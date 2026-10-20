"""Flow matching on PlatonicSplats autoencoder LATENTS, built on ``main_qm9_gen_FM.py``.

A QM9 molecule is a set of atoms with positions and atom-type features. A scene latent from the
PlatonicSplats autoencoder is the same kind of object: a set of 256 tokens, each with a 3D position and
a feature vector (384 for ae_grav_..._trivial). So this script keeps the QM9 flow-matching model as is --
same PlatonicTransformer, same EDM-preconditioned velocity (``_velocity``), same SNR-matched time
schedule, same loss weight, same two samplers -- and swaps only the data side:

  QM9                                   latents
  atoms                                 latent tokens (always 256 per scene)
  one-hot atom types (+charge) / 4      latent features, standardised per channel (unit std)
  atom coordinates, centre removed      latent positions, centre removed, scaled to unit std
  random SO(3) rotation of the batch    `training.augm`: none | yaw | so3 (see the config)
  stability / validity metrics          copying check (generated vs real-new scenes) + saved samples

Generated latents are written in the AUTOENCODER's units (scene centre added back) so PlatonicSplats'
``scripts/latent_fm/sample_latent_fm.py --samples <npz>`` can render them with the frozen decoder.

    # train
    python mains/main_latent_gen_FM.py --config configs/latent_gen.yaml \\
        --dataset.latents_dir=/media/mina/hdd/amlab/nvsm/latents/ae_grav_trivial
    # sample from a checkpoint (writes <testing.output_dir>/samples.npz)
    python mains/main_latent_gen_FM.py --config configs/latent_gen.yaml \\
        --dataset.latents_dir=... --testing.test_ckpt=runs/latent_gen/ckpts/last.ckpt
"""

import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from mains._optim import make_param_groups
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.utils.callbacks import EMACallback, TimerCallback
from platonic_transformers.utils.config_loader import get_arg_parser, load_with_defaults, print_config
from platonic_transformers.utils.utils import RandomSO2AroundAxis, RandomSOd, subtract_mean


# ============================================================================================ data
def load_latents(root, split):
    """``(pos [S,N,3], feat [S,N,F])`` float32 from extract_latents.py shards."""
    files = sorted(glob.glob(os.path.join(root, split, "latents_*.npz")),
                   key=lambda s: int(Path(s).stem.split("_")[1]))
    if not files:
        raise SystemExit(f"no latents in {os.path.join(root, split)} (run extract_latents.py first)")
    data = [np.load(f) for f in files]
    pos = torch.from_numpy(np.concatenate([d["pos"] for d in data]).astype(np.float32))
    feat = torch.from_numpy(np.concatenate([d["feat"] for d in data]).astype(np.float32))
    return pos, feat


class LatentNormaliser:
    """Statistics from the TRAINING latents.

    Positions: each scene's centre is removed (QM9's ``subtract_mean``; the PlatonicTransformer only
    sees relative positions through RoPE anyway) and the result is divided by ONE shared std, so the
    cloud's shape is preserved. The centre is put back at decode time from ``centre_mean``. Features:
    per-channel mean/std -> unit variance, matching ``sigma_data = 1``.
    """

    def __init__(self, pos=None, feat=None, state=None):
        if state is not None:
            for k, v in state.items():
                setattr(self, k, torch.as_tensor(v, dtype=torch.float32))
            return
        centre = pos.mean(1)                                    # [S,3]
        self.centre_mean = centre.mean(0)
        self.centre_std = centre.std(0)
        self.pos_std = (pos - centre[:, None]).std().clamp_min(1e-6)
        f = feat.reshape(-1, feat.shape[-1])
        self.feat_mean = f.mean(0)
        self.feat_std = f.std(0).clamp_min(1e-4)

    def state(self):
        return {k: v.tolist() for k, v in self.__dict__.items()}

    def norm(self, pos, feat):
        pos_c = (pos - pos.mean(-2, keepdim=True)) / self.pos_std
        return pos_c, (feat - self.feat_mean) / self.feat_std

    def denorm(self, pos_c, feat_n):
        dev = pos_c.device
        pos = pos_c * self.pos_std.to(dev) + self.centre_mean.to(dev)
        return pos, feat_n * self.feat_std.to(dev) + self.feat_mean.to(dev)


class LatentSceneDataset(Dataset):
    def __init__(self, pos, feat):
        self.pos, self.feat = pos, feat

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):
        return {"pos": self.pos[i], "x": self.feat[i]}


def collate_latents(items):
    """Flatten a batch of equal-size scenes into the QM9 graph format: pos [B*N,3], x [B*N,F], batch."""
    n = items[0]["pos"].shape[0]
    return {
        "pos": torch.cat([it["pos"] for it in items]),
        "x": torch.cat([it["x"] for it in items]),
        "batch": torch.arange(len(items)).repeat_interleave(n),
    }


def chamfer_to_bank(cloud, bank, chunk=256):
    """Smallest symmetric chamfer distance from ``cloud [N,3]`` to any cloud in ``bank [M,N,3]``."""
    best = float("inf")
    for s in range(0, len(bank), chunk):
        b = bank[s:s + chunk]
        d = torch.cdist(cloud[None].expand(len(b), -1, -1), b)
        cd = 0.5 * (d.min(-1).values.mean(-1) + d.min(-2).values.mean(-1))
        best = min(best, float(cd.min()))
    return best


# ============================================================================================ model
class LatentFMModel(pl.LightningModule):
    """Node-wise conditional flow matching over latent token sets (QM9FMModel with a latent data side)."""

    def __init__(self, config: ml_collections.ConfigDict, feat_dim: int, n_tokens: int, norm_state: dict):
        super().__init__()
        self.save_hyperparameters({"config": config.to_dict(), "feat_dim": feat_dim,
                                   "n_tokens": n_tokens, "norm_state": norm_state})
        self.config = config
        self.feat_dim = feat_dim
        self.n_tokens = n_tokens
        self.normer = LatentNormaliser(state=norm_state)
        solid_name = config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unsupported solid_name '{solid_name}'")
        G = PLATONIC_GROUPS[solid_name].G
        if config.model.hidden_dim // G < feat_dim:
            print(f"[warn] hidden_dim/|G| = {config.model.hidden_dim // G} < {feat_dim} latent features: "
                  f"the scalar input is squeezed through fewer channels per frame than it has (see config)")

        self.net = PlatonicTransformer(
            input_dim=feat_dim + 1,
            input_dim_vec=0,
            hidden_dim=config.model.hidden_dim,
            output_dim=feat_dim,
            output_dim_vec=1,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=config.model.dense_mode,
            scalar_task_level="node",
            vector_task_level="node",
            ffn_readout=config.model.ffn_readout,
            mean_aggregation=config.model.mean_aggregation,
            dropout=config.model.dropout,
            drop_path_rate=config.model.drop_path_rate,
            layer_scale_init_value=config.model.layer_scale_init_value,
            attention=config.model.attention,
            ffn_dim_factor=config.model.ffn_dim_factor,
            rope_sigma=config.model.rope_sigma,
            ape_sigma=config.model.ape_sigma,
            learned_freqs=config.model.learned_freqs,
            freq_init=config.model.freq_init,
            use_key=config.model.use_key,
            rope_on_values=config.model.get("rope_on_values", False),
            attention_backend=config.model.get("attention_backend", "scatter"),
        )
        if config.model.get("compile", False):
            self.net = torch.compile(self.net)

        augm = str(config.training.get("augm", "none")).lower()
        if augm not in ("none", "yaw", "so3"):
            raise ValueError(f"training.augm must be none | yaw | so3, got {augm!r}")
        self.augm = augm
        self.rot_so3 = RandomSOd(3)
        self.rot_yaw = RandomSO2AroundAxis(axis=int(config.training.get("augm_axis", 2)), degrees=180)
        self.ref_train = None   # normalised train positions, for the copying check
        self.ref_val_dist = None

    # -- copying-check reference (set by main) --------------------------------------------------------
    def set_reference(self, train_pos_norm, val_pos_norm):
        self.ref_train = train_pos_norm
        if val_pos_norm is not None and len(val_pos_norm):
            k = min(32, len(val_pos_norm))
            self.ref_val_dist = float(np.median([chamfer_to_bank(c, train_pos_norm) for c in val_pos_norm[:k]]))

    # -- the QM9 flow-matching core, unchanged --------------------------------------------------------
    def _velocity(self, x_t, pos_t, batch, t):
        """EDM-preconditioned flow-matching velocity field (identical to QM9FMModel._velocity)."""
        sigma_data = self.config.diffusion.sigma_data
        t_graph = t.reshape(-1, 1)
        denom = t_graph ** 2 * sigma_data ** 2 + (1 - t_graph) ** 2
        c_in = 1 / denom.sqrt()
        c_skip = (t_graph * sigma_data ** 2 - (1 - t_graph)) / denom
        c_out = sigma_data / denom.sqrt()
        c_noise = (((1 - t_graph) / t_graph).log()) / 4

        c_in_n, c_skip_n, c_out_n, c_noise_n = (v[batch] for v in (c_in, c_skip, c_out, c_noise))
        x_in = c_in_n * x_t
        pos_in = c_in_n * pos_t
        scalar_in = torch.cat([x_in, c_noise_n], dim=-1)
        scalar_out, vector_out = self.net(
            scalar_in, pos_in, batch, vec=None,
            avg_num_nodes=self.config.diffusion.avg_num_nodes,
        )
        v_x = c_skip_n * x_t + c_out_n * scalar_out
        v_pos = c_skip_n * pos_t + c_out_n * vector_out.squeeze(1)
        return v_x, v_pos

    def _sample_t(self, num_graphs, device):
        rnd_normal = torch.randn(num_graphs, device=device)
        sigma = (rnd_normal * self.config.diffusion.P_std + self.config.diffusion.P_mean).exp()
        return 1.0 / (1.0 + sigma)

    def _edm_weight(self, node_t):
        sigma_data = self.config.diffusion.sigma_data
        return (node_t ** 2 * sigma_data ** 2 + (1 - node_t) ** 2) / sigma_data ** 2

    def _augment(self, pos, batch):
        """Per-scene random rotation (yaw about the vertical, or full SO(3)). Features are untouched:
        only valid when the autoencoder's latent features are rotation-invariant (equivariant AE)."""
        if self.augm == "none":
            return pos
        num_graphs = int(batch.max().item()) + 1
        rot = (self.rot_yaw(num_graphs) if self.augm == "yaw" else self.rot_so3(num_graphs))
        rot = rot.reshape(num_graphs, 3, 3).type_as(pos)
        return torch.einsum("nij,nj->ni", rot[batch], pos)

    def _flow_loss(self, batch, augment=False):
        pos = subtract_mean(batch["pos"], batch["batch"])
        x = batch["x"]
        if augment:
            pos = self._augment(pos, batch["batch"])
        num_graphs = int(batch["batch"].max().item()) + 1
        t = self._sample_t(num_graphs, pos.device)
        x0 = torch.randn_like(x)
        pos0 = subtract_mean(torch.randn_like(pos), batch["batch"])
        node_t = t[batch["batch"], None]
        vx, vp = self._velocity((1 - node_t) * x0 + node_t * x,
                                (1 - node_t) * pos0 + node_t * pos, batch["batch"], t)
        weight = self._edm_weight(node_t)
        loss_x = (weight * (vx - (x - x0)).square()).mean()
        loss_pos = (weight * (vp - (pos - pos0)).square()).mean()
        return loss_x + loss_pos, loss_x, loss_pos

    def training_step(self, batch, batch_idx):
        loss, lx, lp = self._flow_loss(batch, augment=True)
        bs = int(batch["batch"].max().item()) + 1
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/loss_feat", lx, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss_pos", lp, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._flow_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=int(batch["batch"].max().item()) + 1)
        return loss

    # -- samplers (QM9's, with a fixed token count) ---------------------------------------------------
    @torch.no_grad()
    def sample(self, num_scenes, num_steps=None, S_churn=None, S_min=0.0, S_max=float("inf"), S_noise=1.0):
        """Returns NORMALISED ``(pos [K,N,3], x [K,N,F])``. Same two samplers as QM9FMModel."""
        self.eval()
        if self.config.diffusion.get("sampler", "fm") == "fm":
            return self._sample_fm_ode(num_scenes, num_steps)
        num_steps = num_steps or self.config.diffusion.num_steps
        if S_churn is None:
            S_churn = self.config.diffusion.get("S_churn", 0.0)
        sigma_min, sigma_max, rho = (self.config.diffusion.sigma_min, self.config.diffusion.sigma_max,
                                     self.config.diffusion.rho)
        batch = torch.arange(num_scenes, device=self.device).repeat_interleave(self.n_tokens)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                       * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])

        def denoise(x_edm, pos_edm, sigma):
            t = 1.0 / (1.0 + sigma)
            x_t, pos_t = t * x_edm, t * pos_edm
            t_full = torch.full((num_scenes,), t.item(), device=self.device)
            vx, vp = self._velocity(x_t, pos_t, batch, t_full)
            return x_t + (1 - t) * vx, pos_t + (1 - t) * vp

        x_edm = torch.randn((len(batch), self.feat_dim), device=self.device)
        pos_edm = subtract_mean(torch.randn((len(batch), 3), device=self.device), batch)
        for i in range(num_steps):
            sigma_cur, sigma_next = sigma_steps[i], sigma_steps[i + 1]
            gamma = min(S_churn / num_steps, 2 ** 0.5 - 1) if S_min <= sigma_cur <= S_max else 0
            sigma_hat = sigma_cur * (1 + gamma)
            noise_scale = (sigma_hat ** 2 - sigma_cur ** 2).sqrt() * S_noise
            x_hat = x_edm + noise_scale * torch.randn_like(x_edm)
            pos_hat = pos_edm + noise_scale * torch.randn_like(pos_edm)
            D_x, D_pos = denoise(x_hat, pos_hat, sigma_hat)
            dx, dpos = (x_hat - D_x) / sigma_hat, (pos_hat - D_pos) / sigma_hat
            x_edm = x_hat + (sigma_next - sigma_hat) * dx
            pos_edm = pos_hat + (sigma_next - sigma_hat) * dpos
            if i < num_steps - 1:
                D_x2, D_pos2 = denoise(x_edm, pos_edm, sigma_next)
                dx2, dpos2 = (x_edm - D_x2) / sigma_next, (pos_edm - D_pos2) / sigma_next
                x_edm = x_hat + (sigma_next - sigma_hat) * (0.5 * dx + 0.5 * dx2)
                pos_edm = pos_hat + (sigma_next - sigma_hat) * (0.5 * dpos + 0.5 * dpos2)
        return self._to_scenes(x_edm, pos_edm, batch, num_scenes)

    @torch.no_grad()
    def _sample_fm_ode(self, num_scenes, num_steps=None):
        num_steps = num_steps or self.config.diffusion.num_steps
        t_start = self.config.diffusion.get("fm_t_start", 0.0)
        t_eps = self.config.diffusion.get("fm_t_eps", 1e-4)
        integrator = self.config.diffusion.get("fm_integrator", "heun")
        batch = torch.arange(num_scenes, device=self.device).repeat_interleave(self.n_tokens)

        def velocity(x, pos, t):
            t_full = torch.full((num_scenes,), float(t), device=self.device).clamp(t_eps, 1 - t_eps)
            vx, vp = self._velocity(x, pos, batch, t_full)
            return vx, subtract_mean(vp, batch)

        x = (1 - t_start) * torch.randn((len(batch), self.feat_dim), device=self.device)
        pos = (1 - t_start) * subtract_mean(torch.randn((len(batch), 3), device=self.device), batch)
        t_steps = torch.linspace(t_start, 1.0, num_steps + 1).tolist()
        for i in range(num_steps):
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            dt = t_next - t_cur
            vx, vp = velocity(x, pos, t_cur)
            x_next, pos_next = x + dt * vx, pos + dt * vp
            if integrator == "heun" and i < num_steps - 1:
                vx2, vp2 = velocity(x_next, pos_next, t_next)
                x_next, pos_next = x + dt * 0.5 * (vx + vx2), pos + dt * 0.5 * (vp + vp2)
            x, pos = x_next, pos_next
        return self._to_scenes(x, pos, batch, num_scenes)

    def _to_scenes(self, x, pos, batch, num_scenes):
        pos = subtract_mean(pos, batch)
        return pos.view(num_scenes, self.n_tokens, 3), x.view(num_scenes, self.n_tokens, self.feat_dim)

    def sample_latents(self, count, chunk=None):
        """Generate ``count`` scenes in the autoencoder's units: ``(pos [K,N,3], feat [K,N,F])`` numpy."""
        chunk = chunk or int(self.config.training.batch_size)
        P, X = [], []
        for s in range(0, count, chunk):
            p, x = self.sample(min(chunk, count - s))
            p, x = self.normer.denorm(p.float(), x.float())
            P.append(p.cpu()); X.append(x.cpu())
        return torch.cat(P).numpy(), torch.cat(X).numpy()

    # -- evaluation: copying check + saved samples ----------------------------------------------------
    def _evaluate_samples(self, count, phase):
        out_dir = Path(self.config.testing.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pos, feat = self.sample_latents(count)
        name = "samples.npz" if phase == "test" else f"samples_epoch{self.current_epoch + 1:05d}.npz"
        np.savez(out_dir / name, pos=pos, feat=feat)
        results = {}
        if self.ref_train is not None:
            p_norm = (torch.from_numpy(pos) - torch.from_numpy(pos).mean(1, keepdim=True)) / self.normer.pos_std
            bank = self.ref_train.to(self.device)
            d = [chamfer_to_bank(c.to(self.device), bank) for c in p_norm[: min(16, count)]]
            results[f"{phase}/copy_dist_gen"] = float(np.median(d))
            if self.ref_val_dist:
                results[f"{phase}/copy_dist_val"] = self.ref_val_dist
                results[f"{phase}/copy_ratio"] = float(np.median(d)) / self.ref_val_dist
        results[f"{phase}/sample_feat_std"] = float(((feat - self.normer.feat_mean.numpy())
                                                    / self.normer.feat_std.numpy()).std())
        for key, value in results.items():
            self.log(key, value, on_epoch=True, logger=True)
        print(f"[{phase}] epoch {self.current_epoch + 1}: {json.dumps({k: round(v, 4) for k, v in results.items()})}"
              f" -> {out_dir / name}")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or (self.current_epoch + 1) % self.config.training.validation_frequency:
            return
        self._evaluate_samples(int(self.config.training.num_val_samples), "val")

    def test_step(self, batch, batch_idx):
        return None

    def on_test_epoch_end(self):
        self._evaluate_samples(int(self.config.testing.num_samples), "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(make_param_groups(self, self.config.optimizer.weight_decay),
                                      lr=self.config.optimizer.lr)
        if self.config.scheduler.use_cosine:
            T_max = self.config.scheduler.get("T_max_epochs", None) or self.trainer.max_epochs
            return {"optimizer": optimizer,
                    "lr_scheduler": CosineAnnealingLR(optimizer, T_max=T_max, eta_min=self.config.optimizer.lr / 100)}
        return optimizer


# ============================================================================================ main
def main(config):
    print_config(config, "Latent Flow Matching Configuration")
    if not config.dataset.latents_dir:
        raise SystemExit("set --dataset.latents_dir=<extract_latents.py --out folder>")
    pl.seed_everything(config.seed)

    pos_tr, feat_tr = load_latents(config.dataset.latents_dir, "train")
    has_val = os.path.isdir(os.path.join(config.dataset.latents_dir, "val"))
    pos_va, feat_va = load_latents(config.dataset.latents_dir, "val") if has_val else (None, None)
    frac = float(config.dataset.get("dataset_fraction", 1.0))
    if frac < 1.0:
        k = max(1, int(len(pos_tr) * frac))
        pos_tr, feat_tr = pos_tr[:k], feat_tr[:k]
    normer = LatentNormaliser(pos_tr, feat_tr)
    pos_tr_n, feat_tr_n = normer.norm(pos_tr, feat_tr)
    S, N, F = feat_tr.shape
    print(f"latents: train {S} scenes x {N} tokens x (3 pos + {F} feat)"
          + (f", val {len(pos_va)} scenes" if has_val else ""))

    kw = dict(batch_size=config.training.batch_size, num_workers=config.system.num_workers,
              collate_fn=collate_latents)
    train_loader = DataLoader(LatentSceneDataset(pos_tr_n, feat_tr_n), shuffle=True, drop_last=True, **kw)
    val_loader = None
    pos_va_n = None
    if has_val:
        pos_va_n, feat_va_n = normer.norm(pos_va, feat_va)
        val_loader = DataLoader(LatentSceneDataset(pos_va_n, feat_va_n), shuffle=False, **kw)

    out_dir = Path(config.testing.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normaliser.json").write_text(json.dumps(normer.state()))

    accelerator = "gpu" if config.system.gpus > 0 and torch.cuda.is_available() else "cpu"
    logger = False
    if config.logging.enabled:
        logger = pl.loggers.WandbLogger(project=config.logging.project_name,
                                        entity=config.logging.get("wandb_identity", None),
                                        config=config.to_dict(), save_dir=str(out_dir))
    callbacks = [pl.callbacks.ModelCheckpoint(dirpath=str(out_dir / "ckpts"), save_last=True,
                                              every_n_epochs=config.training.validation_frequency),
                 TimerCallback()]
    if config.training.get("ema_enabled", False):
        callbacks.append(EMACallback(decay=config.training.get("ema_decay", 0.999),
                                     warmup_steps=config.training.get("ema_warmup_steps", 2000)))
    trainer = pl.Trainer(
        logger=logger, max_epochs=config.training.epochs, callbacks=callbacks,
        accelerator=accelerator, devices=config.system.gpus if accelerator == "gpu" else "auto",
        precision=config.system.precision, gradient_clip_val=config.training.gradient_clip_val,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        default_root_dir=str(out_dir), log_every_n_steps=10,
    )
    model = LatentFMModel(config, feat_dim=F, n_tokens=N, norm_state=normer.state())
    model.set_reference(pos_tr_n, pos_va_n)
    print(f"flow model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | "
          f"copying-check reference (real val -> train): {model.ref_val_dist}")

    test_ckpt = config.testing.test_ckpt
    eval_loader = val_loader if val_loader is not None else train_loader
    if test_ckpt:
        trainer.test(model, eval_loader, ckpt_path=test_ckpt)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.testing.resume_ckpt)
        trainer.test(model, eval_loader, ckpt_path="last")


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/latent_gen.yaml")
    args, unknown_args = parser.parse_known_args()
    main(load_with_defaults(dataset_config=args.config, cli_args=unknown_args))