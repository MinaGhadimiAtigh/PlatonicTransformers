"""NVS flow-matching model.

Adapts QM9FMModel to work with latents from a frozen PlatonicAutoencoder.
Each scene is represented as 256 register tokens, each with:
  - z_feat: (256, 384)  ← scalar features  (replaces atom types)
  - pos_z:  (256, 3)    ← 3D positions      (replaces atom positions)

Key differences from QM9FMModel:
  - No atom type discretization — features are continuous 384-dim vectors
  - Fixed token count (256) — no num_atoms_sampler needed
  - dense_mode=True — tokens already batched as (B, 256, D), no packing needed
  - Frozen encoder runs inside training_step to get (z_feat, pos_z) from images
  - repeats_per_epoch: each scene encoded N times per epoch with different views
  - copy_ratio: nearest-neighbour distance logged to detect memorisation
  - optional fixed global latent normaliser (dataset.norm_passes > 0): per-channel feature
    standardisation and per-axis position centring with ONE scalar position scale; statistics
    are fitted once (optionally cached in dataset.normaliser_path) and saved in the checkpoint
"""

import os
import sys

import ml_collections
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mains._optim import make_param_groups
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.utils.callbacks import EMACallback, TimerCallback
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config,
)
from platonic_transformers.utils.utils import RandomSOd
from psplat.models.platonic_autoencoder import PlatonicAutoencoder


# ---------------------------------------------------------------------------
# Latent normaliser
# ---------------------------------------------------------------------------

class LatentNormaliser:
    """Fixed, invertible global affine map for the autoencoder latents.

    - features:  (x - mean_c) / std_c        per channel
    - positions: (p - mean_xyz) / pos_scale  per-axis mean, ONE scalar scale for all axes

    The statistics are computed once over many scenes and then frozen. Nothing is computed per
    scene, so denormalise() is the exact inverse of normalise(), including the position offset
    (the decoder expects positions in the camera-normalised frame, where the cloud is not centred).
    A scalar position scale keeps the shape of the cloud unchanged.
    """

    def __init__(self, feat_mean, feat_std, pos_mean, pos_scale):
        self.feat_mean = feat_mean   # (1, 1, F)
        self.feat_std  = feat_std    # (1, 1, F)
        self.pos_mean  = pos_mean    # (1, 1, 3)
        self.pos_scale = pos_scale   # scalar tensor

    def normalise(self, pos, feat):
        dev = feat.device
        pos_n  = (pos  - self.pos_mean.to(dev))  / self.pos_scale.to(dev)
        feat_n = (feat - self.feat_mean.to(dev)) / self.feat_std.to(dev)
        return pos_n, feat_n

    def denormalise(self, pos_n, feat_n):
        dev = feat_n.device
        pos  = pos_n  * self.pos_scale.to(dev) + self.pos_mean.to(dev)
        feat = feat_n * self.feat_std.to(dev)  + self.feat_mean.to(dev)
        return pos, feat

    def state_dict(self):
        return {
            "feat_mean": self.feat_mean.detach().cpu(),
            "feat_std":  self.feat_std.detach().cpu(),
            "pos_mean":  self.pos_mean.detach().cpu(),
            "pos_scale": self.pos_scale.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["feat_mean"], d["feat_std"], d["pos_mean"], d["pos_scale"])


class LatentStatsAccumulator:
    """Streaming mean / variance of latents in float64 (no need to keep every latent in memory)."""

    def __init__(self):
        self.n = 0
        self.f_sum = self.f_sq = self.p_sum = self.p_sq = None

    @torch.no_grad()
    def update(self, pos, feat):
        feat = feat.detach().double().reshape(-1, feat.shape[-1]).cpu()
        pos  = pos.detach().double().reshape(-1, pos.shape[-1]).cpu()
        if not (torch.isfinite(feat).all() and torch.isfinite(pos).all()):
            raise RuntimeError("[normaliser] latents contain NaN or infinity")
        if self.f_sum is None:
            self.f_sum = torch.zeros(feat.shape[-1], dtype=torch.float64)
            self.f_sq  = torch.zeros_like(self.f_sum)
            self.p_sum = torch.zeros(pos.shape[-1], dtype=torch.float64)
            self.p_sq  = torch.zeros_like(self.p_sum)
        self.f_sum += feat.sum(0); self.f_sq += (feat ** 2).sum(0)
        self.p_sum += pos.sum(0);  self.p_sq += (pos ** 2).sum(0)
        self.n += feat.shape[0]

    def finalize(self):
        if self.n == 0:
            raise RuntimeError("[normaliser] no latents were accumulated")
        f_mean = self.f_sum / self.n
        f_std  = (self.f_sq / self.n - f_mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
        p_mean = self.p_sum / self.n
        p_var  = (self.p_sq / self.n - p_mean ** 2).clamp(min=0)
        p_scale = p_var.mean().sqrt().clamp(min=1e-6)   # RMS distance from the mean, all axes
        print(f"[normaliser] {self.n} tokens | feat mean |.| avg={f_mean.abs().mean():.4f}  "
              f"feat std min/median/max={f_std.min():.4f}/{f_std.median():.4f}/{f_std.max():.4f}")
        print(f"[normaliser] pos mean={[round(v, 4) for v in p_mean.tolist()]}  "
              f"pos scale={p_scale:.4f}")
        return LatentNormaliser(
            f_mean.float()[None, None], f_std.float()[None, None],
            p_mean.float()[None, None], p_scale.float(),
        )


@torch.no_grad()
def fit_normaliser(autoencoder, train_loader, n_passes=1, device="cuda"):
    print(f"[normaliser] fitting from {n_passes} passes over training set...")
    acc = LatentStatsAccumulator()
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    for pass_i in range(n_passes):
        for batch in train_loader:
            batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            input, _ = autoencoder.process_data(
                batch_d, has_target_image=False, target_has_input=False, compute_rays=True,
            )
            z_feat, pos_z, _ = autoencoder.encode(input)
            acc.update(pos_z, z_feat)
        print(f"[normaliser] pass {pass_i+1}/{n_passes} done")
    return acc.finalize()


def save_normaliser(normaliser, path, ae_ckpt_dir):
    state = normaliser.state_dict()
    state["autoencoder_checkpoint_dir"] = str(ae_ckpt_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_normaliser(path, ae_ckpt_dir):
    state = torch.load(path, map_location="cpu", weights_only=True)
    stored = state.get("autoencoder_checkpoint_dir")
    if stored != str(ae_ckpt_dir):
        raise ValueError(
            f"Normaliser file {path} was fitted for autoencoder '{stored}', but this run uses "
            f"'{ae_ckpt_dir}'. Delete the file to refit, or point normaliser_path elsewhere."
        )
    return LatentNormaliser.from_state_dict(state)


# ---------------------------------------------------------------------------
# Repeated dataset wrapper (repeats_per_epoch)
# ---------------------------------------------------------------------------

class RepeatedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to repeat each scene N times per epoch.

    Each repeat calls __getitem__ fresh, so the ovs.Dataset view-sampler
    picks a different random set of views each time. This multiplies the
    effective number of view combinations seen per epoch by N, acting as
    strong data augmentation for small datasets.

    Example: 512 scenes × repeats=4 = 2048 items per epoch, each with
    a different random view subset from the same scene.
    """

    def __init__(self, dataset, repeats: int):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        # map repeated index back to original scene index
        return self.dataset[idx % len(self.dataset)]


# ---------------------------------------------------------------------------
# Copy ratio utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_copy_ratio(gen_feats, train_feats, threshold_quantile: float = 0.05):
    """Compute nearest-neighbour distance between generated and training latents.

    For each generated scene, find the closest training scene by L2 distance
    on the mean-pooled z_feat (384-dim → scalar distance). Returns:
      - mean_nn_dist:  mean distance to nearest training neighbour
      - copy_ratio:    fraction of generated scenes closer than threshold
                       (threshold = threshold_quantile of pairwise train distances)

    Args:
        gen_feats:   (G, N, F) — generated z_feat from sample()
        train_feats: (T, N, F) — all training z_feat from encode()
        threshold_quantile: quantile of pairwise train distances used as copy threshold

    Returns:
        dict with mean_nn_dist and copy_ratio
    """
    # Mean-pool over tokens → (G, F) and (T, F)
    g = gen_feats.mean(dim=1).float()    # (G, F)
    t = train_feats.mean(dim=1).float()  # (T, F)

    # Pairwise L2 distances: (G, T)
    # ||g_i - t_j||^2 = ||g_i||^2 + ||t_j||^2 - 2 g_i . t_j
    g_sq = (g ** 2).sum(dim=-1, keepdim=True)   # (G, 1)
    t_sq = (t ** 2).sum(dim=-1, keepdim=True).T # (1, T)
    gt   = g @ t.T                               # (G, T)
    dists_gen_train = (g_sq + t_sq - 2 * gt).clamp(min=0).sqrt()  # (G, T)
    nn_dists = dists_gen_train.min(dim=1).values  # (G,) nearest neighbour distance

    # Threshold: quantile of pairwise training distances
    T = t.shape[0]
    if T > 1:
        t_sq2 = (t ** 2).sum(dim=-1, keepdim=True)
        tt = t @ t.T
        dists_train = (t_sq2 + t_sq2.T - 2 * tt).clamp(min=0).sqrt()
        # exclude diagonal (self-distance = 0)
        mask = ~torch.eye(T, dtype=torch.bool, device=t.device)
        pairwise = dists_train[mask]
        threshold = torch.quantile(pairwise.float(), threshold_quantile)
    else:
        threshold = torch.tensor(float("inf"))

    copy_ratio = (nn_dists < threshold).float().mean().item()
    mean_nn_dist = nn_dists.mean().item()

    return {"val/mean_nn_dist": mean_nn_dist, "val/copy_ratio": copy_ratio}


@torch.no_grad()
def mean_nn_distance(query_feats, ref_feats):
    """Mean over queries of the distance to the nearest reference, on mean-pooled z_feat.

    Same distance as compute_copy_ratio. Used for real validation latents against the training
    latents, as a reference for the generated scenes' val/mean_nn_dist.
    Args: query_feats (Q, N, F), ref_feats (R, N, F).
    """
    q = query_feats.mean(dim=1).float()
    r = ref_feats.mean(dim=1).float()
    return torch.cdist(q, r).min(dim=1).values.mean().item()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class NVSFMModel(pl.LightningModule):
    """Node-wise conditional flow matching model for NVS scene latents."""

    def __init__(self, config, autoencoder):
        super().__init__()
        self.config = config

        # Frozen encoder
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        # Dimensions from autoencoder config
        ae_cfg = autoencoder.config
        self.n_tokens  = int(ae_cfg.model.registers.n_reg)
        ae_group_name  = ae_cfg.model.equivariance.solid_name.lower()
        ae_group       = PLATONIC_GROUPS[ae_group_name]
        self.scalar_dim = int(ae_cfg.model.transformer.d) // ae_group.G

        # Flow matching network
        solid_name = config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unsupported solid_name '{solid_name}'")

        self.net = PlatonicTransformer(
            input_dim=self.scalar_dim + 1,
            input_dim_vec=0,
            hidden_dim=config.model.hidden_dim,
            output_dim=self.scalar_dim,
            output_dim_vec=1,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=True,
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

        if config.model.get("compile", True):
            self.net = torch.compile(self.net)

        self.rotation_generator = RandomSOd(3)
        self.normaliser = None
        self.val_dataset = None

        # Relative weights of the two loss terms. Each term is a mean over its own elements, so by default
        # one position coordinate counts 384/3 = 128 times as much as one feature channel. 1.0 / 1.0 is the
        # unchanged sum. Only the feature weight is meant to be tuned; keep pos_loss_weight at 1.0.
        self.feat_loss_weight = float(config.training.get("feat_loss_weight", 1.0))
        self.pos_loss_weight  = float(config.training.get("pos_loss_weight", 1.0))
        if self.feat_loss_weight < 0 or self.pos_loss_weight < 0:
            raise ValueError("feat_loss_weight and pos_loss_weight must be >= 0")
        print(f"[NVSFMModel] loss weights: feat={self.feat_loss_weight:g}  pos={self.pos_loss_weight:g}")

        # Cache of training latents for copy_ratio — built lazily
        self._train_feats_cache = None
        self._val_feats_cache = None

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def set_normaliser(self, normaliser: LatentNormaliser):
        self.normaliser = normaliser
        print(f"[NVSFMModel] normaliser set: "
              f"feat_std={normaliser.feat_std.mean():.4f}  pos_scale={normaliser.pos_scale:.4f}")

    def set_val_dataset(self, val_dataset):
        self.val_dataset = val_dataset

    def set_train_loader(self, train_loader):
        """Store train loader so we can encode all training scenes for copy_ratio."""
        self._train_loader = train_loader

    def set_val_loader(self, val_loader):
        """Store val loader so we can encode real validation scenes as a distance reference."""
        self._val_loader = val_loader

    # The normaliser is not a parameter or buffer, so store it in the checkpoint explicitly.
    # Without it a checkpoint cannot turn generated latents back into the decoder's space.
    def on_save_checkpoint(self, checkpoint):
        if self.normaliser is not None:
            checkpoint["latent_normaliser"] = self.normaliser.state_dict()

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get("latent_normaliser")
        if saved is not None:
            self.normaliser = LatentNormaliser.from_state_dict(saved)
            print("[NVSFMModel] restored latent normaliser from checkpoint")
        elif self.normaliser is not None:
            raise ValueError(
                "This run uses a latent normaliser (dataset.norm_passes > 0) but the checkpoint "
                "has none, so it was trained on raw latents. Set norm_passes: 0 to match it."
            )

    # -----------------------------------------------------------------------
    # Core FM math
    # -----------------------------------------------------------------------

    def _sample_t(self, batch_size, device):
        rnd_normal = torch.randn(batch_size, device=device)
        sigma = (rnd_normal * self.config.diffusion.P_std + self.config.diffusion.P_mean).exp()
        return 1.0 / (1.0 + sigma)

    def _edm_weight(self, t):
        sigma_data = self.config.diffusion.sigma_data
        return (t ** 2 * sigma_data ** 2 + (1 - t) ** 2) / sigma_data ** 2

    def _velocity(self, x_t, pos_t, t):
        sigma_data = self.config.diffusion.sigma_data
        t_ = t[:, None, None]
        denom   = t_ ** 2 * sigma_data ** 2 + (1 - t_) ** 2
        c_in    = 1.0 / denom.sqrt()
        c_skip  = (t_ * sigma_data ** 2 - (1 - t_)) / denom
        c_out   = sigma_data / denom.sqrt()
        c_noise = (((1 - t_) / t_).log() / 4).expand(x_t.shape[0], x_t.shape[1], 1)

        x_in      = c_in * x_t
        pos_in    = c_in * pos_t
        scalar_in = torch.cat([x_in, c_noise], dim=-1)

        scalar_out, vector_out = self.net(
            scalar_in, pos_in, batch=None,
            avg_num_nodes=self.config.diffusion.avg_num_nodes,
        )

        v_x   = c_skip * x_t   + c_out * scalar_out
        v_pos = c_skip * pos_t + c_out * vector_out.squeeze(-2)
        return v_x, v_pos

    # -----------------------------------------------------------------------
    # Encode
    # -----------------------------------------------------------------------

    def _encode(self, batch):
        """Encode a raw image batch → (z_feat, pos_z), optionally normalised."""
        with torch.no_grad():
            input, _ = self.autoencoder.process_data(
                batch, has_target_image=False, target_has_input=False, compute_rays=True,
            )
            z_feat, pos_z, _ = self.autoencoder.encode(input)

        if self.normaliser is not None:
            pos_z, z_feat = self.normaliser.normalise(pos_z, z_feat)
        return z_feat, pos_z

    # -----------------------------------------------------------------------
    # Training / validation steps
    # -----------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x1, pos1 = self._encode(batch)
        B = x1.shape[0]

        if self.config.training.train_augm:
            rot  = self.rotation_generator().type_as(pos1)
            pos1 = torch.einsum("ij,bnj->bni", rot, pos1)

        t   = self._sample_t(B, x1.device)
        t_  = t[:, None, None]
        x0  = torch.randn_like(x1)
        pos0 = torch.randn_like(pos1)
        x_t   = (1 - t_) * x0   + t_ * x1
        pos_t = (1 - t_) * pos0 + t_ * pos1

        vx, vp = self._velocity(x_t, pos_t, t)
        weight = self._edm_weight(t_)
        loss_feat = (weight * (vx - (x1 - x0)).square()).mean()
        loss_pos  = (weight * (vp - (pos1 - pos0)).square()).mean()
        loss = self.feat_loss_weight * loss_feat + self.pos_loss_weight * loss_pos

        # train/loss is the weighted sum that is optimised; the *_feat / *_pos terms and
        # loss_unweighted are the plain values, comparable across runs with different weights.
        self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=B)
        self.log("train/loss_unweighted", loss_feat + loss_pos, on_step=True, on_epoch=True, logger=True, batch_size=B)
        self.log("train/loss_feat", loss_feat, on_step=True, on_epoch=True, logger=True, batch_size=B)
        self.log("train/loss_pos", loss_pos, on_step=True, on_epoch=True, logger=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, pos1 = self._encode(batch)
        B = x1.shape[0]
        t  = self._sample_t(B, x1.device)
        t_ = t[:, None, None]
        x0, pos0 = torch.randn_like(x1), torch.randn_like(pos1)
        x_t   = (1 - t_) * x0   + t_ * x1
        pos_t = (1 - t_) * pos0 + t_ * pos1
        vx, vp = self._velocity(x_t, pos_t, t)
        weight = self._edm_weight(t_)
        loss_feat = (weight * (vx - (x1 - x0)).square()).mean()
        loss_pos  = (weight * (vp - (pos1 - pos0)).square()).mean()
        loss = self.feat_loss_weight * loss_feat + self.pos_loss_weight * loss_pos
        # val/loss (the checkpoint monitor) is the weighted sum; compare runs with val/loss_unweighted.
        self.log("val/loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=B)
        self.log("val/loss_unweighted", loss_feat + loss_pos, on_step=False, on_epoch=True, logger=True, batch_size=B)
        self.log("val/loss_feat", loss_feat, on_step=False, on_epoch=True, logger=True, batch_size=B)
        self.log("val/loss_pos", loss_pos, on_step=False, on_epoch=True, logger=True, batch_size=B)
        return loss

    # -----------------------------------------------------------------------
    # Full validation: render + copy_ratio
    # -----------------------------------------------------------------------

    def _build_train_feats_cache(self, device):
        """Encode all training scenes once and cache z_feat for copy_ratio.

        Uses the stored train_loader. Runs under no_grad, frozen encoder.
        Returns (T, n_tokens, scalar_dim) tensor on CPU.
        """
        return self._build_feats_cache(getattr(self, "_train_loader", None), device, "training")

    def _build_feats_cache(self, loader, device, label):
        """Same as above for any loader (training or validation). (T, n_tokens, scalar_dim) on CPU."""
        if loader is None:
            return None
        print(f"[copy_ratio] encoding {label} set for cache...")
        all_feats = []
        for batch in loader:
            batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                input, _ = self.autoencoder.process_data(
                    batch_d, has_target_image=False, target_has_input=False, compute_rays=True,
                )
                z_feat, _, _ = self.autoencoder.encode(input)
            all_feats.append(z_feat.cpu())
        cache = torch.cat(all_feats, dim=0)  # (T, N, F)
        print(f"[copy_ratio] cached {cache.shape[0]} {label} scenes")
        return cache

    def on_validation_epoch_end(self):
        freq = self.config.training.validation_frequency
        if (self.current_epoch + 1) % freq != 0:
            return

        if not hasattr(self, "val_dataset") or self.val_dataset is None:
            return
        logger = self.trainer.logger if self.trainer is not None else None
        if logger is None or not hasattr(logger, "experiment"):
            return

        try:
            import wandb
            import numpy as np

            n_scenes  = 4
            view_idxs = [0, 3, 6, 9, 12]
            n_views   = len(view_idxs)
            device    = next(self.net.parameters()).device
            sampler   = self.config.diffusion.get("sampler", "fm_ode")
            image_h_w = (128, 128)

            # --- sample generated latents ---
            # Sample more scenes than we render, so the statistics and distances below are less coarse.
            n_stat = 16
            z_all, pos_all = self.sample(n_stat, sampler=sampler)
            z_gen, pos_gen = z_all[:n_scenes], pos_all[:n_scenes]

            # --- statistics of the generated latents, in the flow's own (model) space. With the
            #     normaliser on and a well-trained flow: feat mean ~0, feat std ~1, pos rms ~1,
            #     pos mean ~0. Large deviations mean the samples are off the training distribution. ---
            pos_c = pos_all - pos_all.mean(dim=(0, 1), keepdim=True)
            gen_stats = {
                "val/gen_feat_mean":    z_all.mean().item(),
                "val/gen_feat_std":     z_all.std().item(),
                "val/gen_pos_rms":      pos_c.pow(2).mean().sqrt().item(),
                "val/gen_pos_mean_abs": pos_all.mean(dim=(0, 1)).abs().mean().item(),
                "trainer/global_step":  self.global_step,
            }
            logger.experiment.log(gen_stats)
            print(f"[gen latents] epoch {self.current_epoch} | "
                  f"feat mean={gen_stats['val/gen_feat_mean']:+.3f} std={gen_stats['val/gen_feat_std']:.3f} | "
                  f"pos rms={gen_stats['val/gen_pos_rms']:.3f} mean_abs={gen_stats['val/gen_pos_mean_abs']:.3f}")

            # --- copy_ratio / nearest-neighbour distance: build the caches once per run ---
            if self._train_feats_cache is None:
                self._train_feats_cache = self._build_train_feats_cache(device)
            if self._val_feats_cache is None:
                self._val_feats_cache = self._build_feats_cache(
                    getattr(self, "_val_loader", None), device, "validation")

            if self._train_feats_cache is not None:
                # The caches hold raw AE features, so compare in raw AE space.
                z_cmp = z_all
                if self.normaliser is not None:
                    _, z_cmp = self.normaliser.denormalise(pos_all, z_all)
                cr = compute_copy_ratio(
                    z_cmp.cpu(),
                    self._train_feats_cache,
                    threshold_quantile=self.config.training.get("copy_ratio_quantile", 0.05),
                )
                nn_log = {
                    "val/mean_nn_dist": cr["val/mean_nn_dist"],
                    "val/copy_ratio":   cr["val/copy_ratio"],
                    "trainer/global_step": self.global_step,
                }
                msg = (f"[copy_ratio] epoch {self.current_epoch} | "
                       f"mean_nn_dist={cr['val/mean_nn_dist']:.4f} | "
                       f"copy_ratio={cr['val/copy_ratio']:.3f}")
                if self._val_feats_cache is not None:
                    # Reference: how far REAL validation scenes are from the training scenes.
                    ref = mean_nn_distance(self._val_feats_cache, self._train_feats_cache)
                    nn_log["val/real_mean_nn_dist"] = ref
                    msg += f" | real_val_mean_nn_dist={ref:.4f}"
                logger.experiment.log(nn_log)
                print(msg)

            # --- render images ---
            generated_panels = []
            real_panels      = []

            def to_grid(images):
                imgs = images[0]
                imgs = torch.nan_to_num(imgs, nan=0.0).clamp(0, 1)
                imgs = (imgs * 255).byte().cpu().numpy()
                row  = np.concatenate([imgs[v] for v in range(n_views)], axis=2)
                return row.transpose(1, 2, 0)

            for scene_i in range(n_scenes):
                scene_data = self.val_dataset.scene_batch(
                    scene_idx=scene_i, view_indices=view_idxs, device=device,
                )
                ray_o, ray_d = self.autoencoder.process_data.compute_rays(
                    scene_data["c2w"], scene_data["fxfycxcy"],
                    h=image_h_w[0], w=image_h_w[1], device=device,
                )

                # generated
                z_i   = z_gen[scene_i:scene_i+1]
                pos_i = pos_gen[scene_i:scene_i+1]
                gen_images = self.render(z_i, pos_i, ray_o, ray_d, image_h_w)

                # real encoded
                with torch.no_grad():
                    from easydict import EasyDict as edict
                    enc_ray_o, enc_ray_d = self.autoencoder.process_data.compute_rays(
                        scene_data["c2w"], scene_data["fxfycxcy"],
                        h=image_h_w[0], w=image_h_w[1], device=device,
                    )
                    input_enc = edict({
                        "image": scene_data["image"], "ray_o": enc_ray_o,
                        "ray_d": enc_ray_d, "image_h_w": image_h_w,
                    })
                    z_real, pos_real, _ = self.autoencoder.encode(input_enc)
                real_images = self.render_raw(z_real, pos_real, ray_o, ray_d, image_h_w)

                generated_panels.append(wandb.Image(
                    to_grid(gen_images),
                    caption=f"generated scene {scene_i} | epoch {self.current_epoch}"
                ))
                real_panels.append(wandb.Image(
                    to_grid(real_images),
                    caption=f"real scene {scene_i} | epoch {self.current_epoch}"
                ))

            logger.experiment.log({
                "val/generated_scenes": generated_panels,
                "val/real_scenes":      real_panels,
                "val/render_epoch":     self.current_epoch,
            })

        except Exception as e:
            print(f"[warn] validation rendering skipped: {e}")

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, num_scenes: int, sampler: str = "fm_ode"):
        self.eval()
        compiled_net = self.net
        if hasattr(self.net, "_orig_mod"):
            self.net = self.net._orig_mod
        try:
            if sampler == "fm_ode":
                return self._sample_fm_ode(num_scenes)
            elif sampler == "edm":
                return self._sample_edm(num_scenes)
            else:
                raise ValueError(f"Unknown sampler '{sampler}'. Choose 'fm_ode' or 'edm'.")
        finally:
            self.net = compiled_net

    def _sample_fm_ode(self, num_scenes: int):
        num_steps = self.config.diffusion.num_steps
        order     = self.config.diffusion.get("fm_ode_order", 1)
        device    = next(self.net.parameters()).device

        x   = torch.randn(num_scenes, self.n_tokens, self.scalar_dim, device=device)
        pos = torch.randn(num_scenes, self.n_tokens, 3, device=device)

        eps = 1e-4
        ts  = torch.linspace(eps, 1.0, num_steps + 1, device=device)
        dt  = ts[1] - ts[0]

        for i in range(num_steps):
            t_now  = ts[i].expand(num_scenes)
            t_next = ts[i + 1].expand(num_scenes)

            vx, vp = self._velocity(x, pos, t_now)
            vx = vx.clamp(-10, 10)
            vp = vp.clamp(-10, 10)

            if order == 1:
                x   = x   + dt * vx
                pos = pos + dt * vp
            else:
                x_pred   = (x   + dt * vx).clamp(-50, 50)
                pos_pred = (pos + dt * vp).clamp(-50, 50)
                vx2, vp2 = self._velocity(x_pred, pos_pred, t_next)
                vx2 = vx2.clamp(-10, 10)
                vp2 = vp2.clamp(-10, 10)
                x   = x   + dt * 0.5 * (vx  + vx2)
                pos = pos + dt * 0.5 * (vp  + vp2)

        return x, pos

    def _sample_edm(self, num_scenes: int):
        num_steps = self.config.diffusion.num_steps
        sigma_min = self.config.diffusion.sigma_min
        sigma_max = self.config.diffusion.sigma_max
        rho       = self.config.diffusion.rho
        S_churn   = self.config.diffusion.get("S_churn", 0.0)
        S_noise   = self.config.diffusion.get("S_noise", 1.0)
        device    = next(self.net.parameters()).device

        ramp   = torch.linspace(0, 1, num_steps, device=device)
        sigmas = (sigma_max ** (1/rho) + ramp * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

        x   = torch.randn(num_scenes, self.n_tokens, self.scalar_dim, device=device) * sigma_max
        pos = torch.randn(num_scenes, self.n_tokens, 3, device=device) * sigma_max

        for i in range(num_steps):
            sigma      = sigmas[i]
            sigma_next = sigmas[i + 1]

            if S_churn > 0 and sigma_next > 0:
                gamma     = min(S_churn / num_steps, 2**0.5 - 1)
                sigma_hat = sigma * (1 + gamma)
                x   = x   + torch.randn_like(x)   * S_noise * (sigma_hat**2 - sigma**2).sqrt()
                pos = pos + torch.randn_like(pos) * S_noise * (sigma_hat**2 - sigma**2).sqrt()
            else:
                sigma_hat = sigma

            t_hat  = (1.0 / (1.0 + sigma_hat)).expand(num_scenes)
            t_next = (1.0 / (1.0 + sigma_next)).expand(num_scenes)

            vx, vp   = self._velocity(x, pos, t_hat)
            t_hat_   = t_hat[:, None, None]
            dx       = x   + (1 - t_hat_) * vx
            dpos     = pos + (1 - t_hat_) * vp
            dt_sigma = sigma_next - sigma_hat
            x   = x   + (x   - dx  ) / sigma_hat * dt_sigma
            pos = pos + (pos - dpos) / sigma_hat * dt_sigma

            if sigma_next > 0:
                vx2, vp2 = self._velocity(x, pos, t_next)
                t_next_  = t_next[:, None, None]
                dx2      = x   + (1 - t_next_) * vx2
                dpos2    = pos + (1 - t_next_) * vp2
                x   = x   + (x   - dx  ) / sigma_hat * dt_sigma * 0.5 \
                          + (x   - dx2 ) / sigma_next * dt_sigma * 0.5
                pos = pos + (pos - dpos ) / sigma_hat * dt_sigma * 0.5 \
                          + (pos - dpos2) / sigma_next * dt_sigma * 0.5

        return x, pos

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _decode(self, z_feat, pos_z, ray_o, ray_d, image_h_w):
        if self.autoencoder.latent_mode == "invariant":
            from platonic_transformers.models.platoformer.io import lift_scalars
            z_lift = lift_scalars(z_feat, self.autoencoder.group).flatten(-2)
        else:
            z_lift = z_feat
        return self.autoencoder.decode_views(z_lift, pos_z, ray_o, ray_d, image_h_w)

    @torch.no_grad()
    def render(self, z_feat, pos_z, ray_o, ray_d, image_h_w):
        """Decode GENERATED (normalised-space) latents → images."""
        if self.normaliser is not None:
            pos_z, z_feat = self.normaliser.denormalise(pos_z, z_feat)
        return self._decode(z_feat, pos_z, ray_o, ray_d, image_h_w)

    @torch.no_grad()
    def render_raw(self, z_feat, pos_z, ray_o, ray_d, image_h_w):
        """Decode RAW AE-space latents → images (no denormalisation)."""
        return self._decode(z_feat, pos_z, ray_o, ray_d, image_h_w)

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            make_param_groups(self, self.config.optimizer.weight_decay),
            lr=self.config.optimizer.lr,
        )
        if self.config.scheduler.use_cosine:
            T_max = self.config.scheduler.get("T_max_epochs", None) or self.trainer.max_epochs
            return {"optimizer": optimizer, "lr_scheduler": CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=self.config.optimizer.lr / 100,
            )}
        return optimizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config: ml_collections.ConfigDict, ae_config: ml_collections.ConfigDict):
    import importlib
    module_path, class_name = config.dataset.dataset_name.rsplit(".", 1)
    DatasetClass = getattr(importlib.import_module(module_path), class_name)

    train_manifest = os.path.join(config.dataset.data_dir, "train", "manifest.json")
    val_manifest   = os.path.join(config.dataset.data_dir, "val",   "manifest.json")

    dataset_config = ml_collections.ConfigDict({
        "training": {
            "dataset_name":      config.dataset.dataset_name,
            "dataset_path":      train_manifest,
            "num_views":         10,
            "num_input_views":   5,
            "num_target_views":  5,
            "holdout_views":     2,
            "pose_norm":         "lvsm_mean",
            "scene_scale_factor": 1.35,
            "square_crop":       False,
            "rot_aug":           False,
            "rot_aug_so3":       False,
            "target_has_input":  False,
        },
        "model": {
            "image_tokenizer": {
                "image_size": ae_config.model.image_tokenizer.image_size,
                "patch_size": ae_config.model.image_tokenizer.patch_size,
            }
        },
        "inference": {"if_inference": False},
    })

    train_set = DatasetClass(dataset_config)

    val_config = ml_collections.ConfigDict(dataset_config.to_dict())
    val_config.training.dataset_path = val_manifest
    val_set = DatasetClass(val_config)

    if config.dataset.dataset_fraction < 1.0:
        subset_len = int(len(train_set) * config.dataset.dataset_fraction)
        train_set = torch.utils.data.Subset(
            train_set, torch.randperm(len(train_set))[:subset_len]
        )

    # repeats_per_epoch: wrap train_set so each scene appears N times per epoch
    # with different random view subsets each time (view augmentation)
    repeats = config.dataset.get("repeats_per_epoch", 1)
    if repeats > 1:
        train_set = RepeatedDataset(train_set, repeats)
        print(f"[data] repeats_per_epoch={repeats} → {len(train_set)} items/epoch")

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: ml_collections.ConfigDict) -> None:
    print_config(config, "NVS Flow Matching Configuration")
    torch.set_float32_matmul_precision(config.system.get("float32_matmul_precision", "high"))
    pl.seed_everything(config.seed)

    # Load frozen autoencoder
    ae_ckpt_dir    = config.autoencoder.checkpoint_dir
    ae_config_path = os.path.join(ae_ckpt_dir, "config.yaml")
    ae_config      = load_with_defaults(dataset_config=ae_config_path, cli_args=[])
    autoencoder    = PlatonicAutoencoder(ae_config)
    autoencoder.load_ckpt(ae_ckpt_dir)
    autoencoder.eval()
    print(f"Loaded autoencoder from {ae_ckpt_dir}")

    # Data
    train_loader, val_loader = load_data(config, ae_config)

    # Normaliser (optional): fixed global statistics, fitted once. If dataset.normaliser_path is
    # set, the statistics are saved there and reused by later runs with the same autoencoder.
    norm_passes = config.dataset.get("norm_passes", 0)
    norm_path   = config.dataset.get("normaliser_path", None)
    normaliser  = None
    if norm_passes > 0:
        if norm_path and os.path.isfile(norm_path):
            normaliser = load_normaliser(norm_path, ae_ckpt_dir)
            print(f"[normaliser] loaded fixed statistics from {norm_path}")
        else:
            normaliser = fit_normaliser(
                autoencoder, train_loader,
                n_passes=norm_passes,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            if norm_path:
                save_normaliser(normaliser, norm_path, ae_ckpt_dir)
                print(f"[normaliser] saved fixed statistics to {norm_path}")

    # Model
    model = NVSFMModel(config, autoencoder)
    if normaliser is not None:
        model.set_normaliser(normaliser)
    model.set_val_dataset(val_loader.dataset)
    model.set_train_loader(train_loader)   # for copy_ratio cache
    model.set_val_loader(val_loader)       # for the real-validation distance reference

    # Trainer
    accelerator = "gpu" if config.system.gpus > 0 and torch.cuda.is_available() else "cpu"
    devices     = config.system.gpus if accelerator == "gpu" else "auto"

    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            entity=config.logging.get("wandb_identity", None),
            config=config.to_dict(),
            save_dir=save_dir,
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            every_n_epochs=config.training.validation_frequency,
            save_last=config.checkpoint.save_last,
        ),
        TimerCallback(),
    ]
    if config.training.get("ema_enabled", False):
        callbacks.append(EMACallback(
            decay=config.training.get("ema_decay", 0.999),
            warmup_steps=config.training.get("ema_warmup_steps", 2000),
        ))
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=config.system.precision,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
    )

    test_ckpt   = config.testing.get("test_ckpt", None)
    resume_ckpt = config.testing.get("resume_ckpt", None)

    if test_ckpt is None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
    else:
        trainer.test(model, val_loader, ckpt_path=test_ckpt)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/nvs_fm.yaml")
    args, unknown_args = parser.parse_known_args()
    config = load_with_defaults(dataset_config=args.config, cli_args=unknown_args)
    main(config)