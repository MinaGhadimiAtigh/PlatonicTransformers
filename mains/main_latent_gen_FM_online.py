"""ONLINE latent flow matching: encode scenes from RANDOM views at every step, no saved latents.

Same flow model as ``main_latent_gen_FM.py`` (and so as ``main_qm9_gen_FM.py``): the PlatonicTransformer,
the EDM-preconditioned velocity, the SNR-matched time schedule, the loss weight and both samplers are
inherited unchanged from ``LatentFMModel``. What changes is where the latents come from:

  offline (main_latent_gen_FM.py)              online (this script)
  one latent per scene, from views 0-9,        every step: load each scene with n_input_views (5) views
  extracted once and saved to disk             drawn at random from its non-hold-out views (0-13), encode
                                               them with the FROZEN autoencoder on the GPU, train on that
  512 training examples                        512 scenes x C(14,5)=2002 view subsets x view orders

The autoencoder is held outside the LightningModule's parameters: it is never trained, never in the
optimiser, never in the checkpoints. Only the flow model is saved.

Validation uses FIXED views (0..n_input_views-1 by default) encoded once, and a fixed-seed loss averaged
over ``training.val_repeats`` draws, so the val curve moves only when the model changes.

Runs in the PlatonicSplats environment (the autoencoder needs psplat + its torch). PlatonicTransformers
needs three more pure-python packages there; ``uv run --with`` adds them without changing the env:

    cd <PlatonicTransformers>
    PS=/path/to/PlatonicSplats
    PYTHONPATH=$PS:$PS/LVSM uv run --project $PS \\
        --with ml-collections --with pytorch-lightning --with torch_geometric \\
        python mains/main_latent_gen_FM_online.py --config configs/latent_gen_online.yaml \\
        --dataset.psplat_repo=$PS \\
        --dataset.ae_run=/media/mina/hdd/amlab/nvsm/ae_grav_dense_dec4_v55_20k_trivial_rayinit_bias_noaug \\
        --dataset.data_root=/media/mina/hdd/amlab/nvsm/gravity512v16_dense

    # sample from a checkpoint (writes <testing.output_dir>/samples.npz, AE units)
    ... same command ... --testing.test_ckpt=runs/latent_gen_online/ckpts/last.ckpt

Writes to testing.output_dir: ckpts/, normaliser.json, selftest.json, samples_epochNNNNN.npz, and
latents_fixed/{train,val}/latents_0.npz -- the fixed-view latents in extract_latents.py's format, so
PlatonicSplats' ``sample_latent_fm.py --samples ... --latents <output_dir>/latents_fixed`` works as before.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from mains.main_latent_gen_FM import LatentFMModel, LatentNormaliser
from platonic_transformers.utils.utils import subtract_mean
from platonic_transformers.utils.callbacks import EMACallback, TimerCallback
from platonic_transformers.utils.config_loader import get_arg_parser, load_with_defaults, print_config

KEYS = ("image", "c2w", "fxfycxcy", "index")


# ============================================================================================ autoencoder
def load_psplat(psplat_repo, ae_run, ae_ckpt, data_root, device):
    """Frozen autoencoder + train/val psplat datasets, loaded exactly like render_scene_video.py does."""
    path = Path(psplat_repo) / "scripts" / "visualization" / "render_scene_video.py"
    if not path.exists():
        raise SystemExit(f"{path} not found: --dataset.psplat_repo must be the PlatonicSplats checkout")
    spec = importlib.util.spec_from_file_location("render_scene_video", path)
    vid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vid)
    cfg, ckpt = vid._resolve_run(Path(ae_run), None, ae_ckpt)
    with torch.no_grad():
        ae, ds_train = vid.load_run(cfg, ckpt, data_root, "train", device)
        ds_val = None
        if (Path(data_root) / "val").is_dir():
            _ae_val, ds_val = vid.load_run(cfg, ckpt, data_root, "val", device)
            del _ae_val
    ae.eval()
    ae.requires_grad_(False)
    return ae, ds_train, ds_val, cfg, str(ckpt)


def _field(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


@torch.no_grad()
def pad_views(ae, raw):
    """Make a batch of exactly the chosen input views acceptable to psplat's ProcessData.

    ProcessData.fetch_views (psplat/utils/geometry.py) always splits a batch into inputs and targets:
    the encoder inputs are the FIRST training.num_input_views views, and (target_has_input=False) the
    targets are the views at positions num_views-1 ... num_views-num_target_views, so it needs
    training.num_views (10) views and asserts num_target_views < V. We only want the inputs, so the
    chosen views are put first and the batch is padded to num_views by repeating them; the padding only
    lands in the (unused) target dict. The encoder therefore sees exactly the views we chose, in order.
    """
    tr = ae.config.training
    n_in, v = int(tr.num_input_views), raw["image"].shape[1]
    if v != n_in:
        raise ValueError(f"expected exactly {n_in} input views, got {v}")
    n_tgt = int(tr.get("num_target_views", 0))
    n_total = max(int(tr.get("num_views", n_in)), n_in + n_tgt, n_tgt + 1)
    reps = -(-n_total // v)
    return {k: raw[k].repeat(1, reps, *([1] * (raw[k].ndim - 2)))[:, :n_total]
            for k in KEYS}


@torch.no_grad()
def encode_batch(ae, raw, amp=False):
    """raw: dict of [B,V,...] tensors (V = the chosen input views) on the AE's device
    -> (pos [B,N,3], feat [B,N,F]) fp32."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and raw["image"].is_cuda):
        inp, _ = ae.process_data(pad_views(ae, raw), has_target_image=False,
                                 target_has_input=ae.config.training.get("target_has_input", False),
                                 compute_rays=True)
        z_feat, pos_z, _ = ae.encode(inp)
    return pos_z.float(), z_feat.float()


@torch.no_grad()
def decode_psnr(ae, ds, sid, pos, feat, views, device):
    """PSNR of decoding one latent (pos [N,3], feat [N,F]) at `views` of scene `sid`."""
    ref = ds.scene_batch(int(sid), list(views), device=device)
    h, w = ref["image"].shape[-2:]
    z = feat[None].to(device)
    if ae.latent_mode == "invariant":
        from psplat.models.platoformer.io import lift_scalars
        z = lift_scalars(z, ae.group).flatten(-2)
    ray_o, ray_d = ae.process_data.compute_rays(ref["c2w"], ref["fxfycxcy"], h, w, device=device)
    rgb = ae.decode_views(z, pos[None].to(device), ray_o, ray_d, (h, w)).float().clamp(0, 1)
    gt = ref["image"].float()
    if gt.min() < 0:
        gt = (gt + 1) / 2
    mse = ((rgb - gt) ** 2).flatten(2).mean(-1).clamp_min(1e-10)
    return float((-10 * torch.log10(mse)).mean())


@torch.no_grad()
def render_latent(ae, pos, feat, cams, device):
    """Decode one latent (pos [N,3], feat [N,F], AE units) at cameras (c2w [1,V,4,4], fxfycxcy [1,V,4],
    image size) -> [V,3,H,W] in [0,1] on the CPU."""
    c2w, intr, (h, w) = cams
    z = torch.as_tensor(feat, dtype=torch.float32, device=device)[None]
    if ae.latent_mode == "invariant":
        from psplat.models.platoformer.io import lift_scalars
        z = lift_scalars(z, ae.group).flatten(-2)
    p = torch.as_tensor(pos, dtype=torch.float32, device=device)[None]
    ray_o, ray_d = ae.process_data.compute_rays(c2w, intr, h, w, device=device)
    return ae.decode_views(z, p, ray_o, ray_d, (h, w))[0].float().clamp(0, 1).cpu()


def feat_pos_gap(pos, feat, k=4, seed=0):
    """How much more similar each token's features are to its k spatial nearest neighbours than to k
    random tokens of the same scene (mean cosine similarity). Real latents ~0.5; features placed without
    regard to position ~0. pos [S,N,3], feat [S,N,F] (any units)."""
    g = torch.Generator().manual_seed(seed)
    p = torch.as_tensor(pos).float()
    f = torch.as_tensor(feat).float()
    f = f - f.mean((0, 1), keepdim=True)
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    S, N, _ = p.shape
    d = torch.cdist(p, p) + torch.eye(N)[None] * 1e9
    sim = torch.einsum("snf,smf->snm", f, f)
    near = torch.gather(sim, 2, d.topk(k, largest=False).indices).mean()
    rnd = torch.gather(sim, 2, torch.randint(0, N, (S, N, k), generator=g)).mean()
    return float(near - rnd)


def nearest_cloud(cloud, bank, chunk=256):
    """Index of the cloud in bank [M,N,3] nearest (symmetric chamfer) to cloud [N,3]."""
    best_d, best_i = float("inf"), -1
    for s in range(0, len(bank), chunk):
        b = bank[s:s + chunk]
        d = torch.cdist(cloud[None].expand(len(b), -1, -1), b)
        cd = 0.5 * (d.min(-1).values.mean(-1) + d.min(-2).values.mean(-1))
        j = int(cd.argmin())
        if float(cd[j]) < best_d:
            best_d, best_i = float(cd[j]), s + j
    return best_i


# ============================================================================================ data
class SceneViews(Dataset):
    """One item = one scene's images/cameras for ``n_views`` views (random each call, or fixed).

    Random views are drawn without replacement from the first ``trainable`` views (the hold-out views at
    the end are never used), i.e. the same pool the autoencoder's own training sampled from. Runs in
    DataLoader workers, so image loading overlaps with the GPU.
    """

    def __init__(self, ds, scene_ids, n_views, trainable, fixed_views=None, repeats=1, sort_views=False):
        self.ds, self.scene_ids = ds, list(scene_ids)
        self.n_views, self.trainable = int(n_views), int(trainable)
        self.fixed_views = None if fixed_views is None else list(fixed_views)
        self.repeats, self.sort_views = int(repeats), bool(sort_views)

    def __len__(self):
        return len(self.scene_ids) * self.repeats

    def views_for(self, i):
        if self.fixed_views is not None:
            return self.fixed_views
        v = torch.randperm(self.trainable)[: self.n_views].tolist()
        return sorted(v) if self.sort_views else v

    def __getitem__(self, i):
        sid = self.scene_ids[i % len(self.scene_ids)]
        views = self.views_for(i)
        b = self.ds.scene_batch(int(sid), views, device=torch.device("cpu"))
        item = {k: b[k][0] for k in KEYS}
        item["views"] = torch.tensor(views)
        item["sid"] = torch.tensor(int(sid))
        return item


def encode_all(ae, loader, device, amp):
    """Run a SceneViews loader through the encoder -> (pos [S,N,3], feat [S,N,F], sids [S]) on the CPU."""
    P, F, S = [], [], []
    for raw in loader:
        raw_d = {k: raw[k].to(device, non_blocking=True) for k in KEYS}
        p, f = encode_batch(ae, raw_d, amp)
        P.append(p.cpu()); F.append(f.cpu()); S.append(raw["sid"])
    return torch.cat(P), torch.cat(F), torch.cat(S)


def save_like_extract(root, split, pos, feat, sids, ds):
    out = Path(root) / split
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "latents_0.npz", pos=pos.numpy().astype(np.float32), feat=feat.numpy().astype(np.float32),
             scene_ids=sids.numpy(), scene_names=np.array([ds.scene_names[int(s)] for s in sids], dtype="U64"))


# ============================================================================================ self-test
@torch.no_grad()
def selftest(ae, ds, sid, n_views, trainable, n_avail, device, amp):
    """Stop before training if the encoder does not see exactly the views we give it.

    1. view control  : process_data keeps all n_views given views as encoder input (none dropped / moved
                       to the target side)
    2. determinism   : the same views encode to the same latent
    3. views matter  : different views give a different latent (else random views would be pointless)
    4. same frame    : latents from different random view subsets all decode the HOLD-OUT views (never
                       an input) about equally well, i.e. they live in the same scene frame. A latent from
                       another scene is the "wrong" reference.
    """
    report, fail = {}, []

    def raw_for(views):
        b = ds.scene_batch(int(sid), list(views), device=device)
        return {k: b[k] for k in KEYS}

    g = torch.Generator().manual_seed(0)
    subsets = [torch.randperm(trainable, generator=g)[:n_views].tolist() for _ in range(3)]
    A = subsets[0]

    # 1. view control
    raw = raw_for(A)
    inp, _ = ae.process_data(pad_views(ae, raw), has_target_image=False,
                             target_has_input=ae.config.training.get("target_has_input", False),
                             compute_rays=True)
    img = _field(inp, "image")
    if img is None:
        report["view_control"] = "could not find inp['image'] -- not checked"
        print(f"[selftest] CHECK view control: {report['view_control']}")
    else:
        n_in = int(img.shape[1])
        ok = n_in == n_views
        idx = _field(inp, "index")
        if ok and idx is not None and idx.ndim >= 3:
            got = idx[0, :, 0].long().tolist()
            ok = sorted(got) == sorted(A)
            report["view_control_index"] = got
        report["view_control"] = {"given": A, "encoder_inputs": n_in}
        print(f"[selftest] {'OK   ' if ok else 'FAIL '} view control: gave views {A}, encoder sees {n_in} views"
              + (f" {report.get('view_control_index')}" if "view_control_index" in report else ""))
        if not ok:
            fail.append(f"process_data turned the {n_views} given views into {n_in} encoder inputs "
                        f"(or different ones); random view selection would not be what the encoder sees")

    # 2. determinism
    p1, f1 = encode_batch(ae, raw_for(A), amp)
    p2, f2 = encode_batch(ae, raw_for(A), amp)
    d = max(float((p1 - p2).abs().max()), float((f1 - f2).abs().max()))
    tol = 1e-2 if amp else 1e-4
    report["determinism_maxdiff"] = d
    print(f"[selftest] {'OK   ' if d < tol else 'FAIL '} determinism: same views twice, max diff {d:.2e}")
    if d >= tol:
        fail.append(f"encoding the same views twice differs by {d:.2e}: the encoder is random "
                    "(dropout / jitter left on?)")

    # 3. views matter + order sensitivity
    pB, fB = encode_batch(ae, raw_for(subsets[1]), amp)
    dv = float((f1 - fB).abs().mean() / f1.abs().mean().clamp_min(1e-9))
    pR, fR = encode_batch(ae, raw_for(list(reversed(A))), amp)
    do = float((f1 - fR).abs().mean() / f1.abs().mean().clamp_min(1e-9))
    report.update(views_rel_change=dv, order_rel_change=do)
    print(f"[selftest] {'OK   ' if dv > 1e-3 else 'FAIL '} views matter: other views change the features "
          f"by {100 * dv:.1f}% (same views reversed: {100 * do:.1f}%)")
    if dv <= 1e-3:
        fail.append("different input views give the same latent")

    # 4. same frame: decode the hold-out views from latents of several subsets
    hold = list(range(trainable, n_avail)) or [n_avail - 1]
    ps = []
    for v in subsets:
        p, f = encode_batch(ae, raw_for(v), amp)
        ps.append(decode_psnr(ae, ds, sid, p[0], f[0], hold, device))
    other = int(ds.scene_indices[1 if int(ds.scene_indices[0]) == int(sid) else 0])
    b = ds.scene_batch(other, A, device=device)
    po, fo = encode_batch(ae, {k: b[k] for k in KEYS}, amp)
    ps_other = decode_psnr(ae, ds, sid, po[0], fo[0], hold, device)
    report.update(holdout_views=hold, holdout_psnr=ps, holdout_psnr_other_scene=ps_other)
    spread, gap = max(ps) - min(ps), min(ps) - ps_other
    ok = gap > 2.0
    print(f"[selftest] {'OK   ' if ok else 'CHECK'} same frame: hold-out views {hold} decoded from 3 random "
          f"subsets: {[round(x, 2) for x in ps]} dB (spread {spread:.2f}); from another scene: {ps_other:.2f} dB")
    if not ok:
        print("[selftest]       latents from random views do not decode the hold-out views much better than "
              "another scene's latent does -- look at this before trusting the run")
    return report, fail


# ============================================================================================ model
class OnlineLatentFMModel(LatentFMModel):
    """LatentFMModel whose training batches are raw images, encoded by a frozen autoencoder on the GPU."""

    def __init__(self, config, feat_dim, n_tokens, norm_state, ae=None):
        super().__init__(config, feat_dim, n_tokens, norm_state)
        # plain attribute (not a submodule): no parameters, optimiser state, EMA or checkpoint entries
        object.__setattr__(self, "ae", ae)
        self.encoder_amp = bool(config.dataset.get("encoder_amp", False))
        self.val_repeats = int(config.training.get("val_repeats", 4))
        self.real_tok = None    # real-latent token-size reference, set by set_token_reference

    # -- token-size statistics: the failure mode of the offline run ------------------------------------
    @staticmethod
    def token_sizes(feat_n):
        """Size of each token's normalised feature vector, scaled so a typical real token is ~1."""
        return feat_n.norm(dim=-1) / feat_n.shape[-1] ** 0.5

    def set_token_reference(self, feat_n_real):
        size = self.token_sizes(feat_n_real)
        self.real_tok = {"p99": float(torch.quantile(size.flatten()[:2_000_000], 0.99)),
                         "big": float((size > 2).sum(-1).float().median())}

    def set_gap_reference(self, pos, feat):
        self.real_tok["gap"] = feat_pos_gap(pos[:64], feat[:64])

    def _evaluate_samples(self, count, phase):
        """The parent's evaluation (samples, copying check, feature spread) + token-size statistics.

        The offline run went wrong through a growing tail of oversized tokens (token-size p99 1.9 real
        vs 5.4 at epoch 2000) while everything else looked fine, so these are logged at every sample check:
          token_p99         99th percentile of token size (real: ~1.9)
          big_tokens        median number of tokens per scene with size > 2 (real: a few)
          token_p99_ratio   token_p99 / real; above ~1.3 the renders start to break up
        """
        super()._evaluate_samples(count, phase)
        out_dir = Path(self.config.testing.output_dir)
        name = "samples.npz" if phase == "test" else f"samples_epoch{self.current_epoch + 1:05d}.npz"
        z = np.load(out_dir / name)
        feat_n = (torch.from_numpy(z["feat"]).float() - self.normer.feat_mean) / self.normer.feat_std
        size = self.token_sizes(feat_n)
        res = {f"{phase}/token_p99": float(torch.quantile(size.flatten(), 0.99)),
               f"{phase}/big_tokens": float((size > 2).sum(-1).float().median())}
        res[f"{phase}/feat_pos_gap"] = feat_pos_gap(z["pos"], z["feat"])
        if self.real_tok:
            res[f"{phase}/token_p99_ratio"] = res[f"{phase}/token_p99"] / self.real_tok["p99"]
        for k, v in res.items():
            self.log(k, v, on_epoch=True, logger=True)
        ref = (f" (real: p99 {self.real_tok['p99']:.2f}, big {self.real_tok['big']:.0f}, "
               f"feat_pos_gap {self.real_tok.get('gap', float('nan')):.3f})" if self.real_tok else "")
        print(f"[{phase}] tokens: {json.dumps({k: round(v, 3) for k, v in res.items()})}{ref}")
        self._render_samples(z, phase)

    # -- pictures of the generated scenes at every sample check ----------------------------------------
    def set_render_context(self, ds, ref_sid, views, train_latents, train_pos_n):
        """Cameras to render generated scenes from (a real scene's rig: every scene shares the normalised
        frame), plus the training latents for the nearest-training-scene rows."""
        object.__setattr__(self, "render_ctx", {"ds": ds, "sid": int(ref_sid), "views": list(views),
                                                "train": train_latents, "train_pos_n": train_pos_n})

    @torch.no_grad()
    def _render_samples(self, z, phase):
        ctx = getattr(self, "render_ctx", None)
        k = int(self.config.training.get("num_render", 4))
        if ctx is None or self.ae is None or k <= 0:
            return
        try:
            dev = self.device
            ref = ctx["ds"].scene_batch(ctx["sid"], ctx["views"], device=dev)
            cams = (ref["c2w"], ref["fxfycxcy"], tuple(ref["image"].shape[-2:]))
            gt = ref["image"][0].float().cpu()
            gt = (gt + 1) / 2 if gt.min() < 0 else gt
            rows = [gt]                                                    # row 0: the real camera rig
            tr_pos, tr_feat = ctx["train"]
            for i in range(min(k, len(z["pos"]))):
                rows.append(render_latent(self.ae, z["pos"][i], z["feat"][i], cams, dev))
                p_n = torch.from_numpy(z["pos"][i]).float()
                p_n = (p_n - p_n.mean(0)) / self.normer.pos_std
                j = nearest_cloud(p_n.to(dev), ctx["train_pos_n"].to(dev))
                rows.append(0.55 * render_latent(self.ae, tr_pos[j], tr_feat[j], cams, dev))
            grid = torch.cat([torch.cat(list(r), -1) for r in rows], -2)
            img = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            out = Path(self.config.testing.output_dir) / "renders"
            out.mkdir(parents=True, exist_ok=True)
            name = "test.png" if phase == "test" else f"epoch{self.current_epoch + 1:05d}.png"
            from PIL import Image
            Image.fromarray(img).save(out / name)
            print(f"[{phase}] renders -> {out / name}  (row 1: the real scene whose cameras are used; then per "
                  f"generated scene: its render, and DIMMED below it the nearest training scene)")
        except Exception as e:                                             # never kill a long run over a picture
            print(f"[{phase}] rendering skipped: {type(e).__name__}: {e}")

    def _move_ae(self):
        if self.ae is not None:
            self.ae.to(self.device)

    def on_fit_start(self):
        self._move_ae()

    def on_test_start(self):
        self._move_ae()

    def _normalise(self, pos, feat):
        dev = pos.device
        pos_c = (pos - pos.mean(-2, keepdim=True)) / self.normer.pos_std.to(dev)
        return pos_c, (feat - self.normer.feat_mean.to(dev)) / self.normer.feat_std.to(dev)

    @staticmethod
    def _to_graph(pos, feat):
        B, N, _ = pos.shape
        return {"pos": pos.reshape(B * N, 3), "x": feat.reshape(B * N, -1),
                "batch": torch.arange(B, device=pos.device).repeat_interleave(N)}

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch, dict) and "image" in batch:          # raw training batch -> fresh latents
            pos, feat = encode_batch(self.ae, batch, self.encoder_amp)
            return self._to_graph(*self._normalise(pos, feat))
        return batch                                               # val: pre-encoded, already a graph

    # -- pos_mode "given": flow on FEATURES only, conditioned on the real (clean) positions ----------------
    @property
    def pos_given(self):
        return str(self.config.diffusion.get("pos_mode", "joint")) == "given"

    def _velocity_given(self, x_t, pos, batch, t):
        """Same EDM-preconditioned feature velocity as _velocity, but the positions enter CLEAN (not noised,
        not c_in-scaled): the model only has to choose features for known token positions."""
        sigma_data = self.config.diffusion.sigma_data
        t_graph = t.reshape(-1, 1)
        denom = t_graph ** 2 * sigma_data ** 2 + (1 - t_graph) ** 2
        c_in, c_out = 1 / denom.sqrt(), sigma_data / denom.sqrt()
        c_skip = (t_graph * sigma_data ** 2 - (1 - t_graph)) / denom
        c_noise = (((1 - t_graph) / t_graph).log()) / 4
        c_in_n, c_skip_n, c_out_n, c_noise_n = (v[batch] for v in (c_in, c_skip, c_out, c_noise))
        scalar_out, _ = self.net(torch.cat([c_in_n * x_t, c_noise_n], dim=-1), pos, batch, vec=None,
                                 avg_num_nodes=self.config.diffusion.avg_num_nodes)
        return c_skip_n * x_t + c_out_n * scalar_out

    def _flow_loss(self, batch, augment=False):
        if not self.pos_given:
            return super()._flow_loss(batch, augment)
        pos = subtract_mean(batch["pos"], batch["batch"])
        if augment:
            pos = self._augment(pos, batch["batch"])
        x = batch["x"]
        t = self._sample_t(int(batch["batch"].max().item()) + 1, pos.device)
        x0 = torch.randn_like(x)
        node_t = t[batch["batch"], None]
        vx = self._velocity_given((1 - node_t) * x0 + node_t * x, pos, batch["batch"], t)
        loss_x = (self._edm_weight(node_t) * (vx - (x - x0)).square()).mean()
        return loss_x, loss_x, torch.zeros_like(loss_x)

    @torch.no_grad()
    def sample(self, num_scenes, num_steps=None, **kw):
        if not self.pos_given:
            return super().sample(num_scenes, num_steps, **kw)
        # positions: real (normalised, centred) training-scene positions, drawn at random
        bank = self.ref_train
        idx = torch.randint(0, len(bank), (num_scenes,))
        pos = bank[idx].to(self.device).reshape(-1, 3)
        batch = torch.arange(num_scenes, device=self.device).repeat_interleave(self.n_tokens)
        num_steps = num_steps or self.config.diffusion.num_steps
        t_eps = self.config.diffusion.get("fm_t_eps", 1e-4)
        x = torch.randn((len(batch), self.feat_dim), device=self.device)
        ts = torch.linspace(0.0, 1.0, num_steps + 1).tolist()

        def vel(x, t):
            return self._velocity_given(x, pos, batch, torch.full((num_scenes,), float(t),
                                                                   device=self.device).clamp(t_eps, 1 - t_eps))
        for i in range(num_steps):
            dt = ts[i + 1] - ts[i]
            v1 = vel(x, ts[i])
            x_next = x + dt * v1
            if i < num_steps - 1:
                x_next = x + dt * 0.5 * (v1 + vel(x_next, ts[i + 1]))
            x = x_next
        return self._to_scenes(x, pos, batch, num_scenes)

    def validation_step(self, batch, batch_idx):
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(1000 + batch_idx)
            losses = [self._flow_loss(batch)[0] for _ in range(self.val_repeats)]
        loss = torch.stack(losses).mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=int(batch["batch"].max().item()) + 1)
        return loss


class GraphLatents(Dataset):
    def __init__(self, pos_n, feat_n):
        self.pos, self.feat = pos_n, feat_n

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):
        return {"pos": self.pos[i], "x": self.feat[i]}


def collate_graph(items):
    n = items[0]["pos"].shape[0]
    return {"pos": torch.cat([it["pos"] for it in items]), "x": torch.cat([it["x"] for it in items]),
            "batch": torch.arange(len(items)).repeat_interleave(n)}


# ============================================================================================ main
def main(config):
    print_config(config, "Online Latent Flow Matching Configuration")
    dc = config.dataset
    for k in ("psplat_repo", "ae_run", "data_root"):
        if not dc.get(k):
            raise SystemExit(f"set --dataset.{k}=...")
    pl.seed_everything(config.seed, workers=True)
    device = torch.device("cuda" if config.system.gpus > 0 and torch.cuda.is_available() else "cpu")
    out_dir = Path(config.testing.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ae, ds_tr, ds_va, ae_cfg, ae_ckpt = load_psplat(dc.psplat_repo, dc.ae_run, dc.get("ae_ckpt"),
                                                    dc.data_root, device)
    n_views = int(dc.get("n_input_views") or ae_cfg.training.num_input_views)
    if n_views != int(ae.config.training.num_input_views):
        raise SystemExit(f"n_input_views={n_views}, but the autoencoder's ProcessData always encodes its first "
                         f"training.num_input_views={ae.config.training.num_input_views} views; leave it null")
    if ae.config.training.get("dynamic_input_view_num", False):
        raise SystemExit("the autoencoder config has training.dynamic_input_view_num=true: ProcessData would "
                         "pick a random number of input views. Set it to false in the run's config.yaml")
    n_avail = int(ds_tr.num_views_avail)
    holdout = int(ae_cfg.training.get("holdout_views", 0) or 0)
    trainable = n_avail - holdout
    val_views = list(dc.get("val_views") or range(n_views))
    amp = bool(dc.get("encoder_amp", False))
    print(f"autoencoder: {ae_ckpt}\n  encoder input: {n_views} views, random from views 0-{trainable - 1} "
          f"(views {trainable}-{n_avail - 1} are hold-out, never used); fixed val/reference views {val_views}")

    train_ids = [int(s) for s in ds_tr.scene_indices]
    frac = float(dc.get("dataset_fraction", 1.0))
    if frac < 1.0:
        train_ids = train_ids[: max(1, int(len(train_ids) * frac))]
    val_ids = [int(s) for s in ds_va.scene_indices] if ds_va is not None else []

    # ---- self-test (stops on failure)
    if dc.get("selftest", True):
        report, fail = selftest(ae, ds_tr, train_ids[0], n_views, trainable, n_avail, device, amp)
        (out_dir / "selftest.json").write_text(json.dumps(report, indent=2, default=float))
        if fail:
            raise SystemExit("self-test failed, not training:\n  - " + "\n  - ".join(fail))

    nw = int(config.system.num_workers)
    eval_kw = dict(batch_size=int(config.training.batch_size), num_workers=nw, shuffle=False)

    # ---- fixed-view latents (val set + copy-check reference) and normalisation statistics
    test_ckpt = config.testing.test_ckpt
    print("encoding fixed-view latents (train reference" + (", val" if val_ids else "") + ") ...")
    fix_tr = encode_all(ae, DataLoader(SceneViews(ds_tr, train_ids, n_views, trainable, val_views), **eval_kw),
                        device, amp)
    save_like_extract(out_dir / "latents_fixed", "train", *fix_tr, ds_tr)
    fix_va = None
    if val_ids:
        fix_va = encode_all(ae, DataLoader(SceneViews(ds_va, val_ids, n_views, trainable, val_views), **eval_kw),
                            device, amp)
        save_like_extract(out_dir / "latents_fixed", "val", *fix_va, ds_va)

    if test_ckpt:
        ck = torch.load(test_ckpt, map_location="cpu", weights_only=False)
        norm_state = ck["hyper_parameters"]["norm_state"]
        print(f"sampling mode: normalisation from {test_ckpt}")
    else:
        P, F = [fix_tr[0]], [fix_tr[1]]
        for k in range(int(dc.get("norm_passes", 2))):
            rnd = encode_all(ae, DataLoader(SceneViews(ds_tr, train_ids, n_views, trainable,
                                                       sort_views=dc.get("sort_views", False)), **eval_kw),
                             device, amp)
            P.append(rnd[0]); F.append(rnd[1])
        normer = LatentNormaliser(torch.cat(P), torch.cat(F))
        norm_state = normer.state()
        print(f"normalisation from {sum(len(p) for p in P)} latents (1 fixed-view + "
              f"{len(P) - 1} random-view passes over {len(train_ids)} scenes)")
    (out_dir / "normaliser.json").write_text(json.dumps(norm_state))
    N, F_dim = fix_tr[1].shape[1], fix_tr[1].shape[2]

    model = OnlineLatentFMModel(config, feat_dim=F_dim, n_tokens=N, norm_state=norm_state, ae=ae)
    pos_tr_n, _ = model._normalise(fix_tr[0], fix_tr[1])
    val_loader, pos_va_n = None, None
    if fix_va is not None:
        pos_va_n, feat_va_n = model._normalise(fix_va[0], fix_va[1])
        val_loader = DataLoader(GraphLatents(pos_va_n, feat_va_n), batch_size=int(config.training.batch_size),
                                shuffle=False, collate_fn=collate_graph)
    model.set_reference(pos_tr_n, pos_va_n)
    model.set_token_reference(model._normalise(fix_tr[0], fix_tr[1])[1])
    model.set_gap_reference(fix_tr[0], fix_tr[1])
    if model.pos_given:
        print("pos_mode=given: the flow model generates FEATURES for real token positions (drawn from "
              "training scenes at sampling time). copy_ratio is meaningless in this mode; feat_pos_gap is "
              "the number to watch")
    ds_ref = ds_va if ds_va is not None else ds_tr
    ref_sid = int(ds_ref.scene_indices[0])
    n_cols = int(config.training.get("render_views", 6))
    model.set_render_context(ds_ref, ref_sid, list(range(0, n_avail, max(1, n_avail // n_cols)))[:n_cols],
                             (fix_tr[0], fix_tr[1]), pos_tr_n)
    print(f"real token sizes: p99 {model.real_tok['p99']:.2f}, median {model.real_tok['big']:.0f} tokens "
          f"per scene above 2")
    print(f"flow model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params "
          f"(autoencoder frozen, outside the model) | copying-check reference: {model.ref_val_dist}")

    train_ds = SceneViews(ds_tr, train_ids, n_views, trainable, repeats=int(dc.get("repeats_per_epoch", 1)),
                          sort_views=dc.get("sort_views", False))
    train_loader = DataLoader(train_ds, batch_size=int(config.training.batch_size), shuffle=True, drop_last=True,
                              num_workers=nw, persistent_workers=nw > 0, pin_memory=device.type == "cuda")

    accelerator = "gpu" if device.type == "cuda" else "cpu"
    logger = False
    if config.logging.enabled:
        logger = pl.loggers.WandbLogger(project=config.logging.project_name,
                                        entity=config.logging.get("wandb_identity", None),
                                        config=config.to_dict(), save_dir=str(out_dir))
    # keep EVERY periodic checkpoint (save_top_k=-1): in the offline run the best samples came early
    # (epoch ~200) and only the last checkpoint survived
    callbacks = [pl.callbacks.ModelCheckpoint(dirpath=str(out_dir / "ckpts"), save_last=True,
                                              save_top_k=-1 if config.training.get("keep_all_ckpts", True) else 1,
                                              filename="epoch{epoch:05d}", auto_insert_metric_name=False,
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
    eval_loader = val_loader if val_loader is not None else DataLoader(
        GraphLatents(pos_tr_n, model._normalise(fix_tr[0], fix_tr[1])[1]), batch_size=8, collate_fn=collate_graph)
    if test_ckpt:
        trainer.test(model, eval_loader, ckpt_path=test_ckpt)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.testing.resume_ckpt)
        trainer.test(model, eval_loader, ckpt_path="last")


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/latent_gen_online.yaml")
    args, unknown_args = parser.parse_known_args()
    main(load_with_defaults(dataset_config=args.config, cli_args=unknown_args))