"""Novel view synthesis on an objascenes / OVS split with a Platonic Transformer backbone.

An LVSM-style model (context views + target camera -> target image) in which the plain transformer is
replaced by a PlatonicTransformer, so the model is equivariant to the chosen rotation group and to
translations *by construction* -- no canonical scene frame needed.

Tokens (one per PxP image patch, all views of a scene in one sequence):
    pos     [3]        a point on the patch's centre ray at the depth of the scene centre (RoPE position)
    vec     [5, 3]     unit ray directions: patch centre + 4 patch corners        (rotate with the scene)
    scalars            patch RGB (context) | zeros or noisy RGB (target),
                       + context/target flag, + ray depth, + noise level (flow)  (rotation-invariant)
The network outputs PxPx3 scalars per token; the target tokens are the rendered patches.

Two objectives (--mode):
    regression   deterministic, MSE on target pixels -- the direct LVSM comparison
    flow         conditional flow matching on the target pixels (straight path, EDM-style
                 preconditioning as in main_qm9_gen_FM.py, logit-normal time schedule);
                 sampled from pure noise with a Heun ODE solver.
                 Needs several times more steps than regression: on a toy overfit test, samples stayed
                 noise until the loss fell below ~0.1, then reached ~20 dB. Watch val_psnr, not only loss.

Usage (from the repo root, venv active):
    python mains/main_ovs_nvs.py --train /data/free/train/manifest.json --val /data/free/val/manifest.json
    python mains/main_ovs_nvs.py --train .../manifest.json --mode flow --solid octahedron
    python mains/main_ovs_nvs.py --train .../manifest.json --eval_only --ckpt runs/nvs/last.pt --rot_eval
Run `--help` for every option. Outputs (checkpoints, sample grids, metrics.jsonl) go to --out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> platonic_transformers
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS  # noqa: E402
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer  # noqa: E402


# ============================================================================================ data
class OVSViews(torch.utils.data.Dataset):
    """Scenes of an objascenes split -> (context + target) views with cameras, in a centred frame.

    Only images/ + poses/ (+ specs/ for intrinsics) are read, so any --gt level works.
    The frame is translated to the mean of the selected cameras and scaled so the mean camera
    distance is 1. It is deliberately NOT rotated into a canonical orientation: the equivariant
    backbone makes that unnecessary, and --rot_aug / --rot_eval test exactly that.
    """

    def __init__(self, manifest, num_context, num_target, scenes=None, fixed_views=False,
                 rotate=False, seed=0):
        manifest = Path(manifest).expanduser().resolve()
        self.dir = manifest.parent
        m = json.loads(manifest.read_text())
        self.num_views_avail = int(m["num_views"])
        self.hw = tuple(m["image_size"])
        self.nc, self.nt = num_context, num_target
        if self.nc + self.nt > self.num_views_avail:
            raise ValueError(f"{self.nc}+{self.nt} views requested, scenes only have {self.num_views_avail}")
        c2w, names, K = [], [], []
        for shard in m["shards"]:
            with np.load(self.dir / shard["path"]) as f:
                c2w.append(f["c2w"].astype(np.float32))
                names += [str(n) for n in f["scene_names"]]
            spec = self.dir / shard["path"].replace("poses/", "specs/")
            if spec.exists():
                with np.load(spec) as f:
                    K.append(f["fxfycxcy"].astype(np.float32))
            else:
                K.append(np.tile(np.asarray(m["intrinsics_fxfycxcy"], np.float32), (int(shard["num"]), 1)))
        self.c2w, self.K, self.names = np.concatenate(c2w), np.concatenate(K), names
        self.scenes = list(range(len(names))) if scenes is None else list(scenes)
        self.fixed_views, self.rotate, self.seed = fixed_views, rotate, seed

    def __len__(self):
        return len(self.scenes)

    def _views(self, idx):
        V = self.num_views_avail
        if self.fixed_views:  # deterministic eval: first nc views as context, last nt as targets
            return list(range(self.nc)) + list(range(V - self.nt, V))
        return random.sample(range(V), self.nc + self.nt)

    def __getitem__(self, idx):
        s = self.scenes[idx]
        views = self._views(idx)
        imgs = np.stack([np.asarray(Image.open(self.dir / "images" / self.names[s] / f"{v:02d}.png")
                                    .convert("RGB")) for v in views])
        c2w = torch.from_numpy(self.c2w[s, views]).clone()               # [V,4,4] OpenCV cam->world
        center = c2w[:, :3, 3].mean(0)
        c2w[:, :3, 3] -= center
        c2w[:, :3, 3] /= c2w[:, :3, 3].norm(dim=-1).mean().clamp_min(1e-6)
        if self.rotate:  # random SO(3) rotation of the whole scene (fixed per scene when fixed_views)
            g = torch.Generator().manual_seed(self.seed * 100003 + s) if self.fixed_views else None
            R = random_rotation(g)
            c2w[:, :3, :] = R @ c2w[:, :3, :]
        return {
            "image": torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0,  # [V,3,H,W] in [0,1]
            "c2w": c2w,
            "K": torch.from_numpy(self.K[s]).expand(len(views), 4).clone(),  # [V,4] fx fy cx cy
            "scene": s,
        }


def random_rotation(generator=None):
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r))
    return q * torch.sign(torch.det(q))


def split_scenes(n, val_frac, seed=0):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(1, int(round(n * val_frac)))
    return sorted(idx[k:]), sorted(idx[:k])


# ============================================================================================ tokens
def patch_tokens(c2w, K, hw, p):
    """Per-patch geometry. c2w [B,V,4,4], K [B,V,4] -> pos [B,V,n,3], vec [B,V,n,5,3], depth [B,V,n,1]."""
    H, W = hw
    dev = c2w.device
    ys = torch.arange(0, H, p, device=dev, dtype=torch.float32)
    xs = torch.arange(0, W, p, device=dev, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")                     # patch top-left pixel corners
    gy, gx = gy.reshape(-1), gx.reshape(-1)
    # centre + 4 corners of each patch, in pixel coordinates (pixel edges, not centres)
    offs = torch.tensor([[p / 2, p / 2], [0, 0], [p, 0], [0, p], [p, p]], device=dev)
    u = gx[:, None] + offs[None, :, 0]                                 # [n,5]
    v = gy[:, None] + offs[None, :, 1]
    fx, fy, cx, cy = [K[..., i][..., None, None] for i in range(4)]    # [B,V,1,1]
    d_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u - cx)], -1)  # [B,V,n,5,3]
    R, o = c2w[..., :3, :3], c2w[..., :3, 3]
    d = F.normalize(torch.einsum("bvij,bvnkj->bvnki", R, d_cam), dim=-1)
    depth = o.norm(dim=-1)[..., None, None].expand(*d.shape[:3], 1)   # camera -> scene-centre distance
    pos = o[:, :, None, :] + depth * d[..., 0, :]                      # point on the centre ray
    return pos, d, depth


def patchify(img, p):   # [B,V,3,H,W] -> [B,V,n,p*p*3]
    B, V, C, H, W = img.shape
    x = img.reshape(B, V, C, H // p, p, W // p, p).permute(0, 1, 3, 5, 4, 6, 2)
    return x.reshape(B, V, (H // p) * (W // p), p * p * C)


def unpatchify(x, hw, p):  # [B,V,n,p*p*3] -> [B,V,3,H,W]
    B, V, n, _ = x.shape
    H, W = hw
    x = x.reshape(B, V, H // p, W // p, p, p, 3).permute(0, 1, 6, 2, 4, 3, 5)
    return x.reshape(B, V, 3, H, W)


# ============================================================================================ model
class PlatonicNVS(nn.Module):
    def __init__(self, args, hw):
        super().__init__()
        self.args, self.hw, self.p = args, hw, args.patch_size
        G = PLATONIC_GROUPS[args.solid].G
        self.pix = self.p * self.p * 3
        in_scalars = self.pix + 3   # rgb, is_context flag, ray depth, noise level
        self.net = PlatonicTransformer(
            input_dim=in_scalars, input_dim_vec=5, hidden_dim=args.dim_per_g * G, output_dim=self.pix,
            output_dim_vec=0, nhead=args.heads_per_g * G, num_layers=args.layers, solid_name=args.solid,
            spatial_dim=3, dense_mode=True, scalar_task_level="node", vector_task_level="node",
            ffn_readout=True, attention=True, dropout=0.0, drop_path_rate=args.drop_path,
            rope_sigma=args.rope_sigma, learned_freqs=True, freq_init="spiral", rope_on_values=True,
            qk_norm=args.qk_norm,
        )
        # Zero-init the last readout layer (as in DiT): the model starts by predicting 0 (= grey in
        # regression, zero velocity in flow) instead of a large random offset, and the residual stream
        # is free to differentiate tokens before the head commits to anything.
        last = self.net.scalar_readout[-1] if isinstance(self.net.scalar_readout, nn.Sequential) \
            else self.net.scalar_readout
        nn.init.zeros_(last.kernel)
        if getattr(last, "bias", None) is not None:
            nn.init.zeros_(last.bias)
        if args.compile:
            self.net = torch.compile(self.net)

    def forward(self, batch, target_scalars, noise_level):
        """target_scalars [B,Vt,n,pix] (zeros or noisy rgb), noise_level [B,1] -> output [B,Vt,n,pix]."""
        a, p = self.args, self.p
        img, c2w, K = batch["image"], batch["c2w"], batch["K"]
        B, V = img.shape[:2]
        pos, vec, depth = patch_tokens(c2w, K, self.hw, p)
        n = pos.shape[2]
        ctx_rgb = patchify(img[:, : a.num_context], p) * 2 - 1                 # [-1,1]
        rgb = torch.cat([ctx_rgb, target_scalars], dim=1)                       # [B,V,n,pix]
        is_ctx = torch.zeros(B, V, n, 1, device=img.device)
        is_ctx[:, : a.num_context] = 1
        noise = torch.zeros(B, V, n, 1, device=img.device)
        noise[:, a.num_context:] = noise_level[:, None, None, :]
        x = torch.cat([rgb, is_ctx, depth, noise], -1).reshape(B, V * n, -1)
        out, _ = self.net(x, pos.reshape(B, V * n, 3), None, vec=vec.reshape(B, V * n, 5, 3),
                          return_vectors=False)
        return out.reshape(B, V, n, -1)[:, a.num_context:].float()


# ============================================================================================ objectives
def regression_loss(model, batch):
    a = model.args
    tgt = patchify(batch["image"][:, a.num_context:], model.p) * 2 - 1
    B = tgt.shape[0]
    pred = model(batch, torch.zeros_like(tgt), torch.zeros(B, 1, device=tgt.device))
    if a.bounded_output:  # tanh keeps pixels in [-1,1], like LVSM's sigmoid keeps them in [0,1]
        pred = torch.tanh(pred)
    return F.mse_loss(pred, tgt), pred


def fm_coeffs(t, sd):
    """EDM-style preconditioning for a velocity target (same derivation as main_qm9_gen_FM.py)."""
    d = t ** 2 * sd ** 2 + (1 - t) ** 2
    return d.rsqrt(), (t * sd ** 2 - (1 - t)) / d, sd * d.rsqrt(), d / sd ** 2  # c_in, c_skip, c_out, weight


def fm_velocity(model, batch, z_t, t):
    sd = model.args.sigma_data
    tt = t[:, None, None, None]
    c_in, c_skip, c_out, _ = fm_coeffs(tt, sd)
    tc = t.clamp(1e-4, 1 - 1e-4)
    c_noise = 0.25 * torch.log((1 - tc) / tc)[:, None]
    return c_skip * z_t + c_out * model(batch, c_in * z_t, c_noise)


def flow_loss(model, batch):
    a = model.args
    z1 = patchify(batch["image"][:, a.num_context:], model.p) * 2 - 1
    B = z1.shape[0]
    if a.t_dist == "edm":   # QM9-FM schedule: EDM log-normal sigma, t = 1/(1+sigma). Rarely visits
        sigma = torch.exp(torch.randn(B, device=z1.device) * a.P_std + a.P_mean)  # t<0.3 (~4%), so
        t = 1.0 / (1.0 + sigma)                                  # pair it with --t_start 0.5
    else:                   # logit-normal (SD3-style rectified flow): covers the whole path
        t = torch.sigmoid(torch.randn(B, device=z1.device) * a.t_std + a.t_mean)
    z0 = torch.randn_like(z1)
    tt = t[:, None, None, None]
    z_t = (1 - tt) * z0 + tt * z1
    v = fm_velocity(model, batch, z_t, t)
    w = fm_coeffs(tt, a.sigma_data)[3]
    return (w * (v - (z1 - z0)) ** 2).mean(), None


@torch.no_grad()
def flow_sample(model, batch, steps, t_start=0.0):
    a = model.args
    B = batch["image"].shape[0]
    n = (model.hw[0] // model.p) * (model.hw[1] // model.p)
    z = torch.randn(B, a.num_target, n, model.pix, device=batch["image"].device)
    ts = torch.linspace(t_start, 1.0, steps + 1, device=z.device)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        v0 = fm_velocity(model, batch, z, t0.expand(B))
        z1 = z + (t1 - t0) * v0
        if i < steps - 1:  # Heun correction (skipped on the last step, as in the repo)
            v1 = fm_velocity(model, batch, z1, t1.expand(B))
            z1 = z + (t1 - t0) * 0.5 * (v0 + v1)
        z = z1
    return z


@torch.no_grad()
def render(model, batch):
    """Predicted target images [B,Vt,3,H,W] in [0,1]."""
    if model.args.mode == "flow":
        patches = flow_sample(model, batch, model.args.sample_steps, model.args.t_start)
    else:
        _, patches = regression_loss(model, batch)
    return ((unpatchify(patches, model.hw, model.p) + 1) / 2).clamp(0, 1)


def psnr(pred, gt):
    mse = ((pred - gt) ** 2).flatten(2).mean(-1).clamp_min(1e-10)   # per target view
    return (-10 * torch.log10(mse)).mean().item()


# ============================================================================================ loops
def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, amp, max_batches=None, save_grid=None):
    model.eval()
    scores, base, first = [], [], None
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred = render(model, batch)
        gt = batch["image"][:, model.args.num_context:]
        scores.append(psnr(pred.float(), gt))
        # reference: just averaging the context images (a model must beat this to have learned anything)
        ctx_mean = batch["image"][:, : model.args.num_context].mean(1, keepdim=True).expand_as(gt)
        base.append(psnr(ctx_mean, gt))
        if first is None:
            first = (batch["image"], pred)
    if save_grid and first is not None:
        save_sample_grid(*first, model.args.num_context, save_grid)
    model.train()
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.mean(base))


def save_sample_grid(images, pred, nc, path, max_rows=4):
    """One row per scene: context views | target GT | prediction."""
    rows = []
    for b in range(min(max_rows, images.shape[0])):
        tiles = list(images[b, :nc]) + list(images[b, nc:]) + list(pred[b])
        rows.append(torch.cat([t.float().cpu() for t in tiles], dim=-1))
    grid = (torch.cat(rows, dim=-2).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid).save(path)


def build_loaders(a):
    train_m = json.loads(Path(a.train).expanduser().read_text())
    n_train = int(train_m.get("num_scenes", sum(int(s["num"]) for s in train_m["shards"])))
    if a.val:
        tr_scenes = None
        va_ds_manifest, va_scenes = a.val, None
    else:  # no val split given: hold out a fraction of the training scenes
        tr_scenes, va_scenes = split_scenes(n_train, a.val_frac, a.seed)
        va_ds_manifest = a.train
    if a.scene_limit:
        tr_scenes = (tr_scenes or list(range(n_train)))[: a.scene_limit]
    train_ds = OVSViews(a.train, a.num_context, a.num_target, tr_scenes, fixed_views=False,
                        rotate=a.rot_aug, seed=a.seed)
    val_ds = OVSViews(va_ds_manifest, a.num_context, a.num_target, va_scenes, fixed_views=True,
                      rotate=a.rot_eval, seed=a.seed)
    if a.val_limit:
        val_ds.scenes = val_ds.scenes[: a.val_limit]
    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    train = torch.utils.data.DataLoader(train_ds, a.batch_size, shuffle=True, drop_last=True, **kw)
    val = torch.utils.data.DataLoader(val_ds, a.batch_size, shuffle=False, **kw)
    return train, val, train_ds


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    amp = device.type == "cuda" and not a.no_amp
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    train_loader, val_loader, train_ds = build_loaders(a)
    hw = train_ds.hw
    if hw[0] % a.patch_size or hw[1] % a.patch_size:
        raise ValueError(f"image size {hw} not divisible by patch_size {a.patch_size}")
    if a.rope_sigma <= 0:
        # auto: neighbouring patches sit ~patch/focal apart (in units of the camera-centre distance,
        # which the dataset normalises to 1). Make the highest RoPE frequency resolve that spacing,
        # otherwise attention cannot tell adjacent target patches apart and predicts one mean colour.
        fx = float(train_ds.K[:, 0].mean())
        a.rope_sigma = fx / a.patch_size
        print(f"rope_sigma auto -> {a.rope_sigma:.1f} (focal {fx:.1f}px / patch {a.patch_size}px)")
    model = PlatonicNVS(a, hw).to(device)
    n_tok = (a.num_context + a.num_target) * (hw[0] // a.patch_size) * (hw[1] // a.patch_size)
    print(f"{len(train_ds)} train scenes, {len(val_loader.dataset)} val scenes | image {hw} | "
          f"{n_tok} tokens/scene | {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | "
          f"mode={a.mode} solid={a.solid} | device={device} amp={amp}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step = 0
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        if not a.eval_only and "opt" in ck:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]
        print(f"loaded {a.ckpt} (step {ck.get('step')})")

    if a.eval_only:
        score, base = evaluate(model, val_loader, device, amp, save_grid=out / "eval_samples.png")
        print(json.dumps({"val_psnr": score, "baseline_psnr": base, "rot_eval": a.rot_eval}))
        return

    loss_fn = flow_loss if a.mode == "flow" else regression_loss
    log = open(out / "metrics.jsonl", "a")
    model.train()
    t0, running = time.time(), []
    loss_ema, skipped, gnorm = None, 0, torch.tensor(0.0)
    while step < a.steps:
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, _ = loss_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            lv = loss.item()
            # Spike guard: a batch whose loss is far above the recent typical value (or not finite)
            # is skipped instead of being allowed to wreck the weights. Logged, with its scenes.
            spike = (not math.isfinite(lv)) or (loss_ema is not None and step > a.warmup
                                                and lv > a.spike_factor * loss_ema)
            if spike and a.spike_factor > 0:
                skipped += 1
                print(json.dumps({"step": step, "skipped_spike_loss": lv, "typical_loss": loss_ema,
                                  "scenes": [int(x) for x in batch["scene"]]}))
                sched.step(); step += 1
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step(); sched.step(); step += 1
                running.append(lv)
                loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            if step % a.log_every == 0:
                rec = {"step": step, "loss": float(np.mean(running)) if running else float("nan"),
                       "loss_max": float(np.max(running)) if running else float("nan"),
                       "grad_norm": float(gnorm), "skipped": skipped,
                       "lr": sched.get_last_lr()[0], "sec_per_step": (time.time() - t0) / a.log_every}
                print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
                running, t0, skipped = [], time.time(), 0
            if step % a.eval_every == 0 or step == a.steps:
                score, base = evaluate(model, val_loader, device, amp, a.eval_batches,
                                       save_grid=out / f"samples_{step:07d}.png")
                rec = {"step": step, "val_psnr": score, "baseline_psnr": base}
                print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step, "args": vars(a)}, out / "last.pt")
            if step >= a.steps:
                break
    print(f"done: {out}/last.pt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--train", required=True, help="train split manifest.json")
    g.add_argument("--val", default=None, help="val split manifest.json (default: hold out --val_frac of train)")
    g.add_argument("--val_frac", type=float, default=0.05)
    g.add_argument("--num_context", type=int, default=2)
    g.add_argument("--num_target", type=int, default=1)
    g.add_argument("--scene_limit", type=int, default=0, help=">0: train on the first N scenes only")
    g.add_argument("--val_limit", type=int, default=256, help="max val scenes per evaluation")
    g.add_argument("--rot_aug", action="store_true", help="random SO(3) rotation of every training scene")
    g.add_argument("--rot_eval", action="store_true", help="evaluate on randomly rotated scenes")
    g.add_argument("--workers", type=int, default=4)
    g = p.add_argument_group("model")
    g.add_argument("--mode", choices=("regression", "flow"), default="regression")
    g.add_argument("--solid", default="octahedron", choices=sorted(PLATONIC_GROUPS),
                   help="symmetry group; 'trivial' = plain transformer (the non-equivariant ablation)")
    g.add_argument("--patch_size", type=int, default=8)
    g.add_argument("--layers", type=int, default=8)
    g.add_argument("--dim_per_g", type=int, default=32, help="hidden channels per group element")
    g.add_argument("--heads_per_g", type=int, default=2, help="attention heads per group element")
    g.add_argument("--rope_sigma", type=float, default=0.0,
                   help="highest RoPE frequency (rad per unit); <=0 = auto from focal length / patch size")
    g.add_argument("--drop_path", type=float, default=0.0)
    g.add_argument("--bounded_output", type=int, default=1,
                   help="regression: squash predicted pixels into [-1,1] with tanh (1/0)")
    g.add_argument("--qk_norm", type=int, default=1,
                   help="normalise queries/keys before attention (1/0); prevents sudden loss spikes")
    g.add_argument("--compile", action="store_true")
    g = p.add_argument_group("flow matching")
    g.add_argument("--sigma_data", type=float, default=0.5, help="std of target pixels in [-1,1]")
    g.add_argument("--t_dist", choices=("logit_normal", "edm"), default="logit_normal",
                   help="training time distribution (edm = the QM9-FM schedule)")
    g.add_argument("--t_mean", type=float, default=0.0, help="logit-normal mean")
    g.add_argument("--t_std", type=float, default=1.0, help="logit-normal std")
    g.add_argument("--P_mean", type=float, default=-1.2, help="edm schedule")
    g.add_argument("--P_std", type=float, default=1.2, help="edm schedule")
    g.add_argument("--t_start", type=float, default=0.0, help="sampler start time (0 = pure noise)")
    g.add_argument("--sample_steps", type=int, default=25)
    g = p.add_argument_group("training")
    g.add_argument("--steps", type=int, default=100_000)
    g.add_argument("--batch_size", type=int, default=16)
    g.add_argument("--lr", type=float, default=4e-4)
    g.add_argument("--weight_decay", type=float, default=0.05)
    g.add_argument("--warmup", type=int, default=2000)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--spike_factor", type=float, default=10.0,
                   help="skip a batch whose loss exceeds this x the running loss (0 = never skip)")
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument("--eval_every", type=int, default=2000)
    g.add_argument("--eval_batches", type=int, default=8)
    g.add_argument("--out", default="runs/ovs_nvs")
    g.add_argument("--ckpt", default=None, help="resume from / evaluate this checkpoint")
    g.add_argument("--eval_only", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--cpu", action="store_true")
    g.add_argument("--no_amp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
