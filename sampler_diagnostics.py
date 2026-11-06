#!/usr/bin/env python3
"""Sampler diagnostics for a trained NVS flow-matching checkpoint (no training).

Question: are the blotchy generations caused by HOW we sample, or by what the flow learned?

It loads a Lightning checkpoint (EMA weights by default, exactly what validation uses) and runs:
  1. Latent statistics of REAL validation latents (the reference: ~0 mean, ~1 std after normalisation).
  2. Flow loss by noise level on real latents, against a "no structure" baseline. Shows where the model
     learned something and where it did not (the very high noise levels are rare in training).
  3. Generation with several sampler schedules from the SAME starting noise: statistics + rendered scenes.
  4. Denoising REAL latents from partial noise, decoded and compared with the autoencoder reconstruction.

Add --trace (fast trajectory trace), --decoder-tolerance (how much latent noise the decoder tolerates) or --swap-test
(do generated positions or generated features ruin the images) or --pca-test (is generated feature damage off the
real feature subspace) or --structure (numbers for crowding and for how consistent the features of nearby tokens
are, real against generated; use --scenes 64) or --gen-metrics (distribution metrics on rendered views, generated
against real; use --scenes 128), --real-projection (real features projected onto K principal components),
--ladder (noise ladder: flow against plain noise of the same size) or --structure --height-split (floor against
object tokens) or --position-origin (how far the encoder moves each token from the position it starts at, which the
input cameras fix) to run only that part.
Add --split train to use training scenes.
For a flow conditioned on the starting positions (diffusion.condition_start) generated scenes get the starting positions
of the cameras of the real scenes of --split (validation scenes by default: the flow never trained on their cameras;
those same cameras render the generated scene). --no-start generates without them (needs a model trained with
start_dropout > 0). --copy-train-scenes N (with --gen-metrics and --split val) adds a copy check: how close generated
views are to the reconstructions of N training scenes, against how close real validation views are. --trace is not
available for conditioned flows.
Run from the PlatonicTransformers folder (where mains/ and configs/ are), e.g.
  PYTHONPATH="/media/mina/hdd/amlab/nvsm/PlatonicTransformers:/media/mina/hdd/amlab/nvsm/PlatonicSplats${PYTHONPATH:+:$PYTHONPATH}" \
    python sampler_diagnostics.py \
      --checkpoint mains/logs/Platonic-NVS-FM/<run id>/checkpoints/last.ckpt \
      --config configs/nvs_fm.yaml
Use the YAML that run was trained with. Writes PNGs and sampler_diagnostics.json to a new folder.
"""
import argparse
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

DEFAULT_FM_SCRIPT = "mains/main_nvs_fm_online_v2.py"
VIEW_IDXS = [0, 3, 6, 9, 12]          # same views the validation hook renders
IMAGE_HW = (128, 128)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="Lightning checkpoint, e.g. last.ckpt")
    p.add_argument("--config", type=Path, required=True, help="The flow YAML the run used (nvs_fm.yaml or nvs_fm_b.yaml)")
    p.add_argument("--fm-script", type=Path, default=Path(DEFAULT_FM_SCRIPT))
    p.add_argument("--weights", choices=("ema", "online"), default="ema",
                   help="ema = what validation uses (default); online = raw training weights")
    p.add_argument("--scenes", type=int, default=16, help="Generated scenes per sampler (2 are rendered).")
    p.add_argument("--val-scenes", type=int, default=64, help="Real validation scenes for the references.")
    p.add_argument("--steps", type=int, default=50, help="Sampler steps (Heun samplers use half, same cost).")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("sampler_diagnostics"))
    p.add_argument("--split", choices=("val", "train"), default="val",
                   help="Which scenes supply the real latents and cameras. train = scenes the flow was trained on.")
    p.add_argument("--decoder-tolerance", action="store_true",
                   help="Only measure how much noise in the latent the decoder tolerates (fast).")
    p.add_argument("--swap-test", action="store_true",
                   help="Only test whether generated positions or generated features ruin the images (fast).")
    p.add_argument("--pca-test", action="store_true",
                   help="Only test whether generated feature damage lies off the real feature subspace (fast).")
    p.add_argument("--structure", action="store_true",
                   help="Only compare how tokens are arranged and how consistent the features of nearby tokens are, "
                        "real against generated (numbers, no rendering). Use --scenes 64.")
    p.add_argument("--real-projection", action="store_true",
                   help="Only test whether the decoder renders REAL features projected onto their top-K principal components "
                        "(K 16 to 128; fitted on half of --val-scenes, tested on the other half; use --val-scenes 64).")
    p.add_argument("--ladder", action="store_true",
                   help="Only run the noise ladder: real latents denoised by the flow from partial noise, against real latents plus "
                        "plain noise of the same size, distance to real views at each level (joint-time checkpoints only).")
    p.add_argument("--height-split", action="store_true",
                   help="With --structure: also split the structure numbers into floor tokens and object tokens by height.")
    p.add_argument("--gen-metrics", action="store_true",
                   help="Only compute distribution metrics on rendered views, generated against real (use --scenes 128 and "
                        "--val-scenes 64; run every checkpoint with the same values for a fair comparison).")
    p.add_argument("--feature-net", choices=("auto", "dinov2", "lowlevel"), default="auto",
                   help="Feature network for --gen-metrics; 'lowlevel' skips the download and reports the crude descriptor only.")
    p.add_argument("--trace", action="store_true",
                   help="Only trace where the flow leaves the data distribution along sampling paths (fast).")
    p.add_argument("--position-origin", action="store_true",
                   help="Only measure how far the autoencoder's encoder moves each latent token from its starting position "
                        "(which the input cameras fix), over --val-scenes scenes and --rigs camera sets each. No flow is used.")
    p.add_argument("--no-start", action="store_true",
                   help="Conditioned flow only: generate WITHOUT the starting positions (needs a model trained with start_dropout > 0).")
    p.add_argument("--copy-train-scenes", type=int, default=0,
                   help="With --gen-metrics and --split val: also compare generated views with the reconstructions of this many "
                        "TRAINING scenes (copy check). 0 = off.")
    p.add_argument("--rigs", type=int, default=4,
                   help="With --position-origin: camera sets per scene (the first is the fixed set 0,3,6,9,12, the rest are "
                        "random 5 of the available views).")
    return p.parse_args()


# ----------------------------------------------------------------------------- checkpoint
def load_flow_weights(model, ckpt, weights, normaliser_cls):
    """Put the checkpoint's flow-network weights (EMA or online) and normaliser into `model`."""
    def clean(k):  # torch.compile prefixes parameter names with "_orig_mod."
        return k[len("net."):].replace("_orig_mod.", "")

    net_sd = {clean(k): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
    missing, unexpected = model.net.load_state_dict(net_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match this YAML's network. missing={missing[:3]} "
                           f"unexpected={unexpected[:3]}. Use the YAML the run was trained with.")
    info = {"weights": "online"}
    if weights == "ema":
        callbacks = ckpt.get("callbacks", {})
        key = next((k for k in callbacks if "EMACallback" in str(k)), None)
        state = callbacks.get(key, {}).get("ema_state") if key is not None else None
        if state is None:
            raise RuntimeError("No EMA weights in this checkpoint. Re-run with --weights online.")
        ema = {clean(k): v for k, v in state.items() if k.startswith("net.")}
        params = dict(model.net.named_parameters())
        absent = [n for n in params if n not in ema]
        extra = [n for n in ema if n not in params]
        if absent or extra:
            raise RuntimeError(f"EMA weights do not match the network: absent={absent[:3]} extra={extra[:3]}")
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(ema[n].to(p.dtype).to(p.device))
        info = {"weights": "ema", "ema_decay": callbacks[key].get("decay"),
                "ema_warmup_steps": callbacks[key].get("warmup_steps")}
    saved = ckpt.get("latent_normaliser")
    if saved is not None:
        model.set_normaliser(normaliser_cls.from_state_dict(saved))
        info["normaliser"] = True
    else:
        print("WARNING: checkpoint has no latent normaliser; working in raw latent space.")
        info["normaliser"] = False
    info["global_step"] = int(ckpt.get("global_step", -1))
    info["epoch"] = int(ckpt.get("epoch", -1))
    return info


# ----------------------------------------------------------------------------- statistics
def latent_stats(z, pos):
    """The same statistics the validation hook logs (val/gen_*), plus per-channel and tail facts."""
    z = z.float(); pos = pos.float()
    ch_std = z.reshape(-1, z.shape[-1]).std(0)
    pos_c = pos - pos.mean(dim=(0, 1), keepdim=True)
    return {
        "feat_mean": z.mean().item(), "feat_std": z.std().item(),
        "feat_channel_std_median": ch_std.median().item(), "feat_channel_std_max": ch_std.max().item(),
        "frac_abs_gt_5": (z.abs() > 5).float().mean().item(),
        "pos_rms": pos_c.pow(2).mean().sqrt().item(),
        "pos_mean_abs": pos.mean(dim=(0, 1)).abs().mean().item(),
    }


def train_noise_tail(sigma, p_mean, p_std):
    """Share of training samples whose noise level is >= sigma (training draws sigma log-normally)."""
    z = (math.log(sigma) - p_mean) / p_std
    return 0.5 * math.erfc(z / math.sqrt(2))


def no_structure_baseline(t):
    """Per-coordinate velocity error of the best estimator that ignores all structure, for unit-variance
    data and unit noise (unweighted). A model that learned structure should come in below this."""
    return 2.0 - (2 * t - 1) ** 2 / (t ** 2 + (1 - t) ** 2)


@torch.no_grad()
def loss_by_noise_level(model, z1, pos1, sigmas, p_mean, p_std, draws=4, chunk=16, seed=0, start=None):
    """Flow loss on real latents at fixed noise levels, split into features and positions.
    start (conditioned flow): starting positions of the scenes, [S, N, 3] in flow space. In offset mode the noise is in the
    offset state; the position error is reported in position units."""
    gen = torch.Generator(device=z1.device).manual_seed(seed)
    cond_all = model.as_cond(start, z1.shape[0], z1.device)
    rows = []
    for sigma in sigmas:
        t = 1.0 / (1.0 + sigma)
        acc = {"feat_w": 0.0, "pos_w": 0.0, "feat_u": 0.0, "pos_u": 0.0}
        n = 0
        for _ in range(draws):
            for s in range(0, z1.shape[0], chunk):
                x1, p1 = z1[s:s + chunk], pos1[s:s + chunk]
                cond = None if cond_all is None else cond_all[s:s + chunk]
                p1 = model._pos_to_state(p1, cond)
                _, sc, _ = model._start_parts(cond, x1)
                sc = 1.0 if sc is None else sc
                tb = torch.full((x1.shape[0],), t, device=z1.device)
                t_ = tb[:, None, None]
                x0 = torch.randn(x1.shape, device=x1.device, generator=gen)
                p0 = torch.randn(p1.shape, device=x1.device, generator=gen)
                vx, vp = model._velocity((1 - t_) * x0 + t_ * x1, (1 - t_) * p0 + t_ * p1, tb, start=cond)
                ef = (vx - (x1 - x0)).square(); ep = (sc * (vp - (p1 - p0))).square()
                w = model._edm_weight(t_)
                k = x1.shape[0]
                acc["feat_w"] += (w * ef).mean().item() * k; acc["pos_w"] += (w * ep).mean().item() * k
                acc["feat_u"] += ef.mean().item() * k;      acc["pos_u"] += ep.mean().item() * k
                n += k
        base = no_structure_baseline(t)
        rows.append({
            "sigma": sigma, "t": t, "train_share_at_least_this_noisy": train_noise_tail(sigma, p_mean, p_std),
            "loss_feat": acc["feat_w"] / n, "loss_pos": acc["pos_w"] / n,
            "feat_vs_baseline": acc["feat_u"] / n / base, "pos_vs_baseline": acc["pos_u"] / n / base,
        })
    return rows


# ----------------------------------------------------------------------------- samplers
def sigma_grid(kind, n_steps, sigma_max=20.0, sigma_min=0.002, rho=7.0):
    """Decreasing noise levels, ending in 0 (the data). sigma = (1 - t) / t for t in [0, 1]."""
    if kind == "uniform_t":                       # what validation currently does
        t = np.linspace(1e-4, 1.0, n_steps + 1)
        return (1 - t) / t                        # last entry is exactly 0
    if kind == "log":
        s = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), n_steps))
    elif kind == "karras":
        ramp = np.linspace(0, 1, n_steps)
        s = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    else:
        raise ValueError(kind)
    return np.concatenate([s, [0.0]])


def start_scale(sigma0):
    """Std of x at the starting time for unit-variance data and noise: sqrt((1-t)^2 + t^2)."""
    t0 = 1.0 / (1.0 + sigma0)
    return math.sqrt((1 - t0) ** 2 + t0 ** 2)


@torch.no_grad()
def integrate(model, x, pos, sigmas, heun=False, clamp=None, on_step=None, clip_x1=None, start=None):
    """Integrate the learned velocity field from the first noise level in `sigmas` to 0 (t = 1).

    Euler, or Heun (second order, no second evaluation on the last step, which would be at t = 1).
    `clamp` limits |velocity| like the validation sampler does (None = off). Returns x, pos, info.
    `on_step((i, sigma), x, pos, vx, vp)` is called at the start of every step with the raw velocity.
    `clip_x1` limits the model's clean-feature estimate x1_hat = x + (1-t) v to [-c, c] before stepping
    (the velocity is rebuilt from the clipped estimate; positions are not clipped).
    start: StartCond of a flow conditioned on the starting positions. `pos` is the flow's position STATE (the position itself
    unless the flow is in offset mode); convert the result with model._state_to_pos(pos, start).
    """
    ts = torch.tensor(1.0 / (1.0 + np.asarray(sigmas, dtype=np.float64)), dtype=torch.float32, device=x.device)
    b = x.shape[0]
    n_eval, clamped = 0, []

    def vel(xx, pp, tt, record=None):
        nonlocal n_eval
        vx, vp = model._velocity(xx, pp, tt.expand(b), start=start)
        n_eval += 1
        if on_step is not None and record is not None:
            on_step(record, xx, pp, vx, vp)
        if clip_x1:
            one_minus_t = 1.0 - tt
            vx = ((xx + one_minus_t * vx).clamp(-clip_x1, clip_x1) - xx) / one_minus_t
        if clamp:
            clamped.append(((vx.abs() > clamp).float().mean().item() + (vp.abs() > clamp).float().mean().item()) / 2)
            vx, vp = vx.clamp(-clamp, clamp), vp.clamp(-clamp, clamp)
        return vx, vp

    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        dt = tn - t
        vx, vp = vel(x, pos, t, record=(i, float(sigmas[i])))
        if heun and float(tn) < 1.0:
            vx2, vp2 = vel(x + dt * vx, pos + dt * vp, tn)
            x, pos = x + dt * 0.5 * (vx + vx2), pos + dt * 0.5 * (vp + vp2)
        else:
            x, pos = x + dt * vx, pos + dt * vp
    return x, pos, {"network_evaluations": n_eval, "clamp_fraction": float(np.mean(clamped)) if clamped else 0.0}


def sampler_variants(steps, clip=5.0):
    half = max(steps // 2, 1)
    return [
        # name, noise-level grid, Heun?, velocity clamp, clip of the clean-feature estimate
        ("current: uniform t, Euler, clamp 10",           sigma_grid("uniform_t", steps),   False, 10.0, None),
        ("uniform t, Euler, no clamp",                    sigma_grid("uniform_t", steps),   False, None, None),
        ("log-sigma 20->0.002, Euler",                    sigma_grid("log", steps, 20.0),   False, None, None),
        ("Karras 20->0.002, Heun",                        sigma_grid("karras", half, 20.0), True,  None, None),
        ("Karras 5->0.002, Heun",                         sigma_grid("karras", half, 5.0),  True,  None, None),
        (f"uniform t, Euler, clip x1_hat to +-{clip:g}",  sigma_grid("uniform_t", steps),   False, None, clip),
        (f"Karras 20->0.002, Heun, clip x1_hat to +-{clip:g}", sigma_grid("karras", half, 20.0), True, None, clip),
    ]


@torch.no_grad()
def denoise_from(model, z1, pos1, t0, n_steps, seed=0, on_step=None, start=None):
    """Noise real latents to time t0 (x_t = (1-t0) eps + t0 x1) and integrate back to the data.
    start: starting positions of these scenes for a conditioned flow (tensor [S, N, 3] or StartCond)."""
    gen = torch.Generator(device=z1.device).manual_seed(seed)
    cond = model.as_cond(start, z1.shape[0], z1.device)
    p1 = model._pos_to_state(pos1, cond)
    x = (1 - t0) * torch.randn(z1.shape, device=z1.device, generator=gen) + t0 * z1
    p = (1 - t0) * torch.randn(pos1.shape, device=z1.device, generator=gen) + t0 * p1
    sigmas = sigma_grid("karras", max(n_steps // 2, 1), sigma_max=(1 - t0) / t0)
    xd, pd, _ = integrate(model, x, p, sigmas, heun=True, on_step=on_step, start=cond)
    return xd, model._state_to_pos(pd, cond)


# ----------------------------------------------------------------------------- images
def psnr(a, b):
    """Mean PSNR over views of two [V,C,H,W] image stacks (clipped to [0,1])."""
    a = torch.nan_to_num(a.float(), nan=0.0).clamp(0, 1); b = torch.nan_to_num(b.float(), nan=0.0).clamp(0, 1)
    mse = (a - b).square().mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return float((-10 * torch.log10(mse)).mean())


def _row_pixels(image_stacks):
    """image_stacks: list of [V,C,H,W] tensors (one per scene) -> one uint8 strip."""
    tiles = []
    for k, stack in enumerate(image_stacks):
        s = (torch.nan_to_num(stack.float(), nan=0.0).clamp(0, 1) * 255).byte().cpu().numpy()
        tiles.append(np.concatenate([s[v].transpose(1, 2, 0) for v in range(s.shape[0])], axis=1))
        if k < len(image_stacks) - 1:
            tiles.append(np.full((tiles[-1].shape[0], 10, 3), 255, np.uint8))
    return np.concatenate(tiles, axis=1)


def save_grid(path, rows, label_width=330):
    """rows: list of (label, [image stacks]) -> PNG with labels on the left."""
    from PIL import Image, ImageDraw
    strips = [(label, _row_pixels(stacks)) for label, stacks in rows]
    width = label_width + max(s.shape[1] for _, s in strips)
    height = sum(s.shape[0] + 6 for _, s in strips)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, strip in strips:
        draw.text((6, y + strip.shape[0] // 2 - 6), label, fill="black")
        canvas.paste(Image.fromarray(strip), (label_width, y))
        y += strip.shape[0] + 6
    canvas.save(path)



# ----------------------------------------------------------------------------- trajectory trace
def trace_trajectory(model, x, pos, sigmas, heun=False, clamp=None, z_true=None, p_true=None):
    """Integrate like `integrate`, recording at every step what the model sees and predicts.

    x1_hat = x_t + (1 - t) * v is the model's own estimate of the clean latent at that step.
    For pure noise, expected_std is the std a perfect model would see: sqrt((1-t)^2 + t^2).
    With z_true/p_true (real-latent start) it also records the error of x1_hat against the truth.
    """
    rows = []

    def centred_rms(a):
        return (a - a.mean(dim=(0, 1), keepdim=True)).pow(2).mean().sqrt().item()

    def on_step(record, xx, pp, vx, vp):
        i, sigma = record
        t = 1.0 / (1.0 + sigma)
        xh, ph = xx + (1 - t) * vx, pp + (1 - t) * vp
        row = {"step": i, "sigma": sigma, "expected_std": math.sqrt((1 - t) ** 2 + t ** 2),
               "feat_std": xx.std().item(), "feat_absmax": xx.abs().max().item(), "pos_rms": centred_rms(pp),
               "feat_v_rms": vx.pow(2).mean().sqrt().item(), "feat_v_frac_gt10": (vx.abs() > 10).float().mean().item(),
               "pos_v_rms": vp.pow(2).mean().sqrt().item(),
               "feat_x1hat_std": xh.std().item(), "feat_x1hat_frac_gt5": (xh.abs() > 5).float().mean().item(),
               "pos_x1hat_rms": centred_rms(ph)}
        if z_true is not None:
            row["feat_x1hat_mse"] = (xh - z_true).pow(2).mean().item()
            row["pos_x1hat_mse"] = (ph - p_true).pow(2).mean().item()
        rows.append(row)

    xf, pf, info = integrate(model, x, pos, sigmas, heun=heun, clamp=clamp, on_step=on_step)
    return rows, xf, pf, info


def error_concentration(err):
    """Where a squared-error tensor [scenes, tokens, channels] lives: heavy tail, tokens, channels."""
    flat = err.flatten().float()
    total = flat.sum().clamp(min=1e-12)
    k = max(int(0.01 * flat.numel()), 1)
    per_token = err.float().mean(dim=(0, 2)); per_channel = err.float().mean(dim=(0, 1))
    tok = per_token.topk(min(8, per_token.numel())); ch = per_channel.topk(min(8, per_channel.numel()))
    out = {"mean": flat.mean().item(), "median_entry": flat.median().item(),
           "top1pct_entries_share_of_error": (flat.topk(k).values.sum() / total).item(),
           "top_tokens": [(int(i), float(v / per_token.sum().clamp(min=1e-12))) for v, i in zip(tok.values, tok.indices)],
           "top_channels": [(int(i), float(v / per_channel.sum().clamp(min=1e-12))) for v, i in zip(ch.values, ch.indices)]}
    if per_token.numel() == 256:   # assumed layout (registers.py): 5 view slots x 48 tokens, then 16 scene-centre tokens
        out["mse_by_token_block"] = {f"tokens {a}-{a + 47}": per_token[a:a + 48].mean().item() for a in range(0, 240, 48)}
        out["mse_by_token_block"]["tokens 240-255"] = per_token[240:].mean().item()
    return out


def _pick_steps(n, k=12):
    picks = set(np.round(np.linspace(0, n - 1, k)).astype(int).tolist()) | {0, 1, 2}
    return sorted(i for i in picks if i < n)


def print_trace(title, rows, truth):
    print(f"\n{title}")
    head = "  step | sigma    | feat std (ideal) | x1_hat std | x1_hat |x|>5 | feat |v|>10 | feat max | pos rms | pos x1_hat rms"
    print(head + (" | x1_hat MSE feat / pos" if truth else ""))
    for i in _pick_steps(len(rows)):
        r = rows[i]
        line = (f"  {r['step']:4d} | {r['sigma']:8.3f} | {r['feat_std']:6.3f} ({r['expected_std']:.3f}) | {r['feat_x1hat_std']:9.3f} | "
                f"{100 * r['feat_x1hat_frac_gt5']:9.2f}% | {100 * r['feat_v_frac_gt10']:8.2f}% | {r['feat_absmax']:8.1f} | "
                f"{r['pos_rms']:7.3f} | {r['pos_x1hat_rms']:9.3f}")
        if truth:
            line += f" | {r['feat_x1hat_mse']:.3f} / {r['pos_x1hat_mse']:.3f}"
        print(line)


def run_trace(model, z1, pos1, steps, seed, out_dir):
    """Trace where the learned flow leaves the data distribution. Writes trace.json and trace.png."""
    n = min(16, z1.shape[0])
    z_true, p_true = z1[:n], pos1[:n]
    gen = torch.Generator(device=z1.device).manual_seed(seed)
    eps_x = torch.randn(z_true.shape, device=z1.device, generator=gen)
    eps_p = torch.randn(p_true.shape, device=z1.device, generator=gen)
    results = {}

    for name, kind, heun, smax, n_steps in (
            ("A: from noise, uniform t, Euler, no clamp", "uniform_t", False, None, steps),
            ("B: from noise, Karras 20->0.002, Heun", "karras", True, 20.0, max(steps // 2, 1))):
        sig = sigma_grid(kind, n_steps, smax) if smax else sigma_grid(kind, n_steps)
        s0 = start_scale(sig[0])
        rows, xf, pf, _ = trace_trajectory(model, eps_x * s0, eps_p * s0, sig, heun=heun)
        print_trace(name, rows, truth=False)
        results[name] = {"rows": rows}

    for t0 in (0.7, 0.5):
        sig0 = (1 - t0) / t0
        name = f"C: real latents noised to t={t0} (sigma {sig0:.2f}), Karras Heun"
        sig = sigma_grid("karras", max(steps // 2, 1), sigma_max=sig0)
        x = (1 - t0) * eps_x + t0 * z_true; p = (1 - t0) * eps_p + t0 * p_true
        rows, xf, pf, _ = trace_trajectory(model, x, p, sig, heun=True, z_true=z_true, p_true=p_true)
        print_trace(name, rows, truth=True)
        conc = error_concentration((xf - z_true).pow(2))
        print(f"  final feature error: mean {conc['mean']:.3f}, median entry {conc['median_entry']:.4f}; "
              f"the worst 1% of entries hold {100 * conc['top1pct_entries_share_of_error']:.1f}% of it")
        print("  worst tokens (index: share of error): " + ", ".join(f"{i}: {100 * v:.1f}%" for i, v in conc["top_tokens"][:5]))
        print("  worst channels (index: share of error): " + ", ".join(f"{i}: {100 * v:.1f}%" for i, v in conc["top_channels"][:5]))
        if "mse_by_token_block" in conc:
            print("  mean error by token block (assumed layout): " + ", ".join(f"{k}: {v:.3f}" for k, v in conc["mse_by_token_block"].items()))
        results[name] = {"rows": rows, "final_error": conc}

    (out_dir / "trace.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, res in results.items():
        if name.startswith(("A", "B")):
            r = res["rows"]
            axes[0].plot([x["sigma"] for x in r], [x["feat_std"] for x in r], label=name[:2] + "feature std")
    r = results[next(k for k in results if k.startswith("B"))]["rows"]
    axes[0].plot([x["sigma"] for x in r], [x["expected_std"] for x in r], "k--", label="ideal")
    axes[0].set_title("Feature std of the state along the path"); axes[0].set_ylabel("std of x_t")
    for name, res in results.items():
        if name.startswith("C"):
            r = res["rows"]
            axes[1].plot([x["sigma"] for x in r], [x["feat_x1hat_mse"] for x in r], label=name[:16])
            axes[2].plot([x["sigma"] for x in r], [x["pos_x1hat_mse"] for x in r], label=name[:16])
    axes[1].set_title("Feature error of the model's clean estimate"); axes[1].set_ylabel("MSE of x1_hat vs truth")
    axes[2].set_title("Position error of the model's clean estimate"); axes[2].set_ylabel("MSE of x1_hat vs truth")
    for k, ax in enumerate(axes):
        ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel("noise level sigma (high to low)")
        if k > 0:
            ax.set_yscale("log")
        ax.grid(alpha=0.3); ax.legend(frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "trace.png", dpi=130); plt.close(fig)
    return results



# ----------------------------------------------------------------------------- decoder tolerance
@torch.no_grad()
def build_scene_context(model, ae, dataset, device, edict, n=4):
    """Rays and the clean (normalised) latent of the first n scenes, plus the autoencoder reconstruction."""
    ctx = []
    for i in range(min(n, len(dataset))):
        sd = dataset.scene_batch(scene_idx=int(dataset.scene_indices[i]), view_indices=VIEW_IDXS, device=device)
        ray_o, ray_d = ae.process_data.compute_rays(sd["c2w"], sd["fxfycxcy"], h=IMAGE_HW[0], w=IMAGE_HW[1], device=device)
        enc = edict({"image": sd["image"], "ray_o": ray_o, "ray_d": ray_d, "image_h_w": IMAGE_HW})
        z, p, s0 = ae.encode(enc)
        if model.normaliser is not None:
            p, z = model.normaliser.normalise(p, z)
        ctx.append({"ray_o": ray_o, "ray_d": ray_d, "z": z, "pos": p, "start": model.norm_start(s0),
                    "recon": model.render(z, p, ray_o, ray_d, IMAGE_HW)[0]})
    return ctx


def ctx_starts(ctx, n):
    """Starting positions (flow space) of the input cameras of scene i % len(ctx), for n scenes: [n, N, 3].
    The same cameras render the generated scene i, so latent and cameras share one frame."""
    return torch.cat([ctx[i % len(ctx)]["start"] for i in range(n)])


@torch.no_grad()
def run_decoder_tolerance(model, ctx, out_dir, seed, levels=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0), draws=2):
    """Decode real latents plus Gaussian noise of std `level` (flow space): how precise must the flow be?

    Noise is added to the features only, the positions only, or both. PSNR is against the autoencoder
    reconstruction of the clean latent (so 'inf-like' values mean the decoder ignored the noise).
    """
    gen = torch.Generator(device=ctx[0]["z"].device).manual_seed(seed)

    def decode(c, z, p):
        return model.render(z, p, c["ray_o"], c["ray_d"], IMAGE_HW)[0]

    results = []
    print("\nDecoder tolerance: PSNR (dB) of decoded real latent + noise vs the clean reconstruction")
    print("  noise std | features only | positions only | both")
    for s_ in levels:
        row = {"noise_std": s_}
        for kind in ("features", "positions", "both"):
            scores = []
            for _ in range(draws):
                for c in ctx:
                    nz = torch.randn(c["z"].shape, device=c["z"].device, generator=gen) * s_
                    npos = torch.randn(c["pos"].shape, device=c["z"].device, generator=gen) * s_
                    z = c["z"] + (nz if kind in ("features", "both") else 0)
                    p = c["pos"] + (npos if kind in ("positions", "both") else 0)
                    scores.append(psnr(decode(c, z, p), c["recon"]))
            row[kind] = float(np.mean(scores))
        results.append(row)
        print(f"  {s_:9.2f} | {row['features']:13.2f} | {row['positions']:14.2f} | {row['both']:.2f}")

    rows = [("real: autoencoder reconstruction", [c["recon"] for c in ctx[:2]])]
    for kind, picks in (("features", (0.05, 0.1, 0.2, 0.5)), ("positions", (0.02, 0.05, 0.1))):
        for s_ in picks:
            imgs = []
            for c in ctx[:2]:
                nz = torch.randn(c["z"].shape, device=c["z"].device, generator=gen) * s_
                npos = torch.randn(c["pos"].shape, device=c["z"].device, generator=gen) * s_
                z = c["z"] + (nz if kind == "features" else 0)
                p = c["pos"] + (npos if kind == "positions" else 0)
                imgs.append(decode(c, z, p))
            rows.append((f"{kind} + noise std {s_}", imgs))
    save_grid(out_dir / "3_decoder_tolerance.png", rows)
    (out_dir / "decoder_tolerance.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results



# ----------------------------------------------------------------------------- swap test
@torch.no_grad()
def run_swap_test(model, ctx, generated, token_mean, out_dir):
    """Which half of a generated latent ruins the picture: its positions or its features?

    `generated`: list of (name, z_gen, pos_gen) with at least len(ctx) scenes, in the flow's space.
    `token_mean`: [1, tokens, channels] per-token average of real latents (features with all scene
    content removed). Scene i is rendered with the cameras of real scene i. Prints PSNR against the
    autoencoder reconstruction where that is meaningful, and saves 4_swap_test.png.
    """
    n = len(ctx)

    def draw(zs, ps):
        return [model.render(zs[i], ps[i], ctx[i]["ray_o"], ctx[i]["ray_d"], IMAGE_HW)[0] for i in range(n)]

    real_z = [c["z"] for c in ctx]; real_p = [c["pos"] for c in ctx]
    mean_z = [token_mean.to(c["z"].device) for c in ctx]
    rows = [("real: autoencoder reconstruction", [c["recon"] for c in ctx[:2]])]
    scores = {}
    imgs = draw(mean_z, real_p)
    scores["real positions + mean features"] = float(np.mean([psnr(imgs[i], ctx[i]["recon"]) for i in range(n)]))
    rows.append(("real positions + mean features", imgs[:2]))
    for name, zg, pg in generated:
        gz = [zg[i:i + 1] for i in range(n)]; gp = [pg[i:i + 1] for i in range(n)]
        for label, zs, ps in (
                ("generated latent (positions + features)", gz, gp),
                ("real positions + generated features", gz, real_p),
                ("generated positions + real features", real_z, gp),
                ("generated positions + mean features", mean_z, gp)):
            imgs = draw(zs, ps)
            rows.append((f"{name}: {label}", imgs[:2]))
            if label.startswith("real positions"):
                scores[f"{name}: {label}"] = float(np.mean([psnr(imgs[i], ctx[i]["recon"]) for i in range(n)]))
    print("\nSwap test: PSNR against the autoencoder reconstruction (only where positions are the real ones)")
    for k, v in scores.items():
        print(f"  {k:70s} {v:6.2f} dB")
    save_grid(out_dir / "4_swap_test.png", rows, label_width=430)
    (out_dir / "swap_test.json").write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
    return scores



# ----------------------------------------------------------------------------- feature subspace test
@torch.no_grad()
def feature_pca(z_real):
    """Principal components of the per-token feature vectors of real latents [scenes, tokens, channels]."""
    x = z_real.reshape(-1, z_real.shape[-1]).double()
    mean = x.mean(0)
    xc = x - mean
    evals, evecs = torch.linalg.eigh(xc.T @ xc / (xc.shape[0] - 1))
    order = evals.argsort(descending=True)
    return mean.float(), evecs[:, order].float(), evals[order].clamp(min=0).float()


def project_features(z, mean, basis, k):
    """Keep only the top-k principal components of the features (positions untouched)."""
    c = z - mean
    top = basis[:, :k]
    return mean + (c @ top) @ top.T


def off_subspace_share(z, mean, basis, k):
    """Share of the features' squared deviation from the mean that lies outside the top-k components."""
    c = z - mean
    resid = c - (c @ basis[:, :k]) @ basis[:, :k].T
    return (resid.pow(2).sum() / c.pow(2).sum().clamp(min=1e-12)).item()


@torch.no_grad()
def run_pca_test(model, ctx, generated, z_real, out_dir, ks=(16, 32, 64, 128)):
    """Is the generated-feature damage outside the directions real features occupy?

    Prints how much real features concentrate in few components, how much of each generated sample's
    feature energy lies outside the top-k components, and renders generated scenes with their features
    projected onto those components. If the images clean up, the damage lives off the data subspace.
    """
    mean, basis, evals = feature_pca(z_real)
    cum = (evals.cumsum(0) / evals.sum()).tolist()
    print("\nReal features: share of variance in the top-k principal components")
    print("  " + "  ".join(f"k={k}: {100 * cum[k - 1]:.1f}%" for k in (1, 4, 16, 32, 64, 128, 256) if k <= len(cum)))
    print("\nShare of feature energy OUTSIDE the top-k components (real latents vs generated samples)")
    print("  " + "k".ljust(5) + "real".rjust(9) + "".join(name[:34].rjust(36) for name, _, _ in generated))
    table = {}
    for k in ks:
        real_share = off_subspace_share(z_real, mean, basis, k)
        shares = [off_subspace_share(zg, mean.to(zg.device), basis.to(zg.device), k) for _, zg, _ in generated]
        table[k] = {"real": real_share, **{name: sh for (name, _, _), sh in zip(generated, shares)}}
        print("  " + str(k).ljust(5) + f"{100 * real_share:8.1f}%" + "".join(f"{100 * sh:35.1f}%" for sh in shares))

    n = len(ctx)
    rows = [("real: autoencoder reconstruction", [c["recon"] for c in ctx[:2]])]
    for name, zg, pg in generated:
        for k in (None,) + tuple(ks[::-1]):
            zk = zg if k is None else project_features(zg, mean.to(zg.device), basis.to(zg.device), k)
            imgs = [model.render(zk[i:i + 1], pg[i:i + 1], ctx[i]["ray_o"], ctx[i]["ray_d"], IMAGE_HW)[0] for i in range(min(2, n))]
            rows.append((f"{name}: " + ("generated features as they are" if k is None else f"features kept to top {k} components"), imgs))
    save_grid(out_dir / "5_feature_subspace.png", rows, label_width=470)
    (out_dir / "feature_subspace.json").write_text(json.dumps(
        {"variance_in_top_k": {str(k): cum[k - 1] for k in (1, 4, 16, 32, 64, 128, 256) if k <= len(cum)},
         "energy_outside_top_k": {str(k): v for k, v in table.items()}}, indent=2, allow_nan=False) + "\n")
    return table


@torch.no_grad()
def generate_samples(model, n, steps, seed, device, start=None):
    """Two generated batches from the same starting noise: the current sampler and the clipped Karras one.
    start: starting positions of the n scenes' cameras for a conditioned flow ([n, N, 3] flow space or StartCond);
    None runs without them (needs start_dropout > 0)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    cond = model.as_cond(start, n, device)
    base_x = torch.randn(n, model.n_tokens, model.scalar_dim, device=device, generator=gen)
    base_p = torch.randn(n, model.n_tokens, 3, device=device, generator=gen)
    if getattr(model, "time_coupling", "joint") == "decoupled":
        # the model's own sampler: positions run ahead of features (one sampler only; the other variants
        # assume one time for positions and features)
        zg, pg = model._integrate_decoupled(base_x, base_p, steps, start=cond)
        return [("current: decoupled times, positions first, Euler, clamp 10", zg, pg)]
    variants = sampler_variants(steps)
    out = []
    for name, sigmas_, heun, clamp, clip in (variants[0], variants[-1]):
        s0 = start_scale(sigmas_[0])
        zg, pg, _ = integrate(model, base_x * s0, base_p * s0, sigmas_, heun=heun, clamp=clamp, clip_x1=clip, start=cond)
        out.append((name, zg, model._state_to_pos(pg, cond)))
    return out


# ----------------------------------------------------------------------------- structure test
def structure_metrics(z, pos, slot_mean, edges, ks=(1, 5, 20), near_k=4, view_size=48):
    """How tokens are arranged and how alike the features of nearby tokens are.

    z (S, N, F) and pos (S, N, 3) in the flow's normalised space; slot_mean (1, N, F) is the per-token-slot mean of the
    real latents (47.8% of the feature variance is fixed per slot, so it is removed before comparing features).
    Returns per-scene arrays (to give a spread over scenes) and cosine similarity by position distance.
      pos_nn{k}          mean distance of a token to its k-th nearest token (small = crowded or clumped)
      feat_cos_nearest   mean cosine similarity of the slot-centred features to the near_k nearest tokens by position
      feat_cos_random    the same to all other tokens (baseline, close to 0)
      by_bin             mean cosine similarity of token pairs whose distance falls in the bins given by edges
      by_bin_same_view   the same for pairs from the same input-view slot (token index // view_size, the 16 centre
                         tokens form a sixth group), by_bin_diff_view for pairs from different slots. If real
                         nearby tokens agree only within a view slot, the agreement is view structure, not objects.
      near_same_frac     share of the near_k nearest tokens that belong to the same view slot
    """
    S, N, _ = z.shape
    D = torch.cdist(pos.float(), pos.float())
    eye = torch.eye(N, dtype=torch.bool, device=D.device)
    srt, idx = D.masked_fill(eye, float("inf")).sort(dim=-1)
    r = z.float() - slot_mean.float()
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    C = r @ r.transpose(1, 2)
    out = {f"pos_nn{k}": srt[..., k - 1].mean(dim=-1).cpu().numpy() for k in ks}
    out["feat_cos_nearest"] = torch.gather(C, 2, idx[..., :near_k]).mean(dim=(1, 2)).cpu().numpy()
    out["feat_cos_random"] = (C.masked_fill(eye, 0.0).sum(dim=(1, 2)) / (N * (N - 1))).cpu().numpy()
    grp = torch.clamp(torch.arange(N, device=D.device) // view_size, max=N // view_size)
    same = (grp[:, None] == grp[None, :]).unsqueeze(0)
    nb_same = grp[idx[..., :near_k]] == grp[None, :, None]
    out["near_same_frac"] = nb_same.float().mean(dim=(1, 2)).cpu().numpy()
    bounds = [0.0] + [float(e) for e in edges] + [float("inf")]
    by_bin, by_same, by_diff = [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (D >= lo) & (D < hi) & ~eye
        by_bin.append(float(C[m].mean().item()) if bool(m.any()) else float("nan"))
        ms, md = m & same, m & ~same
        by_same.append(float(C[ms].mean().item()) if bool(ms.any()) else float("nan"))
        by_diff.append(float(C[md].mean().item()) if bool(md.any()) else float("nan"))
    out["by_bin"], out["by_bin_same_view"], out["by_bin_diff_view"] = by_bin, by_same, by_diff
    return out


def distance_edges(pos, quantiles=(0.01, 0.05, 0.2, 0.5)):
    """Bin edges for pair distances: quantiles of the real pairwise distances (same edges for real and generated)."""
    D = torch.cdist(pos.float(), pos.float())
    N = pos.shape[1]
    vals = D[:, ~torch.eye(N, dtype=torch.bool, device=D.device)].flatten()
    if vals.numel() > 4_000_000:
        vals = vals[torch.randperm(vals.numel(), device=vals.device)[:4_000_000]]
    return [float(torch.quantile(vals, q).item()) for q in quantiles]


def _mean_sd(a):
    return float(np.mean(a)), float(np.std(a))


def run_structure(model, z1, pos1, steps, seed, n_gen, device, out_dir, height_split=False, s1=None):
    slot_mean = z1.mean(dim=0, keepdim=True)
    edges = distance_edges(pos1)
    rows = [("real validation latents", structure_metrics(z1, pos1, slot_mean, edges))]
    gens = {}
    for c in range(0, n_gen, 16):
        m_ = min(16, n_gen - c)
        st = None if s1 is None else s1[torch.arange(c, c + m_, device=s1.device) % len(s1)]   # cameras of real scene i
        out = generate_samples(model, m_, steps, seed + c, device, start=st)
        for k, (name, zg, pg) in enumerate(out):
            gens.setdefault(k, (name, [], []))
            gens[k][1].append(zg); gens[k][2].append(pg)
    for name, zs, ps in gens.values():
        rows.append((f"generated: {name}", structure_metrics(torch.cat(zs), torch.cat(ps), slot_mean, edges)))
    keys = ["pos_nn1", "pos_nn5", "pos_nn20", "feat_cos_nearest", "feat_cos_random", "near_same_frac"]
    print(f"\nStructure test: {len(z1)} real scenes, {n_gen} generated scenes per sampler; mean (sd over scenes).")
    print("  Distances are in the flow's normalised space (real position rms = 1).")
    print("  " + f"{'':52s} " + " ".join(f"{k:>17s}" for k in keys))
    summary = {}
    for name, m in rows:
        cells = []
        for k in keys:
            mu, sd = _mean_sd(m[k]); cells.append(f"{mu:8.3f} ({sd:5.3f})")
        print("  " + f"{name:52s} " + " ".join(f"{c:>17s}" for c in cells))
        summary[name] = {k: {"mean": _mean_sd(m[k])[0], "sd_over_scenes": _mean_sd(m[k])[1]} for k in keys}
        for tag in ("by_bin", "by_bin_same_view", "by_bin_diff_view"):
            summary[name]["feat_cos_" + tag] = [None if v != v else v for v in m[tag]]
    labels = [f"< {edges[0]:.3f}"] + [f"{edges[i]:.3f} to {edges[i + 1]:.3f}" for i in range(len(edges) - 1)] + [f">= {edges[-1]:.3f}"]
    print("\n  Cosine similarity of slot-centred features by token-pair distance (bins = real quantiles 1%, 5%, 20%, 50%):")
    print("  " + f"{'distance':20s} " + " ".join(f"{n[:22]:>22s}" for n, _ in rows))
    for j, lab in enumerate(labels):
        print("  " + f"{lab:20s} " + " ".join(f"{m['by_bin'][j]:22.3f}" for _, m in rows))
    for tag, title in (("by_bin_same_view", "pairs from the SAME input-view slot"), ("by_bin_diff_view", "pairs from DIFFERENT input-view slots")):
        print(f"\n  The same, only {title}:")
        for j, lab in enumerate(labels):
            print("  " + f"{lab:20s} " + " ".join(f"{m[tag][j]:22.3f}" for _, m in rows))
    (out_dir / "structure_test.json").write_text(json.dumps({"edges": edges, "bin_labels": labels, "rows": summary},
                                                            indent=2, allow_nan=False) + "\n")
    if height_split:
        run_height_split(z1, pos1, slot_mean, [(f"generated: {name}", torch.cat(zs), torch.cat(ps)) for name, zs, ps in gens.values()], out_dir)



# ----------------------------------------------------------------------------- generation metrics
def _clean_images(x):
    return torch.nan_to_num(x.float(), nan=0.0).clamp(0, 1)


def lowlevel_descriptor(img):
    """[N,3,H,W] in [0,1] -> [N,296] colour, saturation, brightness, gradient and coarse layout descriptor.

    Needs no download. Crude: it sees colours, edge sharpness and where things are, not object identity.
    """
    import torch.nn.functional as F
    img = _clean_images(img).cpu()
    n = img.shape[0]
    mx, mn = img.max(1).values, img.min(1).values
    sat = (mx - mn) / (mx + 1e-6)
    q = (img * 3.999).long().clamp(0, 3)
    rgb_idx = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    rows = []
    for i in range(n):
        h_rgb = torch.bincount(rgb_idx[i].flatten(), minlength=64).float()
        h_sat = torch.histc(sat[i], bins=8, min=0, max=1)
        h_val = torch.histc(mx[i], bins=8, min=0, max=1)
        parts = [h_rgb / h_rgb.sum(), h_sat / h_sat.sum(), h_val / h_val.sum()]
        for s in (1, 2):
            g = gray[i:i + 1, None]
            if s > 1:
                g = F.avg_pool2d(g, s)
            gx = g[..., :, 1:] - g[..., :, :-1]; gy = g[..., 1:, :] - g[..., :-1, :]
            mag = torch.sqrt(gx[..., :-1, :].square() + gy[..., :, :-1].square() + 1e-12)
            hh = torch.histc(torch.log10(mag + 1e-3), bins=12, min=-3, max=0.3)
            parts.append(hh / hh.sum())
        parts.append(F.adaptive_avg_pool2d(img[i:i + 1], 8).flatten())
        rows.append(torch.cat(parts))
    return torch.stack(rows)


def image_scalars(img):
    """Per-image numbers: share of strongly coloured pixels, mean edge strength, share of high-frequency energy."""
    img = _clean_images(img).cpu()
    mx, mn = img.max(1).values, img.min(1).values
    sat = (mx - mn) / (mx + 1e-6)
    colour = ((sat > 0.35) & (mx > 0.25)).float().mean(dim=(1, 2))
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    gx = gray[:, :, 1:] - gray[:, :, :-1]; gy = gray[:, 1:, :] - gray[:, :-1, :]
    edge = torch.sqrt(gx[:, :-1, :].square() + gy[:, :, :-1].square() + 1e-12).mean(dim=(1, 2))
    spec = torch.fft.rfft2(gray - gray.mean(dim=(1, 2), keepdim=True)).abs().square()
    fy = torch.fft.fftfreq(gray.shape[1])[:, None].abs(); fx = torch.fft.rfftfreq(gray.shape[2])[None, :]
    rad = torch.sqrt(fy.square() + fx.square())
    hf = (spec * (rad > 0.25)).sum(dim=(1, 2)) / spec.sum(dim=(1, 2)).clamp(min=1e-12)
    return {"colour_share": colour, "edge_strength": edge, "high_freq_share": hf}


def load_deep_features(device, which="auto"):
    """Returns (name, fn) with fn: [N,3,H,W] in [0,1] -> [N,D] features, or (None, reasons) if nothing could be loaded."""
    import torch.nn.functional as F
    reasons = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def prep(x):
        x = F.interpolate(_clean_images(x).to(device), size=(224, 224), mode="bilinear", align_corners=False)
        return (x - mean) / std

    if which in ("auto", "dinov2"):
        try:
            from transformers import AutoModel
            net = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

            @torch.no_grad()
            def f(x):
                h = net(pixel_values=prep(x)).last_hidden_state
                return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float().cpu()
            return "DINOv2-small (transformers, class token + mean patch token)", f
        except Exception as e:
            reasons.append(f"transformers DINOv2: {type(e).__name__}: {str(e)[:120]}")
        try:
            net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

            @torch.no_grad()
            def f(x):
                o = net.forward_features(prep(x))
                return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], dim=1).float().cpu()
            return "DINOv2-small (torch.hub, class token + mean patch token)", f
        except Exception as e:
            reasons.append(f"torch.hub DINOv2: {type(e).__name__}: {str(e)[:120]}")
    return None, reasons


def _extract(fn, images, chunk=64):
    return torch.cat([fn(images[i:i + chunk]) for i in range(0, len(images), chunk)])


def kid_score(a, b, subset=100, n_subsets=50, seed=0):
    """Unbiased kernel distance (polynomial kernel, Binkowski et al.) averaged over random subsets: (mean, sd)."""
    a, b = a.double(), b.double()
    d, m = a.shape[1], min(subset, len(a), len(b))
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_subsets):
        x = a[torch.randperm(len(a), generator=g)[:m]]; y = b[torch.randperm(len(b), generator=g)[:m]]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        vals.append(float((kxx.sum() - kxx.diag().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
                          - 2 * kxy.mean()))
    return float(np.mean(vals)), float(np.std(vals))


def nearest_similarity(a, b):
    """Cosine similarity of each row of a to its nearest row of b."""
    a = torch.nn.functional.normalize(a.double(), dim=1); b = torch.nn.functional.normalize(b.double(), dim=1)
    return (a @ b.T).max(dim=1).values


def view_similarity_gap(feats, n_views):
    """Mean cosine between different views of the SAME scene minus between views of DIFFERENT scenes."""
    f = torch.nn.functional.normalize(feats.double(), dim=1)
    n_scenes = len(f) // n_views
    sim = f @ f.T
    scene = torch.arange(len(f)) // n_views
    view = torch.arange(len(f)) % n_views
    same_scene, diff_view = scene[:, None] == scene[None, :], view[:, None] != view[None, :]
    within = sim[same_scene & diff_view].mean()
    between = sim[~same_scene & diff_view].mean()
    return float(within), float(between), float(within - between)


@torch.no_grad()
def run_gen_metrics(model, ctx, n_gen, steps, seed, device, out_dir, feature_net="auto", use_start=True, train_ref=None):
    """Distribution metrics on RENDERED views: generated scenes against the autoencoder's reconstructions of real scenes.

    Reference images are the autoencoder reconstructions of the real scenes in `ctx` (same decoder as the generated
    ones, so decoder blur cancels; the numbers measure the gap in the latent distribution, not the decoder).
    Generated scene i is rendered with the cameras of real scene i (modulo the number of real scenes); a flow conditioned on
    the starting positions gets those cameras' starting positions (use_start False: none, needs start_dropout > 0).
    train_ref: reconstructions of TRAINING scenes' views [M, 3, H, W] for the copy check (run with real scenes from the
    validation split): how close generated views are to training views, against how close real validation views are.
    """
    n_real = len(ctx)
    ref = torch.cat([c["recon"] for c in ctx])                  # [n_real * V, 3, H, W]
    n_views = ctx[0]["recon"].shape[0]
    zs, ps, name = [], [], None
    starts = ctx_starts(ctx, n_gen) if (getattr(model, "cond_start", False) and use_start) else None
    for c in range(0, n_gen, 16):
        m_ = min(16, n_gen - c)
        out = generate_samples(model, m_, steps, seed + c, device, start=None if starts is None else starts[c:c + m_])
        name, zg, pg = out[0]                                    # the run's own sampler
        zs.append(zg); ps.append(pg)
    zg, pg = torch.cat(zs), torch.cat(ps)
    gen_stacks = [model.render(zg[i:i + 1], pg[i:i + 1], ctx[i % n_real]["ray_o"], ctx[i % n_real]["ray_d"], IMAGE_HW)[0]
                  for i in range(n_gen)]
    gen = torch.cat(gen_stacks)
    print(f"\nGeneration metrics: {n_gen} generated scenes ({len(gen)} views) against {n_real} real scenes ({len(ref)} views).")
    print(f"  sampler: {name}")

    half = np.arange(n_real) % 2                                  # real-against-real floor: even against odd scenes
    def ref_idx(mask):
        return torch.tensor([i * n_views + v for i in range(n_real) if mask[i] for v in range(n_views)]).long()
    ref_a, ref_b = ref[ref_idx(half == 0)], ref[ref_idx(half == 1)]

    results = {"sampler": name, "n_generated_scenes": n_gen, "n_real_scenes": n_real, "views_per_scene": n_views}
    spaces = []
    ll = lowlevel_descriptor(torch.cat([ref, gen]))
    mu, sd_ = ll[:len(ref)].mean(0), ll[:len(ref)].std(0)
    sd_ = torch.maximum(sd_, sd_.median().clamp(min=1e-6))   # never amplify dimensions that barely vary in the real data
    ll = ((ll - mu) / sd_).clamp(-10, 10)
    spaces.append(("low-level colour/edge/layout descriptor", ll[:len(ref)], ll[len(ref):], ll[:len(ref)][ref_idx(half == 0)],
                   ll[:len(ref)][ref_idx(half == 1)]))
    ft = None
    deep_name, deep = load_deep_features(device, feature_net)
    if deep_name is None:
        print("  deep features not used" + (" (" + "; ".join(deep) + ")" if deep else " (--feature-net lowlevel)") + "; only the low-level metrics are reported.")
        results["deep_feature_error"] = deep
    else:
        fr = _extract(deep, ref); fg = _extract(deep, gen)
        ft = None if train_ref is None else _extract(deep, train_ref)
        spaces.append((deep_name, fr, fg, fr[ref_idx(half == 0)], fr[ref_idx(half == 1)]))

    results["spaces"] = {}
    for sname, fr, fg, fa, fb in spaces:
        kid_gen = kid_score(fr, fg, seed=seed); kid_floor = kid_score(fa, fb, seed=seed)
        nn_gen = nearest_similarity(fg, fa); nn_floor = nearest_similarity(fb, fa)
        thr = float(torch.quantile(nn_floor, 0.95))
        w_r, b_r, g_r = view_similarity_gap(fr, n_views); w_g, b_g, g_g = view_similarity_gap(fg, n_views)
        results["spaces"][sname] = {
            "kid_x1000_generated_vs_real": [1000 * kid_gen[0], 1000 * kid_gen[1]],
            "kid_x1000_real_even_vs_odd_floor": [1000 * kid_floor[0], 1000 * kid_floor[1]],
            "nearest_real_similarity_generated": float(nn_gen.mean()), "nearest_real_similarity_real_floor": float(nn_floor.mean()),
            "share_generated_above_floor_95th_percentile": float((nn_gen > thr).float().mean()),
            "view_similarity_within_between_gap_real": [w_r, b_r, g_r], "view_similarity_within_between_gap_generated": [w_g, b_g, g_g]}
        print(f"\n  Feature space: {sname}")
        print(f"    KID x1000, generated against real   {1000 * kid_gen[0]:9.3f} (sd over subsets {1000 * kid_gen[1]:.3f})")
        print(f"    KID x1000, real against real (floor){1000 * kid_floor[0]:9.3f} (sd {1000 * kid_floor[1]:.3f}; half of the real scenes against the other half)")
        print(f"    nearest real image, cosine similarity: generated {float(nn_gen.mean()):.3f}, real-to-real floor {float(nn_floor.mean()):.3f}; "
              f"{100 * float((nn_gen > thr).float().mean()):.1f}% of generated views are closer than the floor's 95th percentile (copies would be near 100%)")
        print(f"    views of one scene: similarity within {w_r:.3f} / between {b_r:.3f} / gap {g_r:.3f} for real; {w_g:.3f} / {b_g:.3f} / {g_g:.3f} for generated")
        if ft is not None and sname == deep_name:
            nn_gt = nearest_similarity(fg, ft); nn_rt = nearest_similarity(fr, ft)
            thr_t = float(torch.quantile(nn_rt, 0.95)); share_t = float((nn_gt > thr_t).float().mean())
            results["spaces"][sname]["copy_check_vs_training_views"] = {
                "n_training_views": int(len(ft)), "nearest_training_similarity_generated": float(nn_gt.mean()),
                "nearest_training_similarity_real_validation": float(nn_rt.mean()),
                "share_generated_above_real_validation_95th_percentile": share_t}
            print(f"    COPY CHECK against {len(ft)} training views: nearest training view, cosine similarity: generated {float(nn_gt.mean()):.3f}, "
                  f"real validation views {float(nn_rt.mean()):.3f}; {100 * share_t:.1f}% of generated views are closer than the real validation "
                  f"views' 95th percentile (about 5% is expected without copying; copies push the generated similarity up and this share towards 100%)")

    sr, sg = image_scalars(ref), image_scalars(gen)
    print("\n  Image statistics (mean, sd over views):    real          generated")
    results["image_statistics"] = {}
    for k in sr:
        print(f"    {k:22s} {float(sr[k].mean()):10.4f} ({float(sr[k].std()):.4f})   {float(sg[k].mean()):10.4f} ({float(sg[k].std()):.4f})")
        results["image_statistics"][k] = {"real": [float(sr[k].mean()), float(sr[k].std())], "generated": [float(sg[k].mean()), float(sg[k].std())]}
    save_grid(out_dir / "6_gen_metrics_scenes.png", [("real: autoencoder reconstruction", [c["recon"] for c in ctx[:4]]),
                                                     (f"generated: {name}", gen_stacks[:4]), ("generated (next 4)", gen_stacks[4:8])], label_width=330)
    (out_dir / "gen_metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results



# ----------------------------------------------------------------------------- real-feature projection test
def _feature_fn(device, which, ref_images):
    """(name, fn): DINOv2 features when available, else the low-level descriptor standardised on the reference."""
    name, deep = load_deep_features(device, which)
    if name is not None:
        return name, deep
    base = lowlevel_descriptor(ref_images)
    mu, sd_ = base.mean(0), base.std(0)
    sd_ = torch.maximum(sd_, sd_.median().clamp(min=1e-6))
    return "low-level descriptor (not reliable as a quality score)", lambda x: ((lowlevel_descriptor(x) - mu) / sd_).clamp(-10, 10)


def _project(z, center, basis, k):
    c = z - center
    top = basis[:, :k]
    return center + (c @ top) @ top.T


@torch.no_grad()
def run_real_projection(model, ctx, out_dir, ks=(16, 32, 64, 128)):
    """Can the decoder render REAL features that were projected onto their top-K principal components?

    The basis is fitted on the even-numbered real scenes and tested on the odd-numbered ones (so the test is
    out of sample). Two variants: 'global-mean PCA' (the features as they are) and 'slot-centred PCA' (the
    per-token-slot mean of the fit scenes is removed first and added back; 47.8% of the feature variance is
    fixed per slot, so a flow would only have to generate the rest). Positions are untouched. PSNR is against
    the decoder's own reconstruction of the unprojected latent; edge strength, high-frequency share and colour
    share of the render are given as a ratio to that reconstruction (a blurred floor shows as a ratio below 1).
    """
    fit = [c for i, c in enumerate(ctx) if i % 2 == 0]
    test = [c for i, c in enumerate(ctx) if i % 2 == 1]
    zf = torch.cat([c["z"] for c in fit])
    slot_mean = zf.mean(dim=0, keepdim=True)
    g_mean, g_basis, g_evals = feature_pca(zf)
    s_mean, s_basis, s_evals = feature_pca(zf - slot_mean)
    variants = [("global-mean PCA", g_mean, g_basis, g_evals), ("slot-centred PCA", slot_mean + s_mean, s_basis, s_evals)]
    base_stats = image_scalars(torch.cat([c["recon"] for c in test]))
    print(f"\nReal-feature projection: basis fitted on {len(fit)} scenes, tested on {len(test)} other scenes "
          f"({len(test) * test[0]['recon'].shape[0]} views).")
    print("  Share of the fit features' variance in the top-k components: "
          + ", ".join(f"{name.split()[0]} k={k}: {100 * float(ev[:k].sum() / ev.sum()):.1f}%" for name, _, _, ev in variants for k in (32, 64)))
    print(f"  {'variant':18s} {'K':>4s} {'PSNR dB (sd over scenes)':>26s} {'feature MSE':>12s} {'edge ratio':>11s} {'high-freq ratio':>16s} {'colour ratio':>13s}")
    results, rows = [], [("real: autoencoder reconstruction", [c["recon"] for c in test[:2]])]
    for name, center, basis, _ in variants:
        for k in ks[::-1]:
            scores, mses, imgs_all = [], [], []
            for c in test:
                zk = _project(c["z"], center, basis, k)
                img = model.render(zk, c["pos"], c["ray_o"], c["ray_d"], IMAGE_HW)[0]
                scores.append(psnr(img, c["recon"])); mses.append(float((zk - c["z"]).square().mean())); imgs_all.append(img)
            st = image_scalars(torch.cat(imgs_all))
            ratio = {m: (float(st[m].mean() / base_stats[m].mean()) if float(base_stats[m].mean()) > 1e-6 else None) for m in st}
            row = {"variant": name, "k": k, "psnr_mean": float(np.mean(scores)), "psnr_sd_over_scenes": float(np.std(scores)),
                   "feature_mse": float(np.mean(mses)), "edge_ratio": ratio["edge_strength"],
                   "high_freq_ratio": ratio["high_freq_share"], "colour_ratio": ratio["colour_share"]}
            results.append(row)
            print(f"  {name:18s} {k:4d} {row['psnr_mean']:18.2f} ({row['psnr_sd_over_scenes']:4.2f}) {row['feature_mse']:12.4f} "
                  + "".join(("{:>{w}}".format("n/a", w=w) if v is None else "{:{w}.3f}".format(v, w=w)) for v, w in
                            ((row["edge_ratio"], 11), (row["high_freq_ratio"], 16), (row["colour_ratio"], 13))))
            rows.append((f"{name}, top {k}", imgs_all[:2]))
    save_grid(out_dir / "7_real_projection.png", rows, label_width=330)
    (out_dir / "real_projection.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


# ----------------------------------------------------------------------------- noise ladder
@torch.no_grad()
def run_ladder(model, ctx, steps, seed, device, out_dir, feature_net="auto", t0s=(0.9, 0.7, 0.5, 0.3, 0.1), chunk=16, use_start=True):
    """Where does the flow lose 'realness', and is that the flow or the decoder?

    Real latents of the even-numbered scenes are noised to time t0 and denoised by the flow (t0 0.9 = light noise,
    0.1 = almost pure noise), then rendered with their own cameras. For each level the same real latents also get
    plain Gaussian noise of the SAME size (rms error per feature and per position coordinate that the flow's output
    shows) and are rendered. Distance to real views (KID) is measured against the odd-numbered scenes'
    reconstructions; the floor is even-numbered against odd-numbered reconstructions. If the flow's renders are
    much farther from real than the plain-noise renders at the same error size, the flow's error has a structure
    the decoder is sensitive to; if equal, the size of the error is what matters.
    """
    even = [c for i, c in enumerate(ctx) if i % 2 == 0]
    odd = [c for i, c in enumerate(ctx) if i % 2 == 1]
    ref = torch.cat([c["recon"] for c in odd])
    ref_even = torch.cat([c["recon"] for c in even])
    fname, fn = _feature_fn(device, feature_net, torch.cat([ref, ref_even]))
    f_ref = _extract(fn, ref)
    floor = kid_score(f_ref, _extract(fn, ref_even), seed=seed)
    zA = torch.cat([c["z"] for c in even]); pA = torch.cat([c["pos"] for c in even])
    sA = torch.cat([c["start"] for c in even]) if (getattr(model, "cond_start", False) and use_start) else None
    gen = torch.Generator(device=device).manual_seed(seed)

    def render_all(zs, ps):
        return [model.render(zs[i:i + 1], ps[i:i + 1], even[i]["ray_o"], even[i]["ray_d"], IMAGE_HW)[0] for i in range(len(even))]

    print(f"\nNoise ladder: {len(even)} real scenes denoised by the flow; reference {len(odd)} other real scenes. Feature space: {fname}")
    print(f"  KID x1000 floor (real against real): {1000 * floor[0]:.3f} (sd {1000 * floor[1]:.3f})")
    print(f"  {'start t (sigma)':16s} {'rms err feat/pos':>17s} {'PSNR flow':>10s} {'PSNR noise':>11s} {'KID flow':>12s} {'KID same-size noise':>20s}"
          f" {'KID flow pos + real feat':>25s} {'KID real pos + flow feat':>25s}")
    results, rows = [], [("real: autoencoder reconstruction", [c["recon"] for c in even[:2]])]
    for t0 in t0s:
        zs, ps = [], []
        for s in range(0, len(zA), chunk):
            zd, pd = denoise_from(model, zA[s:s + chunk], pA[s:s + chunk], t0, steps, seed=seed + s,
                                  start=None if sA is None else sA[s:s + chunk])
            zs.append(zd); ps.append(pd)
        zd, pd = torch.cat(zs), torch.cat(ps)
        err_f = float((zd - zA).square().mean().sqrt()); err_p = float((pd - pA).square().mean().sqrt())
        imgs_flow = render_all(zd, pd)
        zn = zA + torch.randn(zA.shape, device=device, generator=gen) * err_f
        pn = pA + torch.randn(pA.shape, device=device, generator=gen) * err_p
        imgs_noise = render_all(zn, pn)
        p_flow = float(np.mean([psnr(imgs_flow[i], even[i]["recon"]) for i in range(len(even))]))
        p_noise = float(np.mean([psnr(imgs_noise[i], even[i]["recon"]) for i in range(len(even))]))
        kid_flow = kid_score(f_ref, _extract(fn, torch.cat(imgs_flow)), seed=seed)
        kid_noise = kid_score(f_ref, _extract(fn, torch.cat(imgs_noise)), seed=seed)
        imgs_pos = render_all(zA, pd)       # flow positions, real features
        imgs_feat = render_all(zd, pA)      # real positions, flow features
        kid_pos = kid_score(f_ref, _extract(fn, torch.cat(imgs_pos)), seed=seed)
        kid_feat = kid_score(f_ref, _extract(fn, torch.cat(imgs_feat)), seed=seed)
        p_pos = float(np.mean([psnr(imgs_pos[i], even[i]["recon"]) for i in range(len(even))]))
        p_feat = float(np.mean([psnr(imgs_feat[i], even[i]["recon"]) for i in range(len(even))]))
        results.append({"start_t": t0, "start_sigma": (1 - t0) / t0, "rms_error_feat": err_f, "rms_error_pos": err_p,
                        "psnr_flow": p_flow, "psnr_same_size_noise": p_noise,
                        "kid_x1000_flow": 1000 * kid_flow[0], "kid_x1000_flow_sd": 1000 * kid_flow[1],
                        "kid_x1000_same_size_noise": 1000 * kid_noise[0], "kid_x1000_same_size_noise_sd": 1000 * kid_noise[1],
                        "kid_x1000_flow_positions_real_features": 1000 * kid_pos[0], "psnr_flow_positions_real_features": p_pos,
                        "kid_x1000_real_positions_flow_features": 1000 * kid_feat[0], "psnr_real_positions_flow_features": p_feat})
        print(f"  {t0:.1f} ({(1 - t0) / t0:5.2f})".ljust(18) + f" {err_f:7.3f} / {err_p:6.3f} {p_flow:10.2f} {p_noise:11.2f} "
              f"{1000 * kid_flow[0]:12.3f} {1000 * kid_noise[0]:20.3f} {1000 * kid_pos[0]:25.3f} {1000 * kid_feat[0]:25.3f}")
        rows.append((f"flow, start t={t0}", imgs_flow[:2])); rows.append((f"real + noise of the same size, t={t0}", imgs_noise[:2]))
        if t0 in (0.7, 0.5, 0.3):
            rows.append((f"flow positions + real features, t={t0}", imgs_pos[:2])); rows.append((f"real positions + flow features, t={t0}", imgs_feat[:2]))
    save_grid(out_dir / "8_noise_ladder.png", rows, label_width=330)
    (out_dir / "noise_ladder.json").write_text(json.dumps({"feature_space": fname, "kid_x1000_floor": [1000 * floor[0], 1000 * floor[1]],
                                                           "levels": results}, indent=2, allow_nan=False) + "\n")
    return results


# ----------------------------------------------------------------------------- floor against object split of the structure test
def floor_bands(pos_real, axis=2, bins=60):
    """Height bands from the real positions: the flat floor is the sharp peak of the height histogram.

    floor: height up to the peak plus its half width (on the upper side); objects: above the peak plus two half widths.
    The same absolute cuts are applied to generated positions.
    """
    zv = pos_real[..., axis].reshape(-1).float().cpu().numpy()
    hist, edges = np.histogram(zv, bins=bins)
    k = int(hist.argmax()); width = float(edges[1] - edges[0])
    mode = float(0.5 * (edges[k] + edges[k + 1])); peak = float(hist[k])
    j = k
    while j < len(hist) - 1 and hist[j] > 0.5 * peak:
        j += 1
    hw = max(float(0.5 * (edges[j] + edges[j + 1])) - mode, width)
    return {"axis": axis, "mode": mode, "half_width": hw, "floor_max": mode + hw, "object_min": mode + 2 * hw,
            "share_in_peak_bin": peak / len(zv)}


def band_structure(z, pos, slot_mean, mask, near_k=4, view_size=48, min_tokens=24):
    """Structure numbers computed only among the tokens selected by mask [S, N] (per scene, then averaged)."""
    S, N, _ = z.shape
    grp = torch.clamp(torch.arange(N, device=z.device) // view_size, max=N // view_size)
    r = z.float() - slot_mean.float()
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rows = []
    for s in range(S):
        idx = mask[s].nonzero().squeeze(-1)
        if len(idx) < min_tokens:
            continue
        P = pos[s, idx].float(); D = torch.cdist(P, P)
        D = D.masked_fill(torch.eye(len(idx), dtype=torch.bool, device=D.device), float("inf"))
        k = min(near_k, len(idx) - 1)
        srt, nn = D.topk(k, dim=1, largest=False)
        rs = r[s, idx]; C = rs @ rs.T
        g = grp[idx]
        nn1 = nn[:, 0]; c1 = C[torch.arange(len(idx), device=C.device), nn1]; diff = g[nn1] != g
        rows.append({"tokens": float(len(idx)), "cos_nearest": float(torch.gather(C, 1, nn).mean()),
                     "same_view_share": float((g[nn] == g[:, None]).float().mean()), "nn1_dist": float(srt[:, 0].mean()),
                     "nn1_other_view_share": float(diff.float().mean()),
                     "cos_nn1_same_view": float(c1[~diff].mean()) if bool((~diff).any()) else float("nan"),
                     "cos_nn1_other_view": float(c1[diff].mean()) if bool(diff.any()) else float("nan")})
    if not rows:
        return None
    out = {k: float(np.nanmean([r_[k] for r_ in rows])) for k in rows[0]}
    out["scenes_used"] = len(rows)
    return out


def run_height_split(z1, pos1, slot_mean, gen_sets, out_dir):
    """Does the missing cross-view structure sit in the floor tokens, the object tokens, or both?"""
    b = floor_bands(pos1)
    print(f"\nFloor against object split (height axis {b['axis']}, flow's normalised space): peak at {b['mode']:.3f}, half width "
          f"{b['half_width']:.3f}; floor = height <= {b['floor_max']:.3f}, objects = height > {b['object_min']:.3f}, tokens in between are left out.")
    print(f"  (share of real tokens in the peak bin of the height histogram: {100 * b['share_in_peak_bin']:.1f}%; "
          "a low value means there is no clear flat floor and this split is not meaningful)")
    sets = [("real validation latents", z1, pos1)] + list(gen_sets)
    out = {"bands": b, "sets": {}}
    keys = [("tokens", "tokens per scene", "{:8.1f}"), ("cos_nearest", "feature agreement, 4 nearest", "{:8.3f}"),
            ("same_view_share", "nearest from same view (real < chance)", "{:8.3f}"), ("nn1_dist", "nearest-token distance", "{:8.3f}"),
            ("nn1_other_view_share", "nearest neighbour from another view", "{:8.3f}"),
            ("cos_nn1_same_view", "agreement with it, same view", "{:8.3f}"), ("cos_nn1_other_view", "agreement with it, other view", "{:8.3f}")]
    print("\n  Height occupancy (share of tokens): floor / between / objects, and mean height")
    for name, z, pos in sets:
        h = pos[..., b["axis"]]
        fl = float((h <= b["floor_max"]).float().mean()); ob = float((h > b["object_min"]).float().mean())
        print(f"    {name[:60]:60s} {100 * fl:6.1f}% / {100 * (1 - fl - ob):5.1f}% / {100 * ob:5.1f}%   mean height {float(h.mean()):6.3f}")
        out["sets"][name] = {"floor_share": fl, "object_share": ob, "mean_height": float(h.mean())}
    for band, label in (("floor", "FLOOR tokens"), ("objects", "OBJECT tokens")):
        print(f"\n  {label}")
        print("  " + f"{'':42s}" + "".join(f"{n[:24]:>26s}" for n, _, _ in sets))
        stats = []
        for name, z, pos in sets:
            h = pos[..., b["axis"]]
            mask = (h <= b["floor_max"]) if band == "floor" else (h > b["object_min"])
            s_ = band_structure(z, pos, slot_mean, mask)
            stats.append(s_); out["sets"][name][band] = s_
        for key, label_, fmt in keys:
            cells = [("      n/a" if s_ is None else fmt.format(s_[key])) for s_ in stats]
            print("  " + f"{label_:42s}" + "".join(f"{c:>26s}" for c in cells))
    (out_dir / "height_split.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return out


# ----------------------------------------------------------------------------- main
# ----------------------------------------------------------------------------- position origin
N_RAY_TOKENS = 240        # tokens 0..239 start on the input rays (5 views x 48); the last 16 start at the scene centre
VIEW_SIZE = 48


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def anova_within_share(v):
    """v: [S, R, T, 3] (scene, camera set, token, axis). Share of the per-token variance that differs between camera
    sets of the same scene (within-scene), by a one-way variance decomposition, summed over tokens and axes."""
    v = v.double()
    S, R = v.shape[:2]
    within = v.var(dim=1, unbiased=True).mean(dim=0)                       # [T,3]
    between = (v.mean(dim=1).var(dim=0, unbiased=True) - within / R).clamp_min(0)
    tot = (within + between).sum()
    return float(within.sum() / tot) if tot > 0 else float("nan")


def pos_structure(P, view_size=VIEW_SIZE, near_k=4):
    """P: [S, R, N, 3]. Nearest-token distances and how many of the nearest tokens come from the same view slot."""
    S, R, N, _ = P.shape
    x = P.reshape(S * R, N, 3).double()
    d = torch.cdist(x, x)
    d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    srt, idx = torch.sort(d, dim=2)
    slot = torch.clamp(torch.arange(N) // view_size, max=N // view_size)
    nn_slot = slot[idx[:, :, :near_k]]                                      # [M,N,k]
    same = (nn_slot == slot[None, :, None]).double().mean().item()
    other1 = (slot[idx[:, :, 0]] != slot[None, :]).double().mean().item()
    return {"nn1": srt[:, :, 0].mean().item(), "nn5": srt[:, :, min(4, N - 2)].mean().item(),
            "near_same_view_frac": same, "nearest_from_other_view_frac": other1}


def position_origin_stats(P, P0, n_ray=N_RAY_TOKENS, view_size=VIEW_SIZE):
    """P, P0: [S, R, N, 3], final and starting token positions in the flow's normalised space (real rms about 1).

    Per token group: size of the move |P - P0|, and how much of the spread of P the start explains, three ways
    (start used as is; one global linear map; a separate linear fit per token and axis, the most generous)."""
    P, P0 = P.double(), P0.double()
    S, R, N, _ = P.shape
    n_ray = min(n_ray, N)
    groups = {"ray": slice(0, n_ray)}
    if N > n_ray:
        groups["centre"] = slice(n_ray, N)
    out = {"scenes": S, "camera_sets": R, "tokens": N, "groups": {}}
    for name, sl in groups.items():
        p, p0 = P[:, :, sl], P0[:, :, sl]
        dlt = p - p0
        dn = dlt.norm(dim=-1)
        tot = (p - p.reshape(-1, 3).mean(0)).square().sum(-1).mean()
        ex_unit = 1 - dlt.square().sum(-1).mean() / tot
        ones = torch.ones(p0.reshape(-1, 3).shape[0], 1, dtype=p.dtype)
        X, Y = torch.cat([p0.reshape(-1, 3), ones], 1), p.reshape(-1, 3)
        coef = torch.linalg.lstsq(X, Y).solution
        ex_lin = 1 - (Y - X @ coef).square().sum(-1).mean() / tot
        x = p0.reshape(S * R, -1, 3); y = p.reshape(S * R, -1, 3)
        xm, ym = x.mean(0), y.mean(0)
        varx = (x - xm).square().mean(0)
        alpha = torch.where(varx > 1e-10, ((x - xm) * (y - ym)).mean(0) / varx.clamp_min(1e-10), torch.zeros_like(varx))
        sse = ((y - ym) - alpha * (x - xm)).square().sum(-1).mean()
        ex_slot = 1 - sse / tot
        out["groups"][name] = {
            "tokens": int(p.shape[2]),
            "final_spread_rms": float((tot / 3).sqrt()),          # per coordinate, like the other flow-space numbers
            "start_spread_rms": float(((p0 - p0.reshape(-1, 3).mean(0)).square().sum(-1).mean() / 3).sqrt()),
            "move_rms": float((dlt.square().sum(-1).mean() / 3).sqrt()),   # per coordinate
            "move_over_spread": float((dlt.square().sum(-1).mean() / tot).sqrt()),
            "move_median": float(dn.median()), "move_p90": float(dn.reshape(-1).quantile(0.9)),   # distances (3D)
            "explained_unit_slope": float(ex_unit), "explained_global_linear": float(ex_lin),
            "explained_per_token_linear": float(ex_slot),
            "camera_set_share_final": anova_within_share(p) if R > 1 and S > 1 else None,
            "camera_set_share_start": anova_within_share(p0) if R > 1 and S > 1 else None,
        }
    out["structure_final"] = pos_structure(P, view_size)
    out["structure_start"] = pos_structure(P0, view_size)
    return out


def position_origin_verdict(stats, move_build=0.3, share_build=0.9, move_drop=0.6, share_drop=0.7):
    g = stats["groups"]["ray"]
    if g["move_rms"] < move_build and g["explained_unit_slope"] >= share_build:
        return "BUILD: the cameras fix most of the position structure (move below %.1f and share at least %.0f%%)" % (move_build, 100 * share_build)
    if g["move_rms"] >= move_drop or g["explained_unit_slope"] < share_drop:
        return "DROP: the encoder creates most of the position structure (move %.1f or more, or share below %.0f%%)" % (move_drop, 100 * share_drop)
    return "BETWEEN the thresholds: neither clearly camera-determined nor clearly encoder-created"


def ray_split(patch_fn, patch_size, ray_o, ray_d, p_raw, p0_raw, n_ray=N_RAY_TOKENS, view_size=VIEW_SIZE):
    """Split the move of each ray token into the part along its ray and the part across it (autoencoder frame).

    The ray of a token is found as the patch-mean ray of its assumed view slot that best matches the direction from
    that view's camera to the token's start position; the same match is checked against all five views."""
    o_pm, d_pm = patch_fn(ray_o, ray_d, patch_size)
    if o_pm.dim() != 3 or d_pm.dim() != 4 or o_pm.shape[-1] != 3 or d_pm.shape[-1] != 3:
        raise ValueError(f"unexpected patch_mean_rays shapes {tuple(o_pm.shape)} and {tuple(d_pm.shape)}")
    o = o_pm[0].double().cpu(); d = d_pm[0].double().cpu()
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    V = o.shape[0]
    T = min(n_ray, V * view_size)
    p0 = p0_raw[0, :T].double().cpu(); p = p_raw[0, :T].double().cpu()
    slot = torch.arange(T) // view_size
    diff = p0[:, None, :] - o[None]                                         # [T,V,3]
    unit = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos = torch.einsum("tvc,vkc->tvk", unit, d)                             # [T,V,Np]
    best, arg = cos.max(dim=-1)                                             # [T,V]
    cos_assumed = best[torch.arange(T), slot]
    ok = (best.argmax(dim=1) == slot).double().mean().item()
    u = d[slot, arg[torch.arange(T), slot]]                                 # [T,3]
    delta = p - p0
    par = (delta * u).sum(-1)
    perp = delta - par[:, None] * u
    return {"par2": par.square().sum().item(), "perp2": perp.square().sum().item(), "n": T,
            "cos_assumed_median": cos_assumed.median().item(), "slot_match": ok}


@torch.no_grad()
def collect_position_origin(model, ae, dataset, device, edict, n_scenes, rigs, seed, patch_fn=None):
    """Encode n_scenes scenes under `rigs` camera sets each; return final and starting positions in the flow's space."""
    rng = np.random.RandomState(seed)
    try:
        n_avail = int(dataset.c2w[int(dataset.scene_indices[0])].shape[0])
    except Exception:
        n_avail = 16
        print("  (could not read the number of views per scene; assuming 16)")
    n_scenes = min(n_scenes, len(dataset))
    norm = getattr(model, "normaliser", None)
    scale = float(norm.pos_scale) if norm is not None else 1.0
    Ps, P0s, vis, split_acc, split_err = [], [], [], {"par2": 0.0, "perp2": 0.0, "n": 0, "cos": [], "match": []}, None
    for i in range(n_scenes):
        sidx = int(dataset.scene_indices[i])
        sets = [list(VIEW_IDXS)] + [sorted(rng.choice(n_avail, size=len(VIEW_IDXS), replace=False).tolist()) for _ in range(rigs - 1)]
        rows_p, rows_p0 = [], []
        for views in sets:
            sd_ = dataset.scene_batch(scene_idx=sidx, view_indices=views, device=device)
            ray_o, ray_d = ae.process_data.compute_rays(sd_["c2w"], sd_["fxfycxcy"], h=IMAGE_HW[0], w=IMAGE_HW[1], device=device)
            enc = edict({"image": sd_["image"], "ray_o": ray_o, "ray_d": ray_d, "image_h_w": IMAGE_HW})
            z, p, p0 = ae.encode(enc)
            if norm is not None:
                pn, _ = norm.normalise(p, z); p0n, _ = norm.normalise(p0, z)
            else:
                pn, p0n = p, p0
            rows_p.append(pn[0].float().cpu()); rows_p0.append(p0n[0].float().cpu())
            if patch_fn is not None and split_err is None:
                try:
                    r = ray_split(patch_fn, getattr(ae, "patch_size", 8), ray_o, ray_d, p, p0)
                    split_acc["par2"] += r["par2"]; split_acc["perp2"] += r["perp2"]; split_acc["n"] += r["n"]
                    split_acc["cos"].append(r["cos_assumed_median"]); split_acc["match"].append(r["slot_match"])
                except Exception as e:                                       # optional part: never stop the main result
                    split_err = f"{type(e).__name__}: {e}"
        Ps.append(torch.stack(rows_p)); P0s.append(torch.stack(rows_p0))
        vis.append(sets[0])
    return torch.stack(Ps), torch.stack(P0s), scale, split_acc, split_err


def plot_position_origin(P, P0, path, n_scenes=2, view_size=VIEW_SIZE):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except Exception as e:
        print(f"  (no figure: {e})")
        return False
    n = min(n_scenes, P.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.6 * n), squeeze=False)
    cols = plt.get_cmap("tab10")
    N = P.shape[2]
    slot = np.clip(np.arange(N) // view_size, 0, N // view_size)
    for r in range(n):
        p, p0 = P[r, 0].numpy(), P0[r, 0].numpy()
        for c, (a, b, lab) in enumerate([(0, 1, "x against y (from above)"), (0, 2, "x against z (from the side)")]):
            ax = axes[r, c]
            segs = [[(p0[j, a], p0[j, b]), (p[j, a], p[j, b])] for j in range(N)]
            ax.add_collection(LineCollection(segs, colors=[cols(s) if s < 5 else "black" for s in slot], linewidths=0.8, alpha=0.7))
            ax.scatter(p0[:, a], p0[:, b], s=5, c="0.6", zorder=3)
            ax.scatter(p[:, a], p[:, b], s=9, c=[cols(s) if s < 5 else "black" for s in slot], zorder=4)
            ax.set_aspect("equal"); ax.autoscale(); ax.set_title(f"scene {r}, fixed camera set: {lab}", fontsize=10)
            ax.set_xlabel("flow units"); ax.grid(alpha=0.2)
    fig.suptitle("Token positions: start (grey) to final (colour = input view; black = centre tokens)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def run_position_origin(model, ae, dataset, device, edict, n_scenes, rigs, seed, out_dir, patch_fn=None):
    P, P0, scale, split_acc, split_err = collect_position_origin(model, ae, dataset, device, edict, n_scenes, rigs, seed, patch_fn)
    stats = position_origin_stats(P, P0)
    S, R = P.shape[:2]
    print(f"\nPosition origin: {S} scenes x {R} camera sets each (the first is the fixed set {VIEW_IDXS}, the others are random sorted 5 of the available views).")
    print("Positions are in the flow's space (rms 1 per coordinate, the same units as the flow's position errors: 0.05 per coordinate is what the decoder needs for 30 dB).")
    print("'move' = how far the encoder moves a token from its starting position; 'rms' is per coordinate, 'median' and '90th percentile' are 3D distances.")
    names = {"ray": "ray tokens (first 240, start on the input rays)", "centre": "centre tokens (last 16)"}
    for k, g in stats["groups"].items():
        print(f"\n  {names.get(k, k)}: {g['tokens']} tokens; spread per coordinate of the final positions {g['final_spread_rms']:.3f}, of the starting positions {g['start_spread_rms']:.3f}")
        print(f"    move: rms {g['move_rms']:.3f} per coordinate ({100 * g['move_over_spread']:.0f}% of the spread), median distance {g['move_median']:.3f}, 90th percentile {g['move_p90']:.3f}")
        print(f"    share of the final positions' spread explained by the start: as is {100 * g['explained_unit_slope']:.1f}%, "
              f"one global linear map {100 * g['explained_global_linear']:.1f}%, a linear fit per token and axis {100 * g['explained_per_token_linear']:.1f}%")
        if g["camera_set_share_final"] is not None:
            print(f"    share of the position variance that differs between camera sets of the same scene: final {100 * g['camera_set_share_final']:.1f}%, start {100 * g['camera_set_share_start']:.1f}%")
    sf, ss = stats["structure_final"], stats["structure_start"]
    print("\n  Arrangement of all tokens (chance for the same-view share is about 0.18; real latents earlier: nn1 0.180, nn5 0.439, same-view 0.124):")
    print(f"    {'':8s} {'nn1':>7s} {'nn5':>7s} {'same view':>10s} {'nearest from another view':>26s}")
    for lab, s_ in (("final", sf), ("start", ss)):
        print(f"    {lab:8s} {s_['nn1']:7.3f} {s_['nn5']:7.3f} {s_['near_same_view_frac']:10.3f} {s_['nearest_from_other_view_frac']:26.3f}")
    along = None
    if split_acc["n"] > 0:
        tot = split_acc["par2"] + split_acc["perp2"]
        along = {"share_of_squared_move_along_ray": split_acc["par2"] / tot if tot > 0 else None,
                 "rms_along": math.sqrt(split_acc["par2"] / split_acc["n"]) / scale,                 # one direction
                 "rms_across_per_direction": math.sqrt(split_acc["perp2"] / (2 * split_acc["n"])) / scale,   # two directions
                 "median_cos_start_to_ray": float(np.median(split_acc["cos"])), "view_slot_match": float(np.mean(split_acc["match"]))}
        print(f"\n  Move along the token's ray against across it (flow units): along {along['rms_along']:.3f} (one direction), across {along['rms_across_per_direction']:.3f} per direction (two directions); "
              f"{100 * along['share_of_squared_move_along_ray']:.1f}% of the squared move is along the ray.")
        print(f"    check of the token-to-view assumption: the start lies on its assumed view's ray (median cosine {along['median_cos_start_to_ray']:.4f}); "
              f"the best-matching view is the assumed one for {100 * along['view_slot_match']:.1f}% of tokens (should be near 100%).")
    elif split_err:
        print(f"\n  (along/across split skipped: {split_err})")
    verdict = position_origin_verdict(stats)
    print(f"\n  Proposed rule (thresholds are proposals) on the ray tokens: {verdict}")
    drawn = plot_position_origin(P, P0, out_dir / "9_position_origin.png")
    stats["along_across"] = along
    stats["verdict"] = verdict
    stats["view_sets_first"] = VIEW_IDXS
    stats["figure"] = bool(drawn)
    (out_dir / "position_origin.json").write_text(json.dumps(stats, indent=2, allow_nan=True, default=float) + "\n")
    return stats



def choose_checkpoint(directory):
    candidates = sorted(Path(directory).glob("*.pt"))   # same ordering as PlatonicAutoencoder.load_ckpt
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in {directory}")
    return candidates[-1].resolve()


def main():
    args = parse_args()
    from easydict import EasyDict as edict
    from torch.utils.data import default_collate
    from platonic_transformers.utils.config_loader import load_with_defaults
    from psplat.models.platonic_autoencoder import PlatonicAutoencoder

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    fm_path = args.fm_script.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("nvs_fm_module", fm_path)
    fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)

    config = load_with_defaults(dataset_config=str(args.config), cli_args=[])
    config.model.compile = False                       # plain network: simpler and avoids recompiling
    ae_dir = Path(config.autoencoder.checkpoint_dir)
    ae_config = load_with_defaults(dataset_config=str(ae_dir / "config.yaml"), cli_args=[])
    ae = PlatonicAutoencoder(ae_config)
    ae_ckpt = choose_checkpoint(ae_dir)
    state = torch.load(ae_ckpt, map_location="cpu", weights_only=True)
    ae.load_state_dict(state["model"], strict=True)    # fail loudly, unlike load_ckpt
    del state
    ae.requires_grad_(False); ae.eval()
    print(f"Device: {device}\nAutoencoder: {ae_ckpt} (loaded strictly)")

    train_loader, val_loader = fm.load_data(config, ae_config)
    val_set = val_loader.dataset
    if args.split == "train":
        val_set = train_loader.dataset
        if isinstance(val_set, fm.RepeatedDataset):
            val_set = val_set.dataset
        if not hasattr(val_set, "scene_batch"):
            raise RuntimeError("Training set is a subset wrapper; run with dataset_fraction: 1.0 or use --split val.")
    print(f"Real scenes come from the {args.split} split ({len(val_set)} scenes).")

    model = fm.NVSFMModel(config, ae)
    # Lightning checkpoints contain non-tensor objects; this is your own file.
    ckpt = torch.load(args.checkpoint.expanduser(), map_location="cpu", weights_only=False)
    # The network output means different things in "edm" and "x" feature-prediction mode, and the weights load
    # without error either way, so compare the mode saved in the checkpoint with the config's.
    saved_mode, config_mode = ckpt.get("feat_prediction", "edm"), getattr(model, "feat_prediction", "edm")
    if saved_mode != config_mode:
        raise RuntimeError(f"Checkpoint was trained with diffusion.feat_prediction='{saved_mode}' but --config "
                           f"says '{config_mode}'. Use the config the run was trained with.")
    print(f"Feature prediction mode: {config_mode}")
    saved_coupling, config_coupling = ckpt.get("time_coupling", "joint"), getattr(model, "time_coupling", "joint")
    if saved_coupling != config_coupling:
        raise RuntimeError(f"Checkpoint was trained with diffusion.time_coupling='{saved_coupling}' but --config says "
                           f"'{config_coupling}'. Use the config the run was trained with.")
    print(f"Time coupling: {config_coupling}")
    mine_c = model._conditioning_state()
    saved_c = ckpt.get("conditioning", {"start": False, "mode": "input", "dropout_on": False, "offset_scale": mine_c["offset_scale"]})
    if saved_c["start"] != mine_c["start"] or (mine_c["start"] and (
            saved_c["mode"] != mine_c["mode"] or saved_c["dropout_on"] != mine_c["dropout_on"]
            or (mine_c["mode"] == "offset" and abs(saved_c["offset_scale"] - mine_c["offset_scale"]) > 1e-9))):
        raise RuntimeError(f"Checkpoint was trained with conditioning {saved_c} but --config says {mine_c}. Use the config the run was trained with.")
    if model.cond_start:
        print(f"Conditioning on the starting positions: mode {model.start_mode}, start dropout {model.start_dropout:g}")
    use_start = bool(model.cond_start) and not args.no_start
    if args.no_start and not model.cond_start:
        print("(--no-start ignored: this flow is not conditioned on the starting positions)")
    if args.copy_train_scenes and not args.gen_metrics:
        raise RuntimeError("--copy-train-scenes belongs to --gen-metrics.")
    if args.copy_train_scenes and args.split != "val":
        raise RuntimeError("--copy-train-scenes compares with training scenes, so the real scenes must come from --split val.")
    if config_coupling == "decoupled" and not (args.structure or args.swap_test or args.pca_test or args.decoder_tolerance or args.gen_metrics or args.real_projection or args.position_origin):
        raise RuntimeError("This checkpoint uses separate times for positions and features. The default mode and --trace "
                           "assume one time for both and are not valid for it; use --structure, --swap-test, "
                           "--pca-test, --decoder-tolerance, --gen-metrics, --real-projection or --position-origin.")
    info = load_flow_weights(model, ckpt, args.weights, fm.LatentNormaliser)
    del ckpt
    model.to(device).eval()
    info["checkpoint"] = str(args.checkpoint)
    print(f"Flow checkpoint: {args.checkpoint}\n  weights={info['weights']}  epoch={info['epoch']}  "
          f"step={info['global_step']}  normaliser={info['normaliser']}")
    p_mean, p_std = float(config.diffusion.P_mean), float(config.diffusion.P_std)

    with torch.no_grad():
        # ---- real validation latents (the reference) ----
        n_val = min(args.val_scenes, len(val_set))
        positions = np.unique(np.linspace(0, len(val_set) - 1, n_val).round().astype(int))
        zs, ps, ss = [], [], []
        for s in range(0, len(positions), 8):
            batch = default_collate([val_set[int(i)] for i in positions[s:s + 8]])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            z, p, s_ = model._encode_with_start(batch)  # normalised like training
            zs.append(z); ps.append(p); ss.append(s_)
        z1, pos1 = torch.cat(zs), torch.cat(ps)
        s1 = torch.cat(ss) if model.cond_start else None   # starting positions of these real scenes (conditioned flow)
        S1 = s1 if use_start else None
        real_stats = latent_stats(z1, pos1)
        print(f"\nReal {args.split} latents ({len(z1)} scenes) in the flow's space:")
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in real_stats.items()))

        if args.trace:
            if model.cond_start:
                raise RuntimeError("--trace is not available for a flow conditioned on the starting positions.")
            out_dir = args.out_dir.expanduser().resolve() / ("trace_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            run_trace(model, z1, pos1, args.steps, args.seed, out_dir)
            print(f"\nSaved trace.json and trace.png in {out_dir}")
            return

        if args.decoder_tolerance:
            out_dir = args.out_dir.expanduser().resolve() / ("decoder_tolerance_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            run_decoder_tolerance(model, build_scene_context(model, ae, val_set, device, edict), out_dir, args.seed)
            print(f"\nSaved 3_decoder_tolerance.png and decoder_tolerance.json in {out_dir}")
            return

        if args.swap_test:
            out_dir = args.out_dir.expanduser().resolve() / ("swap_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            ctx = build_scene_context(model, ae, val_set, device, edict)
            generated = generate_samples(model, len(ctx), args.steps, args.seed, device,
                                         start=ctx_starts(ctx, len(ctx)) if use_start else None)
            run_swap_test(model, ctx, generated, z1.mean(dim=0, keepdim=True), out_dir)
            print(f"\nSaved 4_swap_test.png and swap_test.json in {out_dir}")
            return

        if args.structure:
            out_dir = args.out_dir.expanduser().resolve() / ("structure_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            run_structure(model, z1, pos1, args.steps, args.seed, args.scenes, device, out_dir, height_split=args.height_split, s1=S1)
            print(f"\nSaved structure_test.json in {out_dir}")
            return

        if args.position_origin:
            out_dir = args.out_dir.expanduser().resolve() / ("position_origin_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            try:
                from psplat.models.modules.patchify import patch_mean_rays as patch_fn
            except Exception as e:
                print(f"  (patch_mean_rays not importable, along/across split skipped: {e})")
                patch_fn = None
            run_position_origin(model, ae, val_set, device, edict, args.val_scenes, max(1, args.rigs), args.seed, out_dir, patch_fn)
            print(f"\nSaved 9_position_origin.png and position_origin.json in {out_dir}")
            return

        if args.real_projection:
            out_dir = args.out_dir.expanduser().resolve() / ("real_projection_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            ctx = build_scene_context(model, ae, val_set, device, edict, n=min(args.val_scenes, len(val_set)))
            run_real_projection(model, ctx, out_dir)
            print(f"\nSaved 7_real_projection.png and real_projection.json in {out_dir}")
            return

        if args.ladder:
            if config_coupling == "decoupled":
                raise RuntimeError("--ladder uses the joint-time sampler and cannot be run on a decoupled-time checkpoint.")
            out_dir = args.out_dir.expanduser().resolve() / ("noise_ladder_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            ctx = build_scene_context(model, ae, val_set, device, edict, n=min(args.val_scenes, len(val_set)))
            run_ladder(model, ctx, args.steps, args.seed, device, out_dir, feature_net=args.feature_net, use_start=use_start)
            print(f"\nSaved 8_noise_ladder.png and noise_ladder.json in {out_dir}")
            return

        if args.gen_metrics:
            out_dir = args.out_dir.expanduser().resolve() / ("gen_metrics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            ctx = build_scene_context(model, ae, val_set, device, edict, n=min(args.val_scenes, len(val_set)))
            train_ref = None
            if args.copy_train_scenes:
                train_ds = train_loader.dataset
                if isinstance(train_ds, fm.RepeatedDataset):
                    train_ds = train_ds.dataset
                if not hasattr(train_ds, "scene_batch"):
                    raise RuntimeError("Training set is a subset wrapper; run with dataset_fraction: 1.0 to use --copy-train-scenes.")
                tctx = build_scene_context(model, ae, train_ds, device, edict, n=min(args.copy_train_scenes, len(train_ds)))
                train_ref = torch.cat([c["recon"] for c in tctx])
                print(f"Copy check reference: reconstructions of {len(tctx)} training scenes ({len(train_ref)} views).")
            run_gen_metrics(model, ctx, args.scenes, args.steps, args.seed, device, out_dir,
                            feature_net="lowlevel" if args.feature_net == "lowlevel" else args.feature_net,
                            use_start=use_start, train_ref=train_ref)
            print(f"\nSaved 6_gen_metrics_scenes.png and gen_metrics.json in {out_dir}")
            return

        if args.pca_test:
            out_dir = args.out_dir.expanduser().resolve() / ("pca_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            ctx = build_scene_context(model, ae, val_set, device, edict)
            generated = generate_samples(model, len(ctx), args.steps, args.seed, device,
                                         start=ctx_starts(ctx, len(ctx)) if use_start else None)
            run_pca_test(model, ctx, generated, z1, out_dir)
            print(f"\nSaved 5_feature_subspace.png and feature_subspace.json in {out_dir}")
            return

        # ---- loss by noise level ----
        sigmas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        binned = loss_by_noise_level(model, z1, pos1, sigmas, p_mean, p_std, start=S1)
        print("\nFlow loss by noise level on real latents (ratio < 1 = better than a no-structure estimator):")
        print("  sigma | share of training at least this noisy | loss feat / pos | ratio feat / pos")
        for r in binned:
            print(f"  {r['sigma']:5.2f} | {100 * r['train_share_at_least_this_noisy']:9.3f}% "
                  f"| {r['loss_feat']:.3f} / {r['loss_pos']:.3f} | {r['feat_vs_baseline']:.2f} / {r['pos_vs_baseline']:.2f}")

        # ---- rendering context for 4 real scenes ----
        ctx = build_scene_context(model, ae, val_set, device, edict)
        n_render = len(ctx)

        out_dir = args.out_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir.mkdir(parents=True, exist_ok=False)

        # ---- generation with different samplers, same starting noise ----
        gen = torch.Generator(device=device).manual_seed(args.seed)
        n_gen = args.scenes
        base_x = torch.randn(n_gen, model.n_tokens, model.scalar_dim, device=device, generator=gen)
        base_p = torch.randn(n_gen, model.n_tokens, 3, device=device, generator=gen)
        sampler_rows, sampler_results = [("real: autoencoder reconstruction (reference)", [c["recon"] for c in ctx[:2]])], []
        print(f"\nGeneration with {n_gen} scenes per sampler (same starting noise):")
        print("  sampler | evals | feat std | feat |x|>5 | pos rms | clamp active")
        print(f"  {'real latents (reference)':52s} |       | {real_stats['feat_std']:.3f} | {100 * real_stats['frac_abs_gt_5']:.2f}% | {real_stats['pos_rms']:.3f} |")
        cond_gen = model.as_cond(ctx_starts(ctx, n_gen) if use_start else None, n_gen, device)
        for name, sigmas_, heun, clamp, clip in sampler_variants(args.steps):
            s0 = start_scale(sigmas_[0])
            zg, pg, run = integrate(model, base_x * s0, base_p * s0, sigmas_, heun=heun, clamp=clamp, clip_x1=clip, start=cond_gen)
            pg = model._state_to_pos(pg, cond_gen)
            stats = latent_stats(zg, pg)
            sampler_results.append({"name": name, **run, **stats})
            print(f"  {name:52s} | {run['network_evaluations']:5d} | {stats['feat_std']:.3f} | "
                  f"{100 * stats['frac_abs_gt_5']:.2f}% | {stats['pos_rms']:.3f} | {100 * run['clamp_fraction']:.2f}%")
            imgs = [model.render(zg[i:i + 1], pg[i:i + 1], ctx[i]["ray_o"], ctx[i]["ray_d"], IMAGE_HW)[0] for i in range(min(2, n_render))]
            sampler_rows.append((f"{name} ({run['network_evaluations']} evals)", imgs))
        save_grid(out_dir / "1_samplers.png", sampler_rows)

        # ---- denoise real latents from partial noise ----
        z_real = torch.cat([c["z"] for c in ctx]); p_real = torch.cat([c["pos"] for c in ctx])
        denoise_rows, denoise_results = [("real: autoencoder reconstruction of the clean latent", [c["recon"] for c in ctx[:2]])], []
        print("\nDenoising real latents from partial noise (PSNR of the render vs the autoencoder reconstruction):")
        print("  start t (sigma) | latent MSE feat / pos | PSNR")
        for t0 in (0.9, 0.7, 0.5, 0.3, 0.1):
            zd, pd = denoise_from(model, z_real, p_real, t0, args.steps, seed=args.seed,
                                  start=torch.cat([c["start"] for c in ctx]) if use_start else None)
            mse_f = (zd - z_real).square().mean().item(); mse_p = (pd - p_real).square().mean().item()
            imgs = [model.render(zd[i:i + 1], pd[i:i + 1], ctx[i]["ray_o"], ctx[i]["ray_d"], IMAGE_HW)[0] for i in range(n_render)]
            score = float(np.mean([psnr(imgs[i], ctx[i]["recon"]) for i in range(n_render)]))
            denoise_results.append({"start_t": t0, "start_sigma": (1 - t0) / t0, "latent_mse_feat": mse_f,
                                    "latent_mse_pos": mse_p, "psnr_vs_ae_reconstruction": score})
            print(f"  {t0:.1f} ({(1 - t0) / t0:5.2f})   | {mse_f:.4f} / {mse_p:.4f}      | {score:.2f} dB")
            denoise_rows.append((f"start t={t0} (sigma {(1 - t0) / t0:.2f})", imgs[:2]))
        save_grid(out_dir / "2_denoise_real_latents.png", denoise_rows)

    summary = {"checkpoint_info": info, "config": str(args.config), "split": args.split, "seed": args.seed, "real_reference": real_stats,
               "loss_by_noise_level": binned, "samplers": sampler_results, "denoise_real": denoise_results}
    (out_dir / "sampler_diagnostics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"\nSaved 1_samplers.png, 2_denoise_real_latents.png and sampler_diagnostics.json in {out_dir}")


if __name__ == "__main__":
    main()
