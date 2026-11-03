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
against real; use --scenes 128) to run only that part.
Add --split train to use training scenes.
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
    p.add_argument("--gen-metrics", action="store_true",
                   help="Only compute distribution metrics on rendered views, generated against real (use --scenes 128 and "
                        "--val-scenes 64; run every checkpoint with the same values for a fair comparison).")
    p.add_argument("--feature-net", choices=("auto", "dinov2", "lowlevel"), default="auto",
                   help="Feature network for --gen-metrics; 'lowlevel' skips the download and reports the crude descriptor only.")
    p.add_argument("--trace", action="store_true",
                   help="Only trace where the flow leaves the data distribution along sampling paths (fast).")
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
def loss_by_noise_level(model, z1, pos1, sigmas, p_mean, p_std, draws=4, chunk=16, seed=0):
    """Flow loss on real latents at fixed noise levels, split into features and positions."""
    gen = torch.Generator(device=z1.device).manual_seed(seed)
    rows = []
    for sigma in sigmas:
        t = 1.0 / (1.0 + sigma)
        acc = {"feat_w": 0.0, "pos_w": 0.0, "feat_u": 0.0, "pos_u": 0.0}
        n = 0
        for _ in range(draws):
            for s in range(0, z1.shape[0], chunk):
                x1, p1 = z1[s:s + chunk], pos1[s:s + chunk]
                tb = torch.full((x1.shape[0],), t, device=z1.device)
                t_ = tb[:, None, None]
                x0 = torch.randn(x1.shape, device=x1.device, generator=gen)
                p0 = torch.randn(p1.shape, device=x1.device, generator=gen)
                vx, vp = model._velocity((1 - t_) * x0 + t_ * x1, (1 - t_) * p0 + t_ * p1, tb)
                ef = (vx - (x1 - x0)).square(); ep = (vp - (p1 - p0)).square()
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
def integrate(model, x, pos, sigmas, heun=False, clamp=None, on_step=None, clip_x1=None):
    """Integrate the learned velocity field from the first noise level in `sigmas` to 0 (t = 1).

    Euler, or Heun (second order, no second evaluation on the last step, which would be at t = 1).
    `clamp` limits |velocity| like the validation sampler does (None = off). Returns x, pos, info.
    `on_step((i, sigma), x, pos, vx, vp)` is called at the start of every step with the raw velocity.
    `clip_x1` limits the model's clean-feature estimate x1_hat = x + (1-t) v to [-c, c] before stepping
    (the velocity is rebuilt from the clipped estimate; positions are not clipped).
    """
    ts = torch.tensor(1.0 / (1.0 + np.asarray(sigmas, dtype=np.float64)), dtype=torch.float32, device=x.device)
    b = x.shape[0]
    n_eval, clamped = 0, []

    def vel(xx, pp, tt, record=None):
        nonlocal n_eval
        vx, vp = model._velocity(xx, pp, tt.expand(b))
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
def denoise_from(model, z1, pos1, t0, n_steps, seed=0, on_step=None):
    """Noise real latents to time t0 (x_t = (1-t0) eps + t0 x1) and integrate back to the data."""
    gen = torch.Generator(device=z1.device).manual_seed(seed)
    x = (1 - t0) * torch.randn(z1.shape, device=z1.device, generator=gen) + t0 * z1
    p = (1 - t0) * torch.randn(pos1.shape, device=z1.device, generator=gen) + t0 * pos1
    sigmas = sigma_grid("karras", max(n_steps // 2, 1), sigma_max=(1 - t0) / t0)
    xd, pd, _ = integrate(model, x, p, sigmas, heun=True, on_step=on_step)
    return xd, pd


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
        z, p, _ = ae.encode(enc)
        if model.normaliser is not None:
            p, z = model.normaliser.normalise(p, z)
        ctx.append({"ray_o": ray_o, "ray_d": ray_d, "z": z, "pos": p,
                    "recon": model.render(z, p, ray_o, ray_d, IMAGE_HW)[0]})
    return ctx


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
def generate_samples(model, n, steps, seed, device):
    """Two generated batches from the same starting noise: the current sampler and the clipped Karras one."""
    gen = torch.Generator(device=device).manual_seed(seed)
    base_x = torch.randn(n, model.n_tokens, model.scalar_dim, device=device, generator=gen)
    base_p = torch.randn(n, model.n_tokens, 3, device=device, generator=gen)
    if getattr(model, "time_coupling", "joint") == "decoupled":
        # the model's own sampler: positions run ahead of features (one sampler only; the other variants
        # assume one time for positions and features)
        zg, pg = model._integrate_decoupled(base_x, base_p, steps)
        return [("current: decoupled times, positions first, Euler, clamp 10", zg, pg)]
    variants = sampler_variants(steps)
    out = []
    for name, sigmas_, heun, clamp, clip in (variants[0], variants[-1]):
        s0 = start_scale(sigmas_[0])
        zg, pg, _ = integrate(model, base_x * s0, base_p * s0, sigmas_, heun=heun, clamp=clamp, clip_x1=clip)
        out.append((name, zg, pg))
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


def run_structure(model, z1, pos1, steps, seed, n_gen, device, out_dir):
    slot_mean = z1.mean(dim=0, keepdim=True)
    edges = distance_edges(pos1)
    rows = [("real validation latents", structure_metrics(z1, pos1, slot_mean, edges))]
    gens = {}
    for c in range(0, n_gen, 16):
        out = generate_samples(model, min(16, n_gen - c), steps, seed + c, device)
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
def run_gen_metrics(model, ctx, n_gen, steps, seed, device, out_dir, feature_net="auto"):
    """Distribution metrics on RENDERED views: generated scenes against the autoencoder's reconstructions of real scenes.

    Reference images are the autoencoder reconstructions of the real scenes in `ctx` (same decoder as the generated
    ones, so decoder blur cancels; the numbers measure the gap in the latent distribution, not the decoder).
    Generated scene i is rendered with the cameras of real scene i (modulo the number of real scenes).
    """
    n_real = len(ctx)
    ref = torch.cat([c["recon"] for c in ctx])                  # [n_real * V, 3, H, W]
    n_views = ctx[0]["recon"].shape[0]
    zs, ps, name = [], [], None
    for c in range(0, n_gen, 16):
        out = generate_samples(model, min(16, n_gen - c), steps, seed + c, device)
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
    deep_name, deep = load_deep_features(device, feature_net)
    if deep_name is None:
        print("  deep features not used" + (" (" + "; ".join(deep) + ")" if deep else " (--feature-net lowlevel)") + "; only the low-level metrics are reported.")
        results["deep_feature_error"] = deep
    else:
        fr = _extract(deep, ref); fg = _extract(deep, gen)
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


# ----------------------------------------------------------------------------- main
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
    if config_coupling == "decoupled" and not (args.structure or args.swap_test or args.pca_test or args.decoder_tolerance or args.gen_metrics):
        raise RuntimeError("This checkpoint uses separate times for positions and features. The default mode and --trace "
                           "assume one time for both and are not valid for it; use --structure, --swap-test, "
                           "--pca-test, --decoder-tolerance or --gen-metrics.")
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
        zs, ps = [], []
        for s in range(0, len(positions), 8):
            batch = default_collate([val_set[int(i)] for i in positions[s:s + 8]])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            z, p = model._encode(batch)                # normalised like training
            zs.append(z); ps.append(p)
        z1, pos1 = torch.cat(zs), torch.cat(ps)
        real_stats = latent_stats(z1, pos1)
        print(f"\nReal {args.split} latents ({len(z1)} scenes) in the flow's space:")
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in real_stats.items()))

        if args.trace:
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
            generated = generate_samples(model, len(ctx), args.steps, args.seed, device)
            run_swap_test(model, ctx, generated, z1.mean(dim=0, keepdim=True), out_dir)
            print(f"\nSaved 4_swap_test.png and swap_test.json in {out_dir}")
            return

        if args.structure:
            out_dir = args.out_dir.expanduser().resolve() / ("structure_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            run_structure(model, z1, pos1, args.steps, args.seed, args.scenes, device, out_dir)
            print(f"\nSaved structure_test.json in {out_dir}")
            return

        if args.gen_metrics:
            out_dir = args.out_dir.expanduser().resolve() / ("gen_metrics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            ctx = build_scene_context(model, ae, val_set, device, edict, n=min(args.val_scenes, len(val_set)))
            run_gen_metrics(model, ctx, args.scenes, args.steps, args.seed, device, out_dir,
                            feature_net="lowlevel" if args.feature_net == "lowlevel" else args.feature_net)
            print(f"\nSaved 6_gen_metrics_scenes.png and gen_metrics.json in {out_dir}")
            return

        if args.pca_test:
            out_dir = args.out_dir.expanduser().resolve() / ("pca_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            out_dir.mkdir(parents=True, exist_ok=False)
            ctx = build_scene_context(model, ae, val_set, device, edict)
            generated = generate_samples(model, len(ctx), args.steps, args.seed, device)
            run_pca_test(model, ctx, generated, z1, out_dir)
            print(f"\nSaved 5_feature_subspace.png and feature_subspace.json in {out_dir}")
            return

        # ---- loss by noise level ----
        sigmas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        binned = loss_by_noise_level(model, z1, pos1, sigmas, p_mean, p_std)
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
        for name, sigmas_, heun, clamp, clip in sampler_variants(args.steps):
            s0 = start_scale(sigmas_[0])
            zg, pg, run = integrate(model, base_x * s0, base_p * s0, sigmas_, heun=heun, clamp=clamp, clip_x1=clip)
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
            zd, pd = denoise_from(model, z_real, p_real, t0, args.steps, seed=args.seed)
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
