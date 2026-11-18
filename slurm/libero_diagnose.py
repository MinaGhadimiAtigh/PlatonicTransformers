#!/usr/bin/env python3
"""LIBERO flow diagnoser: is the flow learning, and how good are its scenes? No training. Needs one GPU (a short job).

Run it through Slurm, never on the login node:
    sbatch /home/mghadim/PlatonicTransformers/slurm/libero_diagnose.sbatch <config> <checkpoint> <label>

It loads a Lightning checkpoint (EMA weights, as validation uses), builds the data like main_nvs_fm_libero.py, and runs
up to four parts (--parts, default learn,rank,gen; add ladder or run it alone with --parts ladder):

  learn  Does the flow learn anything? On real validation latents it noises them to fixed noise levels, lets the flow
         denoise them and compares the error with simple reference estimators:
           - nothing learned (predict the mean)           - take the noisy input as it is (identity)
           - per-channel shrinkage (knows each channel's variance only)
           - Gaussian with the full feature covariance (knows all second-order structure, every token on its own)
         A flow that has learned structure beyond second-order statistics comes out below the Gaussian reference; where it
         is above, it has not even learned the covariance. Also: a shuffle control (error against ANOTHER scene's clean
         latent), the share of the error that lies outside the top-K principal directions (a width bottleneck shows up
         there), and the velocity-error ratios that W&B logs as val/ratio_*, for cross-checking.
  rank   Do the weak feature directions matter for rendering? Real latents are projected onto their top-K principal
         directions, decoded, and compared (PSNR) with the unprojected decode.
  gen    Generated scenes against real ones. Latents: feature std, variance along the real principal directions, token
         spacing and how alike the features of neighbouring tokens are (real against generated), at 50 and at --steps
         sampler steps. Renders: DINOv2 (or a low-level descriptor) KID of generated views against autoencoder
         reconstructions of real scenes, the real-against-real floor, the autoencoder-against-ground-truth ceiling, and
         multi-view consistency. A picture of ground truth, reconstruction and generated scenes from different tasks.

  ladder Where along the noise range is realness lost? Real validation latents are noised to sigma 0.11, 0.43, 1.0, 2.33 and 9.0
         (start times t = 0.9, 0.7, 0.5, 0.3, 0.1, the same ladder as the gravity runs), denoised by the flow with the usual Euler
         sampler from that point on, decoded, and compared with the autoencoder's own decode of the clean latent (PSNR, DINOv2 KID).
         Four variants: the flow's whole output; the flow's positions with the real features; the real positions with the flow's
         features; and, as a no-denoising reference, the noisy input as it is. "Real positions + flow features" at low sigma
         isolates what the low-noise feature floor costs in rendered pictures.

Everything is logged to W&B (project Platonic-NVS-FM-diag: numbers, one table, one picture; no weights, no checkpoints) and
written to <out-dir>/diagnose.json. --no-wandb turns W&B off.
Run from the PlatonicTransformers folder with PYTHONPATH set to PlatonicTransformers and PlatonicSplats (the sbatch does it).
"""
import argparse
import importlib.util
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HW = (128, 128)
SIGMAS = (0.03, 0.1, 0.2, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0)
RANKS = (32, 64, 128, 192, 256, 384)
LADDER_T0 = (0.9, 0.7, 0.5, 0.3, 0.1)       # start times; sigma = (1 - t) / t = 0.11, 0.43, 1.0, 2.33, 9.0
LADDER_VARIANTS = (("full", "flow output (positions and features)"),
                   ("flow_pos_real_feat", "flow positions + real features"),
                   ("real_pos_flow_feat", "real positions + flow features"),
                   ("noisy_input", "noisy input, not denoised"))


class AttrDict(dict):
    """dict with attribute access (what the autoencoder's encode() expects of its input)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="Lightning checkpoint, e.g. .../checkpoints/snapshot-epoch0499-step64000.ckpt")
    p.add_argument("--config", type=Path, required=True, help="the flow YAML the run was trained with")
    p.add_argument("--label", type=str, default=None, help="name of this diagnosis (default: run id and checkpoint name)")
    p.add_argument("--fm-script", type=Path, default=Path("mains/main_nvs_fm_libero.py"))
    p.add_argument("--weights", choices=("ema", "online"), default="ema")
    p.add_argument("--parts", type=str, default="learn,rank,gen", help="comma list of: learn, rank, gen, ladder")
    p.add_argument("--n-real", type=int, default=160, help="real validation scenes (one per episode) for the references")
    p.add_argument("--n-gen", type=int, default=96, help="generated scenes")
    p.add_argument("--steps", type=int, default=200, help="sampler steps of the main generation (training validation uses 50)")
    p.add_argument("--compare-steps", type=int, default=50, help="second generation with this many steps (latent statistics only)")
    p.add_argument("--n-ladder", type=int, default=48, help="scenes for the noise ladder (spread over the validation episodes)")
    p.add_argument("--ladder-steps", type=int, default=100, help="Euler steps the ladder would use from noise to data (fewer from a later start)")
    p.add_argument("--draws", type=int, default=3, help="noise draws per noise level in the learning test")
    p.add_argument("--n-rank", type=int, default=16, help="scenes for the rank/render test")
    p.add_argument("--feature-net", choices=("auto", "dinov2", "lowlevel"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--out-dir", type=Path, default=Path("diagnose_out"))
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="Platonic-NVS-FM-diag")
    return p.parse_args()


def finite(o):
    """JSON-safe copy: numpy and torch scalars to python, NaN and inf to None."""
    if isinstance(o, dict):
        return {str(k): finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if torch.is_tensor(o):
        o = o.item() if o.numel() == 1 else o.tolist()
        return finite(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


# ----------------------------------------------------------------------------- loading
def load_flow_weights(model, ckpt, weights):
    """Flow-network weights of the checkpoint (EMA or online) into model.net. The normaliser and the mode checks are done
    by model.on_load_checkpoint (called separately), the same code a real resume would use."""
    def clean(k):  # torch.compile prefixes parameter names with "_orig_mod."
        return k[len("net."):].replace("_orig_mod.", "")

    net_sd = {clean(k): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
    missing, unexpected = model.net.load_state_dict(net_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match this YAML's network. missing={missing[:3]} unexpected={unexpected[:3]}. "
                           f"Use the YAML the run was trained with.")
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
        info = {"weights": "ema", "ema_decay": callbacks[key].get("decay"), "ema_warmup_steps": callbacks[key].get("warmup_steps")}
    info["global_step"] = int(ckpt.get("global_step", -1))
    info["epoch"] = int(ckpt.get("epoch", -1))
    return info


def load_everything(args):
    from platonic_transformers.utils.config_loader import load_with_defaults
    from psplat.models.platonic_autoencoder import PlatonicAutoencoder

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    fm_path = args.fm_script.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("nvs_fm_module", fm_path)
    fm = importlib.util.module_from_spec(spec)
    sys.modules["nvs_fm_module"] = fm
    spec.loader.exec_module(fm)

    config = load_with_defaults(dataset_config=str(args.config), cli_args=[])
    config.model.compile = False                      # plain network: no recompiling, same maths
    ae_dir = Path(config.autoencoder.checkpoint_dir)
    ae_config = load_with_defaults(dataset_config=str(ae_dir / "config.yaml"), cli_args=[])
    ae = PlatonicAutoencoder(ae_config)
    ae.load_ckpt(str(ae_dir))                         # exactly what the training script does
    ae.requires_grad_(False); ae.eval()
    pts = sorted(p.name for p in ae_dir.glob("*.pt"))
    print(f"Autoencoder folder {ae_dir}: checkpoints {pts}; load_ckpt picks the last in this order: {pts[-1] if pts else '?'}")

    train_loader, val_loader = fm.load_data(config, ae_config)
    val_set = val_loader.dataset
    print(f"Validation set: {len(val_set)} items")

    model = fm.NVSFMModel(config, ae)
    ckpt = torch.load(args.checkpoint.expanduser(), map_location="cpu", weights_only=False)
    model.on_load_checkpoint(ckpt)                    # checks the modes and restores the latent normaliser
    info = load_flow_weights(model, ckpt, args.weights)
    del ckpt
    model.set_val_dataset(val_set)
    model.to(device).eval()
    info["checkpoint"] = str(args.checkpoint)
    info["device"] = device
    info["conditioning"] = model._conditioning_state()
    print(f"Flow checkpoint {args.checkpoint}\n  weights={info['weights']} epoch={info['epoch']} step={info['global_step']} "
          f"normaliser={'yes' if model.normaliser is not None else 'NO'} start-conditioned={model.cond_start}")
    return fm, config, ae, model, val_set, device, info


# ----------------------------------------------------------------------------- items and contexts
def pick_items(ds, n):
    """Dataset indices of n validation scenes from n different episodes (consecutive indices are frames of one episode)."""
    stride = int(getattr(ds, "frames_per_episode", 0) or 0) or 1
    n_ep = max(len(ds) // stride, 1)
    eps = np.unique(np.linspace(0, n_ep - 1, min(n, n_ep)).round().astype(int))
    return [int((e * stride) % len(ds)) for e in eps]


def suite_of(label):
    l = str(label).lower()
    for key in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "libero_long"):
        if key in l:
            return key
    for key in ("spatial", "object", "goal", "long", "study"):
        if key in l:
            return key
    return "other"


@torch.no_grad()
def encode_items(model, ds, idxs, device, bs=8):
    """Clean (normalised) latents of validation items, exactly as the validation loss sees them: z1, pos1, start s1 (or None)."""
    from torch.utils.data import default_collate
    zs, ps, ss, labels = [], [], [], []
    for s in range(0, len(idxs), bs):
        items = [ds[int(i)] for i in idxs[s:s + bs]]
        labels += [str(it.get("scene_name", "")) if isinstance(it, dict) else "" for it in items]
        batch = default_collate(items)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        z, p, st = model._encode_with_start(batch)
        zs.append(z); ps.append(p); ss.append(st)
    z1, p1 = torch.cat(zs), torch.cat(ps)
    s1 = torch.cat(ss) if model.cond_start else None
    return z1, p1, s1, labels


@torch.no_grad()
def build_ctx(fm, model, ds, idxs, device):
    """Per scene: rays of all its cameras, the clean latent from the encoder cameras, starting positions, the
    autoencoder reconstruction and the ground-truth images of all cameras."""
    ae = model.autoencoder
    ctx = []
    for d_idx in idxs:
        v_idxs, n_in = model._val_views(d_idx, [0, 1, 2, 3, 4, 5, 6])
        sd = fm._scene_batch(ds, d_idx, v_idxs, device)
        ray_o, ray_d = ae.process_data.compute_rays(sd["c2w"], sd["fxfycxcy"], h=HW[0], w=HW[1], device=device)
        enc = AttrDict(image=sd["image"][:, :n_in], ray_o=ray_o[:, :n_in], ray_d=ray_d[:, :n_in], image_h_w=HW)
        z, p, s0 = ae.encode(enc)
        if model.normaliser is not None:
            p, z = model.normaliser.normalise(p, z)
        recon = model.render(z, p, ray_o, ray_d, HW)[0].float().clamp(0, 1)
        ctx.append({"d_idx": int(d_idx), "n_in": int(n_in), "ray_o": ray_o, "ray_d": ray_d, "z": z, "pos": p,
                    "start": model.norm_start(s0), "recon": recon, "gt": sd["image"][0].float().clamp(0, 1)})
    return ctx


# ----------------------------------------------------------------------------- real-latent references
def feature_spectrum(z1):
    """Eigenvalues (descending) and directions of the second-moment matrix of the real feature tokens (about 0, like the
    flow's own PCA monitor). Columns of V are the directions."""
    f = z1.reshape(-1, z1.shape[-1]).double()
    C = (f.T @ f) / f.shape[0]
    ev, V = torch.linalg.eigh(C)
    order = ev.argsort(descending=True)
    return ev[order].clamp_min(1e-12).float(), V[:, order].float(), C.float()


def gauss_oos(CA, CB, s):
    """Cross-fitted version of gauss_mse: the estimator is built from the second-moment matrix CA of one half of the scenes
    and applied to data with second-moment matrix CB (the other half); no optimism from fitting and testing on the same
    latents. W = CA (CA + s^2 I)^-1; error per coordinate = [tr((I-W) CB (I-W)^T) + s^2 tr(W W^T)] / F."""
    F_ = CA.shape[0]
    I = torch.eye(F_, dtype=CA.dtype, device=CA.device)
    W = torch.linalg.solve((CA + s * s * I).T, CA.T).T          # CA (CA + s^2 I)^-1
    M = I - W
    return float((torch.trace(M @ CB @ M.T) + s * s * torch.trace(W @ W.T)) / F_)


def gauss_mse(ev, s):
    """Error (per coordinate) of the best estimator that knows only the covariance eigenvalues ev, at noise level s
    (observation y = x1 + s * noise): mean_i ev_i s^2 / (ev_i + s^2)."""
    return float((ev * s * s / (ev + s * s)).mean())


@torch.no_grad()
def learning_test(model, z1, p1, s1, sigmas, draws, seed, device, ev, V, chunk=16):
    """Denoising error of the flow on real latents at fixed noise levels, against reference estimators. All errors are
    per coordinate, on the clean estimate x_hat = x_t + (1 - t) v (so they are comparable across noise levels)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    S, F_ = z1.shape[0], z1.shape[-1]
    cond_all = model.as_cond(s1, S, device) if model.cond_start else None
    offset = bool(model.cond_start and model.start_mode == "offset")
    # position references (flow space, input mode only: in offset mode the state is not the position)
    pf = p1.reshape(-1, 3).double()
    mu_p = pf.mean(0)
    cov_p = torch.cov((pf - mu_p).T)
    ev_p = torch.linalg.eigvalsh(cov_p).clamp_min(1e-12).float()
    ev_r = None
    if model.cond_start and not offset and s1 is not None:
        X = torch.cat([s1.reshape(-1, 3).double(), torch.ones(pf.shape[0], 1, dtype=torch.float64, device=pf.device)], 1)
        sol = torch.linalg.lstsq(X, pf).solution
        res = pf - X @ sol
        ev_r = torch.linalg.eigvalsh(torch.cov(res.T)).clamp_min(1e-12).float()
    fa, fb = z1[0::2].double().reshape(-1, F_), z1[1::2].double().reshape(-1, F_)
    CA, CB = (fa.T @ fa) / fa.shape[0], (fb.T @ fb) / fb.shape[0]
    second = float((z1.double().reshape(-1, F_) ** 2).mean())
    ch_m2 = (z1.double().reshape(-1, F_) ** 2).mean(0).float()
    rows = []
    for sigma in sigmas:
        t_val = 1.0 / (1.0 + sigma)
        acc = {k: 0.0 for k in ("ef", "ep", "ef_shuf", "ep_shuf", "vf", "vp")}
        e_pc = torch.zeros(F_, dtype=torch.float64, device=device)
        n = 0
        for d in range(draws):
            for s in range(0, S, chunk):
                x1, pos1 = z1[s:s + chunk], p1[s:s + chunk]
                cond = None if cond_all is None else cond_all[s:s + chunk]
                pst = model._pos_to_state(pos1, cond)
                _, psc, _ = model._start_parts(cond, x1)
                psc = 1.0 if psc is None else psc
                B = x1.shape[0]
                tb = torch.full((B,), t_val, device=device)
                t_ = tb[:, None, None]
                x0 = torch.randn(x1.shape, device=device, generator=gen)
                p0 = torch.randn(pst.shape, device=device, generator=gen)
                x_t, p_t = (1 - t_) * x0 + t_ * x1, (1 - t_) * p0 + t_ * pst
                vx, vp = model._velocity(x_t, p_t, tb, start=cond)
                xh, ph = x_t + (1 - t_) * vx, p_t + (1 - t_) * vp
                e = xh - x1
                ep = psc * (ph - pst)
                k = B
                acc["ef"] += e.square().mean().item() * k
                acc["ep"] += ep.square().mean().item() * k
                # shuffle control: the same estimate against the clean latent of ANOTHER scene
                acc["ef_shuf"] += (xh - x1.roll(1, 0)).square().mean().item() * k
                acc["ep_shuf"] += (psc * (ph - pst.roll(1, 0))).square().mean().item() * k
                base = float(model._no_structure_baseline(torch.tensor(t_val)))
                acc["vf"] += ((vx - (x1 - x0)).square().mean().item() / base) * k
                acc["vp"] += ((psc * (vp - (pst - p0))).square().mean().item() / base) * k
                e_pc += ((e.reshape(-1, F_).double() @ V.double().to(device)) ** 2).sum(0)
                n += k
        e_pc = (e_pc / (n * z1.shape[1])).float().cpu()           # error energy per principal direction (per token)
        tot = float(e_pc.sum()) / F_
        row = {
            "sigma": sigma, "train_share_at_least_this_noisy": None,
            "model_mse_feat": acc["ef"] / n, "model_mse_pos": acc["ep"] / n,
            "ref_prior_mean_feat": second, "ref_identity": sigma ** 2,
            "ref_per_channel_feat": float((ch_m2 * sigma ** 2 / (ch_m2 + sigma ** 2)).mean()),
            "ref_gaussian_feat": 0.5 * (gauss_oos(CA, CB, sigma) + gauss_oos(CB, CA, sigma)),
            "ref_gaussian_feat_insample": gauss_mse(ev.cpu(), sigma),
            "ref_gaussian_pos": gauss_mse(ev_p.cpu(), sigma),
            "ref_gaussian_pos_given_start": None if ev_r is None else gauss_mse(ev_r.cpu(), sigma),
            "ref_prior_pos": float(cov_p.trace() / 3),
            "shuffled_mse_feat": acc["ef_shuf"] / n, "shuffled_mse_pos": acc["ep_shuf"] / n,
            "velocity_ratio_feat": acc["vf"] / n, "velocity_ratio_pos": acc["vp"] / n,
            "model_over_gaussian_feat": (acc["ef"] / n) / max(0.5 * (gauss_oos(CA, CB, sigma) + gauss_oos(CB, CA, sigma)), 1e-12),
            "model_over_gaussian_pos": (acc["ep"] / n) / max(gauss_mse(ev_p.cpu(), sigma), 1e-12),
            "feat_error_outside_topK": {str(K): float(e_pc[K:].sum()) / F_ for K in (64, 128, 192, 256) if K < F_},
            "feat_error_total_check": tot,
        }
        rows.append(row)
    ref = {"feature_variance_beyond_topK": {str(K): float(ev[K:].sum() / ev.sum()) for K in RANKS if K < F_},
           "feature_variance_in_top": {str(K): float(ev[:K].sum() / ev.sum()) for K in (16, 64, 128, 256) if K < F_},
           "n_tokens_used": int(z1.shape[0] * z1.shape[1]), "position_cov_eigenvalues": ev_p.tolist(),
           "position_given_start_residual_eigenvalues": None if ev_r is None else ev_r.tolist(),
           "offset_mode_position_references_skipped": offset}
    return rows, ref


# ----------------------------------------------------------------------------- rank test (rendering)
def psnr(a, b):
    """Mean PSNR over views of two [V,3,H,W] stacks (clipped to [0,1])."""
    a = torch.nan_to_num(a.float(), nan=0.0).clamp(0, 1); b = torch.nan_to_num(b.float(), nan=0.0).clamp(0, 1)
    mse = (a - b).square().mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return float((-10 * torch.log10(mse)).mean())


@torch.no_grad()
def rank_render_test(model, ctx, V, ks=RANKS):
    """Real latents projected onto their top-K principal feature directions, decoded, against the full decode."""
    out = {}
    V = V.to(ctx[0]["z"].device)
    for K in ks:
        if K > V.shape[1]:
            continue
        vals = []
        for c in ctx:
            z = c["z"]
            zp = (z @ V[:, :K]) @ V[:, :K].T
            img = model.render(zp, c["pos"], c["ray_o"], c["ray_d"], HW)[0].float().clamp(0, 1)
            vals.append(psnr(img, c["recon"]))
        out[str(K)] = float(np.mean(vals))
    return out


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, ctx, n_gen, steps, seed, device, chunk=32):
    """Generated normalised latents. Scene i uses the cameras of real scene i % len(ctx); a flow conditioned on the starting
    positions gets those cameras' starting positions."""
    model.config.diffusion.num_steps = int(steps)
    sampler = model.config.diffusion.get("sampler", "fm_ode")
    zs, ps = [], []
    for c in range(0, n_gen, chunk):
        m = min(chunk, n_gen - c)
        start = None
        if model.cond_start:
            start = torch.cat([ctx[(c + j) % len(ctx)]["start"] for j in range(m)])
        torch.manual_seed(seed + c)
        z, p = model.sample(m, sampler=sampler, start=start)
        zs.append(z.float()); ps.append(p.float())
    return torch.cat(zs), torch.cat(ps)


@torch.no_grad()
def neighbour_stats(z, p, k=4, near=None, chunk=16):
    """Token spacing and how alike the features of nearby tokens are. z [B,N,F], p [B,N,3] in the flow's space.
    Returns the mean distance to the nearest other token, the mean cosine of a token's features with its k nearest tokens'
    (channel mean removed), and the same for neighbours closer than `near` (a distance)."""
    zn_all = F.normalize(z - z.reshape(-1, z.shape[-1]).mean(0), dim=-1)
    nnd, agree, near_agree, near_cnt = [], [], 0.0, 0
    for s in range(0, z.shape[0], chunk):
        pp, zn = p[s:s + chunk], zn_all[s:s + chunk]
        B, N = pp.shape[:2]
        d = torch.cdist(pp, pp)
        d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        kd, ki = d.topk(k, largest=False)                                   # [B,N,k]
        nb = zn[torch.arange(B, device=z.device)[:, None, None], ki]       # [B,N,k,F]
        cos = (nb * zn[:, :, None]).sum(-1)                                 # [B,N,k]
        nnd.append(kd[..., 0].reshape(-1)); agree.append(cos.reshape(-1))
        if near is not None:
            m = kd < near
            near_agree += float(cos[m].sum()); near_cnt += int(m.sum())
    nnd = torch.cat(nnd); agree = torch.cat(agree)
    out = {"nearest_token_distance": float(nnd.mean()), "neighbour_feature_agreement": float(agree.mean())}
    if near is not None:
        out["agreement_for_pairs_closer_than_real_median"] = near_agree / max(near_cnt, 1)
        out["share_of_pairs_that_close"] = near_cnt / (nnd.numel() * k)
    return out, nnd


def latent_report(z, p, ev, V, real_median_nn=None):
    """Statistics of a set of latents in the flow's space."""
    f = z.reshape(-1, z.shape[-1])
    proj = (f.to(V.device) @ V).square().mean(0)
    ev = ev.to(proj.device)
    groups = {}
    for name, lo, hi in (("top16", 0, 16), ("16to64", 16, 64), ("64to128", 64, 128), ("128plus", 128, None)):
        if lo < ev.numel():
            groups[name] = float(proj[lo:hi].sum() / ev[lo:hi].sum())
    pc = p - p.mean(dim=(0, 1), keepdim=True)
    st, nnd = neighbour_stats(z, p, near=real_median_nn)
    st.update({"feat_std": float(z.std()), "feat_mean": float(z.mean()), "pos_rms": float(pc.pow(2).mean().sqrt()),
               "variance_ratio_along_real_pca": groups})
    return st, nnd


# ----------------------------------------------------------------------------- image metrics
def lowlevel_descriptor(img):
    """[N,3,H,W] in [0,1] -> [N,296] colour, saturation, brightness, gradient and coarse layout descriptor (no download)."""
    img = torch.nan_to_num(img.float(), nan=0.0).clamp(0, 1).cpu()
    mx, mn = img.max(1).values, img.min(1).values
    sat = (mx - mn) / (mx + 1e-6)
    q = (img * 3.999).long().clamp(0, 3)
    rgb_idx = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    rows = []
    for i in range(img.shape[0]):
        h_rgb = torch.bincount(rgb_idx[i].flatten(), minlength=64).float()
        h_sat = torch.histc(sat[i], bins=8, min=0, max=1)
        h_val = torch.histc(mx[i], bins=8, min=0, max=1)
        parts = [h_rgb / h_rgb.sum(), h_sat / h_sat.sum().clamp(min=1), h_val / h_val.sum().clamp(min=1)]
        for s in (1, 2):
            g = gray[i:i + 1, None]
            if s > 1:
                g = F.avg_pool2d(g, s)
            gx = g[..., :, 1:] - g[..., :, :-1]; gy = g[..., 1:, :] - g[..., :-1, :]
            mag = torch.sqrt(gx[..., :-1, :].square() + gy[..., :, :-1].square() + 1e-12)
            hh = torch.histc(torch.log10(mag + 1e-3), bins=12, min=-3, max=0.3)
            parts.append(hh / hh.sum().clamp(min=1))
        parts.append(F.adaptive_avg_pool2d(img[i:i + 1], 8).flatten())
        rows.append(torch.cat(parts))
    return torch.stack(rows)


def load_deep_features(device, which="auto"):
    """(name, fn) with fn: [N,3,H,W] in [0,1] -> [N,D] features, or (None, reasons)."""
    reasons = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def prep(x):
        x = F.interpolate(torch.nan_to_num(x.float(), nan=0.0).clamp(0, 1).to(device), size=(224, 224), mode="bilinear", align_corners=False)
        return (x - mean) / std

    if which in ("auto", "dinov2"):
        try:
            from transformers import AutoModel
            net = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

            @torch.no_grad()
            def f(x):
                h = net(pixel_values=prep(x)).last_hidden_state
                return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float().cpu()
            return "DINOv2-small (class token + mean patch token)", f
        except Exception as e:
            reasons.append(f"transformers DINOv2: {type(e).__name__}: {str(e)[:120]}")
        try:
            net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

            @torch.no_grad()
            def f(x):
                o = net.forward_features(prep(x))
                return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], dim=1).float().cpu()
            return "DINOv2-small (torch.hub)", f
        except Exception as e:
            reasons.append(f"torch.hub DINOv2: {type(e).__name__}: {str(e)[:120]}")
    return None, reasons


def extract(fn, images, chunk=64):
    return torch.cat([fn(images[i:i + chunk]) for i in range(0, len(images), chunk)])


def kid_score(a, b, subset=100, n_subsets=50, seed=0):
    """Unbiased kernel distance (polynomial kernel) averaged over random subsets: (mean, sd)."""
    a, b = a.double(), b.double()
    d, m = a.shape[1], min(subset, len(a), len(b))
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_subsets):
        x = a[torch.randperm(len(a), generator=g)[:m]]; y = b[torch.randperm(len(b), generator=g)[:m]]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        vals.append(float((kxx.sum() - kxx.diag().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1)) - 2 * kxy.mean()))
    return float(np.mean(vals)), float(np.std(vals))


def view_similarity_gap(feats, n_views):
    """Mean cosine between different views of the SAME scene minus between views of DIFFERENT scenes."""
    f = F.normalize(feats.double(), dim=1)
    sim = f @ f.T
    scene = torch.arange(len(f)) // n_views
    view = torch.arange(len(f)) % n_views
    same, diff_view = scene[:, None] == scene[None, :], view[:, None] != view[None, :]
    within, between = sim[same & diff_view].mean(), sim[~same & diff_view].mean()
    return float(within), float(between), float(within - between)


def image_metrics(sets, n_views, feature_net, device, seed):
    """sets: name -> [M,3,H,W] image stacks (n_views consecutive images per scene), with keys 'gen', 'recon', 'gt'.
    KID (x1000) of generated against reconstructions of real scenes, the real-against-real floor (even against odd scenes),
    the autoencoder ceiling (reconstructions against ground truth) and multi-view consistency."""
    gen, recon, gt = sets["gen"], sets["recon"], sets["gt"]
    n_real = len(recon) // n_views
    half = np.arange(n_real) % 2

    def idx(mask):
        return torch.tensor([i * n_views + v for i in range(n_real) if mask[i] for v in range(n_views)]).long()

    spaces, errors = [], None
    ll = lowlevel_descriptor(torch.cat([recon, gen, gt]))
    mu, sd_ = ll[:len(recon)].mean(0), ll[:len(recon)].std(0)
    sd_ = torch.maximum(sd_, sd_.median().clamp(min=1e-6))
    ll = ((ll - mu) / sd_).clamp(-10, 10)
    spaces.append(("low-level descriptor", ll[:len(recon)], ll[len(recon):len(recon) + len(gen)], ll[len(recon) + len(gen):]))
    dname, deep = load_deep_features(device, feature_net)
    if dname is None:
        errors = deep
    else:
        spaces.append((dname, extract(deep, recon), extract(deep, gen), extract(deep, gt)))
    res = {}
    for name, fr, fg, ft in spaces:
        kg, kf = kid_score(fr, fg, seed=seed), kid_score(fr[idx(half == 0)], fr[idx(half == 1)], seed=seed)
        kc = kid_score(ft, fr, seed=seed)
        kgt = kid_score(ft, fg, seed=seed)
        wr, br, gr = view_similarity_gap(fr, n_views); wg, bg, gg = view_similarity_gap(fg, n_views)
        res[name] = {"kid_x1000_generated_vs_reconstruction": [1000 * kg[0], 1000 * kg[1]],
                     "kid_x1000_reconstruction_even_vs_odd_floor": [1000 * kf[0], 1000 * kf[1]],
                     "kid_x1000_reconstruction_vs_ground_truth_ceiling": [1000 * kc[0], 1000 * kc[1]],
                     "kid_x1000_generated_vs_ground_truth": [1000 * kgt[0], 1000 * kgt[1]],
                     "view_gap_real_reconstruction": [wr, br, gr], "view_gap_generated": [wg, bg, gg]}
    return res, errors


def draw_gallery(blocks, n_cols, labels):
    """blocks: list of (title, [rows]) with rows = lists of [3,H,W] tensors. Returns an HxWx3 uint8 array."""
    from PIL import Image, ImageDraw
    H, W = HW
    lw = 150
    rows_total = sum(len(r) for _, r in blocks)
    canvas = Image.new("RGB", (lw + n_cols * W, rows_total * H + len(blocks) * 18), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    y = 0
    for title, rows in blocks:
        dr.text((4, y + 3), title, fill=(0, 0, 0)); y += 18
        for r, row in enumerate(rows):
            dr.text((4, y + 4), str(labels[r])[:22], fill=(60, 60, 60))
            for v in range(min(n_cols, len(row))):
                im = (row[v].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                canvas.paste(Image.fromarray(im), (lw + v * W, y))
            y += H
    return np.asarray(canvas)


def diverse(labels, k):
    """k positions spread over different suites/tasks (round-robin over the suites, then evenly)."""
    groups = {}
    for i, l in enumerate(labels):
        groups.setdefault(suite_of(l), []).append(i)
    out, rnd = [], 0
    while len(out) < min(k, len(labels)):
        added = False
        for g in groups.values():
            if rnd < len(g) and len(out) < k:
                out.append(g[(rnd * 7) % len(g)] if rnd == 0 else g[min(len(g) - 1, (rnd * len(g)) // 4)])
                added = True
        rnd += 1
        if not added:
            break
    out = list(dict.fromkeys(out))
    for i in np.linspace(0, len(labels) - 1, k).round().astype(int):
        if len(out) >= k:
            break
        if int(i) not in out:
            out.append(int(i))
    return out[:k]


# ----------------------------------------------------------------------------- noise ladder
@torch.no_grad()
def ladder_denoise(model, z1, p1, start_all, t0, steps, seed, device, chunk=16):
    """Noise the real latents to time t0 (sigma = (1 - t0) / t0), then run the sampler's own Euler steps (velocity clamp 10)
    from t0 to 1. Returns the flow's features and positions, and the noisy input rescaled to the data scale (x_t / t0)."""
    S = z1.shape[0]
    cond_all = model.as_cond(start_all, S, device) if model.cond_start else None
    gen = torch.Generator(device=device).manual_seed(seed)
    zf, pf, zn, pn = [], [], [], []
    ts = torch.linspace(t0, 1.0, steps + 1, device=device)
    for s in range(0, S, chunk):
        x1, pos1 = z1[s:s + chunk], p1[s:s + chunk]
        cond = None if cond_all is None else cond_all[s:s + chunk]
        pst = model._pos_to_state(pos1, cond)
        B = x1.shape[0]
        x0 = torch.randn(x1.shape, device=device, generator=gen)
        p0 = torch.randn(pst.shape, device=device, generator=gen)
        x, pos = (1 - t0) * x0 + t0 * x1, (1 - t0) * p0 + t0 * pst
        zn.append((x / t0).float()); pn.append(model._state_to_pos(pos / t0, cond).float())
        for i in range(steps):
            dt = ts[i + 1] - ts[i]
            vx, vp = model._velocity(x, pos, ts[i].expand(B), start=cond)
            vx, vp = vx.clamp(-10, 10), vp.clamp(-10, 10)
            x, pos = x + dt * vx, pos + dt * vp
        zf.append(x.float()); pf.append(model._state_to_pos(pos, cond).float())
    return torch.cat(zf), torch.cat(pf), torch.cat(zn), torch.cat(pn)


@torch.no_grad()
def ladder_test(model, ctx, steps, seed, device, feature_net, show):
    """Where along the noise range is realness lost? See the docstring at the top (part ladder). ctx: scenes built by build_ctx.
    show: positions in ctx of the scenes drawn in the picture. Returns (result dict, picture array or None)."""
    S = len(ctx)
    n_views = ctx[0]["recon"].shape[0]
    z1, p1 = torch.cat([c["z"] for c in ctx]), torch.cat([c["pos"] for c in ctx])
    start = torch.cat([c["start"] for c in ctx]) if model.cond_start else None
    recon, gt = torch.cat([c["recon"] for c in ctx]), torch.cat([c["gt"] for c in ctx])
    fname, deep = load_deep_features(device, feature_net)
    if fname is None:
        raw = lowlevel_descriptor(recon)
        mu, sd_ = raw.mean(0), raw.std(0)
        sd_ = torch.maximum(sd_, sd_.median().clamp(min=1e-6))
        fname, deep = "low-level descriptor", (lambda x: ((lowlevel_descriptor(x) - mu) / sd_).clamp(-10, 10))
    f_recon = extract(deep, recon)
    half = np.arange(S) % 2
    idx = lambda m: torch.tensor([i * n_views + v for i in range(S) if m[i] for v in range(n_views)]).long()
    floor = kid_score(f_recon[idx(half == 0)], f_recon[idx(half == 1)], seed=seed)
    ceiling = kid_score(extract(deep, gt), f_recon, seed=seed)
    ae_psnr = float(np.mean([psnr(c["recon"], c["gt"]) for c in ctx]))
    out = {"feature_net": fname, "n_scenes": S, "views_per_scene": n_views,
           "kid_x1000_real_vs_real_floor": 1000 * floor[0], "kid_x1000_autoencoder_vs_ground_truth_ceiling": 1000 * ceiling[0],
           "autoencoder_psnr_vs_ground_truth": ae_psnr, "levels": []}
    shown = {}
    for t0 in LADDER_T0:
        sigma = (1 - t0) / t0
        n_steps = max(2, int(math.ceil(steps * (1 - t0))))
        zf, pf, zn, pn = ladder_denoise(model, z1, p1, start, t0, n_steps, seed, device)
        lev = {"sigma": sigma, "t0": t0, "euler_steps": n_steps,
               "latent_error_feat_flow": float((zf - z1).square().mean()), "latent_error_pos_flow": float((pf - p1).square().mean()),
               "latent_error_feat_noisy_input": float((zn - z1).square().mean()), "latent_error_pos_noisy_input": float((pn - p1).square().mean())}
        variants = {"full": (zf, pf), "flow_pos_real_feat": (z1, pf), "real_pos_flow_feat": (zf, p1), "noisy_input": (zn, pn)}
        for name, (zz, pp) in variants.items():
            try:
                stacks = [model.render(zz[i:i + 1], pp[i:i + 1], ctx[i]["ray_o"], ctx[i]["ray_d"], HW)[0].float().clamp(0, 1) for i in range(S)]
                p_rec = np.array([psnr(stacks[i], ctx[i]["recon"]) for i in range(S)])
                p_gt = np.array([psnr(stacks[i], ctx[i]["gt"]) for i in range(S)])
                kid = kid_score(f_recon, extract(deep, torch.cat(stacks)), seed=seed)
                lev[name] = {"psnr_vs_ae_reconstruction": {"mean": float(p_rec.mean()), "median": float(np.median(p_rec)), "sd": float(p_rec.std())},
                             "psnr_vs_ground_truth_mean": float(p_gt.mean()), "kid_x1000_vs_ae_reconstruction": [1000 * kid[0], 1000 * kid[1]]}
                if name in ("full", "real_pos_flow_feat", "flow_pos_real_feat"):
                    shown[(name, t0)] = [stacks[i][0].cpu() for i in show]
            except Exception:
                lev[name] = {"error": traceback.format_exc()}
                print(f"[ladder] sigma {sigma:.2f} variant {name} FAILED:\n" + traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out["levels"].append(lev)
    pic = None
    try:
        blocks = []
        for name, title in LADDER_VARIANTS[:3]:
            rows = []
            for r, i in enumerate(show):
                row = [ctx[i]["recon"][0].cpu()]
                for t0 in LADDER_T0:
                    row.append(shown[(name, t0)][r])
                rows.append(row)
            blocks.append((f"{title}. Columns: AE reconstruction, then sigma " + ", ".join(f"{(1 - t) / t:.2f}" for t in LADDER_T0) + " (camera 1)", rows))
        pic = draw_gallery(blocks, 1 + len(LADDER_T0), [f"scene {i}" for i in show])
    except Exception:
        out["picture_error"] = traceback.format_exc()
    return out, pic


# ----------------------------------------------------------------------------- main
def loss_share(model, rows, p_mean, p_std):
    """Where the training feature loss comes from, by noise level: the weighted feature loss at the measured noise levels
    (log-sigma interpolation in between, flat beyond the ends) averaged over the training noise distribution.
    Approximate: interpolated between the measured noise levels. Returns the reconstructed mean feature loss and the shares from sigma < 0.43,
    0.43 to 2.3 and above 2.3 (the x-prediction loss floor starts at sigma 0.43)."""
    sig = np.array([r["sigma"] for r in rows])
    t = 1.0 / (1.0 + sig)
    if model.feat_prediction == "x":
        w = (t ** 2 + (1 - t) ** 2) / np.maximum(1 - t, model.xpred_loss_floor) ** 2
        c = w * np.array([r["model_mse_feat"] for r in rows])
    else:
        base = 2.0 - (2 * t - 1) ** 2 / (t ** 2 + (1 - t) ** 2)
        c = (t ** 2 + (1 - t) ** 2) * np.array([r["velocity_ratio_feat"] for r in rows]) * base
    z = np.linspace(-5, 5, 4001); wz = np.exp(-z ** 2 / 2); wz /= wz.sum()
    sg = np.exp(p_mean + p_std * z)
    cs = np.exp(np.interp(np.log(sg), np.log(sig), np.log(np.maximum(c, 1e-12))))
    tot = float((wz * cs).sum())
    out = {"reconstructed_feature_loss": tot}
    for name, lo, hi in (("below_0.43", 0.0, 0.43), ("0.43_to_2.3", 0.43, 2.3), ("above_2.3", 2.3, 1e12)):
        m = (sg >= lo) & (sg < hi)
        out[f"share_of_samples_{name}"] = float(wz[m].sum())
        out[f"share_of_loss_{name}"] = float((wz[m] * cs[m]).sum() / tot)
    return out


def tail(sigma, p_mean, p_std):
    z = (math.log(sigma) - p_mean) / p_std
    return 0.5 * math.erfc(z / math.sqrt(2))


def main():
    args = parse_args()
    t_start = time.time()
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    fm, config, ae, model, val_set, device, info = load_everything(args)
    label = args.label or f"{args.checkpoint.parent.parent.name}-{args.checkpoint.stem}"
    out_dir = args.out_dir.expanduser().resolve() / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    R = {"label": label, "info": info, "args": {k: str(v) for k, v in vars(args).items()}}
    p_mean, p_std = float(config.diffusion.P_mean), float(config.diffusion.P_std)

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, entity=config.logging.get("wandb_identity", None), name=f"diag-{label}",
                            job_type="diagnose", config={"checkpoint": str(args.checkpoint), "config": str(args.config),
                                                         "weights": info["weights"], "epoch": info["epoch"], "steps": args.steps,
                                                         "n_real": args.n_real, "n_gen": args.n_gen})
        except Exception as e:
            print(f"[warn] W&B not available ({type(e).__name__}: {e}); continuing without it.")
    summary = {}

    with torch.no_grad():
        idxs = pick_items(val_set, args.n_real)
        z1, p1, s1, labels = encode_items(model, val_set, idxs, device)
        R["items"] = {"n": len(idxs), "dataset_indices": idxs, "scene_names": labels,
                      "suites": {s: sum(1 for l in labels if suite_of(l) == s) for s in sorted({suite_of(l) for l in labels})}}
        print(f"\nReal validation scenes: {len(idxs)} (one per episode). Scene names, first 6: {labels[:6]}")
        print(f"Suites found in the names: {R['items']['suites']}")
        ev, V, C = feature_spectrum(z1)
        ev_c, V_c = ev.to(device), V.to(device)
        print(f"Real latents in the flow's space: feat std {float(z1.std()):.3f}, pos rms {float((p1 - p1.mean((0, 1))).pow(2).mean().sqrt()):.3f}; "
              f"variance in the top 16 / 64 / 128 / 256 directions: "
              + " / ".join(f"{100 * float(ev[:k].sum() / ev.sum()):.1f}%" for k in (16, 64, 128, 256)))
        R["real_reference"] = {"feat_std": float(z1.std()), "variance_in_top_k": {str(k): float(ev[:k].sum() / ev.sum()) for k in (16, 64, 128, 256)}}

        # ------------------------------------------------------------------ learn
        if "learn" in parts:
            try:
                rows, ref = learning_test(model, z1, p1, s1, SIGMAS, args.draws, args.seed, device, ev, V)
                for r in rows:
                    r["train_share_at_least_this_noisy"] = tail(r["sigma"], p_mean, p_std)
                ref["feature_loss_share"] = loss_share(model, rows, p_mean, p_std)
                R["learn"] = {"rows": rows, "reference": ref}
                print("\n=== LEARN: denoising error per coordinate on the clean estimate (lower is better) ===")
                print("Reference estimators: prior = predict the mean; identity = take the noisy input; Gaussian = knows the full feature covariance.")
                print(f"{'sigma':>6} | {'train%>=':>8} | {'FEATURES model':>14} {'Gaussian':>9} {'identity':>9} {'prior':>7} {'model/Gauss':>11} {'shuffled':>9} | {'POSITIONS model':>15} {'Gaussian':>9} {'model/Gauss':>11} | {'W&B-style ratio f/p':>19}")
                for r in rows:
                    print(f"{r['sigma']:6.2f} | {100 * r['train_share_at_least_this_noisy']:7.2f}% | {r['model_mse_feat']:14.4f} {r['ref_gaussian_feat']:9.4f} "
                          f"{r['ref_identity']:9.4f} {r['ref_prior_mean_feat']:7.3f} {r['model_over_gaussian_feat']:11.2f} {r['shuffled_mse_feat']:9.3f} | "
                          f"{r['model_mse_pos']:15.4f} {r['ref_gaussian_pos']:9.4f} {r['model_over_gaussian_pos']:11.2f} | {r['velocity_ratio_feat']:9.3f} / {r['velocity_ratio_pos']:.3f}")
                if rows[0]["ref_gaussian_pos_given_start"] is not None:
                    print("Positions given the starting positions: Gaussian reference with a linear map from the start: "
                          + ", ".join(f"s={r['sigma']:g}: {r['ref_gaussian_pos_given_start']:.4f}" for r in rows))
                ls = ref["feature_loss_share"]
                print(f"Where the feature training loss comes from (approximate): reconstructed mean loss {ls['reconstructed_feature_loss']:.3f}; "
                      f"noise levels below 0.43 are {100 * ls['share_of_samples_below_0.43']:.0f}% of the training samples and {100 * ls['share_of_loss_below_0.43']:.0f}% of the loss; "
                      f"0.43 to 2.3: {100 * ls['share_of_samples_0.43_to_2.3']:.0f}% / {100 * ls['share_of_loss_0.43_to_2.3']:.0f}%; above 2.3: {100 * ls['share_of_samples_above_2.3']:.0f}% / {100 * ls['share_of_loss_above_2.3']:.0f}%.")
                print(f"Variance of the real features beyond the top K directions: " + ", ".join(f"K={k}: {100 * v:.1f}%" for k, v in ref["feature_variance_beyond_topK"].items()))
                print("Share of the flow's feature ERROR beyond the top K directions (K: error / variance of the data there):")
                for r in rows:
                    print(f"  sigma {r['sigma']:5.2f}: " + ", ".join(f"K={k}: {100 * v / max(r['feat_error_total_check'], 1e-12):.0f}% of the error" for k, v in r["feat_error_outside_topK"].items()))
                for r in rows:
                    summary[f"learn/sigma{r['sigma']:g}/model_mse_feat"] = r["model_mse_feat"]
                    summary[f"learn/sigma{r['sigma']:g}/model_over_gaussian_feat"] = r["model_over_gaussian_feat"]
                    summary[f"learn/sigma{r['sigma']:g}/model_over_gaussian_pos"] = r["model_over_gaussian_pos"]
                    summary[f"learn/sigma{r['sigma']:g}/velocity_ratio_feat"] = r["velocity_ratio_feat"]
                    summary[f"learn/sigma{r['sigma']:g}/velocity_ratio_pos"] = r["velocity_ratio_pos"]
                for k, v in ref["feature_variance_beyond_topK"].items():
                    summary[f"real/variance_beyond_top{k}"] = v
                if wb is not None:
                    cols = ["sigma", "train_share_at_least_this_noisy", "model_mse_feat", "ref_gaussian_feat", "ref_per_channel_feat", "ref_identity",
                            "ref_prior_mean_feat", "shuffled_mse_feat", "model_over_gaussian_feat", "model_mse_pos", "ref_gaussian_pos", "model_over_gaussian_pos",
                            "velocity_ratio_feat", "velocity_ratio_pos"]
                    import wandb
                    wb.log({"learn/table": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
            except Exception:
                print("[learn] FAILED:\n" + traceback.format_exc()); R["learn_error"] = traceback.format_exc()

        ctx = None
        if parts & {"rank", "gen", "ladder"}:
            if "gen" in parts:
                ctx_sel = list(range(len(idxs)))
            else:                                   # fewer scenes: spread them over the validation episodes
                n_ctx = max(args.n_rank if "rank" in parts else 0, args.n_ladder if "ladder" in parts else 0)
                ctx_sel = sorted({int(i) for i in np.linspace(0, len(idxs) - 1, min(n_ctx, len(idxs))).round()})
            ctx_idx = [idxs[i] for i in ctx_sel]
            ctx_labels = [labels[i] for i in ctx_sel]
            ctx = build_ctx(fm, model, val_set, ctx_idx, device)
            ps_ = [psnr(c["recon"], c["gt"]) for c in ctx]
            by_suite = {}
            for c, l, v in zip(ctx, ctx_labels, ps_):
                by_suite.setdefault(suite_of(l), []).append(v)
            R["autoencoder_psnr_vs_ground_truth"] = {"mean": float(np.mean(ps_)), "by_suite": {k: float(np.mean(v)) for k, v in by_suite.items()}}
            print(f"\nAutoencoder reconstruction against ground truth ({len(ctx)} scenes, all {ctx[0]['recon'].shape[0]} cameras): {np.mean(ps_):.2f} dB; "
                  + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in sorted(by_suite.items())))
            summary["ae/psnr_vs_gt"] = float(np.mean(ps_))

        # ------------------------------------------------------------------ rank
        if "rank" in parts:
            try:
                rk = rank_render_test(model, ctx[:args.n_rank], V_c)
                R["rank"] = rk
                print(f"\n=== RANK: real latents projected onto the top-K feature directions, decoded; PSNR against the full decode ({min(args.n_rank, len(ctx))} scenes) ===")
                print("  " + ", ".join(f"K={k}: {v:.2f} dB" for k, v in rk.items()))
                for k, v in rk.items():
                    summary[f"rank/psnr_top{k}"] = v
            except Exception:
                print("[rank] FAILED:\n" + traceback.format_exc()); R["rank_error"] = traceback.format_exc()

        # ------------------------------------------------------------------ gen
        if "gen" in parts:
            try:
                zr, pr = torch.cat([c["z"] for c in ctx]), torch.cat([c["pos"] for c in ctx])
                real_stats, real_nn = latent_report(zr, pr, ev, V_c)
                med = float(real_nn.median())
                real_stats, _ = latent_report(zr, pr, ev, V_c, real_median_nn=med)
                gens = {}
                for steps in sorted({args.compare_steps, args.steps}):
                    zg, pg = generate(model, ctx, args.n_gen, steps, args.seed, device)
                    st, _ = latent_report(zg, pg, ev, V_c, real_median_nn=med)
                    gens[steps] = (zg, pg, st)
                R["gen_latents"] = {"real": real_stats, **{f"generated_{s}_steps": v[2] for s, v in gens.items()}}
                print(f"\n=== GEN: latents (real {len(zr)} scenes; generated {args.n_gen} scenes) ===")
                def line(n, s):
                    g = s["variance_ratio_along_real_pca"]
                    return (f"  {n:24s} feat std {s['feat_std']:.3f} | pos rms {s['pos_rms']:.3f} | variance along real PCA: "
                            + " ".join(f"{k} {v:.2f}" for k, v in g.items())
                            + f" | nearest-token dist {s['nearest_token_distance']:.3f} | neighbour feature agreement {s['neighbour_feature_agreement']:.3f}"
                            + (f" (closer than real median: {s['agreement_for_pairs_closer_than_real_median']:.3f})" if 'agreement_for_pairs_closer_than_real_median' in s else ""))
                print(line("real", real_stats))
                for s, v in gens.items():
                    print(line(f"generated, {s} steps", v[2]))
                    for k in ("feat_std", "pos_rms", "nearest_token_distance", "neighbour_feature_agreement"):
                        summary[f"gen{s}/{k}"] = v[2][k]
                    for k, x in v[2]["variance_ratio_along_real_pca"].items():
                        summary[f"gen{s}/var_ratio_{k}"] = x
                for k in ("feat_std", "nearest_token_distance", "neighbour_feature_agreement"):
                    summary[f"real/{k}"] = real_stats[k]

                zg, pg, _ = gens[args.steps]
                gen_stacks = [model.render(zg[i:i + 1], pg[i:i + 1], ctx[i % len(ctx)]["ray_o"], ctx[i % len(ctx)]["ray_d"], HW)[0].float().clamp(0, 1)
                              for i in range(args.n_gen)]
                n_views = ctx[0]["recon"].shape[0]
                sets = {"gen": torch.cat(gen_stacks), "recon": torch.cat([c["recon"] for c in ctx]), "gt": torch.cat([c["gt"] for c in ctx])}
                met, err = image_metrics(sets, n_views, args.feature_net, device, args.seed)
                R["image_metrics"] = met
                if err:
                    R["deep_feature_errors"] = err
                    print("  (no DINOv2: " + "; ".join(err) + ")")
                print(f"\n=== GEN: rendered views, {args.n_gen} generated scenes ({len(sets['gen'])} views, {args.steps} steps) against {len(ctx)} real scenes ({len(sets['recon'])} views) ===")
                for name, m in met.items():
                    k = m["kid_x1000_generated_vs_reconstruction"]; fl = m["kid_x1000_reconstruction_even_vs_odd_floor"]
                    ce = m["kid_x1000_reconstruction_vs_ground_truth_ceiling"]; gg = m["kid_x1000_generated_vs_ground_truth"]
                    print(f"  [{name}] KID x1000: generated vs real reconstruction {k[0]:.2f} (sd {k[1]:.2f}) | real vs real floor {fl[0]:.2f} | "
                          f"autoencoder vs ground truth {ce[0]:.2f} | generated vs ground truth {gg[0]:.2f}")
                    print(f"     views of one scene: similarity within/between/gap: real {m['view_gap_real_reconstruction'][0]:.3f}/{m['view_gap_real_reconstruction'][1]:.3f}/{m['view_gap_real_reconstruction'][2]:.3f}; "
                          f"generated {m['view_gap_generated'][0]:.3f}/{m['view_gap_generated'][1]:.3f}/{m['view_gap_generated'][2]:.3f}")
                    tag = "dinov2" if name.startswith("DINO") else "lowlevel"
                    summary[f"img/{tag}/kid_gen_vs_recon"] = k[0]; summary[f"img/{tag}/kid_floor"] = fl[0]
                    summary[f"img/{tag}/kid_ae_ceiling"] = ce[0]; summary[f"img/{tag}/kid_gen_vs_gt"] = gg[0]
                    summary[f"img/{tag}/view_gap_real"] = m["view_gap_real_reconstruction"][2]; summary[f"img/{tag}/view_gap_gen"] = m["view_gap_generated"][2]
                pool = min(len(ctx), args.n_gen)          # scene i is generated for the cameras of real scene i
                show = diverse(labels[:pool], 6)
                n_in = ctx[0]["n_in"]
                blocks = [("ground truth (cameras 1-%d of 7)" % n_in, [[ctx[i]["gt"][v] for v in range(n_in)] for i in show]),
                          ("autoencoder reconstruction", [[ctx[i]["recon"][v] for v in range(n_in)] for i in show]),
                          (f"generated from noise, {args.steps} steps, same cameras", [[gen_stacks[i][v] for v in range(n_in)] for i in show])]
                from PIL import Image
                gal = draw_gallery(blocks, n_in, [labels[i] for i in show])
                Image.fromarray(gal).save(out_dir / "gallery.png")
                if wb is not None:
                    import wandb
                    wb.log({"gen/gallery": wandb.Image(gal, caption=f"{label}: ground truth, reconstruction, generated (scene {', '.join(labels[i][:18] for i in show)})")})
            except Exception:
                print("[gen] FAILED:\n" + traceback.format_exc()); R["gen_error"] = traceback.format_exc()

        # ------------------------------------------------------------------ ladder
        if "ladder" in parts:
            try:
                lad_sel = sorted({int(i) for i in np.linspace(0, len(ctx) - 1, min(args.n_ladder, len(ctx))).round()})
                lctx = [ctx[i] for i in lad_sel]
                llabels = [ctx_labels[i] for i in lad_sel]
                show = list(dict.fromkeys(int(i) for i in np.linspace(0, len(lctx) - 1, min(4, len(lctx))).round()))
                lad, pic = ladder_test(model, lctx, args.ladder_steps, args.seed, device, args.feature_net, show)
                lad["scene_names"] = llabels
                R["ladder"] = lad
                print(f"\n=== LADDER: real validation latents noised to sigma, denoised by the flow, decoded ({lad['n_scenes']} scenes x {lad['views_per_scene']} cameras; {lad['feature_net']}) ===")
                print(f"Reference: autoencoder against ground truth {lad['autoencoder_psnr_vs_ground_truth']:.2f} dB; KID x1000 real-vs-real floor {lad['kid_x1000_real_vs_real_floor']:.1f}, "
                      f"autoencoder-vs-ground-truth ceiling {lad['kid_x1000_autoencoder_vs_ground_truth_ceiling']:.1f}.")
                print("PSNR (dB) of the decode against the autoencoder's decode of the CLEAN latent, mean over scenes (higher is better):")
                print(f"{'sigma':>6} {'steps':>5} | " + " ".join(f"{n:>22}" for n, _ in LADDER_VARIANTS))
                for lv in lad["levels"]:
                    print(f"{lv['sigma']:6.2f} {lv['euler_steps']:5d} | " + " ".join(
                        f"{lv[n]['psnr_vs_ae_reconstruction']['mean']:22.2f}" if "psnr_vs_ae_reconstruction" in lv[n] else f"{'failed':>22}" for n, _ in LADDER_VARIANTS))
                print("KID x1000 of the decoded views against the autoencoder's decode of the clean latents (lower is better):")
                print(f"{'sigma':>6} | " + " ".join(f"{n:>22}" for n, _ in LADDER_VARIANTS))
                for lv in lad["levels"]:
                    print(f"{lv['sigma']:6.2f} | " + " ".join(
                        f"{lv[n]['kid_x1000_vs_ae_reconstruction'][0]:22.1f}" if "kid_x1000_vs_ae_reconstruction" in lv[n] else f"{'failed':>22}" for n, _ in LADDER_VARIANTS))
                print("Latent error per coordinate (flow space), flow output against the noisy input: ")
                for lv in lad["levels"]:
                    print(f"  sigma {lv['sigma']:5.2f}: features {lv['latent_error_feat_flow']:.4f} (noisy input {lv['latent_error_feat_noisy_input']:.4f}); "
                          f"positions {lv['latent_error_pos_flow']:.4f} (noisy input {lv['latent_error_pos_noisy_input']:.4f})")
                for lv in lad["levels"]:
                    for n, _ in LADDER_VARIANTS:
                        if "psnr_vs_ae_reconstruction" in lv[n]:
                            summary[f"ladder/sigma{lv['sigma']:.2f}/{n}/psnr_vs_recon"] = lv[n]["psnr_vs_ae_reconstruction"]["mean"]
                            summary[f"ladder/sigma{lv['sigma']:.2f}/{n}/kid_vs_recon"] = lv[n]["kid_x1000_vs_ae_reconstruction"][0]
                summary["ladder/kid_floor"] = lad["kid_x1000_real_vs_real_floor"]
                summary["ladder/kid_ae_ceiling"] = lad["kid_x1000_autoencoder_vs_ground_truth_ceiling"]
                if pic is not None:
                    from PIL import Image
                    Image.fromarray(pic).save(out_dir / "ladder.png")
                    if wb is not None:
                        import wandb
                        wb.log({"ladder/picture": wandb.Image(pic, caption=f"{label}: rows = scenes; blocks = flow output, real positions + flow features, flow positions + real features")})
                if wb is not None:
                    import wandb
                    cols = ["sigma"] + [f"{n}_{m}" for n, _ in LADDER_VARIANTS for m in ("psnr", "kid_x1000")]
                    data = [[lv["sigma"]] + [lv[n].get("psnr_vs_ae_reconstruction", {}).get("mean") if m == "psnr" else (lv[n].get("kid_x1000_vs_ae_reconstruction") or [None])[0]
                                              for n, _ in LADDER_VARIANTS for m in ("psnr", "kid_x1000")] for lv in lad["levels"]]
                    wb.log({"ladder/table": wandb.Table(columns=cols, data=data)})
            except Exception:
                print("[ladder] FAILED:\n" + traceback.format_exc()); R["ladder_error"] = traceback.format_exc()

    # ---------------------------------------------------------------------- reading guide
    guide = []
    L = R.get("learn", {}).get("rows")
    if L:
        by = {r["sigma"]: r for r in L}
        def g(s, key): return by[s][key] if s in by else None
        for s in (1.0, 3.0, 10.0):
            if s in by:
                guide.append(f"sigma {s:g}: feature error {by[s]['model_mse_feat']:.3f} against {by[s]['ref_gaussian_feat']:.3f} for a Gaussian that knows the covariance "
                             f"(model/Gaussian {by[s]['model_over_gaussian_feat']:.2f}; below 1 = structure beyond second-order statistics was learned).")
        if 0.1 in by:
            guide.append(f"sigma 0.1: feature error {by[0.1]['model_mse_feat']:.3f}; identity {by[0.1]['ref_identity']:.3f}, Gaussian {by[0.1]['ref_gaussian_feat']:.3f}. "
                         f"Above these = the flow cannot even pass the clean part of its input through (check the share of the error beyond the top 256 directions).")
    R["reading_guide"] = guide
    R["elapsed_minutes"] = (time.time() - t_start) / 60
    (out_dir / "diagnose.json").write_text(json.dumps(finite(R), indent=2) + "\n")
    if wb is not None:
        try:
            wb.summary.update(finite(summary))
            wb.finish()
        except Exception as e:
            print(f"[warn] W&B summary failed: {e}")
    print("\n=== Reading guide ===")
    for l in guide:
        print("  " + l)
    print(f"\nDone in {R['elapsed_minutes']:.1f} min. Saved diagnose.json" + (" and gallery.png" if (out_dir / 'gallery.png').exists() else "") + f" in {out_dir}")


if __name__ == "__main__":
    main()
