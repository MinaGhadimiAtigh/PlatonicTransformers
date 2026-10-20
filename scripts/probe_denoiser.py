"""What has the flow model actually learned? Probe a checkpoint on REAL val latents (no sampling, no AE).

    python scripts/probe_denoiser.py --ckpt runs/posgiven_rope16/ckpts/last.ckpt \
        --latents runs/latent_gen_online/latents_fixed

1. Feature spectrum of the real latents: how many PCA directions hold 50 / 90 / 99 % of the variance,
   and how different the channels' spreads are. Per-channel standardisation (what the flow model sees)
   gives every channel equal weight; if most of the variance lives in a few directions, most of the
   flow loss is spent on channels that carry little.
2. One-step denoising at several noise levels t: real val latent -> add noise -> model's estimate of the
   clean latent. Compared with the best estimate that uses NO learned structure (the Gaussian-prior
   linear denoiser = what the network gives when its output is 0):
     err model / err none   < 1 = the model has learned something; ~1 = it has learned nothing
     in top-PCA            the same, restricted to the directions that hold 90 % of the variance
     gap estimate          feat_pos_gap of the model's clean estimate (real: ~0.5). High here but low
                           in the samples = the SAMPLER loses it; low here = the NETWORK never learned it
"""

import argparse
import glob
import sys
from pathlib import Path

import ml_collections
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mains.main_latent_gen_FM_online import OnlineLatentFMModel, feat_pos_gap  # noqa: E402
from platonic_transformers.utils.utils import subtract_mean  # noqa: E402


def load(root, split):
    d = [np.load(f) for f in sorted(glob.glob(str(Path(root) / split / "latents_*.npz")))]
    return (torch.from_numpy(np.concatenate([x["pos"] for x in d])).float(),
            torch.from_numpy(np.concatenate([x["feat"] for x in d])).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--scenes", type=int, default=32)
    ap.add_argument("--ts", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. spectrum
    _, ftr = load(a.latents, "train")
    F = ftr.reshape(-1, ftr.shape[-1])
    F = F - F.mean(0)
    ev = torch.linalg.eigvalsh(F.T @ F / len(F)).flip(0).clamp_min(0)
    cum = (ev.cumsum(0) / ev.sum()).numpy()
    k = {q: int(np.searchsorted(cum, q) + 1) for q in (0.5, 0.9, 0.99)}
    ch = F.std(0)
    print(f"[spectrum] {F.shape[1]} channels; PCA directions for 50% / 90% / 99% of the variance: "
          f"{k[0.5]} / {k[0.9]} / {k[0.99]}")
    print(f"[spectrum] channel spreads: min {ch.min():.3g}, median {ch.median():.3g}, max {ch.max():.3g} "
          f"(max/min {float(ch.max() / ch.min().clamp_min(1e-12)):.0f}x)")
    _, U = torch.linalg.eigh(F.T @ F / len(F))
    top = U[:, -k[0.9]:]                                                    # [F, k90] in AE units

    # ---- 2. model
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    cfg = ml_collections.ConfigDict(hp["config"])
    model = OnlineLatentFMModel(cfg, hp["feat_dim"], hp["n_tokens"], hp["norm_state"], ae=None)
    model.load_state_dict(ck["state_dict"])
    ema = ck.get("callbacks", {}).get("EMACallback", {}).get("ema_state")
    if ema:
        sd = model.state_dict()
        sd.update({kk: v for kk, v in ema.items() if kk in sd})
        model.load_state_dict(sd)
    model.to(dev).eval()
    given = model.pos_given
    print(f"[model] {a.ckpt} | pos_mode={'given' if given else 'joint'} | EMA weights: {bool(ema)}")

    pva, fva = load(a.latents, "val")
    pva, fva = pva[: a.scenes], fva[: a.scenes]
    p_n, f_n = model._normalise(pva, fva)
    S, N, Fd = f_n.shape
    pos = subtract_mean(p_n.reshape(-1, 3), torch.arange(S).repeat_interleave(N)).to(dev)
    x = f_n.reshape(-1, Fd).to(dev)
    batch = torch.arange(S, device=dev).repeat_interleave(N)
    fstd = model.normer.feat_std.to(dev)
    topd = top.to(dev)

    def to_top(xn):                                       # normalised feats -> top-PCA coords (AE units)
        return (xn * fstd) @ topd

    print(f"[real]  feat_pos_gap of the clean val latents: {feat_pos_gap(pva, fva):.3f}")
    print(f"{'t':>5s} {'err model/none':>15s} {'in top-PCA':>11s} {'gap estimate':>13s}")
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        for t in a.ts:
            x0 = torch.randn(x.shape, generator=g, device=dev)
            xt = (1 - t) * x0 + t * x
            tt = torch.full((S,), t, device=dev)
            if given:
                vx = model._velocity_given(xt, pos, batch, tt)
            else:
                p0 = subtract_mean(torch.randn(pos.shape, generator=g, device=dev), batch)
                vx, _ = model._velocity(xt, (1 - t) * p0 + t * pos, batch, tt)
            # clean estimates: model, and the no-structure (network output 0) estimate
            sd_ = 1.0
            denom = t ** 2 * sd_ ** 2 + (1 - t) ** 2
            v_none = ((t * sd_ ** 2 - (1 - t)) / denom) * xt
            x1_model, x1_none = xt + (1 - t) * vx, xt + (1 - t) * v_none
            e_m, e_n = (x1_model - x).square().mean(), (x1_none - x).square().mean()
            et_m = (to_top(x1_model) - to_top(x)).square().mean()
            et_n = (to_top(x1_none) - to_top(x)).square().mean()
            gap = feat_pos_gap(pva, (x1_model * fstd).reshape(S, N, Fd).cpu())
            print(f"{t:5.2f} {float(e_m / e_n):15.3f} {float(et_m / et_n):11.3f} {gap:13.3f}")


if __name__ == "__main__":
    main()