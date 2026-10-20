"""Where along the sampling path do generated scenes lose their structure?

    python scripts/probe_sampler.py --ckpt runs/latent_gen_online/ckpts/last.ckpt \
        --latents runs/latent_gen_online/latents_fixed --out runs/latent_gen_online/probe_sampler

Takes REAL val latents, noises them to a level t0 (t0 = 0: pure noise, i.e. normal sampling; t0 = 0.5:
half noise, half real scene), and lets the model integrate from there to t = 1 with the usual sampler
(Heun, same step size). Then measures the neighbour similarity (feat_pos_gap; real ~0.5) of the result.

  good from every t0 except the smallest  -> the model only fails in the high-noise phase, where the
                                             scene layout is decided (and where training samples t rarely)
  bad from every t0                       -> the integration itself goes wrong, at any noise level
Two more columns separate features from positions (meaningful for t0 > 0, where the run starts from a
real scene and should stay close to it):
  gap @real pos   the generated features evaluated at the REAL token positions. High here but low in
                  "gap result" = the features are fine, the generated POSITIONS ended up in the wrong place
  pos moved       median distance each token's generated position ends from its real one, in units of
                  the typical token spacing (>1 = tokens have swapped places with their neighbours)
Writes <out>/from_t0_<t0>.npz (pos, feat in AE units) to render with sample_latent_fm.py --samples.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_denoiser import load  # noqa: E402
from mains.main_latent_gen_FM_online import OnlineLatentFMModel, feat_pos_gap  # noqa: E402
from platonic_transformers.utils.utils import subtract_mean  # noqa: E402
import ml_collections  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50, help="steps for the FULL path t=0..1 (fewer from later t0)")
    ap.add_argument("--eta", type=float, default=0.0,
                    help="0 = the usual deterministic Heun ODE sampler. >0 = stochastic: at every step the "
                         "model's clean estimate is re-noised with this fraction of FRESH noise (1 = all fresh), "
                         "so every state looks like a training state again and errors cannot pile up")
    ap.add_argument("--t0s", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model = OnlineLatentFMModel(ml_collections.ConfigDict(hp["config"]), hp["feat_dim"], hp["n_tokens"],
                                hp["norm_state"], ae=None)
    model.load_state_dict(ck["state_dict"])
    ema = ck.get("callbacks", {}).get("EMACallback", {}).get("ema_state")
    if ema:
        sd = model.state_dict(); sd.update({k: v for k, v in ema.items() if k in sd}); model.load_state_dict(sd)
    model.to(dev).eval()
    given = model.pos_given

    pva, fva = load(a.latents, "val")
    pva, fva = pva[: a.scenes], fva[: a.scenes]
    p_n, f_n = model._normalise(pva, fva)
    S, N, F = f_n.shape
    batch = torch.arange(S, device=dev).repeat_interleave(N)
    x1 = f_n.reshape(-1, F).to(dev)
    p1 = subtract_mean(p_n.reshape(-1, 3).to(dev), batch)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    print(f"[sampler] {'deterministic Heun ODE' if a.eta == 0 else f'stochastic, eta={a.eta}'}")
    print(f"[real] feat_pos_gap of these val latents: {feat_pos_gap(pva, fva):.3f}   "
          f"({'given' if given else 'joint'} model, {a.scenes} scenes)")
    spacing = float((torch.cdist(p1.view(S, N, 3), p1.view(S, N, 3)) + torch.eye(N, device=dev) * 1e9)
                    .min(-1).values.median())
    print(f"{'t0':>5s} {'steps':>6s} {'gap result':>11s} {'gap at start':>13s} {'feat spread':>12s} "
          f"{'gap @real pos':>14s} {'pos moved':>10s}")
    g = torch.Generator(device=dev).manual_seed(0)
    t_eps = 1e-4
    with torch.no_grad():
        for t0 in a.t0s:
            x = (1 - t0) * torch.randn(x1.shape, generator=g, device=dev) + t0 * x1
            p0 = subtract_mean(torch.randn(p1.shape, generator=g, device=dev), batch)
            pos = p1 if given else (1 - t0) * p0 + t0 * p1
            start_gap = feat_pos_gap(pva, (x * model.normer.feat_std.to(dev)).reshape(S, N, F).cpu())
            n_steps = max(1, round(a.steps * (1 - t0)))
            ts = torch.linspace(t0, 1.0, n_steps + 1).tolist()

            def vel(x, pos, t):
                tt = torch.full((S,), float(t), device=dev).clamp(t_eps, 1 - t_eps)
                if given:
                    return model._velocity_given(x, pos, batch, tt), torch.zeros_like(pos)
                vx, vp = model._velocity(x, pos, batch, tt)
                return vx, subtract_mean(vp, batch)

            for i in range(n_steps):
                t, tn = ts[i], ts[i + 1]
                dt = tn - t
                vx, vp = vel(x, pos, t)
                if a.eta > 0:
                    # clean estimate and implied noise; re-noise part of it; step to the next t
                    x1h, p1h = x + (1 - t) * vx, pos + (1 - t) * vp
                    if i == n_steps - 1:
                        x, pos = x1h, (pos if given else p1h)
                        break
                    x0h = (x - t * x1h) / max(1 - t, 1e-4)
                    p0h = (pos - t * p1h) / max(1 - t, 1e-4)
                    k = (1 - a.eta ** 2) ** 0.5
                    x0n = k * x0h + a.eta * torch.randn(x.shape, generator=g, device=dev)
                    p0n = k * p0h + a.eta * subtract_mean(torch.randn(pos.shape, generator=g, device=dev), batch)
                    x = (1 - tn) * x0n + tn * x1h
                    if not given:
                        pos = (1 - tn) * p0n + tn * p1h
                    continue
                xn, pn = x + dt * vx, pos + dt * vp
                if i < n_steps - 1:
                    vx2, vp2 = vel(xn, pn, tn)
                    xn, pn = x + dt * 0.5 * (vx + vx2), pos + dt * 0.5 * (vp + vp2)
                x, pos = xn, pn
            ps, xs = model._to_scenes(x, pos, batch, S)
            P, X = model.normer.denorm(ps.float().cpu(), xs.float().cpu())
            gap = feat_pos_gap(P, X)
            spread = float(xs.std())
            gap_real = feat_pos_gap(pva, X)                     # the SAME features, at the real positions
            moved = float((ps.reshape(-1, 3) - p1).norm(dim=-1).median()) / spacing
            print(f"{t0:5.2f} {n_steps:6d} {gap:11.3f} {start_gap:13.3f} {spread:12.2f} {gap_real:14.3f} "
                  f"{moved:10.2f}")
            if out:
                tag = "" if a.eta == 0 else f"_eta{a.eta:g}"
                np.savez(out / f"from_t0_{t0:.2f}{tag}.npz", pos=P.numpy(), feat=X.numpy())
    if out:
        print(f"wrote {out}/from_t0_*.npz -- render one with sample_latent_fm.py --samples <file>")


if __name__ == "__main__":
    main()