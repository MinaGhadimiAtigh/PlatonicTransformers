"""How precise must a generated latent be? Add noise to REAL latents and render them.

The flow model never produces latents exactly; this measures how much feature error the frozen decoder
tolerates before renders break up into the blocky patches seen in the generated samples.

    PS=/media/mina/hdd/amlab/nvsm/PlatonicSplats
    PYTHONPATH=$PS:$PS/LVSM $PS/.venv/bin/python scripts/decoder_noise_test.py \
        --psplat_repo $PS \
        --ae_run /media/mina/hdd/amlab/nvsm/ae_grav_dense_dec4_v55_20k_trivial_rayinit_bias_noaug \
        --data_root /media/mina/hdd/amlab/nvsm/gravity512v16_dense \
        --latents runs/latent_gen_online/latents_fixed --out runs/latent_gen_online/noise_test.png

Noise is added to the FEATURES only, in normalised units (sigma x each channel's std), positions kept.
Two kinds:
  gauss   every token gets noise of size sigma        (a model that is slightly off everywhere)
  tail    5% of tokens get noise of size 4 x sigma     (a model that is right except for a few tokens,
                                                         like the oversized-token tail we measured)
Prints PSNR of the noisy renders against the CLEAN render of the same latent (and the clean render's
PSNR against the real images, the ceiling), and writes a picture for the first scene: rows = real
images, clean decode, then gauss noise at each sigma.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mains.main_latent_gen_FM_online import load_psplat, render_latent  # noqa: E402


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(-1).clamp_min(1e-10)
    return float((-10 * torch.log10(mse)).mean())


def structured_test(a, ae, ds, pos, feat, sids, views, dev, g):
    """Errors in which every token still looks normal on its own, but tokens no longer fit together:

      shuffle k%   k% of the tokens swap FEATURES among themselves; positions stay. Each feature is a real
                   one from this scene, just at the wrong place (features don't match positions).
      mix k%       k% of the tokens (position + feature) are replaced by tokens of ANOTHER real scene.
                   Each token is real, but the set no longer describes one scene.
      jitter s     every token's POSITION moves by s x the typical token spacing; features stay.
    """
    from PIL import Image
    modes = [("shuffle", 0.1), ("shuffle", 0.3), ("shuffle", 1.0), ("mix", 0.3), ("mix", 0.5),
             ("jitter", 0.5), ("jitter", 1.0)]
    res = {m: [] for m in modes}
    rows0, clean_ps = [], []
    n_sc = min(a.scenes, len(sids))
    for i in range(n_sc):
        ref = ds.scene_batch(int(sids[i]), views, device=dev)
        cams = (ref["c2w"], ref["fxfycxcy"], tuple(ref["image"].shape[-2:]))
        gt = ref["image"][0].float().cpu()
        gt = (gt + 1) / 2 if gt.min() < 0 else gt
        p0, f0 = torch.from_numpy(pos[i]).float(), torch.from_numpy(feat[i]).float()
        n = len(p0)
        clean = render_latent(ae, p0, f0, cams, dev)
        clean_ps.append(psnr(clean, gt))
        if i == 0:
            rows0 += [gt, clean]
        j = (i + 1) % len(sids)
        spacing = torch.cdist(p0, p0).topk(2, largest=False).values[:, 1].median()
        for kind, k in modes:
            p, f = p0.clone(), f0.clone()
            if kind == "shuffle":
                idx = torch.randperm(n, generator=g)[: int(k * n)]
                f[idx] = f0[idx[torch.randperm(len(idx), generator=g)]]
            elif kind == "mix":
                idx = torch.randperm(n, generator=g)[: int(k * n)]
                p[idx] = torch.from_numpy(pos[j][idx.numpy()]).float()
                f[idx] = torch.from_numpy(feat[j][idx.numpy()]).float()
            else:
                p = p0 + k * spacing * torch.randn(p0.shape, generator=g)
            r = render_latent(ae, p, f, cams, dev)
            res[(kind, k)].append(psnr(r, clean))
            if i == 0:
                rows0.append(r)
    print(f"clean decode vs real images: {np.mean(clean_ps):.2f} dB  (the ceiling)")
    print(f"{'damage':>14s}  PSNR vs clean decode")
    for (kind, k), v in res.items():
        print(f"{kind:>8s} {k:5.2f}  {np.mean(v):6.2f} dB")
    out = Path(a.out).with_name(Path(a.out).stem + "_structured.png")
    grid = torch.cat([torch.cat(list(r), -1) for r in rows0], -2)
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)).save(out)
    print(f"wrote {out}: rows = real, clean, " + ", ".join(f"{m} {k}" for m, k in modes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psplat_repo", required=True)
    ap.add_argument("--ae_run", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--latents", required=True, help="latents_fixed folder (or extract_latents.py output)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--out", default="noise_test.png")
    ap.add_argument("--structured", action="store_true",
                    help="instead of random noise, test STRUCTURED errors (see structured_test)")
    a = ap.parse_args()
    dev =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae, ds_tr, ds_va, _, _ = load_psplat(a.psplat_repo, a.ae_run, None, a.data_root, dev)
    ds = ds_va if a.split == "val" and ds_va is not None else ds_tr
    files = sorted(glob.glob(str(Path(a.latents) / "train" / "latents_*.npz")))
    ftr = np.concatenate([np.load(f)["feat"] for f in files]).reshape(-1, np.load(files[0])["feat"].shape[-1])
    std = torch.from_numpy(ftr.std(0)).float()                          # per-channel std (AE units)
    z = [np.load(f) for f in sorted(glob.glob(str(Path(a.latents) / a.split / "latents_*.npz")))]
    pos = np.concatenate([x["pos"] for x in z]); feat = np.concatenate([x["feat"] for x in z])
    sids = np.concatenate([x["scene_ids"] for x in z])

    n_avail = int(ds.num_views_avail)
    views = list(range(0, n_avail, max(1, n_avail // a.views)))[: a.views]
    g = torch.Generator().manual_seed(0)
    if a.structured:
        return structured_test(a, ae, ds, pos, feat, sids, views, dev, g)
    table = {("clean", 0.0): []}
    rows0 = []
    for i in range(min(a.scenes, len(sids))):
        ref = ds.scene_batch(int(sids[i]), views, device=dev)
        cams = (ref["c2w"], ref["fxfycxcy"], tuple(ref["image"].shape[-2:]))
        gt = ref["image"][0].float().cpu()
        gt = (gt + 1) / 2 if gt.min() < 0 else gt
        f0 = torch.from_numpy(feat[i]).float()
        clean = render_latent(ae, pos[i], f0, cams, dev)
        table[("clean", 0.0)].append(psnr(clean, gt))
        if i == 0:
            rows0 += [gt, clean]
        for s in a.sigmas:
            noisy = f0 + s * std * torch.randn(f0.shape, generator=g)
            r = render_latent(ae, pos[i], noisy, cams, dev)
            table.setdefault(("gauss", s), []).append(psnr(r, clean))
            if i == 0:
                rows0.append(r)
            mask = (torch.rand(f0.shape[0], 1, generator=g) < 0.05).float()
            tail = f0 + mask * 4 * s * std * torch.randn(f0.shape, generator=g)
            table.setdefault(("tail", s), []).append(psnr(render_latent(ae, pos[i], tail, cams, dev), clean))

    print(f"clean decode vs real images: {np.mean(table[('clean', 0.0)]):.2f} dB  (the ceiling)")
    print(f"{'sigma':>6s} {'gauss: all tokens':>20s} {'tail: 5% tokens x4':>20s}   (PSNR vs clean decode)")
    for s in a.sigmas:
        print(f"{s:6.2f} {np.mean(table[('gauss', s)]):17.2f} dB {np.mean(table[('tail', s)]):17.2f} dB")
    grid = torch.cat([torch.cat(list(r), -1) for r in rows0], -2)
    from PIL import Image
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)).save(a.out)
    print(f"wrote {a.out}: rows = real, clean decode, then gauss noise sigma = {a.sigmas}")


if __name__ == "__main__":
    main()