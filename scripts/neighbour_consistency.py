"""Do generated features fit their positions? Compare neighbouring tokens, real vs generated.

In a real latent, tokens that sit close together describe the same part of the scene, so their
features are similar. If a model puts the right KINDS of features at the wrong places (what the
"shuffle" damage in decoder_noise_test.py imitates, and what the blocky renders look like), close-by
tokens stop resembling each other, although every token on its own looks normal.

    python scripts/neighbour_consistency.py \
        --latents runs/latent_gen_online/latents_fixed \
        --samples runs/latent_gen_online/samples_epoch00100.npz runs/latent_gen_online/samples_epoch00500.npz

For each set it prints the mean cosine similarity of each token's features to those of its k nearest
tokens in space, and the same for k RANDOM tokens of the same scene. Real latents: neighbours clearly
more similar than random tokens. "gap" = neighbours - random; a gap near 0 means features are placed
without regard to position. Also prints real latents with features shuffled within each scene (gap ~0),
as the floor.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch


def gap(pos, feat, k=4, seed=0):
    """Mean cos-sim to the k spatial nearest neighbours, to k random tokens, and the difference."""
    g = torch.Generator().manual_seed(seed)
    p = torch.as_tensor(pos).float()
    f = torch.as_tensor(feat).float()
    f = f - f.mean((0, 1), keepdim=True)
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    S, N, _ = p.shape
    d = torch.cdist(p, p) + torch.eye(N)[None] * 1e9
    nn = d.topk(k, largest=False).indices                                   # [S,N,k]
    sim = torch.einsum("snf,smf->snm", f, f)                                # [S,N,N]
    near = torch.gather(sim, 2, nn).mean()
    rnd_idx = torch.randint(0, N, (S, N, k), generator=g)
    rnd = torch.gather(sim, 2, rnd_idx).mean()
    return float(near), float(rnd), float(near - rnd)


def load_dir(root, split):
    files = sorted(glob.glob(str(Path(root) / split / "latents_*.npz")))
    d = [np.load(f) for f in files]
    return np.concatenate([x["pos"] for x in d]), np.concatenate([x["feat"] for x in d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_real", type=int, default=64)
    a = ap.parse_args()

    sets = {}
    for split in ("val", "train"):
        try:
            p, f = load_dir(a.latents, split)
            sets[f"real {split}"] = (p[: a.max_real], f[: a.max_real])
        except ValueError:
            pass
    p, f = next(iter(sets.values()))
    g = np.random.default_rng(0)
    sets["real, features shuffled (floor)"] = (p, np.stack([x[g.permutation(len(x))] for x in f]))
    for s in a.samples:
        z = np.load(s)
        sets[Path(s).stem] = (z["pos"], z["feat"])

    print(f"{'set':34s} {'neighbours':>10s} {'random':>8s} {'gap':>7s}   (cosine similarity, k={a.k})")
    for name, (p, f) in sets.items():
        near, rnd, gp = gap(p, f, a.k)
        print(f"{name:34s} {near:10.3f} {rnd:8.3f} {gp:7.3f}")


if __name__ == "__main__":
    main()