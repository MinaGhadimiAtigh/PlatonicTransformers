"""Compare generated latents with real ones, in one picture (no autoencoder needed).

    python scripts/inspect_latent_samples.py \
        --latents /media/mina/hdd/amlab/nvsm/latents/ae_grav_trivial \
        --norm runs/latent_gen/normaliser.json \
        --samples runs/latent_gen/samples_epoch00200.npz runs/latent_gen/samples_epoch01000.npz \
                  runs/latent_gen/samples_epoch02000.npz runs/latent_gen/steps200/samples.npz \
        --out runs/latent_gen/inspect.png

Panels (all features in normalised units, i.e. real data has spread ~1):
  1. spread of each scene's features: real train / real val vs each samples file
  2. spread of each feature CHANNEL: generated / real. 1 = right; a few channels far above 1 means the
     drift is in specific channels, all channels ~1.8 means a global scale-up
  3. length of each token's feature vector (histogram): a long tail = a few exploding tokens
  4. distance of each token's position from the scene centre (histogram): checks the positions
  5-6. one real and one generated scene's token positions (top view), coloured by feature norm
Also prints the numbers behind panels 1-4.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_split(root, split):
    files = sorted(glob.glob(str(Path(root) / split / "latents_*.npz")),
                   key=lambda s: int(Path(s).stem.split("_")[1]))
    d = [np.load(f) for f in files]
    return np.concatenate([x["pos"] for x in d]), np.concatenate([x["feat"] for x in d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--norm", required=True, help="normaliser.json of the flow-matching run")
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", default="inspect.png")
    a = ap.parse_args()

    n = json.load(open(a.norm))
    m, s = np.array(n["feat_mean"]), np.array(n["feat_std"])
    sets = {}
    for split in ("train", "val"):
        try:
            p, f = load_split(a.latents, split)
            sets[f"real {split}"] = (p, (f - m) / s)
        except ValueError:
            pass
    for f in a.samples:
        z = np.load(f)
        name = Path(f).parent.name + "/" + Path(f).stem if Path(f).stem == "samples" else Path(f).stem
        sets[name.replace("samples_", "")] = (z["pos"], (z["feat"] - m) / s)

    real_f = sets["real train"][1].reshape(-1, len(m))
    real_ch = real_f.std(0)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    ax = ax.ravel()
    print(f"{'set':28s} {'scenes':>6s} {'scene spread (median, max)':>28s} {'channel ratio (med, max)':>26s} "
          f"{'token norm p50/p99':>20s} {'pos radius p50/p99':>20s}")
    for i, (k, (p, f)) in enumerate(sets.items()):
        per_scene = f.reshape(len(f), -1).std(1)
        ch_ratio = f.reshape(-1, f.shape[-1]).std(0) / real_ch
        tok = np.linalg.norm(f, axis=-1).ravel() / np.sqrt(f.shape[-1])
        rad = np.linalg.norm(p - p.mean(1, keepdims=True), axis=-1).ravel()
        print(f"{k:28s} {len(f):6d} {np.median(per_scene):14.2f} {per_scene.max():12.2f} "
              f"{np.median(ch_ratio):14.2f} {ch_ratio.max():10.2f} "
              f"{np.percentile(tok, 50):10.2f}/{np.percentile(tok, 99):<8.2f} "
              f"{np.percentile(rad, 50):10.3f}/{np.percentile(rad, 99):<8.3f}")
        col = f"C{i}"
        ax[0].scatter(np.full(len(per_scene), i) + np.random.uniform(-.15, .15, len(per_scene)), per_scene, s=8, color=col)
        if not k.startswith("real train"):
            ax[1].plot(np.sort(ch_ratio)[::-1], label=k, color=col)
        ax[2].hist(tok, bins=100, histtype="step", density=True, label=k, log=True, color=col)
        ax[3].hist(rad, bins=100, histtype="step", density=True, label=k, color=col)
    ax[0].set_xticks(range(len(sets)), list(sets), rotation=30, ha="right")
    ax[0].axhline(1, color="k", lw=.5)
    ax[0].set_title("1. per-scene feature spread (real ~1)")
    ax[1].axhline(1, color="k", lw=.5)
    ax[1].set_title("2. per-channel spread / real, sorted (1 = right)")
    ax[1].set_xlabel("channel (sorted)")
    ax[2].set_title("3. token feature norm / sqrt(F)  (log density)")
    ax[3].set_title("4. token distance from scene centre")
    for j in (1, 2, 3):
        ax[j].legend(fontsize=7)

    gen_key = list(sets)[-1]
    for j, k in ((4, "real val" if "real val" in sets else "real train"), (5, gen_key)):
        p, f = sets[k]
        c = np.linalg.norm(f[0], axis=-1) / np.sqrt(f.shape[-1])
        sc = ax[j].scatter(p[0, :, 0], p[0, :, 1], c=c, s=12, cmap="viridis", vmin=0, vmax=3)
        ax[j].set_aspect("equal")
        ax[j].set_title(f"{5 if j == 4 else 6}. {k}, scene 0: positions (x,y)\ncolour = token feature norm", fontsize=10)
        plt.colorbar(sc, ax=ax[j], fraction=.046)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()