"""check_libero_data.py: what is in the LIBERO shards, and how do the cameras behave? CPU only, no autoencoder, no GPU.

Usage (needs the PlatonicSplats checkout that has psplat/data/libero_shards.py on PYTHONPATH):
    python check_libero_data.py --shards <shard dir with index.json> --ae-dir <autoencoder folder with config.yaml>

It builds the train and validation sets exactly as main_nvs_fm_libero.py does (settings read from the autoencoder's own
training config) and prints: split sizes, cameras, frames per episode, whether the camera rig differs between episodes
and stays fixed within one, how far apart cameras of different episodes are compared with cameras inside one episode
(the same question check_camera_rigs.py asked for the gravity data), one batch, and the loading speed.
"""
import argparse, os, time, random
import numpy as np
import torch
import yaml
import ml_collections
from psplat.data.libero_shards import Dataset


def build(ae, shards, split, inference):
    tr = ae["training"]
    return Dataset(ml_collections.ConfigDict({
        "training": {
            "dataset_name": "psplat.data.libero_shards.Dataset", "dataset_path": str(shards),
            "num_views": int(tr["num_views"]), "num_input_views": int(tr["num_input_views"]),
            "num_target_views": int(tr["num_target_views"]), "pose_norm": str(tr.get("pose_norm", "lvsm_mean")),
            "scene_scale_factor": float(tr.get("scene_scale_factor", 1.35)), "include_wrist": bool(tr.get("include_wrist", False)),
            "frames_per_episode": int(tr.get("frames_per_episode", 16)), "skip_first": int(tr.get("skip_first", 0)),
            "split": split, "split_by": str(tr.get("split_by", "episode")), "holdout_frac": float(tr.get("holdout_frac", 0.1)),
            "test_frac": float(tr.get("test_frac", 0.0)), "holdout_tasks": [str(x) for x in (tr.get("holdout_tasks") or [])],
            "view_select": "first", "square_crop": False, "rot_aug": False, "rot_aug_so3": False, "target_has_input": False,
        },
        "model": {"image_tokenizer": {"image_size": int(ae["model"]["image_tokenizer"]["image_size"]),
                                      "patch_size": int(ae["model"]["image_tokenizer"]["patch_size"])}},
        "inference": {"if_inference": bool(inference)},
    }))


def centres(ds, ep):
    """Camera centres and forward vectors of the static cameras of one episode, in the dataset's normalised frame."""
    row = int(np.flatnonzero(ds.episode_id == ep)[0])
    A_inv, scale = ds._episode_frame(ep)
    c2w = A_inv @ ds.E[row][ds.view_cols]
    c2w[:, :3, 3] /= scale
    return c2w[:, :3, 3].copy(), c2w[:, :3, 2].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--ae-dir", required=True)
    ap.add_argument("--episodes", type=int, default=300, help="episodes used for the camera analysis")
    a = ap.parse_args()
    ae = yaml.safe_load(open(os.path.join(a.ae_dir, "config.yaml")))

    print("== datasets, built as the flow script builds them")
    tr = build(ae, a.shards, "train", False)
    va = build(ae, a.shards, "val", True)
    print(f"train items {len(tr)}, validation items {len(va)}")
    print(f"cameras in the shards: {tr.cameras}")
    print(f"cameras used (wrist {'included' if tr.include_wrist else 'excluded'}): {[tr.cameras[i] for i in tr.view_cols]}  -> {tr.num_views_avail} available, "
          f"{tr.config.training.num_views} per sample ({tr.config.training.num_input_views} input + {tr.config.training.num_target_views} target)")
    print(f"stored image size {tr.stored_size}, encoding {tr.encoding}, pose convention {tr.pose_convention}")
    print(f"tasks in the build: {len(tr.meta['tasks'])}; suites: {tr.meta['suites']}")

    eps_tr = np.unique(tr.episode_id[tr.rows]); eps_va = np.unique(va.episode_id[va.rows])
    print(f"episodes: train {len(eps_tr)}, validation {len(eps_va)}; overlap {len(set(eps_tr) & set(eps_va))}")
    per_ep = np.array([len(r) for r in tr._ep_rows])
    print(f"timesteps per training episode after skip_first: min {per_ep.min()}, median {int(np.median(per_ep))}, max {per_ep.max()}")

    print("\n== camera rigs (static cameras, normalised frame)")
    rng = np.random.default_rng(0)
    ep_list = rng.choice(eps_tr, size=min(a.episodes, len(eps_tr)), replace=False)
    C, F = [], []
    for ep in ep_list:
        c, f = centres(tr, int(ep)); C.append(c); F.append(f)
    C = np.stack(C); F = np.stack(F)                      # [E, V, 3]
    row0 = np.flatnonzero(tr.episode_id == ep_list[0]); st = [i for i, c in enumerate(tr.cameras) if "wrist" not in c.lower()]
    drift = np.abs(tr.E[row0[0]][st] - tr.E[row0[-1]][st]).max()
    print(f"static cameras within one episode, first against last timestep: max difference {drift:.2e}  ({'fixed' if drift < 1e-6 else 'MOVES'})")
    same = np.abs(C - C[0:1]).max(axis=(1, 2))
    print(f"rig of episode 0 against the others: median max coordinate difference {np.median(same[1:]):.3f}  ({'identical rigs' if np.median(same[1:]) < 1e-6 else 'rigs differ between episodes'})")
    within = []
    for e in range(len(C)):
        d = np.linalg.norm(C[e][:, None] - C[e][None], axis=-1); np.fill_diagonal(d, np.inf); within.append(d.min(1))
    within = np.concatenate(within)
    flat = C.reshape(-1, 3); owner = np.repeat(np.arange(len(C)), C.shape[1])
    D = np.linalg.norm(flat[:, None] - flat[None], axis=-1); D[owner[:, None] == owner[None]] = np.inf
    across = D.min(1)
    print(f"nearest other camera inside the same episode: median {np.median(within):.3f}")
    print(f"nearest camera of ANOTHER episode:            median {np.median(across):.3f}  (ratio {np.median(across) / max(np.median(within), 1e-9):.2f}; "
          f"a ratio below 1 means rigs are closer to each other than the cameras inside one rig)")
    rad = np.linalg.norm(C, axis=-1); print(f"distance of cameras from the scene centre: min {rad.min():.2f}, median {np.median(rad):.2f}, max {rad.max():.2f}")
    el = np.degrees(np.arcsin(np.clip(C[..., 2] / np.maximum(rad, 1e-9), -1, 1)))
    print(f"height angle of cameras: min {el.min():.0f}, median {np.median(el):.0f}, max {el.max():.0f} degrees")
    cosf = np.einsum("evi,ewi->evw", F, F)
    cosf[:, np.arange(F.shape[1]), np.arange(F.shape[1])] = -1.0          # ignore a camera against itself
    min_angle = np.degrees(np.arccos(np.clip(cosf.max(axis=(1, 2)), -1, 1)))
    print(f"smallest angle between the viewing directions of two cameras inside one episode: median {np.median(min_angle):.0f} degrees")

    print("\n== one training batch and the loading speed")
    random.seed(0)
    t0 = time.time(); n = 40
    for _ in range(n):
        s = tr[random.randrange(len(tr))]
    dt = time.time() - t0
    print(f"{n / dt:.1f} samples/s from one process")
    print({k: (tuple(v.shape), str(v.dtype)) for k, v in s.items() if torch.is_tensor(v)}, s["scene_name"])
    print(f"image range {float(s['image'].min()):.2f} to {float(s['image'].max()):.2f}")


if __name__ == "__main__":
    main()
