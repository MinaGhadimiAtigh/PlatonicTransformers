"""Print the learned RoPE frequencies of a flow checkpoint (they start at magnitudes between rope_sigma / pairs and rope_sigma).

Usage (from the PlatonicTransformers folder, no PYTHONPATH needed):
  python check_freqs.py mains/logs/Platonic-NVS-FM/<run id>/checkpoints/snapshot-epoch0499-step64000.ckpt
Positions in the flow are normalised to rms 1, so a frequency f gives a phase of f * distance radians; with the
finest frequency near rope_sigma, a position change of 0.05 shifts the phase by only about 0.05 * rope_sigma radians.
"""
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
rows = [(k, v) for k, v in sd.items() if torch.is_tensor(v) and k.split(".")[-1] == "freqs"]
print(f"{len(rows)} frequency tensors found" if rows else "no tensor named 'freqs' found; first keys: " + ", ".join(list(sd)[:8]))
for k, v in rows:
    mag = v.float().norm(dim=-1).flatten()
    print(f"{k[-60:]:60s} shape {tuple(v.shape)}  |f| min {mag.min():7.3f}  median {mag.median():7.3f}  max {mag.max():7.3f}")
if rows:
    allmag = torch.cat([v.float().norm(dim=-1).flatten() for _, v in rows])
    print(f"\nall layers: |f| max {allmag.max():.3f}, 95th percentile {allmag.quantile(0.95):.3f}, median {allmag.median():.3f}")
    print("Reading: if the maximum is still close to rope_sigma (4 in the current configs), the frequencies did not grow during training.")
