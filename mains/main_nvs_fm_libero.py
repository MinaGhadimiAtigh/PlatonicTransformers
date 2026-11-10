"""NVS flow-matching model.

Adapts QM9FMModel to work with latents from a frozen PlatonicAutoencoder.
Each scene is represented as 256 register tokens, each with:
  - z_feat: (256, 384)  ← scalar features  (replaces atom types)
  - pos_z:  (256, 3)    ← 3D positions      (replaces atom positions)

Key differences from QM9FMModel:
  - No atom type discretization — features are continuous 384-dim vectors
  - Fixed token count (256) — no num_atoms_sampler needed
  - dense_mode=True — tokens already batched as (B, 256, D), no packing needed
  - Frozen encoder runs inside training_step to get (z_feat, pos_z) from images
  - repeats_per_epoch: each scene encoded N times per epoch with different views
  - copy_ratio: nearest-neighbour distance logged to detect memorisation
  - optional fixed global latent normaliser (dataset.norm_passes > 0): per-channel feature
    standardisation and per-axis position centring with ONE scalar position scale; statistics
    are fitted once (optionally cached in dataset.normaliser_path) and saved in the checkpoint
  - optional x-prediction for the features (diffusion.feat_prediction: x). The default, edm, is
    the previous behaviour, unchanged. Positions always keep the EDM parametrisation. See the
    comment in NVSFMModel.__init__ and the YAML keys diffusion.xpred_loss_floor / xpred_sample_floor.
  - monitoring logged to W&B (no effect on training): val/ratio_feat_sig* and val/ratio_pos_sig* (velocity error at
    fixed noise levels divided by the no-structure baseline, every validation_frequency epochs),
    val/gen_var_ratio_* and val/real_var_ratio_* (variance of generated / real validation latents in groups of
    PCA directions of the real training latents; 1.0 is ideal), and a checkpoint every
    checkpoint.snapshot_every_n_epochs epochs (default 50, 0 = off).
  - optional conditioning on the STARTING POSITIONS of the registers (diffusion.condition_start, default false = the
    previous behaviour, unchanged). The autoencoder's encoder moves every register from a starting position, which the
    input cameras fix, to its final position. With condition_start the flow gets the starting positions (3 numbers per
    token) as extra input and only generates the final positions and the features. Options (all under diffusion:):
      start_mode: input   the starting positions are extra input channels; the flow still generates the positions
      start_mode: offset  the flow generates the offset (position - start) / start_offset_scale instead, and its attention
                          geometry is the starting position plus the noisy offset
      start_dropout: p    each scene's starting positions are hidden with probability p during training (with an extra
                          'given' channel), so the same network can also generate without them (sample(start=None))
      start_offset_scale  scale of the offset in offset mode (the move has rms about 0.58 per coordinate)
    At generation the starting positions come from camera sets: use those of validation scenes (the flow never trained
    on them), because the starting positions of a training scene's cameras could identify that scene. See StartCond.
  - libero shards (dataset_name psplat.data.libero_shards.Dataset): optional keys dataset.cache_max_items (items encoded for
    the copy_ratio caches, 0 = all), dataset.val_stride (validate on every k-th item only, default 1 = all),
    dataset.norm_max_items (items per normaliser pass, default 0 = the whole training set) and training.limit_train_batches
    (fixed epoch length). All default to the previous behaviour.
"""

import os
import math
import sys

import ml_collections
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mains._optim import make_param_groups
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.utils.callbacks import EMACallback, TimerCallback
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config,
)
from platonic_transformers.utils.utils import RandomSOd
from psplat.models.platonic_autoencoder import PlatonicAutoencoder


# ---------------------------------------------------------------------------
# Latent normaliser
# ---------------------------------------------------------------------------

class LatentNormaliser:
    """Fixed, invertible global affine map for the autoencoder latents.

    - features:  (x - mean_c) / std_c        per channel
    - positions: (p - mean_xyz) / pos_scale  per-axis mean, ONE scalar scale for all axes

    The statistics are computed once over many scenes and then frozen. Nothing is computed per
    scene, so denormalise() is the exact inverse of normalise(), including the position offset
    (the decoder expects positions in the camera-normalised frame, where the cloud is not centred).
    A scalar position scale keeps the shape of the cloud unchanged.
    """

    def __init__(self, feat_mean, feat_std, pos_mean, pos_scale):
        self.feat_mean = feat_mean   # (1, 1, F)
        self.feat_std  = feat_std    # (1, 1, F)
        self.pos_mean  = pos_mean    # (1, 1, 3)
        self.pos_scale = pos_scale   # scalar tensor

    def normalise(self, pos, feat):
        dev = feat.device
        pos_n  = (pos  - self.pos_mean.to(dev))  / self.pos_scale.to(dev)
        feat_n = (feat - self.feat_mean.to(dev)) / self.feat_std.to(dev)
        return pos_n, feat_n

    def normalise_pos(self, pos):
        """Positions only (e.g. the autoencoder's starting positions): the same map as normalise()."""
        return (pos - self.pos_mean.to(pos.device)) / self.pos_scale.to(pos.device)

    def denormalise(self, pos_n, feat_n):
        dev = feat_n.device
        pos  = pos_n  * self.pos_scale.to(dev) + self.pos_mean.to(dev)
        feat = feat_n * self.feat_std.to(dev)  + self.feat_mean.to(dev)
        return pos, feat

    def state_dict(self):
        return {
            "feat_mean": self.feat_mean.detach().cpu(),
            "feat_std":  self.feat_std.detach().cpu(),
            "pos_mean":  self.pos_mean.detach().cpu(),
            "pos_scale": self.pos_scale.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["feat_mean"], d["feat_std"], d["pos_mean"], d["pos_scale"])


class LatentStatsAccumulator:
    """Streaming mean / variance of latents in float64 (no need to keep every latent in memory)."""

    def __init__(self):
        self.n = 0
        self.f_sum = self.f_sq = self.p_sum = self.p_sq = None

    @torch.no_grad()
    def update(self, pos, feat):
        feat = feat.detach().double().reshape(-1, feat.shape[-1]).cpu()
        pos  = pos.detach().double().reshape(-1, pos.shape[-1]).cpu()
        if not (torch.isfinite(feat).all() and torch.isfinite(pos).all()):
            raise RuntimeError("[normaliser] latents contain NaN or infinity")
        if self.f_sum is None:
            self.f_sum = torch.zeros(feat.shape[-1], dtype=torch.float64)
            self.f_sq  = torch.zeros_like(self.f_sum)
            self.p_sum = torch.zeros(pos.shape[-1], dtype=torch.float64)
            self.p_sq  = torch.zeros_like(self.p_sum)
        self.f_sum += feat.sum(0); self.f_sq += (feat ** 2).sum(0)
        self.p_sum += pos.sum(0);  self.p_sq += (pos ** 2).sum(0)
        self.n += feat.shape[0]

    def finalize(self):
        if self.n == 0:
            raise RuntimeError("[normaliser] no latents were accumulated")
        f_mean = self.f_sum / self.n
        f_std  = (self.f_sq / self.n - f_mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
        p_mean = self.p_sum / self.n
        p_var  = (self.p_sq / self.n - p_mean ** 2).clamp(min=0)
        p_scale = p_var.mean().sqrt().clamp(min=1e-6)   # RMS distance from the mean, all axes
        print(f"[normaliser] {self.n} tokens | feat mean |.| avg={f_mean.abs().mean():.4f}  "
              f"feat std min/median/max={f_std.min():.4f}/{f_std.median():.4f}/{f_std.max():.4f}")
        print(f"[normaliser] pos mean={[round(v, 4) for v in p_mean.tolist()]}  "
              f"pos scale={p_scale:.4f}")
        return LatentNormaliser(
            f_mean.float()[None, None], f_std.float()[None, None],
            p_mean.float()[None, None], p_scale.float(),
        )


@torch.no_grad()
def fit_normaliser(autoencoder, train_loader, n_passes=1, device="cuda", max_items=0):
    """max_items > 0: stop each pass after that many items (the loader shuffles, so this is a random subset). The libero
    shards have about 24k items per pass and read slowly, and per-channel mean and std need far fewer."""
    import time
    limit_txt = f" (at most {max_items} items per pass)" if max_items else ""
    print(f"[normaliser] fitting from {n_passes} passes over training set{limit_txt}...", flush=True)
    acc = LatentStatsAccumulator()
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    for pass_i in range(n_passes):
        seen, t0 = 0, time.time()
        for batch in train_loader:
            batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            input, _ = autoencoder.process_data(
                batch_d, has_target_image=False, target_has_input=False, compute_rays=True,
            )
            z_feat, pos_z, _ = autoencoder.encode(input)
            acc.update(pos_z, z_feat)
            seen += z_feat.shape[0]
            if max_items and seen >= max_items:
                break
        dt = time.time() - t0
        print(f"[normaliser] pass {pass_i+1}/{n_passes} done: {seen} items in {dt:.0f} s ({seen / max(dt, 1e-9):.1f} items/s)", flush=True)
    return acc.finalize()


def save_normaliser(normaliser, path, ae_ckpt_dir):
    state = normaliser.state_dict()
    state["autoencoder_checkpoint_dir"] = str(ae_ckpt_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_normaliser(path, ae_ckpt_dir):
    state = torch.load(path, map_location="cpu", weights_only=True)
    stored = state.get("autoencoder_checkpoint_dir")
    if stored != str(ae_ckpt_dir):
        raise ValueError(
            f"Normaliser file {path} was fitted for autoencoder '{stored}', but this run uses "
            f"'{ae_ckpt_dir}'. Delete the file to refit, or point normaliser_path elsewhere."
        )
    return LatentNormaliser.from_state_dict(state)


# ---------------------------------------------------------------------------
# Starting positions (conditioning)
# ---------------------------------------------------------------------------

class StartCond:
    """Starting positions of the register tokens in the flow's (normalised) position space, plus a per-scene flag.

    s       [B, N, 3]  where each register starts before the autoencoder's encoder moves it. With the autoencoder's
                       ray_unproject initialisation and jitter_std 0 these depend only on the input cameras, not on
                       the images (the flow script checks this once against ae.register_init).
    present [B]        1.0 or 0.0: whether the scene's starting positions are given. 0.0 hides them (start dropout in
                       training, unconditional sampling).
    Slicing returns a StartCond for those scenes.
    """

    def __init__(self, s, present=None):
        self.s = s
        self.present = (torch.ones(s.shape[0], device=s.device, dtype=s.dtype) if present is None
                        else present.to(device=s.device, dtype=s.dtype))

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return StartCond(self.s[idx], self.present[idx])


# ---------------------------------------------------------------------------
# Repeated dataset wrapper (repeats_per_epoch)
# ---------------------------------------------------------------------------

class RepeatedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to repeat each scene N times per epoch.

    Each repeat calls __getitem__ fresh, so the ovs.Dataset view-sampler
    picks a different random set of views each time. This multiplies the
    effective number of view combinations seen per epoch by N, acting as
    strong data augmentation for small datasets.

    Example: 512 scenes × repeats=4 = 2048 items per epoch, each with
    a different random view subset from the same scene.
    """

    def __init__(self, dataset, repeats: int):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        # map repeated index back to original scene index
        return self.dataset[idx % len(self.dataset)]


# ---------------------------------------------------------------------------
# Copy ratio utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_copy_ratio(gen_feats, train_feats, threshold_quantile: float = 0.05):
    """Compute nearest-neighbour distance between generated and training latents.

    For each generated scene, find the closest training scene by L2 distance
    on the mean-pooled z_feat (384-dim → scalar distance). Returns:
      - mean_nn_dist:  mean distance to nearest training neighbour
      - copy_ratio:    fraction of generated scenes closer than threshold
                       (threshold = threshold_quantile of pairwise train distances)

    Args:
        gen_feats:   (G, N, F) — generated z_feat from sample()
        train_feats: (T, N, F) — all training z_feat from encode()
        threshold_quantile: quantile of pairwise train distances used as copy threshold

    Returns:
        dict with mean_nn_dist and copy_ratio
    """
    # Mean-pool over tokens → (G, F) and (T, F)
    g = gen_feats.mean(dim=1).float()    # (G, F)
    t = train_feats.mean(dim=1).float()  # (T, F)

    # Pairwise L2 distances: (G, T)
    # ||g_i - t_j||^2 = ||g_i||^2 + ||t_j||^2 - 2 g_i . t_j
    g_sq = (g ** 2).sum(dim=-1, keepdim=True)   # (G, 1)
    t_sq = (t ** 2).sum(dim=-1, keepdim=True).T # (1, T)
    gt   = g @ t.T                               # (G, T)
    dists_gen_train = (g_sq + t_sq - 2 * gt).clamp(min=0).sqrt()  # (G, T)
    nn_dists = dists_gen_train.min(dim=1).values  # (G,) nearest neighbour distance

    # Threshold: quantile of pairwise training distances
    T = t.shape[0]
    if T > 1:
        t_sq2 = (t ** 2).sum(dim=-1, keepdim=True)
        tt = t @ t.T
        dists_train = (t_sq2 + t_sq2.T - 2 * tt).clamp(min=0).sqrt()
        # exclude diagonal (self-distance = 0)
        mask = ~torch.eye(T, dtype=torch.bool, device=t.device)
        pairwise = dists_train[mask]
        threshold = torch.quantile(pairwise.float(), threshold_quantile)
    else:
        threshold = torch.tensor(float("inf"))

    copy_ratio = (nn_dists < threshold).float().mean().item()
    mean_nn_dist = nn_dists.mean().item()

    return {"val/mean_nn_dist": mean_nn_dist, "val/copy_ratio": copy_ratio}


@torch.no_grad()
def mean_nn_distance(query_feats, ref_feats):
    """Mean over queries of the distance to the nearest reference, on mean-pooled z_feat.

    Same distance as compute_copy_ratio. Used for real validation latents against the training
    latents, as a reference for the generated scenes' val/mean_nn_dist.
    Args: query_feats (Q, N, F), ref_feats (R, N, F).
    """
    q = query_feats.mean(dim=1).float()
    r = ref_feats.mean(dim=1).float()
    return torch.cdist(q, r).min(dim=1).values.mean().item()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class NVSFMModel(pl.LightningModule):
    """Node-wise conditional flow matching model for NVS scene latents."""

    def __init__(self, config, autoencoder):
        super().__init__()
        self.config = config

        # Frozen encoder
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        # Dimensions from autoencoder config
        ae_cfg = autoencoder.config
        self.n_tokens  = int(ae_cfg.model.registers.n_reg)
        if int(config.diffusion.avg_num_nodes) != self.n_tokens:
            print(f"[warn] diffusion.avg_num_nodes={config.diffusion.avg_num_nodes} but the autoencoder has "
                  f"{self.n_tokens} register tokens; set avg_num_nodes to {self.n_tokens}.")
        ae_group_name  = ae_cfg.model.equivariance.solid_name.lower()
        ae_group       = PLATONIC_GROUPS[ae_group_name]
        self.scalar_dim = int(ae_cfg.model.transformer.d) // ae_group.G

        # Time coupling of positions and features (see the long comment below, next to feat_prediction).
        self.time_coupling = str(config.diffusion.get("time_coupling", "joint")).lower()
        if self.time_coupling not in ("joint", "decoupled"):
            raise ValueError(f"diffusion.time_coupling must be 'joint' or 'decoupled', got '{self.time_coupling}'")
        self._n_noise_inputs = 2 if self.time_coupling == "decoupled" else 1

        # Conditioning on the starting positions of the registers (all keys optional; default = off = the previous
        # behaviour). The encoder moves each register from a starting position (fixed by the input cameras) to its final
        # position; the final positions of different views then meet on the same surface points. Without the starting
        # positions the flow has to invent the camera arrangement and the scene together.
        #   condition_start: true   the flow gets the starting positions s (3 numbers per token, flow space) as extra
        #        input channels (scalar channels, so only the trivial group is allowed: coordinates would break the
        #        equivariance of a non-trivial one).
        #   start_mode: "input" (default)  the flow still generates the positions p.
        #               "offset"           the flow generates the offset d = (p - s) / start_offset_scale; the network's
        #        attention geometry is s + c_in * scale * (noisy d), i.e. anchored at the starting positions.
        #   start_dropout: probability of hiding s for a training scene (adds a 'given' channel); the same network can then
        #        also generate without starting positions (sample(start=None)). 0 = always given.
        #   start_offset_scale: offset mode only; the move has rms about 0.58 per coordinate in flow space.
        d_ = config.diffusion
        self.cond_start         = bool(d_.get("condition_start", False))
        self.start_mode         = str(d_.get("start_mode", "input")).lower()
        self.start_dropout      = float(d_.get("start_dropout", 0.0))
        self.start_offset_scale = float(d_.get("start_offset_scale", 0.58))
        if self.start_mode not in ("input", "offset"):
            raise ValueError(f"diffusion.start_mode must be 'input' or 'offset', got '{self.start_mode}'")
        if not (0.0 <= self.start_dropout < 1.0):
            raise ValueError("diffusion.start_dropout must be in [0, 1)")
        if self.start_offset_scale <= 0.0:
            raise ValueError("diffusion.start_offset_scale must be > 0")
        if self.cond_start and str(config.model.solid_name).lower() != "trivial":
            raise ValueError("condition_start gives the starting positions as scalar channels, which is only valid for "
                             "model.solid_name 'trivial'.")
        if self.cond_start and config.training.get("train_augm", False):
            raise ValueError("condition_start cannot be combined with training.train_augm (the starting positions would "
                             "need the same rotation).")
        self._n_cond_inputs = (3 + (1 if self.start_dropout > 0 else 0)) if self.cond_start else 0
        self._start_checked = False
        self._val_start_cache = None

        # Flow matching network
        solid_name = config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unsupported solid_name '{solid_name}'")

        self.net = PlatonicTransformer(
            input_dim=self.scalar_dim + self._n_noise_inputs + self._n_cond_inputs,
            input_dim_vec=0,
            hidden_dim=config.model.hidden_dim,
            output_dim=self.scalar_dim,
            output_dim_vec=1,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=True,
            scalar_task_level="node",
            vector_task_level="node",
            ffn_readout=config.model.ffn_readout,
            mean_aggregation=config.model.mean_aggregation,
            dropout=config.model.dropout,
            drop_path_rate=config.model.drop_path_rate,
            layer_scale_init_value=config.model.layer_scale_init_value,
            attention=config.model.attention,
            ffn_dim_factor=config.model.ffn_dim_factor,
            rope_sigma=config.model.rope_sigma,
            ape_sigma=config.model.ape_sigma,
            learned_freqs=config.model.learned_freqs,
            freq_init=config.model.freq_init,
            use_key=config.model.use_key,
            rope_on_values=config.model.get("rope_on_values", False),
            attention_backend=config.model.get("attention_backend", "scatter"),
        )

        if config.model.get("compile", True):
            self.net = torch.compile(self.net)

        self.rotation_generator = RandomSOd(3)
        self.normaliser = None
        self.val_dataset = None

        # Relative weights of the two loss terms. Each term is a mean over its own elements, so by default
        # one position coordinate counts 384/3 = 128 times as much as one feature channel. 1.0 / 1.0 is the
        # unchanged sum. Only the feature weight is meant to be tuned; keep pos_loss_weight at 1.0.
        self.feat_loss_weight = float(config.training.get("feat_loss_weight", 1.0))
        self.pos_loss_weight  = float(config.training.get("pos_loss_weight", 1.0))
        if self.feat_loss_weight < 0 or self.pos_loss_weight < 0:
            raise ValueError("feat_loss_weight and pos_loss_weight must be >= 0")
        print(f"[NVSFMModel] loss weights: feat={self.feat_loss_weight:g}  pos={self.pos_loss_weight:g}")

        # How the network output is turned into the FEATURE velocity (positions always use EDM).
        #   "edm" (default, the previous behaviour): v = c_skip * x_t + c_out * F, EDM preconditioning.
        #   "x": the network outputs the clean features x_hat directly (no skip term) and
        #        v = (x_hat - x_t) / (1 - t). Same network, same sampler, same loss up to weighting.
        #        The network input is still scaled by c_in = 1/sqrt(t^2 + (1-t)^2) (between 1 and 1.41) as in
        #        "edm" mode; JiT feeds the raw x_t. The toy tests used this scaling.
        # xpred_loss_floor: in "x" mode the loss on x_hat is edm_weight / max(1 - t, floor)^2 * (x_hat - x1)^2.
        #   For t < 1 - floor this is exactly the old weighted velocity loss. A small floor (0.05) makes the
        #   weight very large at low noise (up to 400x); 0.3 caps it; 1.0 is close to a plain x-loss.
        # xpred_sample_floor: floor on (1 - t) when x_hat is converted to a velocity for SAMPLING only.
        #   Keep it tiny so the last Euler step lands on x_hat (your sampler ends at t = 1 - 1/num_steps).
        self.feat_prediction    = str(config.diffusion.get("feat_prediction", "edm")).lower()
        self.xpred_loss_floor   = float(config.diffusion.get("xpred_loss_floor", 0.3))
        self.xpred_sample_floor = float(config.diffusion.get("xpred_sample_floor", 1e-3))
        if self.feat_prediction not in ("edm", "x"):
            raise ValueError(f"diffusion.feat_prediction must be 'edm' or 'x', got '{self.feat_prediction}'")
        if not (0.0 < self.xpred_loss_floor <= 1.0) or not (0.0 < self.xpred_sample_floor <= 0.5):
            raise ValueError("xpred_loss_floor must be in (0, 1] and xpred_sample_floor in (0, 0.5]")
        print(f"[NVSFMModel] feature prediction: {self.feat_prediction}"
              + (f"  (loss floor {self.xpred_loss_floor:g}, sampling floor {self.xpred_sample_floor:g})"
                 if self.feat_prediction == "x" else ""))

        # Separate noise levels for positions and features.
        #   "joint" (default, the previous behaviour): one time t for both, the network sees one noise input.
        #   "decoupled": features keep their own time t_f (same distribution as before); positions get their own
        #        time t_p, and the network sees both noise levels (two noise inputs, so a new network, not loadable
        #        from a "joint" checkpoint). Motivation: in real latents tokens from different input views sit at the
        #        same 3D places and share features; generated latents lack this. If positions are settled first, the
        #        feature denoiser can see where the tokens are while the objects are composed (La-Proteina uses
        #        separate times for coordinates and latents, TRELLIS generates structure first).
        # Training: log-sigma of positions and features are standard-normal draws with correlation pos_time_corr, each
        #   with the same marginal as before (P_mean, P_std). With probability pos_clean_prob the positions are nearly
        #   clean (sigma = pos_clean_sigma), because sampling spends its second half there.
        # Sampling: features use the usual uniform-t grid, positions run pos_speedup times faster (t_p = pos_speedup * t),
        #   jump to their clean estimate when t_p reaches 1 / (1 + pos_clean_sigma) and stay fixed afterwards.
        self.pos_time_corr   = float(config.diffusion.get("pos_time_corr", 0.7))
        self.pos_clean_prob  = float(config.diffusion.get("pos_clean_prob", 0.3))
        self.pos_clean_sigma = float(config.diffusion.get("pos_clean_sigma", 0.02))
        self.pos_speedup     = float(config.diffusion.get("pos_speedup", 2.0))
        if not (-1.0 < self.pos_time_corr < 1.0) or not (0.0 <= self.pos_clean_prob <= 1.0) \
                or self.pos_clean_sigma <= 0.0 or self.pos_speedup < 1.0:
            raise ValueError("need -1 < pos_time_corr < 1, 0 <= pos_clean_prob <= 1, pos_clean_sigma > 0, pos_speedup >= 1")
        self._t_clean = 1.0 / (1.0 + self.pos_clean_sigma)
        if self.time_coupling == "decoupled":
            if int(config.diffusion.get("fm_ode_order", 1)) != 1:
                raise ValueError("time_coupling 'decoupled' supports fm_ode_order 1 only")
            print(f"[NVSFMModel] time coupling: decoupled  (corr {self.pos_time_corr:g}, clean prob {self.pos_clean_prob:g}, "
                  f"clean sigma {self.pos_clean_sigma:g}, positions {self.pos_speedup:g}x faster when sampling)")
        else:
            print("[NVSFMModel] time coupling: joint")
        if self.cond_start:
            print(f"[NVSFMModel] conditioning on the starting positions: mode {self.start_mode}, start dropout "
                  f"{self.start_dropout:g}" + (f", offset scale {self.start_offset_scale:g}" if self.start_mode == "offset" else ""))
        else:
            print("[NVSFMModel] conditioning on the starting positions: off")

        # Cache of training latents for copy_ratio — built lazily
        self._train_feats_cache = None
        self._val_feats_cache = None
        self._pca_eigvecs = None   # PCA of the real training latents (flow space), built lazily
        self._pca_eigvals = None

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def set_normaliser(self, normaliser: LatentNormaliser):
        self.normaliser = normaliser
        print(f"[NVSFMModel] normaliser set: "
              f"feat_std={normaliser.feat_std.mean():.4f}  pos_scale={normaliser.pos_scale:.4f}")

    def set_val_dataset(self, val_dataset):
        self.val_dataset = val_dataset

    def set_train_loader(self, train_loader):
        """Store train loader so we can encode all training scenes for copy_ratio."""
        self._train_loader = train_loader

    def set_val_loader(self, val_loader):
        """Store val loader so we can encode real validation scenes as a distance reference."""
        self._val_loader = val_loader

    # The normaliser is not a parameter or buffer, so store it in the checkpoint explicitly.
    # Without it a checkpoint cannot turn generated latents back into the decoder's space.
    def on_save_checkpoint(self, checkpoint):
        if self.normaliser is not None:
            checkpoint["latent_normaliser"] = self.normaliser.state_dict()
        checkpoint["feat_prediction"] = self.feat_prediction
        checkpoint["time_coupling"] = self.time_coupling
        checkpoint["conditioning"] = self._conditioning_state()

    def on_load_checkpoint(self, checkpoint):
        # Checkpoints from before this option have no key and were trained with "edm". Loading weights with
        # the wrong output parametrisation would run without error and give garbage, so refuse.
        saved_mode = checkpoint.get("feat_prediction", "edm")
        if saved_mode != self.feat_prediction:
            raise ValueError(
                f"This checkpoint was trained with diffusion.feat_prediction='{saved_mode}' but the config "
                f"says '{self.feat_prediction}'. Set diffusion.feat_prediction: {saved_mode} to load it."
            )
        saved_coupling = checkpoint.get("time_coupling", "joint")
        if saved_coupling != self.time_coupling:
            raise ValueError(
                f"This checkpoint was trained with diffusion.time_coupling='{saved_coupling}' but the config "
                f"says '{self.time_coupling}'. Set diffusion.time_coupling: {saved_coupling} to load it."
            )
        mine = self._conditioning_state()
        saved_c = checkpoint.get("conditioning", {"start": False, "mode": "input", "dropout_on": False,
                                                  "offset_scale": mine["offset_scale"]})
        if saved_c["start"] != mine["start"] or (mine["start"] and (
                saved_c["mode"] != mine["mode"] or saved_c["dropout_on"] != mine["dropout_on"]
                or (mine["mode"] == "offset" and abs(saved_c["offset_scale"] - mine["offset_scale"]) > 1e-9))):
            raise ValueError(f"This checkpoint was trained with conditioning {saved_c} but the config says {mine}. "
                             f"Use the diffusion.condition_start / start_mode / start_dropout / start_offset_scale of the run.")
        saved = checkpoint.get("latent_normaliser")
        if saved is not None:
            self.normaliser = LatentNormaliser.from_state_dict(saved)
            print("[NVSFMModel] restored latent normaliser from checkpoint")
        elif self.normaliser is not None:
            raise ValueError(
                "This run uses a latent normaliser (dataset.norm_passes > 0) but the checkpoint "
                "has none, so it was trained on raw latents. Set norm_passes: 0 to match it."
            )

    def _conditioning_state(self):
        return {"start": self.cond_start, "mode": self.start_mode, "dropout_on": self.start_dropout > 0,
                "offset_scale": self.start_offset_scale}

    # -----------------------------------------------------------------------
    # Starting positions (conditioning)
    # -----------------------------------------------------------------------

    def as_cond(self, start, B=None, device=None):
        """None, a tensor [B, N, 3] (flow space) or a StartCond -> StartCond, or None for a model without conditioning
        (whatever is passed). A conditioned model given None runs WITHOUT starting positions, which only works if it
        was trained with start_dropout > 0 (it then has a 'given' channel)."""
        if not self.cond_start:
            return None
        if isinstance(start, StartCond):
            return start
        if start is None:
            if self.start_dropout <= 0:
                raise ValueError("This flow was trained without start_dropout, so it cannot generate without the "
                                 "starting positions. Pass start=...")
            if B is None or device is None:
                raise ValueError("as_cond(None) needs B and device")
            return StartCond(torch.zeros(B, self.n_tokens, 3, device=device), torch.zeros(B, device=device))
        return StartCond(start)

    def _start_parts(self, cond, ref):
        """(s_eff [B, N, 3], scale [B, 1, 1], channels [B, N, C]) of a StartCond; hidden scenes get s_eff 0 and scale 1.
        (None, None, None) for a model without conditioning or without start."""
        if cond is None or not self.cond_start:
            return None, None, None
        pres  = cond.present.to(ref.dtype)[:, None, None]
        s_eff = cond.s.to(ref.dtype) * pres
        if self.start_mode == "offset":
            scale = torch.where(pres > 0, torch.full_like(pres, self.start_offset_scale), torch.ones_like(pres))
        else:
            scale = torch.ones_like(pres)
        chans = [s_eff] + ([pres.expand(-1, s_eff.shape[1], 1)] if self.start_dropout > 0 else [])
        return s_eff, scale, torch.cat(chans, dim=-1)

    def _pos_to_state(self, pos, cond):
        """Positions to the flow's position state: unchanged in input mode, the scaled offset from the start in offset mode."""
        if cond is None or not self.cond_start or self.start_mode != "offset":
            return pos
        s_eff, scale, _ = self._start_parts(cond, pos)
        return (pos - s_eff) / scale

    def _state_to_pos(self, state, cond):
        if cond is None or not self.cond_start or self.start_mode != "offset":
            return state
        s_eff, scale, _ = self._start_parts(cond, state)
        return s_eff + scale * state

    def norm_start(self, s):
        """Raw starting positions from the autoencoder to the flow's position space."""
        return s if self.normaliser is None else self.normaliser.normalise_pos(s)

    @torch.no_grad()
    def start_from_rays(self, ray_o, ray_d):
        """Starting positions of the registers from the input cameras' rays alone (no images), in the flow's space.
        ray_o, ray_d: [B, V_in, 3, H, W] as returned by process_data.compute_rays."""
        _, pos = self.autoencoder.register_init(ray_o, ray_d)
        return self.norm_start(pos)

    # -----------------------------------------------------------------------
    # Core FM math
    # -----------------------------------------------------------------------

    def _sample_t(self, batch_size, device):
        rnd_normal = torch.randn(batch_size, device=device)
        sigma = (rnd_normal * self.config.diffusion.P_std + self.config.diffusion.P_mean).exp()
        return 1.0 / (1.0 + sigma)

    def _sample_times(self, batch_size, device):
        """Training times (t_features, t_positions). Joint: the same t twice (same random numbers as before)."""
        if self.time_coupling == "joint":
            t = self._sample_t(batch_size, device)
            return t, t
        d = self.config.diffusion
        z_f = torch.randn(batch_size, device=device)
        z_p = self.pos_time_corr * z_f + math.sqrt(1.0 - self.pos_time_corr ** 2) * torch.randn(batch_size, device=device)
        sigma_f = (z_f * d.P_std + d.P_mean).exp()
        sigma_p = (z_p * d.P_std + d.P_mean).exp()
        clean = torch.rand(batch_size, device=device) < self.pos_clean_prob
        sigma_p = torch.where(clean, torch.full_like(sigma_p, self.pos_clean_sigma), sigma_p)
        return 1.0 / (1.0 + sigma_f), 1.0 / (1.0 + sigma_p)

    def _edm_weight(self, t):
        sigma_data = self.config.diffusion.sigma_data
        return (t ** 2 * sigma_data ** 2 + (1 - t) ** 2) / sigma_data ** 2

    def _edm_coefs(self, t_):
        sigma_data = self.config.diffusion.sigma_data
        denom   = t_ ** 2 * sigma_data ** 2 + (1 - t_) ** 2
        c_in    = 1.0 / denom.sqrt()
        c_skip  = (t_ * sigma_data ** 2 - (1 - t_)) / denom
        c_out   = sigma_data / denom.sqrt()
        c_noise = ((1 - t_) / t_).log() / 4
        return c_in, c_skip, c_out, c_noise

    def _forward_net(self, x_t, pos_t, t, t_pos=None, start=None):
        """Run the network. Returns the raw feature output, the raw position output and the EDM coefficients
        (c_skip, c_out) for features and for positions (shape [B, 1, 1]). In "joint" mode positions use t too.
        In "decoupled" mode t_pos (default: t) is the position time and the network gets both noise levels.
        start: a StartCond, required when the model is conditioned on the starting positions. pos_t is then the position
        STATE (the position itself in input mode, the scaled offset from the start in offset mode)."""
        if self.cond_start and start is None:
            raise ValueError("This flow is conditioned on the starting positions: pass start=StartCond(...) or use "
                             "model.as_cond(None, B, device) for the unconditional mode of a start-dropout model.")
        decoupled = self.time_coupling == "decoupled"
        t_  = t[:, None, None]
        tp_ = t_ if (t_pos is None or not decoupled) else t_pos[:, None, None]
        c_in, c_skip, c_out, c_noise = self._edm_coefs(t_)
        if tp_ is t_:
            c_in_p, c_skip_p, c_out_p, c_noise_p = c_in, c_skip, c_out, c_noise
        else:
            c_in_p, c_skip_p, c_out_p, c_noise_p = self._edm_coefs(tp_)
        B, N = x_t.shape[0], x_t.shape[1]
        noise_inputs = [c_noise.expand(B, N, 1)]
        if decoupled:
            noise_inputs.append(c_noise_p.expand(B, N, 1))

        x_in      = c_in * x_t
        pos_in    = c_in_p * pos_t
        s_eff, s_scale, cond_in = self._start_parts(start, x_t)
        if s_eff is not None and self.start_mode == "offset":
            pos_in = s_eff + c_in_p * s_scale * pos_t        # geometry anchored at the starting positions
        scalar_in = torch.cat([x_in] + noise_inputs + ([cond_in] if cond_in is not None else []), dim=-1)

        scalar_out, vector_out = self.net(
            scalar_in, pos_in, batch=None,
            avg_num_nodes=self.config.diffusion.avg_num_nodes,
        )
        return scalar_out, vector_out.squeeze(-2), (c_skip, c_out), (c_skip_p, c_out_p)

    def _velocity(self, x_t, pos_t, t, t_pos=None, start=None):
        scalar_out, vector_out, (c_skip, c_out), (c_skip_p, c_out_p) = self._forward_net(x_t, pos_t, t, t_pos, start=start)
        t_ = t[:, None, None]
        if self.feat_prediction == "x":
            # the network output IS the clean-feature estimate x_hat
            v_x = (scalar_out - x_t) / (1 - t_).clamp_min(self.xpred_sample_floor)
        else:
            v_x = c_skip * x_t + c_out * scalar_out
        v_pos = c_skip_p * pos_t + c_out_p * vector_out
        return v_x, v_pos

    def _flow_losses(self, x1, pos1, x0, pos0, t, t_pos=None, start=None):
        """Weighted feature and position losses, and the plain feature MSE of the clean estimate.

        edm mode: identical to the previous code (EDM-weighted velocity error).
        x mode:   features use edm_weight / max(1 - t, xpred_loss_floor)^2 * (x_hat - x1)^2, which equals the
                  EDM-weighted velocity error for t < 1 - floor. Positions are unchanged.
        feat_xhat_mse is (x_hat - x1)^2 averaged, with x_hat = x_t + (1 - t) v in edm mode, so it can be
        compared across the two modes.
        pos1 are POSITIONS (flow space); with start in offset mode the flow's position state is the scaled offset from
        the starting positions, and the position loss is in those units (pos0 is noise in state space)."""
        pos1 = self._pos_to_state(pos1, start)
        t_  = t[:, None, None]
        tp_ = t_ if (t_pos is None or self.time_coupling != "decoupled") else t_pos[:, None, None]
        x_t   = (1 - t_)  * x0   + t_  * x1
        pos_t = (1 - tp_) * pos0 + tp_ * pos1

        scalar_out, vector_out, (c_skip, c_out), (c_skip_p, c_out_p) = self._forward_net(x_t, pos_t, t, t_pos, start=start)
        weight   = self._edm_weight(t_)
        weight_p = self._edm_weight(tp_)

        v_pos     = c_skip_p * pos_t + c_out_p * vector_out
        loss_pos  = (weight_p * (v_pos - (pos1 - pos0)).square()).mean()
        if self.feat_prediction == "x":
            x_hat     = scalar_out
            w_x       = weight / (1 - t_).clamp_min(self.xpred_loss_floor).square()
            loss_feat = (w_x * (x_hat - x1).square()).mean()
        else:
            v_x       = c_skip * x_t + c_out * scalar_out
            loss_feat = (weight * (v_x - (x1 - x0)).square()).mean()
            x_hat     = x_t + (1 - t_) * v_x
        feat_xhat_mse = (x_hat - x1).square().mean().detach()
        return loss_feat, loss_pos, feat_xhat_mse

    # -----------------------------------------------------------------------
    # Monitoring helpers (logging only)
    # -----------------------------------------------------------------------

    _MONITOR_SIGMAS = (0.1, 0.3, 1.0, 3.0, 10.0)
    _PCA_GROUPS = (("top16", 0, 16), ("16to64", 16, 64), ("64to128", 64, 128), ("weak128plus", 128, None))

    @staticmethod
    def _no_structure_baseline(t):
        """Per-coordinate velocity error of the best estimator that ignores all structure (unit-variance data,
        unit noise, unweighted). Same definition as sampler_diagnostics.py. A model that learned structure
        comes in below it, so the ratios below are < 1 where it learned something and about 1 where it did not."""
        return 2.0 - (2 * t - 1) ** 2 / (t ** 2 + (1 - t) ** 2)

    @torch.no_grad()
    def _noise_level_ratios(self, x1, pos1, batch_idx, start=None):
        """Velocity error / baseline at fixed noise levels, with fixed noise so epochs are comparable.
        With start in offset mode the noise is in the offset state; the position error is reported in position units."""
        out = {}
        pos1 = self._pos_to_state(pos1, start)
        _, p_scale, _ = self._start_parts(start, x1)
        if p_scale is None:
            p_scale = 1.0
        gen = torch.Generator(device=x1.device)
        gen.manual_seed(1234 + int(batch_idx))
        B = x1.shape[0]
        for sigma in self._MONITOR_SIGMAS:
            t_val = 1.0 / (1.0 + sigma)
            t  = torch.full((B,), t_val, device=x1.device, dtype=x1.dtype)
            t_ = t[:, None, None]
            x0   = torch.randn(x1.shape,   device=x1.device, dtype=x1.dtype, generator=gen)
            pos0 = torch.randn(pos1.shape, device=x1.device, dtype=x1.dtype, generator=gen)
            x_t   = (1 - t_) * x0   + t_ * x1
            pos_t = (1 - t_) * pos0 + t_ * pos1
            vx, vp = self._velocity(x_t, pos_t, t, start=start)
            base = self._no_structure_baseline(torch.tensor(t_val)).item()
            out[f"val/ratio_feat_sig{sigma:g}"] = (vx - (x1 - x0)).square().mean() / base
            out[f"val/ratio_pos_sig{sigma:g}"]  = (p_scale * (vp - (pos1 - pos0))).square().mean() / base
            if self.time_coupling == "decoupled":
                # the situation of the second half of sampling: features at this noise level, positions nearly clean
                tp_c  = torch.full_like(t, self._t_clean)
                pos_c = (1 - self._t_clean) * pos0 + self._t_clean * pos1
                vx_c, _ = self._velocity(x_t, pos_c, t, tp_c, start=start)
                out[f"val/ratio_feat_sig{sigma:g}_posclean"] = (vx_c - (x1 - x0)).square().mean() / base
        return out

    def _to_flow_feats(self, feats):
        """Raw AE features (..., F) to the flow's space (per-channel standardisation if a normaliser is used)."""
        if self.normaliser is None:
            return feats
        _, z = self.normaliser.normalise(torch.zeros(*feats.shape[:-1], 3, device=feats.device), feats)
        return z

    @torch.no_grad()
    def _ensure_pca(self):
        """PCA (second-moment matrix, about 0) of the real TRAINING latents in the flow's space, from the cache
        that copy_ratio already builds. Returns False while that cache is not available."""
        if self._pca_eigvecs is not None:
            return True
        cache = self._train_feats_cache
        if cache is None:
            return False
        F = cache.shape[-1]
        C = torch.zeros(F, F, dtype=torch.float64)
        n = 0
        for chunk in cache.split(256):
            z = self._to_flow_feats(chunk).reshape(-1, F).float()
            C += (z.T @ z).double()
            n += z.shape[0]
        evals, evecs = torch.linalg.eigh(C / n)
        order = evals.argsort(descending=True)
        self._pca_eigvals = evals[order].clamp_min(1e-12).float()
        self._pca_eigvecs = evecs[:, order].float()
        print(f"[pca] real training latents: top-16 / 64 / 128 hold "
              f"{100 * self._pca_eigvals[:16].sum() / self._pca_eigvals.sum():.1f}% / "
              f"{100 * self._pca_eigvals[:64].sum() / self._pca_eigvals.sum():.1f}% / "
              f"{100 * self._pca_eigvals[:128].sum() / self._pca_eigvals.sum():.1f}% of the variance")
        return True

    @torch.no_grad()
    def _var_ratios(self, z_flat):
        """Variance of z_flat (M, F, flow space) along the real PCA directions, per group, divided by the real
        variance in that group. 1.0 is ideal; above 1 means too much energy in those directions."""
        v   = self._pca_eigvecs.to(z_flat.device)
        lam = self._pca_eigvals.to(z_flat.device)
        second_moment = (z_flat.float() @ v).square().mean(0)
        out = {}
        for name, lo, hi in self._PCA_GROUPS:
            if lo >= lam.numel():
                continue
            out[name] = (second_moment[lo:hi].sum() / lam[lo:hi].sum()).item()
        return out


    # -----------------------------------------------------------------------
    # Encode
    # -----------------------------------------------------------------------

    def _encode_with_start(self, batch):
        """Encode a raw image batch → (z_feat, pos_z, start), optionally normalised. start is the registers' starting
        position from the same encoder pass in the flow's space (None unless the model is conditioned on it)."""
        with torch.no_grad():
            input, _ = self.autoencoder.process_data(
                batch, has_target_image=False, target_has_input=False, compute_rays=True,
            )
            z_feat, pos_z, pos_init = self.autoencoder.encode(input)
            if self.cond_start and not self._start_checked:
                # The starting positions must be reproducible from the cameras alone, otherwise the ones used at
                # generation time (computed from rays) would differ from the ones the flow trained with.
                ref = self.autoencoder.register_init(input.ray_o, input.ray_d)[1]
                diff = (ref - pos_init).abs().max().item()
                print(f"[NVSFMModel] starting positions: max |encode() start - register_init(rays)| = {diff:.2e}"
                      + ("" if diff < 1e-4 else "  WARNING: not reproducible from the rays alone (jitter?)."))
                self._start_checked = True

        start = None
        if self.cond_start:
            start = self.norm_start(pos_init)
        if self.normaliser is not None:
            pos_z, z_feat = self.normaliser.normalise(pos_z, z_feat)
        return z_feat, pos_z, start

    def _encode(self, batch):
        """Encode a raw image batch → (z_feat, pos_z), optionally normalised."""
        z_feat, pos_z, _ = self._encode_with_start(batch)
        return z_feat, pos_z

    # -----------------------------------------------------------------------
    # Training / validation steps
    # -----------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x1, pos1, start_s = self._encode_with_start(batch)
        B = x1.shape[0]
        cond = None
        if self.cond_start:
            present = (torch.rand(B, device=x1.device) >= self.start_dropout).to(x1.dtype)
            cond = StartCond(start_s, present)

        if self.config.training.train_augm:
            rot  = self.rotation_generator().type_as(pos1)
            pos1 = torch.einsum("ij,bnj->bni", rot, pos1)

        t, t_pos = self._sample_times(B, x1.device)
        x0  = torch.randn_like(x1)
        pos0 = torch.randn_like(pos1)

        loss_feat, loss_pos, feat_xhat_mse = self._flow_losses(x1, pos1, x0, pos0, t, t_pos, start=cond)
        loss = self.feat_loss_weight * loss_feat + self.pos_loss_weight * loss_pos

        # train/loss is the weighted sum that is optimised; the *_feat / *_pos terms and
        # loss_unweighted are the plain values, comparable across runs with different weights.
        self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=B)
        self.log("train/loss_unweighted", loss_feat + loss_pos, on_step=True, on_epoch=True, logger=True, batch_size=B)
        self.log("train/loss_feat", loss_feat, on_step=True, on_epoch=True, logger=True, batch_size=B)
        self.log("train/loss_pos", loss_pos, on_step=True, on_epoch=True, logger=True, batch_size=B)
        self.log("train/feat_xhat_mse", feat_xhat_mse, on_step=True, on_epoch=True, logger=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, pos1, start_s = self._encode_with_start(batch)
        B = x1.shape[0]
        cond = StartCond(start_s) if self.cond_start else None      # validation: starting positions given
        t, t_pos = self._sample_times(B, x1.device)
        x0, pos0 = torch.randn_like(x1), torch.randn_like(pos1)
        loss_feat, loss_pos, feat_xhat_mse = self._flow_losses(x1, pos1, x0, pos0, t, t_pos, start=cond)
        loss = self.feat_loss_weight * loss_feat + self.pos_loss_weight * loss_pos
        # val/loss (the checkpoint monitor) is the weighted sum; compare runs with val/loss_unweighted.
        self.log("val/loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=B)
        self.log("val/loss_unweighted", loss_feat + loss_pos, on_step=False, on_epoch=True, logger=True, batch_size=B)
        self.log("val/loss_feat", loss_feat, on_step=False, on_epoch=True, logger=True, batch_size=B)
        self.log("val/loss_pos", loss_pos, on_step=False, on_epoch=True, logger=True, batch_size=B)
        self.log("val/feat_xhat_mse", feat_xhat_mse, on_step=False, on_epoch=True, logger=True, batch_size=B)
        if self.cond_start and self.start_dropout > 0:
            # the same batch with the starting positions hidden: how much do they help?
            lf_n, lp_n, _ = self._flow_losses(x1, pos1, x0, pos0, t, t_pos, start=StartCond(start_s, torch.zeros(B, device=x1.device)))
            self.log("val/loss_feat_nostart", lf_n, on_step=False, on_epoch=True, logger=True, batch_size=B)
            self.log("val/loss_pos_nostart", lp_n, on_step=False, on_epoch=True, logger=True, batch_size=B)
        # Velocity error at fixed noise levels over the no-structure baseline, on the same epochs as the
        # generation statistics (every validation_frequency epochs); logging only.
        if (self.current_epoch + 1) % self.config.training.validation_frequency == 0:
            for name, value in self._noise_level_ratios(x1, pos1, batch_idx, start=cond).items():
                self.log(name, value, on_step=False, on_epoch=True, logger=True, batch_size=B)
        return loss

    # -----------------------------------------------------------------------
    # Full validation: render + copy_ratio
    # -----------------------------------------------------------------------

    def _build_train_feats_cache(self, device):
        """Encode all training scenes once and cache z_feat for copy_ratio.

        Uses the stored train_loader. Runs under no_grad, frozen encoder.
        Returns (T, n_tokens, scalar_dim) tensor on CPU.
        """
        return self._build_feats_cache(getattr(self, "_train_loader", None), device, "training")

    def _build_feats_cache(self, loader, device, label):
        """Same as above for any loader (training or validation). (T, n_tokens, scalar_dim) on CPU."""
        if loader is None:
            return None
        print(f"[copy_ratio] encoding {label} set for cache...")
        ds_cfg = getattr(self.config, "dataset", None)
        limit = int(ds_cfg.get("cache_max_items", 0) or 0) if ds_cfg is not None else 0   # 0: the whole loader
        all_feats, got = [], 0
        for batch in loader:
            if limit and got >= limit:
                break
            batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                input, _ = self.autoencoder.process_data(
                    batch_d, has_target_image=False, target_has_input=False, compute_rays=True,
                )
                z_feat, _, _ = self.autoencoder.encode(input)
            all_feats.append(z_feat.cpu())
            got += z_feat.shape[0]
        cache = torch.cat(all_feats, dim=0)  # (T, N, F)
        if limit:
            cache = cache[:limit]
        print(f"[copy_ratio] cached {cache.shape[0]} {label} scenes")
        return cache

    def on_validation_epoch_end(self):
        freq = self.config.training.validation_frequency
        if (self.current_epoch + 1) % freq != 0:
            return

        if not hasattr(self, "val_dataset") or self.val_dataset is None:
            return
        logger = self.trainer.logger if self.trainer is not None else None
        if logger is None or not hasattr(logger, "experiment"):
            return

        try:
            import wandb
            import numpy as np

            n_scenes  = 4
            view_idxs = [0, 3, 6, 9, 12]      # gravity default; the libero shards choose their own, see _val_views
            n_views   = len(self._val_views(self._val_index(0), view_idxs)[0])
            device    = next(self.net.parameters()).device
            sampler   = self.config.diffusion.get("sampler", "fm_ode")
            image_h_w = (128, 128)

            # --- sample generated latents ---
            # Sample more scenes than we render, so the statistics and distances below are less coarse.
            n_stat = 16
            # A conditioned flow gets the starting positions of validation scenes' cameras (the same cameras that
            # render the scene below); the first n_scenes are the scenes rendered.
            cond_val = self._val_cond(n_stat, view_idxs, device) if self.cond_start else None
            z_all, pos_all = self.sample(n_stat, sampler=sampler, start=cond_val)
            z_gen, pos_gen = z_all[:n_scenes], pos_all[:n_scenes]

            # --- statistics of the generated latents, in the flow's own (model) space. With the
            #     normaliser on and a well-trained flow: feat mean ~0, feat std ~1, pos rms ~1,
            #     pos mean ~0. Large deviations mean the samples are off the training distribution. ---
            pos_c = pos_all - pos_all.mean(dim=(0, 1), keepdim=True)
            gen_stats = {
                "val/gen_feat_mean":    z_all.mean().item(),
                "val/gen_feat_std":     z_all.std().item(),
                "val/gen_pos_rms":      pos_c.pow(2).mean().sqrt().item(),
                "val/gen_pos_mean_abs": pos_all.mean(dim=(0, 1)).abs().mean().item(),
                "trainer/global_step":  self.global_step,
            }
            logger.experiment.log(gen_stats)
            print(f"[gen latents] epoch {self.current_epoch} | "
                  f"feat mean={gen_stats['val/gen_feat_mean']:+.3f} std={gen_stats['val/gen_feat_std']:.3f} | "
                  f"pos rms={gen_stats['val/gen_pos_rms']:.3f} mean_abs={gen_stats['val/gen_pos_mean_abs']:.3f}")

            # --- copy_ratio / nearest-neighbour distance: build the caches once per run ---
            if self._train_feats_cache is None:
                self._train_feats_cache = self._build_train_feats_cache(device)
            if self._val_feats_cache is None:
                self._val_feats_cache = self._build_feats_cache(
                    getattr(self, "_val_loader", None), device, "validation")

            # --- variance of the generated latents along the REAL principal directions (own try block so a
            #     failure here cannot skip the rendering below) ---
            try:
                if self._ensure_pca():
                    n_f = z_all.shape[-1]
                    pca_log = {f"val/gen_var_ratio_{k}": v
                               for k, v in self._var_ratios(z_all.reshape(-1, n_f)).items()}
                    if self._val_feats_cache is not None:
                        z_real_ref = self._to_flow_feats(self._val_feats_cache[:16]).reshape(-1, n_f).to(device)
                        pca_log.update({f"val/real_var_ratio_{k}": v
                                        for k, v in self._var_ratios(z_real_ref).items()})
                    pca_log["trainer/global_step"] = self.global_step
                    logger.experiment.log(pca_log)
                    print("[pca] epoch {} | generated / real variance: ".format(self.current_epoch)
                          + "  ".join(f"{k.replace('val/gen_var_ratio_', '')}={v:.2f}"
                                      for k, v in pca_log.items() if k.startswith("val/gen_var_ratio_")))
            except Exception as e:
                print(f"[warn] PCA variance ratios skipped: {e}")

            if self._train_feats_cache is not None:
                # The caches hold raw AE features, so compare in raw AE space.
                z_cmp = z_all
                if self.normaliser is not None:
                    _, z_cmp = self.normaliser.denormalise(pos_all, z_all)
                cr = compute_copy_ratio(
                    z_cmp.cpu(),
                    self._train_feats_cache,
                    threshold_quantile=self.config.training.get("copy_ratio_quantile", 0.05),
                )
                nn_log = {
                    "val/mean_nn_dist": cr["val/mean_nn_dist"],
                    "val/copy_ratio":   cr["val/copy_ratio"],
                    "trainer/global_step": self.global_step,
                }
                msg = (f"[copy_ratio] epoch {self.current_epoch} | "
                       f"mean_nn_dist={cr['val/mean_nn_dist']:.4f} | "
                       f"copy_ratio={cr['val/copy_ratio']:.3f}")
                if self._val_feats_cache is not None:
                    # Reference: how far REAL validation scenes are from the training scenes.
                    ref = mean_nn_distance(self._val_feats_cache, self._train_feats_cache)
                    nn_log["val/real_mean_nn_dist"] = ref
                    msg += f" | real_val_mean_nn_dist={ref:.4f}"
                logger.experiment.log(nn_log)
                print(msg)

            # --- render images ---
            generated_panels = []
            real_panels      = []

            def to_grid(images):
                imgs = images[0]
                imgs = torch.nan_to_num(imgs, nan=0.0).clamp(0, 1)
                imgs = (imgs * 255).byte().cpu().numpy()
                row  = np.concatenate([imgs[v] for v in range(n_views)], axis=2)
                return row.transpose(1, 2, 0)

            for scene_i in range(n_scenes):
                d_idx = self._val_index(scene_i)
                v_idxs, n_in = self._val_views(d_idx, view_idxs)
                scene_data = _scene_batch(self.val_dataset, d_idx, v_idxs, device)
                ray_o, ray_d = self.autoencoder.process_data.compute_rays(
                    scene_data["c2w"], scene_data["fxfycxcy"],
                    h=image_h_w[0], w=image_h_w[1], device=device,
                )

                # generated
                z_i   = z_gen[scene_i:scene_i+1]
                pos_i = pos_gen[scene_i:scene_i+1]
                gen_images = self.render(z_i, pos_i, ray_o, ray_d, image_h_w)

                # real encoded
                with torch.no_grad():
                    from easydict import EasyDict as edict
                    # the encoder sees the first n_in cameras only (all of them for the gravity data)
                    enc_ray_o, enc_ray_d = ray_o[:, :n_in], ray_d[:, :n_in]
                    input_enc = edict({
                        "image": scene_data["image"][:, :n_in], "ray_o": enc_ray_o,
                        "ray_d": enc_ray_d, "image_h_w": image_h_w,
                    })
                    z_real, pos_real, _ = self.autoencoder.encode(input_enc)
                real_images = self.render_raw(z_real, pos_real, ray_o, ray_d, image_h_w)

                generated_panels.append(wandb.Image(
                    to_grid(gen_images),
                    caption=f"generated scene {scene_i} | epoch {self.current_epoch}"
                ))
                real_panels.append(wandb.Image(
                    to_grid(real_images),
                    caption=f"real scene {scene_i} | epoch {self.current_epoch}"
                ))

            logger.experiment.log({
                "val/generated_scenes": generated_panels,
                "val/real_scenes":      real_panels,
                "val/render_epoch":     self.current_epoch,
            })

        except Exception as e:
            print(f"[warn] validation rendering skipped: {e}")

    def _val_index(self, i):
        """Dataset index of the i-th validation scene shown. With frames_per_episode (libero shards) consecutive indices
        are frames of one episode, so step over a whole episode to show different episodes; gravity: the index itself."""
        ds = self.val_dataset
        stride = int(getattr(ds, "frames_per_episode", 0) or 0) or 1
        return (int(i) * stride) % len(ds)

    def _val_views(self, d_idx, default):
        """(camera indices, number of encoder inputs) for validation scene d_idx. Gravity: the fixed list, all of it is
        input. Libero shards: the num_views cameras of the episode with the num_input_views most spread-out viewing
        directions first (the first ones are the encoder's input, the rest are extra cameras to render)."""
        ds = self.val_dataset
        if hasattr(ds, "spread_views") and hasattr(ds, "episode_id"):
            tr = ds.config.training
            n_in, n_all = int(tr.num_input_views), int(tr.num_views)
            row = ds._row_for(int(d_idx))
            order = ds.spread_views(int(ds.episode_id[row]), n_in)
            return list(order[:n_all]), n_in
        return list(default), len(default)

    @torch.no_grad()
    def _val_cond(self, n, view_idxs, device):
        """StartCond for n validation scenes: starting positions from the input cameras (view_idxs) of validation
        scene i, computed from the rays alone. Cached."""
        if self._val_start_cache is None or self._val_start_cache.shape[0] != n:
            ss = []
            for i in range(n):
                d_idx = self._val_index(i)
                v_idxs, n_in = self._val_views(d_idx, view_idxs)
                sd = _scene_batch(self.val_dataset, d_idx, v_idxs, device)
                ray_o, ray_d = self.autoencoder.process_data.compute_rays(
                    sd["c2w"][:, :n_in], sd["fxfycxcy"][:, :n_in], h=128, w=128, device=device)
                ss.append(self.start_from_rays(ray_o, ray_d))
            self._val_start_cache = torch.cat(ss)
        return StartCond(self._val_start_cache)

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, num_scenes: int, sampler: str = "fm_ode", start=None):
        """Generate num_scenes latents (flow space). start: for a flow conditioned on the starting positions, a StartCond
        or a tensor [num_scenes, N, 3] (flow space, e.g. start_from_rays of validation scenes' cameras); None runs without
        them, which needs a model trained with start_dropout > 0. Ignored by a model without conditioning."""
        self.eval()
        cond = self.as_cond(start, num_scenes, next(self.net.parameters()).device)
        if cond is not None and len(cond) != num_scenes:
            raise ValueError(f"start has {len(cond)} scenes, expected {num_scenes}")
        compiled_net = self.net
        if hasattr(self.net, "_orig_mod"):
            self.net = self.net._orig_mod
        try:
            if sampler == "fm_ode":
                return self._sample_fm_ode(num_scenes, cond)
            elif sampler == "edm":
                if self.time_coupling == "decoupled":
                    raise ValueError("The 'edm' sampler uses one time for positions and features; "
                                     "use 'fm_ode' with time_coupling 'decoupled'.")
                return self._sample_edm(num_scenes, cond)
            else:
                raise ValueError(f"Unknown sampler '{sampler}'. Choose 'fm_ode' or 'edm'.")
        finally:
            self.net = compiled_net

    def _sample_fm_ode(self, num_scenes: int, start=None):
        num_steps = self.config.diffusion.num_steps
        order     = self.config.diffusion.get("fm_ode_order", 1)
        device    = next(self.net.parameters()).device

        x   = torch.randn(num_scenes, self.n_tokens, self.scalar_dim, device=device)
        pos = torch.randn(num_scenes, self.n_tokens, 3, device=device)
        if self.time_coupling == "decoupled":
            return self._integrate_decoupled(x, pos, num_steps, start=start)

        eps = 1e-4
        ts  = torch.linspace(eps, 1.0, num_steps + 1, device=device)
        dt  = ts[1] - ts[0]

        for i in range(num_steps):
            t_now  = ts[i].expand(num_scenes)
            t_next = ts[i + 1].expand(num_scenes)

            vx, vp = self._velocity(x, pos, t_now, start=start)
            vx = vx.clamp(-10, 10)
            vp = vp.clamp(-10, 10)

            if order == 1:
                x   = x   + dt * vx
                pos = pos + dt * vp
            else:
                x_pred   = (x   + dt * vx).clamp(-50, 50)
                pos_pred = (pos + dt * vp).clamp(-50, 50)
                vx2, vp2 = self._velocity(x_pred, pos_pred, t_next, start=start)
                vx2 = vx2.clamp(-10, 10)
                vp2 = vp2.clamp(-10, 10)
                x   = x   + dt * 0.5 * (vx  + vx2)
                pos = pos + dt * 0.5 * (vp  + vp2)

        return x, self._state_to_pos(pos, start)

    def _integrate_decoupled(self, x, pos, num_steps, start=None):
        """Euler sampler with separate position and feature times (time_coupling 'decoupled').
        Features follow the usual uniform grid t_f. Positions follow t_p = pos_speedup * t_f; at the step where
        t_p would pass 1 / (1 + pos_clean_sigma) they jump to their clean estimate and stay fixed, and the network is
        told t_p = 1 / (1 + pos_clean_sigma) from then on (the situation it saw in training with clean positions).
        The velocity clamp of the joint sampler is kept.
        pos is the position STATE (see _forward_net); the returned positions are converted back with start."""
        start  = self.as_cond(start, x.shape[0], x.device)
        B      = x.shape[0]
        device = x.device
        eps    = 1e-4
        ts     = torch.linspace(eps, 1.0, num_steps + 1, device=device)
        dt     = ts[1] - ts[0]
        t_cap  = torch.tensor(self._t_clean, device=device)
        lam    = self.pos_speedup
        pos_done = False
        for i in range(num_steps):
            t_now  = ts[i]
            tp_now = t_cap if pos_done else torch.minimum(lam * t_now, t_cap)
            vx, vp = self._velocity(x, pos, t_now.expand(B), tp_now.expand(B), start=start)
            vx = vx.clamp(-10, 10)
            vp = vp.clamp(-10, 10)
            x = x + dt * vx
            if not pos_done:
                tp_next = lam * ts[i + 1]
                if tp_next >= t_cap:
                    pos = pos + (1 - tp_now) * vp          # land on the clean position estimate
                    pos_done = True
                else:
                    pos = pos + (tp_next - tp_now) * vp
        return x, self._state_to_pos(pos, start)

    def _sample_edm(self, num_scenes: int, start=None):
        num_steps = self.config.diffusion.num_steps
        sigma_min = self.config.diffusion.sigma_min
        sigma_max = self.config.diffusion.sigma_max
        rho       = self.config.diffusion.rho
        S_churn   = self.config.diffusion.get("S_churn", 0.0)
        S_noise   = self.config.diffusion.get("S_noise", 1.0)
        device    = next(self.net.parameters()).device

        ramp   = torch.linspace(0, 1, num_steps, device=device)
        sigmas = (sigma_max ** (1/rho) + ramp * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

        x   = torch.randn(num_scenes, self.n_tokens, self.scalar_dim, device=device) * sigma_max
        pos = torch.randn(num_scenes, self.n_tokens, 3, device=device) * sigma_max

        for i in range(num_steps):
            sigma      = sigmas[i]
            sigma_next = sigmas[i + 1]

            if S_churn > 0 and sigma_next > 0:
                gamma     = min(S_churn / num_steps, 2**0.5 - 1)
                sigma_hat = sigma * (1 + gamma)
                x   = x   + torch.randn_like(x)   * S_noise * (sigma_hat**2 - sigma**2).sqrt()
                pos = pos + torch.randn_like(pos) * S_noise * (sigma_hat**2 - sigma**2).sqrt()
            else:
                sigma_hat = sigma

            t_hat  = (1.0 / (1.0 + sigma_hat)).expand(num_scenes)
            t_next = (1.0 / (1.0 + sigma_next)).expand(num_scenes)

            vx, vp   = self._velocity(x, pos, t_hat, start=start)
            t_hat_   = t_hat[:, None, None]
            dx       = x   + (1 - t_hat_) * vx
            dpos     = pos + (1 - t_hat_) * vp
            dt_sigma = sigma_next - sigma_hat
            x   = x   + (x   - dx  ) / sigma_hat * dt_sigma
            pos = pos + (pos - dpos) / sigma_hat * dt_sigma

            if sigma_next > 0:
                vx2, vp2 = self._velocity(x, pos, t_next, start=start)
                t_next_  = t_next[:, None, None]
                dx2      = x   + (1 - t_next_) * vx2
                dpos2    = pos + (1 - t_next_) * vp2
                x   = x   + (x   - dx  ) / sigma_hat * dt_sigma * 0.5 \
                          + (x   - dx2 ) / sigma_next * dt_sigma * 0.5
                pos = pos + (pos - dpos ) / sigma_hat * dt_sigma * 0.5 \
                          + (pos - dpos2) / sigma_next * dt_sigma * 0.5

        return x, self._state_to_pos(pos, start)

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _decode(self, z_feat, pos_z, ray_o, ray_d, image_h_w):
        if self.autoencoder.latent_mode == "invariant":
            from platonic_transformers.models.platoformer.io import lift_scalars
            z_lift = lift_scalars(z_feat, self.autoencoder.group).flatten(-2)
        else:
            z_lift = z_feat
        return self.autoencoder.decode_views(z_lift, pos_z, ray_o, ray_d, image_h_w)

    @torch.no_grad()
    def render(self, z_feat, pos_z, ray_o, ray_d, image_h_w):
        """Decode GENERATED (normalised-space) latents → images."""
        if self.normaliser is not None:
            pos_z, z_feat = self.normaliser.denormalise(pos_z, z_feat)
        return self._decode(z_feat, pos_z, ray_o, ray_d, image_h_w)

    @torch.no_grad()
    def render_raw(self, z_feat, pos_z, ray_o, ray_d, image_h_w):
        """Decode RAW AE-space latents → images (no denormalisation)."""
        return self._decode(z_feat, pos_z, ray_o, ray_d, image_h_w)

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            make_param_groups(self, self.config.optimizer.weight_decay),
            lr=self.config.optimizer.lr,
        )
        if self.config.scheduler.use_cosine:
            T_max = self.config.scheduler.get("T_max_epochs", None) or self.trainer.max_epochs
            return {"optimizer": optimizer, "lr_scheduler": CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=self.config.optimizer.lr / 100,
            )}
        return optimizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ovs_sets(DatasetClass, config, ae_config):
    """The gravity (psplat.data.ovs.Dataset) train and validation sets: unchanged from before."""
    train_manifest = os.path.join(config.dataset.data_dir, "train", "manifest.json")
    val_manifest   = os.path.join(config.dataset.data_dir, "val",   "manifest.json")

    dataset_config = ml_collections.ConfigDict({
        "training": {
            "dataset_name":      config.dataset.dataset_name,
            "dataset_path":      train_manifest,
            "num_views":         10,
            "num_input_views":   5,
            "num_target_views":  5,
            "holdout_views":     2,
            "pose_norm":         "lvsm_mean",
            "scene_scale_factor": 1.35,
            "square_crop":       False,
            "rot_aug":           False,
            "rot_aug_so3":       False,
            "target_has_input":  False,
        },
        "model": {
            "image_tokenizer": {
                "image_size": ae_config.model.image_tokenizer.image_size,
                "patch_size": ae_config.model.image_tokenizer.patch_size,
            }
        },
        "inference": {"if_inference": False},
    })

    train_set = DatasetClass(dataset_config)

    val_config = ml_collections.ConfigDict(dataset_config.to_dict())
    val_config.training.dataset_path = val_manifest
    val_set = DatasetClass(val_config)
    return train_set, val_set


def _scene_batch(ds, idx, views, device):
    """ds.scene_batch for any dataset class: ovs names its first argument scene_idx, libero_shards names it i."""
    import inspect
    first = list(inspect.signature(ds.scene_batch).parameters)[0]
    return ds.scene_batch(**{first: int(idx)}, view_indices=list(views), device=device)


def _libero_sets(DatasetClass, config, ae_config):
    """Train and validation sets of psplat.data.libero_shards.Dataset.

    Everything that defines what a sample is (views per sample, pose normalisation, camera set, frames per episode,
    split) is taken from the AUTOENCODER's own training config, so the flow sees the data the autoencoder was trained
    on; any of these keys can be overridden under dataset: in the flow config. dataset.data_dir is the shard directory
    (the one with index.json and shard_*.h5). Train uses split=train with random frames and random cameras; validation
    uses split=val with deterministic frames (if_inference) so the same scenes are shown every time.
    """
    d, atr = config.dataset, ae_config.training

    def pick(key, default=None):
        v = d.get(key, None)
        return v if v is not None else atr.get(key, default)

    def one(split, inference):
        return ml_collections.ConfigDict({
            "training": {
                "dataset_name":       str(d.dataset_name),
                "dataset_path":       str(d.data_dir),
                "num_views":          int(pick("num_views")),
                "num_input_views":    int(pick("num_input_views")),
                "num_target_views":   int(pick("num_target_views")),
                "pose_norm":          str(pick("pose_norm", "lvsm_mean")),
                "scene_scale_factor": float(pick("scene_scale_factor", 1.35)),
                "include_wrist":      bool(pick("include_wrist", False)),
                "frames_per_episode": int(pick("frames_per_episode", 16)),
                "skip_first":         int(pick("skip_first", 0)),
                "split":              split,
                "split_by":           str(pick("split_by", "episode")),
                "holdout_frac":       float(pick("holdout_frac", 0.1)),
                "test_frac":          float(pick("test_frac", 0.0)),
                "holdout_tasks":      [str(x) for x in (pick("holdout_tasks", []) or [])],
                "view_select":        str(d.get("view_select", "first")),
                "square_crop":        bool(pick("square_crop", False)),
                "rot_aug":            False,
                "rot_aug_so3":        False,
                "target_has_input":   False,
            },
            "model": {
                "image_tokenizer": {
                    "image_size": ae_config.model.image_tokenizer.image_size,
                    "patch_size": ae_config.model.image_tokenizer.patch_size,
                }
            },
            "inference": {"if_inference": bool(inference)},
        })

    train_set = DatasetClass(one("train", False))
    val_set   = DatasetClass(one("val", True))
    n_in, n_all = int(pick("num_input_views")), int(pick("num_views"))
    print(f"[data] libero shards: {len(train_set)} train items, {len(val_set)} validation items; "
          f"{n_all} cameras per sample, the first {n_in} are the encoder inputs", flush=True)
    return train_set, val_set


def load_data(config: ml_collections.ConfigDict, ae_config: ml_collections.ConfigDict):
    import importlib
    module_path, class_name = config.dataset.dataset_name.rsplit(".", 1)
    DatasetClass = getattr(importlib.import_module(module_path), class_name)

    if "libero_shards" in str(config.dataset.dataset_name):
        train_set, val_set = _libero_sets(DatasetClass, config, ae_config)
    else:
        train_set, val_set = _ovs_sets(DatasetClass, config, ae_config)

    if config.dataset.dataset_fraction < 1.0:
        subset_len = int(len(train_set) * config.dataset.dataset_fraction)
        train_set = torch.utils.data.Subset(
            train_set, torch.randperm(len(train_set))[:subset_len]
        )

    # repeats_per_epoch: wrap train_set so each scene appears N times per epoch
    # with different random view subsets each time (view augmentation)
    repeats = config.dataset.get("repeats_per_epoch", 1)
    if repeats > 1:
        train_set = RepeatedDataset(train_set, repeats)
        print(f"[data] repeats_per_epoch={repeats} → {len(train_set)} items/epoch")

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    # val_stride > 1: validate on every val_stride-th item only (a fixed subset, same every epoch). Lightning validates the
    # whole loader every epoch; the libero validation split has about 3000 items (the gravity one 64 scenes), which would
    # cost more reading than the training epoch itself. A stride that does not divide frames_per_episode (16) takes
    # different frames of every episode. The dataset object stays whole: rendering picks scenes through it.
    val_stride = int(config.dataset.get("val_stride", 1) or 1)
    val_sampler = list(range(0, len(val_set), val_stride)) if val_stride > 1 else None
    if val_sampler is not None:
        print(f"[data] validation: every {val_stride}th item -> {len(val_sampler)} of {len(val_set)} items", flush=True)
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: ml_collections.ConfigDict) -> None:
    print_config(config, "NVS Flow Matching Configuration")
    torch.set_float32_matmul_precision(config.system.get("float32_matmul_precision", "high"))
    pl.seed_everything(config.seed)

    # Load frozen autoencoder
    ae_ckpt_dir    = config.autoencoder.checkpoint_dir
    ae_config_path = os.path.join(ae_ckpt_dir, "config.yaml")
    ae_config      = load_with_defaults(dataset_config=ae_config_path, cli_args=[])
    autoencoder    = PlatonicAutoencoder(ae_config)
    autoencoder.load_ckpt(ae_ckpt_dir)
    autoencoder.eval()
    print(f"Loaded autoencoder from {ae_ckpt_dir}")

    # Data
    train_loader, val_loader = load_data(config, ae_config)

    # Normaliser (optional): fixed global statistics, fitted once. If dataset.normaliser_path is
    # set, the statistics are saved there and reused by later runs with the same autoencoder.
    norm_passes = config.dataset.get("norm_passes", 0)
    norm_path   = config.dataset.get("normaliser_path", None)
    normaliser  = None
    if norm_passes > 0:
        if norm_path and os.path.isfile(norm_path):
            normaliser = load_normaliser(norm_path, ae_ckpt_dir)
            print(f"[normaliser] loaded fixed statistics from {norm_path}")
        else:
            normaliser = fit_normaliser(
                autoencoder, train_loader,
                n_passes=norm_passes,
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_items=int(config.dataset.get("norm_max_items", 0) or 0),
            )
            if norm_path:
                save_normaliser(normaliser, norm_path, ae_ckpt_dir)
                print(f"[normaliser] saved fixed statistics to {norm_path}")

    # Model
    model = NVSFMModel(config, autoencoder)
    if normaliser is not None:
        model.set_normaliser(normaliser)
    model.set_val_dataset(val_loader.dataset)
    model.set_train_loader(train_loader)   # for copy_ratio cache
    model.set_val_loader(val_loader)       # for the real-validation distance reference

    # Trainer
    accelerator = "gpu" if config.system.gpus > 0 and torch.cuda.is_available() else "cpu"
    devices     = config.system.gpus if accelerator == "gpu" else "auto"

    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            entity=config.logging.get("wandb_identity", None),
            config=config.to_dict(),
            save_dir=save_dir,
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            every_n_epochs=config.training.validation_frequency,
            save_last=config.checkpoint.save_last,
        ),
        TimerCallback(),
    ]
    # Extra checkpoints every N epochs (never overwritten), so a run can be inspected at several stages.
    snapshot_every = int(config.checkpoint.get("snapshot_every_n_epochs", 50))
    if snapshot_every > 0:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            filename="snapshot-epoch{epoch:04d}-step{step}",
            auto_insert_metric_name=False,
            every_n_epochs=snapshot_every,
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ))
    if config.training.get("ema_enabled", False):
        callbacks.append(EMACallback(
            decay=config.training.get("ema_decay", 0.999),
            warmup_steps=config.training.get("ema_warmup_steps", 2000),
        ))
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=config.system.precision,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        # optional fixed epoch length (the libero shards have tens of thousands of samples per pass, the gravity data 512 scenes)
        limit_train_batches=(int(config.training.get("limit_train_batches", 0)) or 1.0),
    )

    test_ckpt   = config.testing.get("test_ckpt", None)
    resume_ckpt = config.testing.get("resume_ckpt", None)

    if test_ckpt is None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
    else:
        trainer.test(model, val_loader, ckpt_path=test_ckpt)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/nvs_fm.yaml")
    args, unknown_args = parser.parse_known_args()
    config = load_with_defaults(dataset_config=args.config, cli_args=unknown_args)
    main(config)