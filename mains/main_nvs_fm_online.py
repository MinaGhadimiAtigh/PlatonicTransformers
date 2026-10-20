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
"""

import os
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


class NVSFMModel(pl.LightningModule):
    """Node-wise conditional flow matching model for NVS scene latents."""

    def __init__(self, config, autoencoder):
        """
        Args:
            config: ml_collections.ConfigDict — same structure as QM9FM config.
            autoencoder: frozen PlatonicAutoencoder instance — used to encode
                         raw image batches into (z_feat, pos_z) during training.
        """
        super().__init__()
        self.config = config

        # --- Frozen encoder ---
        # We keep the full autoencoder so we can call .process_data() and .encode()
        # in training_step. All parameters are frozen — no gradients flow into it.
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        # --- Dimensions (read from the autoencoder's own config) ---
        # n_reg comes from autoencoder config.model.registers.n_reg
        # d     comes from autoencoder config.model.transformer.d
        # With trivial group (G=1), invariant mode gives d/G = d, so scalar_dim = d
        ae_cfg = autoencoder.config
        self.n_tokens = int(ae_cfg.model.registers.n_reg)
        ae_group_name = ae_cfg.model.equivariance.solid_name.lower()
        ae_group = PLATONIC_GROUPS[ae_group_name]
        self.scalar_dim = int(ae_cfg.model.transformer.d) // ae_group.G

        # --- Flow matching network ---
        # Mirrors QM9FMModel.net but with:
        #   input_dim  = scalar_dim + 1  (384 features + 1 c_noise time embedding)
        #   output_dim = scalar_dim      (384, predicts velocity in feature space)
        #   output_dim_vec = 1           (one 3D vector per token, velocity in position space)
        #   dense_mode = True            (input is (B, 256, D), not packed flat)
        solid_name = config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unsupported solid_name '{solid_name}'")

        self.net = PlatonicTransformer(
            input_dim=self.scalar_dim + 1,       # 385: features + c_noise
            input_dim_vec=0,                      # no vector input
            hidden_dim=config.model.hidden_dim,
            output_dim=self.scalar_dim,           # 384: velocity in feature space
            output_dim_vec=1,                     # velocity in position space (3D)
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=True,                      # (B, 256, D) format, no batch index
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

        # Random SO(3) rotation for data augmentation during training
        self.rotation_generator = RandomSOd(3)

    def _sample_t(self, batch_size, device):
        """Sample one flow-matching time t per scene.

        Identical to QM9FMModel: uses EDM's log-normal sigma schedule mapped
        to t via t = 1/(1+sigma), focusing training on intermediate timesteps.

        Returns shape (B,) — one t per scene, not per token.
        """
        rnd_normal = torch.randn(batch_size, device=device)
        sigma = (rnd_normal * self.config.diffusion.P_std + self.config.diffusion.P_mean).exp()
        return 1.0 / (1.0 + sigma)

    def _edm_weight(self, t):
        """Per-token loss weight = 1/c_out(t)^2.

        Identical to QM9FMModel. Bounded in [0.5, 1] — never explodes.
        t has shape (B, 1, 1) when called from training_step, broadcasting
        over the (B, n_tokens, scalar_dim) feature tensor.
        """
        sigma_data = self.config.diffusion.sigma_data
        return (t ** 2 * sigma_data ** 2 + (1 - t) ** 2) / sigma_data ** 2

    def _velocity(self, x_t, pos_t, t):
        """EDM-preconditioned flow-matching velocity field.

        Same math as QM9FMModel._velocity, adapted for dense (B, N, D) format.

        Args:
            x_t:   (B, n_tokens, scalar_dim)  — interpolated features at time t
            pos_t: (B, n_tokens, 3)           — interpolated positions at time t
            t:     (B,)                        — one time value per scene

        Returns:
            v_x:   (B, n_tokens, scalar_dim)  — predicted velocity in feature space
            v_pos: (B, n_tokens, 3)           — predicted velocity in position space
        """
        sigma_data = self.config.diffusion.sigma_data

        # t is (B,) → reshape to (B, 1, 1) so it broadcasts over (B, n_tokens, D)
        # In QM9FM this was done via t[batch] to expand per-atom;
        # here dense_mode means we just unsqueeze twice.
        t_ = t[:, None, None]                                        # (B, 1, 1)

        denom  = t_ ** 2 * sigma_data ** 2 + (1 - t_) ** 2          # (B, 1, 1)
        c_in   = 1.0 / denom.sqrt()                                  # normalize input
        c_skip = (t_ * sigma_data ** 2 - (1 - t_)) / denom          # linear baseline
        c_out  = sigma_data / denom.sqrt()                           # residual scale
        # c_noise: encode t as log SNR, appended to features as time embedding
        # shape (B, n_tokens, 1) so it can be cat'd along the feature dim
        c_noise = (((1 - t_) / t_).log() / 4).expand(
            x_t.shape[0], x_t.shape[1], 1                           # (B, n_tokens, 1)
        )

        # Normalize inputs to unit variance
        x_in   = c_in * x_t                                          # (B, n_tokens, scalar_dim)
        pos_in = c_in * pos_t                                         # (B, n_tokens, 3)

        # Append time embedding to scalar features
        scalar_in = torch.cat([x_in, c_noise], dim=-1)               # (B, n_tokens, scalar_dim+1)

        # Network call — dense_mode=True, so batch=None
        scalar_out, vector_out = self.net(
            scalar_in, pos_in, batch=None,
            avg_num_nodes=self.config.diffusion.avg_num_nodes,
        )
        # scalar_out: (B, n_tokens, scalar_dim)
        # vector_out: (B, n_tokens, 1, 3) → squeeze to (B, n_tokens, 3)

        # Blend linear baseline with network residual
        v_x   = c_skip * x_t   + c_out * scalar_out                  # (B, n_tokens, scalar_dim)
        v_pos = c_skip * pos_t + c_out * vector_out.squeeze(-2)       # (B, n_tokens, 3)

        return v_x, v_pos

    def _encode(self, batch):
        """Run the frozen autoencoder encoder on a raw image batch.

        Calls process_data() to compute rays from cameras, then encode()
        to get the scene latent. Runs under torch.no_grad() since the
        autoencoder is frozen.

        Args:
            batch: dict from ovs.Dataset with keys:
                   image (B, V, 3, H, W), c2w (B, V, 4, 4), fxfycxcy (B, V, 4), ...

        Returns:
            x1:   (B, n_tokens, scalar_dim)  — clean scene features  (our "data")
            pos1: (B, n_tokens, 3)           — clean token positions
        """
        with torch.no_grad():
            # process_data splits views and computes Plücker rays
            # input gets the first num_input_views views
            input, _ = self.autoencoder.process_data(
                batch,
                has_target_image=False,   # we don't need target images for encoding
                target_has_input=False,
                compute_rays=True,
            )
            # encode returns (z_feat, pos_z, pos_reg_init)
            # z_feat: (B, n_tokens, scalar_dim)
            # pos_z:  (B, n_tokens, 3)
            z_feat, pos_z, _ = self.autoencoder.encode(input)

        return z_feat, pos_z

    def training_step(self, batch, batch_idx):
        # --- Step 1: encode raw images → clean latent (x1, pos1) ---
        x1, pos1 = self._encode(batch)           # (B, n_tokens, scalar_dim), (B, n_tokens, 3)
        B = x1.shape[0]

        # --- Step 2: optional SO(3) augmentation on positions ---
        if self.config.training.train_augm:
            rot = self.rotation_generator().type_as(pos1)            # (3, 3)
            pos1 = torch.einsum("ij,bnj->bni", rot, pos1)           # rotate all tokens

        # --- Step 3: sample t — one per scene ---
        t = self._sample_t(B, x1.device)                            # (B,)

        # --- Step 4: sample noise (x0, pos0) ~ N(0, I) ---
        x0   = torch.randn_like(x1)                                 # (B, n_tokens, scalar_dim)
        pos0 = torch.randn_like(pos1)                               # (B, n_tokens, 3)

        # --- Step 5: interpolate to get (x_t, pos_t) ---
        t_ = t[:, None, None]                                       # (B, 1, 1) for broadcasting
        x_t   = (1 - t_) * x0   + t_ * x1                         # (B, n_tokens, scalar_dim)
        pos_t = (1 - t_) * pos0 + t_ * pos1                        # (B, n_tokens, 3)

        # --- Step 6: predict velocity ---
        vx, vp = self._velocity(x_t, pos_t, t)

        # --- Step 7: weighted MSE loss against true velocity (x1-x0, pos1-pos0) ---
        weight = self._edm_weight(t_)                               # (B, 1, 1)
        loss = (
            (weight * (vx - (x1 - x0)).square()).mean()
          + (weight * (vp - (pos1 - pos0)).square()).mean()
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, pos1 = self._encode(batch)
        B = x1.shape[0]
        t = self._sample_t(B, x1.device)
        t_ = t[:, None, None]
        x0, pos0 = torch.randn_like(x1), torch.randn_like(pos1)
        x_t   = (1 - t_) * x0   + t_ * x1
        pos_t = (1 - t_) * pos0 + t_ * pos1
        vx, vp = self._velocity(x_t, pos_t, t)
        weight = self._edm_weight(t_)
        loss = (
            (weight * (vx - (x1 - x0)).square()).mean()
          + (weight * (vp - (pos1 - pos0)).square()).mean()
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 logger=True, batch_size=B)
        return loss

    def set_val_dataset(self, val_dataset):
        """Store a reference to the val dataset so on_validation_epoch_end
        can access full scene cameras. Called from main() after model creation."""
        self.val_dataset = val_dataset

    def on_validation_epoch_end(self):
        """Render generated and real scenes and log to wandb.

        For each of n_scenes:
          - sample a new latent from noise → render from 5 val cameras
          - encode a real val scene        → render from same 5 val cameras
        Log both as image grids side by side so we can track quality over time.
        """
        # only run every validation_frequency epochs to save time
        freq = self.config.training.validation_frequency
        if (self.current_epoch + 1) % freq != 0:
            return

        # need wandb logger and val dataset
        if not hasattr(self, "val_dataset") or self.val_dataset is None:
            return
        logger = self.trainer.logger if self.trainer is not None else None
        if logger is None or not hasattr(logger, "experiment"):
            return

        try:
            import wandb
            import numpy as np

            n_scenes   = 4    # number of scenes to render
            view_idxs  = [0, 3, 6, 9, 12]   # 5 evenly spaced views from the 16 available
            n_views    = len(view_idxs)
            device     = next(self.net.parameters()).device
            sampler    = self.config.diffusion.get("sampler", "fm_ode")
            image_h_w  = (128, 128)

            # --- sample generated latents ---
            z_gen, pos_gen = self.sample(n_scenes, sampler=sampler)
            # z_gen:   (n_scenes, 256, 384)
            # pos_gen: (n_scenes, 256, 3)

            generated_panels = []   # will hold wandb.Image objects
            real_panels      = []

            for scene_i in range(n_scenes):
                # --- get cameras from a fixed val scene ---
                # use scene_i as scene index so we always use the same scenes
                scene_data = self.val_dataset.scene_batch(
                    scene_idx=scene_i,
                    view_indices=view_idxs,
                    device=device,
                )
                # scene_data shapes: image (1,5,3,H,W), c2w (1,5,4,4), fxfycxcy (1,5,4)

                # compute rays from cameras
                ray_o, ray_d = self.autoencoder.process_data.compute_rays(
                    scene_data["c2w"],
                    scene_data["fxfycxcy"],
                    h=image_h_w[0],
                    w=image_h_w[1],
                    device=device,
                )
                # ray_o: (1, n_views, 3, H, W)
                # ray_d: (1, n_views, 3, H, W)

                # --- render generated scene ---
                z_i   = z_gen[scene_i:scene_i+1]     # (1, 256, 384)
                pos_i = pos_gen[scene_i:scene_i+1]   # (1, 256, 3)
                # Normalize generated latents to unit std before decoding.
                # The untrained network produces out-of-distribution latents
                # (large values) that cause NaN in the decoder. After training
                # this normalization becomes a no-op since the FM model learns
                # to produce latents matching the real distribution (std ≈ 1.5).
                z_std = z_i.std().clamp(min=1e-6)
                z_i   = z_i / z_std
                gen_images = self.render(z_i, pos_i, ray_o, ray_d, image_h_w)
                # gen_images: (1, n_views, 3, H, W)

                # --- encode real scene and render ---
                # directly feed the 5 views into the encoder
                # scene_data already has image (1,5,3,H,W) and c2w (1,5,4,4)
                # so we can call encode() after computing rays manually
                with torch.no_grad():
                    enc_ray_o, enc_ray_d = self.autoencoder.process_data.compute_rays(
                        scene_data["c2w"],
                        scene_data["fxfycxcy"],
                        h=image_h_w[0],
                        w=image_h_w[1],
                        device=device,
                    )
                    from easydict import EasyDict as edict
                    input_enc = edict({
                        "image":     scene_data["image"],   # (1, 5, 3, H, W)
                        "ray_o":     enc_ray_o,             # (1, 5, 3, H, W)
                        "ray_d":     enc_ray_d,             # (1, 5, 3, H, W)
                        "image_h_w": image_h_w,
                    })
                    z_real, pos_real, _ = self.autoencoder.encode(input_enc)
                real_images = self.render(z_real, pos_real, ray_o, ray_d, image_h_w)
                # real_images: (1, n_views, 3, H, W)

                # --- convert to wandb image grid (one row = one scene, cols = views) ---
                # squeeze batch dim, clamp to [0,1], convert to uint8
                def to_grid(images):
                    # images: (1, n_views, 3, H, W)
                    imgs = images[0]
                    imgs = torch.nan_to_num(imgs, nan=0.0)  # replace nan with black
                    imgs = imgs.clamp(0, 1)                 # (n_views, 3, H, W)
                    imgs = (imgs * 255).byte().cpu().numpy() # (n_views, 3, H, W)
                    # concatenate views horizontally → (3, H, n_views*W)
                    row = np.concatenate([imgs[v] for v in range(n_views)], axis=2)
                    # to HWC for wandb
                    return row.transpose(1, 2, 0)           # (H, n_views*W, 3)

                generated_panels.append(wandb.Image(
                    to_grid(gen_images),
                    caption=f"generated scene {scene_i} | epoch {self.current_epoch}"
                ))
                real_panels.append(wandb.Image(
                    to_grid(real_images),
                    caption=f"real scene {scene_i} | epoch {self.current_epoch}"
                ))

            # log to wandb
            logger.experiment.log({
                "val/generated_scenes": generated_panels,
                "val/real_scenes":      real_panels,
                "val/render_epoch":     self.current_epoch,
            })

        except Exception as e:
            print(f"[warn] validation rendering skipped: {e}")

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, num_scenes: int, sampler: str = "fm_ode"):
        """Generate num_scenes scene latents from pure noise.

        Args:
            num_scenes: number of scenes to generate.
            sampler:    "fm_ode" — simple Euler/Heun ODE integration (default)
                        "edm"   — Karras sigma schedule with stochasticity

        Returns:
            z_feat: (num_scenes, n_tokens, scalar_dim)  — generated scene features
            pos_z:  (num_scenes, n_tokens, 3)           — generated token positions
        """
        self.eval()

        # torch.compile can produce NaN during inference with no_grad due to
        # kernel selection issues. We temporarily swap to the uncompiled network
        # for sampling and restore it afterwards.
        compiled_net = self.net
        if hasattr(self.net, "_orig_mod"):
            self.net = self.net._orig_mod

        try:
            if sampler == "fm_ode":
                return self._sample_fm_ode(num_scenes)
            elif sampler == "edm":
                return self._sample_edm(num_scenes)
            else:
                raise ValueError(f"Unknown sampler '{sampler}'. Choose 'fm_ode' or 'edm'.")
        finally:
            self.net = compiled_net

    def _sample_fm_ode(self, num_scenes: int):
        """Plain FM ODE sampler — Euler or Heun integration on a uniform time grid.

        Integrates dz/dt = v(z, t) from t=0 (noise) to t=1 (data) using
        num_steps uniform steps. Heun (2nd-order) is used when
        config.diffusion.fm_ode_order == 2, otherwise Euler (1st-order).

        Returns:
            z_feat: (num_scenes, n_tokens, scalar_dim)
            pos_z:  (num_scenes, n_tokens, 3)
        """
        num_steps = self.config.diffusion.num_steps
        order     = self.config.diffusion.get("fm_ode_order", 2)  # 1=Euler, 2=Heun
        device    = next(self.net.parameters()).device

        # Start from pure noise at t=0
        x   = torch.randn(num_scenes, self.n_tokens, self.scalar_dim, device=device)
        pos = torch.randn(num_scenes, self.n_tokens, 3, device=device)

        # Uniform time grid from eps → 1
        # We avoid t=0 exactly because c_noise = log((1-t)/t) → -inf at t=0,
        # causing NaN in _velocity. A small epsilon is numerically identical
        # to starting at pure noise but avoids the singularity.
        eps = 1e-4
        ts  = torch.linspace(eps, 1.0, num_steps + 1, device=device)  # (num_steps+1,)
        dt  = ts[1] - ts[0]                                            # scalar

        for i in range(num_steps):
            t_now  = ts[i].expand(num_scenes)     # (B,) — same t for all scenes in batch
            t_next = ts[i + 1].expand(num_scenes)

            # Velocity at current point
            # clamp to prevent NaN explosion from untrained network
            vx, vp = self._velocity(x, pos, t_now)
            vx  = vx.clamp(-10, 10)
            vp  = vp.clamp(-10, 10)

            if order == 1:
                # Euler step
                x   = x   + dt * vx
                pos = pos + dt * vp

            else:
                # Heun step (2nd-order):
                # 1. Euler predictor to t_next
                x_pred   = x   + dt * vx
                pos_pred = pos + dt * vp
                # clamp predictor before second velocity call —
                # an untrained network can produce large velocities that push
                # x_pred far out of distribution, causing NaN in _velocity.
                # After training this clamp is effectively never active.
                x_pred   = x_pred.clamp(-50, 50)
                pos_pred = pos_pred.clamp(-50, 50)
                # 2. Velocity at predicted point
                vx2, vp2 = self._velocity(x_pred, pos_pred, t_next)
                vx2 = vx2.clamp(-10, 10)
                vp2 = vp2.clamp(-10, 10)
                # 3. Corrector: average of both velocities
                x   = x   + dt * 0.5 * (vx  + vx2)
                pos = pos + dt * 0.5 * (vp  + vp2)

        return x, pos

    def _sample_edm(self, num_scenes: int):
        """EDM-style stochastic sampler — Karras sigma schedule + Heun correction.

        Converts between EDM's sigma and FM's t via t = 1/(1+sigma),
        and uses the Karras et al. noise schedule for better sample quality.
        Optionally adds stochasticity via S_churn.

        Returns:
            z_feat: (num_scenes, n_tokens, scalar_dim)
            pos_z:  (num_scenes, n_tokens, 3)
        """
        num_steps = self.config.diffusion.num_steps
        sigma_min = self.config.diffusion.sigma_min
        sigma_max = self.config.diffusion.sigma_max
        rho       = self.config.diffusion.rho
        S_churn   = self.config.diffusion.get("S_churn", 0.0)
        S_noise   = self.config.diffusion.get("S_noise", 1.0)
        device    = next(self.net.parameters()).device

        # Karras sigma schedule: sigmas[0]=sigma_max ... sigmas[-1]=sigma_min
        ramp   = torch.linspace(0, 1, num_steps, device=device)
        sigmas = (sigma_max ** (1/rho) + ramp * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])  # append 0 at the end

        # Start from pure noise scaled by sigma_max
        x   = torch.randn(num_scenes, self.n_tokens, self.scalar_dim, device=device) * sigma_max
        pos = torch.randn(num_scenes, self.n_tokens, 3, device=device) * sigma_max

        for i in range(num_steps):
            sigma     = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Optional stochasticity: inject extra noise then step back
            # (Karras et al. Algorithm 2, lines 6-8)
            if S_churn > 0 and sigma_next > 0:
                gamma     = min(S_churn / num_steps, 2**0.5 - 1)
                sigma_hat = sigma * (1 + gamma)
                noise_x   = torch.randn_like(x)   * S_noise
                noise_pos = torch.randn_like(pos) * S_noise
                x   = x   + noise_x   * (sigma_hat**2 - sigma**2).sqrt()
                pos = pos + noise_pos * (sigma_hat**2 - sigma**2).sqrt()
            else:
                sigma_hat = sigma

            # Convert sigma → t for our FM velocity field
            # t = 1/(1+sigma) is the FM↔EDM correspondence
            t_hat  = (1.0 / (1.0 + sigma_hat )).expand(num_scenes)   # (B,)
            t_next = (1.0 / (1.0 + sigma_next)).expand(num_scenes)   # (B,)

            # Velocity at current sigma_hat
            vx, vp = self._velocity(x, pos, t_hat)

            # EDM denoiser estimate D from velocity:
            # D = x_t + (1-t)*v  →  the "clean" prediction
            t_hat_ = t_hat[:, None, None]
            dx   = x   + (1 - t_hat_) * vx
            dpos = pos + (1 - t_hat_) * vp

            # Euler step in sigma space
            dt_sigma = sigma_next - sigma_hat
            x   = x   + (x   - dx  ) / sigma_hat * dt_sigma
            pos = pos + (pos - dpos) / sigma_hat * dt_sigma

            # Heun 2nd-order correction (skip at last step)
            if sigma_next > 0:
                vx2, vp2 = self._velocity(x, pos, t_next)
                t_next_  = t_next[:, None, None]
                dx2   = x   + (1 - t_next_) * vx2
                dpos2 = pos + (1 - t_next_) * vp2
                x   = x   + (x   - dx  ) / sigma_hat * dt_sigma * 0.5 \
                          + (x   - dx2 ) / sigma_next * dt_sigma * 0.5
                pos = pos + (pos - dpos ) / sigma_hat * dt_sigma * 0.5 \
                          + (pos - dpos2) / sigma_next * dt_sigma * 0.5

        return x, pos

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def render(self, z_feat, pos_z, ray_o, ray_d, image_h_w):
        """Decode generated latents into images using the frozen autoencoder decoder.

        Args:
            z_feat:    (B, n_tokens, scalar_dim)  — from sample()
            pos_z:     (B, n_tokens, 3)           — from sample()
            ray_o:     (B, Vt, 3, H, W)           — target camera ray origins
            ray_d:     (B, Vt, 3, H, W)           — target camera ray directions
            image_h_w: (H, W)                     — output image resolution

        Returns:
            images: (B, Vt, 3, H, W)  — rendered images
        """
        # Re-lift invariant latent back to equivariant tokens for the decoder
        # (mirrors PlatonicAutoencoder.forward())
        if self.autoencoder.latent_mode == "invariant":
            from platonic_transformers.models.platoformer.io import lift_scalars
            z_lift = lift_scalars(z_feat, self.autoencoder.group).flatten(-2)
        else:
            z_lift = z_feat

        return self.autoencoder.decode_views(
            z_lift, pos_z, ray_o, ray_d, image_h_w,
        )

    def configure_optimizers(self):
        # make_param_groups separates weight decay from bias/norm parameters
        optimizer = torch.optim.AdamW(
            make_param_groups(self, self.config.optimizer.weight_decay),
            lr=self.config.optimizer.lr,
        )
        if self.config.scheduler.use_cosine:
            T_max = self.config.scheduler.get("T_max_epochs", None) or self.trainer.max_epochs
            return {"optimizer": optimizer, "lr_scheduler": CosineAnnealingLR(
                optimizer, T_max=T_max,
                eta_min=self.config.optimizer.lr / 100,
            )}
        return optimizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config: ml_collections.ConfigDict, ae_config: ml_collections.ConfigDict):
    """Load train and val ovs.Dataset instances and wrap in DataLoaders.

    Args:
        config:    FM model config (nvs_fm.yaml)
        ae_config: frozen autoencoder config — needed for model.image_tokenizer.*
                   which the dataset reads to resize images correctly.

    The dataset reads from three config sections:
        config.training.*              — view counts, pose norm, augmentation
        config.model.image_tokenizer.* — image_size and patch_size for resizing
        config.inference.*             — inference mode flags
    """
    import importlib
    module_path, class_name = config.dataset.dataset_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    DatasetClass = getattr(module, class_name)

    train_manifest = os.path.join(config.dataset.data_dir, "train", "manifest.json")
    val_manifest   = os.path.join(config.dataset.data_dir, "val",   "manifest.json")

    # Build dataset config with all keys the dataset actually reads.
    # model.image_tokenizer comes from the autoencoder config — it defines
    # the image_size and patch_size the encoder expects, so we must match it.
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
            "rot_aug":           False,      # ovs.Dataset rejects rot_aug=True
            "rot_aug_so3":       False,
            "target_has_input":  False,
        },
        # Taken directly from the autoencoder config so image sizes match exactly
        "model": {
            "image_tokenizer": {
                "image_size": ae_config.model.image_tokenizer.image_size,   # 128
                "patch_size": ae_config.model.image_tokenizer.patch_size,   # 8
            }
        },
        "inference": {
            "if_inference": False,
        },
    })

    train_set = DatasetClass(dataset_config)

    # Val uses identical config but points at val manifest
    val_config = ml_collections.ConfigDict(dataset_config.to_dict())
    val_config.training.dataset_path = val_manifest
    val_set = DatasetClass(val_config)

    if config.dataset.dataset_fraction < 1.0:
        subset_len = int(len(train_set) * config.dataset.dataset_fraction)
        train_set = torch.utils.data.Subset(
            train_set, torch.randperm(len(train_set))[:subset_len]
        )

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: ml_collections.ConfigDict) -> None:
    print_config(config, "NVS Flow Matching Configuration")
    torch.set_float32_matmul_precision(
        config.system.get("float32_matmul_precision", "high")
    )
    pl.seed_everything(config.seed)

    # --- Load frozen autoencoder ---
    # We need the autoencoder's own config to build it correctly,
    # then load the checkpoint weights.
    ae_ckpt_dir = config.autoencoder.checkpoint_dir
    ae_config_path = os.path.join(ae_ckpt_dir, "config.yaml")
    ae_config = load_with_defaults(dataset_config=ae_config_path, cli_args=[])
    autoencoder = PlatonicAutoencoder(ae_config)
    autoencoder.load_ckpt(ae_ckpt_dir)
    autoencoder.eval()
    print(f"Loaded autoencoder from {ae_ckpt_dir}")

    # --- Data ---
    train_loader, val_loader = load_data(config, ae_config)

    # --- Model ---
    model = NVSFMModel(config, autoencoder)
    # give model access to val dataset for camera-based rendering in on_validation_epoch_end
    model.set_val_dataset(val_loader.dataset)

    # --- Trainer setup ---
    if config.system.gpus > 0 and torch.cuda.is_available():
        accelerator, devices = "gpu", config.system.gpus
    else:
        accelerator, devices = "cpu", "auto"

    if config.logging.enabled:
        save_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "logs"
        )
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
            monitor=config.checkpoint.monitor,   # "val/loss"
            mode=config.checkpoint.mode,         # "min"
            every_n_epochs=config.training.validation_frequency,
            save_last=config.checkpoint.save_last,
        ),
        TimerCallback(),
    ]
    if config.training.get("ema_enabled", False):
        callbacks.append(EMACallback(
            decay=config.training.get("ema_decay", 0.999),
            warmup_steps=config.training.get("ema_warmup_steps", 2000),
        ))
    if config.logging.enabled:
        callbacks.append(
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        )

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
    )

    # --- Train or test ---
    test_ckpt = config.testing.get("test_ckpt", None)
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