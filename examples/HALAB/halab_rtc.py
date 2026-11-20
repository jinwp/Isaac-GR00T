#!/usr/bin/env python3
"""Training-time RTC for the GR00T N1.7 action head (OpenHLM ``pi0_ha.py`` port).

Training-time action conditioning for real-time chunking (arXiv 2512.05964), as OpenHLM's
``ha_pi05_full`` trains it (``use_training_time_rtc=True``, ``rtc_max_delay=10``,
``rtc_mask_loss=True``):

* per sample a delay ``d ~ U{0, ..., max_delay - 1}``; the first ``d`` chunk rows were
  committed by the previous inference, so they stay clean and only the rest is noised
  (``d = 0`` keeps plain flow matching in the mix);
* the flow time is per frame: prefix rows sit at the clean end, the suffix at the
  sampled ``t``;
* the loss covers the suffix rows only.

GR00T's convention is the reverse of openpi's: ``x_t = (1 - t) * noise + t * a`` and
``v = a - noise``, so a clean row has ``t = 1`` (openpi: ``sigma = 0``).  The action
encoder takes the per-frame time (its sinusoidal encoding already handles ``(B, T)``;
only its input check is relaxed here).  The DiT's AdaLN conditioning is one timestep per
sample, so it keeps the suffix ``t``.

Inference: ``get_action(inputs, options={"halab_rtc_prefix": a, "halab_rtc_delay": d})``
pins the first ``d`` rows of ``a`` (normalized, ``(B, H, D)``) at the clean time on every
Euler step and generates the rest; without these options GR00T's own sampler runs
unchanged.  ``install`` patches the classes in this process (no Isaac-GR00T file edits).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.embodiment_conditioned_mlp import MultiEmbodimentActionEncoder, swish


CLEAN_T = 1.0  # GR00T flow time of a clean action


def _encoder_forward(self, actions, timesteps, cat_ids):
    """``MultiEmbodimentActionEncoder.forward`` that also takes per-frame ``(B, T)`` timesteps."""

    B, T, _ = actions.shape
    if timesteps.dim() == 1 and timesteps.shape[0] == B:
        timesteps = timesteps.unsqueeze(1).expand(-1, T)
    elif timesteps.shape != (B, T):
        raise ValueError(f"timesteps must be (B,) or (B, T)={B, T}, got {tuple(timesteps.shape)}")
    a_emb = self.W1(actions, cat_ids)
    tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
    x = swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1), cat_ids))
    return self.W3(x, cat_ids)


def _run_dit(head, sa_embs, vl_embeds, timestep, backbone_output, train: bool):
    """The DiT call exactly as GR00T's forward / sampler make it."""

    if train:
        kwargs = {"encoder_attention_mask": backbone_output.backbone_attention_mask, "return_all_hidden_states": True}
    else:
        kwargs = {}
    if head.config.use_alternate_vl_dit:
        kwargs.update(image_mask=backbone_output.image_mask, backbone_attention_mask=backbone_output.backbone_attention_mask)
    out = head.model(hidden_states=sa_embs, encoder_hidden_states=vl_embeds, timestep=timestep, **kwargs)
    return out[0] if train else out


def _embed_actions(head, noisy, t_frame_disc, embodiment_id, state_features):
    action_features = head.action_encoder(noisy, t_frame_disc, embodiment_id)
    if head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
        action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
    return torch.cat((state_features, action_features), dim=1)


def install(max_delay: int = 10, mask_loss: bool = True) -> None:
    """Patch the action head for training-time RTC and the prefix-pinned sampler."""

    if max_delay < 1:
        raise ValueError("max_delay must be >= 1")
    MultiEmbodimentActionEncoder.forward = _encoder_forward

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # GR00T ``Gr00tN1d7ActionHead.forward`` with the RTC prefix (see module docstring).
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)
        state_features = self.state_encoder(action_input.state, embodiment_id)
        if self.training and self.state_dropout_prob > 0:
            do_dropout = torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
            state_features = state_features * (1 - do_dropout[:, None, None].to(dtype=state_features.dtype))

        actions = action_input.action
        B, H, _ = actions.shape
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(B, device=actions.device, dtype=actions.dtype)  # (B,), suffix time
        delay = torch.randint(0, max_delay, (B,), device=actions.device)
        prefix = torch.arange(H, device=actions.device)[None, :] < delay[:, None]  # (B, H)
        t_frame = torch.where(prefix, torch.full_like(t[:, None].expand(B, H), CLEAN_T), t[:, None].expand(B, H))
        noisy_trajectory = (1 - t_frame[..., None]) * noise + t_frame[..., None] * actions
        velocity = actions - noise

        t_discretized = (t * self.num_timestep_buckets).long()
        t_frame_disc = (t_frame * self.num_timestep_buckets).long()
        sa_embs = _embed_actions(self, noisy_trajectory, t_frame_disc, embodiment_id, state_features)
        model_output = _run_dit(self, sa_embs, vl_embeds, t_discretized, backbone_output, train=True)
        pred_actions = self.action_decoder(model_output, embodiment_id)[:, -H:]

        action_mask = action_input.action_mask
        if mask_loss:
            action_mask = action_mask * (~prefix)[..., None].to(action_mask.dtype)
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,  # suffix rows only: the one-step metrics follow it
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    Gr00tN1d7ActionHead.forward = forward

    original_sampler = Gr00tN1d7ActionHead.get_action_with_features

    @torch.no_grad()
    def get_action_with_features(
        self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options=None
    ):
        if not options or "halab_rtc_prefix" not in options:
            return original_sampler(
                self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options
            )
        vl_embeds = backbone_features
        B, device = vl_embeds.shape[0], vl_embeds.device
        H = self.config.action_horizon
        delay = int(options["halab_rtc_delay"])
        prefix_actions = options["halab_rtc_prefix"].to(device=device, dtype=vl_embeds.dtype)[:, :H]
        pin = torch.arange(H, device=device)[None, :, None] < delay  # (1, H, 1)
        actions = torch.randn((B, H, self.action_dim), dtype=vl_embeds.dtype, device=device)
        actions = torch.where(pin, prefix_actions, actions)
        n = self.num_inference_timesteps
        for step in range(n):
            t_cont = step / float(n)
            t_disc = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((B,), t_disc, device=device)
            t_frame_disc = torch.where(
                pin[..., 0].expand(B, H),
                torch.full((B, H), int(CLEAN_T * self.num_timestep_buckets), device=device),
                torch.full((B, H), t_disc, device=device),
            )
            sa_embs = _embed_actions(self, actions, t_frame_disc, embodiment_id, state_features)
            model_output = _run_dit(self, sa_embs, vl_embeds, timesteps, backbone_output, train=False)
            pred_velocity = self.action_decoder(model_output, embodiment_id)[:, -H:]
            actions = torch.where(pin, prefix_actions, actions + pred_velocity / n)
        return BatchFeature(data={"action_pred": actions, "backbone_features": vl_embeds, "state_features": state_features})

    Gr00tN1d7ActionHead.get_action_with_features = get_action_with_features


def rtc_options(target: torch.Tensor, delay: int) -> dict[str, Any]:
    """``get_action`` options pinning the first ``delay`` rows of ``target`` (normalized actions)."""

    return {"halab_rtc_prefix": target, "halab_rtc_delay": int(delay)}
