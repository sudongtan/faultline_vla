"""Readout patching and activation capture for SmolVLA.

During inference the VLM runs once over the prefix and caches each layer's keys and values.
The action expert reads that cache at every layer on every denoising step. Replacing a cached
layer changes only what the expert reads there, leaving the VLM's own computation untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

IMAGE_TOKENS = 64
AGENTVIEW_TOKENS = slice(0, IMAGE_TOKENS)
ARM_DIMS = 6


@dataclass
class Prefill:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    pad_masks: torch.Tensor


def action_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Mean L2 distance over the chunk, on the 6 arm dimensions."""
    difference = prediction[:, :ARM_DIMS] - reference[:, :ARM_DIMS]
    return float(np.linalg.norm(difference, axis=1).mean())


class ReadoutPatcher:
    """Run the prefill and denoising separately so caches can be mixed."""

    def __init__(self, runner):
        self.runner = runner
        self.policy = runner.policy
        self.model = runner.policy.model
        self.backbone = self.model.vlm_with_expert

    @property
    def num_layers(self) -> int:
        return self.backbone.num_vlm_layers

    def _prefix(self, batch: dict):
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, image_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        return self.model.embed_prefix(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )

    @torch.inference_mode()
    def prefill(self, batch: dict) -> Prefill:
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        embeddings, pad_masks, attention_masks = self._prefix(batch)
        attention = make_att_2d_masks(pad_masks, attention_masks)
        positions = torch.cumsum(pad_masks, dim=1) - 1
        _, cache = self.backbone.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
        )
        return Prefill(
            keys=[layer.keys for layer in cache.layers],
            values=[layer.values for layer in cache.layers],
            pad_masks=pad_masks,
        )

    @torch.inference_mode()
    def decode(self, prefill: Prefill, noise: torch.Tensor) -> np.ndarray:
        """Denoise from a prefill and return the un-normalized chunk (steps, 7)."""
        from lerobot.policies.common.flow_matching import euler_integrate

        cache = DynamicCache()
        for index, (keys, values) in enumerate(zip(prefill.keys, prefill.values, strict=True)):
            cache.update(keys, values, index)
        actions = euler_integrate(
            lambda x_t, timestep: self.model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefill.pad_masks,
                past_key_values=cache,
                timestep=timestep,
            ),
            noise,
            self.policy.config.num_steps,
        )
        dimension = self.policy.config.action_feature.shape[0]
        return self.runner.postprocessor(actions[:, :, :dimension]).cpu().numpy()[0]

    @staticmethod
    def hybrid(
        base: Prefill, donor: Prefill, layers: Iterable[int], tokens: slice | None = None
    ) -> Prefill:
        """Copy ``base``, replacing the given layers with ``donor`` (optionally some tokens only)."""
        keys, values = list(base.keys), list(base.values)
        for layer in layers:
            if tokens is None:
                keys[layer] = donor.keys[layer]
                values[layer] = donor.values[layer]
                continue
            keys[layer] = keys[layer].clone()
            values[layer] = values[layer].clone()
            keys[layer][..., tokens, :] = donor.keys[layer][..., tokens, :]
            values[layer][..., tokens, :] = donor.values[layer][..., tokens, :]
        return Prefill(keys=keys, values=values, pad_masks=base.pad_masks)

    @torch.inference_mode()
    def prefill_with_connector(self, batch: dict, agentview: torch.Tensor) -> Prefill:
        """Prefill with the agentview connector output replaced, shaped (tokens, dim).

        The connector runs once per camera, agentview first, so only its first call is replaced.
        """
        connector = self.backbone.get_vlm_model().connector
        calls = []

        def hook(_module, _inputs, output):
            calls.append(None)
            if len(calls) > 1:
                return None
            value = (output[0] if isinstance(output, tuple) else output).clone()
            value[0] = agentview
            return (value, *output[1:]) if isinstance(output, tuple) else value

        handle = connector.register_forward_hook(hook)
        try:
            return self.prefill(batch)
        finally:
            handle.remove()

    @torch.inference_mode()
    def capture(self, batch: dict, vision_blocks: Iterable[int]) -> dict[str, torch.Tensor]:
        """Capture agentview activations along the pipeline, each shaped (tokens, dim)."""
        vision = self.backbone.get_vlm_model().vision_model.encoder.layers
        connector = self.backbone.get_vlm_model().connector
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def store(name):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                captured.setdefault(name, value[0].detach())

            return hook

        for block in vision_blocks:
            handles.append(vision[block].register_forward_hook(store(f"vision_{block:02d}")))
        handles.append(connector.register_forward_hook(store("connector")))
        try:
            prefill = self.prefill(batch)
        finally:
            for handle in handles:
                handle.remove()
        for layer, values in enumerate(prefill.values):
            tokens = values[0, :, AGENTVIEW_TOKENS, :]
            captured[f"vlm_{layer:02d}"] = tokens.permute(1, 0, 2).flatten(1)
        return captured


def layer_bins(num_layers: int, bin_size: int) -> list[list[int]]:
    return [list(range(i, min(i + bin_size, num_layers))) for i in range(0, num_layers, bin_size)]
