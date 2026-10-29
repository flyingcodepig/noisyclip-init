"""One-encoder foreground alignment through the existing classifier geometry.

The auxiliary branch uses the existing head values with gradients detached;
the vision and post-layer normalization still receive alignment gradients.
The teacher is the stopped original prediction, not an additional model.
"""
from __future__ import annotations
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from scene.foreground_encoder import SceneFactorizer
from scene.token_partition import separate_scene


class ForegroundAlignment(SceneFactorizer):
    """Keep the original scene decision and add a training-only foreground readout."""

    def forward(self, images, background_pairing=None, *, return_diagnostics=False):
        output = self.trunk.vision_model(pixel_values=images, interpolate_pos_encoding=True)
        cls, foreground, background = separate_scene(
            output.last_hidden_state, self.focus_temperature.exp().clamp(.025, .35))
        foreground = F.layer_norm(foreground, (768,))
        background = F.layer_norm(background, (768,))
        foreground_gain = .12 * self.fg_level.sigmoid()
        background_gain = .02 + .06 * self.bg_level.sigmoid()
        scene = cls + foreground_gain * foreground - background_gain * background
        logits, descriptor, native = self._decision(scene)
        result = dict(scene_logits=logits, object_descriptor=descriptor, scene_native=native)
        if background_pairing is not None:
            exchanged = cls + foreground_gain * foreground - background_gain * background[background_pairing]
            result['exchanged_logits'] = self._decision(exchanged)[0]
        foreground_native = self.trunk.vision_model.post_layernorm(foreground)
        foreground_descriptor = F.normalize(
            self.trunk.visual_projection(foreground_native).float(), dim=-1)
        head = F.normalize(self.label_basis.detach(), dim=-1)
        scale = self.score_scale.detach().exp().clamp(1, 100)
        result['foreground_logits'] = scale * foreground_descriptor @ head.T
        if return_diagnostics:
            result['alignment_diagnostics'] = dict(
                foreground_native=foreground_native,
                foreground_descriptor=foreground_descriptor,
                final_token_states=output.last_hidden_state)
        return result


def confidence_weighted_alignment(main_logits, foreground_logits, temperature=2.):
    """Mean max-probability-weighted KL(main/T || foreground/T), scaled by T²."""
    if temperature != 2.:
        raise ValueError('The matched foreground comparison fixes temperature to 2')
    stopped = main_logits.detach().float()
    teacher_log = (stopped / temperature).log_softmax(-1)
    teacher_probability = teacher_log.exp()
    confidence = stopped.softmax(-1).amax(-1)
    foreground_log = (foreground_logits.float() / temperature).log_softmax(-1)
    divergence = (teacher_probability * (teacher_log - foreground_log)).sum(-1)
    loss = (temperature**2 * confidence * divergence).mean()
    return loss, dict(confidence=confidence, per_image_KL=divergence,
                     teacher_probability=teacher_probability)
