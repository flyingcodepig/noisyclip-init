"""One-encoder FG contribution from foreground-versus-background evidence.

The existing classifier reads both pooled regions. Its stopped probability
gap chooses how much FG enters the original scene representation. No extra
parameters, classifier, encoder or primary/foreground agreement filter is used.
"""
from __future__ import annotations
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from scene.foreground_encoder import SceneFactorizer
from scene.token_partition import separate_scene


class EvidenceRoutedScene(SceneFactorizer):
    def __init__(self, official, routing_strength=0., initialize_pretrained=True):
        if routing_strength not in (0., 1.):
            raise ValueError('This matched comparison permits routing strength0 or1')
        super().__init__(official, initialize_pretrained=initialize_pretrained)
        self.routing_strength = float(routing_strength)

    def forward(self, images, background_pairing=None, *, return_diagnostics=False):
        tokens = self.trunk.vision_model(pixel_values=images, interpolate_pos_encoding=True)
        cls, foreground, background = separate_scene(
            tokens.last_hidden_state, self.focus_temperature.exp().clamp(.025, .35))
        foreground = F.layer_norm(foreground, (768,))
        background = F.layer_norm(background, (768,))
        original_gain = .12 * self.fg_level.sigmoid()
        background_gain = .02 + .06 * self.bg_level.sigmoid()
        original_field = cls + original_gain * foreground - background_gain * background
        original_logits, original_descriptor, original_native = self._decision(original_field)

        foreground_native = self.trunk.vision_model.post_layernorm(foreground)
        foreground_descriptor = F.normalize(
            self.trunk.visual_projection(foreground_native).float(), dim=-1)
        head = F.normalize(self.label_basis.detach(), dim=-1)
        scale = self.score_scale.detach().exp().clamp(1, 100)
        foreground_logits = scale * foreground_descriptor @ head.T
        # Only foreground has the existing differentiable KL auxiliary. The
        # background readout is a routing measurement, not a new BG objective.
        with torch.no_grad():
            background_native = self.trunk.vision_model.post_layernorm(background)
            background_descriptor = F.normalize(
                self.trunk.visual_projection(background_native).float(), dim=-1)
            background_logits = scale * background_descriptor @ head.T
            fg_probability = foreground_logits.detach().float().softmax(-1)
            bg_probability = background_logits.detach().float().softmax(-1)
            selected_class = fg_probability.argmax(-1)
            reliability = (fg_probability.gather(1, selected_class[:, None])
                - bg_probability.gather(1, selected_class[:, None])).flatten().clamp(0., 1.)

        if self.routing_strength == 0.:
            # Reuse the exact original tensors: no zero-weight computational edge.
            gain = original_gain.expand(len(images))
            logits, descriptor, native = original_logits, original_descriptor, original_native
        else:
            gain = original_gain + (.08 - original_gain).clamp_min(0.) * reliability
            logits, descriptor, native = self._decision(
                cls + gain[:, None] * foreground - background_gain * background)
        result = dict(scene_logits=logits, object_descriptor=descriptor, scene_native=native,
            ungated_scene_logits=original_logits, foreground_logits=foreground_logits,
            routing_reliability=reliability, effective_foreground_gain=gain,
            original_foreground_gain=original_gain, routing_selected_class=selected_class)
        if background_pairing is not None:
            # The recipient's route stays fixed while only its BG residual changes.
            if self.routing_strength == 0.:
                exchanged_field = cls + original_gain * foreground - background_gain * background[background_pairing]
            else:
                exchanged_field = cls + gain[:, None] * foreground - background_gain * background[background_pairing]
            result['exchanged_logits'] = self._decision(exchanged_field)[0]
        if return_diagnostics:
            result['routing_diagnostics'] = dict(
                foreground_probability=fg_probability, background_probability=bg_probability,
                foreground_native=foreground_native, foreground_descriptor=foreground_descriptor,
                background_logits=background_logits, final_token_states=tokens.last_hidden_state,
                no_main_class_agreement_requirement=True, recipient_gate_reused_for_exchange=True)
        return result
