import math
import torch
from torch import nn
from torch.nn import functional as F
from model_base import VisualClassifier


class Network(VisualClassifier):
    def __init__(self, official_path, config, initialize_official=True):
        super().__init__(official_path, config["size"], config["scale"], initialize_official=initialize_official, num_classes=config.get('num_classes', 500))
        self.gate = nn.Parameter(torch.zeros(()))
        self.log_temperature = nn.Parameter(torch.tensor(config["patch_temperature"]).log())
        self.contrast_norm = nn.LayerNorm(768, elementwise_affine=False)
        self.fraction, self.gate_bound = config["foreground_fraction"], config["gate_bound"]

    def forward(self, images):
        output = self.backbone.vision_model(images, interpolate_pos_encoding=False)
        tokens = output.last_hidden_state
        cls, patch = tokens[:, 0], tokens[:, 1:]
        cosine = (F.normalize(cls.float(), dim=-1)[:, None] * F.normalize(patch.float(), dim=-1)).sum(-1)
        count = max(1, math.ceil(patch.shape[1] * self.fraction))
        selected = cosine.topk(count, -1).indices
        foreground_mask = torch.zeros_like(cosine, dtype=torch.bool).scatter_(1, selected, True)
        score = cosine / self.log_temperature.exp().clamp(.025, .35)
        score = score.masked_fill(~foreground_mask, -1e4)
        attention = score.softmax(-1).to(patch.dtype)
        foreground = (patch * attention[:, :, None]).sum(1)
        background_mask = (~foreground_mask).to(patch.dtype)
        background = (patch * background_mask[:, :, None]).sum(1) / background_mask.sum(1, keepdim=True).clamp_min(1)
        pooled = cls + self.gate_bound * self.gate.tanh() * self.contrast_norm(foreground - background)
        embedding = self.backbone.visual_projection(self.backbone.vision_model.post_layernorm(pooled))
        return self.classify(embedding)
