"""Single CLIP ViT-B/32 student with the documented foreground/background head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionModelWithProjection


class AModel(nn.Module):
    def __init__(self, official_path, classes=500, size=448):
        super().__init__()
        self.backbone = CLIPVisionModelWithProjection.from_pretrained(
            official_path, local_files_only=True, attn_implementation="sdpa"
        )
        emb = self.backbone.vision_model.embeddings
        old = emb.position_embedding.weight.detach()
        patch = old[1:].reshape(1, 7, 7, 768).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, (size // 32, size // 32), mode="bicubic", align_corners=False)
        weight = torch.cat((old[:1], patch.permute(0, 2, 3, 1).reshape(-1, 768)))
        emb.position_embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        emb.register_buffer("position_ids", torch.arange(len(weight))[None], persistent=False)
        emb.image_size = size
        emb.num_patches = (size // 32) ** 2
        emb.num_positions = len(weight)
        self.backbone.visual_projection.requires_grad_(False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.log_temperature = nn.Parameter(torch.tensor(0.135).log())
        self.contrast_norm = nn.LayerNorm(768, elementwise_affine=False)
        self.head = nn.Linear(512, classes, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(30.0).log())

    def forward(self, image):
        tokens = self.backbone.vision_model(image, interpolate_pos_encoding=False).last_hidden_state
        cls, patch = tokens[:, 0], tokens[:, 1:]
        sim = (F.normalize(cls.float(), dim=-1)[:, None] * F.normalize(patch.float(), dim=-1)).sum(-1)
        k = max(1, math.ceil(patch.shape[1] * 0.24))
        mask = torch.zeros_like(sim, dtype=torch.bool).scatter_(1, sim.topk(k, -1).indices, True)
        attention = (sim / self.log_temperature.exp().clamp(0.025, 0.35)).masked_fill(~mask, -1e4).softmax(-1)
        foreground = (patch * attention[..., None]).sum(1)
        background = (patch * (~mask)[..., None]).sum(1) / (~mask).sum(1, keepdim=True).clamp_min(1)
        pooled = cls + 0.28 * self.gate.tanh() * self.contrast_norm(foreground - background)
        visual = self.backbone.vision_model.post_layernorm(pooled)
        feature = F.normalize(self.backbone.visual_projection(visual).float(), dim=-1)
        logits = self.log_scale.exp().clamp(1, 100) * feature @ F.normalize(self.head.weight.float(), dim=-1).T
        return logits, feature, F.normalize(visual.float(), dim=-1)
