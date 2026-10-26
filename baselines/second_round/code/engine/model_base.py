from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig

class VisualClassifier(nn.Module):

    def __init__(self, official_path, size, scale, initialize_official=True):
        super().__init__()
        if initialize_official:
            self.backbone = CLIPVisionModelWithProjection.from_pretrained(str(official_path), local_files_only=True, attn_implementation='sdpa')
        else:
            config = CLIPVisionConfig.from_pretrained(str(official_path), local_files_only=True)
            config._attn_implementation = 'sdpa'
            self.backbone = CLIPVisionModelWithProjection(config)
        emb = self.backbone.vision_model.embeddings
        initial = emb.position_embedding.weight.detach().cpu()
        patch = initial[1:].reshape(1, 7, 7, 768).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, (size // 32, size // 32), mode='bicubic', align_corners=False)
        weights = torch.cat((initial[:1], patch.permute(0, 2, 3, 1).reshape(-1, 768)))
        emb.position_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        emb.register_buffer('position_ids', torch.arange(len(weights))[None], persistent=False)
        emb.image_size = size
        emb.num_patches = (size // 32) ** 2
        emb.num_positions = len(weights)
        self.backbone.visual_projection.requires_grad_(False)
        self.head = nn.Linear(512, 750, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(float(scale)).log())

    def classify(self, embedding):
        feature = F.normalize(embedding.float(), dim=-1)
        logits = self.log_scale.exp().clamp(1, 100) * feature @ F.normalize(self.head.weight, dim=-1).T
        return (logits, feature)

    def forward(self, images):
        return self.classify(self.backbone(images, interpolate_pos_encoding=False).image_embeds)

def load_model(workspace, config, root, initialize_official=True):
    import importlib.util
    source = Path(workspace) / 'src/model.py'
    spec = importlib.util.spec_from_file_location('native_model', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Network(root / 'weights/clip-vit-base-patch32', config, initialize_official=initialize_official)
