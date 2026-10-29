"""One visual encoder; background exchange is a TRAIN-only feature operation."""
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionModelWithProjection, CLIPConfig
from scene.token_partition import separate_scene


class SceneFactorizer(nn.Module):
    def __init__(self,official,initialize_pretrained=True):
        super().__init__()
        if initialize_pretrained:
            self.trunk=CLIPVisionModelWithProjection.from_pretrained(str(official),local_files_only=True,attn_implementation='sdpa')
        else:
            config=CLIPConfig.from_pretrained(str(official),local_files_only=True).vision_config
            config._attn_implementation='sdpa'
            self.trunk=CLIPVisionModelWithProjection(config)
        self.trunk.visual_projection.requires_grad_(False)
        self.label_basis=nn.Parameter(torch.randn(500,512)*.02)
        self.score_scale=nn.Parameter(torch.tensor(30.).log())
        self.fg_level=nn.Parameter(torch.tensor(-2.))
        self.bg_level=nn.Parameter(torch.tensor(-2.5))
        self.focus_temperature=nn.Parameter(torch.tensor(.12).log())

    def _decision(self,scene):
        visual=self.trunk.vision_model.post_layernorm(scene)
        descriptor=F.normalize(self.trunk.visual_projection(visual).float(),dim=-1)
        logits=self.score_scale.exp().clamp(1,100)*descriptor@F.normalize(self.label_basis,dim=-1).T
        return logits,descriptor,visual

    def forward(self,images,background_pairing=None):
        output=self.trunk.vision_model(pixel_values=images,interpolate_pos_encoding=True)
        cls,foreground,background=separate_scene(output.last_hidden_state,self.focus_temperature.exp().clamp(.025,.35))
        foreground=F.layer_norm(foreground,(768,));background=F.layer_norm(background,(768,))
        foreground_gain=.12*self.fg_level.sigmoid();background_gain=.02+.06*self.bg_level.sigmoid()
        field=cls+foreground_gain*foreground-background_gain*background
        logits,descriptor,native=self._decision(field)
        result={'scene_logits':logits,'object_descriptor':descriptor,'scene_native':native}
        if background_pairing is not None:
            altered=cls+foreground_gain*foreground-background_gain*background[background_pairing]
            result['exchanged_logits']=self._decision(altered)[0]
        return result
