"""Separate confident object patches from the least related scene patches."""
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
import math
import torch
from torch.nn import functional as F


def separate_scene(sequence,temperature):
    reference,patches=sequence[:,0],sequence[:,1:]
    relation=(F.normalize(reference.float(),dim=-1)[:,None]*F.normalize(patches.float(),dim=-1)).sum(-1)
    foreground_count=max(1,math.ceil(patches.shape[1]*.22));background_count=max(1,math.ceil(patches.shape[1]*.35))
    foreground_index=relation.topk(foreground_count,dim=-1).indices
    background_index=relation.topk(background_count,dim=-1,largest=False).indices
    chosen=torch.zeros_like(relation,dtype=torch.bool).scatter_(1,foreground_index,True)
    focus=(relation/temperature).masked_fill(~chosen,-1e4).softmax(-1).to(patches.dtype)
    object_field=(patches*focus[:,:,None]).sum(1)
    background=patches.gather(1,background_index[:,:,None].expand(-1,-1,patches.shape[-1])).mean(1)
    return reference,object_field,background
