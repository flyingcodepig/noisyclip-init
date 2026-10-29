"""RGB bicubic source canvases and atomic file utilities."""
from __future__ import annotations
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))

import csv

import hashlib

import json

from pathlib import Path

import time

import numpy as np

from PIL import Image

import torch

from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    writing = path.with_suffix('.writing.json')
    writing.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    writing.replace(path)

class Photos(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert('RGB')
            width, height = image.size
            scale = 512 / min(width, height)
            if (width, height) != (512, 512):
                image = image.resize((max(512, int(round(width * scale))),
                                      max(512, int(round(height * scale)))), Image.Resampling.BICUBIC)
            width, height = image.size
            left, top = (width - 512) // 2, (height - 512) // 2
            raw = np.array(image.crop((left, top, left + 512, top + 512)), copy=True)
        return torch.from_numpy(raw).permute(2, 0, 1), index

