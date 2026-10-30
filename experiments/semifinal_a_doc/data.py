"""Semifinal-only image inventory and reproducible views."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as I

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def inventory(root, train=True):
    root = Path(root)
    if train:
        classes = sorted(p for p in root.iterdir() if p.is_dir())
        if [p.name for p in classes] != [f"{i:04d}" for i in range(500)]:
            raise ValueError("Expected class folders 0000 through 0499")
        rows = [(str(p), int(folder.name)) for folder in classes for p in sorted(folder.glob("*.jpg"))]
    else:
        rows = [(str(p), -1) for p in sorted(root.glob("*.jpg"))]
    expected = 90197 if train else 49857
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} images, found {len(rows)}")
    return rows


def open_rgb(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def centered(image, ratio=6 / 7):
    width, height = image.size
    side = max(1, round(min(width, height) * ratio))
    left, top = (width - side) // 2, (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def padded(image):
    width, height = image.size
    side = max(width, height)
    rgb = tuple(round(v * 255) for v in MEAN)
    canvas = Image.new("RGB", (side, side), rgb)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def view(image, kind, crop=1.0, flip=False, size=448):
    image = padded(image) if kind == "full" else centered(image)
    if kind == "center":
        image = image.resize((384, 384), Image.Resampling.BICUBIC)
    if crop < 1:
        image = centered(image, crop)
    if flip:
        image = ImageOps.mirror(image)
    return T.Compose([T.Resize((size, size), interpolation=I.BICUBIC), T.ToTensor(), T.Normalize(MEAN, STD)])(image)


class Images(Dataset):
    def __init__(self, rows, seed=41291, train=False):
        self.rows, self.seed, self.train = rows, seed, train
        self.epoch = 0
        self.strong = T.Compose([
            T.RandomResizedCrop(448, scale=(0.7, 1.0), ratio=(0.75, 4 / 3), interpolation=I.BICUBIC),
            T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2, 0.2),
            T.RandAugment(1, 5), T.ToTensor(), T.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, epoch = index
        else:
            epoch = self.epoch
        path, label = self.rows[index]
        image = open_rgb(path)
        if not self.train:
            return image, label, index
        # Separate deterministic seed per sample and epoch. The worker RNG is restored afterwards.
        state = random.getstate()
        torch_state = torch.get_rng_state()
        random.seed(self.seed + epoch * 1000003 + index)
        torch.manual_seed(self.seed + epoch * 1000003 + index)
        try:
            if random.random() < 0.46:
                base = padded(image)
            else:
                base = centered(image).resize((384, 384), Image.Resampling.BICUBIC)
            strong = self.strong(base)
            if random.random() < 0.17:
                other = random.randrange(len(self.rows))
                while self.rows[other][1] == label:
                    other = random.randrange(len(self.rows))
                background = view(open_rgb(self.rows[other][0]), "center")
                inner = round(448 * random.uniform(0.37, 0.81))
                small = F.interpolate(strong[None], size=(inner, inner), mode="bilinear", align_corners=False, antialias=True)[0]
                ox, oy = random.randrange(449 - inner), random.randrange(449 - inner)
                ramp = torch.arange(inner).float()
                edge = torch.minimum(ramp / 8, (inner - 1 - ramp) / 8).clamp(0, 1)
                mask = (edge[:, None] * edge[None, :])[None]
                region = background[:, oy:oy + inner, ox:ox + inner]
                background[:, oy:oy + inner, ox:ox + inner] = region * (1 - mask) + small * mask
                strong = background
            weak = view(image, "center", flip=random.random() < 0.5)
        finally:
            random.setstate(state)
            torch.set_rng_state(torch_state)
        return strong, weak, label, index


def save_inventory(rows, path):
    Path(path).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
