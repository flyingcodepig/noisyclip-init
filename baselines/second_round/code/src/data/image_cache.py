from __future__ import annotations
import json
import zlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
CACHE_SIZE = 384
_WORKER_SIZE = CACHE_SIZE

def _load_one(p: str):
    size = _WORKER_SIZE
    try:
        with Image.open(p) as im:
            im = im.convert('RGB')
            w, h = im.size
            s = size / min(w, h)
            if (w, h) != (size, size):
                im = im.resize((max(size, int(round(w * s))), max(size, int(round(h * s)))), Image.BICUBIC)
            w, h = im.size
            l, t = ((w - size) // 2, (h - size) // 2)
            im = im.crop((l, t, l + size, t + size))
            return (np.asarray(im, dtype=np.uint8), True)
    except Exception:
        return (np.zeros((size, size, 3), dtype=np.uint8), False)

def _init_worker(size: int) -> None:
    global _WORKER_SIZE
    _WORKER_SIZE = size

def build_cache(paths: list[Path], out_path: Path, size: int=CACHE_SIZE, workers: int=48, square: bool=True) -> Path:
    from concurrent.futures import ProcessPoolExecutor
    out_path = Path(out_path)
    if out_path.exists():
        print(f'[缓存] 已存在，跳过: {out_path}')
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(paths)
    arr = np.lib.format.open_memmap(out_path.with_suffix('.tmp.npy'), mode='w+', dtype=np.uint8, shape=(n, size, size, 3))
    print(f'[缓存] 解码 {n} 张 -> {out_path}  ({n * size * size * 3 / 1000000000.0:.1f} GB)')
    done = 0
    failed: list[int] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(size,)) as ex:
        for i, (a, ok) in enumerate(ex.map(_load_one, [str(p) for p in paths], chunksize=32)):
            arr[i] = a
            if not ok:
                failed.append(i)
            done += 1
            if done % 20000 == 0:
                print(f'  ... {done}/{n}')
    arr.flush()
    del arr
    out_path.with_suffix('.tmp.npy').rename(out_path)
    if failed:
        fl = out_path.with_suffix('.failed.txt')
        fl.write_text('\n'.join((str(paths[i]) for i in failed)))
        rate = len(failed) / max(1, n)
        print(f'[缓存] ⚠️ {len(failed)} 张解码失败（{rate:.3%}），已写入全黑图并记录清单 {fl.name}')
        print(f'       前几个: {[paths[i].name for i in failed[:5]]}')
        if rate > 0.001:
            print('!' * 68)
            print(f'⚠️⚠️  解码失败率 {rate:.2%} 超过 0.1% —— **建议停下来查**。')
            print('      这些图在训练里是全黑方块，会污染训练且不报错。')
            print('!' * 68)
    else:
        print(f'[缓存] 全部 {n} 张解码成功 ✓')
    print(f'[缓存] 完成 {out_path}')
    return out_path

class CachedImageDataset(Dataset):
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __init__(self, cache_path: Path, indices: np.ndarray, labels: list[int] | None=None, size: int=224, train: bool=True, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3), jitter: float=0.3, wideshot_p: float=0.0, wideshot_scale=(0.15, 0.5), degrade_p: float=0.0, erase_p: float=0.0, wideshot_bg: str='image', wideshot_feather: float=0.0, wideshot_subject_crop: float=1.0, legacy_rng: bool=False, seed_mode: str='hash', base_seed: int=0, dual_view: bool=False, randaug_n: int=0, randaug_m: int=9):
        self.arr = np.load(cache_path, mmap_mode='r')
        self.indices = np.asarray(indices)
        self.labels = labels
        self.size = size
        self.train = train
        self.scale, self.ratio, self.jitter = (scale, ratio, jitter)
        self.wideshot_p = wideshot_p
        self.wideshot_scale = wideshot_scale
        self.degrade_p = degrade_p
        self.erase_p = erase_p
        self.wideshot_bg = wideshot_bg
        self.wideshot_feather = wideshot_feather
        self.wideshot_subject_crop = wideshot_subject_crop
        self._rng = None
        self.epoch = 0
        self.legacy_rng = legacy_rng
        self.seed_mode = seed_mode
        self.base_seed = int(base_seed)
        self.randaug_n = int(randaug_n)
        self.randaug_m = int(randaug_m)
        self.dual_view = bool(dual_view)
        if randaug_n > 0:
            from torchvision.transforms import v2 as _v2
            self._randaug = _v2.RandAugment(num_ops=randaug_n, magnitude=randaug_m)
        else:
            self._randaug = None

    def set_epoch(self, ep: int) -> None:
        self.epoch = int(ep)

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, img: np.ndarray, pos: int) -> torch.Tensor:
        t = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float().div_(255.0)
        H = t.shape[-1]
        if self.train:
            if self.legacy_rng:
                if self._rng is None:
                    self._rng = np.random.default_rng(12345)
            elif self.seed_mode == 'worker':
                if self._rng is None:
                    wi = torch.utils.data.get_worker_info()
                    wid = 0 if wi is None else int(wi.id)
                    seed = (self.base_seed * 2654435761 + wid * 2246822507 + 374761393) % 2 ** 32
                    self._rng = np.random.default_rng(seed)
            else:
                gid = int(self.indices[pos])
                if self.seed_mode == 'simple':
                    seed = (gid * 1000003 + self.epoch * 7919) % 2 ** 32
                elif self.seed_mode == 'content':
                    h = zlib.crc32(img.tobytes())
                    seed = (h ^ self.epoch * 2654435761) % 2 ** 32
                else:
                    seed = (gid * 2654435761 + self.epoch * 2246822507 + 3266489909) % 2 ** 32
                self._rng = np.random.default_rng(seed)
            for _ in range(10):
                area = H * H * self._rng.uniform(*self.scale)
                ar = np.exp(self._rng.uniform(np.log(self.ratio[0]), np.log(self.ratio[1])))
                w = int(round(np.sqrt(area * ar)))
                h = int(round(np.sqrt(area / ar)))
                if w <= H and h <= H:
                    x = self._rng.integers(0, H - w + 1)
                    y = self._rng.integers(0, H - h + 1)
                    break
            else:
                w = h = H
                x = y = 0
            t = t[:, y:y + h, x:x + w]
            if self._rng.random() < 0.5:
                t = torch.flip(t, dims=[2])
            if self.jitter > 0:
                j = self.jitter
                b = 1 + self._rng.uniform(-j, j)
                c = 1 + self._rng.uniform(-j, j)
                s = 1 + self._rng.uniform(-j, j)
                g = t.mean(dim=0, keepdim=True)
                t = ((t - g) * s + g).mul_(c).add_(b - 1.0).clamp_(0, 1)
        t = F.interpolate(t.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False, antialias=True)[0]
        if self.train and self.degrade_p > 0 and (self._rng.random() < self.degrade_p):
            k = self._rng.integers(0, 3)
            if k == 0:
                sig = float(self._rng.uniform(0.6, 2.0))
                r = max(1, int(round(sig * 2)))
                ax = torch.arange(-r, r + 1, dtype=torch.float32)
                ker = torch.exp(-ax ** 2 / (2 * sig ** 2))
                ker = ker / ker.sum()
                t = F.conv2d(F.conv2d(t[None], ker.view(1, 1, 1, -1).expand(3, 1, 1, -1), padding=(0, r), groups=3), ker.view(1, 1, -1, 1).expand(3, 1, -1, 1), padding=(r, 0), groups=3)[0]
            elif k == 1:
                f = float(self._rng.uniform(0.25, 0.7))
                s = max(8, int(self.size * f))
                t = F.interpolate(t[None], size=(s, s), mode='bilinear', align_corners=False, antialias=True)
                t = F.interpolate(t, size=(self.size, self.size), mode='bilinear', align_corners=False, antialias=True)[0]
            else:
                t = (t + torch.randn_like(t) * float(self._rng.uniform(0.01, 0.06))).clamp_(0, 1)
        if self.train and self.erase_p > 0 and (self._rng.random() < self.erase_p):
            for _ in range(self._rng.integers(1, 4)):
                eh = int(self.size * self._rng.uniform(0.05, 0.25))
                ew = int(self.size * self._rng.uniform(0.05, 0.25))
                if eh < 2 or ew < 2:
                    continue
                oy = int(self._rng.integers(0, self.size - eh + 1))
                ox = int(self._rng.integers(0, self.size - ew + 1))
                t[:, oy:oy + eh, ox:ox + ew] = float(self._rng.uniform(0, 1))
        if self.train and self.wideshot_p > 0 and (self._rng.random() < self.wideshot_p):
            f = float(self._rng.uniform(*self.wideshot_scale))
            inner = max(8, int(round(self.size * f)))
            src = t
            sc = self.wideshot_subject_crop
            if sc < 1.0:
                ch, cw = (t.shape[-2], t.shape[-1])
                sh, sw = (max(8, int(round(ch * sc))), max(8, int(round(cw * sc))))
                y0, x0 = ((ch - sh) // 2, (cw - sw) // 2)
                src = t[:, y0:y0 + sh, x0:x0 + sw]
            small = F.interpolate(src.unsqueeze(0), size=(inner, inner), mode='bilinear', align_corners=False, antialias=True)[0]
            mode = self.wideshot_bg
            if mode == 'mix':
                mode = ('image', 'noise', 'color')[int(self._rng.integers(0, 3))]
            if mode == 'noise':
                canvas = torch.randn(3, self.size, self.size) * 0.35 + 0.5
            elif mode == 'color':
                canvas = torch.rand(3, 1, 1).expand(3, self.size, self.size).clone()
            else:
                cand = self.indices
                if self.labels is not None and len(self.indices) > 1:
                    lab_i = self.labels[pos]
                    other = np.flatnonzero(np.asarray(self.labels) != lab_i)
                    cand = self.indices[other] if other.size else self.indices
                bg_i = int(cand[self._rng.integers(0, len(cand))])
                bg = np.array(self.arr[bg_i], copy=True)
                bg = torch.from_numpy(bg).permute(2, 0, 1).float().div_(255.0)
                canvas = F.interpolate(bg.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False, antialias=True)[0]
            oy = int(self._rng.integers(0, self.size - inner + 1))
            ox = int(self._rng.integers(0, self.size - inner + 1))
            canvas = canvas.clone()
            if self.wideshot_feather > 0:
                k = min(max(1, int(round(self.wideshot_feather))), max(1, inner // 2))
                a = np.ones((inner, inner), dtype=np.float32)
                ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
                a[:k, :] *= ramp[:, None]
                a[-k:, :] *= ramp[::-1, None]
                a[:, :k] *= ramp[None, :]
                a[:, -k:] *= ramp[::-1][None, :]
                am = torch.from_numpy(a).unsqueeze(0)
                region = canvas[:, oy:oy + inner, ox:ox + inner]
                canvas[:, oy:oy + inner, ox:ox + inner] = region * (1.0 - am) + small * am
            else:
                canvas[:, oy:oy + inner, ox:ox + inner] = small
            t = canvas
        if self.train and self._randaug is not None:
            t = self._randaug(t)
        return (t - self.MEAN) / self.STD

    def __getitem__(self, i: int):
        gi = int(self.indices[i])
        out = {'image': self._augment(self.arr[gi], i), 'index': i}
        if self.dual_view:
            out['image2'] = self._augment(self.arr[gi], i)
        if self.labels is not None:
            out['label'] = self.labels[i]
        return out

def collate_cached(batch):
    out = {'image': torch.stack([b['image'] for b in batch]), 'index': torch.tensor([b['index'] for b in batch])}
    if 'image2' in batch[0]:
        out['image2'] = torch.stack([b['image2'] for b in batch])
    if 'label' in batch[0]:
        out['label'] = torch.tensor([b['label'] for b in batch])
    return out

def cache_meta(cache_path: Path) -> dict:
    m = Path(cache_path).with_suffix('.json')
    return json.loads(m.read_text()) if m.exists() else {}
