"""Local scoring: CLIP-consensus reference + clean-subset accuracy. No TEST labels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = Path('/root/b1-data-20261027/train')


class ValImages(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert('RGB')
            return self.processor(images=im, return_tensors='pt')['pixel_values'][0]


def main():
    torch.set_num_threads(4)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    index = json.loads((ROOT / 'data/val_index.json').read_text())
    paths = [TRAIN_ROOT / row[0] for row in index]
    labels = np.asarray([row[1] for row in index], dtype=np.int64)

    folder = ROOT / 'weights/clip-vit-base-patch32'
    processor = CLIPImageProcessor.from_pretrained(str(folder), local_files_only=True)
    model = CLIPVisionModelWithProjection.from_pretrained(str(folder), local_files_only=True).cuda().eval()

    features = []
    with torch.inference_mode():
        loader = DataLoader(ValImages(paths, processor), batch_size=512, num_workers=8, pin_memory=True)
        for number, batch in enumerate(loader):
            emb = model(batch.cuda()).image_embeds
            features.append(F.normalize(emb, dim=-1).half().cpu().float().numpy())
            if number % 5 == 0:
                print('val features', number * 512, '/', len(paths), flush=True)
    del model
    torch.cuda.empty_cache()

    features = np.concatenate(features).astype(np.float32)  # (9020, 512), normalized
    centers = np.load(ROOT / 'engine/initial_centers.npy').astype(np.float32)  # (500, 512), normalized
    consensus = (features @ centers.T).argmax(-1)  # nearest frozen-CLIP class center
    np.save(ROOT / 'engine/val_teacher.npy', features)
    np.save(ROOT / 'engine/val_consensus.npy', consensus)

    clean_mask = (consensus == labels)
    print(f'val={len(labels)}  given==consensus: {clean_mask.sum()} ({clean_mask.mean():.3f})', flush=True)

    results = {}
    run_names = sorted(p.name for p in (ROOT / 'runs').iterdir()
                       if p.is_dir() and (p / 'validation_predictions.csv').exists())
    for run in run_names:
        rows = list(csv.DictReader(open(ROOT / f'runs/{run}/validation_predictions.csv')))
        pred = np.asarray([int(r['prediction']) for r in rows], dtype=np.int64)
        given = np.asarray([int(r['given_label']) for r in rows], dtype=np.int64)
        assert len(pred) == len(labels) and (given == labels).all(), run
        results[run] = dict(
            given_label_acc=float((pred == labels).mean()),
            consensus_acc=float((pred == consensus).mean()),
            clean_subset_acc=float((pred[clean_mask] == labels[clean_mask]).mean()) if clean_mask.any() else None,
        )
    out = dict(consensus_agreement=float(clean_mask.mean()), results=results)
    print(json.dumps(out, indent=2))
    (ROOT / 'runs/local_score.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
