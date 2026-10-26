import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'engine'))
from infer import predict, sha
from fetch_weights import main as fetch_official


class TrainingImages(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert('RGB')
            return self.processor(images=image, return_tensors='pt')['pixel_values'][0]


def official_arrays(features, labels):
    features = features.astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    centers = np.zeros((750,512),np.float32)
    np.add.at(centers, labels, features)
    centers /= np.bincount(labels,minlength=750)[:,None]
    centers /= np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
    return features, centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind',choices=('official',),default='official')
    parser.add_argument('--batch',type=int,default=512,help='Official 224-pixel feature batch size')
    parser.add_argument('--workers',type=int,default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    manifest = json.loads((ROOT/'training_cache_recipe.json').read_text())
    index = json.loads((ROOT/'data/train_index.json').read_text())
    paths = [ROOT/'data/official/train'/row[0] for row in index]
    labels = np.asarray([row[1] for row in index],dtype=np.int64)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} TRAIN images. Place official images under data/official')
    np.save(ROOT/'engine/train_labels.npy',labels)
    if args.kind in ('official','all'):
        fetch_official()
        folder = ROOT/'weights/clip-vit-base-patch32'
        processor = CLIPImageProcessor.from_pretrained(str(folder),local_files_only=True)
        model = CLIPVisionModelWithProjection.from_pretrained(str(folder),local_files_only=True).cuda().eval()
        loader = DataLoader(TrainingImages(paths,processor),batch_size=args.batch,num_workers=args.workers,pin_memory=True)
        features = []
        with torch.inference_mode():
            for number,pixels in enumerate(loader):
                embedding = model(pixels.cuda()).image_embeds
                features.append(F.normalize(embedding,dim=-1).half().cpu().float().numpy())
                if number%50 == 0:
                    print('Official TRAIN features',number*args.batch,len(paths),flush=True)
        del model
        torch.cuda.empty_cache()
        features,centers = official_arrays(np.concatenate(features),labels)
        np.save(ROOT/'engine/train_teacher.npy',features)
        np.save(ROOT/'engine/initial_centers.npy',centers)
    required = ['engine/train_labels.npy','cache/train_candidate_labels.npy']
    if args.kind in ('official','all'):
        required += ['engine/train_teacher.npy','engine/initial_centers.npy']
    for name in required:
        assert sha(ROOT/name) == manifest['expected_sha256'][name], f'Cache differs: {name}; check environment, image files and batch sizes. Do not bypass this check.'
    print('Regenerated TRAIN inputs passed the original checksums')


if __name__ == '__main__':
    main()
