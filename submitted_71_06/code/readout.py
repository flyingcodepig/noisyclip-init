import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'engine'
sys.path.insert(0, str(ENGINE))
from infer import sha, balance
from confidence_propagation import propagate_with_floor
from balanced_query_readout import sinkhorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--policy', type=Path, default=ROOT/'inference_policy.json')
    parser.add_argument('--new-checkpoint',action='store_true',help='Use a newly trained checkpoint bound to this feature cache')
    parser.add_argument('--names',type=Path,default=ROOT/'data/test_names.json')
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1', '2', '3')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    base, out = args.base.resolve(), args.output.resolve()
    policy = json.loads(args.policy.read_text())
    binding = json.loads((base/'test_binding.json').read_text())
    checkpoint = args.model or next(ROOT.parent.glob('*-模型文件.pt'))
    assert binding['actual_views'] == policy['classification_views']
    assert sha(checkpoint) == binding['checkpoint_sha256']
    if not args.new_checkpoint:
        assert sha(checkpoint) == '9048b62c2ceff91743087c475d7a5f9f31b44f7864c37418fbccba4348'
    assert sha(base/'test.npz') == binding['feature_cache_sha256']
    with np.load(base/'test.npz') as data:
        views = torch.from_numpy(data['probabilities']).cuda()
        features = F.normalize(torch.from_numpy(data['features']).cuda(), dim=-1)
        native = F.normalize(torch.from_numpy(data['visual']).cuda(), dim=-1)
    names = json.loads(args.names.read_text())['names']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert views.shape == (len(policy['view_weights']),len(names),state['config'].get('num_classes',500)) and native.shape == (len(names),768)
    assert all(torch.isfinite(x).all() for x in (views, features, native))
    centers = F.normalize(state['model']['head.weight'].cuda(), dim=-1)
    del state
    pp = views.clamp_min(1e-9).pow(1/policy['temperature'])
    pp /= pp.sum(-1, keepdim=True)
    weights = torch.tensor(policy['view_weights'],device='cuda')[:,None,None]
    assert abs(sum(policy['view_weights'])-1) < 1e-8
    posterior = (pp*weights).sum(0)
    h = policy['head_prior_fraction']
    seed = balance((1-h)*posterior+h*(15*features@centers.T).softmax(-1),policy['epsilon'])
    n, k = len(native), policy['neighbors']
    index = torch.empty(n,k,dtype=torch.long,device='cuda')
    similarity = torch.empty(n,k,device='cuda')
    for start in range(0,n,384):
        sim = native[start:start+384]@native.T
        length = len(sim)
        sim[torch.arange(length,device='cuda'),torch.arange(start,start+length,device='cuda')] = -torch.inf
        values, ids = sim.topk(k,-1)
        similarity[start:start+length],index[start:start+length] = values,ids
    assert not (index == torch.arange(n,device='cuda')[:,None]).any()
    root = seed.sqrt()
    affinity = torch.empty_like(similarity)
    for start in range(0,n,256):
        affinity[start:start+256] = (root[start:start+256,None]*root[index[start:start+256]]).sum(-1)
    logweight = (similarity-similarity.max(-1,keepdim=True).values)*policy['sharpness']
    local = policy['local_fraction']
    if local:
        distance = (1-similarity).clamp_min(1e-5)
        radius = distance[:,policy['local_rank']-1].sqrt()
        local_log = -distance/(radius[:,None]*radius[index]).clamp_min(1e-8)
        local_log -= local_log.max(-1,keepdim=True).values
        logweight = (1-local)*logweight+local*local_log
    edge = policy['nonmutual_fraction']
    mutual = (index[index] == torch.arange(n,device='cuda')[:,None,None]).any(-1)
    weight = logweight.exp()*seed.max(-1).values[index].pow(policy['confidence_power'])*affinity*(edge+(1-edge)*mutual.float())
    q = propagate_with_floor(seed,index,weight,policy['alpha'],policy['iterations'],policy['minimum_propagation'])
    assert len(names) == len(set(names)) == n
    z = sinkhorn(q, policy['assignment_temperature'], policy['balance_strength'], policy['assignment_iterations'])
    labels = z.argmax(-1).cpu().tolist()
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='') as stream:
        csv.writer(stream).writerows((name,f'{label:04d}') for name,label in zip(names,labels))
    if not args.new_checkpoint:
        assert sha(out) == 'c33c089278063b2e8bb861c50104631b99961b59bd17f32f785771e7aedcfd41', 'Prediction differs: check the original images, environment and view extraction.'
    print(json.dumps(dict(prediction=str(out),prediction_sha256=sha(out),rows=n)),flush=True)

if __name__ == '__main__': main()
