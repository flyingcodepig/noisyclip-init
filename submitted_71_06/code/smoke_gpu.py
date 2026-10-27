import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'engine'))
from model_base import load_model
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cfg = json.loads((ROOT/'train_config_stage1.json').read_text())
model = load_model(ROOT, cfg, ROOT).cuda()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
batch = int(sys.argv[1]) if len(sys.argv)>1 else 48
started = time.monotonic()
for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    images = torch.randn(batch, 3, cfg['size'], cfg['size'], device='cuda')
    with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
        logits, features = model(images)
    loss = logits.float().square().mean() + features.float().square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps(dict(batch=batch,step=step,loss=loss.item(),peak_gb=torch.cuda.max_memory_allocated()/1e9,seconds=time.monotonic()-started)),flush=True)
