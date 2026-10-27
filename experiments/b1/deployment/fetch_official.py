"""Download pinned official bytes from original host or mirror; hash is mandatory."""
from pathlib import Path
import sys
import urllib.request
import shutil
ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(ROOT))
from fetch_weights import REVISION, EXPECTED, digest

target = ROOT / 'weights/clip-vit-base-patch32/pytorch_model.bin'
if target.exists():
    assert digest(target) == EXPECTED
else:
    errors = []
    for host in ['huggingface.co', 'hf-mirror.com']:
        try:
            url = f'https://{host}/openai/clip-vit-base-patch32/resolve/{REVISION}/pytorch_model.bin'
            tmp = target.with_suffix('.downloading')
            with urllib.request.urlopen(url, timeout=45) as response, tmp.open('wb') as f:
                shutil.copyfileobj(response, f, length=8*1024*1024)
            assert digest(tmp) == EXPECTED, 'Official weight checksum mismatch'
            tmp.replace(target)
            break
        except Exception as exc:
            errors.append(f'{host}: {type(exc).__name__}: {exc}')
            print(errors[-1], flush=True)
    else:
        raise RuntimeError(errors)
print('Official initialization SHA256 verified', flush=True)
