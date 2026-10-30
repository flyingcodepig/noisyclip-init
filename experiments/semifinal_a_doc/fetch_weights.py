"""Fetch the pinned official initialization used only for training."""
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
EXPECTED = 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    target = ROOT/'weights/clip-vit-base-patch32/pytorch_model.bin'
    if target.exists():
        assert digest(target) == EXPECTED, 'Official initialization checksum mismatch'
        print('Official initialization verified')
        return
    temporary = target.with_suffix('.downloading')
    url = f'https://huggingface.co/openai/clip-vit-base-patch32/resolve/{REVISION}/pytorch_model.bin'
    try:
        with urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream, length=8*1024*1024)
        assert digest(temporary) == EXPECTED, 'Downloaded initialization checksum mismatch'
        temporary.rename(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print('Official initialization downloaded and verified')


if __name__ == '__main__':
    main()
