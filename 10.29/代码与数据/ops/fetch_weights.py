"""Fetch and verify the official OpenAI CLIP ViT-B/32 initialization."""
from __future__ import annotations
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))

import argparse
import hashlib
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
BASE_URL = f"https://huggingface.co/openai/clip-vit-base-patch32/resolve/{REVISION}"
FILES = {
    "config.json": "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
    "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    "pytorch_model.bin": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
}
CHUNK_BYTES = 32 * 1024 * 1024


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_file(url, target, expected_sha, retries=3):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected_sha:
            raise ValueError(f"Existing {target.name} has a different SHA256; select an empty destination")
        print(f"Verified {target.name}; using existing file", flush=True)
        return
    temporary = target.with_suffix(target.suffix + ".download")
    if temporary.exists() and sha(temporary) == expected_sha:
        temporary.replace(target)
        print(f"Verified completed download: {target.name}", flush=True)
        return
    failures = 0
    while True:
        offset = temporary.stat().st_size if temporary.exists() else 0
        try:
            request = Request(url, headers={"Range": f"bytes={offset}-{offset + CHUNK_BYTES - 1}",
                                           "Cache-Control": "no-cache"})
            with urlopen(request, timeout=120) as response:
                if response.status == 206:
                    interval = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if interval is None:
                        raise ValueError("A partial response is missing its byte range")
                    first, last, total = map(int, interval.groups())
                    if first != offset or last < first or last >= total:
                        raise ValueError("The download server returned a different byte range")
                    expected_bytes = last - first + 1
                    mode = "ab"
                elif response.status == 200:
                    # Servers may ignore Range. Restart instead of appending duplicate bytes.
                    offset = 0
                    length = response.headers.get("Content-Length")
                    total = int(length) if length is not None else None
                    expected_bytes, mode = total, "wb"
                else:
                    raise ValueError(f"Unexpected download status: {response.status}")
                received = 0
                with temporary.open(mode) as stream:
                    while expected_bytes is None or received < expected_bytes:
                        size = 8 * 1024 * 1024 if expected_bytes is None else min(8 * 1024 * 1024, expected_bytes - received)
                        chunk = response.read(size)
                        if not chunk:
                            break
                        stream.write(chunk)
                        received += len(chunk)
                if expected_bytes is not None and received != expected_bytes:
                    raise OSError("The response ended before its declared byte count")
            current = temporary.stat().st_size
            print(f"Downloaded {target.name}: {current}/{total or 'unknown'} bytes", flush=True)
            if total is not None and current < total:
                continue
            if sha(temporary) != expected_sha:
                temporary.unlink()
                raise ValueError(f"Downloaded {target.name} failed SHA256 verification")
            temporary.replace(target)
            print(f"Verified download: {target.name}", flush=True)
            return
        except Exception as error:
            if isinstance(error, HTTPError) and error.code == 416 and temporary.exists():
                temporary.unlink()
            failures += 1
            print(f"Download retry {failures}/{retries}: {target.name} ({type(error).__name__})", flush=True)
            if failures >= retries:
                raise
            time.sleep(2 * failures)


def ensure_clip(destination=None, check_only=False):
    destination = Path(destination or ROOT / "settings/clip").resolve()
    for filename, expected in FILES.items():
        path = destination / filename
        if check_only:
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f"Missing or invalid official initialization: {filename}")
            print(f"Verified {filename}", flush=True)
        else:
            download_file(f"{BASE_URL}/{filename}", path, expected)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ensure_clip(args.destination, args.check_only)
