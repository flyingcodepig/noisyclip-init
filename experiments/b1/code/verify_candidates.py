import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_key(path):
    crc = 0
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            size += len(block)
            crc = zlib.crc32(block, crc)
    return size, crc & 0xFFFFFFFF


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-zip", type=Path)
    source.add_argument("--train-root", type=Path)
    parser.add_argument("--index", type=Path, default=ROOT / "data/train_index.json")
    parser.add_argument("--candidates", type=Path, default=ROOT / "cache/train_candidate_labels.npy")
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    rows = json.loads(args.index.read_text(encoding="utf-8"))
    values = np.load(args.candidates, mmap_mode="r")
    if len(rows) != 148695 or values.shape != (len(rows), 7) or values.dtype != np.int64:
        raise ValueError("Candidate array shape or type mismatch")
    expected = args.expected_sha or json.loads((ROOT / "training_cache_recipe.json").read_text())["expected_sha256"]["cache/train_candidate_labels.npy"]
    if digest(args.candidates) != expected:
        raise ValueError("Candidate array checksum mismatch")
    keys = []
    groups = defaultdict(list)
    if args.train_zip:
        with zipfile.ZipFile(args.train_zip) as archive:
            infos = {item.filename: item for item in archive.infolist() if not item.is_dir()}
            for name, label in rows:
                if name.split("/", 1)[0] != f"{label:04d}":
                    raise ValueError("Directory label mismatch")
                item = infos.get("train/" + name)
                if item is None:
                    raise FileNotFoundError("train/" + name)
                key = item.file_size, item.CRC
                keys.append(key)
                groups[key].append((name, label))
            for members in groups.values():
                if len({label for _, label in members}) > 1:
                    hashes = {hashlib.sha256(archive.read("train/" + name)).digest() for name, _ in members}
                    if len(hashes) != 1:
                        raise ValueError("Cross-label content collision")
    else:
        for name, label in rows:
            if name.split("/", 1)[0] != f"{label:04d}":
                raise ValueError("Directory label mismatch")
            path = args.train_root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            key = file_key(path)
            keys.append(key)
            groups[key].append((name, label))
        for members in groups.values():
            if len({label for _, label in members}) > 1:
                if len({digest(args.train_root / name) for name, _ in members}) != 1:
                    raise ValueError("Cross-label content collision")
    for index, ((name, label), key) in enumerate(zip(rows, keys)):
        actual = values[index].tolist()
        expected_row = sorted({item_label for _, item_label in groups[key]})
        if actual != expected_row + [-1] * (7 - len(expected_row)) or label not in actual:
            raise ValueError(f"Candidate mismatch at row {index}: {name}")
    print(json.dumps({"status": "verified", "source": "official_train", "rows": len(rows), "candidate_sha256": expected}), flush=True)


if __name__ == "__main__":
    main()
