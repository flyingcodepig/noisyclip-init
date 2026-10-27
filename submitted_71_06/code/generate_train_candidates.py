from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zlib
import zipfile

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_key(path: Path) -> tuple[int, int]:
    crc = 0
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
    return size, crc & 0xFFFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-zip", type=Path)
    source.add_argument("--train-root", type=Path)
    parser.add_argument("--index", type=Path, default=Path(__file__).resolve().parent / "data/train_index.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--expected-sha", help="Optional published candidate array SHA256")
    args = parser.parse_args()

    rows = json.loads(args.index.read_text(encoding="utf-8"))
    if len(rows) != 148695:
        raise RuntimeError(f"Unexpected TRAIN row count: {len(rows)}")
    keys = []
    by_key: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)

    if args.train_zip:
        with zipfile.ZipFile(args.train_zip) as archive:
            infos = {item.filename: item for item in archive.infolist() if not item.is_dir()}
            for name, label in rows:
                if name.split("/", 1)[0] != f"{label:04d}":
                    raise RuntimeError(f"Directory label mismatch: {name}")
                item = infos.get("train/" + name)
                if item is None:
                    raise FileNotFoundError("train/" + name)
                key = item.file_size, item.CRC
                keys.append(key)
                by_key[key].append((name, label))
            for members in by_key.values():
                if len({label for _, label in members}) > 1:
                    hashes = {hashlib.sha256(archive.read("train/" + name)).digest() for name, _ in members}
                    if len(hashes) != 1:
                        raise RuntimeError("Size/CRC collision across labels")
    else:
        root = args.train_root.resolve()
        for name, label in rows:
            if name.split("/", 1)[0] != f"{label:04d}":
                raise RuntimeError(f"Directory label mismatch: {name}")
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            key = file_key(path)
            keys.append(key)
            by_key[key].append((name, label))
        for members in by_key.values():
            if len({label for _, label in members}) > 1:
                hashes = {digest(root / name) for name, _ in members}
                if len(hashes) != 1:
                    raise RuntimeError("Size/CRC collision across labels")

    labels = np.full((len(rows), 7), -1, dtype=np.int64)
    max_candidates = 0
    for index, key in enumerate(keys):
        candidates = sorted({label for _, label in by_key[key]})
        max_candidates = max(max_candidates, len(candidates))
        if len(candidates) > 7:
            raise RuntimeError("Candidate count exceeds supported width 7")
        labels[index, : len(candidates)] = candidates

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, labels)
    output_sha = digest(args.output)
    if args.expected_sha and output_sha != args.expected_sha:
        raise RuntimeError(f"Candidate SHA mismatch: {output_sha}")
    info = {
        "source": "official TRAIN exact-content groups and given labels only",
        "reference_read": False,
        "labels_order_verified": True,
        "rows": len(rows),
        "max_candidates": max_candidates,
        "index_sha256": digest(args.index),
        "candidate_sha256": output_sha,
    }
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
