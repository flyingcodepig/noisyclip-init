"""Score frozen rematch predictions; labels are never imported by training code."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 37444 or any(len(row) != 2 for row in rows):
        raise ValueError("Expected 37,444 two-column rows")
    mapping = {name.strip(): label.strip() for name, label in rows}
    if len(mapping) != len(rows) or any(len(label) != 4 or not label.isdigit() or int(label) >= 750 for label in mapping.values()):
        raise ValueError("Duplicate names or invalid class IDs")
    return mapping


def split_by_class(labels):
    groups = defaultdict(list)
    for name, label in labels.items():
        groups[label].append(name)
    development, holdout = set(), set()
    for names in groups.values():
        names.sort(key=lambda n: hashlib.sha256(("rematch-41291:" + n).encode()).digest())
        cutoff = round(len(names) * 0.7)
        development.update(names[:cutoff]); holdout.update(names[cutoff:])
    assert len(development | holdout) == len(labels) and not development & holdout
    return development, holdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--prediction", required=True)
    p.add_argument("--holdout", action="store_true", help="Report held-out result after selection is frozen")
    args = p.parse_args()
    truth, pred = read_csv(args.labels), read_csv(args.prediction)
    if set(truth) != set(pred): raise ValueError("Prediction filenames do not match reference labels")
    dev, hold = split_by_class(truth)
    selected = hold if args.holdout else dev
    correct = sum(pred[name] == truth[name] for name in selected)
    print(json.dumps({"split": "holdout" if args.holdout else "development", "correct": correct,
                      "total": len(selected), "accuracy": correct / len(selected),
                      "label_sha256": hashlib.sha256(Path(args.labels).read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__": main()
