"""Validate the semifinal CSV/ZIP against the exact test image inventory."""
import argparse
import csv
import zipfile
from pathlib import Path
from data import inventory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", required=True)
    p.add_argument("--zip", required=True)
    args = p.parse_args()
    expected = {Path(path).name for path, _ in inventory(args.test_root, train=False)}
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        if names != ["pred_results.csv"]:
            raise ValueError(f"ZIP must contain only pred_results.csv: {names}")
        lines = archive.read(names[0]).decode("utf-8-sig").splitlines()
    rows = list(csv.reader(lines))
    if len(rows) != 49857: raise ValueError(f"Expected 49857 rows, got {len(rows)}")
    if any(len(row) != 2 for row in rows): raise ValueError("Every row needs filename,class_id")
    actual = [row[0] for row in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Duplicate, missing, or extra test filenames")
    if any(row[1] != row[1].strip() or len(row[1]) != 4 or not row[1].isdigit() or not 0 <= int(row[1]) < 500 for row in rows):
        raise ValueError("Classes must be 0000..0499, with no whitespace")
    print(f"OK: {len(rows)} unique test images, valid class IDs, single CSV in ZIP")


if __name__ == "__main__": main()
