from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--test-list", required=True, help="UTF-8 text file, one exact filename per line"
    )
    parser.add_argument(
        "--class-map", required=True, help="JSON mapping class_id to internal index"
    )
    parser.add_argument("--header", choices=["yes", "no"], default="no")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    test_names = [
        line.rstrip("\r\n")
        for line in Path(args.test_list).read_text(encoding="utf-8").splitlines()
        if line
    ]
    class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
    allowed_classes = set(class_map)
    class_pattern = re.compile(r"^[0-9]{4}$")

    rows: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if args.header == "yes":
            header = next(reader, None)
            if header != ["filename", "class_id"]:
                raise ValueError(f"unexpected header: {header}")
        for line_number, row in enumerate(reader, start=2 if args.header == "yes" else 1):
            if len(row) != 2:
                raise ValueError(f"line {line_number}: expected 2 columns, found {len(row)}")
            filename, class_id = row
            if filename != filename.strip() or class_id != class_id.strip():
                raise ValueError(f"line {line_number}: leading/trailing whitespace")
            if not class_pattern.fullmatch(class_id):
                raise ValueError(f"line {line_number}: class ID must be four digits: {class_id!r}")
            if class_id not in allowed_classes:
                raise ValueError(f"line {line_number}: unknown class ID: {class_id}")
            rows.append((filename, class_id))

    predicted_names = [item[0] for item in rows]
    if len(predicted_names) != len(set(predicted_names)):
        raise ValueError("submission contains duplicate filenames")
    expected_set = set(test_names)
    predicted_set = set(predicted_names)
    missing = sorted(expected_set - predicted_set)
    extra = sorted(predicted_set - expected_set)
    if missing or extra:
        raise ValueError(f"filename mismatch: missing={missing[:10]}, extra={extra[:10]}")
    if len(rows) != len(test_names):
        raise ValueError(f"row count mismatch: expected {len(test_names)}, got {len(rows)}")

    print(
        json.dumps({"status": "ok", "rows": len(rows), "classes": len(allowed_classes)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
