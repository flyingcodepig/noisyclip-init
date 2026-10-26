import argparse
from pathlib import Path
import zipfile


FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    if args.csv.name != "pred_results.csv":
        raise ValueError("CSV filename must be pred_results.csv")
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("pred_results.csv", date_time=FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(info, args.csv.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


if __name__ == "__main__":
    main()
