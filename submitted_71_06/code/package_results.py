import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    if args.csv.name != "pred_results.csv":
        raise ValueError("CSV filename must be pred_results.csv")
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(args.csv, arcname="pred_results.csv")


if __name__ == "__main__":
    main()
