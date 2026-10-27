"""Verify an extracted delivery using only Python's standard library."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        path=ROOT/name
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==expected['sha256'],name
        assert path.stat().st_size==expected['bytes'],name
    with zipfile.ZipFile(ROOT/'results/pred_results.zip') as archive:
        assert archive.namelist()==['pred_results.csv'] and archive.testzip() is None
        assert archive.read('pred_results.csv')==(ROOT/'results/pred_results.csv').read_bytes()
    print('ALL_PACKAGE_FILES_VERIFIED',len(manifest['files']))

if __name__=='__main__':main()
