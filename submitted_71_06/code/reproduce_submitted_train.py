"""Start the archived original training recipe only after verifying its original TRAIN."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    source=ROOT/'train.zip'
    assert source.is_file(),'Place original train.zip in code/train.zip'
    expected=json.loads((ROOT.parent/'metadata/semifinal_input_manifest.json').read_text())['train.zip']
    with source.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    assert actual==expected,'Training archive differs from this model version'
    (ROOT/'TRAIN_UPLOAD_VERIFIED').write_text(actual+'\n')
    subprocess.run(['bash',str(ROOT/'run_train_semifinal.sh')],cwd=ROOT,check=True)

if __name__=='__main__':main()
