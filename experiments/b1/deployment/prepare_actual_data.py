"""Read actual archives; CRC/decode/SHA checks; write new data, never source code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'code'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archives', type=Path, default=Path('/mnt/半决赛数据集'))
    p.add_argument('--storage', type=Path, default=Path('/root/b1-data-20261027'))
    a = p.parse_args()
    out = ROOT / 'data/a100'
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'source': str(a.archives), 'test_labels_read': False, 'splits': {}}
    for split in ['train', 'test']:
        archive = a.archives / f'{split}.zip'
        archive_hash = sha(archive)
        target = a.storage / split
        target.mkdir(parents=True, exist_ok=True)
        rows, hashes = [], []
        with zipfile.ZipFile(archive) as z:
            infos = sorted((i for i in z.infolist() if not i.is_dir()), key=lambda i: i.filename)
            assert len({i.filename for i in infos}) == len(infos)
            for no, info in enumerate(infos):
                name = PurePosixPath(info.filename)
                if name.is_absolute() or '..' in name.parts or name.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    raise ValueError(f'Unexpected archive member: {name}')
                expected_parts = 2 if split == 'train' else 1
                assert len(name.parts) == expected_parts
                dest = target / str(name)
                # Reading to EOF validates ZIP CRC, including on resumed preparations.
                h = hashlib.sha256()
                tmp = dest.with_suffix(dest.suffix + '.extracting')
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, tmp.open('wb') as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk); h.update(chunk)
                if dest.exists():
                    if sha(dest) != h.hexdigest():
                        raise ValueError(f'Existing extraction differs: {dest}')
                    tmp.unlink()
                else:
                    tmp.replace(dest)
                with Image.open(dest) as im:
                    im.load()  # no silently substituted black image
                hashes.append([str(name), h.hexdigest()])
                if split == 'train':
                    label = int(name.parts[0]); assert name.parts[0] == f'{label:04d}'
                    rows.append([str(name), label])
                else:
                    rows.append(str(name))
                if no % 10000 == 0:
                    print(json.dumps({'stage': 'extract_verify', 'split': split, 'rows': no}), flush=True)
        if split == 'train':
            classes = sorted({row[1] for row in rows})
            assert classes == list(range(500)), classes
            index_name, content = 'train_index.json', rows
        else:
            index_name, content = 'test_names.json', {'names': rows}
        for filename, content_item in [(index_name, content), (f'{split}_sha256.json', hashes)]:
            path = out / filename
            text = json.dumps(content_item, ensure_ascii=False, separators=(',', ':')) + '\n'
            if path.exists() and path.read_text() != text:
                raise ValueError(f'Existing manifest differs: {path}')
            path.write_text(text)
        link = ROOT / 'data/official' / split
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists():
            assert link.resolve() == target.resolve()
        else:
            link.symlink_to(target, target_is_directory=True)
        manifest['splits'][split] = {'archive_sha256': archive_hash, 'rows': len(rows), 'root': str(target), 'index_sha256': sha(out / index_name), 'image_manifest_sha256': sha(out / f'{split}_sha256.json')}
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__':
    main()
