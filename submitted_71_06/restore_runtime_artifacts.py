"""Restore omitted runtime files from the original verified local delivery."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA = '82d8c496089d87d118932e30cbe9b99ca4ecad90a777c5e5fe5534f362a1aadc'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    if digest(args.archive) != ARCHIVE_SHA:
        raise ValueError('Archive SHA256 mismatch')
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text(encoding='utf-8'))
    prefix = manifest['package'] + '/'
    with zipfile.ZipFile(args.archive) as package:
        for name, record in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError('Unsafe archive path: ' + name)
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError('Path escapes package: ' + name)
            if target.exists():
                if digest(target) != record['sha256'] or target.stat().st_size != record['bytes']:
                    raise ValueError('Existing file differs: ' + name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.restoring')
            hasher = hashlib.sha256()
            size = 0
            try:
                with package.open(prefix + name) as source, temporary.open('xb') as output:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
                        hasher.update(block)
                        size += len(block)
                if hasher.hexdigest() != record['sha256'] or size != record['bytes']:
                    raise ValueError('Restored file checksum mismatch: ' + name)
                temporary.replace(target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
    print('All original package files restored and verified')

if __name__ == '__main__':
    main()
