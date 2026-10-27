"""Check direct-ZIP cache pixels against the original file-based functions."""
import ast
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from PIL import Image, ImageOps
from build_semifinal_cache import decode, ROOT

def function_from_file(path, name, namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

def main():
    original=function_from_file(ROOT/'src/data/image_cache.py','_load_one',dict(Image=Image,np=np,_WORKER_SIZE=512))
    complete=function_from_file(ROOT/'build_cache.py','complete',dict(Image=Image,ImageOps=ImageOps,np=np))
    with zipfile.ZipFile(ROOT.parent/'train.zip' if not (ROOT/'train.zip').exists() else ROOT/'train.zip') as archive:
        names=sorted(n for n in archive.namelist() if not n.endswith('/'))
        # The local workspace stores the archive one directory above the code.
        import build_semifinal_cache as cache_module
        if not (ROOT/'train.zip').exists():
            cache_module.LOCAL.archive=archive
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as folder:
            for name in [names[int(i)] for i in np.linspace(0,len(names)-1,24)]:
                target=Path(folder)/'sample.jpg'; target.write_bytes(archive.read(name))
                a,b=decode(name)
                expected,ok=original(str(target))
                assert ok and np.array_equal(a,expected),name
                assert np.array_equal(b,complete(target)),name
    print('24 TRAIN images: both direct-ZIP caches exactly match the original pixel arrays')

if __name__=='__main__': main()
