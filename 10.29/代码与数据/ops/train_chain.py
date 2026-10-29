"""Run the recorded 8+2+2+2 epoch training recipe on official TRAIN images."""
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'initial', 'alignment', 'refine'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'var/runs/train')
    parser.add_argument('--cache', type=Path, default=ROOT / 'var/cache/train')
    parser.add_argument('--clip-dir', type=Path, default=ROOT / 'settings/clip')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    stages = [
        ('initial', ['training/schedule.py', '--destination', str(output / 'initial'), '--exchange-strength', '1']),
        ('alignment', ['training/calibrate.py', '--parent', str(output / 'initial/scene_final.pt'),
                       '--destination', str(output / 'alignment'), '--alignment-weight', '0.05']),
        ('refine', ['training/continue_scene.py', '--parent', str(output / 'alignment/alignment_final.pt'),
                    '--destination', str(output / 'refine'), '--routing-strength', '0'])]
    if args.dry_run:
        print(json.dumps(dict(epochs=[8, 2, 2, 2], images_per_epoch=90197, batch_size=88,
            total_images=1262758, optimizer_steps=14350, final_route=0, stages=[name for name, _ in stages]), indent=2))
        return
    from ops.runtime import check_training_inputs
    check_training_inputs(args.cache, args.clip_dir)
    environment = dict(os.environ, AIC_CACHE_DIR=str(args.cache.resolve()), AIC_CLIP_DIR=str(args.clip_dir.resolve()))
    for name, command in stages:
        if args.stage in ('all', name):
            subprocess.run([sys.executable, str(ROOT / command[0]), *command[1:]], env=environment, check=True)

if __name__ == '__main__':
    main()
