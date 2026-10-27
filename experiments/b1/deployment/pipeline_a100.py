"""500-class single-A100 orchestration and read-only streaming monitor."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

DEPLOY = Path(__file__).resolve().parent
CODE = DEPLOY.parent / 'code'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--after', type=float, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-prediction', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run):
        parser.error('Invalid run name')
    folder = CODE / 'runs' / (args.run + '_pipeline')
    status = folder / 'status.json'
    events = folder / 'progress.log'
    session = 'b1_' + args.run
    if args.monitor:
        offset = 0
        print('Waiting for server status; Ctrl+C disconnects monitoring only.', flush=True)
        for attempt in range(30):
            if status.exists() and json.loads(status.read_text()).get('updated', 0) >= args.after:
                break
            time.sleep(1)
        else:
            launch = CODE / 'runs' / (args.run + '.launch.log')
            if launch.exists():
                print(launch.read_text(errors='replace'), flush=True)
            raise SystemExit('No pipeline status. Check launch log and tmux session.')
        while True:
            if events.exists():
                with events.open() as stream:
                    stream.seek(offset)
                    text = stream.read()
                    offset = stream.tell()
                if text:
                    print(text, end='', flush=True)
            record = json.loads(status.read_text())
            if record['state'] in ('completed', 'failed'):
                raise SystemExit(0 if record['state'] == 'completed' else 1)
            if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode:
                raise SystemExit('Server session ended without completion status; inspect logs before resuming.')
            time.sleep(2)

    import fcntl
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['RUN_PREFIX'] = args.run
    os.environ['PYTHON'] = sys.executable
    (CODE / 'runs').mkdir(exist_ok=True)
    lock = (CODE / 'runs/.a100.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another B1 pipeline holds the training lock.')
    if not args.resume and any(p.exists() for p in [folder, CODE/'runs'/f'{args.run}_stage1', CODE/'runs'/f'{args.run}_stage2', CODE/'runs'/f'{args.run}_predictions']):
        raise SystemExit('Run already exists; use --resume or a new run ID.')
    if args.resume and not folder.exists():
        raise SystemExit('Cannot resume a nonexistent managed run.')
    folder.mkdir(exist_ok=True)
    stage = 'preflight'
    def emit(message, state='running', **extra):
        record = dict(state=state, stage=stage, run=args.run, updated=time.time(), time=time.strftime('%Y-%m-%d %H:%M:%S'), message=message, **extra)
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        temporary.replace(status)
        with events.open('a') as stream:
            stream.write(f"[{record['time']}] [{stage}] {message}\n")

    def run(label, command):
        nonlocal stage
        stage = label
        emit('开始')
        with (folder / f'{label}.log').open('a') as log:
            log.write('\nCOMMAND ' + json.dumps(command) + '\n'); log.flush()
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line); log.flush()
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        emit(line.strip())
                        continue
                    if obj.get('stage') == 'training':
                        emit(f"{obj['step']}/{obj['target']} 更新 | loss={obj['mean_loss']:.5f} | {obj['seconds_per_step']:.2f}秒/更新 | 显存峰值={obj['peak_allocated_gib']:.2f}GiB", step=obj['step'], target=obj['target'])
                    else:
                        emit(json.dumps(obj, ensure_ascii=False))
                result = process.wait()
            if result:
                raise RuntimeError(f'{label} exited {result}; log: {folder / (label + ".log")}')
        emit('完成')

    py = sys.executable
    try:
        emit('检查500类配置、GPU及输入')
        for s in ('stage1', 'stage2'):
            cfg = json.loads((CODE/f'train_config_a100_{s}.json').read_text())
            assert cfg['num_classes'] == 500, 'Only the fixed 500-class dataset is supported'
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if gpu.strip():
            raise RuntimeError('GPU has active processes; refusing concurrent training')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
        (folder/'run.json').write_text(json.dumps(dict(run=args.run, revision=revision, python=py, classes=500), indent=2))
        run('01_数据与缓存准备', ['bash', str(DEPLOY/'prepare_a100.sh')])
        # Reuse an existing exact config/source/quality gate; do not repeat fallback batches.
        gate = subprocess.run([py, str(DEPLOY/'check_smoke_gate.py')], capture_output=True, text=True)
        (folder/'gate_check.log').write_text(gate.stdout + gate.stderr)
        if gate.returncode:
            run('02_最终配置冒烟与恢复验证', [py, str(DEPLOY/'smoke_a100.py')])
        else:
            stage = '02_复用已通过验证'
            emit('源码、配置与质量绑定未变，跳过重复完整冒烟')

        def train(s, target, label):
            checkpoint = CODE/'runs'/f'{args.run}_{s}'/'last.pt'
            command = [py, str(CODE/'engine/train_single_gpu.py'), '--workspace', str(CODE), '--config', str(CODE/f'train_config_a100_{s}.json'), '--run', f'{args.run}_{s}', '--until', str(target)]
            if checkpoint.exists():
                command += ['--resume']
            elif s == 'stage2':
                command += ['--restart-from', str(CODE/'runs'/f'{args.run}_stage1'/'last.pt')]
            run(label, command)
            run(label+'_验收', [py, str(DEPLOY/'check_checkpoint.py'), str(checkpoint), str(CODE/f'train_config_a100_{s}.json'), str(target), str(CODE)])
        train('stage1', 4096, '03_阶段一')
        train('stage2', 1536, '04_阶段二前段')
        train('stage2', 2048, '05_阶段二续训')
        if not args.skip_prediction:
            run('06_推理与预测验收', ['bash', str(DEPLOY/'predict_a100.sh')])
        stage = '完成'
        emit('正式训练完成' + ('；按要求未运行推理' if args.skip_prediction else '；预测验收完成'), state='completed')
    except BaseException as exc:
        emit(f'{type(exc).__name__}: {exc}', state='failed')
        raise

if __name__ == '__main__':
    main()
