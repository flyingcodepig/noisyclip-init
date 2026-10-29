"""Train all stages and export one EMA checkpoint; Linux/CUDA runtime."""
import argparse,hashlib,io,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import prepare_optimization as prep
import prepare_robust_v3 as builder
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,default=ROOT/'workspace');args=parser.parse_args()
    torch.set_num_threads(4)
    labels=np.load(ROOT/'engine/train_labels.npy');teacher=np.load(ROOT/'engine/train_teacher.npy',mmap_mode='r')
    assert len(labels)==len(teacher) and set(labels.tolist())==set(range(500))
    configs={name+'.json':json.loads((ROOT/'configs'/f'{name}.json').read_text()) for name in ['stage1','stage2','refine']}
    buffer=io.BytesIO();np.save(buffer,np.arange(len(labels)))
    for cfg in configs.values():cfg['candidate_labels_sha256']=prep.digest_candidate(labels)
    configs['refine.json']['training_indices_sha256']=hashlib.sha256(buffer.getvalue()).hexdigest()
    prep.BASE=args.workspace.resolve().parent
    ws=prep.prepare(args.workspace.name,np.arange(len(labels)),np.array([],dtype=np.int64),labels,teacher,configs)
    for name in ['robust_v3_support.py','prepare_robust_v3.py']:shutil.copy2(ROOT/name,ws/name)
    helper=(ws/'robust_v3_support.py').read_text().replace('    def evaluate(self,step,anchors):','    def evaluate(self,step,anchors):\n        if not self.cfg.get("validation_indices"):return')
    (ws/'robust_v3_support.py').write_text(helper)
    (ws/'src/model.py').write_text(builder.model_source())
    (ws/'engine/train_base.py').write_text(builder.accelerated_base())
    (ws/'engine/train_initial.py').write_text(prep.patch_trainer())
    source=builder.trainer_source().replace("    teacher = torch.from_numpy(np.load(HERE / 'train_teacher.npy')).cuda()",builder.optimizer_migration()+"    teacher = torch.from_numpy(np.load(HERE / 'train_teacher.npy')).cuda()")
    (ws/'engine/train_refine.py').write_text(source)
    for stage,until in [('stage1',4096),('stage2',2845),('refine',1707)]:
        previous={'stage2':'stage1','refine':'stage2'}.get(stage);last=ws/'runs'/stage/'last.pt'
        script={'stage1':'train_initial.py','stage2':'train_base.py','refine':'train_refine.py'}[stage]
        command=[sys.executable,str(ws/'engine'/script),'--workspace',str(ws),'--config',str(ws/f'{stage}.json'),'--run',stage,'--until',str(until)]
        if last.exists():
            state=torch.load(last,map_location='cpu',weights_only=False)
            if state['step']>=until:continue
            del state;command+=['--resume']
        elif previous:
            initial=ws/'runs'/previous/'last.pt';command+=['--restart-from',str(initial)]
            if stage=='refine':
                metadata=json.loads((ws/'PREPARED.json').read_text());metadata['initial_checkpoint_sha256']=prep.digest(initial)
                (ws/'PREPARED.json').write_text(json.dumps(metadata,indent=2))
        if stage=='refine':
            cfg=configs['refine.json'];cfg['training_indices_sha256']=prep.digest(ws/'data/training_global_indices.npy')
            (ws/'refine.json').write_text(json.dumps(cfg,indent=2))
        subprocess.run(command,check=True)
    state=torch.load(ws/'runs/refine/last.pt',map_location='cpu',weights_only=False)
    state['model'].update(state['robust']['shadow'])
    torch.save(dict(model=state['model'],config=state['config'],step=state['step'],weights='ema'),ws/'model.pt')
    print(ws/'model.pt')

if __name__=='__main__':main()
