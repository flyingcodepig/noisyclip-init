"""Bind the semifinal TRAIN metadata to fresh single-GPU training configs."""
import json
from pathlib import Path
import torch
from fetch_weights import digest
ROOT=Path(__file__).resolve().parent

def main():
    metadata=json.loads((ROOT/'cache/train_candidate_labels.json').read_text())
    assert metadata['rows']==90197 and metadata['classes']==500 and metadata['reference_read'] is False
    candidate_hash=digest(ROOT/'cache/train_candidate_labels.npy')
    batch=json.loads((ROOT/'hardware_profile.json').read_text())['microbatch']
    assert batch in (12,24,48) and 192%batch==0
    for source,target in [('train_config_stage1.json','semifinal_stage1.json'),('train_config.json','semifinal_stage2.json')]:
        cfg=json.loads((ROOT/source).read_text())
        cfg.update(num_classes=500,batch=batch,gradient_accumulation=192//batch,workers=6,candidate_labels_sha256=candidate_hash)
        path=ROOT/target
        if path.exists():
            assert json.loads(path.read_text())==cfg, 'Existing run configuration differs'
        else:
            path.write_text(json.dumps(cfg,indent=2))
    import transformers,numpy,PIL,platform,subprocess
    actual=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,pillow=PIL.__version__,gpu=torch.cuda.get_device_name(0),microbatch=batch,gradient_accumulation=192//batch,effective_batch=192,contrast_scope=f'microbatch of {batch}; original four-GPU global scope was 192',train_rows=90197,classes=500)
    (ROOT/'semifinal_actual_environment.json').write_text(json.dumps(actual,indent=2))
    subprocess.run(['python','-m','pip','freeze'],stdout=(ROOT/'requirements.actual.txt').open('w'),check=True)
    print(json.dumps(actual),flush=True)

if __name__=='__main__': main()
