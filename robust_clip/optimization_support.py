"""Training-only index mapping, gradual unfreezing and fixed holdout evaluation."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

def allowed_parameter(name,cfg):
    if not cfg.get('gradual_unfreeze'):return True
    if name.startswith('backbone.'):
        if name.startswith('backbone.vision_model.post_layernorm.'):return True
        prefix='backbone.vision_model.encoder.layers.'
        if name.startswith(prefix):return int(name[len(prefix):].split('.')[0])>=12-cfg['unfreeze_last_blocks']
        return False
    return True

def configure_optimizer_scope(model,cfg):
    if cfg.get('gradual_unfreeze'):
        for name,p in model.named_parameters():p.requires_grad_(p.requires_grad and allowed_parameter(name,cfg))

def set_training_phase(model,cfg,step,out):
    if not cfg.get('gradual_unfreeze'):return
    warm=step<int(cfg.get('head_warmup_steps',0))
    phase='head_only' if warm else 'last_blocks'
    for name,p in model.named_parameters():
        enabled=allowed_parameter(name,cfg) and name!='backbone.visual_projection.weight'
        if warm:enabled=enabled and (name.startswith('head.') or name=='log_scale')
        p.requires_grad_(enabled)
    if getattr(model,'_optimization_phase',None)!=phase:
        model._optimization_phase=phase
        event=dict(step=step,phase=phase,trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        with (out/'phase_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        print(json.dumps(dict(stage='training_phase',**event)),flush=True)

def adjust_unfreeze_lr(group,cfg,step):
    warm=int(cfg.get('head_warmup_steps',0))
    if cfg.get('gradual_unfreeze') and warm and group['name']!='head':
        group['lr']*=max(0.,min(1.,(step-warm+1)/cfg.get('unfreeze_warmup_steps',128)))

def load_training_indices(root,cfg,labels):
    indices=np.load(root/cfg['training_indices'])
    assert indices.ndim==1 and len(indices)==len(labels)==len(np.unique(indices))
    if cfg.get('validation_indices'):
        validation=np.load(root/cfg['validation_indices'])
        assert not np.intersect1d(indices,validation).size,'Train/validation image leakage'
    source_labels=np.load(root/'data/source_labels.npy')
    assert np.array_equal(labels,source_labels[indices]),'Global cache indices and local labels differ'
    return indices

def evaluate_holdout(model,cfg,root,out,step,anchors):
    if not cfg.get('validation_indices'):return
    status=out/'validation_status.json'
    if status.exists():
        previous=json.loads(status.read_text())
        if previous.get('stage')=='completed' and previous.get('step')==step and (out/f'validation_step_{step:06d}.npz').exists():return
    status.write_text(json.dumps(dict(stage='evaluating',step=step,unix_time=time.time())))
    indices=np.load(root/cfg['validation_indices'])
    labels=np.load(root/'data/source_labels.npy')[indices]
    cache=np.load(root/'image_cache/semi_train_512.npy',mmap_mode='r')
    from infer import pixels
    # Original complete and center 1.0 were verified identical for ALL heldout and test images.
    views=[(1.,False,.39),(.86,True,.21),(.74,False,.40)]
    allprob=[];viewprob=[[],[],[]];was_training=model.training
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
    try:
        model.eval()
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            for start in range(0,len(indices),48):
                selected=indices[start:start+48];posterior=None
                for view_no,(crop,flip,weight) in enumerate(views):
                    images=pixels(cache,selected,cfg['size'],crop,flip)
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(images)
                    probability=logits.float().softmax(-1).clamp_min(1e-9).pow(1/2.85)
                    probability/=probability.sum(-1,keepdim=True)
                    viewprob[view_no].append(probability.cpu().numpy())
                    posterior=probability*weight if posterior is None else posterior+probability*weight
                allprob.append(posterior.cpu().numpy())
        probability=np.concatenate(allprob);prediction=probability.argmax(-1);correct=prediction==labels
        top5=np.argpartition(probability,-5,axis=1)[:,-5:]
        np.savez(out/f'validation_step_{step:06d}.npz',indices=indices,labels=labels,prediction=prediction,
            top5=top5,confidence=probability.max(-1))
        per_class=[float(correct[labels==c].mean()) for c in np.unique(labels)]
        counts=np.bincount(np.load(root/'engine/train_labels.npy'),minlength=500)
        grouped={}
        for name,mask in [('tail_under50',counts[labels]<50),('mid_50_149',(counts[labels]>=50)&(counts[labels]<150)),('head_150plus',counts[labels]>=150)]:
            grouped[name]=dict(rows=int(mask.sum()),top1=float(correct[mask].mean()) if mask.any() else None)
        record=dict(step=step,rows=len(labels),top1=float(correct.mean()),correct=int(correct.sum()),macro=float(np.mean(per_class)),
            top5=float((top5==labels[:,None]).any(-1).mean()),
            nll=float(-np.log(np.maximum(probability[np.arange(len(labels)),labels],1e-12)).mean()),groups=grouped,
            training_phase=getattr(model,'_optimization_phase','full_backbone'),independent_of_this_run_training=True,
            labels='original noisy training labels; not clean or online accuracy',unix_time=time.time())
        if cfg.get('gradual_unfreeze') and step<=cfg.get('head_warmup_steps',0):
            deviation=max(float((p.detach()-anchors[n]).abs().max()) for n,p in model.backbone.named_parameters() if n in anchors)
            assert deviation==0.,'Frozen backbone changed during head warmup'
            record['frozen_backbone_max_abs_change']=deviation
        with (out/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        best_path=out/'best.json';best=json.loads(best_path.read_text()) if best_path.exists() else None
        improved=best is None or (record['top1'],-record['nll'])>(best['top1'],-best['nll'])
        if improved or (best.get('step')==step and not (out/'best_predictions.npz').exists()):
            torch.save(dict(model=model.state_dict(),config=cfg,step=step,validation=record),out/'best.tmp.pt')
            (out/'best.tmp.pt').replace(out/'best.pt')
            best_path.write_text(json.dumps(record,indent=2))
            np.savez(out/'best_predictions.npz',indices=indices,labels=labels,prediction=prediction,
                probabilities=probability,view_probabilities=np.stack([np.concatenate(v) for v in viewprob]),step=step)
        status.write_text(json.dumps(dict(stage='completed',**record),indent=2))
        print(json.dumps(dict(stage='validation',**record)),flush=True)
    finally:
        model.train(was_training)
        random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state(saved[3])
