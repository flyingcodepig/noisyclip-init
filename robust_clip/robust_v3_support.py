"""Training-subset reliability, GPU EMA, and spatial-region consistency."""
import contextlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.data import Dataset, DataLoader

class CacheRows(Dataset):
    def __init__(self,path,indices):
        self.cache=np.load(path,mmap_mode='r');self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        return torch.from_numpy(np.array(self.cache[int(self.indices[index])],copy=True))

def cached_batches(root,indices,batch,workers=8):
    return DataLoader(CacheRows(root/'image_cache/semi_train_512.npy',indices),batch_size=batch,
                      num_workers=workers,pin_memory=True,prefetch_factor=3,shuffle=False)

def upload_pixels(raw):
    return raw.to('cuda',non_blocking=True).permute(0,3,1,2).float().div_(255)

def view_pixels(base,size,crop=1.,flip=False):
    side=max(32,round(base.shape[-1]*crop));offset=(base.shape[-1]-side)//2
    image=base[:,:,offset:offset+side,offset:offset+side]
    image=F.interpolate(image,size=(size,size),mode='bicubic',align_corners=False,antialias=True)
    if flip:image=image.flip(-1)
    mean=image.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
    std=image.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
    return (image-mean)/std

def attention_context():
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])

def cuda_prefetch(loader):
    """Overlap the next pinned batch's H2D transfer with the current batch."""
    iterator=iter(loader);stream=torch.cuda.Stream()
    def load():
        try:batch=next(iterator)
        except StopIteration:return None
        with torch.cuda.stream(stream):
            for key in ['image','label','index']:
                batch[key]=batch[key].to('cuda',non_blocking=True)
        return batch
    pending=load()
    while pending is not None:
        torch.cuda.current_stream().wait_stream(stream)
        current=pending
        for key in ['image','label','index']:current[key].record_stream(torch.cuda.current_stream())
        pending=load()
        yield current

class RobustState:
    def __init__(self, model, cfg, root, out, indices, labels, resume=None):
        self.model,self.cfg,self.root,self.out=model,cfg,root,out
        self.indices,self.labels=indices,labels
        self.enabled=cfg.get('robust_weights',False)
        self.shadow={n:p.detach().clone() for n,p in model.named_parameters()}
        self.weights=torch.ones(len(labels),device='cuda')
        self.posterior=None;self.updates=0;self.last_refresh=-1
        self.prototype=self.prototype_score()
        if resume:
            self.shadow={n:t.to('cuda') for n,t in resume['shadow'].items()}
            self.weights=resume['weights'].to('cuda')
            self.posterior=resume['posterior'].to('cuda') if resume['posterior'] is not None else None
            self.updates=resume['updates'];self.last_refresh=resume['last_refresh']

    def prototype_score(self):
        if not self.enabled:return None
        feat=torch.from_numpy(np.load(self.root/'engine/train_teacher.npy')).cuda().float()
        feat=F.normalize(feat,dim=-1);labels=torch.as_tensor(self.labels,device='cuda')
        centers=torch.from_numpy(np.load(self.root/'engine/initial_centers.npy')).cuda().float()
        similarity=(feat*centers[labels]).sum(-1)
        # Trim only while estimating a prototype: every image remains in training.
        robust=[]
        for c in range(500):
            mask=labels==c;values=similarity[mask];selected=feat[mask][values>=torch.quantile(values,.4)]
            robust.append(F.normalize(selected.mean(0),dim=0))
        centers=torch.stack(robust);score=[]
        for start in range(0,len(labels),4096):
            y=labels[start:start+4096];sim=feat[start:start+4096]@centers.T
            own=sim.gather(1,y[:,None]).squeeze(1)
            sim.scatter_(1,y[:,None],-2.)
            score.append(torch.sigmoid((own-sim.max(-1).values)/.035))
        return torch.cat(score)

    @contextlib.contextmanager
    def ema_model(self):
        # No architecture changes, no independent-model voting at inference.
        saved={n:p.detach().clone() for n,p in self.model.named_parameters()}
        with torch.no_grad():
            for n,p in self.model.named_parameters():p.copy_(self.shadow[n])
        try:yield self.model
        finally:
            with torch.no_grad():
                for n,p in self.model.named_parameters():p.copy_(saved[n])

    @torch.no_grad()
    def update(self):
        self.updates+=1
        decay=min(self.cfg.get('ema_decay',.995),(1+self.updates)/(10+self.updates))
        params=dict(self.model.named_parameters())
        torch._foreach_lerp_(list(self.shadow.values()),[params[n].detach() for n in self.shadow],1-decay)

    def state_dict(self):
        return dict(shadow=self.shadow,weights=self.weights,posterior=self.posterior,
                    updates=self.updates,last_refresh=self.last_refresh)

    @torch.no_grad()
    def refresh(self,step):
        if not self.enabled or self.last_refresh==step:return
        shared=self.root.parent/'initial_reliability.npz'
        if step==0 and shared.exists():
            with np.load(shared) as data:
                assert np.array_equal(data['indices'],self.indices)
                assert np.array_equal(data['labels'],self.labels)
                assert str(data['initial_checkpoint_sha256'])==json.loads((self.root/'PREPARED.json').read_text())['initial_checkpoint_sha256']
                self.weights=torch.from_numpy(data['weights']).cuda()
                self.posterior=torch.from_numpy(data['posterior']).cuda()
            self.last_refresh=0
            print(json.dumps(dict(stage='initial_reliability_reused',rows=len(self.indices),validation_used=False,test_used=False)),flush=True)
            return
        saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
        was_training=self.model.training;started=time.monotonic();self.model.eval()
        report=dict(stage='reliability_sweep',step=step,rows=len(self.indices),validation_used=False,test_used=False)
        (self.out/'reliability_status.json').write_text(json.dumps(report));print(json.dumps(report),flush=True)
        allprob=[];allscore=[];batch=self.cfg.get('inference_batch',96)
        try:
            with self.ema_model(),attention_context():
                for batch_no,raw in enumerate(cached_batches(self.root,self.indices,batch)):
                    start=batch_no*batch;views=[];base=upload_pixels(raw)
                    for crop,flip in [(1.,False),(.86,True)]:
                        x=view_pixels(base,self.cfg['size'],crop,flip)
                        with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=self.model(x)
                        views.append((logits.float()/2.).softmax(-1))
                    p,q=views;m=(p+q)*.5
                    y=torch.as_tensor(self.labels[start:start+batch],device='cuda')
                    ratio=(p.gather(1,y[:,None]).squeeze(1)*q.gather(1,y[:,None]).squeeze(1)).sqrt()
                    ratio/=((p.max(-1).values*q.max(-1).values).sqrt()+1e-8)
                    js=.5*((p*(p.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(-1)+(q*(q.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(-1))
                    agreement=(1-js/math.log(2)).clamp(0,1)
                    trust=.65*ratio+.15*ratio*agreement+.20*self.prototype[start:start+batch]
                    allscore.append(trust);allprob.append(m.to(torch.float16))
                    if start%(batch*100)==0:
                        print(json.dumps(dict(stage='reliability_progress',step=step,done=min(start+batch,len(self.indices)),total=len(self.indices))),flush=True)
            score=torch.cat(allscore);new=.30+.70*score
            self.weights=new if self.last_refresh<0 else .8*self.weights+.2*new
            self.posterior=torch.cat(allprob);self.last_refresh=step
            if step==0:
                temporary=shared.with_suffix('.tmp.npz')
                np.savez(temporary,indices=self.indices,labels=self.labels,weights=self.weights.cpu().numpy(),
                         posterior=self.posterior.cpu().numpy(),
                         initial_checkpoint_sha256=json.loads((self.root/'PREPARED.json').read_text())['initial_checkpoint_sha256'])
                temporary.replace(shared)
            w=self.weights.cpu().numpy();counts=np.bincount(self.labels,minlength=500)
            means=np.bincount(self.labels,weights=w,minlength=500)/counts
            report.update(stage='completed',seconds=time.monotonic()-started,min=float(w.min()),mean=float(w.mean()),max=float(w.max()),
                          quantiles=np.quantile(w,[0,.1,.5,.9,1]).tolist(),min_class_mean=float(means.min()),
                          hard_deleted=0,labels_modified=0,indices_sha256=self.cfg['training_indices_sha256'])
            np.savez(self.out/f'reliability_{step:06d}.npz',indices=self.indices,weights=w,labels=self.labels,class_mean_weights=means)
            (self.out/'reliability_status.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        finally:
            self.model.train(was_training)
            random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state(saved[3])

    def classification(self,logits,y,index,coefficient,perm,mixed):
        logp=logits.float().log_softmax(-1);w=self.weights[index]
        a=-logp.gather(1,y[:,None]).squeeze(1)
        if mixed:
            b=-logp.gather(1,y[perm,None]).squeeze(1);other=w[perm]
            return (coefficient*w*a+(1-coefficient)*other*b).mean()/self.weights.mean().detach()
        q=self.cfg['gce_q'];floor=self.cfg.get('gce_ce_floor',.025)
        per=(1-floor)*(1-(-q*a).exp())/q+floor*a
        return (w*per).mean()/self.weights.mean().detach()

    def local_loss(self,logits,y,index,mixed,step):
        zero=logits.new_zeros(())
        if mixed or not self.cfg.get('local_consistency_weight',0):return zero
        locals_=getattr(self.model,'_region_logits',None)
        if not locals_:return zero
        target=(logits.detach().float()/2.).softmax(-1)
        if self.posterior is not None:target=.5*target+.5*self.posterior[index].float()
        target=target/target.sum(-1,keepdim=True)
        reliability=self.weights[index]
        mask=(logits.detach().argmax(-1)==y)&(reliability>=.6)
        # Keep the loss graph even when no sample is selected.
        terms=[F.kl_div((z.float()/2.).log_softmax(-1),target,reduction='none').sum(-1)*4 for z in locals_]
        per=torch.stack(terms).mean(0)
        return (per*reliability*mask).sum()/mask.sum().clamp_min(1)*min(1.,step/256)

    def evaluate(self,step,anchors):
        from optimization_support import evaluate_holdout
        evaluate_holdout(self.model,self.cfg,self.root,self.out,step,anchors)
        ema_out=self.out/'ema';ema_out.mkdir(exist_ok=True)
        with self.ema_model():evaluate_holdout(self.model,self.cfg,self.root,ema_out,step,anchors)

