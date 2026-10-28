"""Class-capped detached contrastive memory; an adaptation, not full SNSCL."""
import torch
from torch.nn import functional as F

class ContrastQueue:
    def __init__(self, classes, dim, cfg, device):
        self.cfg=dict(cfg);self.classes=classes;self.dim=dim
        shape=(classes,cfg['capacity_per_class'])
        self.features=torch.zeros(*shape,dim,device=device)
        self.ids=torch.full(shape,-1,dtype=torch.long,device=device)
        self.steps=torch.full_like(self.ids,-1)
        self.pointer=torch.zeros(classes,dtype=torch.long,device=device)
        self.totals={}

    def active(self,step):
        return (self.ids>=0)&(self.steps<=step)&((step-self.steps)<=self.cfg['max_age_updates'])

    def loss(self,features,labels,ids,trusted,temperature,step):
        active=self.active(step)
        memory=self.features[active].detach()
        memory_ids=self.ids[active]
        memory_labels=torch.arange(self.classes,device=features.device)[:,None].expand_as(active)[active]
        keys=torch.cat([features,memory]);key_labels=torch.cat([labels,memory_labels]);key_ids=torch.cat([ids,memory_ids])
        key_trust=torch.cat([trusted,torch.ones_like(memory_labels,dtype=torch.bool)])
        # Same image must not be a positive OR a negative, including historical entries.
        excluded=ids[:,None]==key_ids[None,:]
        positive=(labels[:,None]==key_labels[None,:])&trusted[:,None]&key_trust[None,:]&~excluded
        count=positive.sum(-1)
        logp=(features@keys.T/temperature).masked_fill(excluded,-10000.).log_softmax(-1)
        row_loss=-(logp*positive).sum(-1)/count.clamp_min(1)
        valid=count>0
        loss=(row_loss*valid).sum()/valid.sum().clamp_min(1)
        return loss,dict(contrast_rows=len(labels),positive_rows=int(valid.sum()),positive_pairs=int(positive.sum()),
            within_batch_positive_rows=int(positive[:,:len(labels)].any(-1).sum()),memory_positive_pairs=int(positive[:,len(labels):].sum()))

    @torch.no_grad()
    def enqueue(self,features,labels,ids,logits,frozen_agreement,step):
        values,pred=logits.detach().float().softmax(-1).topk(2,dim=-1)
        eligible=(pred[:,0]==labels)&(values[:,0]>=self.cfg['min_confidence'])&((values[:,0]-values[:,1])>=self.cfg['min_margin'])
        if self.cfg['require_frozen_loo_agreement']:eligible &= frozen_agreement.bool()
        if step<self.cfg['warmup_updates']:eligible &= False
        normalized=F.normalize(features.detach().float(),dim=-1)
        for i in torch.where(eligible)[0].tolist():
            c=int(labels[i]);old=torch.where(self.ids[c]==ids[i])[0]
            if len(old):slot=int(old[0])
            else:
                slot=int(self.pointer[c]);self.pointer[c]=(slot+1)%self.cfg['capacity_per_class']
            self.features[c,slot].copy_(normalized[i]);self.ids[c,slot]=ids[i];self.steps[c,slot]=step
        return dict(enqueue_candidates=len(labels),enqueued=int(eligible.sum()))

    def observe(self,metrics):
        for k,v in metrics.items():self.totals[k]=self.totals.get(k,0)+v

    def flush_metrics(self,step):
        active=self.active(step);out={'queue_'+k:v for k,v in self.totals.items()};n=max(1,self.totals.get('contrast_rows',0))
        out.update(queue_positive_row_fraction=self.totals.get('positive_rows',0)/n,
            queue_within_batch_positive_row_fraction=self.totals.get('within_batch_positive_rows',0)/n,
            queue_active_entries=int(active.sum()),queue_covered_classes=int(active.any(-1).sum()))
        self.totals={};return out

    def state_dict(self):
        return dict(version=1,cfg=self.cfg,classes=self.classes,dim=self.dim,totals=dict(self.totals),
            **{k:getattr(self,k).detach().cpu().clone() for k in ['features','ids','steps','pointer']})

    def load_state_dict(self,state):
        assert state['version']==1 and state['cfg']==self.cfg and state['classes']==self.classes and state['dim']==self.dim
        for k in ['features','ids','steps','pointer']:
            src=state[k];dst=getattr(self,k);assert src.shape==dst.shape and src.dtype==dst.dtype
            if src.is_floating_point():assert torch.isfinite(src).all()
            dst.copy_(src.to(dst.device))
        assert (self.ids>=-1).all() and ((self.pointer>=0)&(self.pointer<self.cfg['capacity_per_class'])).all()
        self.totals=dict(state['totals'])
