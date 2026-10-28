"""Focused tests: equivalence, reliability gates, self exclusion, expiry and restore."""
import copy,io,sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent/'engine'))
from contrast_queue import ContrastQueue
from trusted_contrast import global_trusted_contrast
cfg=dict(capacity_per_class=2,max_age_updates=3,warmup_updates=1,min_confidence=.5,min_margin=.2,require_frozen_loo_agreement=True)
torch.manual_seed(7)
q=ContrastQueue(3,4,cfg,'cpu')
f=F.normalize(torch.randn(4,4),dim=-1).requires_grad_();y=torch.tensor([0,0,1,2]);ids=torch.arange(4);trusted=torch.ones(4,dtype=torch.bool)
a,_=q.loss(f,y,ids,trusted,.12,0);b,_=global_trusted_contrast(f,y,trusted,.12)
assert torch.allclose(a,b)
ga=torch.autograd.grad(a,f,retain_graph=True)[0];gb=torch.autograd.grad(b,f)[0];assert torch.allclose(ga,gb)
logits=F.one_hot(y,3).float()*10
assert q.enqueue(f,y,ids,logits,trusted,0)['enqueued']==0
assert q.enqueue(f,y,ids,logits,torch.zeros_like(trusted),1)['enqueued']==0
assert q.enqueue(f,y,ids,torch.zeros_like(logits),trusted,1)['enqueued']==0
assert q.enqueue(f,y,ids,logits,trusted,1)['enqueued']==4
# Reinsert same image: no duplicate slots.
q.enqueue(f,y,ids,logits,trusted,2);assert int((q.ids>=0).sum())==4
single=f[2:3].detach().requires_grad_()
loss,stats=q.loss(single,y[2:3],ids[2:3],trusted[:1],.12,2)
assert stats['positive_pairs']==0 and loss.item()==0
loss,stats=q.loss(single,y[2:3],torch.tensor([99]),trusted[:1],.12,2)
assert stats['memory_positive_pairs']==1 and torch.isfinite(loss)
loss.backward();assert torch.isfinite(single.grad).all() and not q.features.requires_grad
# Enforce class cap and expiration.
q.enqueue(f[:3],torch.zeros(3,dtype=torch.long),torch.tensor([10,11,12]),torch.tensor([[10.,0,0]]).expand(3,-1),trusted[:3],2)
assert int((q.ids[0]>=0).sum())==2
assert int(q.active(6).sum())==0
q.observe(dict(contrast_rows=4,positive_rows=2))
buffer=io.BytesIO();torch.save(q.state_dict(),buffer);buffer.seek(0)
r=ContrastQueue(3,4,cfg,'cpu');r.load_state_dict(torch.load(buffer,weights_only=False))
for key in ['features','ids','steps','pointer']:assert torch.equal(getattr(q,key),getattr(r,key))
assert q.totals==r.totals
l1,s1=q.loss(f,y,ids,trusted,.12,3);l2,s2=r.loss(f,y,ids,trusted,.12,3)
assert torch.equal(l1,l2) and s1==s2
q.enqueue(f,y,ids,logits,trusted,3);r.enqueue(f,y,ids,logits,trusted,3)
assert torch.equal(q.features,r.features) and torch.equal(q.ids,r.ids)
print('Queue tests passed: baseline loss/gradients, gates, self exclusion, cap, expiry, serialized continuation.')
