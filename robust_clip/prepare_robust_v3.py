import math
from pathlib import Path
import prepare_optimization as prep
ROOT=Path(__file__).resolve().parent


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)

def trainer_source():
    s = prep.patch_trainer()
    s = replace_once(s, 'from model_base import load_model', 'from model_base import load_model\nfrom robust_v3_support import RobustState, attention_context, cuda_prefetch')
    s = replace_once(s, "weight_decay=cfg['wd'])", "weight_decay=cfg['wd'], fused=True)")
    s = replace_once(s, 'with sdpa_kernel(SDPBackend.MATH), torch.autocast', 'with attention_context(), torch.autocast')
    s = replace_once(s, '    own_features, own_probability = (None, None)', "    robust=RobustState(model,cfg,ROOT,out,global_indices,labels,checkpoint.get('robust') if args.resume else None)\n    if robust.last_refresh<0:robust.refresh(start)\n    own_features, own_probability = (None, None)")
    s = replace_once(s, 'rng=rngs, official_initialization=True', 'rng=rngs, robust=robust.state_dict(), official_initialization=True')
    s = replace_once(s, '            if sample_weights is not None:', "            if cfg.get('robust_weights'):\n                cls_loss=robust.classification(logits,y,index,coefficient,perm,mixed)\n            local_consistency=robust.local_loss(logits,y,index,mixed,step)\n            if sample_weights is not None:")
    s = replace_once(s, "global_trusted_contrast(features, y, logits.detach().argmax(-1) == y, cfg['supcon_temperature'])", "global_trusted_contrast(features, y, (logits.detach().argmax(-1) == y) & (robust.weights[index]>=.6), cfg['supcon_temperature'])")
    s = replace_once(s, '            assert torch.isfinite(loss)', "            loss=loss+cfg.get('local_consistency_weight',0)*local_consistency\n            assert torch.isfinite(loss)")
    s = replace_once(s, '            optimizer.step()', '            optimizer.step()\n            robust.update()')
    s = replace_once(s, 'for batch_no, batch in enumerate(loader):', 'for batch_no, batch in enumerate(cuda_prefetch(loader)):')
    s = replace_once(s, "            assert torch.isfinite(loss), f'Nonfinite training loss at step {step}'", "            torch._assert_async(torch.isfinite(loss), f'Nonfinite training loss at step {step}')")
    s = replace_once(s, "            assert torch.isfinite(norm), f'Nonfinite gradient at step {step}'", "            torch._assert_async(torch.isfinite(norm), f'Nonfinite gradient at step {step}')")
    s = replace_once(s, 'rolling.append([float(loss.detach()), float(cls_loss.detach()), float(drift.detach()), float(anchor.detach()), float(norm)])', 'rolling.append(torch.stack([loss.detach(),cls_loss.detach(),drift.detach(),anchor.detach(),norm.detach()]))')
    s = replace_once(s, 'average = np.mean(rolling[-25:], axis=0).tolist()', 'average = torch.stack(rolling[-25:]).mean(0).cpu().tolist()')
    s = replace_once(s, 'geometry_loss=float(geometry.detach()), seconds=', 'geometry_loss=float(geometry.detach()), local_consistency=float(local_consistency.detach()), ema_updates=robust.updates, peak_vram_gb=torch.cuda.max_memory_allocated()/2**30, seconds=')
    s = replace_once(s, "            if step % cfg['save_every'] == 0 or step == args.until:", "            if step % cfg.get('reliability_refresh_every',768)==0 and step<args.until:\n                robust.refresh(step)\n            if step % cfg['save_every'] == 0 or step == args.until:")
    s = replace_once(s, '                save()\n                evaluate_holdout(model,cfg,ROOT,out,step,anchors)', '                robust.evaluate(step,anchors)\n                save()')
    s = replace_once(s, '    step, epoch = (start, start // steps_per_epoch)', "    if cfg.get('robust_weights') and not (out/'initial_reference/best.json').exists():\n        reference=out/'initial_reference';reference.mkdir(exist_ok=True)\n        evaluate_holdout(model,cfg,ROOT,reference,0,anchors)\n    step, epoch = (start, start // steps_per_epoch)")
    s = replace_once(s, '        evaluate_holdout(model,cfg,ROOT,out,step,anchors)', '        robust.evaluate(step,anchors)')
    compile(s, 'train_robust_v3.py', 'exec')
    return s

def model_source():
    s = (ROOT / 'src/model.py').read_text()
    s = replace_once(s, '        self.gate =', "        self.local_enabled=bool(config.get('local_consistency_weight',0))\n        self.gate =")
    old = '        return self.classify(embedding)'
    new = "        result=self.classify(embedding)\n        self._region_logits=None\n        if self.training and self.local_enabled:\n            side=math.isqrt(patch.shape[1]);assert side*side==patch.shape[1]\n            coordinates=torch.arange(side,device=patch.device)\n            yy,xx=torch.meshgrid(coordinates,coordinates,indexing='ij')\n            coords=torch.stack((yy.flatten(),xx.flatten()),dim=-1)\n            first=cosine.detach().argmax(-1)\n            distance=(coords[None]-coords[first,None]).abs().amax(-1)\n            second=cosine.detach().masked_fill(distance<side//2,-1e4).argmax(-1)\n            region_logits=[]\n            for center in [first,second]:\n                spatial=(coords[None]-coords[center,None]).abs().amax(-1)<=max(1,side//4)\n                region_score=cosine/self.log_temperature.exp().clamp(.025,.35)\n                local_attention=region_score.masked_fill(~spatial,-1e4).softmax(-1).to(patch.dtype)\n                local_foreground=(patch*local_attention[:,:,None]).sum(1)\n                local_pooled=cls+self.gate_bound*self.gate.tanh()*self.contrast_norm(local_foreground-background)\n                local_embedding=self.backbone.visual_projection(self.backbone.vision_model.post_layernorm(local_pooled))\n                region_logits.append(self.classify(local_embedding)[0])\n            self._region_logits=region_logits\n        return result"
    s = replace_once(s, old, new)
    compile(s, 'model.py', 'exec')
    return s

def accelerated_base():
    from prepare_optimization import patch_trainer
    from prepare_robust_v3 import replace_once as rep
    s = patch_trainer()
    s = rep(s, 'from model_base import load_model', 'from model_base import load_model\nfrom robust_v3_support import attention_context,cuda_prefetch')
    s = rep(s, "weight_decay=cfg['wd'])", "weight_decay=cfg['wd'],fused=True)")
    s = rep(s, 'with sdpa_kernel(SDPBackend.MATH), torch.autocast', 'with attention_context(), torch.autocast')
    s = rep(s, 'for batch_no, batch in enumerate(loader):', 'for batch_no, batch in enumerate(cuda_prefetch(loader)):')
    s = rep(s, "            assert torch.isfinite(loss), f'Nonfinite training loss at step {step}'", "            torch._assert_async(torch.isfinite(loss), f'Nonfinite training loss at step {step}')")
    s = rep(s, "            assert torch.isfinite(norm), f'Nonfinite gradient at step {step}'", "            torch._assert_async(torch.isfinite(norm), f'Nonfinite gradient at step {step}')")
    s = rep(s, 'rolling.append([float(loss.detach()), float(cls_loss.detach()), float(drift.detach()), float(anchor.detach()), float(norm)])', 'rolling.append(torch.stack([loss.detach(),cls_loss.detach(),drift.detach(),anchor.detach(),norm.detach()]))')
    s = rep(s, 'average = np.mean(rolling[-25:], axis=0).tolist()', 'average = torch.stack(rolling[-25:]).mean(0).cpu().tolist()')
    s = rep(s, "    teacher = torch.from_numpy(np.load(HERE / 'train_teacher.npy')).cuda()", optimizer_migration() + "    teacher = torch.from_numpy(np.load(HERE / 'train_teacher.npy')).cuda()")
    compile(s, 'train_full_base.py', 'exec')
    return s

def optimizer_migration():
    return "    for group in optimizer.param_groups:group['fused']=True\n    for parameter,state in optimizer.state.items():\n        if 'step' in state:state['step']=state['step'].to(device=parameter.device)\n"
