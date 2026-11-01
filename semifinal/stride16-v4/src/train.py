"""A-route training. One GPU A100; resume preserves optimizer, EMA and history."""
import argparse
import copy
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import Images
from model import AModel


def robust(logits, target, candidate, weights, q, floor):
    prob = logits.float().softmax(-1)
    choices = candidate.clamp_min(0)
    mass = prob.gather(1, choices)
    mass = (mass * (candidate >= 0)).sum(1).clamp(1e-8, 1)
    gce = (1 - mass.pow(q)) / q
    ce = F.cross_entropy(logits.float(), target, reduction="none")
    return weights[target] * ((1 - floor) * gce + floor * ce)


def lr_multiplier(step, cfg):
    main, tail, warm = cfg["main_steps"], cfg["tail_steps"], cfg["warmup_steps"]
    if step < warm: return (step + 1) / warm
    if step < main:
        x = (step - warm) / max(1, main - warm)
        return 0.05 + 0.95 * (1 + math.cos(math.pi * x)) / 2
    x = min(1, (step - main) / max(1, tail))
    return 0.05 - 0.04 * x


def save(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--official-weights", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, help="Stop and checkpoint after this step; useful for a disposable smoke run")
    p.add_argument("--micro-batch", type=int, help="Override microbatch while preserving effective batch 192")
    p.add_argument("--workers", type=int, help="Data loader workers")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.micro_batch is not None:
        if args.micro_batch <= 0 or 192 % args.micro_batch:
            raise ValueError("Microbatch must divide 192")
        cfg["batch_size"] = args.micro_batch
        cfg["accumulation"] = 192 // args.micro_batch
    if args.workers is not None:
        cfg["workers"] = args.workers
    prepared, run = Path(args.prepared), Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    rows = json.loads((prepared / "train_inventory.json").read_text(encoding="utf-8"))
    data = np.load(prepared / "train_inputs.npz")
    if len(rows) != cfg.get("expected_train", 90197) or len(data["labels"]) != len(rows): raise ValueError("Invalid train inventory")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA A100 required")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    model = AModel(args.official_weights, cfg["classes"], cfg["image_size"]).to(device)
    model.head.weight.data.copy_(torch.tensor(data["centers"], device=device))
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    anchors = {name: value.detach().clone() for name, value in model.backbone.named_parameters() if value.requires_grad}
    groups = {name: [] for name in ("head", "pool", "early", "late")}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.startswith("head") or name == "log_scale": group = "head"
        elif name in ("gate", "log_temperature"): group = "pool"
        elif "encoder.layers." in name:
            layer = int(name.split("encoder.layers.")[1].split(".")[0])
            group = "early" if layer < 8 else "late"
        elif "embeddings" in name or "pre_layrnorm" in name: group = "early"
        else: group = "late"
        groups[group].append(param)
    optim = torch.optim.AdamW([{"params": values, "lr": cfg["lr_" + name], "base_lr": cfg["lr_" + name]}
                               for name, values in groups.items() if values], weight_decay=cfg["weight_decay"])
    labels = torch.tensor(data["labels"].astype(np.int64), device=device)
    candidate = torch.tensor(data["candidate"].astype(np.int64), device=device)
    teacher = torch.tensor(data["teacher"].astype(np.float32), device=device)
    loo = torch.tensor(data["loo_agree"], device=device)
    canonical = torch.tensor(data["canonical"].astype(np.int64), device=device)
    queue = torch.zeros((cfg["classes"], cfg["queue_per_class"], 512), device=device)
    queue_age = torch.full((cfg["classes"], cfg["queue_per_class"]), -100000, device=device, dtype=torch.long)
    queue_canonical = torch.full_like(queue_age, -1)
    queue_next = torch.zeros(cfg["classes"], device=device, dtype=torch.long)
    count = torch.bincount(labels, minlength=cfg["classes"]).float()
    weights = ((1 - 0.999) / (1 - 0.999 ** count)).float()
    weights /= weights.mean()
    # Documented A route tracks repeat visits by identical image content.
    history_size = int(canonical.max()) + 1 if cfg.get("history_by_canonical", False) else len(labels)
    history_class = torch.full((history_size,), -1, device=device, dtype=torch.long)
    history_step = torch.full((history_size,), -100000, device=device, dtype=torch.long)
    history_ok = torch.zeros((history_size,), device=device, dtype=torch.bool)
    step = 0
    if args.resume:
        state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if state["config"] != cfg: raise ValueError("Resume config changed")
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
        optim.load_state_dict(state["optimizer"])
        history_class = state["history_class"].to(device); history_step = state["history_step"].to(device)
        history_ok = state["history_ok"].to(device)
        queue = state["queue"].to(device); queue_age = state["queue_age"].to(device)
        queue_canonical = state["queue_canonical"].to(device); queue_next = state["queue_next"].to(device)
        step = state["step"]
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
    dataset = Images(rows, cfg["seed"], train=True)
    batch = cfg["batch_size"]
    total = cfg["main_steps"] + cfg["tail_steps"]
    target_steps = total if args.stop_after is None else min(total, args.stop_after)
    permutations = {}
    while step < target_steps:
        # Build 128 exact optimizer steps. Sampler position is derived solely from step.
        sequence = []
        for position in range(step * cfg["accumulation"] * batch,
                              min(target_steps, step + 128) * cfg["accumulation"] * batch):
            epoch, offset = divmod(position, len(rows))
            if epoch not in permutations:
                permutations = {epoch: torch.randperm(len(rows), generator=torch.Generator().manual_seed(cfg["seed"] + epoch)).tolist()}
            sequence.append((permutations[epoch][offset], epoch))
        loader = DataLoader(dataset, batch_size=batch, sampler=sequence, num_workers=cfg["workers"], pin_memory=True, drop_last=True)
        iterator = iter(loader)
        while step < target_steps:
            optim.zero_grad(set_to_none=True)
            losses = []
            enqueue = []
            for micro in range(cfg["accumulation"]):
                try: strong, weak, y, ids = next(iterator)
                except StopIteration: break
                strong, weak, y, ids = strong.to(device), weak.to(device), y.to(device), ids.to(device)
                mixing = random.random()
                if mixing < cfg["mixup_probability"] + cfg["cutmix_probability"]:
                    perm = torch.randperm(len(y), device=device)
                    if mixing < cfg["mixup_probability"]:
                        lam = np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"])
                        strong = lam * strong + (1 - lam) * strong[perm]
                    else:
                        lam = np.random.beta(cfg["cutmix_alpha"], cfg["cutmix_alpha"])
                        side = strong.shape[-1]
                        width = int(side * math.sqrt(1 - lam))
                        cx, cy = random.randrange(side), random.randrange(side)
                        x0, x1 = max(0, cx - width // 2), min(side, cx + width // 2)
                        y0, y1 = max(0, cy - width // 2), min(side, cy + width // 2)
                        strong[:, :, y0:y1, x0:x1] = strong[perm, :, y0:y1, x0:x1]
                        lam = 1 - (x1 - x0) * (y1 - y0) / side**2
                    target_feature = F.normalize(lam * teacher[ids] + (1 - lam) * teacher[ids[perm]], dim=-1)
                else:
                    lam, perm, target_feature = None, None, teacher[ids]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, feature, _ = model(strong)
                    if lam is None:
                        classification = robust(logits, y, candidate[ids], weights, cfg["gce_q"], cfg["ce_floor"]).mean()
                    else:
                        a = robust(logits, y, candidate[ids], weights, cfg["mix_gce_q"], cfg["ce_floor"])
                        b = robust(logits, y[perm], candidate[ids[perm]], weights, cfg["mix_gce_q"], cfg["ce_floor"])
                        classification = (lam * a + (1 - lam) * b).mean()
                    frozen_loss = F.smooth_l1_loss(feature, target_feature, beta=0.1)
                    loss = classification + cfg["frozen_weight"] * frozen_loss
                    if lam is None:
                        student_prob = logits.float().softmax(-1)
                        student_top = student_prob.topk(2, -1).values
                        trusted = (student_prob.argmax(-1) == y) & (student_top[:, 0] >= 0.5) & ((student_top[:, 0] - student_top[:, 1]) >= 0.2) & loo[ids]
                        if step >= cfg["queue_start"] and trusted.any():
                            valid = (step - queue_age).clamp_min(0) <= cfg["queue_max_age"]
                            flat_feature = queue[valid].detach()
                            flat_label = torch.arange(cfg["classes"], device=device)[:, None].expand_as(valid)[valid]
                            flat_canonical = queue_canonical[valid]
                            if len(flat_feature):
                                anchor = F.normalize(feature[trusted].float(), dim=-1)
                                similarity = anchor @ flat_feature.T / cfg["supcon_temperature"]
                                distinct = canonical[ids[trusted], None] != flat_canonical[None]
                                similarity = similarity.masked_fill(~distinct, -1e4)
                                positive = (y[trusted, None] == flat_label[None]) & distinct
                                usable = positive.any(-1)
                                if usable.any():
                                    logp = similarity.log_softmax(-1)
                                    contrast = -(logp * positive).sum(-1) / positive.sum(-1).clamp_min(1)
                                    loss = loss + cfg["supcon_weight"] * contrast[usable].mean()
                        enqueue.append((feature[trusted].detach(), y[trusted].detach(), canonical[ids[trusted]].detach()))
                    with torch.no_grad():
                        ema_logits, ema_feature, _ = ema(weak if lam is None else strong)
                    if lam is not None:
                        loss = loss + cfg["ema_feature_weight"] * F.smooth_l1_loss(feature, ema_feature, beta=0.1)
                    elif step >= cfg["soft_start"]:
                        prob = ema_logits.float().softmax(-1)
                        top2 = prob.topk(2, -1).values
                        pred = prob.argmax(-1)
                        good = (top2[:, 0] >= 0.9) & ((top2[:, 0] - top2[:, 1]) >= 0.4)
                        history_ids = canonical[ids] if cfg.get("history_by_canonical", False) else ids
                        stable = good & history_ok[history_ids] & (history_class[history_ids] == pred) & ((step - history_step[history_ids]) <= cfg["soft_max_gap"]) & (history_step[history_ids] < step)
                        if stable.any():
                            class_freq = torch.bincount(pred[stable], minlength=cfg["classes"]).clamp_min(1)
                            class_w = 1 / class_freq[pred[stable]].float()
                            class_w *= stable.sum() / class_w.sum()
                            soft_ce = -(prob[stable] * F.log_softmax(logits[stable].float(), -1)).sum(-1)
                            ramp = min(1, (step - cfg["soft_start"]) / cfg["soft_ramp"])
                            loss = loss + cfg["soft_weight"] * ramp * (soft_ce * class_w).sum() / len(y)
                        history_class[history_ids], history_step[history_ids], history_ok[history_ids] = pred, step, good
                    sp_terms = [((param - anchors[name]) ** 2).sum() for name, param in model.backbone.named_parameters() if name in anchors]
                    sp_count = sum(param.numel() for name, param in model.backbone.named_parameters() if name in anchors)
                    sp = torch.stack(sp_terms).sum() / sp_count
                    loss = loss + cfg["l2sp_weight"] * sp
                (loss / cfg["accumulation"]).backward()
                losses.append(float(loss.detach()))
            if len(losses) != cfg["accumulation"]: break
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            factor = lr_multiplier(step, cfg)
            for group in optim.param_groups: group["lr"] = group["base_lr"] * factor
            optim.step()
            with torch.no_grad():
                for feat, lab, cid in enqueue:
                    for f, label, canon in zip(feat, lab, cid):
                        target = int(label); slot = int(queue_next[target])
                        queue[target, slot] = F.normalize(f.float(), dim=-1)
                        queue_age[target, slot] = step
                        queue_canonical[target, slot] = canon
                        queue_next[target] = (slot + 1) % cfg["queue_per_class"]
            with torch.no_grad():
                for e, s in zip(ema.parameters(), model.parameters()): e.lerp_(s, 1 - cfg["ema_decay"])
            step += 1
            if step % 50 == 0: print(json.dumps({"step": step, "loss": sum(losses) / len(losses), "lr_factor": factor}), flush=True)
            if step % 256 == 0 or step in (cfg["main_steps"], total, target_steps):
                state = {"config": cfg, "step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                         "optimizer": optim.state_dict(), "history_class": history_class.cpu(), "history_step": history_step.cpu(),
                         "history_ok": history_ok.cpu(), "queue": queue.cpu(), "queue_age": queue_age.cpu(),
                         "queue_canonical": queue_canonical.cpu(), "queue_next": queue_next.cpu(),
                         "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                         "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
                save(run / "last.pt", state)
                if step == total: save(run / "final_ema.pt", {"config": cfg, "step": step, "model": ema.state_dict()})


if __name__ == "__main__": main()
