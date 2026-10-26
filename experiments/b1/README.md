# B1: Trust-driven robust fine-tuning

B1 builds on `baselines/second_round` (B0), whose reported score is 78–79.
**B1 is an implementation, not a validated score improvement.** Its new settings
are transparent starting defaults, not parameters selected against TEST labels.
The B0 snapshot remains unchanged.

## What changes

1. Hash TRAIN image bytes with SHA-256 to form exact duplicate groups.
2. Cross-fit three lightweight ridge classification heads on **frozen official
   CLIP features**, assigning each entire duplicate group to one fold. All head
   fitting and neighbor voting for a held-out fold use only the other folds.
3. Average features within each group and give each reference group one vote.
   Combine out-of-fold probabilities and neighbor evidence to estimate trust.
   Class-relative support is tempered by absolute support; trust has a 0.1 floor.
4. Preserve candidate labels from duplicate groups. Blend toward soft evidence
   only when the OOF classifier and neighbors agree and evidence reaches 0.5.
   No original index labels are overwritten and no samples are deleted.
5. Weight classification and consistency terms inversely by duplicate-group
   size (normalized to mean one). This is loss weighting, not a new sampler;
   feature anchoring and contrastive regularization retain B0 reductions.
6. Train with expected soft-label GCE plus a soft CE floor, and symmetric
   stop-gradient KL between two independently augmented views on **unmixed**
   batches. Low-trust rows receive more consistency weight. Mixup/CutMix retain
   each source row's trust and duplicate weight separately.
7. First 512 stage-one steps update only the head and scale. Then update the last
   four visual transformer blocks, post-layernorm, patch mechanism and head.
   Earlier blocks, position embeddings and visual projection remain frozen.
   Layer learning rates decay by 0.8 toward earlier blocks. Stage two has no
   additional head-only warmup and preserves B1 optimizer state.

Warmup gradients are set to `None` after backward, so AdamW cannot decay frozen
parameters. The DDP parameter set stays stable across the transition. This
implementation changes update eligibility at step 512; it still computes the
backward graph during warmup and is not a compute-optimized head-only trainer.

The architecture, class count, input size and graph-readout policy remain B0's.
The first-stage trainer is a wrapper around one shared training implementation.
LoRA, near-duplicate detection, hard relabeling, online trust updates, EMA
teachers and local-crop branches are deliberately not included in this B1.

## Layout

| File | Role |
|---|---|
| `code/build_quality.py` | TRAIN-only feature/index checks, byte grouping and quality artifacts |
| `code/engine/trust_quality.py` | Group folds, OOF ridge heads, neighbor voting and soft targets |
| `code/engine/trust_training.py` | Soft supervision, consistency, freeze policy and artifact validation |
| `code/quality_config.json` | Explicit cleaning/cross-fitting defaults |
| `code/train_config_stage1.json` / `train_config.json` | Stage settings and B1 additions |
| `code/train_stage1.sh` / `train_stage2.sh` | Preparation and training |
| `code/resume_train.sh` | Resume an interrupted B1 stage |
| `code/run_train_all.sh` | Both stages, final inference and ZIP |

## Prepare inputs

Use the Linux/WSL CUDA environment from `code/requirements.txt`. From the
repository root install dependencies with:

```bash
python -m pip install -r experiments/b1/code/requirements.txt
```

Supply authorized original files separately (they are not committed):

```text
experiments/b1/code/data/train_index.json
experiments/b1/code/data/test_names.json
experiments/b1/code/data/official/train/<class>/<image>
experiments/b1/code/data/official/test/<image>
```

The TRAIN feature extractor must use the official initialization. Do not replace
its features with features from the trained B0 model: that would invalidate the
out-of-fold interpretation. The original feature/label checksum checks remain
enabled. To reuse existing official feature or image caches, link/copy the
original files to their expected B1 paths; never substitute another image order.
Raw images and indices must still be present for source verification.

## Run

From the repository root:

```bash
# Full pipeline: needs TRAIN and TEST, four training GPUs by default.
bash experiments/b1/code/run_train_all.sh

# Or run the stages separately (training does not require TEST).
bash experiments/b1/code/train_stage1.sh
bash experiments/b1/code/train_stage2.sh

# Resume after interruption. Keep the same configuration and world size.
bash experiments/b1/code/resume_train.sh stage1
bash experiments/b1/code/resume_train.sh stage2

# Predict using the B1 stage-two model. Use a fresh feature-cache path.
INFERENCE_CACHE="$PWD/experiments/b1/code/inference_cache_repeat" \
  bash experiments/b1/code/run_predict.sh
```

Stage one runs 4096 steps; stage two restarts its schedule and runs to 1536,
then resumes to 2048 with the inherited 3072-step schedule. Direct resume of an
interrupted stage two targets 2048. Existing training runs are not overwritten.
B0 checkpoints cannot be passed as B1 stage-transition checkpoints: B1 requires
matching quality hashes and the B1 optimizer structure.

The first-stage script builds `cache/train_quality.npz` and its JSON report.
If preparation succeeded but training has not started, start the trainer directly
instead of rebuilding the quality cache:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone \
  --nproc_per_node=4 experiments/b1/code/engine/train_stage1.py \
  --workspace experiments/b1/code \
  --config experiments/b1/code/train_config_stage1.json --run stage1 --until 4096
```

Quality computation defaults to CPU; `python experiments/b1/code/build_quality.py
--device cuda` enables GPU computation. `--force` is required to replace an
existing quality cache. A changed cache invalidates existing B1 run bindings.
The classifier is a ridge-regression head followed by temperature softmax;
its probabilities/trust scores are **heuristics, not calibrated correctness
probabilities or an implementation of Confident Learning**.

## Outputs and resource considerations

- `code/cache/train_quality.npz`: soft targets, trust, duplicate weights,
  group hashes, fold assignments, OOF probabilities and neighbor probabilities.
- `code/cache/train_quality.json`: input hashes, settings, per-class trust and
  classes absent from each fold's fitting data. Inspect missing-class warnings
  before relying on the resulting soft labels.
- `code/runs/stage*/last.pt`: model, optimizer, RNG/worker state and quality hash.
- `code/runs/stage*/model.pt`: inference checkpoint, configuration and quality hash.
- `results/pred_results.csv` / `pred_results.zip`: final predictions.

Two augmented views double the forward image count on unmixed batches. Defaults
retain B0's batch 48 per GPU; memory feasibility has not been verified. Reduce
batch in both stage configs if needed and record the changed global batch.
Group kNN computation is blocked but still approximately quadratic in the number
of distinct groups; quality preparation can be expensive on CPU. With 148695
rows and 750 classes, each dense probability array is about 446 MB; allow several
GB of RAM and about 1.4 GB for the quality NPZ, in addition to B0 image caches.

Quality folds are used for cross-fitting supervision, **not** an unbiased final
model evaluation split: final training uses all TRAIN rows. For tuning or reporting
new accuracy, construct a separate group-isolated evaluation protocol. TEST is
used only by the inherited prediction/readout scripts, never by quality building.

## Verification

```bash
python -m unittest discover -s experiments/b1/tests -v
```

Tests cover duplicate isolation, held-out label independence for both head and
neighbor predictions, conservative correction, group weights, mixed-label loss,
two-view gradients, and actual optimizer updates through warmup/unfreezing.
Local verification uses CPU PyTorch. Full CUDA/DDP execution, original feature
regeneration, memory use and B1 scores still require the training server.
