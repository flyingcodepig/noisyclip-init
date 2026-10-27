# C1: one change to submitted-71-06

Base: submitted-71-06 commit 4769a9b. No B1 trained checkpoint is used.
Only methodological change: early visual backbone learning rate x0.5
(encoder blocks 0..7, embeddings, pre-layernorm). Blocks 8..11, post-layernorm,
head and patch mechanism retain original rates. Visual projection remains frozen.
Losses, augmentation, microbatch 48 / accumulation 4, seeds, stage budgets
4096 and 2048 (stage2 schedule 3072), optimizer preservation and weighted
four-view validation follow the submitted recipe.

Split seed 20261028; largest-remainder per-class allocation gives exactly 9020
held-out images and 81177 training images. SHA256 groups must be unique for
this dataset; otherwise preparation stops instead of splitting duplicate groups.
Near duplicates are not detected. Labels are original noisy TRAIN labels.
No competition TEST is read. Only TRAIN-subset features initialize the centers.
Existing frozen official per-image features are reused with fixed SHA256;
they were extracted in the B1 environment (not learned B1 representations).
Their values differ from the teammate's archived features, so this is not a
bitwise reproduction of the 4090 environment.

Runtime changes: lazily decode original images to the exact center/complete
512-pixel cache recipe, retain stream augmentation and checkpoint state;
avoid rebuilding 132GB caches. CPU speed may affect throughput. No image
decode failure is replaced with a fake image. Small three-update smoke/resume
uses a separate run directory and full formal config. It is not an exact
uninterrupted-vs-resumed equivalence test. Stage1/2 endpoints are evaluated.

Run on the existing authorized server after local commit/push and server pull:
`bash experiments/c1_early_lr/code/run.sh`
Use tmux and capture stdout/stderr. `runs/pipeline_status.txt`, stage
`status.json`, metrics and `validation.json` show progress. Rerunning the same
script resumes the last complete training checkpoint; do not run duplicates.
No full-data retraining, competition prediction or platform submission occurs.

This is one candidate, not a demonstrated improvement over 71.06. There is
no retrained baseline with this split, so held-out accuracy cannot be directly
compared to the online score. Split inputs, environment and Git revision are
recorded with runtime outputs.
