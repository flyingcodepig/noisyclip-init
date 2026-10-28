# C2: reliable cross-batch contrastive queue

Based on C1 commit 5b7e394. Only stage1 (4096 optimizer updates) runs.
All C1 split files, frozen official features, TRAIN-only initial centers,
loss coefficients, augmentation, seed, LR groups and four-view validation
are reused. C1 is unchanged. No competition TEST is accessed.

Method change: append detached historical keys to the existing supervised
contrastive loss. No momentum encoder, label correction or sample removal.
This is SNSCL-inspired, not a reproduction of SNSCL. Per-class FIFO capacity
8 (maximum 4000 keys); age limit 256 optimizer updates; start admission at
update 128. Admit only unmixed samples whose prediction equals the given
label, top probability >=0.5, top-two margin >=0.2, and frozen CLIP nearest
class prototype agrees. The own-class prototype excludes the image itself;
all prototypes use only the 81177 TRAIN-subset images. These thresholds are
untuned starting values, not validated noise-detection guarantees.

Current-batch anchor/positive eligibility stays C1's rule. Historical keys
are stop-gradient; entries with the same sample ID are masked from both
positives and negatives. Repeated IDs update their existing class slot.
Admission occurs after each unmixed microbatch backward. Age is measured
in optimizer updates. Queue tensors, FIFO pointers, timestamps and logging
counters are checkpointed; incompatible/missing queue state cannot resume.

Observe queue class coverage, admission count, positive-row fractions with
and without historical keys, contrastive loss and validation accuracy.
Empty-queue loss and gradient equivalence, gates, self exclusion, expiry,
capacity and nonempty serialized continuation have focused unit tests.
The separate three-update GPU smoke restores a real training checkpoint;
its queue remains empty during admission warmup. Observe the formal run
past step128 to verify populated GPU queue behavior.

Run after local commit/push and server pull:
`bash experiments/c2_queue/code/run_queue.sh`
Use tmux with output redirected to runs/pipeline.log. Completed evaluation
writes runs/stage1/validation.json and runs/comparison_c1.json, including
paired corrections/regressions versus C1 stage1 on the same 9020 images.
No second stage or full-data retraining is launched.
