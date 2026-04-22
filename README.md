# Continual Learning on HAR Dataset

Comparing four anti-forgetting strategies on the UCI Human Activity Recognition dataset using a shared MLP architecture trained sequentially on 3 binary tasks.

---

## Methods compared

| Method | Strategy | Stores data? |
|---|---|---|
| Fine-tuning | Baseline — no protection | No |
| Experience Replay | Circular buffer of past samples | Yes |
| LwF | Knowledge distillation from old model | No |
| EWC | Fisher-weighted regularisation penalty | No |

---

## Dataset

**UCI HAR** — smartphone accelerometer + gyroscope readings from 30 volunteers performing 6 activities, split into 3 sequential binary tasks:

- **T1** Walking vs Walking Upstairs
- **T2** Walking Downstairs vs Sitting
- **T3** Standing vs Laying

561 features · ~10,000 samples · architecture: `561 → 256 → 256 → 1`

---

## Results

| Method | Avg Accuracy | BWT | Forgetting |
|---|---|---|---|
| Fine-tuning | 0.673 | −0.465 | 0.310 |
| **Experience Replay** | **0.901** | **−0.119** | **0.079** |
| LwF | 0.699 | −0.435 | 0.290 |
| EWC | 0.666 | −0.463 | 0.311 |

Experience Replay is the clear winner — 4× less forgetting than the baseline with a 2000-sample buffer.

EWC fails structurally due to the shared single-output head redefining semantics across tasks (flat performance across λ ∈ {1, 10, 100, 500, 2000}).

---

## Metrics

- **Average Accuracy** — mean accuracy across all tasks after training on the last task
- **BWT** (Backward Transfer) — how much new tasks hurt old ones (negative = forgetting)
- **FWT** (Forward Transfer) — how much past learning helps future tasks vs random baseline
- **Forgetting** — gap between peak and final accuracy per task

