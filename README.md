# Trace the Ace: Matrix-Aware Muon-LoRA

Reproducible code for our K-12 AI Infrastructure **Trace the Ace** solution and
the optimizer study that followed it. The task predicts whether a student will
answer correctly after a long tutoring dialogue.

**[Read the solution report (PDF)](docs/solution.pdf)**

## Main result

We replaced AdamW with Muon for every trainable LoRA factor and the sequence
classification matrix of Qwen3-4B. With the backbone, 35,072 training rows,
rank-256 adapters, one epoch, four-GPU global batch 4, and inference format held
fixed, an optimizer-specific learning rate of `2.5e-5` produced:

| Single neural model | Raw Log Loss | Robust-Platt Log Loss | AUROC |
|---|---:|---:|---:|
| AdamW, `1e-5` | 0.623250 | 0.593883 | 0.650021 |
| **Muon, `2.5e-5`** | **0.616465** | **0.591436** | **0.653200** |

The paired 2,000-repeat session bootstrap estimated a calibrated Log Loss gain
of `0.002441`, with a 95% interval of `[0.000650, 0.004234]`.

The Muon numbers use released public/A labels. The
[finalized leaderboard](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/leaderboard/)
uses private/B labels, so its winning `0.59233` Log Loss is not directly
comparable. Our best official submission moved from `0.59663` on A to `0.59449`
on B, which makes stronger Muon B performance plausible but unmeasured. The
Muon run was evaluated after submissions closed and was not a leaderboard
entry.

## Why Muon inside LoRA?

LoRA learns matrix factors `A` and `B` while freezing the pretrained backbone.
AdamW scales their entries as independent coordinates. Muon orthogonalizes each
matrix momentum update with Newton-Schulz iterations before applying it. We
used Muon for **505 trainable matrices and zero AdamW fallback matrices**, and
saved the complete runtime membership audit beside the checkpoint.

The gain was not explained by the final score matrix alone. Opposite routing
controls, Muon adapters with an AdamW head and AdamW adapters with a Muon head,
both regressed. The useful recipe required coordinated optimization of the
adapted representation and task head. The learning-rate response was also
non-monotonic: `2.5e-5` was best in our grid, while `3e-5` and `1e-4` failed.

## Repository layout

```text
configs/                         Frozen training recipe
docs/solution.md                 Four-page solution narrative
results/frozen_test_metrics.json Aggregate results only
src/trace_ace/data.py            Competition CSV and transcript preparation
src/trace_ace/headtail.py        35/65 long-context allocation
src/trace_ace/train.py           Four-GPU ms-swift Muon-LoRA launcher
src/trace_ace/muon_audit.py      Runtime optimizer membership and finite checks
src/trace_ace/predict.py         Sequence-classification inference
src/trace_ace/xgb_head.py        Frozen five-model hidden-state XGBoost stage
src/trace_ace/calibration.py     Frozen beta and robust-Platt mappings
src/trace_ace/evaluate.py        ID-safe Log Loss and AUROC evaluation
tests/                           Lightweight preprocessing/calibration tests
```

Competition data, labels, per-row predictions, pretrained weights, and LoRA
checkpoints are intentionally excluded.

## Environment

The reference run used Python 3.10, PyTorch with CUDA, `ms-swift==4.4.2`,
Qwen3-4B-Instruct-2507, and four GPUs. Install the package and training extras:

```bash
python -m pip install -e ".[train,test]"
git clone https://github.com/MoonshotAI/Moonlight.git third_party/Moonlight
```

Muon is loaded through ms-swift's optimizer plugin. `trace_ace.muon_audit`
patches only the repository-path/eager-execution compatibility needed by
ms-swift 4.4.2; it does not change Muon's update equation.

## Prepare data

Download the competition data from the official platform. Do not commit it.
Build ordinary full-dialogue training JSONL:

```bash
trace-ace-prepare \
  --features data/train_features.csv \
  --labels data/train_labels.csv \
  --transcripts data/train_transcripts \
  --output data/train.jsonl
```

Build Head-tail 35/65 inference JSONL. Short examples remain unchanged; long
examples retain the learning objective, 35% of the early dialogue budget, and
65% of the latest dialogue budget:

```bash
trace-ace-prepare \
  --features data/test_features.csv \
  --transcripts data/test_transcripts \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --headtail \
  --output data/test_headtail.jsonl
```

## Train

The launcher reproduces the frozen `2.5e-5` recipe and prints the exact command
before execution. Run `--dry-run` first when adapting paths.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 trace-ace-train \
  --dataset data/train.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --muon-repo third_party/Moonlight \
  --output checkpoints/qwen3-4b-muon-lora
```

At startup, the callback writes `muon_optimizer_audit.json` and aborts unless
all trainable matrices are assigned to Muon. It also rejects non-finite logged
losses, gradient norms, learning rates, or trained parameters.

## Predict and evaluate

```bash
trace-ace-predict \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter checkpoints/qwen3-4b-muon-lora/checkpoint-8768 \
  --jsonl data/test_headtail.jsonl \
  --output predictions/raw.csv \
  --features-output outputs/test_features.npz

trace-ace-evaluate \
  --predictions predictions/raw.csv \
  --labels data/test_labels.csv
```

The optional XGBoost stage consumes the exported final-token hidden vectors.
Its five configurations and beta mapping were selected on grouped development
predictions and are checked in as frozen constants:

```bash
python -m pip install -e ".[xgb]"

trace-ace-xgb train \
  --features outputs/train_features.npz \
  --labels data/train_labels.csv \
  --output checkpoints/xgb

trace-ace-xgb predict \
  --features outputs/test_features.npz \
  --models checkpoints/xgb \
  --output predictions/xgb.csv
```

Do not select XGBoost trials or calibration coefficients on the released test
labels. The checked-in trial set is the frozen result of the development audit.

The checked-in calibration constants were fitted on session-grouped development
predictions. They must not be refitted on a test set. For a new dataset, fit new
calibrators exclusively from grouped out-of-fold predictions.

## Reproducibility boundaries

- The Muon and AdamW comparison uses each optimizer's selected learning rate;
  it is a recipe comparison, not an equal-learning-rate causal estimate.
- Released labels were read only after Muon predictions and their SHA-256 hash
  had been frozen.
- No released test rows, labels, predictions, or model weights are included.
- The direct Hugging Face predictor is a portable reference. For exact
  competition-runtime parity, preserve the ms-swift `qwen3_nothinking` template
  and verify logits on a fixed smoke set before scoring.

## License and citations

This repository is released under the MIT License. Qwen, ms-swift, Moonlight,
Muon, LoRA, XGBoost, and the competition dataset retain their own licenses.
Please cite the corresponding upstream projects and papers listed in
[`docs/solution.md`](docs/solution.md).
