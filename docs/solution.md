# Matrix-Aware LoRA Optimization for Calibrated Tutoring Assessment

## 1. Summary

Predicting whether a student answered correctly from a tutoring transcript is not only a
language-understanding problem. The primary metric, Log Loss, also requires reliable
probabilities. We therefore built a system with three explicit stages: (1) preserve the
learning objective and the most useful evidence from a long tutor-student dialogue, (2)
learn an outcome representation with a LoRA-tuned Qwen3-4B classifier, and (3) calibrate the
result with a lightweight prediction head.

Our submitted system used the final valid-token hidden state from Qwen3-4B, a five-model
XGBoost ensemble, and cross-fitted beta calibration. It achieved **0.5966 Log Loss and
0.6464 AUROC**, compared with **0.6000 and 0.6480** for the native Qwen classifier. This
tradeoff was informative: the language model supplied useful ranking, while the second
stage improved the probability geometry required by Log Loss.

| Competition submission | Log Loss | AUROC |
|---|---:|---:|
| Qwen3-4B LoRA classification head | 0.6000 | **0.6480** |
| **Final-token five-model XGBoost + beta** | **0.5966** | 0.6464 |
| Head-tail + turn network + objective prior | 0.5975 | 0.6460 |

The middle row was our best leaderboard result and is the primary submitted solution. The
last row is an important ablation: individually plausible additions did not necessarily
survive distribution shift when stacked together.

Our strongest method extension concerns optimization inside LoRA. Muon was developed and
scaled mainly for dense matrix optimization in language model training [3]. LoRA instead
freezes the backbone and learns factorized low-rank updates `Delta W = (alpha/r) B A` [2].
We applied Muon's matrix-orthogonalized momentum to all trainable LoRA factors and the
classification matrix, then audited actual optimizer membership at runtime. At its best
tested learning rate, this **Muon-LoRA recipe** improved both discrimination and
calibration before any new model was added: frozen robust-Platt Log Loss fell from
**0.593883 to 0.591436**, raw Log Loss fell from **0.623250 to 0.616465**, and AUROC rose
from **0.650021 to 0.653200**. The single model still uses one Qwen forward pass and a
linear classification head at inference.

The Muon result uses released public/A labels, whereas the finalized leaderboard uses
private/B labels, so it cannot be ranked directly against the winning **0.59233 / 0.65088**
row [7]. Our best official submission moved from **0.59663 / 0.64465** on A to
**0.59449 / 0.64985** on B. That direction makes stronger Muon B performance plausible,
but no Muon B prediction exists. The run was evaluated after submissions closed and was
not a leaderboard entry.

We do not claim to have invented Muon for low-rank adaptation. Recent work confirms that
LoRA's factorization creates a genuine geometric ambiguity and motivates specialized
low-rank optimization [4]. Our contribution is an end-to-end application and evidence
chain for long-context educational classification: exact optimizer auditing,
optimizer-specific learning rate scaling, frozen calibration, opposite module-routing
controls, and optimizer-diverse ensembling.

## 2. Data, Validation, and Transcript Construction

For each response, we combined the target learning objective with the chronologically
ordered tutoring transcript. Role markers distinguished tutor and student utterances.
Identifiers such as `response_id`, `session_id`, `learning_objective_id`, and
`utterance_id` were excluded from model text. Timestamps were retained when available
because they preserve conversational order without exposing the outcome label.

A tutoring session can generate several response-objective rows. Random row splitting
would therefore leak nearly identical conversations across train and validation. We grouped
every model selection split by `session_id`, maintained an untouched development partition,
and estimated uncertainty by resampling sessions rather than rows. Calibration parameters
were fitted only from grouped development predictions and then frozen.

The maximum model length was 10,240 tokens. For long conversations, our Head-tail policy
retained the task prefix and allocated 35% of the content budget to the beginning and 65%
to the end. This preserves the learning objective, early task setup, and the conclusion of
the interaction. A 2x2 audit crossed ordinary and Head-tail inputs at training and
inference. A Head-tail-trained model evaluated with ordinary formatting degraded sharply,
showing that truncation is part of the learned task definition rather than a harmless
inference switch.

## 3. Outcome Representation and Probability Head

We fine-tuned Qwen3-4B-Instruct-2507 [1] as a two-class sequence classifier using ms-swift.
The base recipe used one epoch, LoRA rank 256 and alpha 512, CE loss, AdamW with learning
rate 1e-5, cosine decay, 3% warmup, weight decay 0.1, bfloat16, SDPA attention, gradient
checkpointing, and global batch size four.

For each example, we saved the final layer at the last non-padding token, a 2,560-
dimensional vector denoted `h_final`, and the two classification logits. The final token is
a natural aggregation point in a causal Transformer because it follows both the objective
and all retained dialogue evidence.

XGBoost [5] was not used as a generic replacement classifier. We optimized it as a
regularized probability head over two complementary feature views. Two ensemble members
consumed `h_final`; three consumed `[h_final; logits]`, allowing the trees to combine the
continuous representation with the language model's native decision margin. Tree depth,
learning rate, row and feature subsampling, minimum child weight, L1/L2 regularization, and
the number of boosting rounds were selected only from session-grouped development results.
The arithmetic ensemble improved monotonically through the five selected members; adding
weaker trials did not help. This gave us diversity without another Transformer forward pass.

| Probability head | Validation Log Loss | Validation AUROC |
|---|---:|---:|
| Native Qwen classifier | 0.5634 | 0.7196 |
| Frozen final-token linear probe | 0.5371 | 0.7324 |
| Five-model XGBoost mean | 0.5360 | **0.7327** |
| XGBoost mean + beta calibration | **0.5356** | **0.7327** |

We then applied beta calibration [6] to the averaged probability:

`p_cal = sigmoid(a log(p) - b log(1-p) + c)`.

The coefficients were learned with grouped cross-fitting, so the calibrator never fitted a
row using that row's in-fold prediction. This division of labor is useful: Qwen models
semantic interaction evidence, XGBoost learns a regularized nonlinear decision surface,
model averaging controls tree variance, and beta calibration corrects asymmetric
probability distortion without changing ranking.

## 4. Muon for Low-Rank Adaptation

### 4.1 Why the transfer is nontrivial

AdamW rescales individual coordinates of each trainable factor. Muon instead constructs a
momentum matrix `M_t` and approximately orthogonalizes it with Newton-Schulz iterations
before applying the update [3]:

`M_t = mu M_(t-1) + G_t`,   `O_t approx M_t (M_t^T M_t)^(-1/2)`,

with the transposed form for wide matrices. Weight decay and shape-aware update scaling
follow the scalable Muon implementation. This changes training geometry, not the LoRA
architecture or inference graph.

For LoRA, `(B R)(R^(-1) A) = B A`, so many factor pairs describe the same effective
adapter. Applying a matrix optimizer separately to `A` and `B` is therefore not equivalent
to applying it to the dense update `Delta W`. Rather than claiming a new optimizer, we
studied whether the practical factor-wise implementation produces a useful task model.

Our 36-block model applies LoRA to `q`, `k`, `v`, `o`, `gate`, `up`, and `down`
projections, producing 504 trainable adapter matrices. The two-class score matrix is the
505th. A runtime callback enumerated every parameter by name and shape and verified
**505 Muon matrices, zero AdamW fallback matrices**, four-GPU DDP, global batch four,
finite losses and gradients, and the registered learning rate. This prevents a common
experimental failure mode in which an optimizer is named in configuration but only a
subset of intended parameters actually uses it.

### 4.2 Optimizer-specific scale

We kept the 35,072 training rows, backbone, CE objective, LoRA rank and alpha, seed, epoch
count, four-GPU execution, global batch, and input policy fixed. Muon learning rate was the
only changed training hyperparameter across its main sweep. All evaluation mappings were
frozen before scoring.

| Optimizer recipe | Raw Log Loss | Frozen beta Log Loss | Frozen robust-Platt Log Loss | AUROC |
|---|---:|---:|---:|---:|
| AdamW, LR1e-5 | 0.623250 | 0.594181 | 0.593883 | 0.650021 |
| Muon, LR 2e-5 | 0.619220 | 0.592745 | 0.592714 | **0.653213** |
| Muon, LR 2.5e-5 | **0.616465** | **0.592053** | **0.591436** | 0.653200 |
| Muon, LR 3e-5 | 0.640174 | 0.603389 | 0.600297 | 0.636753 |
| Muon, LR 1e-4 | - | - | 0.6329 | 0.5475 |

The response is sharply non-monotonic. Muon benefited from a larger optimizer-specific
step than the AdamW recipe, peaked at 2.5e-5 in our grid, and deteriorated by 3e-5. Against
AdamW, the best robust-Platt result improved Log Loss by 0.002447. A paired bootstrap over
8,241 sessions estimated a mean gain of 0.002441 with 95% interval
[0.000650, 0.004234]. The raw endpoint also improved, so the result is not created by
post-processing. At the selected 2.5e-5 rate, the calibrated single-model score of
0.591436 crossed the numeric level of the private/B winner on the public/A audit. Because
the rows differ, this is evidence of headroom rather than a leaderboard comparison.

This comparison establishes a useful recipe, not a pure causal estimate of optimizer
identity: AdamW and Muon use their selected learning rates, and we do not have matched
AdamW runs at 2e-5 or 2.5e-5. The negative high-rate boundary and frozen mappings make the
claim narrower but stronger: matrix normalization changes the useful update scale, yet does
not make Muon insensitive to learning rate.

### 4.3 Mechanism controls

We tested whether the gain came only from the two-class score matrix. In one full-data run,
Muon optimized the Transformer LoRA matrices while AdamW optimized the score head. In the
opposite audit, AdamW optimized all 504 LoRA matrices while Muon optimized only the score
matrix. Both variants regressed. In the reverse audit, untouched beta Log Loss increased
from 0.5415 to 0.5460 and AUROC fell from 0.7254 to 0.7175; the session-bootstrap Log Loss
gain was -0.004508 with 95% interval [-0.006976, -0.002084]. These opposite controls reject
the simple explanation that Muon merely found a better linear head. The useful behavior
requires coordinated optimization of the adapted representation and task head.

## 5. Complementary Predictions

Muon also changed the error pattern. AdamW and Muon raw logits correlated 0.9188, leaving
enough diversity for a blend. A 40% AdamW / 60% Muon average of beta-calibrated
classification probabilities reached **0.591466 Log Loss / 0.654527 AUROC**.

We also retained an AdamW XGBoost branch built from block 28 rather than the final layer.
Its standalone beta result, 0.594894 Log Loss, was weaker than the best Muon classifier,
but its calibrated output correlated only 0.8792 with Muon's. A 40% Layer-28 XGBoost /
60% Muon classifier blend therefore reached the strongest observed two-branch result,
**0.590509 / 0.657425**. This shows why we did not select representation layers solely by
standalone score: an intermediate layer can preserve complementary semantic information
that becomes valuable in an ensemble. The gain combines optimizer, representation-depth,
and prediction-head diversity.

The complete improvement path is summarized below. Official B rows and public/A rows are
separated explicitly; only values within one evaluation column are directly comparable.

| System | Evaluation | Log Loss | AUROC | Main lesson |
|---|---|---:|---:|---|
| Official #1, oleh | Private/B | 0.59233 | 0.65088 | Finalized leaderboard reference |
| Official #2, appleswim | Private/B | 0.59241 | 0.64735 | Finalized leaderboard reference |
| Official #3, Team Chicken | Private/B | 0.59280 | 0.64737 | Finalized leaderboard reference |
| **Our official #4 submission** | **Private/B** | **0.59449** | **0.64985** | Official result |
| Same official submission | Public/A | 0.59663 | 0.64465 | A was harder for this submitted model |
| AdamW Qwen + Head-tail + robust Platt | Public/A audit | 0.593883 | 0.650021 | Context allocation and calibration |
| **Muon LR 2.5e-5 + robust Platt** | **Public/A audit** | **0.591436** | **0.653200** | Best single neural public/A result |
| AdamW + Muon LR 2e-5 beta blend | Public/A audit | 0.591466 | 0.654527 | Optimizer diversity |
| **Layer-28 XGBoost + Muon LR 2e-5** | **Public/A audit** | **0.590509** | **0.657425** | Best observed two-branch result |

These fusion weights were inspected on released labels, so we treat them as diagnostic
evidence rather than independent model selection. The single model Muon result is the main
optimizer finding; a deployable ensemble would select its branches and weights from grouped
out-of-fold (OOF) predictions. This distinction also avoids attributing the XGBoost fusion gain solely to
Muon.

## 6. Compact Ablation Results

We report unsuccessful ideas because they sharpen the positive conclusions without taking
over the main narrative.

| Ablation | Evaluation | Log Loss / AUROC | Decision |
|---|---|---:|---|
| Final-mean pooling + linear head | Leaderboard | 0.6381 / 0.6026 | Reject unstable pooling |
| Final-mean pooling + XGBoost | Leaderboard | 0.6084 / 0.6287 | Still below final token |
| Head-tail + turn network + prior 0.20 | Leaderboard | 0.5975 / 0.6460 | Did not beat 0.5966 |
| Tail-only + robust Platt | Frozen released test | 0.594402 / 0.649034 | No stable gain over Head-tail |
| XGBoost + frozen turn network | Frozen released test | 0.596341 / 0.648259 | Worse than XGBoost alone |
| XGBoost + objective prior 0.20 | Frozen released test | 0.595604 / 0.649253 | No useful improvement |
| Label smoothing 0.05 | Frozen released test | 0.602361 / 0.623283 | Reject |
| Objective-guided evidence routing | Frozen released test | 0.603463 / 0.620377 | Reject |
| Muon LR 3e-5 / 1e-4 | Frozen released test | 0.600297 / 0.636753; 0.6329 / 0.5475 | LR boundary failure |
| Partial teacher-rationale training | Frozen released test | 0.6005 / 0.6336 | Reject implemented CoT pipeline |

Additional checkpoint averaging, Focal loss, LoRA-rank, two-epoch, and experimental
optimizer variants also failed to exceed the retained systems. The pattern is consistent:
simple calibration, final-token or intermediate-layer representations, and a narrowly tuned
Muon recipe transferred; additional architectural complexity usually did not.

## 7. Lessons and Limitations

Three lessons transfer beyond this competition. First, optimize the probability pipeline,
not only the backbone: representation quality, decision head, and calibration solve
different parts of Log Loss. Second, long-context allocation is part of model training and
must match inference. Third, parameter-efficient fine-tuning still has meaningful optimizer
geometry. A small trainable parameter count does not make the optimizer irrelevant.

Our Muon evidence is limited to one main training seed and an optimizer-specific LR sweep.
The strongest blend is not independently selected, and the released evaluation is not a
substitute for a new hidden test set. The model also predicts outcomes rather than causal
tutoring quality. It should support review or follow-up decisions, not serve as a standalone
grading, tutor-evaluation, or placement system.

## References

[1] A. Yang et al. "Qwen3 Technical Report." arXiv:2505.09388, 2025.

[2] E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.

[3] J. Liu et al. "Muon is Scalable for LLM Training." arXiv:2502.16982, 2025.

[4] V. Bogachev et al. "LoRA meets Riemannion: Muon Optimizer for
Parametrization-independent Low-Rank Adapters." ICLR, 2026.

[5] T. Chen and C. Guestrin. "XGBoost: A Scalable Tree Boosting System." KDD, 2016.

[6] M. Kull, T. Silva Filho, and P. Flach. "Beta calibration: a well-founded and easily
implemented improvement on logistic calibration for binary classifiers." AISTATS, 2017.

[7] K-12 AI Infrastructure Program. "Trace the Ace: Final Leaderboard." 2026.
https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/leaderboard/
