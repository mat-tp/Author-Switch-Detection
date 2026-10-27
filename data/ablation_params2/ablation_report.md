# Ablation study report

- Training pairs: 2439074
- Evaluation pairs (held-out test half): 259653
- Pair features in the full model: 76
- Baseline feature settings: `case=preserve|char=[3]x256|pos=[2]x128(heuristic)|fw=true|full=false|pair=ABSOLUTE|zclip=5.0`
- Seeds: [42] (single run, no variance estimate)
- Total run time: 494.0 min

**Metric:** macro-F1 (the mean of the F1 of the *change* class and of the *no change* class, pooled over all boundaries - the measure PAN 2026 ranks by) on the held-out test half. The F1 of the change class alone is in `ablation_results.csv`. Thresholds (all three models) are chosen on the validation data by macro-F1, never on the evaluation data. Parameters and pipeline settings are ranked by **validation** macro-F1.

## 0. The difficulty levels

PAN 2026 scores **easy, medium and hard as three separate datasets** (the best systems use a separate recipe for each), so every table below reports the levels separately. The pairs of one level can be very different from another (medium has very few changes), so a threshold that is best for all pairs pooled is not best for any one level. Two numbers are therefore kept for every level: *pooled thr.* = the one threshold chosen on the pooled validation data (how the first version reported), and *own thr.* = the threshold chosen on that level's validation data only. The *own thr.* number is the one that corresponds to a submission for that dataset and is used in all per-level columns of this report. Run `-Ddifficulty=easy|medium|hard` to train and evaluate on one level only.

**Baselines on the evaluation pairs of each level** (macro-F1; the official test baselines are in `docs/RESEARCH.md`):

| Level | Pairs | Share of changes | Always "no change" | Always "change" |
|---|---:|---:|---:|---:|
| easy | 56685 | 29.7 % | 0.4128 | 0.2290 |
| medium | 139254 | 4.5 % | 0.4886 | 0.0427 |
| hard | 63714 | 21.2 % | 0.4409 | 0.1746 |

**The baseline setting of each model, per level** (evaluation macro-F1):

| Model | Level | Pooled thr. | Own thr. | Own threshold |
|---|---|---:|---:|---:|
| Neural Network | easy | 0.9234 | 0.9435 | 0.5300 |
| Neural Network | medium | 0.5052 | 0.5051 | 0.7000 |
| Neural Network | hard | 0.4455 | 0.5439 | 0.3800 |

## 1. Feature-group ablation

`Importance(G) = MacroF1_all - MacroF1_without_G`. Positive = the group carries useful information. Near 0 = it adds nothing the other groups do not already provide. Negative = the model does better without it. "Only G" trains on that group alone. The per-difficulty columns are the same difference on easy / medium / hard pairs, each level with the threshold chosen on that level's own validation data (PAN scores every level as its own dataset).

## 2. Parameter ablation

One hyperparameter is changed at a time; everything else stays at the baseline of Main.java. `◀` = the baseline value, `★` = best validation macro-F1 of that parameter. The train − validation gap shows over-fitting.

### Neural Network

**hidden_layers**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 16 | 0.7207 | 0.7210 | 0.7131 | -0.0003 | +0.0018 | ★ |
| 32-16 | 0.7209 | 0.7209 | 0.7137 | -0.0000 | +0.0024 |  |
| 64-32 | 0.7205 | 0.7206 | 0.7124 | -0.0001 | +0.0012 |  |
| 128-64-32 | 0.7202 | 0.7203 | 0.7127 | -0.0001 | +0.0014 |  |
| 128-64-32-16-8 | 0.7193 | 0.7201 | 0.7113 | -0.0008 | +0.0000 | ◀  |

**l2_lambda**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7374 | 0.7379 | 0.7294 | -0.0005 | +0.0182 | ★ |
| 1.0E-4 | 0.7337 | 0.7340 | 0.7268 | -0.0003 | +0.0156 |  |
| 1.0E-3 | 0.7291 | 0.7291 | 0.7216 | +0.0000 | +0.0104 |  |
| 1.0E-2 | 0.7193 | 0.7201 | 0.7113 | -0.0008 | +0.0000 | ◀  |
| 1.0E-1 | 0.4616 | 0.4609 | 0.4622 | +0.0007 | -0.2491 |  |

**dropout**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7221 | 0.7223 | 0.7143 | -0.0002 | +0.0030 | ★ |
| 0.1 | 0.7207 | 0.7206 | 0.7126 | +0.0000 | +0.0013 |  |
| 0.2 | 0.7193 | 0.7201 | 0.7113 | -0.0008 | +0.0000 | ◀  |
| 0.3 | 0.7191 | 0.7192 | 0.7116 | -0.0000 | +0.0004 |  |
| 0.5 | 0.7187 | 0.7187 | 0.7109 | +0.0000 | -0.0004 |  |

**learning_rate**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1.0E-4 | 0.7218 | 0.7219 | 0.7139 | -0.0001 | +0.0026 | ★ |
| 5.0E-4 | 0.7201 | 0.7208 | 0.7119 | -0.0007 | +0.0006 |  |
| 1.0E-3 | 0.7193 | 0.7201 | 0.7113 | -0.0008 | +0.0000 | ◀  |
| 5.0E-3 | 0.7173 | 0.7171 | 0.7102 | +0.0002 | -0.0011 |  |
| 1.0E-2 | 0.7175 | 0.7171 | 0.7100 | +0.0005 | -0.0013 |  |

**optimizer**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| adam | 0.7193 | 0.7201 | 0.7113 | -0.0008 | +0.0000 | ◀  |
| sgd | 0.7204 | 0.7203 | 0.7128 | +0.0000 | +0.0016 | ★ |

**batch_size**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 32 | 0.7197 | 0.7197 | 0.7115 | +0.0000 | +0.0002 |  |
| 64 | 0.7193 | 0.7201 | 0.7113 | -0.0008 | +0.0000 | ◀  |
| 128 | 0.7200 | 0.7200 | 0.7126 | -0.0000 | +0.0013 |  |
| 256 | 0.7200 | 0.7203 | 0.7129 | -0.0003 | +0.0016 | ★ |

**feature_clip**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7194 | 0.7202 | 0.7114 | -0.0008 | +0.0001 |  |
| 3.0 | 0.7204 | 0.7207 | 0.7127 | -0.0003 | +0.0015 | ★ |
| 5.0 | 0.7193 | 0.7201 | 0.7113 | -0.0008 | +0.0000 | ◀  |

**patience**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.7202 | 0.7201 | 0.7126 | +0.0001 | +0.0013 |  |
| 3 | 0.7194 | 0.7190 | 0.7120 | +0.0004 | +0.0007 |  |
| 8 | 0.7193 | 0.7201 | 0.7113 | -0.0008 | +0.0000 | ◀ ★ |

### Best value of each parameter

Each parameter is tuned on its own (the others stay at baseline), so combining all the best values is a *candidate* that still has to be trained and checked, not a guaranteed improvement.

**Neural Network**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| hidden_layers | 16 | 0.7210 | 0.7131 | 128-64-32-16-8 | 0.7113 |
| l2_lambda | 0.0 | 0.7379 | 0.7294 | 1.0E-2 | 0.7113 |
| dropout | 0.0 | 0.7223 | 0.7143 | 0.2 | 0.7113 |
| learning_rate | 1.0E-4 | 0.7219 | 0.7139 | 1.0E-3 | 0.7113 |
| optimizer | sgd | 0.7203 | 0.7128 | adam | 0.7113 |
| batch_size | 256 | 0.7203 | 0.7129 | 64 | 0.7113 |
| feature_clip | 3.0 | 0.7207 | 0.7127 | 5.0 | 0.7113 |
| patience | 8 | 0.7201 | 0.7113 | 8 | 0.7113 |

## 4. How to read this (limitations to state in the paper)

- **Correlated groups.** If two groups carry the same information (e.g. sentence length and word length), removing one costs little because the other compensates, so a small importance does not prove a group is useless. The "only G" column helps tell the two cases apart.
- **One parameter at a time** ignores interactions between parameters (the decision-tree grid is the only place interactions are looked at).
- **Single run vs seeds.** Without several seeds there is no variance estimate; differences of a few thousandths of F1 should not be interpreted. Run with `-Dseeds=42,123,456` for the final numbers.
- **POS tags** come from the lightweight rule-based tagger (features.HeuristicPosTagger) unless `pos_tagger: corenlp` is in the pipeline table, so the POS result describes that tagger.
- **Pipeline settings are compared one at a time** around the baseline. The best value of each family is a candidate, not a proven combination (the optional `char_grid` family looks at n x buckets together).
- Parameters and pipeline settings are chosen on validation macro-F1; the test column is reported once, for the chosen values, and should not be used to pick settings.
