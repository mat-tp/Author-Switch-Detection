# Ablation study report

- Training pairs: 2439074
- Evaluation pairs (held-out test half): 259653
- Pair features in the full model: 76
- Baseline feature settings: `case=preserve|char=[3]x256|pos=[2]x128(heuristic)|fw=true|full=false|pair=ABSOLUTE|zclip=5.0`
- Seeds: [42] (single run, no variance estimate)
- Total run time: 2122.3 min

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
| Neural Network | easy | 0.9357 | 0.9551 | 0.5600 |
| Neural Network | medium | 0.5092 | 0.5096 | 0.6150 |
| Neural Network | hard | 0.4456 | 0.5654 | 0.4000 |

## 1. Feature-group ablation

`Importance(G) = MacroF1_all - MacroF1_without_G`. Positive = the group carries useful information. Near 0 = it adds nothing the other groups do not already provide. Negative = the model does better without it. "Only G" trains on that group alone. The per-difficulty columns are the same difference on easy / medium / hard pairs, each level with the threshold chosen on that level's own validation data (PAN scores every level as its own dataset).

## 2. Parameter ablation

One hyperparameter is changed at a time; everything else stays at the baseline of Main.java. `◀` = the baseline value, `★` = best validation macro-F1 of that parameter. The train − validation gap shows over-fitting.

### Neural Network

**hidden_layers**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 16 | 0.7310 | 0.7312 | 0.7238 | -0.0002 | -0.0029 |  |
| 32-16 | 0.7325 | 0.7331 | 0.7254 | -0.0007 | -0.0013 |  |
| 64-32 | 0.7333 | 0.7337 | 0.7259 | -0.0004 | -0.0007 |  |
| 128-64-32 | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ◀ ★ |
| 128-64-32-16-8 | 0.7325 | 0.7330 | 0.7249 | -0.0005 | -0.0017 |  |

**l2_lambda**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7370 | 0.7376 | 0.7289 | -0.0006 | +0.0022 | ★ |
| 1.0E-4 | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ◀  |
| 1.0E-3 | 0.7286 | 0.7288 | 0.7212 | -0.0003 | -0.0054 |  |
| 1.0E-2 | 0.7196 | 0.7201 | 0.7115 | -0.0004 | -0.0152 |  |
| 1.0E-1 | 0.7073 | 0.7073 | 0.7006 | -0.0000 | -0.0261 |  |

**dropout**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7359 | 0.7357 | 0.7273 | +0.0002 | +0.0007 | ★ |
| 0.1 | 0.7341 | 0.7346 | 0.7264 | -0.0006 | -0.0003 |  |
| 0.2 | 0.7343 | 0.7350 | 0.7268 | -0.0007 | +0.0001 |  |
| 0.3 | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ◀  |
| 0.5 | 0.7320 | 0.7321 | 0.7247 | -0.0001 | -0.0019 |  |

**learning_rate**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1.0E-4 | 0.7405 | 0.7401 | 0.7323 | +0.0003 | +0.0057 | ★ |
| 5.0E-4 | 0.7355 | 0.7357 | 0.7276 | -0.0002 | +0.0010 |  |
| 1.0E-3 | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ◀  |
| 5.0E-3 | 0.7269 | 0.7273 | 0.7200 | -0.0005 | -0.0066 |  |
| 1.0E-2 | 0.7232 | 0.7231 | 0.7161 | +0.0001 | -0.0105 |  |

**optimizer**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| adam | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ◀  |
| sgd | 0.7371 | 0.7367 | 0.7286 | +0.0003 | +0.0020 | ★ |

**batch_size**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 32 | 0.7324 | 0.7322 | 0.7251 | +0.0002 | -0.0015 |  |
| 64 | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ◀  |
| 128 | 0.7340 | 0.7342 | 0.7261 | -0.0002 | -0.0005 |  |
| 256 | 0.7359 | 0.7362 | 0.7281 | -0.0003 | +0.0015 | ★ |

**feature_clip**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7332 | 0.7335 | 0.7258 | -0.0004 | -0.0008 |  |
| 3.0 | 0.7337 | 0.7336 | 0.7265 | +0.0001 | -0.0002 |  |
| 5.0 | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ◀ ★ |

**patience**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.7333 | 0.7336 | 0.7264 | -0.0003 | -0.0003 |  |
| 3 | 0.7333 | 0.7335 | 0.7260 | -0.0002 | -0.0006 |  |
| 8 | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ★ |
| 10 | 0.7334 | 0.7339 | 0.7266 | -0.0005 | +0.0000 | ◀  |

### Best value of each parameter

Each parameter is tuned on its own (the others stay at baseline), so combining all the best values is a *candidate* that still has to be trained and checked, not a guaranteed improvement.

**Neural Network**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| hidden_layers | 128-64-32 | 0.7339 | 0.7266 | 128-64-32 | 0.7266 |
| l2_lambda | 0.0 | 0.7376 | 0.7289 | 1.0E-4 | 0.7266 |
| dropout | 0.0 | 0.7357 | 0.7273 | 0.3 | 0.7266 |
| learning_rate | 1.0E-4 | 0.7401 | 0.7323 | 1.0E-3 | 0.7266 |
| optimizer | sgd | 0.7367 | 0.7286 | adam | 0.7266 |
| batch_size | 256 | 0.7362 | 0.7281 | 64 | 0.7266 |
| feature_clip | 5.0 | 0.7339 | 0.7266 | 5.0 | 0.7266 |
| patience | 8 | 0.7339 | 0.7266 | 10 | 0.7266 |

## 4. How to read this (limitations to state in the paper)

- **Correlated groups.** If two groups carry the same information (e.g. sentence length and word length), removing one costs little because the other compensates, so a small importance does not prove a group is useless. The "only G" column helps tell the two cases apart.
- **One parameter at a time** ignores interactions between parameters (the decision-tree grid is the only place interactions are looked at).
- **Single run vs seeds.** Without several seeds there is no variance estimate; differences of a few thousandths of F1 should not be interpreted. Run with `-Dseeds=42,123,456` for the final numbers.
- **POS tags** come from the lightweight rule-based tagger (features.HeuristicPosTagger) unless `pos_tagger: corenlp` is in the pipeline table, so the POS result describes that tagger.
- **Pipeline settings are compared one at a time** around the baseline. The best value of each family is a candidate, not a proven combination (the optional `char_grid` family looks at n x buckets together).
- Parameters and pipeline settings are chosen on validation macro-F1; the test column is reported once, for the chosen values, and should not be used to pick settings.
