# Ablation study report

- Training pairs: 2439074
- Evaluation pairs (held-out test half): 259653
- Pair features in the full model: 76
- Baseline feature settings: `case=preserve|char=[3]x256|pos=[2]x128(heuristic)|fw=true|full=false|pair=ABSOLUTE|zclip=5.0`
- Seeds: [42] (single run, no variance estimate)
- Total run time: 226.7 min

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
| Decision Tree | easy | 0.9156 | 0.9230 | 0.6950 |
| Decision Tree | medium | 0.5031 | 0.5031 | 0.7900 |
| Decision Tree | hard | 0.4511 | 0.5521 | 0.4200 |

## 1. Feature-group ablation

`Importance(G) = MacroF1_all - MacroF1_without_G`. Positive = the group carries useful information. Near 0 = it adds nothing the other groups do not already provide. Negative = the model does better without it. "Only G" trains on that group alone. The per-difficulty columns are the same difference on easy / medium / hard pairs, each level with the threshold chosen on that level's own validation data (PAN scores every level as its own dataset).

## 2. Parameter ablation

One hyperparameter is changed at a time; everything else stays at the baseline of Main.java. `◀` = the baseline value, `★` = best validation macro-F1 of that parameter. The train − validation gap shows over-fitting.

### Decision Tree

**max_depth**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.6903 | 0.6906 | 0.6838 | -0.0003 | -0.0309 |  |
| 2 | 0.7045 | 0.7046 | 0.6970 | -0.0001 | -0.0177 |  |
| 3 | 0.7107 | 0.7109 | 0.7033 | -0.0002 | -0.0114 |  |
| 4 | 0.7164 | 0.7156 | 0.7091 | +0.0007 | -0.0056 |  |
| 5 | 0.7189 | 0.7179 | 0.7117 | +0.0011 | -0.0029 |  |
| 6 | 0.7215 | 0.7210 | 0.7135 | +0.0005 | -0.0011 |  |
| 8 | 0.7266 | 0.7255 | 0.7165 | +0.0011 | +0.0018 |  |
| 10 | 0.7315 | 0.7269 | 0.7200 | +0.0046 | +0.0054 | ★ |
| 15 | 0.7491 | 0.7208 | 0.7147 | +0.0283 | +0.0000 | ◀  |
| 20 | 0.7752 | 0.7070 | 0.7011 | +0.0683 | -0.0135 |  |

**min_samples_split**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 2 | 0.7499 | 0.7210 | 0.7147 | +0.0289 | +0.0001 |  |
| 10 | 0.7491 | 0.7208 | 0.7147 | +0.0283 | +0.0000 | ◀  |
| 20 | 0.7469 | 0.7210 | 0.7150 | +0.0259 | +0.0003 |  |
| 50 | 0.7432 | 0.7223 | 0.7164 | +0.0209 | +0.0018 |  |
| 100 | 0.7400 | 0.7227 | 0.7179 | +0.0172 | +0.0032 |  |
| 500 | 0.7340 | 0.7254 | 0.7191 | +0.0086 | +0.0044 | ★ |

**min_samples_leaf**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.7494 | 0.7207 | 0.7149 | +0.0286 | +0.0002 |  |
| 2 | 0.7491 | 0.7208 | 0.7147 | +0.0283 | +0.0000 | ◀  |
| 5 | 0.7461 | 0.7201 | 0.7145 | +0.0260 | -0.0001 |  |
| 10 | 0.7460 | 0.7198 | 0.7142 | +0.0262 | -0.0004 |  |
| 25 | 0.7440 | 0.7216 | 0.7149 | +0.0223 | +0.0002 |  |
| 100 | 0.7360 | 0.7259 | 0.7202 | +0.0101 | +0.0056 | ★ |

**min_impurity_decrease**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7491 | 0.7208 | 0.7147 | +0.0283 | +0.0000 | ◀  |
| 1.0E-5 | 0.7491 | 0.7208 | 0.7147 | +0.0283 | +0.0000 |  |
| 1.0E-4 | 0.7489 | 0.7210 | 0.7149 | +0.0279 | +0.0003 |  |
| 1.0E-3 | 0.7449 | 0.7232 | 0.7165 | +0.0216 | +0.0019 | ★ |
| 1.0E-2 | 0.7245 | 0.7194 | 0.7128 | +0.0051 | -0.0019 |  |

**class_weights**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| true | 0.7491 | 0.7208 | 0.7147 | +0.0283 | +0.0000 | ◀ ★ |
| false | 0.7532 | 0.7185 | 0.7118 | +0.0346 | -0.0029 |  |

**split_bins**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.7491 | 0.7208 | 0.7147 | +0.0283 | +0.0000 | ◀  |
| 16 | 0.7457 | 0.7215 | 0.7154 | +0.0243 | +0.0007 | ★ |
| 32 | 0.7483 | 0.7212 | 0.7141 | +0.0270 | -0.0006 |  |
| 64 | 0.7482 | 0.7210 | 0.7144 | +0.0272 | -0.0002 |  |
| 128 | 0.7468 | 0.7195 | 0.7144 | +0.0273 | -0.0003 |  |
| 255 | 0.7474 | 0.7202 | 0.7146 | +0.0272 | -0.0001 |  |

**Decision tree: max_depth (rows) × min_samples_leaf (columns), Validation macro-F1**

| depth \ leaf | 1 | 5 | 25 | 100 |
|---|---:|---:|---:|---:|
| 2 | 0.7046 | 0.7046 | 0.7046 | 0.7046 |
| 3 | 0.7109 | 0.7109 | 0.7109 | 0.7109 |
| 5 | 0.7179 | 0.7179 | 0.7179 | 0.7179 |
| 8 | 0.7255 | 0.7255 | 0.7255 | 0.7256 |
| 12 | 0.7259 | 0.7260 | 0.7263 | 0.7272 |

**Decision tree: max_depth (rows) × min_samples_leaf (columns), Test macro-F1**

| depth \ leaf | 1 | 5 | 25 | 100 |
|---|---:|---:|---:|---:|
| 2 | 0.6970 | 0.6970 | 0.6970 | 0.6970 |
| 3 | 0.7033 | 0.7033 | 0.7033 | 0.7033 |
| 5 | 0.7117 | 0.7117 | 0.7117 | 0.7117 |
| 8 | 0.7165 | 0.7165 | 0.7165 | 0.7166 |
| 12 | 0.7186 | 0.7185 | 0.7192 | 0.7207 |

### Best value of each parameter

Each parameter is tuned on its own (the others stay at baseline), so combining all the best values is a *candidate* that still has to be trained and checked, not a guaranteed improvement.

**Decision Tree**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| max_depth | 10 | 0.7269 | 0.7200 | 15 | 0.7147 |
| min_samples_split | 500 | 0.7254 | 0.7191 | 10 | 0.7147 |
| min_samples_leaf | 100 | 0.7259 | 0.7202 | 2 | 0.7147 |
| min_impurity_decrease | 1.0E-3 | 0.7232 | 0.7165 | 0.0 | 0.7147 |
| class_weights | true | 0.7208 | 0.7147 | true | 0.7147 |
| split_bins | 16 | 0.7215 | 0.7154 | 0 | 0.7147 |

## 4. How to read this (limitations to state in the paper)

- **Correlated groups.** If two groups carry the same information (e.g. sentence length and word length), removing one costs little because the other compensates, so a small importance does not prove a group is useless. The "only G" column helps tell the two cases apart.
- **One parameter at a time** ignores interactions between parameters (the decision-tree grid is the only place interactions are looked at).
- **Single run vs seeds.** Without several seeds there is no variance estimate; differences of a few thousandths of F1 should not be interpreted. Run with `-Dseeds=42,123,456` for the final numbers.
- **POS tags** come from the lightweight rule-based tagger (features.HeuristicPosTagger) unless `pos_tagger: corenlp` is in the pipeline table, so the POS result describes that tagger.
- **Pipeline settings are compared one at a time** around the baseline. The best value of each family is a candidate, not a proven combination (the optional `char_grid` family looks at n x buckets together).
- Parameters and pipeline settings are chosen on validation macro-F1; the test column is reported once, for the chosen values, and should not be used to pick settings.
