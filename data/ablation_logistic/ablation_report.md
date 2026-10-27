# Ablation study report

- Training pairs: 2439074
- Evaluation pairs (held-out test half): 259653
- Pair features in the full model: 76
- Baseline feature settings: `case=preserve|char=[3]x256|pos=[2]x128(heuristic)|fw=true|full=false|pair=ABSOLUTE|zclip=5.0`
- Seeds: [42] (single run, no variance estimate)
- Total run time: 206.4 min

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
| Logistic Regression | easy | 0.9236 | 0.9455 | 0.6250 |
| Logistic Regression | medium | 0.5082 | 0.5084 | 0.6900 |
| Logistic Regression | hard | 0.4464 | 0.5437 | 0.4150 |

## 1. Feature-group ablation

`Importance(G) = MacroF1_all - MacroF1_without_G`. Positive = the group carries useful information. Near 0 = it adds nothing the other groups do not already provide. Negative = the model does better without it. "Only G" trains on that group alone. The per-difficulty columns are the same difference on easy / medium / hard pairs, each level with the threshold chosen on that level's own validation data (PAN scores every level as its own dataset).

## 2. Parameter ablation

One hyperparameter is changed at a time; everything else stays at the baseline of Main.java. `◀` = the baseline value, `★` = best validation macro-F1 of that parameter. The train − validation gap shows over-fitting.

### Logistic Regression

**learning_rate**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.01 | 0.7160 | 0.7161 | 0.7086 | -0.0002 | -0.0071 |  |
| 0.05 | 0.7217 | 0.7215 | 0.7147 | +0.0002 | -0.0009 |  |
| 0.1 | 0.7226 | 0.7228 | 0.7157 | -0.0002 | +0.0000 | ◀ ★ |
| 0.3 | 0.7180 | 0.7188 | 0.7114 | -0.0008 | -0.0043 |  |
| 0.6 | 0.7157 | 0.7166 | 0.7091 | -0.0009 | -0.0066 |  |
| 1.0 | 0.7145 | 0.7152 | 0.7076 | -0.0008 | -0.0080 |  |

**iterations**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 100 | 0.7096 | 0.7101 | 0.7029 | -0.0005 | -0.0128 |  |
| 500 | 0.7203 | 0.7206 | 0.7133 | -0.0003 | -0.0024 |  |
| 1000 | 0.7217 | 0.7215 | 0.7147 | +0.0002 | -0.0010 |  |
| 2000 | 0.7226 | 0.7228 | 0.7157 | -0.0002 | +0.0000 | ◀  |
| 2500 | 0.7228 | 0.7231 | 0.7162 | -0.0002 | +0.0005 |  |
| 5000 | 0.7232 | 0.7239 | 0.7162 | -0.0006 | +0.0006 |  |
| 10000 | 0.7242 | 0.7245 | 0.7171 | -0.0003 | +0.0014 | ★ |

**l2_lambda**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7226 | 0.7227 | 0.7157 | -0.0001 | +0.0001 |  |
| 1.0E-4 | 0.7226 | 0.7228 | 0.7157 | -0.0002 | +0.0000 | ◀  |
| 1.0E-3 | 0.7226 | 0.7228 | 0.7158 | -0.0003 | +0.0001 | ★ |
| 1.0E-2 | 0.7215 | 0.7216 | 0.7146 | -0.0001 | -0.0011 |  |
| 1.0E-1 | 0.7178 | 0.7178 | 0.7101 | -0.0000 | -0.0056 |  |

**class_weights**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| true | 0.7226 | 0.7228 | 0.7157 | -0.0002 | +0.0000 | ◀  |
| false | 0.7231 | 0.7229 | 0.7162 | +0.0002 | +0.0005 | ★ |

### Best value of each parameter

Each parameter is tuned on its own (the others stay at baseline), so combining all the best values is a *candidate* that still has to be trained and checked, not a guaranteed improvement.

**Logistic Regression**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| learning_rate | 0.1 | 0.7228 | 0.7157 | 0.1 | 0.7157 |
| iterations | 10000 | 0.7245 | 0.7171 | 2000 | 0.7157 |
| l2_lambda | 1.0E-3 | 0.7228 | 0.7158 | 1.0E-4 | 0.7157 |
| class_weights | false | 0.7229 | 0.7162 | true | 0.7157 |

## 4. How to read this (limitations to state in the paper)

- **Correlated groups.** If two groups carry the same information (e.g. sentence length and word length), removing one costs little because the other compensates, so a small importance does not prove a group is useless. The "only G" column helps tell the two cases apart.
- **One parameter at a time** ignores interactions between parameters (the decision-tree grid is the only place interactions are looked at).
- **Single run vs seeds.** Without several seeds there is no variance estimate; differences of a few thousandths of F1 should not be interpreted. Run with `-Dseeds=42,123,456` for the final numbers.
- **POS tags** come from the lightweight rule-based tagger (features.HeuristicPosTagger) unless `pos_tagger: corenlp` is in the pipeline table, so the POS result describes that tagger.
- **Pipeline settings are compared one at a time** around the baseline. The best value of each family is a candidate, not a proven combination (the optional `char_grid` family looks at n x buckets together).
- Parameters and pipeline settings are chosen on validation macro-F1; the test column is reported once, for the chosen values, and should not be used to pick settings.
