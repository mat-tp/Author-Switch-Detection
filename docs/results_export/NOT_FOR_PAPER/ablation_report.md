# Ablation study report

- Training pairs: 50000
- Evaluation pairs (held-out test half): 150000
- Pair features in the full model: 76
- Baseline feature settings: `case=preserve|char=[3]x256|pos=[2]x128(heuristic)|fw=true|full=false|pair=ABSOLUTE|zclip=5.0`
- Seeds: [42] (single run, no variance estimate)
- Total run time: 102.1 min

**Metric:** macro-F1 (the mean of the F1 of the *change* class and of the *no change* class, pooled over all boundaries - the measure PAN 2026 ranks by) on the held-out test half. The F1 of the change class alone is in `ablation_results.csv`. Thresholds (all three models) are chosen on the validation data by macro-F1, never on the evaluation data. Parameters and pipeline settings are ranked by **validation** macro-F1.

## 0. The difficulty levels

PAN 2026 scores **easy, medium and hard as three separate datasets** (the best systems use a separate recipe for each), so every table below reports the levels separately. The pairs of one level can be very different from another (medium has very few changes), so a threshold that is best for all pairs pooled is not best for any one level. Two numbers are therefore kept for every level: *pooled thr.* = the one threshold chosen on the pooled validation data (how the first version reported), and *own thr.* = the threshold chosen on that level's validation data only. The *own thr.* number is the one that corresponds to a submission for that dataset and is used in all per-level columns of this report. Run `-Ddifficulty=easy|medium|hard` to train and evaluate on one level only.

**Baselines on the evaluation pairs of each level** (macro-F1; the official test baselines are in `docs/RESEARCH.md`):

| Level | Pairs | Share of changes | Always "no change" | Always "change" |
|---|---:|---:|---:|---:|
| easy | 50000 | 29.7 % | 0.4128 | 0.2290 |
| medium | 50000 | 4.4 % | 0.4887 | 0.0424 |
| hard | 50000 | 21.1 % | 0.4410 | 0.1743 |

**The baseline setting of each model, per level** (evaluation macro-F1):

| Model | Level | Pooled thr. | Own thr. | Own threshold |
|---|---|---:|---:|---:|
| Decision Tree | easy | 0.9224 | 0.9318 | 0.4875 |
| Decision Tree | medium | 0.4991 | 0.4990 | 0.6475 |
| Decision Tree | hard | 0.4460 | 0.5393 | 0.4025 |
| Logistic Regression | easy | 0.9285 | 0.9345 | 0.0450 |
| Logistic Regression | medium | 0.5025 | 0.5052 | 0.2650 |
| Logistic Regression | hard | 0.4560 | 0.5362 | 0.0050 |
| Neural Network | easy | 0.9322 | 0.9471 | 0.5650 |
| Neural Network | medium | 0.5022 | 0.5024 | 0.7150 |
| Neural Network | hard | 0.4475 | 0.5254 | 0.3750 |

## 1. Feature-group ablation

`Importance(G) = MacroF1_all - MacroF1_without_G`. Positive = the group carries useful information. Near 0 = it adds nothing the other groups do not already provide. Negative = the model does better without it. "Only G" trains on that group alone. The per-difficulty columns are the same difference on easy / medium / hard pairs, each level with the threshold chosen on that level's own validation data (PAN scores every level as its own dataset).

### Decision Tree

Baseline (all 76 features): macro-F1 = 0.7493

| Rank | Feature group | Macro-F1 without G | Importance | Δ easy | Δ medium | Δ hard | Macro-F1 only G | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | CHAR_NGRAM | 0.7440 | **+0.0053** | +0.0045 | -0.0026 | -0.0055 | 0.6413 | useful |
| 2 | FLESCH_KINCAID | 0.7481 | **+0.0012** | +0.0017 | -0.0015 | -0.0000 | 0.7158 | negligible |
| 3 | FLESCH_READING_EASE | 0.7483 | **+0.0010** | -0.0018 | -0.0020 | -0.0092 | 0.7216 | negligible |
| 4 | PUNCTUATION_MARKS | 0.7485 | **+0.0008** | +0.0071 | -0.0008 | +0.0000 | 0.6237 | negligible |
| 5 | POS_RATIOS | 0.7487 | **+0.0006** | +0.0004 | -0.0017 | +0.0003 | 0.5766 | negligible |
| 6 | POS_NGRAM | 0.7489 | **+0.0004** | -0.0006 | -0.0024 | -0.0000 | 0.6013 | negligible |
| 7 | WORD_LENGTH | 0.7490 | **+0.0003** | -0.0005 | -0.0011 | -0.0001 | 0.6036 | negligible |
| 8 | GUNNING_FOG | 0.7493 | **+0.0001** | +0.0005 | +0.0001 | -0.0013 | 0.7101 | negligible |
| 9 | LEXICAL_DIVERSITY | 0.7493 | **+0.0000** | +0.0002 | -0.0002 | +0.0000 | 0.5870 | negligible |
| 10 | ORTHOGRAPHY | 0.7493 | **+0.0000** | -0.0001 | +0.0005 | +0.0001 | 0.5439 | negligible |
| 11 | FREQUENT_WORDS | 0.7493 | **-0.0000** | +0.0001 | -0.0000 | -0.0001 | 0.5425 | negligible |
| 12 | TFIDF | 0.7493 | **-0.0000** | +0.0000 | -0.0000 | +0.0000 | 0.6151 | negligible |
| 13 | CONTRACTIONS | 0.7493 | **-0.0000** | -0.0000 | -0.0000 | +0.0000 | 0.5163 | negligible |
| 14 | LAYOUT | 0.7494 | **-0.0001** | -0.0001 | -0.0003 | +0.0003 | 0.5029 | negligible |
| 15 | PUNCTUATION | 0.7494 | **-0.0001** | -0.0004 | +0.0002 | -0.0000 | 0.5658 | negligible |
| 16 | EMOTION | 0.7495 | **-0.0002** | -0.0005 | -0.0000 | -0.0001 | 0.5327 | negligible |
| 17 | FUNCTION_WORDS | 0.7505 | **-0.0011** | +0.0007 | -0.0005 | -0.0024 | 0.5667 | negligible |
| 18 | LENGTH | 0.7509 | **-0.0016** | +0.0049 | -0.0007 | -0.0077 | 0.6992 | negligible |

### Logistic Regression

Baseline (all 76 features): macro-F1 = 0.7481

| Rank | Feature group | Macro-F1 without G | Importance | Δ easy | Δ medium | Δ hard | Macro-F1 only G | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | CHAR_NGRAM | 0.7444 | **+0.0037** | +0.0068 | +0.0019 | +0.0123 | 0.5976 | negligible |
| 2 | POS_RATIOS | 0.7446 | **+0.0036** | +0.0019 | +0.0015 | -0.0036 | 0.5893 | negligible |
| 3 | PUNCTUATION | 0.7469 | **+0.0012** | +0.0005 | +0.0021 | +0.0001 | 0.5673 | negligible |
| 4 | WORD_LENGTH | 0.7471 | **+0.0010** | +0.0017 | -0.0009 | +0.0026 | 0.5894 | negligible |
| 5 | FLESCH_READING_EASE | 0.7473 | **+0.0009** | +0.0005 | +0.0007 | +0.0016 | 0.7226 | negligible |
| 6 | PUNCTUATION_MARKS | 0.7475 | **+0.0006** | -0.0009 | +0.0025 | +0.0040 | 0.5952 | negligible |
| 7 | GUNNING_FOG | 0.7476 | **+0.0005** | +0.0009 | +0.0005 | +0.0020 | 0.7123 | negligible |
| 8 | LAYOUT | 0.7476 | **+0.0005** | -0.0008 | +0.0005 | -0.0001 | 0.5029 | negligible |
| 9 | LENGTH | 0.7478 | **+0.0003** | +0.0004 | +0.0005 | +0.0010 | 0.7079 | negligible |
| 10 | EMOTION | 0.7479 | **+0.0002** | -0.0001 | +0.0001 | -0.0008 | 0.5153 | negligible |
| 11 | FLESCH_KINCAID | 0.7480 | **+0.0001** | -0.0001 | +0.0012 | +0.0000 | 0.7176 | negligible |
| 12 | FREQUENT_WORDS | 0.7480 | **+0.0001** | +0.0004 | +0.0006 | +0.0004 | 0.5428 | negligible |
| 13 | POS_NGRAM | 0.7482 | **-0.0001** | -0.0001 | +0.0009 | +0.0021 | 0.5922 | negligible |
| 14 | FUNCTION_WORDS | 0.7482 | **-0.0001** | -0.0018 | +0.0014 | -0.0002 | 0.5733 | negligible |
| 15 | TFIDF | 0.7484 | **-0.0003** | -0.0004 | +0.0004 | +0.0005 | 0.6165 | negligible |
| 16 | ORTHOGRAPHY | 0.7486 | **-0.0005** | -0.0020 | +0.0022 | +0.0041 | 0.5465 | negligible |
| 17 | CONTRACTIONS | 0.7487 | **-0.0006** | -0.0004 | -0.0006 | -0.0005 | 0.5049 | negligible |
| 18 | LEXICAL_DIVERSITY | 0.7489 | **-0.0008** | -0.0017 | +0.0023 | +0.0013 | 0.5953 | negligible |

### Neural Network

Baseline (all 76 features): macro-F1 = 0.7533

| Rank | Feature group | Macro-F1 without G | Importance | Δ easy | Δ medium | Δ hard | Macro-F1 only G | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | FLESCH_READING_EASE | 0.7503 | **+0.0029** | +0.0049 | -0.0010 | +0.0138 | 0.7228 | negligible |
| 2 | POS_RATIOS | 0.7507 | **+0.0025** | +0.0063 | +0.0009 | +0.0098 | 0.5950 | negligible |
| 3 | WORD_LENGTH | 0.7525 | **+0.0007** | +0.0029 | -0.0001 | -0.0007 | 0.5859 | negligible |
| 4 | CONTRACTIONS | 0.7529 | **+0.0004** | +0.0007 | -0.0004 | +0.0002 | 0.4798 | negligible |
| 5 | CHAR_NGRAM | 0.7530 | **+0.0002** | +0.0027 | -0.0018 | +0.0088 | 0.4493 | negligible |
| 6 | FUNCTION_WORDS | 0.7532 | **+0.0000** | +0.0012 | -0.0006 | +0.0012 | 0.5449 | negligible |
| 7 | LAYOUT | 0.7533 | **-0.0001** | +0.0002 | +0.0003 | +0.0017 | 0.4975 | negligible |
| 8 | PUNCTUATION | 0.7534 | **-0.0002** | +0.0019 | -0.0006 | +0.0029 | 0.5667 | negligible |
| 9 | EMOTION | 0.7535 | **-0.0002** | +0.0006 | -0.0005 | +0.0027 | 0.5094 | negligible |
| 10 | LENGTH | 0.7535 | **-0.0002** | +0.0030 | -0.0020 | -0.0017 | 0.6891 | negligible |
| 11 | ORTHOGRAPHY | 0.7536 | **-0.0003** | +0.0004 | -0.0011 | +0.0000 | 0.5466 | negligible |
| 12 | FLESCH_KINCAID | 0.7536 | **-0.0003** | +0.0000 | -0.0019 | -0.0024 | 0.7176 | negligible |
| 13 | GUNNING_FOG | 0.7537 | **-0.0004** | +0.0001 | -0.0022 | -0.0004 | 0.7118 | negligible |
| 14 | PUNCTUATION_MARKS | 0.7537 | **-0.0004** | +0.0006 | +0.0009 | +0.0235 | 0.6068 | negligible |
| 15 | TFIDF | 0.7538 | **-0.0005** | +0.0002 | -0.0012 | +0.0001 | 0.6169 | negligible |
| 16 | POS_NGRAM | 0.7538 | **-0.0005** | +0.0008 | -0.0021 | +0.0028 | 0.5603 | negligible |
| 17 | LEXICAL_DIVERSITY | 0.7539 | **-0.0006** | +0.0012 | -0.0005 | -0.0012 | 0.5488 | negligible |
| 18 | FREQUENT_WORDS | 0.7539 | **-0.0007** | +0.0009 | -0.0012 | +0.0013 | 0.5401 | negligible |

### Importance across the models

A group that matters for ALL models is much stronger evidence than one that matters for one.

| Feature group | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| LENGTH | -0.0016 | +0.0003 | -0.0002 |
| FREQUENT_WORDS | -0.0000 | +0.0001 | -0.0007 |
| PUNCTUATION | -0.0001 | +0.0012 | -0.0002 |
| PUNCTUATION_MARKS | +0.0008 | +0.0006 | -0.0004 |
| ORTHOGRAPHY | +0.0000 | -0.0005 | -0.0003 |
| LAYOUT | -0.0001 | +0.0005 | -0.0001 |
| WORD_LENGTH | +0.0003 | +0.0010 | +0.0007 |
| TFIDF | -0.0000 | -0.0003 | -0.0005 |
| EMOTION | -0.0002 | +0.0002 | -0.0002 |
| LEXICAL_DIVERSITY | +0.0000 | -0.0008 | -0.0006 |
| FLESCH_READING_EASE | +0.0010 | +0.0009 | +0.0029 |
| GUNNING_FOG | +0.0001 | +0.0005 | -0.0004 |
| FLESCH_KINCAID | +0.0012 | +0.0001 | -0.0003 |
| CONTRACTIONS | -0.0000 | -0.0006 | +0.0004 |
| FUNCTION_WORDS | -0.0011 | -0.0001 | +0.0000 |
| POS_RATIOS | +0.0006 | +0.0036 | +0.0025 |
| POS_NGRAM | +0.0004 | -0.0001 | -0.0005 |
| CHAR_NGRAM | +0.0053 | +0.0037 | +0.0002 |

### Importance by difficulty level (mean over the models)

Mean of `MacroF1_all - MacroF1_without_G` over the models, per level. A group that helps on **hard** is a style signal (topic is constant there); a group that helps only on **easy** may be a topic signal.

| Feature group | easy | medium | hard |
|---|---:|---:|---:|
| LENGTH | +0.0027 | -0.0007 | -0.0028 |
| FREQUENT_WORDS | +0.0004 | -0.0002 | +0.0005 |
| PUNCTUATION | +0.0007 | +0.0006 | +0.0010 |
| PUNCTUATION_MARKS | +0.0023 | +0.0009 | +0.0092 |
| ORTHOGRAPHY | -0.0006 | +0.0005 | +0.0014 |
| LAYOUT | -0.0003 | +0.0002 | +0.0006 |
| WORD_LENGTH | +0.0014 | -0.0007 | +0.0006 |
| TFIDF | -0.0001 | -0.0003 | +0.0002 |
| EMOTION | +0.0000 | -0.0001 | +0.0006 |
| LEXICAL_DIVERSITY | -0.0001 | +0.0005 | +0.0000 |
| FLESCH_READING_EASE | +0.0012 | -0.0008 | +0.0021 |
| GUNNING_FOG | +0.0005 | -0.0005 | +0.0001 |
| FLESCH_KINCAID | +0.0005 | -0.0007 | -0.0008 |
| CONTRACTIONS | +0.0001 | -0.0003 | -0.0001 |
| FUNCTION_WORDS | +0.0000 | +0.0001 | -0.0005 |
| POS_RATIOS | +0.0029 | +0.0002 | +0.0022 |
| POS_NGRAM | +0.0000 | -0.0012 | +0.0016 |
| CHAR_NGRAM | +0.0046 | -0.0008 | +0.0052 |

## 2. Parameter ablation

One hyperparameter is changed at a time; everything else stays at the baseline of Main.java. `◀` = the baseline value, `★` = best validation macro-F1 of that parameter. The train − validation gap shows over-fitting.

### Decision Tree

**max_depth**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.6909 | 0.7189 | 0.7228 | -0.0279 | -0.0265 |  |
| 5 | 0.7216 | 0.7460 | 0.7493 | -0.0244 | +0.0000 | ◀ ★ |
| 6 | 0.7301 | 0.7445 | 0.7482 | -0.0145 | -0.0011 |  |
| 20 | 0.8114 | 0.6990 | 0.7025 | +0.1124 | -0.0468 |  |

**min_samples_split**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 2 | 0.7216 | 0.7460 | 0.7493 | -0.0244 | +0.0000 |  |
| 20 | 0.7216 | 0.7460 | 0.7493 | -0.0244 | +0.0000 | ◀  |
| 50 | 0.7213 | 0.7461 | 0.7494 | -0.0248 | +0.0001 | ★ |
| 500 | 0.7194 | 0.7457 | 0.7486 | -0.0263 | -0.0007 |  |

**min_samples_leaf**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.7216 | 0.7460 | 0.7493 | -0.0243 | -0.0001 |  |
| 5 | 0.7216 | 0.7460 | 0.7493 | -0.0244 | +0.0000 | ◀ ★ |
| 10 | 0.7215 | 0.7460 | 0.7493 | -0.0245 | -0.0000 |  |
| 100 | 0.7194 | 0.7457 | 0.7486 | -0.0263 | -0.0007 |  |

**min_impurity_decrease**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7216 | 0.7460 | 0.7493 | -0.0244 | +0.0000 | ◀  |
| 1.0E-4 | 0.7216 | 0.7460 | 0.7493 | -0.0244 | +0.0000 |  |
| 1.0E-2 | 0.7208 | 0.7460 | 0.7493 | -0.0253 | +0.0000 | ★ |

**class_weights**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| true | 0.7216 | 0.7460 | 0.7493 | -0.0244 | +0.0000 | ◀  |
| false | 0.7203 | 0.7481 | 0.7525 | -0.0278 | +0.0032 | ★ |

**split_bins**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.7216 | 0.7460 | 0.7493 | -0.0244 | +0.0000 | ◀  |
| 64 | 0.7210 | 0.7465 | 0.7496 | -0.0256 | +0.0003 | ★ |
| 255 | 0.7216 | 0.7458 | 0.7487 | -0.0242 | -0.0006 |  |

**Decision tree: max_depth (rows) × min_samples_leaf (columns), Validation macro-F1**

| depth \ leaf | 5 | 100 |
|---|---:|---:|
| 3 | 0.7427 | 0.7427 |
| 8 | 0.7434 | 0.7472 |

**Decision tree: max_depth (rows) × min_samples_leaf (columns), Test macro-F1**

| depth \ leaf | 5 | 100 |
|---|---:|---:|
| 3 | 0.7459 | 0.7459 |
| 8 | 0.7483 | 0.7510 |

### Logistic Regression

**learning_rate**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.01 | 0.7187 | 0.7516 | 0.7548 | -0.0328 | +0.0067 | ★ |
| 0.3 | 0.7123 | 0.7483 | 0.7510 | -0.0360 | +0.0029 |  |
| 0.6 | 0.7099 | 0.7456 | 0.7481 | -0.0357 | +0.0000 | ◀  |
| 1.0 | 0.7088 | 0.7446 | 0.7464 | -0.0358 | -0.0017 |  |

**iterations**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 100 | 0.6927 | 0.7221 | 0.7257 | -0.0294 | -0.0225 |  |
| 2500 | 0.7109 | 0.7453 | 0.7477 | -0.0345 | -0.0004 |  |
| 10000 | 0.7099 | 0.7456 | 0.7481 | -0.0357 | +0.0000 | ◀ ★ |

**l2_lambda**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7099 | 0.7456 | 0.7481 | -0.0357 | +0.0000 | ◀ ★ |
| 1.0E-3 | 0.7103 | 0.7450 | 0.7484 | -0.0347 | +0.0003 |  |
| 1.0E-1 | 0.7047 | 0.7378 | 0.7410 | -0.0331 | -0.0071 |  |

**class_weights**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| true | 0.7099 | 0.7456 | 0.7481 | -0.0357 | +0.0000 | ◀  |
| false | 0.7226 | 0.7570 | 0.7593 | -0.0344 | +0.0112 | ★ |

### Neural Network

**hidden_layers**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 16 | 0.7217 | 0.7523 | 0.7549 | -0.0307 | +0.0016 | ★ |
| 64-32 | 0.7211 | 0.7513 | 0.7550 | -0.0302 | +0.0018 |  |
| 128-64-32-16-8 | 0.7199 | 0.7519 | 0.7533 | -0.0320 | +0.0000 | ◀  |

**l2_lambda**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7440 | 0.7565 | 0.7590 | -0.0126 | +0.0058 |  |
| 1.0E-3 | 0.7327 | 0.7583 | 0.7602 | -0.0256 | +0.0069 | ★ |
| 0.01 | 0.7199 | 0.7519 | 0.7533 | -0.0320 | +0.0000 | ◀  |
| 1.0E-1 | 0.4613 | 0.4480 | 0.4493 | +0.0133 | -0.3040 |  |

**dropout**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7194 | 0.7516 | 0.7543 | -0.0322 | +0.0011 |  |
| 0.2 | 0.7199 | 0.7519 | 0.7533 | -0.0320 | +0.0000 | ◀ ★ |
| 0.5 | 0.7186 | 0.7500 | 0.7530 | -0.0314 | -0.0002 |  |

**learning_rate**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1.0E-4 | 0.7205 | 0.7530 | 0.7560 | -0.0325 | +0.0028 | ★ |
| 1.0E-3 | 0.7199 | 0.7519 | 0.7533 | -0.0320 | +0.0000 | ◀  |
| 1.0E-2 | 0.7163 | 0.7478 | 0.7506 | -0.0315 | -0.0027 |  |

**optimizer**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| adam | 0.7199 | 0.7519 | 0.7533 | -0.0320 | +0.0000 | ◀  |
| sgd | 0.7188 | 0.7522 | 0.7547 | -0.0334 | +0.0014 | ★ |

**batch_size**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 32 | 0.7173 | 0.7494 | 0.7521 | -0.0320 | -0.0012 |  |
| 64 | 0.7199 | 0.7519 | 0.7533 | -0.0320 | +0.0000 | ◀ ★ |
| 128 | 0.7182 | 0.7516 | 0.7541 | -0.0334 | +0.0008 |  |
| 256 | 0.7207 | 0.7516 | 0.7540 | -0.0309 | +0.0007 |  |

**feature_clip**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.7206 | 0.7517 | 0.7532 | -0.0311 | -0.0001 |  |
| 3.0 | 0.7174 | 0.7513 | 0.7543 | -0.0338 | +0.0010 |  |
| 5.0 | 0.7199 | 0.7519 | 0.7533 | -0.0320 | +0.0000 | ◀ ★ |

**patience**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.7183 | 0.7506 | 0.7533 | -0.0323 | +0.0001 |  |
| 3 | 0.7189 | 0.7509 | 0.7541 | -0.0320 | +0.0008 |  |
| 8 | 0.7199 | 0.7519 | 0.7533 | -0.0320 | +0.0000 | ◀ ★ |

### Best value of each parameter

Each parameter is tuned on its own (the others stay at baseline), so combining all the best values is a *candidate* that still has to be trained and checked, not a guaranteed improvement.

**Decision Tree**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| max_depth | 5 | 0.7460 | 0.7493 | 5 | 0.7493 |
| min_samples_split | 50 | 0.7461 | 0.7494 | 20 | 0.7493 |
| min_samples_leaf | 5 | 0.7460 | 0.7493 | 5 | 0.7493 |
| min_impurity_decrease | 1.0E-2 | 0.7460 | 0.7493 | 0.0 | 0.7493 |
| class_weights | false | 0.7481 | 0.7525 | true | 0.7493 |
| split_bins | 64 | 0.7465 | 0.7496 | 0 | 0.7493 |

**Logistic Regression**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| learning_rate | 0.01 | 0.7516 | 0.7548 | 0.6 | 0.7481 |
| iterations | 10000 | 0.7456 | 0.7481 | 10000 | 0.7481 |
| l2_lambda | 0.0 | 0.7456 | 0.7481 | 0.0 | 0.7481 |
| class_weights | false | 0.7570 | 0.7593 | true | 0.7481 |

**Neural Network**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| hidden_layers | 16 | 0.7523 | 0.7549 | 128-64-32-16-8 | 0.7533 |
| l2_lambda | 1.0E-3 | 0.7583 | 0.7602 | 0.01 | 0.7533 |
| dropout | 0.2 | 0.7519 | 0.7533 | 0.2 | 0.7533 |
| learning_rate | 1.0E-4 | 0.7530 | 0.7560 | 1.0E-3 | 0.7533 |
| optimizer | sgd | 0.7522 | 0.7547 | adam | 0.7533 |
| batch_size | 64 | 0.7519 | 0.7533 | 64 | 0.7533 |
| feature_clip | 5.0 | 0.7519 | 0.7533 | 5.0 | 0.7533 |
| patience | 8 | 0.7519 | 0.7533 | 8 | 0.7533 |

## 3. Feature-pipeline ablation

The features are computed again for every setting (casing, character n-gram order and buckets, POS n-grams, pair representation, ...) and the models are trained with the baseline hyperparameters. Every cell is `validation / test` macro-F1; `◀` = the baseline setting, `★` = best validation macro-F1 of that model within the family.

### Casing of the text (preserve / lower-case / hybrid)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| preserve ◀ | 76 | 0.7460 / 0.7493 ★ | 0.7456 / 0.7481 ★ | 0.7519 / 0.7533 ★ |
| lowercase | 76 | 0.7445 / 0.7467 | 0.7449 / 0.7473 | 0.7510 / 0.7534 |
| hybrid | 76 | 0.7450 / 0.7475 | 0.7455 / 0.7482 | 0.7518 / 0.7536 |

### Character n-gram order n

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| n=1 | 76 | 0.7428 / 0.7442 | 0.7450 / 0.7465 | 0.7514 / 0.7530 |
| n=3 ◀ | 76 | 0.7460 / 0.7493 | 0.7456 / 0.7481 | 0.7519 / 0.7533 |
| n=1,2 | 80 | 0.7481 / 0.7493 ★ | 0.7473 / 0.7494 ★ | 0.7520 / 0.7543 ★ |
| none | 72 | 0.7397 / 0.7440 | 0.7418 / 0.7444 | 0.7500 / 0.7530 |

### Hash buckets of the character profile

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| buckets=16 | 76 | 0.7397 / 0.7441 | 0.7425 / 0.7453 | 0.7513 / 0.7530 |
| buckets=256 ◀ | 76 | 0.7460 / 0.7493 | 0.7456 / 0.7481 | 0.7519 / 0.7533 ★ |
| buckets=2048 | 76 | 0.7557 / 0.7561 ★ | 0.7483 / 0.7504 ★ | 0.7518 / 0.7537 |

### Typed character n-grams (affix + punctuation)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| all n-grams ◀ | 76 | 0.7460 / 0.7493 ★ | 0.7456 / 0.7481 ★ | 0.7519 / 0.7533 |
| typed (affix+punct) | 76 | 0.7436 / 0.7469 | 0.7454 / 0.7478 | 0.7520 / 0.7537 ★ |

### POS n-gram order

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| none | 72 | 0.7460 / 0.7489 | 0.7457 / 0.7482 | 0.7516 / 0.7538 |
| n=2 ◀ | 76 | 0.7460 / 0.7493 | 0.7456 / 0.7481 | 0.7519 / 0.7533 ★ |
| n=3 | 76 | 0.7460 / 0.7490 ★ | 0.7455 / 0.7484 | 0.7516 / 0.7534 |
| n=1,2,3 | 84 | 0.7460 / 0.7493 | 0.7466 / 0.7489 ★ | 0.7512 / 0.7529 |

### Hash buckets of the POS profile

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| buckets=32 | 76 | 0.7462 / 0.7491 | 0.7457 / 0.7484 ★ | 0.7521 / 0.7532 ★ |
| buckets=128 ◀ | 76 | 0.7460 / 0.7493 | 0.7456 / 0.7481 | 0.7519 / 0.7533 |
| buckets=512 | 76 | 0.7462 / 0.7489 ★ | 0.7454 / 0.7480 | 0.7516 / 0.7537 |

### POS tagger

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| heuristic ◀ | 76 | 0.7460 / 0.7493 ★ | 0.7456 / 0.7481 ★ | 0.7519 / 0.7533 ★ |

### Function-word profile

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| off | 72 | 0.7474 / 0.7504 ★ | 0.7455 / 0.7478 | 0.7499 / 0.7533 |
| on ◀ | 76 | 0.7460 / 0.7493 | 0.7456 / 0.7481 ★ | 0.7519 / 0.7533 ★ |

### Pair representation (how two sentences become one vector)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| absolute ◀ | 76 | 0.7460 / 0.7493 | 0.7456 / 0.7481 | 0.7519 / 0.7533 |
| absolute_and_mean | 139 | 0.7462 / 0.7494 | 0.7496 / 0.7503 | 0.7644 / 0.7663 ★ |
| absolute_and_product | 139 | 0.7493 / 0.7509 ★ | 0.7510 / 0.7530 ★ | 0.7589 / 0.7604 |

### Cut of the z-scores of the scalar features

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| off | 76 | 0.7460 / 0.7493 | 0.7450 / 0.7481 | 0.7504 / 0.7525 |
| +-3 | 76 | 0.7443 / 0.7478 | 0.7469 / 0.7498 ★ | 0.7542 / 0.7558 ★ |
| +-5 ◀ | 76 | 0.7460 / 0.7493 ★ | 0.7456 / 0.7481 | 0.7519 / 0.7533 |
| +-10 | 76 | 0.7460 / 0.7493 | 0.7451 / 0.7484 | 0.7511 / 0.7531 |

### Compressed (4 measures per profile) vs full (every bucket difference)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| compressed (4 measures) ◀ | 76 | 0.7460 / 0.7493 ★ | 0.7456 / 0.7481 ★ | 0.7519 / 0.7533 |
| full (every bucket) | 674 | 0.7458 / 0.7491 | 0.7310 / 0.7329 | 0.7589 / 0.7607 ★ |

## 3b. Feature-pipeline ablation, by difficulty level

The same runs as section 3, but the evaluation macro-F1 of each level separately (threshold chosen on that level's validation data). Cell = easy / medium / hard.

### Casing of the text (preserve / lower-case / hybrid)

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| preserve ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| lowercase | 0.9323 / 0.4991 / 0.5386 | 0.9340 / 0.5027 / 0.5363 | 0.9445 / 0.5034 / 0.5236 |
| hybrid | 0.9315 / 0.4985 / 0.5389 | 0.9351 / 0.5060 / 0.5363 | 0.9473 / 0.5026 / 0.5165 |

### Character n-gram order n

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| n=1 | 0.9277 / 0.4998 / 0.5482 | 0.9339 / 0.5051 / 0.5328 | 0.9464 / 0.5018 / 0.5141 |
| n=3 ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| n=1,2 | 0.9306 / 0.4975 / 0.5380 | 0.9384 / 0.5064 / 0.5369 | 0.9467 / 0.5041 / 0.5260 |
| none | 0.9273 / 0.5016 / 0.5447 | 0.9278 / 0.5033 / 0.5239 | 0.9444 / 0.5043 / 0.5165 |

### Hash buckets of the character profile

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| buckets=16 | 0.9278 / 0.5016 / 0.5449 | 0.9296 / 0.5045 / 0.5256 | 0.9459 / 0.5039 / 0.5247 |
| buckets=256 ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| buckets=2048 | 0.9425 / 0.5019 / 0.5625 | 0.9399 / 0.5067 / 0.5379 | 0.9474 / 0.5027 / 0.5233 |

### Typed character n-grams (affix + punctuation)

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| all n-grams ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| typed (affix+punct) | 0.9294 / 0.4974 / 0.5414 | 0.9342 / 0.5045 / 0.5333 | 0.9472 / 0.5030 / 0.5184 |

### POS n-gram order

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| none | 0.9324 / 0.5014 / 0.5393 | 0.9346 / 0.5043 / 0.5341 | 0.9464 / 0.5046 / 0.5226 |
| n=2 ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| n=3 | 0.9323 / 0.5010 / 0.5390 | 0.9341 / 0.5043 / 0.5358 | 0.9466 / 0.5029 / 0.5154 |
| n=1,2,3 | 0.9314 / 0.4990 / 0.5388 | 0.9361 / 0.5056 / 0.5293 | 0.9455 / 0.5017 / 0.5259 |

### Hash buckets of the POS profile

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| buckets=32 | 0.9325 / 0.4980 / 0.5386 | 0.9351 / 0.5042 / 0.5348 | 0.9476 / 0.5029 / 0.5245 |
| buckets=128 ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| buckets=512 | 0.9337 / 0.4991 / 0.5447 | 0.9341 / 0.5051 / 0.5368 | 0.9472 / 0.5019 / 0.5244 |

### POS tagger

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| heuristic ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |

### Function-word profile

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| off | 0.9311 / 0.4994 / 0.5417 | 0.9358 / 0.5027 / 0.5363 | 0.9449 / 0.5048 / 0.5236 |
| on ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |

### Pair representation (how two sentences become one vector)

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| absolute ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| absolute_and_mean | 0.9301 / 0.4992 / 0.5406 | 0.9355 / 0.5092 / 0.5586 | 0.9640 / 0.5077 / 0.5278 |
| absolute_and_product | 0.9333 / 0.5005 / 0.5325 | 0.9388 / 0.5055 / 0.5493 | 0.9501 / 0.5016 / 0.5440 |

### Cut of the z-scores of the scalar features

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| off | 0.9317 / 0.4990 / 0.5393 | 0.9357 / 0.5032 / 0.5333 | 0.9454 / 0.5023 / 0.5200 |
| +-3 | 0.9348 / 0.5003 / 0.5451 | 0.9346 / 0.5031 / 0.5372 | 0.9480 / 0.5017 / 0.5277 |
| +-5 ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| +-10 | 0.9317 / 0.4990 / 0.5393 | 0.9357 / 0.5041 / 0.5340 | 0.9466 / 0.5035 / 0.5228 |

### Compressed (4 measures per profile) vs full (every bucket difference)

| Setting | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| compressed (4 measures) ◀ | 0.9318 / 0.4990 / 0.5393 | 0.9345 / 0.5052 / 0.5362 | 0.9471 / 0.5024 / 0.5254 |
| full (every bucket) | 0.9307 / 0.4990 / 0.5393 | 0.9086 / 0.5094 / 0.5442 | 0.9558 / 0.5074 / 0.5466 |

## 4. How to read this (limitations to state in the paper)

- **Correlated groups.** If two groups carry the same information (e.g. sentence length and word length), removing one costs little because the other compensates, so a small importance does not prove a group is useless. The "only G" column helps tell the two cases apart.
- **One parameter at a time** ignores interactions between parameters (the decision-tree grid is the only place interactions are looked at).
- **Single run vs seeds.** Without several seeds there is no variance estimate; differences of a few thousandths of F1 should not be interpreted. Run with `-Dseeds=42,123,456` for the final numbers.
- **POS tags** come from the lightweight rule-based tagger (features.HeuristicPosTagger) unless `pos_tagger: corenlp` is in the pipeline table, so the POS result describes that tagger.
- **Pipeline settings are compared one at a time** around the baseline. The best value of each family is a candidate, not a proven combination (the optional `char_grid` family looks at n x buckets together).
- Parameters and pipeline settings are chosen on validation macro-F1; the test column is reported once, for the chosen values, and should not be used to pick settings.
