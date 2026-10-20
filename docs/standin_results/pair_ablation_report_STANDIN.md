# Ablation study report

- Training pairs: 20000
- Evaluation pairs (held-out test half): 6727
- Pair features in the full model: 76
- Baseline feature settings: `case=preserve|char=[3]x256|pos=[2]x128(heuristic)|fw=true|full=false|pair=ABSOLUTE|zclip=5.0`
- Seeds: [42] (single run, no variance estimate)
- Total run time: 8.9 min

**Metric:** macro-F1 (the mean of the F1 of the *change* class and of the *no change* class, pooled over all boundaries - the measure PAN 2026 ranks by) on the held-out test half. The F1 of the change class alone is in `ablation_results.csv`. Thresholds (all three models) are chosen on the validation data by macro-F1, never on the evaluation data. Parameters and pipeline settings are ranked by **validation** macro-F1.

## 1. Feature-group ablation

`Importance(G) = MacroF1_all - MacroF1_without_G`. Positive = the group carries useful information. Near 0 = it adds nothing the other groups do not already provide. Negative = the model does better without it. "Only G" trains on that group alone. The per-difficulty columns are the same difference on easy / medium / hard pairs.

### Decision Tree

Baseline (all 76 features): macro-F1 = 0.5521

| Rank | Feature group | Macro-F1 without G | Importance | Δ easy | Δ medium | Δ hard | Macro-F1 only G | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | FUNCTION_WORDS | 0.5415 | **+0.0106** | -0.0018 | +0.0191 | +0.0109 | 0.5461 | useful |
| 2 | PUNCTUATION_MARKS | 0.5435 | **+0.0086** | +0.0189 | +0.0110 | -0.0099 | 0.5122 | useful |
| 3 | POS_NGRAM | 0.5440 | **+0.0081** | +0.0294 | +0.0093 | -0.0204 | 0.5394 | useful |
| 4 | LENGTH | 0.5468 | **+0.0053** | -0.0293 | +0.0119 | +0.0305 | 0.5267 | useful |
| 5 | LAYOUT | 0.5519 | **+0.0002** | +0.0015 | -0.0006 | +0.0002 | 0.5530 | negligible |
| 6 | FREQUENT_WORDS | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5341 | negligible |
| 7 | PUNCTUATION | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5171 | negligible |
| 8 | TFIDF | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5099 | negligible |
| 9 | EMOTION | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.4397 | negligible |
| 10 | LEXICAL_DIVERSITY | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5289 | negligible |
| 11 | FLESCH_READING_EASE | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5290 | negligible |
| 12 | GUNNING_FOG | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5277 | negligible |
| 13 | FLESCH_KINCAID | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5048 | negligible |
| 14 | CONTRACTIONS | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5259 | negligible |
| 15 | POS_RATIOS | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5367 | negligible |
| 16 | CHAR_NGRAM | 0.5521 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | 0.5378 | negligible |
| 17 | WORD_LENGTH | 0.5534 | **-0.0013** | +0.0099 | +0.0017 | -0.0162 | 0.5263 | negligible |
| 18 | ORTHOGRAPHY | 0.5555 | **-0.0034** | -0.0317 | -0.0131 | +0.0334 | 0.5325 | negligible |

### Logistic Regression

Baseline (all 76 features): macro-F1 = 0.5678

| Rank | Feature group | Macro-F1 without G | Importance | Δ easy | Δ medium | Δ hard | Macro-F1 only G | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | LAYOUT | 0.5548 | **+0.0130** | +0.0227 | +0.0054 | +0.0053 | 0.5530 | useful |
| 2 | ORTHOGRAPHY | 0.5549 | **+0.0129** | +0.0090 | +0.0061 | +0.0190 | 0.5289 | useful |
| 3 | FUNCTION_WORDS | 0.5593 | **+0.0085** | +0.0086 | +0.0070 | +0.0093 | 0.5398 | useful |
| 4 | POS_NGRAM | 0.5614 | **+0.0064** | -0.0014 | +0.0036 | +0.0155 | 0.5421 | useful |
| 5 | FREQUENT_WORDS | 0.5646 | **+0.0033** | +0.0017 | -0.0038 | +0.0151 | 0.5262 | negligible |
| 6 | PUNCTUATION | 0.5664 | **+0.0015** | +0.0077 | -0.0013 | -0.0009 | 0.5297 | negligible |
| 7 | FLESCH_KINCAID | 0.5674 | **+0.0005** | +0.0008 | +0.0029 | -0.0026 | 0.5320 | negligible |
| 8 | CHAR_NGRAM | 0.5674 | **+0.0005** | -0.0020 | -0.0016 | +0.0057 | 0.5366 | negligible |
| 9 | FLESCH_READING_EASE | 0.5678 | **+0.0000** | +0.0008 | +0.0007 | -0.0021 | 0.5310 | negligible |
| 10 | POS_RATIOS | 0.5679 | **-0.0001** | -0.0034 | -0.0026 | +0.0064 | 0.5322 | negligible |
| 11 | EMOTION | 0.5681 | **-0.0003** | +0.0000 | +0.0000 | -0.0006 | 0.4397 | negligible |
| 12 | TFIDF | 0.5684 | **-0.0006** | +0.0134 | -0.0016 | -0.0154 | 0.5173 | negligible |
| 13 | GUNNING_FOG | 0.5685 | **-0.0007** | +0.0010 | -0.0000 | -0.0028 | 0.5235 | negligible |
| 14 | LENGTH | 0.5692 | **-0.0014** | +0.0048 | -0.0020 | -0.0055 | 0.5399 | negligible |
| 15 | CONTRACTIONS | 0.5697 | **-0.0019** | -0.0007 | -0.0004 | -0.0038 | 0.5257 | negligible |
| 16 | WORD_LENGTH | 0.5704 | **-0.0026** | +0.0001 | +0.0002 | -0.0053 | 0.5309 | negligible |
| 17 | LEXICAL_DIVERSITY | 0.5713 | **-0.0034** | -0.0016 | -0.0016 | -0.0045 | 0.5249 | negligible |
| 18 | PUNCTUATION_MARKS | 0.5715 | **-0.0036** | +0.0202 | -0.0031 | -0.0284 | 0.5352 | negligible |

### Neural Network

Baseline (all 76 features): macro-F1 = 0.5619

| Rank | Feature group | Macro-F1 without G | Importance | Δ easy | Δ medium | Δ hard | Macro-F1 only G | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | LAYOUT | 0.5512 | **+0.0106** | +0.0045 | +0.0149 | +0.0109 | 0.5528 | useful |
| 2 | ORTHOGRAPHY | 0.5539 | **+0.0079** | -0.0018 | +0.0019 | +0.0222 | 0.5264 | useful |
| 3 | FUNCTION_WORDS | 0.5578 | **+0.0041** | +0.0109 | +0.0095 | -0.0071 | 0.5434 | negligible |
| 4 | EMOTION | 0.5611 | **+0.0008** | +0.0197 | +0.0166 | -0.0250 | 0.4397 | negligible |
| 5 | FREQUENT_WORDS | 0.5625 | **-0.0007** | +0.0134 | +0.0051 | -0.0154 | 0.5245 | negligible |
| 6 | CONTRACTIONS | 0.5627 | **-0.0008** | +0.0010 | +0.0120 | -0.0097 | 0.5289 | negligible |
| 7 | PUNCTUATION | 0.5631 | **-0.0012** | +0.0235 | +0.0096 | -0.0312 | 0.5253 | negligible |
| 8 | WORD_LENGTH | 0.5634 | **-0.0016** | +0.0293 | +0.0169 | -0.0425 | 0.5296 | negligible |
| 9 | LENGTH | 0.5635 | **-0.0016** | +0.0294 | +0.0119 | -0.0378 | 0.5298 | negligible |
| 10 | TFIDF | 0.5635 | **-0.0016** | +0.0022 | -0.0028 | -0.0034 | 0.5142 | negligible |
| 11 | GUNNING_FOG | 0.5639 | **-0.0021** | +0.0081 | +0.0040 | -0.0152 | 0.5257 | negligible |
| 12 | LEXICAL_DIVERSITY | 0.5640 | **-0.0022** | +0.0149 | -0.0015 | -0.0183 | 0.5280 | negligible |
| 13 | CHAR_NGRAM | 0.5646 | **-0.0027** | -0.0021 | +0.0016 | -0.0024 | 0.4397 | negligible |
| 14 | FLESCH_KINCAID | 0.5652 | **-0.0033** | -0.0012 | +0.0022 | -0.0066 | 0.5316 | negligible |
| 15 | PUNCTUATION_MARKS | 0.5663 | **-0.0044** | +0.0051 | +0.0048 | -0.0218 | 0.5489 | negligible |
| 16 | FLESCH_READING_EASE | 0.5663 | **-0.0045** | -0.0014 | -0.0009 | -0.0084 | 0.5280 | negligible |
| 17 | POS_NGRAM | 0.5667 | **-0.0048** | +0.0078 | +0.0013 | -0.0206 | 0.5432 | negligible |
| 18 | POS_RATIOS | 0.5717 | **-0.0099** | +0.0025 | +0.0069 | -0.0234 | 0.5310 | removal helps |

### Importance across the models

A group that matters for ALL models is much stronger evidence than one that matters for one.

| Feature group | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|
| LENGTH | +0.0053 | -0.0014 | -0.0016 |
| FREQUENT_WORDS | +0.0000 | +0.0033 | -0.0007 |
| PUNCTUATION | +0.0000 | +0.0015 | -0.0012 |
| PUNCTUATION_MARKS | +0.0086 | -0.0036 | -0.0044 |
| ORTHOGRAPHY | -0.0034 | +0.0129 | +0.0079 |
| LAYOUT | +0.0002 | +0.0130 | +0.0106 |
| WORD_LENGTH | -0.0013 | -0.0026 | -0.0016 |
| TFIDF | +0.0000 | -0.0006 | -0.0016 |
| EMOTION | +0.0000 | -0.0003 | +0.0008 |
| LEXICAL_DIVERSITY | +0.0000 | -0.0034 | -0.0022 |
| FLESCH_READING_EASE | +0.0000 | +0.0000 | -0.0045 |
| GUNNING_FOG | +0.0000 | -0.0007 | -0.0021 |
| FLESCH_KINCAID | +0.0000 | +0.0005 | -0.0033 |
| CONTRACTIONS | +0.0000 | -0.0019 | -0.0008 |
| FUNCTION_WORDS | +0.0106 | +0.0085 | +0.0041 |
| POS_RATIOS | +0.0000 | -0.0001 | -0.0099 |
| POS_NGRAM | +0.0081 | +0.0064 | -0.0048 |
| CHAR_NGRAM | +0.0000 | +0.0005 | -0.0027 |

## 2. Parameter ablation

One hyperparameter is changed at a time; everything else stays at the baseline of Main.java. `◀` = the baseline value, `★` = best validation macro-F1 of that parameter. The train − validation gap shows over-fitting.

### Decision Tree

**max_depth**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.4799 | 0.4597 | 0.4380 | +0.0202 | -0.1141 |  |
| 5 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 | ◀ ★ |
| 6 | 0.6239 | 0.5570 | 0.5498 | +0.0669 | -0.0023 |  |
| 20 | 0.8705 | 0.5489 | 0.5284 | +0.3216 | -0.0237 |  |

**min_samples_split**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 2 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 |  |
| 20 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 | ◀  |
| 50 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 |  |
| 500 | 0.6163 | 0.5630 | 0.5536 | +0.0533 | +0.0015 | ★ |

**min_samples_leaf**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 | ★ |
| 5 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 | ◀  |
| 10 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 |  |
| 100 | 0.6201 | 0.5620 | 0.5538 | +0.0581 | +0.0017 |  |

**min_impurity_decrease**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 | ◀ ★ |
| 1.0E-4 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 |  |
| 1.0E-2 | 0.6178 | 0.5372 | 0.5374 | +0.0806 | -0.0147 |  |

**class_weights**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| true | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 | ◀ ★ |
| false | 0.6183 | 0.5535 | 0.5496 | +0.0648 | -0.0025 |  |

**split_bins**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.6221 | 0.5621 | 0.5521 | +0.0600 | +0.0000 | ◀  |
| 64 | 0.6238 | 0.5628 | 0.5519 | +0.0610 | -0.0002 | ★ |
| 255 | 0.5892 | 0.5464 | 0.5326 | +0.0428 | -0.0195 |  |

**Decision tree: max_depth (rows) × min_samples_leaf (columns), Validation macro-F1**

| depth \ leaf | 5 | 100 |
|---|---:|---:|
| 3 | 0.5373 | 0.5373 |
| 8 | 0.5527 | 0.5658 |

**Decision tree: max_depth (rows) × min_samples_leaf (columns), Test macro-F1**

| depth \ leaf | 5 | 100 |
|---|---:|---:|
| 3 | 0.5393 | 0.5393 |
| 8 | 0.5439 | 0.5534 |

### Logistic Regression

**learning_rate**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.01 | 0.6185 | 0.5665 | 0.5598 | +0.0521 | -0.0080 |  |
| 0.3 | 0.6315 | 0.5831 | 0.5665 | +0.0484 | -0.0014 | ★ |
| 0.6 | 0.6404 | 0.5828 | 0.5678 | +0.0575 | +0.0000 | ◀  |
| 1.0 | 0.6461 | 0.5744 | 0.5626 | +0.0717 | -0.0053 |  |

**iterations**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 100 | 0.6076 | 0.5620 | 0.5493 | +0.0456 | -0.0186 |  |
| 600 | 0.6404 | 0.5828 | 0.5678 | +0.0575 | +0.0000 | ◀  |
| 2500 | 0.6330 | 0.5863 | 0.5680 | +0.0467 | +0.0001 |  |
| 10000 | 0.6398 | 0.5915 | 0.5723 | +0.0483 | +0.0044 | ★ |

**l2_lambda**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.6404 | 0.5828 | 0.5678 | +0.0575 | +0.0000 | ◀ ★ |
| 1.0E-3 | 0.6390 | 0.5816 | 0.5641 | +0.0574 | -0.0038 |  |
| 1.0E-1 | 0.6109 | 0.5607 | 0.5572 | +0.0503 | -0.0107 |  |

**class_weights**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| true | 0.6404 | 0.5828 | 0.5678 | +0.0575 | +0.0000 | ◀ ★ |
| false | 0.6339 | 0.5751 | 0.5646 | +0.0587 | -0.0032 |  |

### Neural Network

**hidden_layers**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 16 | 0.6457 | 0.5752 | 0.5639 | +0.0706 | +0.0021 |  |
| 64-32 | 0.6397 | 0.5808 | 0.5653 | +0.0589 | +0.0035 | ★ |
| 128-64-32-16-8 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ◀  |

**l2_lambda**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.6477 | 0.5795 | 0.5688 | +0.0683 | +0.0069 | ★ |
| 1.0E-3 | 0.6563 | 0.5772 | 0.5548 | +0.0792 | -0.0071 |  |
| 0.01 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ◀  |
| 1.0E-1 | 0.4362 | 0.4373 | 0.4397 | -0.0011 | -0.1221 |  |

**dropout**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.6490 | 0.5773 | 0.5669 | +0.0717 | +0.0051 | ★ |
| 0.2 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ◀  |
| 0.5 | 0.4620 | 0.4731 | 0.4775 | -0.0111 | -0.0844 |  |

**learning_rate**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 1.0E-4 | 0.5635 | 0.5555 | 0.5439 | +0.0080 | -0.0180 |  |
| 1.0E-3 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ◀ ★ |
| 1.0E-2 | 0.4362 | 0.4373 | 0.4397 | -0.0011 | -0.1221 |  |

**optimizer**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| adam | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ◀ ★ |
| sgd | 0.6201 | 0.5689 | 0.5645 | +0.0511 | +0.0027 |  |

**batch_size**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 32 | 0.6232 | 0.5772 | 0.5636 | +0.0460 | +0.0018 | ★ |
| 64 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ◀  |
| 128 | 0.6257 | 0.5716 | 0.5607 | +0.0541 | -0.0011 |  |
| 256 | 0.6253 | 0.5687 | 0.5589 | +0.0566 | -0.0030 |  |

**feature_clip**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0.0 | 0.6228 | 0.5743 | 0.5626 | +0.0485 | +0.0008 |  |
| 3.0 | 0.6293 | 0.5730 | 0.5645 | +0.0563 | +0.0026 |  |
| 5.0 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ◀ ★ |

**patience**

| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ★ |
| 3 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 |  |
| 8 | 0.6334 | 0.5746 | 0.5619 | +0.0588 | +0.0000 | ◀  |

### Best value of each parameter

Each parameter is tuned on its own (the others stay at baseline), so combining all the best values is a *candidate* that still has to be trained and checked, not a guaranteed improvement.

**Decision Tree**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| max_depth | 5 | 0.5621 | 0.5521 | 5 | 0.5521 |
| min_samples_split | 500 | 0.5630 | 0.5536 | 20 | 0.5521 |
| min_samples_leaf | 1 | 0.5621 | 0.5521 | 5 | 0.5521 |
| min_impurity_decrease | 0.0 | 0.5621 | 0.5521 | 0.0 | 0.5521 |
| class_weights | true | 0.5621 | 0.5521 | true | 0.5521 |
| split_bins | 64 | 0.5628 | 0.5519 | 0 | 0.5521 |

**Logistic Regression**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| learning_rate | 0.3 | 0.5831 | 0.5665 | 0.6 | 0.5678 |
| iterations | 10000 | 0.5915 | 0.5723 | 600 | 0.5678 |
| l2_lambda | 0.0 | 0.5828 | 0.5678 | 0.0 | 0.5678 |
| class_weights | true | 0.5828 | 0.5678 | true | 0.5678 |

**Neural Network**

| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |
|---|---|---:|---:|---|---:|
| hidden_layers | 64-32 | 0.5808 | 0.5653 | 128-64-32-16-8 | 0.5619 |
| l2_lambda | 0.0 | 0.5795 | 0.5688 | 0.01 | 0.5619 |
| dropout | 0.0 | 0.5773 | 0.5669 | 0.2 | 0.5619 |
| learning_rate | 1.0E-3 | 0.5746 | 0.5619 | 1.0E-3 | 0.5619 |
| optimizer | adam | 0.5746 | 0.5619 | adam | 0.5619 |
| batch_size | 32 | 0.5772 | 0.5636 | 64 | 0.5619 |
| feature_clip | 5.0 | 0.5746 | 0.5619 | 5.0 | 0.5619 |
| patience | 0 | 0.5746 | 0.5619 | 8 | 0.5619 |

## 3. Feature-pipeline ablation

The features are computed again for every setting (casing, character n-gram order and buckets, POS n-grams, pair representation, ...) and the models are trained with the baseline hyperparameters. Every cell is `validation / test` macro-F1; `◀` = the baseline setting, `★` = best validation macro-F1 of that model within the family.

### Casing of the text (preserve / lower-case / hybrid)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| preserve ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 | 0.5746 / 0.5619 |
| lowercase | 76 | 0.5554 / 0.5560 | 0.5785 / 0.5480 | 0.5656 / 0.5554 |
| hybrid | 76 | 0.5609 / 0.5507 | 0.5829 / 0.5679 ★ | 0.5762 / 0.5599 ★ |

### Character n-gram order n

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| n=1 | 76 | 0.5621 / 0.5521 ★ | 0.5806 / 0.5665 | 0.5747 / 0.5607 ★ |
| n=3 ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 | 0.5746 / 0.5619 |
| n=1,2 | 80 | 0.5621 / 0.5521 ★ | 0.5831 / 0.5666 ★ | 0.5739 / 0.5574 |
| none | 72 | 0.5621 / 0.5521 ★ | 0.5818 / 0.5674 | 0.5735 / 0.5646 |

### Hash buckets of the character profile

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| buckets=16 | 76 | 0.5621 / 0.5521 ★ | 0.5809 / 0.5666 | 0.5740 / 0.5633 |
| buckets=256 ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 ★ | 0.5746 / 0.5619 ★ |
| buckets=2048 | 76 | 0.5547 / 0.5431 | 0.5824 / 0.5704 | 0.5734 / 0.5570 |

### Typed character n-grams (affix + punctuation)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| all n-grams ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 ★ | 0.5746 / 0.5619 ★ |
| typed (affix+punct) | 76 | 0.5621 / 0.5521 ★ | 0.5818 / 0.5655 | 0.5727 / 0.5637 |

### POS n-gram order

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| none | 72 | 0.5578 / 0.5440 | 0.5805 / 0.5614 | 0.5734 / 0.5667 |
| n=2 ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 ★ | 0.5746 / 0.5619 ★ |
| n=3 | 76 | 0.5578 / 0.5442 | 0.5790 / 0.5617 | 0.5721 / 0.5598 |
| n=1,2,3 | 84 | 0.5621 / 0.5521 ★ | 0.5818 / 0.5685 | 0.5352 / 0.5338 |

### Hash buckets of the POS profile

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| buckets=32 | 76 | 0.5570 / 0.5435 | 0.5804 / 0.5645 | 0.5738 / 0.5595 |
| buckets=128 ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 ★ | 0.5746 / 0.5619 |
| buckets=512 | 76 | 0.5477 / 0.5534 | 0.5825 / 0.5673 | 0.5753 / 0.5621 ★ |

### POS tagger

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| heuristic ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 ★ | 0.5746 / 0.5619 ★ |

### Function-word profile

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| off | 72 | 0.5566 / 0.5416 | 0.5736 / 0.5591 | 0.5795 / 0.5704 ★ |
| on ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 ★ | 0.5746 / 0.5619 |

### Pair representation (how two sentences become one vector)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| absolute ◀ | 76 | 0.5621 / 0.5521 | 0.5828 / 0.5678 | 0.5746 / 0.5619 |
| absolute_and_mean | 139 | 0.5722 / 0.5656 ★ | 0.5983 / 0.5869 | 0.5970 / 0.5906 ★ |
| absolute_and_product | 139 | 0.5327 / 0.5311 | 0.6062 / 0.5900 ★ | 0.5951 / 0.5882 |

### Cut of the z-scores of the scalar features

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| off | 76 | 0.5621 / 0.5521 ★ | 0.5783 / 0.5662 | 0.5730 / 0.5597 |
| +-3 | 76 | 0.5362 / 0.5381 | 0.5778 / 0.5722 | 0.5741 / 0.5690 |
| +-5 ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 ★ | 0.5746 / 0.5619 |
| +-10 | 76 | 0.5621 / 0.5521 ★ | 0.5784 / 0.5683 | 0.5754 / 0.5622 ★ |

### Compressed (4 measures per profile) vs full (every bucket difference)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| compressed (4 measures) ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 | 0.5746 / 0.5619 |
| full (every bucket) | 674 | 0.5494 / 0.5484 | 0.5878 / 0.5685 ★ | 0.5914 / 0.5794 ★ |

### Character n-gram order x buckets (grid)

| Setting | Pair features | Decision Tree | Logistic Regression | Neural Network |
|---|---:|---:|---:|---:|
| n=1 x 64 | 76 | 0.5621 / 0.5521 ★ | 0.5805 / 0.5659 | 0.5751 / 0.5627 |
| n=3 x 256 ◀ | 76 | 0.5621 / 0.5521 ★ | 0.5828 / 0.5678 ★ | 0.5746 / 0.5619 |
| n=5 x 1024 | 76 | 0.5614 / 0.5554 | 0.5821 / 0.5668 | 0.5763 / 0.5644 ★ |

## 4. How to read this (limitations to state in the paper)

- **Correlated groups.** If two groups carry the same information (e.g. sentence length and word length), removing one costs little because the other compensates, so a small importance does not prove a group is useless. The "only G" column helps tell the two cases apart.
- **One parameter at a time** ignores interactions between parameters (the decision-tree grid is the only place interactions are looked at).
- **Single run vs seeds.** Without several seeds there is no variance estimate; differences of a few thousandths of F1 should not be interpreted. Run with `-Dseeds=42,123,456` for the final numbers.
- **POS tags** come from the lightweight rule-based tagger (features.HeuristicPosTagger) unless `pos_tagger: corenlp` is in the pipeline table, so the POS result describes that tagger.
- **Pipeline settings are compared one at a time** around the baseline. The best value of each family is a candidate, not a proven combination (the optional `char_grid` family looks at n x buckets together).
- Parameters and pipeline settings are chosen on validation macro-F1; the test column is reported once, for the chosen values, and should not be used to pick settings.
