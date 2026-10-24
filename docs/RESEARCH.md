# Research notes: features, models, sentence-pair representation, retrieval

Team redblinkinglight (T. P. Matlala, University of Johannesburg) - PAN 2026 Multi-Author Writing Style Analysis (MAWSA).

This file explains **why** the project computes what it computes, which choices are open questions, and
what the ablation study is built to answer. Every number about PAN 2026 in it was checked against the
official pages on 2026-10-19 (links in section 9). Anything about *results* is marked as either
**PAN data** (not available in the build sandbox) or **stand-in** (a smoke test, see section 8).

---------------------------------------------------------------------------------------------------

## 1. The task, as the organisers define it

| item | value | source |
|---|---|---|
| unit | for every pair of consecutive sentences: 1 = author changes, 0 = same author | task page |
| data | English **fan fiction** crawled from Archive of Our Own, 100 largest fandoms, top 100 character pairs per fandom, 2 to 4 authors per document | overview, section on the dataset |
| size | easy / medium / hard, **15,000 documents each**, split 70 / 15 / 15 | overview |
| difficulty | easy = topics vary (topic helps), medium = little topical variety, hard = one topic (only style helps) | task page |
| measure | **macro-F1** over all sentence pairs, each dataset scored separately | task page |
| baseline "always 0" | 0.410 / 0.489 / 0.441 (easy / medium / hard) | overview |
| baseline "always 1" | 0.234 / 0.042 / 0.173 | overview |
| best 2026 results | easy 1.000 (stylefusion, aimoment); medium 0.768 and hard 0.780 (datateam-tug); all DeBERTa-family encoders, some fused with LightGBM on stylometric features | overview |
| submission | `software -i INPUT -o OUTPUT`, one `solution-problem-N.json` with `{"changes": [0,1,...]}` per `problem-N.txt`, via TIRA | task page |

What this means for expectations: a model built only on hand-crafted features computed per sentence
is expected to sit far below that on medium and hard (this is an expectation from the literature, not a measured result: the
stand-in numbers below are not evidence for it, they only show the code works). The 2026 winners on medium and hard are around 0.77. The
contribution of this project is therefore an *interpretable, from-scratch, fully ablated* baseline, not a
leaderboard attempt, and the paper should say so.

---------------------------------------------------------------------------------------------------

## 2. Mapping of the review points to the project

| review point | what was done | where | status |
|---|---|---|---|
| TIRA deployment returned a constant 1 | `TiraRunner` loads the saved model, **refuses to start without one** (no silent fallback), keeps one output label per sentence pair, and writes the file for every input; `scripts/tira_selftest.sh` checks that predictions are not constant | `TiraRunner.java`, `Dockerfile`, `scripts/tira_selftest.sh` | tested locally on stand-in data; **Docker image and TIRA itself not built or run here** |
| results on the full dataset | `Main`, `Ablation`, `RetrievalMain` read `dataset/mawsa26-pan-zenodo-DATA/` (all three difficulties) and report overall + per difficulty; `-Dfull.validation=true` uses the whole validation split | `cli/Main.java`, `data/DatasetLoader.java` | **needs the Zenodo data (not reachable from the sandbox)** |
| justification of features and classifiers | sections 3 - 5 of this file, and the ablation (section 7) | this file | text done, numbers need real data |
| not all described features were used | the paper described 14 features + a trigram profile + a 16-feature pair vector. The code now has **63 scalar features + 3 profile types**, every one named in `FeatureExtractor.SCALAR_NAMES` and grouped in `FeatureGroup` so each group can be switched off | `features/` | done; the paper text must be rewritten to match (section 10) |
| title format | "Team redblinkinglight at PAN 2026: ..." | manuscript | **to do in the paper** (section 10) |
| official CLEF template | the draft still uses `llncs`; the CLEF 2026 working notes are published in CEUR-WS, whose CEURART template is what the PAN notebook instructions ask for | manuscript | **to do in the paper**, please check the notebook instructions you received |
| citations `bevendorff:2026b`, `zangerle:2026` | rebuilt from sources that could be opened: arXiv 2602.09147 (12 authors) and CEUR-WS Vol-4283 paper388. The ECIR extended abstract (`bevendorff:2026a`) lacked two authors; its volume number and DOI could not be confirmed and are left as a note | `scripts/manuscript/references.bib` | done, **verify the two flagged fields** |
| macro-F1 instead of switch-class F1 | `Metrics.calcMacroF1` is the model-selection and threshold criterion everywhere | `models/Metrics.java` | done |

---------------------------------------------------------------------------------------------------

## 3. Features: what is computed and why

All features are computed **per sentence** (so the same sentence has the same vector in any document) and
are fixed-size, whatever the sentence length. Position in the document is carried by the *pair* (left /
right neighbour) and by the retrieval model's *window* (section 6), never by variable-length vectors.

### 3.1 The groups

| group (`FeatureGroup`) | features | motivation |
|---|---|---|
| LENGTH | log word count, log character count | length is habit-dependent, and the log tames the long tail; also the strongest single signal in the first paper |
| FREQUENT_WORDS, FUNCTION_WORDS | share of very frequent words, share of function words, plus the function-word **profile** | function words are the classic topic-independent authorship marker (Mosteller and Wallace 1964; surveys: Stamatatos 2009, Koppel et al. 2009). They matter most on the *hard* level, where topic is fixed |
| PUNCTUATION, PUNCTUATION_MARKS | overall density, variety, and one rate per mark (comma, period, semicolon, colon, !, ?, quote, apostrophe, bracket, dash, ellipsis, repeated marks) | punctuation habit survives topic control and is very cheap; fan-fiction authors differ strongly in dash, ellipsis and quote style |
| ORTHOGRAPHY, LAYOUT | upper-case ratio, ALL-CAPS words, capitalised words, starts with capital / dialogue / conjunction, ends with terminal mark, non-ASCII, spacing anomalies, letter elongation | typing and formatting habits; **these need the original casing**, which is why `-Dcasing` is part of the ablation |
| WORD_LENGTH | mean, std and 6 length buckets | cheap proxy for vocabulary level |
| LEXICAL_DIVERSITY | type-token ratio, hapax ratio, Simpson's D, Yule's K, word-frequency class | richness measures; for a single 15 to 20 word sentence these are noisy, so expect them to matter little, which the ablation will confirm or refute |
| FLESCH_READING_EASE, GUNNING_FOG, FLESCH_KINCAID | the three readability scores | the first paper found them largely redundant with length; they are separate groups so the ablation can show it |
| TFIDF, EMOTION | mean tf-idf; NRC emotion score | content-leaning; kept only to test whether they help or only add topic. `emotion_score` is 0 when the NRC lexicon file is missing (it is not bundled) |
| CONTRACTIONS | contraction ratio | informal-register marker |
| POS_RATIOS | share of 14 coarse tags + tag entropy | syntax without lexical content |
| CHAR_NGRAM, POS_NGRAM | hashed n-gram **profiles** (below) | the bulk of the signal in most authorship work |

### 3.2 The profiles (fixed size, order of n, buckets)

A profile counts character n-grams (or POS-tag n-grams, or function words) into a **fixed number of hash
buckets** (feature hashing, Weinberger et al. 2009), then divides by the total, so the block is a
distribution that does not depend on sentence length. This is the answer to the first question of this
project ("fixed size, sentences of varying length"): *count into buckets, normalise, compare*. Variable
length never reaches the classifier.

Settings that are ablated, because the literature does not settle them for 15-word sentences:

* **n = 1..5, and combinations.** Character 3-grams are the usual default and 4 - 5-grams add some
  topic. For very short texts, lower orders are denser and less noisy. `-Dchar.n=1,2,3,4,5` makes one block per order.
* **Buckets 64 / 128 / 256 / 512 / 1024.** Few buckets: dense but collisions; many: sparse for a single sentence. A sentence of 100 characters has at most ~100 distinct trigrams, so more than 512 buckets is mostly empty.
* **Typed n-grams** (`-Dchar.typed=true`): only affix and punctuation n-grams (Sapkota et al. 2015 found these carry style and that content-word n-grams carry topic). Useful on the hard level.
* **Casing** (`-Dcasing=preserve|lower|hybrid`): preserve keeps capitals and ALL-CAPS as signal; lower-casing merges "The"/"the" and removes name noise; *hybrid* gives the scalars the original text and the profiles the lower-cased text. All three are reported separately, as requested.
* **POS n-grams** (`-Dpos.n`, `-Dpos.buckets`, `-Dpos.tagger=heuristic|corenlp`). The heuristic tagger is a rule/lexicon tagger written in the project (no external model, so it also runs inside the TIRA container). It is **not** as accurate as CoreNLP; `corenlp` uses the bridge to the Stanford tagger when its jar is available (untested here). The ablation row "no POS profile" shows how much the tagger matters at all.

### 3.3 Standardisation

Scalars are z-scored with the **training** mean/std and clipped at +-`z.clip` (default 5). This is the
Burrows' Delta idea (Burrows 2002): compare authors in units of how much a feature normally varies. It
also makes the logistic regression and the network well-conditioned. The scaler is fitted on training
sentences only and saved inside the model.

---------------------------------------------------------------------------------------------------

## 4. Representing a sentence pair (the "SentencePair / FeaturePair" question)

For sentences A and B with feature vectors a and b (z-scored scalars + profile blocks), the pair
vector fed to the classifier can be built in several ways. All are implemented in
`data/SentencePair.java` and selected with `-Dpair=`:

| `pair` | scalar part of the vector (u, v = z-scored scalars of sentence 1, 2) | width | idea |
|---|---|---|---|
| `absolute` (default) | \|u - v\| | S | symmetric, "how much do they differ"; the choice of the first paper |
| `signed` | u - v | S | keeps the direction (A longer than B is not B longer than A); not symmetric |
| `mean` | (u + v) / 2 | S | the *level* of the pair, cannot see a change on its own |
| `absolute_and_mean` | [\|u - v\|, (u + v)/2] | 2S | \|u - v\| is the same for (1, 2) and (101, 102); the mean says where on the scale the difference is |
| `relative` | \|u - v\| / (\|u\| + \|v\| + eps), on raw values | S | scale-free difference |
| `concat` | [u, v, \|u - v\|] | 3S | the classification vector of Sentence-BERT (Reimers and Gurevych 2019); gives the model the sentences too |
| `absolute_and_product` | [\|u - v\|, u * v] | 2S | difference + agreement; Reimers and Gurevych found u * v adds nothing on top of the concatenation, the ablation tests that for style features |

The hashed profiles (character n-grams, POS n-grams, function words) always enter through **profile measures**
unless the full pair is asked for (next paragraph), whatever `pair` is.

Additionally, the profile part of the pair can be **compressed** (default: 4 measures per profile block: cosine, Jensen-Shannon
distance, total variation, support Jaccard, which gives a few columns instead of 256) or **full**
(`-Dfull=true`: every bucket difference as a column). Compression is fast and removes noise; full lets a
non-linear model find *which* n-grams differ. The ablation reports both.

The first paper used |a - b| over 14 features plus one trigram cosine. That choice was never compared
with another representation, which is exactly what a reviewer asks for; the `pair` sweep fixes it.

---------------------------------------------------------------------------------------------------

## 5. Why exactly these three classifiers

The submission has three models (kept, as requested). They are written from scratch in Java, share the
same data and metric, and each answers a different question:

| model | motivation | known weakness | what the ablation checks |
|---|---|---|---|
| **Logistic regression** (batch gradient descent, L2, optional class weights) | a *linear* baseline; its weights are directly readable ("a longer sentence raises the switch probability by ..."), which is the interpretability story of the paper | cannot model interactions such as "a difference in length matters only when the punctuation also differs" | learning rate, iterations, L2, class weights |
| **Neural network** (MLP, Adam, early stopping on validation macro-F1) | the standard non-linear learner for dense features; shows whether interactions between features exist at all | needs more tuning, not directly interpretable | hidden sizes, learning rate, epochs, dropout / L2 as implemented |
| **Decision tree** (exact or histogram splits, leaf probabilities, class weights) | non-linear, **rule-readable** ("if \|Δ length\| > x and \|Δ commas\| > y then switch"), and the usual weak learner of the boosted systems that did well in PAN 2025 / 2026 | single trees overfit and are unstable; probabilities from leaves are coarse | depth, leaf size, impurity, bins, class weights |

Decisions common to all three:

* **Threshold tuned on validation for macro-F1** (not F1 of the change class, and not 0.5), since the classes are imbalanced (about 14 % changes in the first paper's partition, and a few % in the medium level whose "always 1" baseline is 0.042).
* **Class weights** are an ablation axis: the first paper showed they matter for the tree and not for the logistic regression once the threshold is tuned.
* **Honest protocol**: extractor / scaler fit on *train only*; first half of validation for every decision, second half held out as the test number.

The weak point to keep in mind: all three see *only the two sentences of the pair*. Winners use context
windows (datateam-tug: two sentences each side) and fine-tuned encoders. The retrieval model below is the
project's way to use context without leaving the interpretable feature world.

---------------------------------------------------------------------------------------------------

## 6. The retrieval model (new)

**Idea.** Do not pair sentences. Embed *each sentence on its own* from the same features, and learn an
embedding in which sentences of the same author are close and those of another author are far apart.
Then the same model does both:

* **retrieval**: "which sentences of this document are written like this one" (rank by cosine);
* **switch detection**: compare the mean embedding of the `w` sentences left of a position to the `w` on the right; a large distance = a change. `w` (1, 2, 3) is chosen on validation, so this is where *position/context* enters.

**Input** (`FeatureExtractor.embeddingInput`): z-scored scalars, plus the profiles passed through
`sqrt` and multiplied by `ret.scale` (default 4). The square root turns a distribution into the
Hellinger representation, so Euclidean geometry on it equals the Hellinger distance (Vedaldi and
Zisserman 2012 use the same map for additive kernels); it stops the few common n-grams dominating.

**Network and loss** (`EmbeddingNetwork`): MLP (hidden 128) -> `ret.dim` (32) -> L2 normalisation, trained
with the triplet loss max(0, m + d(a,p)^2 - d(a,n)^2) (Schroff et al. 2015, the siamese idea of Bromley
et al. 1993). Weak labels come for free from the data:

* positive = another sentence of the **same segment** (same author for certain);
* negative = a sentence of the **adjacent segment** (different author for certain);
* two segments that are *not* adjacent can have the same author (a two-author document alternates), the dataset does not say so, so they are **never used as labels**, neither in training nor in evaluation. This makes the retrieval numbers a *lower bound*.

Optional: hardest-of-k negative mining, negatives from other documents.

**Evaluation** (`RetrievalEvaluator`): P@1, Recall@5, MRR, MAP and triplet accuracy for retrieval; AUC and macro-F1 for switch detection, always next to two baselines: cosine of the raw feature vectors and random vectors. A learned model that does not beat raw features has learned nothing and the report will say so.

**Honest status.** On the stand-in data with unseen "authors" the learned embedding was **not** better
than the raw-feature cosine (it overfits the training authors; see section 8). Whether it helps on
the real data is exactly what `RetrievalMain` / the Retrieve page measure. Do not claim a benefit before that run.

### 6a. Author counts from the truth files, and the author-count estimate

**Training labels.** Each truth file gives the number of authors and the change flags. `data/SegmentAuthorship.java` turns them into
what is certain about two segments: neighbouring segments differ; if the number of authors equals the number of
segments, all segments differ; with two authors the segments alternate, so segments an even distance apart share
an author and an odd distance apart do not; in every other case the relation is unknown and the pair is not used. `data/TripletSampler.java` draws
anchor, positive and hardest-of-N negative under this rule (`-Dret.authorcounts=true`, the default). The positive can come from a non-adjacent segment
of the same author with probability `-Dret.farpositive` (0.2). `SamplerCheck` found no wrong label in the drawn triplets on the stand-in data. No accuracy
gain from this sampling has been shown yet; compare `-Dret.authorcounts=false` and `true` on the real data.

**Author-count estimate.** `analysis/DocumentSegmentation.java`: the change scores of the retrieval model are compared with the
decision threshold of the detected difficulty level; every change closes a segment and the number of segments is the estimate. Segments are not authors: a document with
12 segments and 4 authors (the sample truth file) is estimated at 12. The estimate is therefore an upper bound whenever an
author returns and a lower bound when a change is missed. `RetrievalMain` reports the exact-match rate and the mean absolute and signed error against the truth files.


---------------------------------------------------------------------------------------------------

## 7. The ablation study

Run on the real data (the same commands work on the stand-in):

```
scripts/ablation.sh all            # pair models: feature groups, model settings, pipeline grid
scripts/ablation.sh retrieval      # retrieval model sweep (RetrievalMain ablate)
```

| axis | values | model(s) | output |
|---|---|---|---|
| feature group, "ALL - group" and "ONLY group" | 19 groups | 3 pair models | `ablation_results.csv` |
| character n | 1, 2, 3, 4, 5, 1-5 | pair, retrieval | pipeline grid / `retrieval_ablation.csv` |
| character buckets | 64, 128, 256, 512, 1024 | pair, retrieval | same |
| casing | preserve, lower, hybrid | pair, retrieval | same |
| typed n-grams | on / off | pair | same |
| POS n, POS buckets, tagger | 1-3, 64/128/256, heuristic / corenlp | pair | same |
| pair representation | 7 representations x compressed / full | pair | same |
| classifier settings | tree depth / leaf / impurity / bins / weights; logistic lr / iterations / L2 / weights; NN hidden / lr / epochs | each | same |
| embedding settings | dim, margin, hidden, mining, cross-document negatives, profile scale, window | retrieval | `retrieval_ablation.csv` |

**Difficulty levels.** PAN scores easy, medium and hard as three separate datasets. The study therefore (1) reports every
table per level, (2) chooses the decision threshold of a level on *that level's* validation data (the earlier version used
one pooled threshold for all levels, which hurts the level with few changes; on stand-in data the level's own threshold gave
+0.02 to +0.04 macro-F1), (3) prints the all-0 / all-1 baselines of each level next to the results, (4) has a table of
the importance of each feature group per level (a group that helps on *hard* is a style signal, one that helps only on *easy*
may be a topic signal), and (5) can run on a single level with `-Ddifficulty=easy|medium|hard`. `Main` prints the own threshold
of each level; for the submission of one dataset pass it to the runner: `TiraRunner ... --threshold T`.

**The level is detected, not given.** In use (TIRA, the web app) nobody tells the system whether a document is easy, medium or
hard, so a threshold per level is only usable if the level can be found from the text. `models/DifficultyDetector` does that:
it is a 3-class softmax regression (written from scratch like the other models, class-balanced) on 16 document-level
features: how similar the content words of the sentences are to each other (hashed, function words excluded, because the
levels are defined by topic variety) and how much the style measures vary inside the document. It is trained on the
TRAIN split only. `models/LevelRouter` then picks the threshold of the detected level from `NAME.ser.levels`, which
`DifficultyMain calibrate` writes from the first half of each level's validation split (the second half stays untouched).
If the detector or the thresholds are missing, the router does **not** guess: it uses the pooled threshold and says why.
`DifficultyMain evaluate` reports three numbers per level on the held-out half: pooled threshold, threshold of the TRUE level
(the upper bound), and threshold of the DETECTED level, so the cost of detecting the level is measured, not assumed.
Measured accuracy of the detector: see section 8a (88.8% of the held-out validation documents).
`TiraRunner --difficulty auto` (default) decides per document,
`auto-dataset` decides once per dataset from the mean probability (more stable, because TIRA scores one level at a time),
`easy|medium|hard` forces a level and `none` uses the pooled threshold.

**First real-data run (author's own run of `Main` and a partial `Ablation`, before the detector existed; approximate values).**
Overall macro-F1 about 0.71. Per level: easy about 0.91 - 0.93, medium about 0.50 (the level of the all-0 baseline 0.489, so
no usable signal yet), hard about 0.45 with the pooled threshold and about 0.55 with its own threshold. So the pooled
threshold was hiding a small real signal on hard. In the partial ablation the importance of the feature groups was about 0
for every group, and one readability feature alone gave about 0.68 overall and about 0.90 on easy: on easy, almost any
document-level difference separates the authors (topic), which is why easy must be reported separately and why a single
overall number is misleading. 15 neural-network ablation runs ran out of memory (see the VM notes in the README). These
values come from a run that was not repeated with several seeds and are not final.

Reading rules for the paper (these are what make the ablation *support* the features rather than just list numbers):

1. Decide on **validation** columns; report **test** columns.
2. A difference below about **0.01 macro-F1 is within noise** for one seed; use `-Dseeds=42,123,456` (mean +- std are recorded) before saying a feature helps.
3. Report "ALL - group" (what is lost) *and* "ONLY group" (what it can do alone); together they separate redundant groups from useless ones.
4. Always show the per-difficulty columns: a feature that helps easy (topic) and not hard (style) is a topic feature.

---------------------------------------------------------------------------------------------------

## 8. What the code produced so far (STAND-IN data: a smoke test, NOT PAN results)

The PAN data are on Zenodo, which the build sandbox cannot reach. To exercise every code path the whole
study was run once on stand-in data (`scripts/make_pseudo_mawsa.py`: sentences from public-domain books,
"author" = book; the validation books are **not** in training). Settings were "quick": 20,000 training pairs,
**6,727 evaluation pairs, one seed**. The files are in `docs/standin_results/`. Stand-in text is not fan
fiction, "difficulty" is synthetic, and the macro-F1 values (~0.55 - 0.60) say nothing about PAN. What *can*
be learned from them is limited to: the code runs, the study produces the tables, and the points below.

**What the stand-in run does show about the study design** (not about authorship):

* With 6,727 evaluation pairs, one seed and quick training, differences below about **0.01 are noise**
  (the same setting moves by that much when training changes slightly). Most rows of the study fall inside that band.
  The final study needs the real data and `-Dseeds=42,123,456`.
* **Pair representation** was the only axis with a gap that was consistent over all three models:
  `absolute_and_mean` and `absolute_and_product` beat `absolute` by +0.02 to +0.03 macro-F1 for tree,
  logistic regression and network (e.g. network 0.5619 -> 0.5906 / 0.5882). The quick run covered only
  these two of the six alternatives. This is the first thing to confirm on PAN, and if it holds, `absolute` is the wrong default.
* **Full bucket differences** (674 columns) instead of the compressed measures helped the network (0.5619 -> 0.5794) and not the tree.
* **Character n-gram order, bucket count and typed n-grams** moved nothing beyond noise in the pair models, and the
  tree never used the character block at all (identical numbers for every n). The "only CHAR_NGRAM"
  row for the network is 0.4397, the same as "only EMOTION" - the all-0 prediction - so the network learned nothing from that block alone in the quick run.
  The stand-in books may simply not differ in character n-grams in a way four summary measures can show; do not generalise.
* **Casing:** lower-casing was never clearly better; hybrid was equal to preserve. Consistent with the idea that capital letters carry
  author habits (the LAYOUT and ORTHOGRAPHY groups were among the most useful for logistic regression and the network).
* **Feature groups** that came out useful for at least two of the three models: FUNCTION_WORDS, and LAYOUT/ORTHOGRAPHY (linear and network),
  POS_NGRAM (tree and logistic regression). Readability, lexical diversity, TF-IDF, emotion and contractions were negligible or negative for every model.
  `emotion_score` is 0 in this run (no NRC lexicon), so its row is empty by construction.
  POS_RATIOS removal *helped* the network by 0.0099, a candidate for dropping or for regularising the network more.
* **Model settings:** the tree is best at depth 5 on stand-in (depth 20: train 0.87 vs test 0.53, clearly overfitting); logistic regression keeps improving
  with more iterations (10,000: 0.5723 vs 0.5678 at 600), i.e. it was under-trained at the default; the network did slightly
  better with fewer layers (64-32), no dropout and no L2. Defaults are not changed on this evidence.

**Retrieval model, 33 variants** (`retrieval_ablation_STANDIN.csv`; unseen books; switch detection on the held-out half):

| | macro-F1 | AUC | retrieval MAP |
|---|---:|---:|---:|
| learned embedding, baseline | 0.596 | 0.664 | 0.345 |
| cosine of the raw features | 0.614 | 0.690 | 0.352 |
| best learned variant (scale = 2) | 0.613 | 0.690 | 0.359 |

No variant beat the raw-feature cosine in a way outside noise (the learned AUC ranged 0.59 - 0.69 against 0.62 - 0.70 for raw
features in the same runs). Window 5 was worse than 1 - 3 (AUC 0.59). So **there is no evidence yet that the learned embedding helps**; on authors it has not seen,
it overfits to the training authors. That is a finding to report honestly, and it may change on PAN where training has many more authors.

---------------------------------------------------------------------------------------------------

## 8a. Results on the REAL PAN 2026 data (author's run on a Google VM, Debian, Java 25, 4 cores, 31 GB)

All numbers: macro-F1 on the **second half of the validation split** (259,653 sentence pairs, 3,375 documents; nothing used
it for training, settings or thresholds), one run, one seed. Source logs: `train.log`, `difficulty.log`, `evaluate.log`.
Official all-0 baseline on this half: easy 0.413, medium 0.489, hard 0.441 (all 0.462). Thresholds are chosen on the first
validation half of each level.

**Models, one pooled threshold (what the first study did).**

| model | easy | medium | hard | all |
|---|---:|---:|---:|---:|
| decision tree | 0.914 | 0.501 | 0.444 | 0.712 |
| logistic regression | 0.914 | 0.508 | 0.449 | 0.710 |
| neural network | 0.923 | 0.505 | 0.446 | 0.711 |
| retrieval (extra) | 0.850 | 0.536 | 0.487 | 0.662 |

**Threshold of the true level / threshold of the DETECTED level (the level is not given) / one level voted for the whole level folder.**

| model | level | true level | detected | folder vote |
|---|---|---:|---:|---:|
| tree | easy / medium / hard | 0.935 / 0.503 / 0.553 | 0.887 / 0.503 / 0.552 | 0.935 / 0.503 / 0.553 |
| logistic | easy / medium / hard | 0.935 / 0.506 / 0.538 | 0.906 / 0.507 / 0.536 | 0.935 / 0.506 / 0.538 |
| neural | easy / medium / hard | 0.944 / 0.505 / 0.544 | 0.905 / 0.506 / 0.544 | 0.944 / 0.505 / 0.544 |
| retrieval | easy / medium / hard | 0.899 / 0.541 / 0.570 | 0.874 / 0.541 / 0.569 | 0.899 / 0.541 / 0.570 |

Macro-F1 over all pairs: pooled / true level / detected / folder vote = tree 0.712 / 0.719 / 0.707 / 0.719; logistic
0.710 / 0.729 / 0.719 / 0.729; neural 0.711 / 0.722 / 0.712 / 0.722; retrieval 0.662 / 0.722 / 0.711 / 0.722.

**Difficulty detector.** Accuracy 0.888 on the 3,375 held-out documents (first half: 0.882). Rows are the true level:
easy [883, 40, 202], medium [39, 1083, 3], hard [81, 14, 1030] (columns easy / medium / hard). So medium is separated from
hard almost perfectly, and the confusion is between **easy and hard**, which both have documents of similar length.

What this supports, and what it does not:

1. **Tuning the threshold per level matters most for hard.** Hard goes from the baseline level (0.44 - 0.49) to 0.54 - 0.57 with its own
   threshold. This is the same effect as in the first run and it is the result with the clearest support.
2. **Detecting the level costs little when it is done for the whole dataset.** The folder vote chose the right level for all
   three folders, and its scores equal the true-level scores. Per document, the detector costs about 0.010 macro-F1 against
   the true level (it mostly loses on easy: 202 easy documents get the hard threshold). For the three pair models the
   per-document detector is **not better than the pooled threshold overall** (-0.004 to +0.009, within noise); for the retrieval model it
   is clearly better (+0.049) because its pooled threshold was a poor fit for hard.
3. **Medium is still at the baseline level for the three pair models** (0.50 - 0.51 against 0.489). The retrieval model
   is the only one with a visible gain there (0.536 pooled, 0.541 with its level threshold, about +0.05).
4. **The three pair models are equal overall** (0.710 - 0.712). The difference between them is smaller than 0.01 and is within noise
   for one seed; do not rank them without `-Dseeds=42,123,456`.
5. **Easy is mostly topic.** Easy reaches 0.91 - 0.94 with simple features; this says little about style.
6. **For context** the best 2026 systems reach about 0.77 on medium and hard. These models are an interpretable baseline,
   well below that, and the paper must say so.

Caveat on the detector: the document features show that medium documents have about 3 times as many sentences as easy
or hard ones (mean of ln(1+sentences): easy 3.59, medium 4.80, hard 3.78), and the sentence count is one of the 16 inputs. Part
of the 0.888 therefore comes from a property of how this dataset was built. If another test set is built differently,
the detector must be retrained and re-evaluated; do not claim it detects "topic variety" in general.

## 9. References (all checked to exist; open the link before quoting page numbers)

Task and data
* Zangerle, Mayerl, Potthast, Stein. *Overview of the Multi-Author Writing Style Analysis Task at PAN 2026.* CLEF 2026 Working Notes, CEUR-WS Vol-4283. https://ceur-ws.org/Vol-4283/paper388.pdf
* Bevendorff et al. *Overview of PAN 2026 ...* arXiv:2602.09147. https://arxiv.org/abs/2602.09147
* Task page: https://pan.webis.de/clef26/pan26-web/style-change-detection.html
* Zangerle, Mayerl, Potthast, Stein. *Overview ... at PAN 2025.* CEUR-WS Vol-4038 (Reddit data, baselines 0.178 / 0.439). https://ceur-ws.org/Vol-4038/paper_279.pdf
* Voznyuk, Tammaa (team datateam-tug). *Contrastive Windowed Boundary Classification for Multi-Author Writing Style Analysis.* CEUR-WS Vol-4283 paper429. https://ceur-ws.org/Vol-4283/paper429.pdf

Style change detection
* Zlatkova et al. *An Ensemble-Rich Multi-Aspect Approach Towards Robust Style Change Detection.* PAN at CLEF 2018 notebook; and Kopev et al. *Recursive Style Breach Detection with Multifaceted Ensemble Learning.* arXiv:1906.06917 (TF-IDF + hand-made features, SVM / RF / AdaBoost / MLP / LightGBM; PAN 2018 winner).

Features
* Stamatatos. *A survey of modern authorship attribution methods.* JASIST 60(3), 538-556, 2009.
* Sapkota, Bethard, Montes, Solorio. *Not All Character N-grams Are Created Equal: A Study in Authorship Attribution.* NAACL-HLT 2015, 93-102. https://aclanthology.org/N15-1010/
* Koppel, Schler, Argamon. *Computational methods in authorship attribution.* JASIST 60(1), 9-26, 2009.
* Burrows. *"Delta": a measure of stylistic difference and a guide to likely authorship.* Literary and Linguistic Computing 17(3), 267-287, 2002.
* Mosteller, Wallace. *Inference and Disputed Authorship: The Federalist.* 1964.
* Weinberger et al. *Feature Hashing for Large Scale Multitask Learning.* ICML 2009. https://arxiv.org/abs/0902.2206
* Flesch 1948; Gunning 1952; Kincaid et al. 1975 (readability); Yule 1944 (K); Simpson 1949 (D).

Embeddings
* Reimers, Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP-IJCNLP 2019 (from the header comment in `SentencePair.java`; not re-opened this session).
* Bromley et al. *Signature verification using a "siamese" time delay neural network.* NIPS 1993.
* Schroff, Kalenichenko, Philbin. *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR 2015.
* Vedaldi, Zisserman. *Efficient Additive Kernels via Explicit Feature Maps.* IEEE TPAMI 34(3), 2012.

Items without a link above (Mosteller-Wallace, Bromley, Burrows, Koppel, Yule, Simpson) were cited from
memory of the standard references and not re-opened in this session; check the bibliographic details
before they go in the paper.

---------------------------------------------------------------------------------------------------

## 10. Mismatches between the submitted paper and the code (fix in the new paper)

1. **Data.** The submitted PDF called the data forum posts; the PAN 2026 data are **fan fiction**
   (the draft source in `scripts/manuscript/main.tex` already says so).
2. **Metric.** The PDF reports the F1 of the switch class. PAN ranks by **macro-F1**. Every table must be
   regenerated with macro-F1, and the official baselines (0.410 / 0.489 / 0.441 and 0.234 / 0.042 / 0.173) shown next to the results.
3. **Features.** The PDF describes 14 features + a trigram profile and a 16-feature pair vector. Not all
   of them were used by the code that produced the results, which a reviewer noticed. The new code has
   63 named scalar features and three profile types; the paper must describe what `FeatureConfig.signature()`
   of the submitted model says, nothing more.
4. **Classifiers.** Three from-scratch models, kept. The paper must describe the settings actually saved with
   the model (GUI: Results -> Model detail), not the defaults of an earlier version.
5. **TIRA.** The constant-1 output came from deployment, not from the model. The new runner fails if the model is missing; run `scripts/tira_selftest.sh` and compare with the validation numbers before submitting.
6. **Title and template.** "Team redblinkinglight at PAN 2026: <short description>" and the CEUR-WS (CEURART) template, not `llncs`.
7. **Results.** The paper's numbers are from a partition of the training data. The new paper must report the
   full PAN data, per difficulty, with the held-out test split, and must not mix in the stand-in numbers.
8. **Context.** The winners are ~0.77 on medium and hard. Say clearly that this work is an interpretable baseline.
