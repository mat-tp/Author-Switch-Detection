# Evaluating Interpretable Stylometric Features for Sentence-Level Author Switch Detection

**Author:** TP. Matlala (Tshephang Phuti-A-Nkutu Matlala), University of Johannesburg
**Shared task:** PAN 2026 Multi-Author Writing Style Analysis (CLEF 2026), team *redblinkinglight*

---

## 1. Problem description

A document is a list of sentences, and it may have been written by more than one author. For every two
neighbouring sentences the system must say whether the author **changes** between them (1) or stays the
same (0). Only the way the sentences are written can be used, not who wrote them.

The task comes in three difficulty levels, scored as three separate datasets:

| Level | What the sentences of one document look like |
|---|---|
| Easy | different topics, so a topic change often gives the author away |
| Medium | little topic variety |
| Hard | all sentences on the same topic, only style is left |

The score is the **macro-F1** over all sentence pairs (the mean of the F1 of "change" and "no change").
The data are English fan fiction.

## 2. Purpose of the project

The purpose is **not** to repeat earlier studies. It is to take stylometric features that earlier work found
useful and adapt them to this task: we collected a large set of features per sentence and we evaluate them.
The project does three things:

1. extracts many features from each sentence (below) and compares different ways to turn two sentences into one input;
2. trains three simple models written from scratch (and one extra retrieval model), so every step can be explained;
3. runs a large **ablation study** to see which features, settings and models matter for detecting style
   changes, **separately for the easy, medium and hard levels**.

## 3. Pipeline of the models

All models share the same first two steps (read sentences, compute features). They differ after that.

| Step | Logistic regression | Neural network | Decision tree | Retrieval model (extra) |
|---|---|---|---|---|
| 1. Input | two neighbouring sentences | two neighbouring sentences | two neighbouring sentences | **one** sentence at a time |
| 2. Features | 63 scalar features + 3 hashed profiles per sentence | same | same | same |
| 3. Representation | pair vector: difference of the two sentences (7 variants, default absolute difference) | same | same | the sentence's own vector (scalars standardised, profiles square-rooted) |
| 4. Learning | weights by gradient descent, L2, class weights | MLP, Adam, dropout, early stopping | splits on the best gain, leaf probabilities, class weights | MLP that outputs a 32-number embedding, triplet loss: same author close, different author far |
| 5. Decision | probability of a change | probability of a change | probability of a change | distance between the average embedding left and right of a position |
| 6. Threshold | tuned on validation (macro-F1), **per difficulty level**; the level is **detected from the text**, never given | same | same | same |
| 7. Output | 0/1 per sentence pair | same | same | same, plus "which sentences sound alike" |
| Why this model | readable weights | finds feature interactions | readable rules | uses context without pairing sentences |

Data protocol: features and standardisation are fitted on the training data only. The validation split of
PAN is cut in two halves by document: the first half chooses thresholds and settings, the second half is
the held-out test.

## 4. Features extracted

Each sentence gets **63 scalar features** (the table groups them) plus three **hashed profiles**: a fixed
number of buckets that count n-grams, so the vector size does not depend on the sentence length.

| Group | Count | What it is | Why |
|---|---:|---|---|
| Length | 2 | words, characters (log) | authors differ in how long their sentences are |
| Frequent words / rarity | 2 | share of very common words, word frequency class | habit of simple or rare words |
| Punctuation | 2 | punctuation density and variety | punctuation habits |
| Punctuation marks | 12 | rate of each mark (, . ; : ! ? quotes ' brackets dash ellipsis repeats) | very personal habits, topic-free |
| Orthography | 6 | digits, capitals, ALL-CAPS, non-ASCII, stretched letters | typing style |
| Layout | 5 | starts with capital / dialogue / conjunction, ends with a full stop, odd spacing | formatting habits |
| Word length | 8 | mean, spread, six length buckets | vocabulary level |
| TF-IDF, emotion | 2 | mean TF-IDF, NRC emotion score | content-related, tested to see if they help or only add topic |
| Lexical diversity | 4 | type-token ratio, hapax, Simpson, Yule's K | richness of vocabulary |
| Readability | 3 | Flesch reading ease, Gunning fog, Flesch-Kincaid | text complexity |
| Contractions | 1 | share of *didn't, it's, I'll* | formal or informal |
| Function words | 1 + profile | share and profile of *the, of, and, but ...* | classic topic-free authorship signal |
| POS tags | 15 | share of each part of speech and their entropy | sentence structure without the words |
| POS n-gram profile | hashed | counts of tag sequences | syntax habits |
| Character n-gram profile | hashed | counts of character sequences (n = 1 to 5, 64 to 1024 buckets) | the strongest classic signal |

The settings that change how these are computed (letter case, n, buckets, pair representation) are all
ablated. They are described in [docs/RESEARCH.md](docs/RESEARCH.md).

## 5. Project structure

```
Author-Switch-Detection/
├── README.md
├── pom.xml, mvnw, .mvn/              Maven (web app)
├── Dockerfile                        TIRA image
├── docs/
│   ├── RESEARCH.md                   why these features/models, the ablation plan, paper checklist
│   └── standin_results/              smoke-test outputs (NOT PAN results)
├── scripts/
│   ├── build.sh / build.bat          train the models, self-checks
│   ├── ablation.sh / ablation.bat    the ablation study
│   ├── tira_selftest.sh              checks the TIRA runner
│   ├── make_pseudo_mawsa.py          small stand-in data to test the code
│   └── manuscript/, slides/          paper and slides
├── src/main/java/
│   ├── Main.java                     train tree / logistic / neural
│   ├── Ablation.java, AblationReport.java   the ablation study and its report
│   ├── RetrievalMain.java            the retrieval model
│   ├── DifficultyMain.java           difficulty detector + threshold per level (train / calibrate / evaluate)
│   ├── TiraRunner.java               PAN command line (-i / -o)
│   ├── SelfCheck.java                39 checks
│   ├── data/                         dataset loading, sentence pairs, matrices
│   ├── features/                     feature extraction, groups, settings, POS tagger
│   ├── models/                       the models, metrics, saving
│   └── com/hyp/authorswitchdetection/    the Spring Boot web app
├── src/main/resources/               web pages, style, application.properties
├── data/models/                      trained models (created by training)
├── data/ablation/                    ablation results (created by the study)
└── dataset/                          the data (see below, not in git)
```

## 6. Files needed to train the models

```
dataset/
├── mawsa26-pan-zenodo-DATA/                 PAN 2026 MAWSA data (Zenodo)
│   ├── easy/   train/ and validation/       problem-N.txt   one sentence per line
│   ├── medium/ train/ and validation/       truth-problem-N.json   {"changes": [0,1,0,...]}
│   └── hard/   train/ and validation/
└── word-processing/
    ├── frequentWords/google-10000-english.txt
    └── NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
```

If a lexicon file is missing, the program prints `File not found` and that feature becomes 0, so check the
log. Java 21 (a full JDK) is needed. Maven is needed only for the web app.

## 7. How to run

### 7.1 Terminal training

```bash
./scripts/build.sh                 # decision tree, logistic regression, neural network
./scripts/build.sh neural          # one model: tree | logistic | neural
./scripts/build.sh retrieval       # the retrieval model
./scripts/build.sh check           # self-checks
```

**Difficulty detection (the level is never given at evaluation).** Train the detector once, then calibrate the
saved models. Calibrating needs **no retraining**: it only chooses the threshold of each level for the models you already have.

```bash
./scripts/build.sh difficulty      # detector -> data/models/difficulty.ser
./scripts/build.sh calibrate       # threshold per level of every saved model -> data/models/NAME.ser.levels
./scripts/build.sh evaluate        # pooled threshold vs true level vs DETECTED level (held-out half)
```

The prediction code (web app and TIRA) then detects the level of each document and uses that level's threshold. Without
these two files it falls back to the pooled threshold and prints why. The accuracy of the detector on the PAN data is
only known after `evaluate` has been run; medium against hard may be hard to tell apart.

Models are saved in `data/models/` and the scores in `data/model-metrics.json`. Options go in `JAVA_OPTS`:

```bash
JAVA_OPTS="-Xmx16g" ./scripts/build.sh                              # more memory
JAVA_OPTS="-Dsubset=50000" ./scripts/build.sh tree                  # quick trial
JAVA_OPTS="-Dchar.n=2,3,4 -Dchar.buckets=512 -Dcasing=hybrid -Dpair=absolute_and_mean" ./scripts/build.sh
JAVA_OPTS="-Ddifficulty=hard" ./scripts/build.sh                    # train and score one level only
```

The result is printed overall, per level, and per level with that level's own threshold.

**Ablation study** (results in `data/ablation/`: `ablation_report.md`, `ablation_results.csv`, `ablation_tables.tex`):

```bash
./scripts/ablation.sh                  # feature groups + model settings, all models, all levels
./scripts/ablation.sh pipeline         # n-gram order 1-5, buckets, casing, POS, pair representation
./scripts/ablation.sh retrieval        # the retrieval model
JAVA_OPTS="-Xmx24g -Dseeds=42,123,456" ./scripts/ablation.sh        # final run: 3 seeds
JAVA_OPTS="-Ddifficulty=medium" ./scripts/ablation.sh               # the whole study on one level
JAVA_OPTS="-Dquick=true -Dsubset=50000" ./scripts/ablation.sh       # trial run first
```

### 7.2 Maven / Spring Boot web app

```bash
./mvnw spring-boot:run             # or: mvn spring-boot:run
```

Open `http://localhost:8080` and log in. The first run downloads Maven and the libraries, so it needs
internet. Pages: Predict, Train, Results, Inspect, Retrieve, Ablation, Explore.

- No model is bundled. Train one first (Train page or section 7.1); until then Predict says so.
- A default account `admin` / `admin123` is created on first start. **Change it** in
  `application.properties` (`app.admin.username`, `app.admin.password`) before anyone else can reach the app.

### 7.3 TIRA

```bash
./scripts/tira_selftest.sh                                  # checks the output format, no Docker
docker build -t asd-pan2026 .
docker run --rm -v $PWD/in:/in:ro -v $PWD/out:/out asd-pan2026 -i /in -o /out --model neural --threshold 0.30
```

The runner stops with an error if the model is missing, so it can never answer with a constant. `--difficulty`
chooses how the threshold is found: `auto` (default, the level of every document is detected), `auto-dataset` (one level
for the whole input, from the mean probability), `easy|medium|hard` (forced), `none` (the pooled threshold). `--threshold T`
overrides everything. The detector (`difficulty.ser`) and `NAME.ser.levels` are in `data/models/`, which the Dockerfile copies.

### 7.4 Training on a VM (memory)

- Set the heap to about 75% of the RAM: `JAVA_OPTS="-Xmx24g"` on a 32 GB machine.
- The retrieval model is the largest. It trains on at most 150,000 sentences by default (`-Dret.maxsentences=...` to change it).
- Run the neural-network ablation as its own process (`JAVA_OPTS="-Dmodels=neural" ./scripts/ablation.sh`); in the first
  run on a small machine 15 of its runs ran out of memory.
- Use `-Dsubset=50000` for a trial, and `nohup` or `tmux` so a closed terminal does not stop a long run.

## 8. Conclusion

The purpose of the project is not to duplicate the previous studies, but to adopt them into the current
research. We collected a great number of features from each sentence and we want to evaluate them. We are
also conducting a large ablation of the models, to get an overview of how important the system and its
features are for detecting style changes, and how this differs across the difficulty levels of the dataset.

## 9. References

**Data**
1. Zangerle, E., Mayerl, M., Stein, B., Potthast, M. *PAN26 Multi-Author Writing Style Analysis* (dataset). Zenodo. https://doi.org/10.5281/zenodo.19068842
2. Zangerle, E., Mayerl, M., Potthast, M., Stein, B. *Overview of the Multi-Author Writing Style Analysis Task at PAN 2026.* CLEF 2026 Working Notes, CEUR-WS Vol-4283. https://ceur-ws.org/Vol-4283/paper388.pdf
3. Task page: https://pan.webis.de/clef26/pan26-web/style-change-detection.html

**Lexicons**
4. Mohammad, S., Turney, P. *Crowdsourcing a word-emotion association lexicon* (NRC Emotion Lexicon). Computational Intelligence 29(3), 2013.
5. google-10000-english word list (J. Kaufman, from Google's Trillion Word Corpus).

**Other**
6. Stamatatos, E. *A survey of modern authorship attribution methods.* JASIST 60(3), 2009.
7. Sapkota, U., Bethard, S., Montes, M., Solorio, T. *Not All Character N-grams Are Created Equal.* NAACL-HLT 2015.
8. Kopev, D. et al. *Recursive Style Breach Detection with Multifaceted Ensemble Learning.* arXiv:1906.06917.
9. Weinberger, K. et al. *Feature Hashing for Large Scale Multitask Learning.* ICML 2009.
10. Schroff, F., Kalenichenko, D., Philbin, J. *FaceNet.* CVPR 2015 (triplet loss).
11. Reimers, N., Gurevych, I. *Sentence-BERT.* EMNLP-IJCNLP 2019 (sentence-pair representations).
12. Flesch (1948), Gunning (1952), Kincaid et al. (1975): readability scores.

More references and the reasoning behind each choice are in [docs/RESEARCH.md](docs/RESEARCH.md).
