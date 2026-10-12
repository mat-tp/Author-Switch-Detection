# Author-Switch-Detection

Detects whether the author changes between two consecutive sentences, based on writing style rather than topic.
A Spring Boot web app wraps a from-scratch ML pipeline (no ML libraries): log in, train models, inspect
sentence-level stylistic features, and run live predictions on text pairs. The same pipeline also runs from the
command line, including an automated ablation study.

**Research question:** can an author switch be detected from relatively stable writing habits instead of topic?
PAN 2026 makes this important: its hard split keeps sentences on the same topic, so systems must rely on style.

---

## Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run the web app](#run-the-web-app)
- [Train from the command line](#train-from-the-command-line)
- [Ablation study](#ablation-study)
- [Feature set](#feature-set)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Neural network details](#neural-network-details)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Models:** Logistic Regression, Decision Tree and Neural Network, all implemented from scratch.
  Random Forest is a placeholder (`models/RandomForest.java` is empty) and has no trained model yet.
- **Web interface:** login, model training panel, results page, feature inspection, dataset exploration
  and interactive prediction.
- **One shared pipeline:** the CLI and the web app use the same feature extraction and training steps.
- **Dynamic model loading:** the best configuration found during training is saved and picked up
  without restarting the app.
- **Automated ablation study:** feature-group and hyperparameter ablation with CSV, Markdown and LaTeX output.

---

## Prerequisites

- **Java:** JDK 21 or newer (the project targets 21). A full JDK is needed, not just a JRE, because the
  scripts call `javac`.
- **Maven:** 3.x, or use the included wrapper `./mvnw`.
- **Dataset and lexicons** (not stored in git, see [Setup](#setup)).

---

## Setup

Place the data under `dataset/`:

```
dataset/
├── mawsa26-pan-zenodo-DATA/                       # PAN MAWSA 2026 training / validation data
└── word-processing/
    ├── frequentWords/google-10000-english.txt
    └── NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
```

The two lexicon files feed the frequent-word and emotion features. If one is missing, the extractor prints
`File not found` and that feature becomes 0, which silently weakens the results, so check them first.

> **Models trained before a feature change must be retrained.** A saved model remembers how many features it
> was trained with. If the feature set has since changed, the app logs
> `model was trained with N scalar features but the code now has M - retrain the model`
> and that model is not loaded. See [Train from the command line](#train-from-the-command-line).

---

## Run the web app

```bash
./mvnw spring-boot:run
```

Or build and run the executable JAR:

```bash
./mvnw clean package -DskipTests
java -jar target/Author-Switch-Detection.jar
```

Open `http://localhost:8080`. A default admin account (`admin` / `admin123`) is created on first start.
**Change it** (`app.admin.*` in `application.properties`) before exposing the app anywhere.

---

## Train from the command line

`scripts/build.sh` (Linux/macOS) and `scripts/build.bat` (Windows) compile and train. Models are saved to
`data/models/` and metrics to `data/model-metrics.json`, the same places the web app reads from.

```bash
./scripts/build.sh              # all models
./scripts/build.sh tree         # decision tree only
./scripts/build.sh logistic     # logistic regression only
./scripts/build.sh neural       # neural network only
./scripts/build.sh clean        # delete out/ and exit
```

Useful options go through `JAVA_OPTS`:

```bash
JAVA_OPTS="-Xmx8g" ./scripts/build.sh all           # more memory
JAVA_OPTS="-Dsubset=50000" ./scripts/build.sh tree  # quick trial on a subset
```

Training from the web panel runs the same steps (`DatasetLoader` → `FeatureExtractor` → `train()`), tunes
hyperparameters on the validation split, and saves the best model to `data/models/`.

---

## Ablation study

`scripts/ablation.sh` runs the study and writes everything to `data/ablation/`. Results are written while the
study runs, so an interrupted run keeps what already finished.

- **Feature-group ablation:** each model is trained on all features, then once with each feature group removed
  (`Importance(G) = F1_all - F1_without_G`), and once with each group on its own. Groups are defined in
  `features/FeatureGroup.java`.
- **Parameter ablation:** one hyperparameter at a time (tree depth, min samples, impurity, class weights;
  logistic regression learning rate and iterations; neural network layers, L2, dropout, learning rate,
  batch size, clipping, patience), plus a decision-tree depth × min-leaf grid.
- Settings are ranked by **validation** F1. The test split is reported but never used to choose a setting.

| Output                 | Content                                             |
| ---------------------- | --------------------------------------------------- |
| `ablation_results.csv` | one row per run                                     |
| `ablation_report.md`   | importance ranking, parameter tables, best settings |
| `ablation_tables.tex`  | LaTeX tables for the manuscript                     |

### Running one model at a time

Yes. The `models` option takes `tree`, `logistic` and `neural`, alone or comma-separated:

```bash
JAVA_OPTS="-Dmodels=tree" ./scripts/ablation.sh                  # decision tree only
JAVA_OPTS="-Dmodels=logistic" ./scripts/ablation.sh              # logistic regression only
JAVA_OPTS="-Dmodels=neural" ./scripts/ablation.sh                # neural network only
JAVA_OPTS="-Dmodels=tree,logistic" ./scripts/ablation.sh         # any subset
```

Combine it with a mode to run only part of the study for that model:

```bash
JAVA_OPTS="-Dmodels=tree" ./scripts/ablation.sh features         # feature groups only
JAVA_OPTS="-Dmodels=neural" ./scripts/ablation.sh params         # hyperparameters only
```

### All ablation options

| Command / option                 | Effect                                                       |
| -------------------------------- | ------------------------------------------------------------ |
| `./scripts/ablation.sh`          | features + parameters, all three models                      |
| `./scripts/ablation.sh features` | feature-group ablation only                                  |
| `./scripts/ablation.sh params`   | parameter ablation only                                      |
| `-Dmodels=tree,logistic,neural`  | which models to run (default: all three)                     |
| `-Dseeds=42,123,456`             | repeat with several seeds, report mean ± std (default: `42`) |
| `-Donly=false`                   | skip the "only group G" runs (faster)                        |
| `-Dquick=true -Dsubset=50000`    | fast trial run on a subset                                   |
| `-Dablation.nnEpochs=20`         | shorter neural network runs                                  |
| `-Dablation.lrIterations=2000`   | fewer logistic regression iterations                         |
| `-Dablation.out=path/`           | output folder (default: `data/ablation/`)                    |
| `-Xmx24g`                        | more heap, for the full dataset                              |

Combine several in one `JAVA_OPTS`, for example:

```bash
JAVA_OPTS="-Xmx24g -Dmodels=neural -Dseeds=42,123,456 -Dablation.nnEpochs=20" ./scripts/ablation.sh params
```

Tip: do a quick trial run first (`-Dquick=true -Dsubset=50000`) to check the setup before a long run.

---

## Feature set

Per-sentence scalar features live in one list, `FeatureExtractor.SCALAR_NAMES` (14 today):

| Group             | Features                                                                            |
| ----------------- | ----------------------------------------------------------------------------------- |
| Length            | log word count, log character count                                                 |
| Vocabulary        | frequent-word ratio, mean TF-IDF, type-token ratio                                  |
| Punctuation       | punctuation density, punctuation variety                                            |
| Orthography       | digit ratio, uppercase ratio, average word length                                   |
| Emotion           | NRC emotion score                                                                   |
| Readability       | Flesch Reading Ease, Gunning Fog, Flesch-Kincaid Grade                              |
| Character n-grams | 256-bucket hashed character 3-gram profile (cosine similarity + bucket differences) |

For each sentence pair, the models see the absolute z-score difference of every scalar feature, the n-gram
cosine similarity, an overall distance, and optionally the 256 n-gram bucket differences. Training (CLI and
web), the ablation study, the Inspect/Explore pages and the decision-tree rule names all read the same list.
POS ratios are not implemented yet and are not part of the vector.

**Adding a feature:**

1. Append its name to `SCALAR_NAMES` and fill the slot in `FeatureExtractor.getFeatures`.
2. Add one line to `features/FeatureGroup.java`; the ablation then includes it automatically.
3. Retrain the models (old ones are rejected, see [Setup](#setup)).

Note that the three readability scores are strongly correlated, so removing just one in the ablation can look
unimportant because the other two compensate.

---

## Project structure

```
Author-Switch-Detection/
├── src/main/java/
│   ├── Main.java                        # CLI training pipeline
│   ├── Ablation.java, AblationReport.java   # automated ablation study
│   ├── com/hyp/authorswitchdetection/   # Spring Boot controllers, services, web config
│   ├── data/                            # dataset loaders, sentence pairs
│   ├── features/                        # feature extraction and feature groups
│   └── models/                          # classifiers and metrics
├── src/main/resources/
│   ├── templates/                       # Thymeleaf views
│   └── application.properties
├── scripts/                             # build.sh/.bat, ablation.sh, manuscript and slides
├── data/                                # users, trained models, metrics, ablation output
├── dataset/                             # training data and lexicons (not in git)
└── pom.xml
```

---

## Configuration

Main settings in `src/main/resources/application.properties`:

| Property                                    | Default                            | Purpose                                   |
| ------------------------------------------- | ---------------------------------- | ----------------------------------------- |
| `app.users.file`                            | `data/users.json`                  | user accounts                             |
| `app.model-metrics.file`                    | `data/model-metrics.json`          | metrics shown on the Results page         |
| `app.trained-models.dir`                    | `data/models`                      | where trained models are saved and loaded |
| `app.dataset.root`                          | `dataset/mawsa26-pan-zenodo-DATA/` | training data                             |
| `app.admin.username` / `app.admin.password` | `admin` / `admin123`               | default admin account                     |

A model in `data/models/` takes priority over the baseline bundled in `src/main/resources/ml-models/`.

---

## Neural network details

The custom network uses mini-batch gradient descent and supports:

- L2 weight regularization
- Dropout layers
- Class weighting
- Early stopping on validation loss

---

# References

1. PAN Dataset
