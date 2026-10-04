# Author-Switch-Detection

A Spring Boot web app that wraps hyProjectTP's author-switch-detection models
(Logistic Regression, Decision Tree, Random Forest, Neural Network) behind a
browser UI: sign in, train, inspect features, and get predictions on sentence
pairs.

## Requirements

- Java 25
- Maven (the bundled `mvnw` / `mvnw.cmd` wrapper works too, no local Maven
  install needed)

## Running the web app

```bash
./mvnw spring-boot:run
```

or, after packaging:

```bash
./mvnw clean package
java -jar target/Author-Switch-Detection.jar
```

The app starts on `http://localhost:8080`. Sign in, then use the Train page
to kick off feature learning + parameter learning from the browser — this
calls the same pipeline described below, just triggered over HTTP instead of
the command line.

## Project layout

```
Author-Switch-Detection/
├── src/main/java/
│   ├── com/hyp/authorswitchdetection/   # Spring Boot app: controllers, services, web glue
│   ├── data/                            # DatasetLoader, SentencePair (from hyProjectTP)
│   ├── features/                        # FeatureExtractor, TextUtils (from hyProjectTP)
│   └── models/                          # LogisticRegression, DecisionTree, RandomForest,
│                                         # NeuralNetwork, Metrics (from hyProjectTP)
├── src/main/resources/
│   ├── templates/                       # Thymeleaf pages
│   └── application.properties
├── dataset/                             # users.json + the MAWSA dataset root
└── pom.xml
```

`data`, `features`, and `models` are kept as their own top-level packages
(not folded into `com.hyp.authorswitchdetection`) so they stay a drop-in
match for hyProjectTP's source — a model trained via hyProjectTP's CLI and
one trained through this web app are running the exact same code.

## Feature learning and parameter learning

Both happen through the same pipeline, in two ways:

1. **CLI (hyProjectTP style):** compile and run hyProjectTP's own `Main.java`
   against the dataset directly, the same way you would in the standalone
   hyProjectTP project. This performs feature learning (fitting the scaler /
   TF-IDF / frequent-word stats over the training corpus) and parameter
   learning (fitting each model's weights) exactly as hyProjectTP does.

2. **Web (this app):** the Train page calls `RealTrainer`, which runs the
   identical `DatasetLoader` → `FeatureExtractor` → model `train(...)` pipeline
   in-process, tries a small grid of feature-set/hyperparameter combinations
   per model, and keeps whichever configuration scores best on the
   validation split. The winning model is serialized to
   `app.trained-models.dir` (`data/models` by default) and is picked up
   immediately by `MlModelLoader` for prediction — no restart needed.

Either path produces a model compatible with the other, since both share the
same `data`/`features`/`models` code.

## Lexicon / frequent-words files

`FeatureExtractor` (in `features/`) reads two large reference files for the
mean TF-IDF and mean emotion-score features:

```
dataset/word-processing/frequentWords/google-10000-english.txt
dataset/word-processing/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
```

These paths are relative to the app's working directory at runtime — for
this Spring Boot app, that's the `Author-Switch-Detection/` project root
(where `pom.xml` lives), the same as for hyProjectTP's own `Main.java`. Place
the files there yourself; they're not bundled in this repo (they're large
and excluded via `.gitignore`) and `FeatureExtractor` degrades gracefully
(features read as 0) if they're missing, rather than failing to start.

## Users dataset

Login users are stored in `dataset/users.json` (configurable via the
`app.users-file` property in `application.properties`), alongside the MAWSA
dataset root (`app.dataset.root`). This keeps everything under `dataset/`
rather than mixing runtime user data into a generic `data/` folder — `data/`
is reserved for build artifacts (trained model `.ser` files).

## Neural network

The `NeuralNetwork` in `models/` trains with plain mini-batch gradient
descent — no Adam or other adaptive optimizer. It still supports L2
regularization, dropout, class weighting, and early stopping via a
validation split, same as before.
