# Author-Switch-Detection

A Spring Boot web application for author-switch detection. It provides a browser interface to authenticate users, execute model training workflows, inspect sentence-level stylistic features, and run real-time predictions on text pairs.

---

## Features

- **Classification Models:** Logistic Regression, Decision Tree, Random Forest, and Neural Network.
- **Web Interface:** User login, model training control panel, feature visualization, and interactive inference.
- **Shared Pipeline:** Consistent feature extraction and training execution across both CLI and web entry points.
- **Dynamic Model Loading:** Automatically serializes and loads the best-performing hyperparameter configuration without requiring a application restart.

---

## Prerequisites

- **Java:** JDK 25 (or JDK 21 compatibility)
- **Build Tool:** Maven 3.x (or the included Maven Wrapper `./mvnw`)

---

## Quick Start

### Run via Maven Wrapper

```bash
./mvnw spring-boot:run
```

## Build and Run Executable JAR

```bash
./mvnw clean package -DskipTests
java -jar target/Author-Switch-Detection.jar
```

Access the web interface at `http://localhost:8080`.

## Project Structure

```
Author-Switch-Detection/
├── src/main/java/
│   ├── com/hyp/authorswitchdetection/   # Spring Boot controllers, services, and web configuration
│   ├── data/                            # Dataset loaders and sentence pair models
│   ├── features/                        # Stylistic and lexical feature extraction
│   └── models/                          # Machine learning classifiers and metrics evaluation
├── src/main/resources/
│   ├── templates/                       # Thymeleaf views
│   └── application.properties
├── dataset/                             # User data and dataset root
└── pom.xml
```

## Feature Learning & Training PipelineCLI Execution: Run

- Main.java directly against the raw dataset to perform feature scaling, TF-IDF calculation, and model parameter optimization.

- Web Training: Triggering training from the web panel executes the same in-process pipeline (DatasetLoader $\rightarrow$ FeatureExtractor $\rightarrow$ train()), runs hyperparameter tuning across validation splits, and persists the top-scoring model to data/models/.

## External Resources & Configuration

Lexicon Files
The feature extraction pipeline utilizes external lexical files for stylistic density calculation:

dataset/word-processing/frequentWords/google-10000-english.txt

dataset/word-processing/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

## Storage Directories

Users Data: dataset/users.json (configured via app.users-file)

Serialized Models: data/models/ (configured via app.trained-models.dir)

## Neural Network Details

The custom neural network implementation uses mini-batch gradient descent and supports:

L2 Weight Regularization

Dropout layers

Class weighting

Early stopping based on validation loss
EOF

## Ablation Study (automated)

`scripts/ablation.sh` runs the whole study and writes the results to `data/ablation/`.

- **Feature-group ablation:** every model is trained on all features, then once with each feature group
  removed (`Importance(G) = F1_all - F1_without_G`), and once with each group on its own. Groups are defined
  in `features/FeatureGroup.java`; a new feature group needs ONE line there and is then included automatically.
- **Parameter ablation:** one hyperparameter at a time (decision tree depth / min samples / impurity /
  class weights, logistic regression learning rate / iterations, neural network layers / L2 / dropout /
  learning rate / batch size / clipping / patience) plus a decision-tree depth x min-leaf grid.
- Parameters are ranked by validation F1; the test split is reported but never used to pick a setting.
- Output: `ablation_results.csv` (one row per run), `ablation_report.md`, `ablation_tables.tex`.

```bash
./scripts/ablation.sh                                         # everything
JAVA_OPTS="-Dseeds=42,123,456 -Xmx24g" ./scripts/ablation.sh  # mean ± std over 3 seeds
JAVA_OPTS="-Dquick=true -Dsubset=50000" ./scripts/ablation.sh # fast trial
```
