#!/bin/bash

clear

# Tshephang Matlala
# HYP TRAINING ML MODELS
# Trains and compares all the models (Decision Tree, Logistic Regression,
# Neural Network) using ONLY javac/java — no Maven, no internet access and
# no IDE needed. The Spring Boot web app (mvnw) is a separate, optional
# thing; this script is for training/testing the models straight from the
# command line.
#
# Usage:
#   ./scripts/build.sh              runs all the models (train + validation + test)
#   ./scripts/build.sh tree         runs only one: tree | logistic | neural
#   ./scripts/build.sh clean        deletes out/ and exits (nothing is run)
#
# Version for Linux / macOS systems

# JAVA CONFIGURATION

# Use JAVA_HOME if it has already been configured.
# Otherwise, try to find Java automatically.
if [ -z "$JAVA_HOME" ]; then
    if command -v java >/dev/null 2>&1; then
        JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(which java)")")")"
    fi
fi

if [ -n "$JAVA_HOME" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

echo
echo "        HYP - Project"
echo
echo "JAVA_HOME:"
echo "${JAVA_HOME:-<not set, relying on PATH>}"
echo

# WHICH MODEL(S)
# No argument = all the models, or give one: tree | logistic | neural
MODEL="${1:-all}"

if [ "$MODEL" = "clean" ]; then
    cd "$(dirname "$0")/.." || exit 1
    echo "Removing $PWD/out ..."
    rm -rf out
    echo "Done."
    exit 0
fi

case "$MODEL" in
    all|tree|logistic|neural) ;;
    *)
        echo
        echo "ERROR: Unknown model '$MODEL' (use: all | tree | logistic | neural | clean)"
        exit 1
        ;;
esac

# CHECK JAVA
echo "~~~ Checking Java ~~~"

if ! command -v javac >/dev/null 2>&1; then
    echo
    echo "ERROR: javac could not be found."
    echo "Install a JDK (17+) and either add it to PATH or set JAVA_HOME, e.g.:"
    echo "  export JAVA_HOME=/path/to/jdk"
    exit 1
fi

if ! command -v java >/dev/null 2>&1; then
    echo
    echo "ERROR: java could not be found."
    exit 1
fi

javac -version

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Could not check javac version."
    exit 1
fi

java -version

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Could not check java version."
    exit 1
fi

echo
echo "Java is ready."
echo

# MOVE TO PROJECT ROOT (this script lives in <project root>/scripts)
cd "$(dirname "$0")/.." || {
    echo
    echo "ERROR: Could not move to project directory."
    exit 1
}

echo "Project directory:"
pwd
echo

# PROJECT PATHS
# Main.java (plus the data / features / models packages it uses) lives
# under the normal Maven source folder, so this script and a Maven build
# compile from exactly the same source of truth.
SRC="$PWD/src/main/java"
OUT="$PWD/out"
MODELS_DIR="$PWD/data/models"
METRICS_FILE="$PWD/data/model-metrics.json"
DATASET_DIR="$PWD/dataset/mawsa26-pan-zenodo-DATA"

# CLEAN
echo "              > Cleaning previously compiled classes"
echo

if [ -d "$OUT" ]; then
    find "$OUT" -type f -name "*.class" -delete
else
    mkdir -p "$OUT"
fi

mkdir -p "$MODELS_DIR"

echo "Cleaning complete."
echo

# DATASET CHECK (a friendly warning instead of a wall of "directory does not
# exist" messages from inside DatasetLoader)
if [ ! -d "$DATASET_DIR" ]; then
    echo "WARNING: Dataset folder not found at:"
    echo "  $DATASET_DIR"
    echo "Training will fail immediately with 'No training data was loaded.'"
    echo "Place the MAWSA dataset there first (see README.md)."
    echo
fi

# COMPILE
# Only Main.java is passed to javac; javac follows its imports (data.*,
# features.*, models.*) automatically via -sourcepath, without touching
# the Spring Boot (com.hyp.*) classes, so no Spring/Maven dependencies are
# needed on the classpath.
echo "          > Compiling Java Source Code"
echo

javac \
    -sourcepath "$SRC" \
    -d "$OUT" \
    "$SRC/Main.java"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Compilation failed."
    exit 1
fi

echo
echo "Compilation successful."
echo

# RUN
# Main trains on ALL 3 difficulty splits combined (same as RealTrainer, used by
# the Spring web app's Train page), with the 11 compressed features (see
# USE_FULL_FEATURES in Main.java). It tunes the decision threshold on one half
# of the validation split, and reports final numbers (accuracy, precision,
# recall, F1, balanced accuracy) on the OTHER half (the held-out "test" half),
# which the model never saw during training or thresholding. The trained model
# is saved under data/models/, and a row is appended to data/model-metrics.json
# - the SAME files the Spring Boot web app (./mvnw spring-boot:run) reads for
# prediction and for the Results page, so this run shows up there too.
#
# Extra JVM options can be given with JAVA_OPTS, for example:
#   JAVA_OPTS="-Xmx8g" ./scripts/build.sh all          more heap (Main prints
#                                                      how much it wants)
#   JAVA_OPTS="-Dsubset=50000" ./scripts/build.sh tree quick trial run
echo "          > Running ML Model(s): $MODEL"
echo

java $JAVA_OPTS -cp "$OUT" Main "$MODEL"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Program execution failed."
    exit 1
fi

echo
echo "        > ML Training Finished"
echo "        > Trained model(s) saved under: $MODELS_DIR"
echo "        > Metrics appended to:          $METRICS_FILE"
echo "        > (open the web app's Results page to see them)"
echo

exit 0
