#!/usr/bin/env bash
#
# Trains and evaluates the models of the project (Linux and macOS; build.bat is the Windows version).
#
# Usage:
#   ./scripts/build.sh              train the decision tree, logistic regression and neural network
#   ./scripts/build.sh tree         train one pair model: tree | logistic | neural
#   ./scripts/build.sh retrieval    train the retrieval (embedding) model
#   ./scripts/build.sh difficulty   train the difficulty detector (data/models/difficulty.ser)
#   ./scripts/build.sh calibrate    choose the threshold of every difficulty level for every saved model
#                                   (NAME.ser.levels); needs no retraining
#   ./scripts/build.sh evaluate     compare pooled, true-level and detected-level thresholds on the held-out half
#   ./scripts/build.sh check        run the self-checks (gradients, metrics, features, saving and loading)
#   ./scripts/build.sh clean        delete the compiled classes in out/ and stop
#
# What a training run does (cli.Main, cli.RetrievalMain):
#   - It trains on all three difficulty levels together, the same way the web app's Train page does.
#   - The feature settings come from features/FeatureConfig.java and can be changed with -D options,
#     for example -Dchar.n=1,2,3 -Dchar.buckets=512 -Dcasing=lower.
#   - The decision threshold is tuned on one half of the PAN validation split. The final numbers
#     (accuracy, precision, recall, F1, balanced accuracy, macro-F1) are measured on the other half,
#     which the model never saw during training or threshold selection.
#   - The model is saved under data/models/ and a row is appended to data/model-metrics.json.
#     These are the files the web app (./mvnw spring-boot:run) reads, so the run also shows up on its Results page.
#
# JVM options are passed through the JAVA_OPTS environment variable:
#   JAVA_OPTS="-Xmx16g" ./scripts/build.sh                  more memory
#   JAVA_OPTS="-Dsubset=50000" ./scripts/build.sh tree      quick trial run on a subset of the training pairs
#   JAVA_OPTS="-Ddifficulty=hard" ./scripts/build.sh        train and score one difficulty level only

set -u

# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

TARGET="${1:-all}"

case "$TARGET" in
    all|tree|logistic|neural|retrieval|difficulty|calibrate|evaluate|check|clean) ;;
    *) asd_fail "unknown target '$TARGET' (use: all | tree | logistic | neural | retrieval | difficulty | calibrate | evaluate | check | clean)" ;;
esac

# "clean" needs neither Java nor a compilation.
if [ "$TARGET" = "clean" ]; then
    asd_enter_project_root
    echo "Removing $ASD_OUT"
    rm -rf "$ASD_OUT"
    echo "Done."
    exit 0
fi

asd_configure_java
asd_require_jdk
asd_enter_project_root
asd_warn_if_dataset_missing

mkdir -p "$ASD_MODELS_DIR"
asd_clean_classes
asd_compile cli/Main.java cli/RetrievalMain.java cli/DifficultyMain.java cli/SelfCheck.java cli/TiraRunner.java

echo
echo "Running: $TARGET"
echo

# Each target is one entry point of the cli package; the pair models share cli.Main and receive their name.
case "$TARGET" in
    retrieval)  java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -cp "$ASD_OUT" cli.RetrievalMain ;;
    difficulty) java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -cp "$ASD_OUT" cli.DifficultyMain train ;;
    calibrate)  java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -cp "$ASD_OUT" cli.DifficultyMain calibrate ;;
    evaluate)   java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -cp "$ASD_OUT" cli.DifficultyMain evaluate ;;
    check)      java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -cp "$ASD_OUT" cli.SelfCheck ;;
    *)          java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -cp "$ASD_OUT" cli.Main "$TARGET" ;;
esac
status=$?
[ "$status" -eq 0 ] || asd_fail "the program stopped with exit code $status."

echo
echo "Finished: $TARGET"
case "$TARGET" in
    all|tree|logistic|neural|retrieval|difficulty|calibrate)
        echo "Trained models and thresholds are under: $ASD_MODELS_DIR"
        echo "Metrics are appended to:                 $ASD_METRICS_FILE (shown on the web app's Results page)"
        ;;
esac
