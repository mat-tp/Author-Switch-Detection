#!/bin/bash

clear

# HYP - AUTOMATED ABLATION STUDY
# Feature-group ablation (Importance(G) = F1_all - F1_without_G) and parameter
# ablation for the decision tree, logistic regression and neural network.
#
# Usage:
#   ./scripts/ablation.sh                 features + parameters + pipeline, all 3 models
#   ./scripts/ablation.sh features        feature-group ablation only
#   ./scripts/ablation.sh params          parameter ablation only
#   ./scripts/ablation.sh pipeline        pipeline settings: character n-gram order n = 1..5, bucket sizes,
#                                         n x buckets grid (-Dpipeline.grid=true), casing, POS n-grams,
#                                         pair representation, z-score clipping, tagger
#   ./scripts/ablation.sh retrieval       the retrieval model: embedding size, margin, hard-negative mining,
#                                         n-gram orders, buckets, casing, window (writes retrieval_ablation.csv)
#
# Options go in through JAVA_OPTS (-D...), for example:
#   JAVA_OPTS="-Xmx24g" ./scripts/ablation.sh
#   JAVA_OPTS="-Dmodels=tree" ./scripts/ablation.sh params          only the tree
#   JAVA_OPTS="-Dseeds=42,123,456" ./scripts/ablation.sh            mean +- std over 3 seeds
#   JAVA_OPTS="-Dquick=true -Dsubset=50000" ./scripts/ablation.sh   fast trial run
#   JAVA_OPTS="-Donly=false" ./scripts/ablation.sh features         skip the "only G" runs
#   JAVA_OPTS="-Dablation.nnEpochs=20" ./scripts/ablation.sh        shorter neural network runs
#   JAVA_OPTS="-Ddifficulty=hard" ./scripts/ablation.sh             the whole study on ONE PAN dataset (easy | medium | hard)
#                                                                   (default: all three levels, every table is split by level)
#
# The dataset is dataset/mawsa26-pan-zenodo-DATA/ unless -Ddataset=... is given (JAVA_OPTS).
#
# Results (written while running, so an interrupted run keeps what finished):
#   data/ablation/ablation_results.csv    one row per run
#   data/ablation/ablation_report.md      importance ranking, parameter tables, best settings
#   data/ablation/ablation_tables.tex     LaTeX tables for the manuscript

MODE="${1:-all}"

case "$MODE" in
    all|features|params|pipeline|retrieval) ;;
    *)
        echo "ERROR: Unknown mode '$MODE' (use: all | features | params | pipeline | retrieval)"
        exit 1
        ;;
esac

if [ -z "$JAVA_HOME" ]; then
    if command -v java >/dev/null 2>&1; then
        JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(which java)")")")"
    fi
fi
if [ -n "$JAVA_HOME" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

if ! command -v javac >/dev/null 2>&1; then
    echo "ERROR: javac could not be found (install a JDK 17+ or set JAVA_HOME)."
    exit 1
fi

cd "$(dirname "$0")/.." || exit 1

SRC="$PWD/src/main/java"
OUT="$PWD/out"
DATASET_DIR="$PWD/dataset/mawsa26-pan-zenodo-DATA"

mkdir -p "$OUT"

if [ ! -d "$DATASET_DIR" ]; then
    echo "WARNING: Dataset folder not found at: $DATASET_DIR"
fi

echo "          > Compiling"
# Ablation.java reuses Main's data pipeline; javac follows the imports through -sourcepath
javac -encoding UTF-8 -sourcepath "$SRC" -d "$OUT" "$SRC/Ablation.java" "$SRC/RetrievalMain.java"

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed."
    exit 1
fi

echo "          > Running the ablation study: $MODE"
echo

if [ "$MODE" = "retrieval" ]; then
    mkdir -p data/ablation
    java $JAVA_OPTS -Dret.csv="${RET_CSV:-data/ablation/retrieval_ablation.csv}" -cp "$OUT" RetrievalMain ablate
else
    java $JAVA_OPTS -cp "$OUT" Ablation "$MODE"
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Ablation study failed."
    exit 1
fi

echo
echo "        > Results: $PWD/data/ablation/  (ablation_report.md, ablation_results.csv, retrieval_ablation.csv)"
exit 0
