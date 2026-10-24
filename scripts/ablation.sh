#!/usr/bin/env bash
#
# Runs the automated ablation study (Linux and macOS; ablation.bat is the Windows version).
#
# The study measures which feature groups and settings matter:
#   - Feature-group importance:  Importance(G) = F1(all features) - F1(all features without G)
#   - Parameter ablation of the decision tree, the logistic regression and the neural network.
#
# Usage:
#   ./scripts/ablation.sh                features + parameters + pipeline, all three models
#   ./scripts/ablation.sh features       feature-group ablation only
#   ./scripts/ablation.sh params         parameter ablation only
#   ./scripts/ablation.sh pipeline       pipeline settings: character n-gram order 1..5, bucket sizes, the n x buckets
#                                        grid (-Dpipeline.grid=true), casing, POS n-grams, pair representation,
#                                        z-score clipping and the POS tagger
#   ./scripts/ablation.sh retrieval      the retrieval model: embedding size, margin, hard-negative mining,
#                                        n-gram orders, buckets, casing and window
#
# Options are passed through the JAVA_OPTS environment variable:
#   JAVA_OPTS="-Xmx24g" ./scripts/ablation.sh                         more memory
#   JAVA_OPTS="-Dmodels=tree" ./scripts/ablation.sh params            one model only
#   JAVA_OPTS="-Dseeds=42,123,456" ./scripts/ablation.sh              mean and standard deviation over three seeds
#   JAVA_OPTS="-Dquick=true -Dsubset=50000" ./scripts/ablation.sh     fast trial run
#   JAVA_OPTS="-Donly=false" ./scripts/ablation.sh features           skip the "only G" runs
#   JAVA_OPTS="-Dablation.nnEpochs=20" ./scripts/ablation.sh          shorter neural network runs
#   JAVA_OPTS="-Ddifficulty=hard" ./scripts/ablation.sh               the whole study on one level (easy | medium | hard);
#                                                                     by default all three levels are run and every table is split by level
#   JAVA_OPTS="-Ddataset=path/to/data/" ./scripts/ablation.sh         another dataset than dataset/mawsa26-pan-zenodo-DATA/
#
# The retrieval study writes to data/ablation/retrieval_ablation.csv; set RET_CSV to choose another file.
#
# Results are written while the study runs, so an interrupted run keeps what has finished:
#   data/ablation/ablation_results.csv   one row per run
#   data/ablation/ablation_report.md     importance ranking, parameter tables and the best settings
#   data/ablation/ablation_tables.tex    LaTeX tables for the manuscript

set -u

# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

MODE="${1:-all}"

case "$MODE" in
    all|features|params|pipeline|retrieval) ;;
    *) asd_fail "unknown mode '$MODE' (use: all | features | params | pipeline | retrieval)" ;;
esac

asd_configure_java
asd_require_jdk
asd_enter_project_root
asd_warn_if_dataset_missing

# cli.Ablation reuses the data pipeline of the other programs; javac follows the imports.
asd_compile cli/Ablation.java cli/RetrievalMain.java

echo
echo "Running the ablation study: $MODE"
echo

if [ "$MODE" = "retrieval" ]; then
    mkdir -p "$ASD_ROOT/data/ablation"
    java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -Dret.csv="${RET_CSV:-data/ablation/retrieval_ablation.csv}" \
        -cp "$ASD_OUT" cli.RetrievalMain ablate
else
    java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -cp "$ASD_OUT" cli.Ablation "$MODE"
fi
status=$?
[ "$status" -eq 0 ] || asd_fail "the ablation study stopped with exit code $status."

echo
echo "Finished. Results are in $ASD_ROOT/data/ablation/"
echo "  ablation_report.md, ablation_results.csv, ablation_tables.tex, retrieval_ablation.csv"
