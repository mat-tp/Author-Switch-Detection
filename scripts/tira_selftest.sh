#!/usr/bin/env bash
#
# Tests the TIRA runner from start to finish without Docker.
#
# It compiles cli.TiraRunner, runs it twice and checks the results:
#   1. Without a model the runner must exit with an error and write no output at all.
#   2. With a trained model it must write one solution-problem-N.json per problem, each holding exactly
#      (number of sentences - 1) values, and the predictions must not all be the same value.
#
# Usage: ./scripts/tira_selftest.sh [INPUT_DIR] [MODEL]
#   INPUT_DIR  folder of problem-N.txt files; defaults to dataset/mawsa26-pan-zenodo-DATA/easy/validation
#   MODEL      neural | logistic | tree | retrieval; defaults to neural
#
# JVM options are passed through the JAVA_OPTS environment variable. A trained model must exist
# in data/models/ (see build.sh); the output is written to a temporary folder that is removed afterwards.

set -u

# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

INPUT="${1:-dataset/mawsa26-pan-zenodo-DATA/easy/validation}"
MODEL="${2:-neural}"

asd_configure_java
asd_require_jdk
asd_enter_project_root
asd_compile cli/TiraRunner.java

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo
echo "Check 1: without a model the runner must fail and write nothing"
java -Dmodels.dir="$WORK/no-models/" -cp "$ASD_OUT" cli.TiraRunner -i "$INPUT" -o "$WORK/out-none"
if [ $? -eq 0 ] || [ -e "$WORK/out-none" ]; then
    asd_fail "the runner produced output without a model."
fi
echo "ok"

echo
echo "Check 2: with the $MODEL model"
java ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"} -cp "$ASD_OUT" cli.TiraRunner -i "$INPUT" -o "$WORK/out" --model "$MODEL" \
    || asd_fail "the runner failed."

# The output is verified against the input folder: one file per problem, the right length, not a constant.
python3 - "$INPUT" "$WORK/out" <<'PYTHON'
import glob
import json
import os
import sys

input_dir, output_dir = sys.argv[1], sys.argv[2]
problems = pairs = predicted_changes = 0

for problem_file in sorted(glob.glob(os.path.join(input_dir, "**", "problem-*.txt"), recursive=True)):
    relative_folder = os.path.relpath(os.path.dirname(problem_file), input_dir)
    problem_id = os.path.basename(problem_file)[len("problem-"):-len(".txt")]
    solution_file = os.path.join(output_dir, relative_folder, "solution-problem-%s.json" % problem_id)
    assert os.path.exists(solution_file), "missing " + solution_file

    with open(problem_file, encoding="utf-8") as handle:
        sentences = [line for line in handle.read().splitlines() if line.strip()]
    with open(solution_file, encoding="utf-8") as handle:
        changes = json.load(handle)["changes"]

    assert len(changes) == max(0, len(sentences) - 1), "wrong number of values in " + solution_file
    problems += 1
    pairs += len(changes)
    predicted_changes += sum(changes)

share = 100.0 * predicted_changes / max(1, pairs)
print("problems %d, pairs %d, predicted changes %d (%.1f%%)" % (problems, pairs, predicted_changes, share))
assert problems > 0, "no problems were found in the input folder"
assert pairs < 50 or 0 < predicted_changes < pairs, "the predictions are a constant"
print("OK")
PYTHON
