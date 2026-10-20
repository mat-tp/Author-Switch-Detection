#!/bin/bash
# Checks the TIRA runner end to end, WITHOUT Docker: compiles it, runs it on a small folder of problems with a
# trained model, and checks that
#   - one solution-problem-N.json exists per problem, with exactly (sentences - 1) values
#   - the predictions are not one constant
#   - with no model it stops with a non-zero exit code and writes nothing
#
# Usage: ./scripts/tira_selftest.sh [INPUT_DIR] [MODEL]
#   INPUT_DIR defaults to dataset/mawsa26-pan-zenodo-DATA/easy/validation (problem-N.txt + truth-problem-N.json)
#   MODEL     neural | logistic | tree | retrieval (default neural)
set -u
cd "$(dirname "$0")/.." || exit 1

INPUT="${1:-dataset/mawsa26-pan-zenodo-DATA/easy/validation}"
MODEL="${2:-neural}"
OUT="$PWD/out"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

javac -encoding UTF-8 -sourcepath src/main/java -d "$OUT" src/main/java/TiraRunner.java || exit 1

echo "--- 1. no model: must fail and write nothing"
java -Dmodels.dir="$WORK/no-models/" -cp "$OUT" TiraRunner -i "$INPUT" -o "$WORK/out-none"
if [ $? -eq 0 ] || [ -e "$WORK/out-none" ]; then echo "FAIL: ran without a model"; exit 1; fi
echo "ok"

echo "--- 2. with the $MODEL model"
java ${JAVA_OPTS:-} -cp "$OUT" TiraRunner -i "$INPUT" -o "$WORK/out" --model "$MODEL" || { echo "FAIL: runner failed"; exit 1; }

python3 - "$INPUT" "$WORK/out" <<'PY'
import json, glob, os, sys
inp, out = sys.argv[1], sys.argv[2]
n = ones = total = 0
for f in sorted(glob.glob(os.path.join(inp, "**", "problem-*.txt"), recursive=True)):
    rel = os.path.relpath(os.path.dirname(f), inp)
    pid = os.path.basename(f)[len("problem-"):-4]
    sol = os.path.join(out, rel, "solution-problem-%s.json" % pid)
    assert os.path.exists(sol), "missing " + sol
    sentences = [l for l in open(f, encoding="utf-8").read().splitlines() if l.strip()]
    changes = json.load(open(sol))["changes"]
    assert len(changes) == max(0, len(sentences) - 1), "wrong length in " + sol
    n += 1; total += len(changes); ones += sum(changes)
print("problems %d, pairs %d, predicted changes %d (%.1f%%)" % (n, total, ones, 100.0 * ones / max(1, total)))
assert n > 0, "no problems"
assert total < 50 or 0 < ones < total, "constant prediction"
print("OK")
PY
