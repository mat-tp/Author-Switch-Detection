#!/usr/bin/env bash
#
# Shared helpers of the project's shell scripts (build.sh, ablation.sh, tira_selftest.sh).
# This file is sourced, never run directly:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
#
# After sourcing, call the functions in this order:
#     asd_configure_java        find a JDK and put it first on the PATH
#     asd_require_jdk           stop with a clear message when javac or java is missing
#     asd_enter_project_root    change into the project root and define the path variables
#     asd_warn_if_dataset_missing
#     asd_compile <source file>...   compile command line programs into $ASD_OUT
#
# Variables defined for the calling script:
#     ASD_ROOT         absolute path of the project root
#     ASD_SRC          Java source root (the same folder Maven compiles)
#     ASD_OUT          folder for the compiled classes of the command line tools
#     ASD_MODELS_DIR   folder for the trained models
#     ASD_METRICS_FILE json file the web app reads for its Results page
#     ASD_DATASET_DIR  folder with the PAN 2026 MAWSA data
#     ASD_JAVA_OPTS    JAVA_OPTS of the caller, split into words (use ${ASD_JAVA_OPTS[@]+"${ASD_JAVA_OPTS[@]}"})

# The oldest JDK the command line tools are known to compile with.
ASD_MINIMUM_JDK=21

# Absolute path of the folder this file lives in (scripts/lib).
ASD_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prints an error message and ends the calling script with a non-zero exit code.
#   $1  message
asd_fail() {
    echo >&2
    echo "ERROR: $1" >&2
    exit 1
}

# Chooses the JDK: JAVA_HOME when it is set, otherwise the one that "java" on the PATH belongs to.
# The chosen JDK is placed first on the PATH so that javac and java come from the same installation.
asd_configure_java() {
    if [ -z "${JAVA_HOME:-}" ] && command -v java >/dev/null 2>&1; then
        JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
    fi
    if [ -n "${JAVA_HOME:-}" ]; then
        export JAVA_HOME
        export PATH="$JAVA_HOME/bin:$PATH"
    fi
    echo "JAVA_HOME: ${JAVA_HOME:-<not set, relying on PATH>}"
}

# Checks that javac and java exist, prints their versions and warns about a JDK that is older than
# ASD_MINIMUM_JDK. Ends the script when either program is missing.
asd_require_jdk() {
    command -v javac >/dev/null 2>&1 \
        || asd_fail "javac was not found. Install a JDK ($ASD_MINIMUM_JDK or newer) and either add it to the PATH or set JAVA_HOME."
    command -v java >/dev/null 2>&1 \
        || asd_fail "java was not found. Install a JDK ($ASD_MINIMUM_JDK or newer) and either add it to the PATH or set JAVA_HOME."

    javac -version
    java -version 2>&1 | head -n 1

    local major
    major="$(javac -version 2>&1 | sed -n 's/^javac \([0-9][0-9]*\).*/\1/p')"
    if [ -n "$major" ] && [ "$major" -lt "$ASD_MINIMUM_JDK" ]; then
        echo "WARNING: JDK $major found, JDK $ASD_MINIMUM_JDK or newer is expected (the web app needs 25, see pom.xml)."
    fi
}

# Changes into the project root (the parent folder of scripts/) and defines the path variables.
asd_enter_project_root() {
    cd "$ASD_LIB_DIR/../.." || asd_fail "could not move to the project directory."
    ASD_ROOT="$PWD"
    ASD_SRC="$ASD_ROOT/src/main/java"
    ASD_OUT="$ASD_ROOT/out"
    ASD_MODELS_DIR="$ASD_ROOT/data/models"
    ASD_METRICS_FILE="$ASD_ROOT/data/model-metrics.json"
    ASD_DATASET_DIR="$ASD_ROOT/dataset/mawsa26-pan-zenodo-DATA"

    # JAVA_OPTS holds JVM options as one string, for example "-Xmx16g -Dsubset=50000".
    # It is split into words once, here, so that every java call receives each option as its own argument.
    read -r -a ASD_JAVA_OPTS <<< "${JAVA_OPTS:-}"

    echo "Project directory: $ASD_ROOT"
}

# Warns when the PAN data is not where the programs expect it. The programs themselves would stop
# with a less helpful message from inside the dataset loader.
asd_warn_if_dataset_missing() {
    if [ ! -d "$ASD_DATASET_DIR" ]; then
        echo "WARNING: the dataset folder was not found at:"
        echo "  $ASD_DATASET_DIR"
        echo "  Training and the ablation study need it (see README.md, section 6)."
    fi
}

# Deletes the class files of an earlier compilation, so that no stale class survives a renamed or removed source file.
asd_clean_classes() {
    if [ -d "$ASD_OUT" ]; then
        find "$ASD_OUT" -type f -name '*.class' -delete
    else
        mkdir -p "$ASD_OUT"
    fi
}

# Compiles the given programs into $ASD_OUT. javac follows their imports through -sourcepath, so only the
# entry points are listed; the Spring Boot classes are never reached, hence no Maven dependency is needed.
#   $@  source files relative to src/main/java, for example cli/Main.java
asd_compile() {
    local sources=()
    local file
    for file in "$@"; do
        [ -f "$ASD_SRC/$file" ] || asd_fail "source file not found: $ASD_SRC/$file"
        sources+=("$ASD_SRC/$file")
    done
    mkdir -p "$ASD_OUT"
    echo "Compiling: $*"
    javac -encoding UTF-8 -sourcepath "$ASD_SRC" -d "$ASD_OUT" "${sources[@]}" \
        || asd_fail "compilation failed."
    echo "Compilation successful."
}
