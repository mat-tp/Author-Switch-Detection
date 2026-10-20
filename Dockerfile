# TIRA / PAN 2026 submission image (NOT TESTED in the sandbox this was written in: no Docker, no TIRA access).
#
# Before building, put next to this file:
#   data/models/<the trained model>.ser          trained with Main / RetrievalMain on the real PAN data
#   dataset/word-processing/...                  the two lexicons (frequent words, NRC emotion), the model
#                                                was trained with them, so they must be here too
# Build:  docker build -t asd-pan2026 .
# Local test, same command line TIRA uses:
#   docker run --rm -v $PWD/some-input:/in:ro -v $PWD/some-out:/out asd-pan2026 -i /in -o /out --model neural
FROM eclipse-temurin:21-jdk AS build
WORKDIR /app
COPY src/main/java src/main/java
RUN mkdir -p out && javac -encoding UTF-8 -sourcepath src/main/java -d out \
    src/main/java/TiraRunner.java src/main/java/SelfCheck.java

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /app/out out
COPY data/models data/models
COPY dataset/word-processing dataset/word-processing

# TiraRunner stops with an error (non-zero exit code) when the model is missing or out of date.
# It never writes a default prediction. The model can also be chosen with the MODEL environment variable.
ENTRYPOINT ["java", "-Xmx4g", "-cp", "out", "TiraRunner"]
