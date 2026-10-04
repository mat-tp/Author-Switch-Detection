package com.hyp.authorswitchdetection.service;

import data.DatasetLoader;
import data.SentencePair;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Reads the MAWSA-style dataset (same format hyProjectTP's Main.java trains
 * on: difficulty/split/problem-*.txt + matching truth-problem-*.json) off
 * disk and turns it into the per-difficulty numbers the Explore page shows.
 *
 * Uses DatasetLoader (from hyProjectTP) to parse the sentence pairs, so the
 * counts here always match what Main.java would actually train on.
 */
@Service
public class DatasetExploreService {

    private static final String[] DIFFICULTIES = { "easy", "medium", "hard" };
    private static final String TRAIN_SPLIT = "train";

    private final String datasetRoot;

    public DatasetExploreService(
            @Value("${app.dataset.root:dataset/mawsa26-pan-zenodo-DATA/}") String datasetRoot) {
        this.datasetRoot = datasetRoot;
    }

    public String getDatasetRoot() {
        return datasetRoot;
    }

    // true once at least one difficulty folder has a train split present on disk
    public boolean isConfigured() {

        for (String difficulty : DIFFICULTIES) {
            File splitDir = new File(datasetRoot, difficulty + File.separator + TRAIN_SPLIT);
            if (splitDir.isDirectory()) {
                return true;
            }
        }

        return false;
    }

    // stats for every difficulty that has data on disk (skips the rest)
    public List<DifficultyStats> computeStats() {

        List<DifficultyStats> stats = new ArrayList<>();

        for (String difficulty : DIFFICULTIES) {
            File splitDir = new File(datasetRoot, difficulty + File.separator + TRAIN_SPLIT);
            if (!splitDir.isDirectory()) {
                continue;
            }
            stats.add(computeStatsForDifficulty(difficulty, splitDir));
        }

        return stats;
    }

    private DifficultyStats computeStatsForDifficulty(String difficulty, File splitDir) {

        List<SentencePair> pairs = DatasetLoader.loadSplit(datasetRoot, TRAIN_SPLIT, difficulty);
        int nProblems = countProblemFiles(splitDir);

        int nSwitches = 0;
        SentencePair exampleSwitch = null;
        SentencePair exampleSameAuthor = null;

        for (SentencePair pair : pairs) {

            if (pair.getLabel() == 1) {
                nSwitches++;
                if (exampleSwitch == null) {
                    exampleSwitch = pair;
                }
            } else if (exampleSameAuthor == null) {
                exampleSameAuthor = pair;
            }
        }

        DifficultyStats result = new DifficultyStats();
        result.difficulty = difficulty;
        result.nProblems = nProblems;
        result.nPairs = pairs.size();
        // every problem with n sentences contributes n-1 pairs, so summing
        // (pairs + 1) per problem gives back the total sentence count
        result.nSentences = pairs.size() + nProblems;
        result.nSwitches = nSwitches;
        result.switchRatePercent = pairs.isEmpty() ? 0.0 : round(100.0 * nSwitches / pairs.size());
        result.exampleSwitchPair = toExample(exampleSwitch);
        result.exampleSameAuthorPair = toExample(exampleSameAuthor);

        return result;
    }

    private PairExample toExample(SentencePair pair) {

        if (pair == null) {
            return null;
        }

        PairExample example = new PairExample();
        example.sentenceA = pair.getSentence1();
        example.sentenceB = pair.getSentence2();
        // same real hyProjectTP scalar features Inspect shows, via SentenceFeatures
        example.featuresA = new SentenceFeatures(pair.getSentence1(), 0, 2).toNamedMap();
        example.featuresB = new SentenceFeatures(pair.getSentence2(), 1, 2).toNamedMap();

        return example;
    }

    // counting problem-*.txt files directly, same filter DatasetLoader itself uses
    private int countProblemFiles(File splitDir) {

        File[] files = splitDir.listFiles();
        if (files == null) {
            return 0;
        }

        int count = 0;
        for (File file : files) {
            String name = file.getName();
            if (file.isFile() && name.startsWith("problem-") && name.endsWith(".txt")) {
                count++;
            }
        }

        return count;
    }

    private static double round(double v) {
        return Math.round(v * 100.0) / 100.0;
    }

    public static class DifficultyStats {
        public String difficulty;
        public int nProblems;
        public int nSentences;
        public int nPairs;
        public int nSwitches;
        public double switchRatePercent;
        public PairExample exampleSwitchPair;
        public PairExample exampleSameAuthorPair;
    }

    public static class PairExample {
        public String sentenceA;
        public String sentenceB;
        public Map<String, Double> featuresA;
        public Map<String, Double> featuresB;
    }
}
