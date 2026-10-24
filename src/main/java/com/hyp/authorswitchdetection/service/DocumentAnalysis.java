package com.hyp.authorswitchdetection.service;

import java.util.ArrayList;
import java.util.List;

/**
 * Everything the Analyse page shows about one document: its difficulty, the raw stylometric features of its
 * sentences, and the author segmentation derived from the boundary scores of the retrieval model.
 *
 * <p>The three parts are independent of each other. A part that cannot be computed (no trained model, no
 * difficulty detector) carries its own explanation in a {@code ...Problem} field instead of failing the page.</p>
 */
public class DocumentAnalysis {

    /** One sentence with the segment it belongs to and the score of the boundary after it. */
    public static class SentenceView {
        public int index;
        public String text;
        public int segment;
        /** Change score between this sentence and the next one; {@code null} for the last sentence. */
        public Double scoreToNext;
        /** Whether a change is predicted between this sentence and the next one. */
        public boolean splitAfter;
    }

    /** A run of sentences without a predicted author change. */
    public static class SegmentView {
        public int number;
        public int firstSentence;
        public int lastSentence;
        public int sentenceCount;
        public int wordCount;
    }

    /** Summary of one raw feature across all sentences of the document. */
    public static class FeatureSummary {
        public String name;
        public double mean;
        public double standardDeviation;
        public double minimum;
        public double maximum;
    }

    public int sentenceCount;
    public boolean truncatedFeatures;

    /** Difficulty detection: {@code null} level means it was not available, see {@link #difficultyProblem}. */
    public String difficultyLevel;
    public double[] difficultyProbabilities;
    public String[] difficultyLevels;
    public String difficultyProblem;

    /** Boundary detection and author estimate; empty when the retrieval model is not available. */
    public String segmentationProblem;
    public String modelDescription;
    public double threshold;
    public String thresholdSource;
    public int window;
    public int predictedChanges;
    public int estimatedAuthors;
    public double[] changeScores;
    public List<SentenceView> sentences = new ArrayList<>();
    public List<SegmentView> segments = new ArrayList<>();

    /** Raw features, computed with a corpus-free extractor so they do not depend on any trained model. */
    public List<String> featureNames = new ArrayList<>();
    public List<FeatureSummary> featureSummaries = new ArrayList<>();
    public List<double[]> featureRows = new ArrayList<>();

    public boolean hasSegmentation() {
        return segmentationProblem == null;
    }

    public boolean hasDifficulty() {
        return difficultyProblem == null;
    }
}
