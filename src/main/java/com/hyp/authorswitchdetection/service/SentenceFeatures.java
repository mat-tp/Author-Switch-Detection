package com.hyp.authorswitchdetection.service;

import features.FeatureExtractor;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Display wrapper around one sentence's real hyProjectTP feature vector.
 * Every value here comes straight out of FeatureExtractor.getFeatures(...),
 * the same call the trained models are built on — nothing is reimplemented
 * or invented on this side. sentencePosition is the only field that isn't
 * one of hyProjectTP's 9 scalar features; it's just bookkeeping for pair
 * ordering on the Inspect/Explore pages.
 */
public class SentenceFeatures {

    // shared, corpus-less extractor: getFeatures(...) doesn't need a training
    // corpus to run, only fitScaler/TF-IDF-over-a-corpus does, and we're not
    // standardizing or training here, just reading the raw 9 scalars for
    // display. Lazily built once and reused for every sentence.
    private static volatile FeatureExtractor sharedExtractor;

    private static FeatureExtractor extractor() {
        FeatureExtractor extractor = sharedExtractor;
        if (extractor == null) {
            synchronized (SentenceFeatures.class) {
                extractor = sharedExtractor;
                if (extractor == null) {
                    extractor = new FeatureExtractor(null);
                    extractor.loadFrequentWords();
                    sharedExtractor = extractor;
                }
            }
        }
        return extractor;
    }

    public final String rawText;

    // hyProjectTP's 9 scalar features, in FeatureExtractor.getFeatures order
    public final double logWordCount;
    public final double logCharCount;
    public final double frequentWordRatio;
    public final double punctuationDensity;
    public final double digitRatio;
    public final double uppercaseRatio;
    public final double avgWordLength;
    public final double meanTfIdf;
    public final double meanEmotionScore;

    // not a feature, just the sentence's position in the document (0..1)
    public final double sentencePosition;

    public SentenceFeatures(String sentence, int index, int totalSentences) {
        this.rawText = sentence;

        double[] scalars = extractor().getFeatures(sentence);

        this.logWordCount = round(scalars[0]);
        this.logCharCount = round(scalars[1]);
        this.frequentWordRatio = round(scalars[2]);
        this.punctuationDensity = round(scalars[3]);
        this.digitRatio = round(scalars[4]);
        this.uppercaseRatio = round(scalars[5]);
        this.avgWordLength = round(scalars[6]);
        this.meanTfIdf = round(scalars[7]);
        this.meanEmotionScore = round(scalars[8]);

        this.sentencePosition = totalSentences <= 1 ? 0.0 : round(index / (double) (totalSentences - 1));
    }

    private static double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }

    /** Flat vector, used both for display and Euclidean style-distance. */
    public double[] toVector() {
        return new double[]{
                logWordCount, logCharCount, frequentWordRatio, punctuationDensity,
                digitRatio, uppercaseRatio, avgWordLength, meanTfIdf, meanEmotionScore,
                sentencePosition
        };
    }

    public Map<String, Double> toNamedMap() {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("log_word_count", logWordCount);
        m.put("log_char_count", logCharCount);
        m.put("frequent_word_ratio", frequentWordRatio);
        m.put("punctuation_density", punctuationDensity);
        m.put("digit_ratio", digitRatio);
        m.put("uppercase_ratio", uppercaseRatio);
        m.put("avg_word_length", avgWordLength);
        m.put("mean_tfidf", meanTfIdf);
        m.put("mean_emotion_score", meanEmotionScore);
        m.put("sentence_position", sentencePosition);
        return m;
    }
}
