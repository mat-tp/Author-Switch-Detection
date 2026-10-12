package com.hyp.authorswitchdetection.service;

import features.FeatureExtractor;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Display wrapper around one sentence's real hyProjectTP feature vector.
 * Every value here comes straight out of FeatureExtractor.getFeatures(...),
 * the same call the trained models are built on — nothing is reimplemented
 * or invented on this side. sentencePosition is the only field that isn't
 * one of the scalar features (FeatureExtractor.SCALAR_NAMES); it's just bookkeeping for pair
 * ordering on the Inspect/Explore pages.
 */
public class SentenceFeatures {

    // shared, corpus-less extractor: getFeatures(...) doesn't need a training
    // corpus to run, only fitScaler/TF-IDF-over-a-corpus does, and we're not
    // standardizing or training here, just reading the raw scalars for
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

    // every scalar feature, name -> rounded value, in FeatureExtractor.SCALAR_NAMES order
    private final Map<String, Double> scalars = new LinkedHashMap<>();

    // not a feature, just the sentence's position in the document (0..1)
    public final double sentencePosition;

    public SentenceFeatures(String sentence, int index, int totalSentences) {
        this.rawText = sentence;

        double[] values = extractor().getFeatures(sentence);
        for (int i = 0; i < FeatureExtractor.SCALAR_COUNT; i++) {
            scalars.put(FeatureExtractor.SCALAR_NAMES[i], round(values[i]));
        }

        this.sentencePosition = totalSentences <= 1 ? 0.0 : round(index / (double) (totalSentences - 1));
    }

    private static double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }

    /** Flat vector (scalar features, then the sentence position), used for display. */
    public double[] toVector() {
        double[] vector = new double[scalars.size() + 1];
        int i = 0;
        for (double v : scalars.values()) {
            vector[i++] = v;
        }
        vector[i] = sentencePosition;
        return vector;
    }

    public Map<String, Double> toNamedMap() {
        Map<String, Double> m = new LinkedHashMap<>(scalars);
        m.put("sentence_position", sentencePosition);
        return m;
    }
}
