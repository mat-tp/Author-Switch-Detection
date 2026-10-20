package com.hyp.authorswitchdetection.service;

import java.util.ArrayList;
import java.util.List;

/**
 * Everything the Inspect page shows: for each sentence the features the model
 * extracts, and between neighbouring sentences the pair-feature vector the
 * model is given. Values are shown as computed, nothing is scaled for display.
 */
public class InspectResult {

    public static class SentenceView {
        public int index;              // 1-based
        public String text;
        public double[] raw;           // the scalar features as extracted
        public double[] zScore;        // the same, standardised with the TRAINING mean / std (what the model uses)
        public int nGramBucketsUsed;   // how many of the hashed profile buckets (character / POS n-grams) are non-zero
        public double[] toNext;        // pair-feature vector to the next sentence, null for the last sentence
    }

    public String modelName;
    public String note;            // e.g. "no trained model, z-scores are relative to this text only", null = nothing to say
    public String featureConfig;   // FeatureConfig.signature() of the extractor that computed these values
    public int nSentences;
    public int nPairs;
    public String featureDescription;  // FeatureExtractor.describe(): what the vectors consist of
    public int sentenceVectorLength;   // scalar features + profile buckets of one sentence
    public int profileBuckets;         // the hashed / function-word buckets of one sentence
    public String[] sentenceFeatureNames;
    public String[] pairFeatureNames;
    public List<SentenceView> sentences = new ArrayList<>();
}
