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
        public double[] raw;           // the 9 scalar features as extracted
        public double[] zScore;        // the same, standardised with the TRAINING mean / std (what the model uses)
        public int nGramBucketsUsed;   // how many of the 256 hashed trigram buckets are non-zero
        public double[] toNext;        // pair-feature vector to the next sentence, null for the last sentence
    }

    public String modelName;
    public int nSentences;
    public int nPairs;
    public String[] sentenceFeatureNames;
    public String[] pairFeatureNames;
    public List<SentenceView> sentences = new ArrayList<>();
}
