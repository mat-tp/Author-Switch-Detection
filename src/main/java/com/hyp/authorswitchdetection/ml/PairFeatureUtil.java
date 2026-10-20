package com.hyp.authorswitchdetection.ml;

import features.FeatureExtractor;

import java.util.ArrayList;
import java.util.List;

/**
 * Turning raw sentences into the SAME feature vector a model was trained on.
 *
 * Everything is done by the FeatureExtractor that was saved together with the model (it carries the feature
 * settings, the z-score parameters and the TF-IDF table of the training data) - never by a new one,
 * otherwise the numbers would not line up. This class only calls it: the pair vector
 * (FeatureExtractor.pairVector) is the one code path the command line, the ablation and the web app share,
 * so there is nothing here that has to be kept in sync by hand.
 */
public final class PairFeatureUtil {

    private PairFeatureUtil() {
        // utility class, not meant to be instantiated
    }

    /** The per-sentence scalar features, in FeatureExtractor.getFeatures order. */
    public static final String[] SENTENCE_FEATURE_NAMES = FeatureExtractor.SCALAR_NAMES.clone();

    /** Pair feature vector of two sentences, ready for a loaded model. */
    public static double[] pairFeatures(FeatureExtractor extractor, String sentence1, String sentence2) {
        return extractor.pairVector(extractor.getFeatures(sentence1), extractor.getFeatures(sentence2));
    }

    /**
     * The pair vectors of all neighbouring sentences of a document: every sentence is extracted ONCE
     * (and not twice, as the sentence is in two pairs).
     */
    public static double[][] neighbourPairFeatures(FeatureExtractor extractor, List<String> sentences) {

        List<double[]> raw = new ArrayList<>();
        for (String sentence : sentences) {
            raw.add(extractor.getFeatures(sentence));
        }

        double[][] matrix = new double[Math.max(0, sentences.size() - 1)][];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = extractor.pairVector(raw.get(i), raw.get(i + 1));
        }
        return matrix;
    }

    /** The names of the pair vector columns of this extractor. */
    public static String[] pairFeatureNames(FeatureExtractor extractor) {
        return extractor.pairFeatureNames();
    }
}
