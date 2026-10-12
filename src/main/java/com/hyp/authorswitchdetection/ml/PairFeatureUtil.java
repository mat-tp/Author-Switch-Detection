package com.hyp.authorswitchdetection.ml;

import data.SentencePair;
import features.FeatureExtractor;

/*
    TODO: (1) Keep this in sync manually with Main.java's getPairFeatures/makePairFeatures
    if the standalone project's feature pipeline ever changes.
*/

/**
 * Turning a pair of raw sentences into the SAME feature vector a given model was trained on:
 * full (COMPRESSED_PAIR_COUNT + 256 features) or compressed (COMPRESSED_PAIR_COUNT features), since
 * RealTrainer's grid search can pick either one per model.
 *
 * This mirrors, step by step, what Main.getPairFeatures do:
 * 1. get the scalar (+ 256 n-gram bucket, full only) features of each sentence
 * 2. standardize the scalar part (z-score, fitted on the training data)
 * 3. build the pair vector: absolute scalar differences + n-gram cosine +
 * overall distance (+ absolute n-gram bucket differences, full only)
 * 4. full only: standardize the n-gram bucket differences too
 *
 * The FeatureExtractor given here must be the one loaded together with the
 * model (it already has the fitted z-score / TF-IDF / n-gram parameters from
 * training baked in), never a new one, otherwise the numbers won't line up.
 */
public final class PairFeatureUtil {

    private PairFeatureUtil() {
        // utility class, not meant to be instantiated
    }

    /** The per-sentence scalar features, in FeatureExtractor.getFeatures order. */
    public static final String[] SENTENCE_FEATURE_NAMES = FeatureExtractor.SCALAR_NAMES.clone();

    /**
     * The compressed pair features, in SentencePair.getPairFeatures order:
     * one absolute z-score difference per scalar feature, the n-gram cosine
     * similarity, and the overall (RMS) z-score distance.
     */
    public static final String[] PAIR_FEATURE_NAMES = buildPairFeatureNames();

    private static String[] buildPairFeatureNames() {
        int s = FeatureExtractor.SCALAR_COUNT;
        String[] names = new String[s + 2];
        for (int i = 0; i < s; i++) {
            names[i] = "diff_" + FeatureExtractor.SCALAR_NAMES[i];
        }
        names[s] = "ngram_cosine_similarity";
        names[s + 1] = "overall_distance";
        return names;
    }

    /** Name of pair feature i; the full vector continues with the n-gram bucket differences. */
    public static String pairFeatureName(int i) {
        return i < PAIR_FEATURE_NAMES.length ? PAIR_FEATURE_NAMES[i]
                : "diff_ngram_bucket_" + (i - PAIR_FEATURE_NAMES.length);
    }

    /**
     * Pairwise feature vector for two sentences, ready to be fed straight into
     * a loaded model's predict / predictProbabilities.
     *
     * @param extractor    the (already fitted) extractor loaded alongside the model
     * @param sentence1    first sentence
     * @param sentence2    second sentence
     * @param fullFeatures true = all features (scalars + 256 n-gram buckets), false = compressed (only possible for a model
     *                     RealTrainer's grid search picked "compressed" for)
     * @return the standardized pair feature vector
     */
    public static double[] pairFeatures(FeatureExtractor extractor, String sentence1, String sentence2,
            boolean fullFeatures) {

        // z-scores of both sentences, using the training parameters
        double[] features1 = extractor.standardize(extractor.getFeatures(sentence1));
        double[] features2 = extractor.standardize(extractor.getFeatures(sentence2));

        // label is unknown here, -1 is only a placeholder (same trick as Main.predictNewPair)
        SentencePair pair = new SentencePair(sentence1, sentence2, -1);

        if (!fullFeatures) {
            return pair.getPairFeatures(features1, features2);
        }

        double[] pairFeatures = pair.getFullPairFeatures(features1, features2);

        // same n-gram difference z-score parameters as the training data
        return extractor.standardizeNGramDifferences(pairFeatures);
    }
}
