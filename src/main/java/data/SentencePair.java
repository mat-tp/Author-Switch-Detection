package data;

import features.FeatureExtractor;

/*
    TODO: (1) For representation of PAIRS
    // This will be experimented
    1. just pure difference
    2. absolute difference: provides the same difference even for underlying different vectors (Problematic)
    3. Mean difference: (f1 + f2) / 2, can provide a better difference than abs difference.
    4. 1 + 2 ; PairRepresentation = [|f1-f2|, (f1 + f2) /2 ]
    5. Assuming that none of my features are negative:

        Representation R = |f1 - f2| / (f1 + f2 + epsilon)
 */
public class SentencePair {

    public enum Representation {
        ABSOLUTE, SIGNED, MEAN, ABSOLUTE_AND_MEAN, RELATIVE
    }

    public static final double EPSILON = 1e-8;

    private String sentence1;
    private String sentence2;
    private int label;

    /**
     * Constructor for a pair of consecutive sentences.
     *
     * @param sentence1 first sentence
     * @param sentence2 second sentence
     * @param label     ground-truth change label
     */
    public SentencePair(String sentence1, String sentence2, int label) {

        this.sentence1 = sentence1;
        this.sentence2 = sentence2;
        this.label = label;
    }

    /**
     * Returns the first sentence.
     *
     * @return first sentence
     */
    public String getSentence1() {
        return sentence1;
    }

    /**
     * Returns the second sentence.
     *
     * @return second sentence
     */
    public String getSentence2() {
        return sentence2;
    }

    /**
     * Returns the ground-truth label.
     *
     * @return label
     */
    public int getLabel() {
        return label;
    }

    // public double[] getPairFeatures(double[] features1, double[] features2) {

    // return getPairFeatures(features1, features2, Representation.ABSOLUTE);
    // }

    /**
     * Computing pairwise features for neighbouring sentences.
     *
     * @param features1 features of the 1st sentence
     * @param features2 features of the 2nd sentence
     * @return array of features for the pair
     */
    public double[] getPairFeatures(double[] features1, double[] features2) {

        if (features1.length != features2.length) {
            throw new IllegalArgumentException("features1 and features2 must have same length");
        }

        int scalarCount = FeatureExtractor.SCALAR_COUNT;

        if (features1.length <= scalarCount) {
            throw new IllegalArgumentException("features must come from FeatureExtractor.getFeatures");
        }

        double[] pairFeatures = new double[scalarCount + 2];
        double squaredDistance = 0.0;

        for (int i = 0; i < scalarCount; i++) {
            // calculate the absolute difference
            double difference = features1[i] - features2[i];
            pairFeatures[i] = Math.abs(difference);
            squaredDistance += difference * difference;
        }

        // also adding the cosine similarity between the N-Gram portion of the features
        pairFeatures[scalarCount] = FeatureExtractor.calcNGramCosine(features1, features2);
        // and the standardized gaussian distance that follow the normal dist
        pairFeatures[scalarCount + 1] = Math.sqrt(squaredDistance / scalarCount);

        return pairFeatures;
    }

    /**
     * Computing the pairwise features for neighbouring sentences WITHOUT
     * compressing the N-Gram portion.
     *
     * The first (scalarCount + 2) features are the same as in getPairFeatures:
     * [0..8] absolute z-score differences, [9] n-gram cosine similarity,
     * [10] overall z-score distance
     * Followed by the absolute difference of every hashed character 3-gram bucket:
     * [11..266] (256 buckets)
     *
     * @param features1 features of the 1st sentence
     * @param features2 features of the 2nd sentence
     * @return array of features for the pair (all the features, not compressed)
     */
    public double[] getFullPairFeatures(double[] features1, double[] features2) {

        // the compressed features come first (also checks the lengths)
        double[] compressedFeatures = getPairFeatures(features1, features2);

        int scalarCount = FeatureExtractor.SCALAR_COUNT;
        int nGramCount = features1.length - scalarCount;

        double[] pairFeatures = new double[compressedFeatures.length + nGramCount];
        System.arraycopy(compressedFeatures, 0, pairFeatures, 0, compressedFeatures.length);

        // absolute difference of each n-gram bucket
        for (int i = 0; i < nGramCount; i++) {
            pairFeatures[compressedFeatures.length + i] = Math
                    .abs(features1[scalarCount + i] - features2[scalarCount + i]);
        }

        return pairFeatures;
    }
}