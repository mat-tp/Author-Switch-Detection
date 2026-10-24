package util;

/**
 * Four ways of comparing two non-negative profiles (hashed n-gram counts turned into shares):
 * cosine similarity, Jensen-Shannon distance, total variation distance and Jaccard overlap.
 *
 * <p>They are computed in one pass over the profiles, because the pair matrix of the whole training set
 * is built from them.</p>
 */
public final class DistributionDistances {

    /** How many numbers {@link #compare} writes. */
    public static final int MEASURE_COUNT = 4;

    private DistributionDistances() {
        // utility class, not meant to be instantiated
    }

    /**
     * Compares a block of two profiles and writes four numbers: cosine similarity, Jensen-Shannon distance
     * (base 2, so between 0 and 1), total variation distance and Jaccard overlap of the non-empty buckets.
     * When one of the two profiles is empty the numbers are 0, 1, 1, 0 (nothing in common).
     *
     * @param first    the first profile vector
     * @param second   the second profile vector
     * @param offset   index of the first bucket of the block
     * @param size     number of buckets in the block
     * @param out      the array that receives the four measures
     * @param outIndex index in {@code out} of the first measure
     */
    public static void compare(double[] first, double[] second, int offset, int size, double[] out, int outIndex) {

        double dot = 0.0;
        double firstNormSquared = 0.0;
        double secondNormSquared = 0.0;
        double totalVariation = 0.0;
        double divergence = 0.0;
        int inBoth = 0;
        int inEither = 0;
        double firstSum = 0.0;
        double secondSum = 0.0;

        for (int bucket = offset; bucket < offset + size; bucket++) {

            double p = first[bucket];
            double q = second[bucket];

            dot += p * q;
            firstNormSquared += p * p;
            secondNormSquared += q * q;
            firstSum += p;
            secondSum += q;
            totalVariation += Math.abs(p - q);

            double midpoint = (p + q) / 2.0;
            if (p > 0.0) {
                divergence += 0.5 * p * InformationTheory.log2(p / midpoint);
            }
            if (q > 0.0) {
                divergence += 0.5 * q * InformationTheory.log2(q / midpoint);
            }

            if (p > 0.0 || q > 0.0) {
                inEither++;
                if (p > 0.0 && q > 0.0) {
                    inBoth++;
                }
            }
        }

        if (firstSum == 0.0 || secondSum == 0.0) {
            out[outIndex] = 0.0;
            out[outIndex + 1] = 1.0;
            out[outIndex + 2] = 1.0;
            out[outIndex + 3] = 0.0;
            return;
        }

        out[outIndex] = dot / (Math.sqrt(firstNormSquared) * Math.sqrt(secondNormSquared));
        out[outIndex + 1] = Math.sqrt(Math.max(0.0, Math.min(1.0, divergence)));
        out[outIndex + 2] = Math.min(1.0, totalVariation / 2.0);
        out[outIndex + 3] = inEither == 0 ? 0.0 : (double) inBoth / inEither;
    }
}
