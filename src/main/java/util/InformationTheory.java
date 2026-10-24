package util;

/**
 * Logarithms used by the entropy and divergence measures.
 */
public final class InformationTheory {

    private InformationTheory() {
        // utility class, not meant to be instantiated
    }

    /**
     * Logarithm to the base 2.
     *
     * @param value a positive number
     * @return the base-2 logarithm
     */
    public static double log2(double value) {
        return Math.log(value) / Math.log(2.0);
    }
}
