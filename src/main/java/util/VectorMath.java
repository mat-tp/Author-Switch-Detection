package util;

/**
 * Basic vector arithmetic shared by the embedding model, the evaluators and the feature extractor.
 */
public final class VectorMath {

    /** Added to a norm before dividing, so that an all-zero vector does not produce NaN. */
    public static final double NORM_EPSILON = 1e-12;

    private VectorMath() {
        // utility class, not meant to be instantiated
    }

    /**
     * Dot product.
     *
     * @param first  the first vector
     * @param second the second vector, at least as long as the first
     * @return the sum of the products of the components
     */
    public static double dot(double[] first, double[] second) {
        double sum = 0.0;
        for (int index = 0; index < first.length; index++) {
            sum += first[index] * second[index];
        }
        return sum;
    }

    /**
     * Euclidean length of a vector.
     *
     * @param vector the vector
     * @return the square root of the sum of squares
     */
    public static double norm(double[] vector) {
        return Math.sqrt(dot(vector, vector));
    }

    /**
     * Cosine similarity, 1 for vectors that point in the same direction and -1 for opposite directions.
     * An all-zero vector gives 0 instead of NaN.
     *
     * @param first  the first vector
     * @param second the second vector
     * @return the cosine of the angle between the vectors
     */
    public static double cosineSimilarity(double[] first, double[] second) {
        return dot(first, second) / (Math.sqrt(dot(first, first) * dot(second, second)) + NORM_EPSILON);
    }

    /**
     * Cosine distance scaled to the range 0 (same direction) to 1 (opposite direction).
     *
     * @param first  the first vector
     * @param second the second vector
     * @return half of one minus the cosine similarity
     */
    public static double scaledCosineDistance(double[] first, double[] second) {
        return (1.0 - cosineSimilarity(first, second)) / 2.0;
    }

    /**
     * Copies a vector and divides it by its length.
     *
     * @param vector the vector, which is not modified
     * @return a new vector of length 1 (or all zeros when the input is all zeros)
     */
    public static double[] unitVector(float[] vector) {
        double sumOfSquares = 0.0;
        for (float component : vector) {
            sumOfSquares += (double) component * component;
        }
        double norm = Math.sqrt(sumOfSquares) + NORM_EPSILON;
        double[] result = new double[vector.length];
        for (int index = 0; index < vector.length; index++) {
            result[index] = vector[index] / norm;
        }
        return result;
    }

    /**
     * Divides a vector by its length, in place.
     *
     * @param vector the vector that is changed
     */
    public static void normaliseInPlace(double[] vector) {
        double norm = norm(vector);
        for (int index = 0; index < vector.length; index++) {
            vector[index] /= norm;
        }
    }

    /**
     * Root of the mean squared difference of the first components of two vectors.
     *
     * @param first  the first vector
     * @param second the second vector
     * @param length how many leading components are compared
     * @return the root mean squared difference
     */
    public static double rootMeanSquareDifference(double[] first, double[] second, int length) {
        double sum = 0.0;
        for (int index = 0; index < length; index++) {
            double difference = first[index] - second[index];
            sum += difference * difference;
        }
        return Math.sqrt(sum / length);
    }

    /**
     * Adds the rows of a matrix over a range of rows.
     *
     * @param rows      the rows
     * @param firstRow  first row, inclusive
     * @param lastRow   last row, inclusive
     * @param width     length of every row
     * @return the component-wise sum
     */
    public static double[] sumRows(double[][] rows, int firstRow, int lastRow, int width) {
        double[] sum = new double[width];
        for (int row = firstRow; row <= lastRow; row++) {
            for (int component = 0; component < width; component++) {
                sum[component] += rows[row][component];
            }
        }
        return sum;
    }
}
