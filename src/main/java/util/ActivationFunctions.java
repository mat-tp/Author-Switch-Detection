package util;

/**
 * Activation functions of the logistic, neural and softmax models.
 */
public final class ActivationFunctions {

    private ActivationFunctions() {
        // utility class, not meant to be instantiated
    }

    /**
     * Logistic function.
     *
     * @param score a real number
     * @return a probability between 0 and 1
     */
    public static double sigmoid(double score) {
        return 1 / (1 + Math.exp(-score));
    }

    /**
     * Logistic function applied to every score.
     *
     * @param scores real numbers
     * @return one probability per score
     */
    public static double[] sigmoid(double[] scores) {
        double[] probabilities = new double[scores.length];
        for (int index = 0; index < scores.length; index++) {
            probabilities[index] = sigmoid(scores[index]);
        }
        return probabilities;
    }

    /**
     * Softmax with the largest score subtracted first, which avoids overflow and changes nothing else.
     *
     * @param scores one score per class
     * @return probabilities that add up to 1
     */
    public static double[] softmax(double[] scores) {
        double largest = Double.NEGATIVE_INFINITY;
        for (double score : scores) {
            largest = Math.max(largest, score);
        }
        double[] probabilities = new double[scores.length];
        double total = 0.0;
        for (int index = 0; index < scores.length; index++) {
            probabilities[index] = Math.exp(scores[index] - largest);
            total += probabilities[index];
        }
        for (int index = 0; index < scores.length; index++) {
            probabilities[index] /= total;
        }
        return probabilities;
    }
}
