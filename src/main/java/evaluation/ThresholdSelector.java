package evaluation;

/**
 * Calibrates continuous scores into binary classification decisions by evaluating and selecting
 * the optimal decision threshold against validation data.
 *
 * <p>Centralizes threshold selection logic to ensure all models within the pipeline evaluate
 * boundaries uniformly.</p>
 */
public final class ThresholdSelector {

    /** Resolution of the search grid: 200 discrete steps yielding thresholds of 0.005, 0.010, ... 0.995. */
    private static final int STEPS = 200;

    private ThresholdSelector() {
        // utility class, not meant to be instantiated
    }

    /**
     * Applies a hard decision threshold to an array of continuous scores.
     *
     * <p>A sentence pair is classified as an author change only when its score is strictly
     * greater than the threshold.</p>
     *
     * @param scores    Array of continuous probabilities or calculated change scores.
     * @param threshold The scalar decision boundary.
     * @return Integer array containing 1 for an author change and 0 for no change, per score.
     */
    public static int[] apply(double[] scores, double threshold) {
        int[] decisions = new int[scores.length];
        for (int index = 0; index < scores.length; index++) {
            decisions[index] = scores[index] > threshold ? 1 : 0;
        }
        return decisions;
    }

    /**
     * Finds the decision threshold that yields the highest macro-F1 score on validation data.
     *
     * @param scores      Array of predicted scores generated from the validation dataset.
     * @param truthValues Array of binary ground-truth labels for the validation dataset.
     * @return The optimal decision threshold.
     */
    public static double findBestThreshold(double[] scores, int[] truthValues) {
        return findBestThreshold(scores, truthValues, true);
    }

    /**
     * Finds the optimal threshold by performing a linear grid search in steps of 0.005.
     *
     * <p>If multiple consecutive thresholds yield identical peak performance (a plateau),
     * the median of those thresholds is returned to prevent the decision boundary from
     * sitting on a brittle edge.</p>
     *
     * @param scores      Array of predicted scores generated from the validation dataset.
     * @param truthValues Array of binary ground-truth labels for the validation dataset.
     * @param macro       {@code true} to optimize for macro-F1; {@code false} to optimize strictly for the 'change' class F1 score.
     * @return The optimized double threshold, defaulting to 0.5 if no threshold yields a score above zero.
     */
    public static double findBestThreshold(double[] scores, int[] truthValues, boolean macro) {

        double bestScore = -1.0;
        int firstBest = -1;
        int lastBest = -1;

        for (int step = 1; step < STEPS; step++) {

            int[] decisions = apply(scores, step / (double) STEPS);
            double score = macro ? Metrics.calcMacroF1(decisions, truthValues)
                    : Metrics.calcF1(decisions, truthValues);

            if (score > bestScore + 1e-12) {
                bestScore = score;
                firstBest = step;
                lastBest = step;
            } else if (Math.abs(score - bestScore) <= 1e-12 && lastBest == step - 1) {
                lastBest = step;
            }
        }

        return firstBest > 0 ? (firstBest + lastBest) / 2.0 / STEPS : 0.5;
    }
}