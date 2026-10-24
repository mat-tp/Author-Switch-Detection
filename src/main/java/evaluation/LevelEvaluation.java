package evaluation;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * The evaluation steps that the command-line trainer, the web trainer and the ablation study share:
 * choosing thresholds on validation data and breaking a result down by difficulty level.
 *
 * <p>A scorer turns a pair-feature matrix into one change score per row. Thresholds are always chosen
 * on validation data and then applied to the evaluation data, never the other way round.</p>
 */
public final class LevelEvaluation {

    /** Threshold used when there is no validation data to choose one on. */
    public static final double DEFAULT_THRESHOLD = 0.5;

    private LevelEvaluation() {
        // utility class, not meant to be instantiated
    }

    /**
     * Chooses the pooled threshold, the one with the best macro-F1 on the validation data.
     *
     * @param scorer     turns a pair-feature matrix into change scores
     * @param validation the validation matrix, or {@code null} when there is none
     * @param truth      the truth of the validation matrix
     * @return the threshold, {@link #DEFAULT_THRESHOLD} without validation data
     */
    public static double chooseThreshold(Function<double[][], double[]> scorer, double[][] validation, int[] truth) {
        return validation == null ? DEFAULT_THRESHOLD
                : ThresholdSelector.findBestThreshold(scorer.apply(validation), truth);
    }

    /**
     * Chooses one threshold per difficulty level, each on the validation data of that level only.
     *
     * @param scorer      turns a pair-feature matrix into change scores
     * @param validation  validation matrices by level; a missing or empty level is skipped
     * @param truth       validation truth by level
     * @param levels      the levels in the order they are reported
     * @return the threshold of every level that has validation data, in the order of {@code levels}
     */
    public static Map<String, Double> chooseLevelThresholds(Function<double[][], double[]> scorer,
            Map<String, double[][]> validation, Map<String, int[]> truth, Iterable<String> levels) {

        Map<String, Double> thresholds = new LinkedHashMap<>();
        for (String level : levels) {
            double[][] matrix = validation.get(level);
            if (matrix != null && matrix.length > 0) {
                thresholds.put(level, ThresholdSelector.findBestThreshold(scorer.apply(matrix), truth.get(level)));
            }
        }
        return thresholds;
    }

    /**
     * Scores every difficulty level of the evaluation data with one threshold and describes the result.
     *
     * @param scorer     turns a pair-feature matrix into change scores
     * @param threshold  the threshold to apply
     * @param evaluation evaluation matrices by level; a missing or empty level is skipped
     * @param truth      evaluation truth by level
     * @param levels     the levels in the order they are reported
     * @param label      printed before every level ('Test', 'Validation'), or {@code null} to print nothing extra
     * @return one summary line per level, joined with a semicolon
     *         (see {@link Metrics#formatBreakdownEntry}), empty when no level had data
     */
    public static String breakdown(Function<double[][], double[]> scorer, double threshold,
            Map<String, double[][]> evaluation, Map<String, int[]> truth, Iterable<String> levels, String label) {

        StringBuilder summary = new StringBuilder();
        for (String level : levels) {
            double[][] matrix = evaluation.get(level);
            if (matrix == null || matrix.length == 0) {
                continue;
            }
            if (label != null) {
                System.out.println();
                System.out.println(label + " (" + level + "):");
            }
            int[] decisions = ThresholdSelector.apply(scorer.apply(matrix), threshold);
            if (summary.length() > 0) {
                summary.append(';');
            }
            summary.append(Metrics.formatBreakdownEntry(level, decisions, truth.get(level)));
        }
        return summary.toString();
    }

    /**
     * Macro-F1 of one level when the level's own threshold is used.
     *
     * @param scorer     turns a pair-feature matrix into change scores
     * @param threshold  the threshold of the level
     * @param matrix     the evaluation matrix of the level
     * @param truth      the evaluation truth of the level
     * @return the macro-F1
     */
    public static double macroF1(Function<double[][], double[]> scorer, double threshold, double[][] matrix,
            int[] truth) {
        return Metrics.calcMacroF1(ThresholdSelector.apply(scorer.apply(matrix), threshold), truth);
    }
}
