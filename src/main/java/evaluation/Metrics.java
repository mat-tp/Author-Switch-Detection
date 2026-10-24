package evaluation;

import java.util.Locale;

/**
 * Scores for a set of 0/1 predictions against the truth, as printed by the command line and stored in
 * the result files. The numbers come from {@link ConfusionMatrix}.
 */
public final class Metrics {

    /** Index of the accuracy in the array returned by {@link #evaluate}. */
    public static final int ACCURACY = 0;
    /** Index of the precision of the class 'change'. */
    public static final int PRECISION = 1;
    /** Index of the recall of the class 'change'. */
    public static final int RECALL = 2;
    /** Index of the F1 score of the class 'change'. */
    public static final int F1_CHANGE = 3;
    /** Index of the balanced accuracy. */
    public static final int BALANCED_ACCURACY = 4;
    /** Index of the macro-F1, the official measure of the PAN 2026 task. */
    public static final int MACRO_F1 = 5;

    private Metrics() {
        // utility class, not meant to be instantiated
    }

    /**
     * Prints the confusion matrix and the scores, and returns them.
     *
     * @param predictions 0 or 1 for every pair
     * @param truthValues 0 or 1 for every pair
     * @return {accuracy, precision, recall, f1, balancedAccuracy, macroF1}
     */
    public static double[] evaluate(int[] predictions, int[] truthValues) {

        double[] result = evaluateQuietly(predictions, truthValues);
        ConfusionMatrix matrix = ConfusionMatrix.of(predictions, truthValues);

        System.out.println("Confusion matrix (rows = truth, columns = predicted):");
        System.out.println("             pred 0   pred 1");
        System.out.println("  truth 0  " + String.format("%7d  %7d", matrix.trueNegatives, matrix.falsePositives));
        System.out.println("  truth 1  " + String.format("%7d  %7d", matrix.falseNegatives, matrix.truePositives));
        System.out.println(String.format(
                "Accuracy=%.4f (majority baseline=%.4f) | Balanced Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1(change)=%.4f | Macro-F1=%.4f",
                result[ACCURACY], matrix.majorityBaseline(), result[BALANCED_ACCURACY], result[PRECISION],
                result[RECALL], result[F1_CHANGE], result[MACRO_F1]));

        return result;
    }

    /**
     * The same scores as {@link #evaluate}, without printing. The ablation study evaluates hundreds of runs.
     *
     * @param predictions 0 or 1 for every pair
     * @param truthValues 0 or 1 for every pair
     * @return {accuracy, precision, recall, f1, balancedAccuracy, macroF1}
     * @throws IllegalArgumentException when the arrays are empty or of different length
     */
    public static double[] evaluateQuietly(int[] predictions, int[] truthValues) {

        if (predictions == null || truthValues == null || predictions.length == 0
                || predictions.length != truthValues.length) {
            throw new IllegalArgumentException("predictions and truthValues must be non-empty and the same length");
        }

        ConfusionMatrix matrix = ConfusionMatrix.of(predictions, truthValues);
        return new double[] { matrix.accuracy(), matrix.precision(), matrix.recall(), matrix.f1Change(),
                matrix.balancedAccuracy(), matrix.macroF1() };
    }

    /**
     * Macro-F1: the mean of the F1 score of 'change' and of 'same author', over the pooled pairs of all
     * documents. This is the measure the PAN 2026 task ranks by.
     *
     * @param predictions 0 or 1 for every pair
     * @param truthValues 0 or 1 for every pair
     * @return the macro-F1
     */
    public static double calcMacroF1(int[] predictions, int[] truthValues) {
        return ConfusionMatrix.of(predictions, truthValues).macroF1();
    }

    /**
     * Balanced accuracy, the mean of the recall of both classes.
     *
     * @param predictions 0 or 1 for every pair
     * @param truthValues 0 or 1 for every pair
     * @return the balanced accuracy
     */
    public static double calcBalancedAccuracy(int[] predictions, int[] truthValues) {
        return ConfusionMatrix.of(predictions, truthValues).balancedAccuracy();
    }

    /**
     * F1 score of the class 'change' only.
     *
     * @param predictions 0 or 1 for every pair
     * @param truthValues 0 or 1 for every pair
     * @return the F1 score of the class 'change'
     */
    public static double calcF1(int[] predictions, int[] truthValues) {
        return ConfusionMatrix.of(predictions, truthValues).f1Change();
    }

    /**
     * Counts the outcomes of a prediction.
     *
     * @param predictions 0 or 1 for every pair
     * @param truthValues 0 or 1 for every pair
     * @return {truePositives, falsePositives, trueNegatives, falseNegatives}
     */
    public static int[] countOutcomes(int[] predictions, int[] truthValues) {
        return ConfusionMatrix.of(predictions, truthValues).toArray();
    }

    /**
     * Prints the scores of one difficulty level and returns a one-line summary of them. The lines of
     * easy, medium and hard, joined with a semicolon, are the per-difficulty breakdown of the Results page.
     *
     * @param label       the difficulty name, for example 'easy'
     * @param predictions 0 or 1 for the pairs of that difficulty
     * @param truthValues 0 or 1 for the pairs of that difficulty
     * @return for example 'easy: n=54, accuracy=0.6296, precision=0.7727, ...'
     */
    public static String formatBreakdownEntry(String label, int[] predictions, int[] truthValues) {

        double[] result = evaluate(predictions, truthValues);

        return String.format(Locale.ROOT,
                "%s: n=%d, accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, balancedAccuracy=%.4f, macroF1=%.4f",
                label, truthValues.length, result[ACCURACY], result[PRECISION], result[RECALL], result[F1_CHANGE],
                result[BALANCED_ACCURACY], result[MACRO_F1]);
    }
}
