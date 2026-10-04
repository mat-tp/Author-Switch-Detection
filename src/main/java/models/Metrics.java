package models;

/*TODO: (1) 
    - Add confusion matrix
    - Add Easy/Medium/Hard evaluation
    - Ensure threshold is selected only from validation
    - Save all experiment results 
*/

public class Metrics {

    /**
     * Prints the confusion matrix, the precision, recall, F1 and balanced
     * accuracy of class 1 (change), and the accuracy of always predicting the
     * majority class.
     *
     * Balanced accuracy is the average of the recall obtained on each class
     * (sensitivity and specificity, averaged). Unlike plain accuracy, it is
     * not inflated by an imbalanced class distribution: a model that always
     * predicts the majority class scores 0.5, no matter how skewed the
     * dataset is.
     * 
     * @param predictions 0 / 1 predictions
     * @param truthValues truth values
     * @return {accuracy, precision, recall, f1, balancedAccuracy}
     */
    public static double[] evaluate(int[] predictions, int[] truthValues) {

        if (predictions == null || truthValues == null || predictions.length == 0
                || predictions.length != truthValues.length) {
            throw new IllegalArgumentException("predictions and truthValues must be non-empty and the same length");
        }

        int[] counts = countOutcomes(predictions, truthValues);
        int truePositives = counts[0];
        int falsePositives = counts[1];
        int trueNegatives = counts[2];
        int falseNegatives = counts[3];

        int total = predictions.length;
        int positives = truePositives + falseNegatives;
        int negatives = trueNegatives + falsePositives;

        double accuracy = (double) (truePositives + trueNegatives) / total;
        double precision = (truePositives + falsePositives) == 0 ? 0.0
                : (double) truePositives / (truePositives + falsePositives);
        double recall = positives == 0 ? 0.0 : (double) truePositives / positives; // sensitivity / TPR
        double f1 = (precision + recall) == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);

        double specificity = negatives == 0 ? 0.0 : (double) trueNegatives / negatives; // TNR
        double balancedAccuracy = (recall + specificity) / 2.0;

        // accuracy of a "model" that always predicts the bigger class
        double baseline = (double) Math.max(positives, total - positives) / total;

        System.out.println("Confusion matrix (rows = truth, columns = predicted):");
        System.out.println("             pred 0   pred 1");
        System.out.println("  truth 0  " + String.format("%7d  %7d", trueNegatives, falsePositives));
        System.out.println("  truth 1  " + String.format("%7d  %7d", falseNegatives, truePositives));
        System.out.println(String.format(
                "Accuracy=%.4f (majority baseline=%.4f) | Balanced Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
                accuracy, baseline, balancedAccuracy, precision, recall, f1));

        return new double[] { accuracy, precision, recall, f1, balancedAccuracy };
    }

    /**
     * Calculates only the balanced accuracy (average of the recall of each
     * class), without printing.
     *
     * @param predictions 0 / 1 predictions
     * @param truthValues truth values
     * @return balanced accuracy
     */
    public static double calcBalancedAccuracy(int[] predictions, int[] truthValues) {

        int[] counts = countOutcomes(predictions, truthValues);
        int truePositives = counts[0];
        int falsePositives = counts[1];
        int trueNegatives = counts[2];
        int falseNegatives = counts[3];

        int positives = truePositives + falseNegatives;
        int negatives = trueNegatives + falsePositives;

        double recall = positives == 0 ? 0.0 : (double) truePositives / positives;
        double specificity = negatives == 0 ? 0.0 : (double) trueNegatives / negatives;

        return (recall + specificity) / 2.0;
    }

    /**
     * Prints the confusion matrix + metrics for one difficulty level (via
     * evaluate()) and returns a single human-readable summary line for it, e.g.
     * "easy: n=54, accuracy=0.6296, precision=0.7727, recall=0.5313, f1=0.6296,
     * balancedAccuracy=0.6520". Joining these lines with ";" for "easy",
     * "medium" and "hard" gives the per-difficulty breakdown shown on the
     * Results page (see ModelMetrics.difficultyBreakdownCsv) — used by both
     * Main.java and the web app's RealTrainer, so the two report breakdowns
     * the exact same way.
     *
     * @param label       difficulty name, e.g. "easy"
     * @param predictions 0 / 1 predictions for that difficulty's pairs only
     * @param truthValues truth values for that difficulty's pairs only
     */
    public static String formatBreakdownEntry(String label, int[] predictions, int[] truthValues) {

        double[] result = evaluate(predictions, truthValues);

        return String.format(java.util.Locale.ROOT,
                "%s: n=%d, accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, balancedAccuracy=%.4f",
                label, truthValues.length, result[0], result[1], result[2], result[3],
                result.length > 4 ? result[4] : 0.0);
    }

    /**
     * Calculates only the F1 of class 1 (change), without printing.
     *
     * @param predictions 0 / 1 predictions
     * @param truthValues truth values
     * @return f1
     */
    public static double calcF1(int[] predictions, int[] truthValues) {

        int[] counts = countOutcomes(predictions, truthValues);
        int truePositives = counts[0];
        int falsePositives = counts[1];
        int falseNegatives = counts[3];

        double precision = (truePositives + falsePositives) == 0 ? 0.0
                : (double) truePositives / (truePositives + falsePositives);
        double recall = (truePositives + falseNegatives) == 0 ? 0.0
                : (double) truePositives / (truePositives + falseNegatives);

        return (precision + recall) == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
    }

    /**
     * @return {truePositives, falsePositives, trueNegatives, falseNegatives}
     */
    public static int[] countOutcomes(int[] predictions, int[] truthValues) {

        int truePositives = 0;
        int falsePositives = 0;
        int trueNegatives = 0;
        int falseNegatives = 0;

        for (int i = 0; i < predictions.length; i++) {

            if (predictions[i] == 1 && truthValues[i] == 1) {
                truePositives++;

            } else if (predictions[i] == 1 && truthValues[i] == 0) {
                falsePositives++;

            } else if (predictions[i] == 0 && truthValues[i] == 0) {
                trueNegatives++;

            } else {
                falseNegatives++;
            }
        }

        return new int[] { truePositives, falsePositives, trueNegatives, falseNegatives };
    }
}