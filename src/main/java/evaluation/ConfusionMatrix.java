package evaluation;

/**
 * The four outcomes of a binary decision, with every score of the project derived from them.
 *
 * <p>The positive class is 1, an author change between two neighbouring sentences. The other class, 0, is
 * 'same author'. All metrics of the project (command line, ablation, web pages, TIRA runner) are computed
 * here, so that they cannot drift apart.</p>
 */
public final class ConfusionMatrix {

    public final int truePositives;
    public final int falsePositives;
    public final int trueNegatives;
    public final int falseNegatives;

    private ConfusionMatrix(int truePositives, int falsePositives, int trueNegatives, int falseNegatives) {
        this.truePositives = truePositives;
        this.falsePositives = falsePositives;
        this.trueNegatives = trueNegatives;
        this.falseNegatives = falseNegatives;
    }

    /**
     * Counts the outcomes of a prediction.
     *
     * @param predictions 0 or 1 for every pair
     * @param truth       0 or 1 for every pair, same length
     * @return the confusion matrix
     */
    public static ConfusionMatrix of(int[] predictions, int[] truth) {
        int truePositives = 0;
        int falsePositives = 0;
        int trueNegatives = 0;
        int falseNegatives = 0;
        for (int index = 0; index < predictions.length; index++) {
            if (predictions[index] == 1) {
                if (truth[index] == 1) {
                    truePositives++;
                } else {
                    falsePositives++;
                }
            } else if (truth[index] == 0) {
                trueNegatives++;
            } else {
                falseNegatives++;
            }
        }
        return new ConfusionMatrix(truePositives, falsePositives, trueNegatives, falseNegatives);
    }

    /** @return the number of pairs */
    public int total() {
        return truePositives + falsePositives + trueNegatives + falseNegatives;
    }

    /** @return the number of real changes in the truth */
    public int actualChanges() {
        return truePositives + falseNegatives;
    }

    /** @return the number of pairs without a change in the truth */
    public int actualSame() {
        return trueNegatives + falsePositives;
    }

    /** @return the number of changes that were predicted */
    public int predictedChanges() {
        return truePositives + falsePositives;
    }

    /** @return the share of correct predictions */
    public double accuracy() {
        return ratio(truePositives + trueNegatives, total());
    }

    /** @return the share of predicted changes that are real changes */
    public double precision() {
        return ratio(truePositives, predictedChanges());
    }

    /** @return the share of real changes that were found (sensitivity) */
    public double recall() {
        return ratio(truePositives, actualChanges());
    }

    /** @return the share of unchanged pairs that were left alone */
    public double specificity() {
        return ratio(trueNegatives, actualSame());
    }

    /** @return the mean of recall and specificity, which a constant prediction can not inflate */
    public double balancedAccuracy() {
        return (recall() + specificity()) / 2.0;
    }

    /** @return the F1 score of the class 'change' */
    public double f1Change() {
        return f1(truePositives, falsePositives, falseNegatives);
    }

    /** @return the F1 score of the class 'same author' */
    public double f1Same() {
        return f1(trueNegatives, falseNegatives, falsePositives);
    }

    /** @return the mean of both F1 scores, the official measure of the PAN 2026 task */
    public double macroF1() {
        return (f1Change() + f1Same()) / 2.0;
    }

    /** @return the accuracy of always predicting the bigger class */
    public double majorityBaseline() {
        return ratio(Math.max(actualChanges(), actualSame()), total());
    }

    /** @return {truePositives, falsePositives, trueNegatives, falseNegatives} */
    public int[] toArray() {
        return new int[] { truePositives, falsePositives, trueNegatives, falseNegatives };
    }

    private static double f1(int truePositives, int falsePositives, int falseNegatives) {
        int denominator = 2 * truePositives + falsePositives + falseNegatives;
        return denominator == 0 ? 0.0 : 2.0 * truePositives / denominator;
    }

    private static double ratio(int part, int whole) {
        return whole == 0 ? 0.0 : (double) part / whole;
    }
}
