package com.hyp.authorswitchdetection.service;

import evaluation.ConfusionMatrix;

/**
 * The predicted labels compared with the uploaded truth values, as shown on the prediction page.
 * The positive class is 1, an author switch. Every number comes from {@link ConfusionMatrix}.
 */
public class EvaluationMetrics {

    public int total;
    public int truePositives;
    public int falsePositives;
    public int trueNegatives;
    public int falseNegatives;

    /** Switches in the truth file. */
    public int actualSwitches;
    /** Switches the model predicted. */
    public int predictedSwitches;

    public double accuracy;
    public double precision;
    /** Sensitivity. */
    public double recall;
    public double specificity;
    /** F1 score of the class 'switch'. */
    public double f1;
    /** Mean of the F1 scores of both classes, the measure PAN 2026 ranks by. */
    public double macroF1;
    public double balancedAccuracy;
    /** Accuracy of always predicting the bigger class. */
    public double majorityBaseline;

    /**
     * Compares predictions with the truth.
     *
     * @param predictions 0 or 1 per sentence pair
     * @param truth       0 or 1 per sentence pair
     * @return the scores
     */
    public static EvaluationMetrics of(int[] predictions, int[] truth) {

        ConfusionMatrix matrix = ConfusionMatrix.of(predictions, truth);

        EvaluationMetrics metrics = new EvaluationMetrics();
        metrics.total = matrix.total();
        metrics.truePositives = matrix.truePositives;
        metrics.falsePositives = matrix.falsePositives;
        metrics.trueNegatives = matrix.trueNegatives;
        metrics.falseNegatives = matrix.falseNegatives;
        metrics.actualSwitches = matrix.actualChanges();
        metrics.predictedSwitches = matrix.predictedChanges();
        metrics.accuracy = matrix.accuracy();
        metrics.precision = matrix.precision();
        metrics.recall = matrix.recall();
        metrics.specificity = matrix.specificity();
        metrics.f1 = matrix.f1Change();
        metrics.macroF1 = matrix.macroF1();
        metrics.balancedAccuracy = matrix.balancedAccuracy();
        metrics.majorityBaseline = matrix.majorityBaseline();
        return metrics;
    }
}
