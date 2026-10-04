package com.hyp.authorswitchdetection.service;

import models.Metrics;

/**
 * Predicted labels compared with the uploaded truth values.
 * Positive class = 1 = author switch. Same definitions as models.Metrics.
 */
public class EvaluationMetrics {

    public int total;
    public int truePositives;
    public int falsePositives;
    public int trueNegatives;
    public int falseNegatives;

    public int actualSwitches;    // switches in the truth file
    public int predictedSwitches; // switches the model predicted

    public double accuracy;
    public double precision;
    public double recall;         // sensitivity
    public double specificity;
    public double f1;
    public double balancedAccuracy;
    public double majorityBaseline; // accuracy of always predicting the bigger class

    public static EvaluationMetrics of(int[] predictions, int[] truth) {

        EvaluationMetrics m = new EvaluationMetrics();
        int[] counts = Metrics.countOutcomes(predictions, truth); // {TP, FP, TN, FN}

        m.total = predictions.length;
        m.truePositives = counts[0];
        m.falsePositives = counts[1];
        m.trueNegatives = counts[2];
        m.falseNegatives = counts[3];

        m.actualSwitches = m.truePositives + m.falseNegatives;
        m.predictedSwitches = m.truePositives + m.falsePositives;
        int actualSame = m.trueNegatives + m.falsePositives;

        m.accuracy = ratio(m.truePositives + m.trueNegatives, m.total);
        m.precision = ratio(m.truePositives, m.predictedSwitches);
        m.recall = ratio(m.truePositives, m.actualSwitches);
        m.specificity = ratio(m.trueNegatives, actualSame);
        m.f1 = (m.precision + m.recall) == 0.0 ? 0.0 : 2.0 * m.precision * m.recall / (m.precision + m.recall);
        m.balancedAccuracy = (m.recall + m.specificity) / 2.0;
        m.majorityBaseline = ratio(Math.max(m.actualSwitches, actualSame), m.total);

        return m;
    }

    private static double ratio(int part, int whole) {
        return whole == 0 ? 0.0 : (double) part / whole;
    }
}
