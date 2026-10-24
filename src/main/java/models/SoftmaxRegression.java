package models;

import util.ActivationFunctions;

import java.io.Serializable;

/**
 * Logistic regression for MORE than two classes (softmax regression), written the same way as LogisticRegression:
 * batch gradient descent on the cross-entropy, with an L2 penalty and optional class weights.
 *
 * It is used by the DifficultyDetector (easy / medium / hard), not for the author switch itself.
 *
 * p(class c | x) = exp(w_c . x + b_c) / sum_k exp(w_k . x + b_k)
 */
public class SoftmaxRegression implements Serializable {

    private static final long serialVersionUID = 1L;

    private final int classes;
    private final int features;

    // weights[c][j] for j < features, weights[c][features] is the bias of class c
    private final double[][] weights;

    public SoftmaxRegression(int classes, int features) {
        this.classes = classes;
        this.features = features;
        this.weights = new double[classes][features + 1];
    }

    public int classes() {
        return classes;
    }

    public int features() {
        return features;
    }

    /** The probabilities of the classes for one feature vector. */
    public double[] probabilities(double[] x) {

        double[] scores = new double[classes];

        for (int c = 0; c < classes; c++) {
            double sum = weights[c][features];
            for (int j = 0; j < features; j++) {
                sum += weights[c][j] * x[j];
            }
            scores[c] = sum;
        }
        return ActivationFunctions.softmax(scores);
    }

    /** The class with the highest probability. */
    public int predict(double[] x) {

        double[] p = probabilities(x);
        int best = 0;
        for (int c = 1; c < classes; c++) {
            if (p[c] > p[best]) {
                best = c;
            }
        }
        return best;
    }

    /**
     * Mean cross-entropy (plus the L2 penalty) and its gradient, over all rows.
     *
     * @param gradient same shape as the weights, filled by this call (may be null when only the loss is wanted)
     */
    public double lossAndGradient(double[][] x, int[] y, double[] classWeight, double l2, double[][] gradient) {

        int n = x.length;
        double loss = 0.0;
        double weightSum = 0.0;

        if (gradient != null) {
            for (double[] row : gradient) {
                java.util.Arrays.fill(row, 0.0);
            }
        }

        for (int i = 0; i < n; i++) {

            double w = classWeight == null ? 1.0 : classWeight[y[i]];
            double[] p = probabilities(x[i]);

            loss -= w * Math.log(Math.max(p[y[i]], 1e-12));
            weightSum += w;

            if (gradient != null) {
                for (int c = 0; c < classes; c++) {
                    double error = w * (p[c] - (c == y[i] ? 1.0 : 0.0));
                    for (int j = 0; j < features; j++) {
                        gradient[c][j] += error * x[i][j];
                    }
                    gradient[c][features] += error;
                }
            }
        }

        double penalty = 0.0;
        for (int c = 0; c < classes; c++) {
            for (int j = 0; j < features; j++) { // the bias is not penalised
                penalty += weights[c][j] * weights[c][j];
            }
        }

        if (gradient != null) {
            for (int c = 0; c < classes; c++) {
                for (int j = 0; j <= features; j++) {
                    gradient[c][j] /= weightSum;
                    if (j < features) {
                        gradient[c][j] += l2 * weights[c][j];
                    }
                }
            }
        }
        return loss / weightSum + 0.5 * l2 * penalty;
    }

    /** Sets one weight (used by the gradient check of SelfCheck). */
    public void setWeight(int c, int j, double value) {
        weights[c][j] = value;
    }

    public double getWeight(int c, int j) {
        return weights[c][j];
    }

    /**
     * Trains by batch gradient descent.
     *
     * @param balanced true = every class counts the same, whatever its number of rows
     */
    public static SoftmaxRegression train(double[][] x, int[] y, int classes, int iterations, double rate, double l2,
            boolean balanced) {

        int features = x[0].length;
        SoftmaxRegression model = new SoftmaxRegression(classes, features);

        double[] classWeight = null;
        if (balanced) {
            int[] counts = new int[classes];
            for (int label : y) {
                counts[label]++;
            }
            classWeight = new double[classes];
            for (int c = 0; c < classes; c++) {
                classWeight[c] = counts[c] == 0 ? 0.0 : (double) y.length / (classes * counts[c]);
            }
        }

        double[][] gradient = new double[classes][features + 1];
        // momentum: the same plain gradient step, but a little faster on the long flat part of the loss
        double[][] velocity = new double[classes][features + 1];

        for (int it = 0; it < iterations; it++) {

            model.lossAndGradient(x, y, classWeight, l2, gradient);

            for (int c = 0; c < classes; c++) {
                for (int j = 0; j <= features; j++) {
                    velocity[c][j] = 0.9 * velocity[c][j] - rate * gradient[c][j];
                    model.weights[c][j] += velocity[c][j];
                }
            }
        }
        return model;
    }
}
