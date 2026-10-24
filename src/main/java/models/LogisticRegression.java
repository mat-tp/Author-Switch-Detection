package models;

import util.ActivationFunctions;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class LogisticRegression {

    // consistent random sampling for repeatability of results
    private static final long RANDOM_SEED = 42L;

    // the samples are cut in chunks that are calculated in parallel (the sums are added in the SAME order
    // every time, so the result does not depend on the speed of the threads)
    private static final int CHUNK_SIZE = 2048;

    // the training stops when the loss changes less than this (relative) for 'STALL_ITERATIONS' iterations
    private static final double STALL_TOLERANCE = 1e-9;
    private static final int STALL_ITERATIONS = 20;

    public static double[] gradiantDescent(double[][] featureMatrix, int[] truthValues, int nIterations,
            double learningRate) {

        return gradiantDescent(featureMatrix, truthValues, nIterations, learningRate, false);
    }

    public static double[] calcSampleWeights(int[] truthValues, boolean useClassWeights) {

        double[] sampleWeights = new double[truthValues.length];
        // Since I have Binary class values I can just focus on It.
        double weightForZero = 1.0;
        double weightForOne = 1.0;

        if (useClassWeights) {

            int ones = 0;
            int zeros = 0;

            for (int truth : truthValues) {
                ones += truth; // aggregated
            }
            // the remain equate the number of zeros
            zeros = truthValues.length - ones;

            // ensure that they exist to be able to balance
            if (ones > 0 && zeros > 0) {
                // following the compute class weight implementation
                int N = truthValues.length;
                int uniqueClasses = 2;

                // (double) is needed, otherwise Java divides as integers and drops the decimals
                // (e.g. a weight of 0.67 becomes 0, and that class is ignored while training)
                weightForZero = (double) N / (uniqueClasses * zeros);
                weightForOne = (double) N / (uniqueClasses * ones);

            }

        }

        // Update the wights for each index
        for (int sample = 0; sample < truthValues.length; sample++) {

            sampleWeights[sample] = truthValues[sample] == 1 ? weightForOne : weightForZero;
        }

        return sampleWeights;
    }

    /**
     * Binary cross- entropy for a single sample
     *
     * @param prediction probability from a sigmoid function
     * @param truth      truth value (0 / 1)
     * @return loss of the sample
     */
    private static double calcSampleLoss(double prediction, int truth) {
        // Avoiding negative infinity, Log(0)
        double epsilon = 1e-12;
        double pred = Math.min(Math.max(prediction, epsilon), 1 - epsilon);

        return truth == 1 ? -Math.log(pred) : -Math.log(1.0 - pred);
    }

    public static double[] gradiantDescent(double[][] featureMatrix, int[] truthValues, int nIterations,
            double learningRate, boolean useClassWeights) {

        return gradiantDescent(featureMatrix, truthValues, nIterations, learningRate, useClassWeights, null);
    }

    /**
     * Same as above, but also records the training loss into lossHistoryOut as it
     * goes (one value per iteration), so it can be charted afterwards. Pass null
     * when the loss history is not needed, same as the overload above.
     */
    public static double[] gradiantDescent(double[][] featureMatrix, int[] truthValues, int nIterations,
            double learningRate, boolean useClassWeights, List<Double> lossHistoryOut) {

        return gradiantDescent(featureMatrix, truthValues, nIterations, learningRate, useClassWeights, 0.0,
                lossHistoryOut);
    }

    /**
     * Full batch gradient descent with an optional L2 penalty.
     *
     * @param l2Lambda L2 regularisation of the weights (0 = none, the bias is never
     *                 penalised). It keeps the weights of the (many) correlated pair features
     *                 small, which matters with the wide full pair features.
     */
    public static double[] gradiantDescent(double[][] featureMatrix, int[] truthValues, int nIterations,
            double learningRate, boolean useClassWeights, double l2Lambda, List<Double> lossHistoryOut) {
        /*
         * This method defines a Gradient Algorithm: A method for finding weights and
         * biases for the Logistic Regression and Neural Network.
         *
         * Logistic Regression:
         * - Find the score (z) = (features * weights for features at i)
         * - Convert into 0/1 using sigmoid function, gives the prediction
         * - calculate the error
         * - calculate the gradients for weights & update the weights : Repeat
         */

        // ensure values are valid
        if (featureMatrix == null || truthValues == null || featureMatrix.length == 0
                || featureMatrix.length != truthValues.length) {
            throw new IllegalArgumentException("featureMatrix and truthValues must be non-empty and the same length");
        }
        if (l2Lambda < 0.0) {
            throw new IllegalArgumentException("l2Lambda can not be negative");
        }

        int nFeatures = featureMatrix[0].length;

        double[] weightsBiasArray = new double[nFeatures + 1];
        double[] weights = new double[nFeatures];

        Random random = new Random(RANDOM_SEED);

        // generate random values for the weights & bias
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextGaussian() * 0.01;

        }

        double bias = 0.0;

        /*
         * If No class:
         * Assigning the class weights (1(O Binary class):1 (1 Binary class))
         * else:
         * Assigns different weights; due to class Imbalance-ness
         */

        double[] sampleWeights = calcSampleWeights(truthValues, useClassWeights);

        double[] weightGradients = new double[nFeatures];
        double previousLoss = Double.NaN;
        int stalled = 0;

        // updating after each iteration
        for (int iteration = 0; iteration < nIterations; iteration++) {

            // the data loss, and the gradients (written into weightGradients, the bias gradient is returned last)
            double[] lossAndBiasGradient = lossAndGradient(featureMatrix, truthValues, sampleWeights, weights, bias,
                    weightGradients);
            double loss = lossAndBiasGradient[0];
            double biasGradient = lossAndBiasGradient[1];

            // L2 penalty: (lambda / 2) * ||w||^2 has the gradient lambda * w
            double penalty = 0.0;
            if (l2Lambda > 0.0) {
                for (int feature = 0; feature < nFeatures; feature++) {
                    weightGradients[feature] += l2Lambda * weights[feature];
                    penalty += weights[feature] * weights[feature];
                }
                penalty *= l2Lambda / 2.0;
            }

            // update the weights
            for (int feature = 0; feature < nFeatures; feature++) {

                weights[feature] -= learningRate * weightGradients[feature];
            }

            bias -= learningRate * biasGradient;

            if (lossHistoryOut != null) {
                lossHistoryOut.add(loss);
            }

            // verbose : showing that the loss is going down the iterations
            if (iteration == 0 || (iteration + 1) % 100 == 0 || iteration == nIterations - 1) {
                System.out.println(String.format("  Iteration %d / %d | Loss: %.6f", iteration + 1, nIterations,
                        loss + penalty));
            }

            // converged: the loss does not move any more
            if (!Double.isNaN(previousLoss)
                    && Math.abs(previousLoss - loss) <= STALL_TOLERANCE * Math.max(1.0, Math.abs(previousLoss))) {
                stalled++;
                if (stalled >= STALL_ITERATIONS) {
                    System.out.println(String.format("  Converged at iteration %d (loss %.6f)", iteration + 1, loss));
                    break;
                }
            } else {
                stalled = 0;
            }
            previousLoss = loss;

        }

        for (int feature = 0; feature < nFeatures; feature++) {
            weightsBiasArray[feature] = weights[feature];
        }

        weightsBiasArray[nFeatures] = bias;

        return weightsBiasArray;
    }

    /**
     * The average (class weighted) loss and its gradients over ALL the samples.
     *
     * @param weightGradients receives the gradient of every weight (it is cleared first)
     * @return {loss, gradient of the bias}
     */
    private static double[] lossAndGradient(double[][] featureMatrix, int[] truthValues, double[] sampleWeights,
            double[] weights, double bias, double[] weightGradients) {

        int nSample = featureMatrix.length;
        int nFeatures = weights.length;
        int nChunks = (nSample + CHUNK_SIZE - 1) / CHUNK_SIZE;

        // one result per chunk: {loss, biasGradient, weightGradients...}
        double[][] partial = new double[nChunks][];

        IntStream.range(0, nChunks).parallel().forEach(chunk -> {

            double[] result = new double[2 + nFeatures];
            int from = chunk * CHUNK_SIZE;
            int to = Math.min(from + CHUNK_SIZE, nSample);

            // calculations for each sample, weights & losses (errors)
            for (int sample = from; sample < to; sample++) {

                double[] row = featureMatrix[sample];
                double zScore = bias; // initialising the score to bias value

                for (int feature = 0; feature < nFeatures; feature++) {
                    zScore += row[feature] * weights[feature];
                }

                // calculating the probability score using the sigmoid function.
                double prediction = ActivationFunctions.sigmoid(zScore);
                // using the class wights to adjust and focus correctly based on Class-weights
                // instead
                double error = (prediction - truthValues[sample]) * sampleWeights[sample];

                // Calculating the Gradients
                for (int feature = 0; feature < nFeatures; feature++) {
                    result[2 + feature] += row[feature] * error;
                }
                result[1] += error;
                result[0] += sampleWeights[sample] * calcSampleLoss(prediction, truthValues[sample]);
            }

            partial[chunk] = result;
        });

        // added in chunk order
        java.util.Arrays.fill(weightGradients, 0.0);
        double loss = 0.0;
        double biasGradient = 0.0;

        for (double[] result : partial) {
            loss += result[0];
            biasGradient += result[1];
            for (int feature = 0; feature < nFeatures; feature++) {
                weightGradients[feature] += result[2 + feature];
            }
        }

        // Average the gradients over all samples
        for (int feature = 0; feature < nFeatures; feature++) {
            weightGradients[feature] /= nSample;
        }

        return new double[] { loss / nSample, biasGradient / nSample };
    }

    /**
     * Gradient check (used by SelfCheck): the largest relative difference between the gradient of
     * the code and the numerical gradient (loss(w + h) - loss(w - h)) / 2h, over all weights and
     * the bias. A correct gradient gives about 1e-7 or less.
     */
    public static double maxGradientError(double[][] featureMatrix, int[] truthValues, double[] weights, double bias,
            boolean useClassWeights) {

        double[] sampleWeights = calcSampleWeights(truthValues, useClassWeights);
        int nFeatures = weights.length;

        double[] analytic = new double[nFeatures];
        double biasGradient = lossAndGradient(featureMatrix, truthValues, sampleWeights, weights, bias, analytic)[1];

        double h = 1e-6;
        double worst = 0.0;
        double[] scratch = new double[nFeatures];

        for (int feature = -1; feature < nFeatures; feature++) {

            double[] plus = weights.clone();
            double[] minus = weights.clone();
            double biasPlus = bias;
            double biasMinus = bias;

            if (feature < 0) {
                biasPlus += h;
                biasMinus -= h;
            } else {
                plus[feature] += h;
                minus[feature] -= h;
            }

            double lossPlus = lossAndGradient(featureMatrix, truthValues, sampleWeights, plus, biasPlus, scratch)[0];
            double lossMinus = lossAndGradient(featureMatrix, truthValues, sampleWeights, minus, biasMinus,
                    scratch)[0];

            double numeric = (lossPlus - lossMinus) / (2.0 * h);
            double code = feature < 0 ? biasGradient : analytic[feature];
            double relative = Math.abs(numeric - code) / Math.max(1e-8, Math.abs(numeric) + Math.abs(code));

            worst = Math.max(worst, relative);
        }

        return worst;
    }

    public static double[] predictProbabilities(double[][] featureMatrix, double[] weightsBiasArray) {

        int nFeatures = weightsBiasArray.length - 1;
        double[] probabilities = new double[featureMatrix.length];

        IntStream.range(0, featureMatrix.length).parallel().forEach(sample -> {

            // sample check
            if (featureMatrix[sample].length != nFeatures) {
                throw new IllegalArgumentException("Sample " + sample + " has " + featureMatrix[sample].length
                        + " features, but the model has " + nFeatures);
            }

            double zScore = weightsBiasArray[nFeatures]; // bias value

            for (int feature = 0; feature < nFeatures; feature++) {
                zScore += featureMatrix[sample][feature] * weightsBiasArray[feature];
            }

            probabilities[sample] = ActivationFunctions.sigmoid(zScore);

        });

        return probabilities;
    }

    /**
     * Predicts a single sample (used for inference on a new pair).
     *
     * @param features         features of the pair
     * @param weightsBiasArray weights followed by the bias
     * @param threshold        probability from which the sample is a change
     * @return 0 = no change, 1 = change
     */
    public static int predict(double[] features, double[] weightsBiasArray, double threshold) {

        return predictProbabilities(new double[][] { features }, weightsBiasArray)[0] > threshold ? 1 : 0;
    }
}
