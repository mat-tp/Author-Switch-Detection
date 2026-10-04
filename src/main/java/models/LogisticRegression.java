package models;

import java.util.List;
import java.util.Random;

public class LogisticRegression {

    // consistent random sampling for repeatability of results
    private static final long RANDOM_SEED = 42L;

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
        // TODO: refactor to a separate method
        if (featureMatrix == null || truthValues == null || featureMatrix.length == 0
                || featureMatrix.length != truthValues.length) {
            throw new IllegalArgumentException("featureMatrix and truthValues must be non-empty and the same length");
        }

        int nSample = featureMatrix.length;
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

        // updating after each iteration
        for (int iteration = 0; iteration < nIterations; iteration++) {

            double[] weightGradients = new double[nFeatures];
            double biasGradient = 0.0;
            double loss = 0.0;

            // calculations for each sample, weights & losses (errors)
            for (int sample = 0; sample < featureMatrix.length; sample++) {

                double zScore = bias; // initialising the score to bias value

                for (int feature = 0; feature < nFeatures; feature++) {

                    zScore += featureMatrix[sample][feature] * weights[feature];

                }

                // calculating the probability score using the sigmoid function.
                double prediction = sigmoid(zScore);
                // using the class wights to adjust and focus correctly based on Class-weights
                // instead
                double error = (prediction - truthValues[sample]) * sampleWeights[sample];

                // Calculating the Gradients
                for (int feature = 0; feature < nFeatures; feature++) {
                    weightGradients[feature] += featureMatrix[sample][feature] * error;
                }
                biasGradient += error;

                loss += sampleWeights[sample] * calcSampleLoss(prediction, truthValues[sample]);

            }

            // Average the feature gradients over all samples
            for (int feature = 0; feature < nFeatures; feature++) {

                weightGradients[feature] /= nSample;
            }

            biasGradient /= nSample;
            loss /= nSample;

            // update the weights
            for (int feature = 0; feature < featureMatrix[0].length; feature++) {

                weights[feature] -= learningRate * weightGradients[feature];
            }

            bias -= learningRate * biasGradient;

            if (lossHistoryOut != null) {
                lossHistoryOut.add(loss);
            }

            // verbose : showing that the loss is going down the iterations
            if (iteration == 0 || (iteration + 1) % 100 == 0 || iteration == nIterations - 1) {
                System.out.println(String.format("  Iteration %d / %d | Loss: %.6f", iteration + 1, nIterations, loss));
            }

        }

        for (int feature = 0; feature < nFeatures; feature++) {
            weightsBiasArray[feature] = weights[feature];
        }

        weightsBiasArray[nFeatures] = bias;

        return weightsBiasArray;
    }

    public static double[] predictProbabilities(double[][] featureMatrix, double[] weightsBiasArray) {

        int nFeatures = weightsBiasArray.length - 1;
        double[] probabilities = new double[featureMatrix.length];

        for (int sample = 0; sample < featureMatrix.length; sample++) {

            // sample check
            if (featureMatrix[sample].length != nFeatures) {
                throw new IllegalArgumentException("Sample " + sample + " has " + featureMatrix[sample].length
                        + " features, but the model has " + nFeatures);
            }

            double zScore = weightsBiasArray[nFeatures]; // bias value

            for (int feature = 0; feature < nFeatures; feature++) {
                zScore += featureMatrix[sample][feature] * weightsBiasArray[feature];
            }

            probabilities[sample] = sigmoid(zScore);

        }

        return probabilities;
    }

    /**
     * Getting the predictions from batch processing (Array of probilities)
     * 
     * @param probabilities
     * @param threshold
     * @return
     */
    public static int[] predict(double[] probabilities, double threshold) {
        int[] predictions = new int[probabilities.length];

        for (int i = 0; i < probabilities.length; i++) {
            predictions[i] = probabilities[i] > threshold ? 1 : 0;
        }

        return predictions;
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

        return predict(predictProbabilities(new double[][] { features }, weightsBiasArray), threshold)[0];
    }

    /**
     * Searches for the threshold with the best F1 of class 1 (change).
     * Use it on the VALIDATION data only, then keep this threshold for the test
     * data.
     *
     * @param probabilities probabilities on the validation data
     * @param truthValues   truth values of the validation data
     * @return the best threshold
     */
    public static double findBestThreshold(double[] probabilities, int[] truthValues) {

        double bestThreshold = 0.5;
        double bestF1 = -1.0;

        for (int step = 1; step < 100; step++) {

            double threshold = step / 100.0;
            double f1 = Metrics.calcF1(predict(probabilities, threshold), truthValues);

            if (f1 > bestF1) {
                bestF1 = f1;
                bestThreshold = threshold;
            }
        }

        return bestThreshold;
    }

    /**
     * This method computes sigmoid activation
     *
     * @param scores These are the product of features (X) multiplied by their
     *               weights
     * @return double[] Array of predictions for array of scores
     */
    public static double[] sigmoid(double[] scores) {

        double[] predictions = new double[scores.length];

        for (int i = 0; i < scores.length; i++) {

            predictions[i] = sigmoid(scores[i]);
        }

        return predictions;
    }

    /**
     * calculates sigmoid activation function
     *
     * @param score linear score from multiplying weights with each feature
     * @return probability Score
     */
    public static double sigmoid(double score) {

        return 1 / (1 + Math.exp(-score));
    }

}