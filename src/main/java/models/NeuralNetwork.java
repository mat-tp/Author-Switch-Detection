package models;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static models.LogisticRegression.sigmoid;

public class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 1L;

    // consistent random sampling for repeatability of results
    private static final long RANDOM_SEED = 42L;

    /*
     * The training data is only needed while training, so it is transient:
     * otherwise saving the model would also save the whole feature matrix.
     */
    transient double[][] featureMatrix;
    transient int[] truthValues;

    // parameters for the network= for representing the whole network with more
    // layers
    /*
     * weights[layerNumberInTheNetwork][PreviousNeuron][CurrentNeuron]
     * 
     * featureMatrix → X
     * 
     * weights[0] → X → H1
     * weights[1] → H1 → H2
     * weights[2] → H2 → Output (with 2 hidden layers)
     * 
     * bias[0] → H1
     * bias[1] → H2
     * bias[2] → Output
     */
    double[][][] weights;
    double[][] bias;

    int inputSize; // Corresponds to feature length of each sample
    int[] hiddenSizes; // {26, 27, 27}
    int outputSize; // Binary output.

    /*
     * Every feature is clipped to [-featureClip, featureClip] before it goes into the network
     */
    double featureClip;

    public NeuralNetwork(double[][] featureMatrix, int[] truthValues, int[] hiddenSizes) {

        this(featureMatrix, truthValues, hiddenSizes, 0.0);
    }

    public NeuralNetwork(double[][] featureMatrix, int[] truthValues, int[] hiddenSizes, double featureClip) {

        // ensure values are valid
        if (featureMatrix == null || truthValues == null || featureMatrix.length == 0
                || featureMatrix.length != truthValues.length) {
            throw new IllegalArgumentException("featureMatrix and truthValues must be non-empty and the same length");
        }
        if (hiddenSizes == null || hiddenSizes.length == 0) {
            throw new IllegalArgumentException("at least one hidden layer is needed");
        }
        if (featureClip < 0.0) {
            throw new IllegalArgumentException("featureClip can not be negative (0 = no clipping)");
        }

        this.featureMatrix = featureMatrix;
        this.truthValues = truthValues;
        this.inputSize = featureMatrix[0].length;
        this.hiddenSizes = hiddenSizes;
        this.outputSize = 1;
        this.featureClip = featureClip;

        initWeights();
    }

    private void initWeights() {
        int numLayers = hiddenSizes.length + 1;
        weights = new double[numLayers][][];
        bias = new double[numLayers][];

        Random rand = new Random(RANDOM_SEED);
        int prevSize = inputSize;

        for (int layer = 0; layer < numLayers; layer++) {
            int currentLayerNeuronsSize = neuronsInLayer(layer);

            weights[layer] = new double[prevSize][currentLayerNeuronsSize];
            bias[layer] = new double[currentLayerNeuronsSize]; // biases start at 0

            // He initialisation, since the hidden layers use ReLU
            double initScale = Math.sqrt(2.0 / prevSize);
            for (int prevNeuron = 0; prevNeuron < prevSize; prevNeuron++) {
                for (int currNeuron = 0; currNeuron < currentLayerNeuronsSize; currNeuron++) {
                    weights[layer][prevNeuron][currNeuron] = rand.nextGaussian() * initScale;
                }
            }

            // updating the pointers for prev and curr for the next layer...
            prevSize = currentLayerNeuronsSize;
        }
    }

    /**
     * Number of neurons of a layer: the hidden sizes first, the last layer is the
     * output layer.
     */
    private int neuronsInLayer(int layer) {
        return layer < hiddenSizes.length ? hiddenSizes[layer] : outputSize;
    }

    /**
     * Trains the network with plain mini-batch gradient descent.
     * 
     * If the validation data is given, the weights from the epoch with the lowest
     * validation loss are kept, and the training stops early when it did not
     * improve for 'patience' epochs.
     *
     * @param nEpochs          maximum passes over the training data
     * @param learningRate     step size for the gradient update
     * @param batchSize        samples used for each weight update
     * @param l2Lambda         L2 regularisation (0 = none)
     * @param dropoutRate      chance that a hidden neuron is switched off for a
     *                         sample while training (0 = none, 0.2 = 20%). Never
     *                         used when predicting.
     * @param useClassWeights  same idea as in the Logistic Regression
     * @param validationMatrix validation features, null = no early stopping
     * @param validationTruth  validation truth values
     * @param patience         epochs without improvement before stopping (0 =
     *                         never stop early)
     */
    public void train(int nEpochs, double learningRate, int batchSize, double l2Lambda, double dropoutRate,
            boolean useClassWeights, double[][] validationMatrix, int[] validationTruth, int patience) {

        train(nEpochs, learningRate, batchSize, l2Lambda, dropoutRate, useClassWeights, validationMatrix,
                validationTruth, patience, null);
    }

    /**
     * Same as above, but also records {trainLoss, validationLoss} into
     * lossHistoryOut for every epoch (validationLoss is left out, array of size 1,
     * when there is no validation data), so it can be charted afterwards. Pass
     * null when the loss history is not needed, same as the overload above.
     */
    public void train(int nEpochs, double learningRate, int batchSize, double l2Lambda, double dropoutRate,
            boolean useClassWeights, double[][] validationMatrix, int[] validationTruth, int patience,
            List<double[]> lossHistoryOut) {

        int nSample = featureMatrix.length;
        int numLayers = hiddenSizes.length + 1;

        boolean useValidation = validationMatrix != null && validationTruth != null && validationMatrix.length > 0;

        if (useValidation && validationMatrix.length != validationTruth.length) {
            throw new IllegalArgumentException("validationMatrix and validationTruth must be the same length");
        }

        if (dropoutRate < 0.0 || dropoutRate >= 1.0) {
            throw new IllegalArgumentException("dropoutRate must be from 0 up to (not including) 1");
        }

        // the clipped copies, the original matrices stay untouched
        double[][] trainMatrix = clipAll(featureMatrix);
        double[][] validationClipped = useValidation ? clipAll(validationMatrix) : null;

        // using the same class weights as the Logistic Regression
        double[] sampleWeights = LogisticRegression.calcSampleWeights(truthValues, useClassWeights);
        double[] validationWeights = useValidation
                ? LogisticRegression.calcSampleWeights(validationTruth, useClassWeights)
                : null;

        // gradients of the current batch, same shape as the weights and bias
        double[][][] weightGradients = new double[numLayers][][];
        double[][] biasGradients = new double[numLayers][];

        for (int layer = 0; layer < numLayers; layer++) {
            weightGradients[layer] = new double[weights[layer].length][weights[layer][0].length];
            biasGradients[layer] = new double[bias[layer].length];
        }

        // buffers that are reused for every sample
        double[][] activations = new double[numLayers + 1][];
        double[][] errors = new double[numLayers][];
        for (int layer = 0; layer < numLayers; layer++) {
            activations[layer + 1] = new double[neuronsInLayer(layer)];
            errors[layer] = new double[neuronsInLayer(layer)];
        }

        // the order of the samples, shuffled before each epoch
        int[] order = new int[nSample];
        for (int i = 0; i < nSample; i++) {
            order[i] = i;
        }
        Random random = new Random(RANDOM_SEED);

        // dropout is only for the hidden layers, and only while training
        Random dropoutRandom = dropoutRate > 0.0 ? new Random(RANDOM_SEED + 1) : null;
        double keepScale = dropoutRate > 0.0 ? 1.0 / (1.0 - dropoutRate) : 1.0;

        // for the early stopping
        double bestValidationLoss = Double.POSITIVE_INFINITY;
        int bestEpoch = 0;
        int epochsWithoutImprovement = 0;
        double[][][] bestWeights = null;
        double[][] bestBias = null;

        for (int epoch = 1; epoch <= nEpochs; epoch++) {

            shuffle(order, random);
            double loss = 0.0;

            for (int start = 0; start < nSample; start += batchSize) {

                int end = Math.min(start + batchSize, nSample);

                clearGradients(weightGradients, biasGradients);

                for (int position = start; position < end; position++) {

                    int sample = order[position];
                    double prediction = forwardProp(trainMatrix[sample], activations, dropoutRandom, dropoutRate);

                    loss += sampleWeights[sample] * calculateLoss(prediction, truthValues[sample]);

                    backProp(activations, truthValues[sample], sampleWeights[sample], errors, weightGradients,
                            biasGradients, keepScale);
                }

                updateWeights(weightGradients, biasGradients, end - start, learningRate, l2Lambda);
            }

            loss /= nSample;

            // verbose : showing that the loss is going down the epochs
            if (!useValidation) {
                System.out.println(String.format("  Epoch %d / %d | Loss: %.6f", epoch, nEpochs, loss));
                if (lossHistoryOut != null) {
                    lossHistoryOut.add(new double[] { loss });
                }
                continue;
            }

            double validationLoss = calculateAverageLoss(validationClipped, validationTruth, validationWeights);
            System.out.println(String.format("  Epoch %d / %d | Loss: %.6f | Validation loss: %.6f", epoch, nEpochs,
                    loss, validationLoss));

            if (lossHistoryOut != null) {
                lossHistoryOut.add(new double[] { loss, validationLoss });
            }

            if (validationLoss < bestValidationLoss) {
                bestValidationLoss = validationLoss;
                bestEpoch = epoch;
                epochsWithoutImprovement = 0;
                bestWeights = copyOf(weights);
                bestBias = copyOf(bias);
            } else {
                epochsWithoutImprovement++;

                if (patience > 0 && epochsWithoutImprovement >= patience) {
                    System.out.println("  Early stopping: no improvement for " + patience + " epochs");
                    break;
                }
            }
        }

        // going back to the best epoch, the last one can already be over-fitted
        if (useValidation && bestWeights != null) {
            weights = bestWeights;
            bias = bestBias;
            System.out.println(String.format("  Using the weights of epoch %d (validation loss %.6f)", bestEpoch,
                    bestValidationLoss));
        }
    }

    /*
     * The scores are calculated by multiplying the values (results from the
     * previous layer) by their weights, following the Logistic Regression.
     * 
     * I care about the 'activations' from each network layer to the next, so all of
     * them are kept, backProp needs them. activations[0] is the sample itself and
     * activations[layer + 1] is the output of the layer.
     * 
     * The hidden layers use relu and the output layer uses sigmoid (0/1).
     * 
     * Dropout (only when dropoutRandom is given, so only while training): each
     * hidden neuron is switched off (0) with the chance dropoutRate. The neurons
     * that
     * stay are multiplied by 1 / (1 - dropoutRate), so the size of the scores stays
     * the same and nothing has to change when predicting.
     */
    private double forwardProp(double[] sample, double[][] activations, Random dropoutRandom, double dropoutRate) {

        activations[0] = sample;
        double keepScale = dropoutRandom != null ? 1.0 / (1.0 - dropoutRate) : 1.0;
        int numLayers = hiddenSizes.length + 1;

        // looping through for all the layers
        for (int layerN = 0; layerN < numLayers; layerN++) {

            // X[,...]  -> actuvascore = (X* we)
            // size for the destination neuron
            int currentNeuronSize = weights[layerN][0].length;

            // I want to distinguish output layer to be handled differently
            boolean isOutputLayer = (layerN == numLayers - 1);

            for (int neuron = 0; neuron < currentNeuronSize; neuron++) {

                // get the bias using the 'layer of the network' And 'which neuron of the
                // network'
                double zScore = bias[layerN][neuron];

                for (int a = 0; a < activations[layerN].length; a++) {
                    zScore += activations[layerN][a] * weights[layerN][a][neuron];
                }

                if (isOutputLayer) {
                    activations[layerN + 1][neuron] = sigmoid(zScore);
                } else if (dropoutRandom != null && dropoutRandom.nextDouble() < dropoutRate) {
                    activations[layerN + 1][neuron] = 0.0; // dropped for this sample
                } else {
                    activations[layerN + 1][neuron] = relu(zScore) * keepScale;
                }
            }
        }

        return activations[numLayers][0];
    }

    // no dropout: for predicting and for the validation loss
    private double forwardProp(double[] sample, double[][] activations) {

        return forwardProp(sample, activations, null, 0.0);
    }

    /*
     * Back propagation of one sample, it adds the gradients of the sample to the
     * gradients of the batch.
     * 
     * For a sigmoid output with the binary cross entropy loss, the error of the
     * output neuron is just (prediction - truth) (times the class weight, like in
     * the Logistic Regression). Going back, the error of a neuron is the sum of the
     * errors of the next layer times the weights, and it is 0 if the relu of that
     * neuron was not active (or it was dropped). The neurons that stayed after
     * dropout were scaled by keepScale, so their errors are scaled the same way.
     */
    private void backProp(double[][] activations, int truth, double sampleWeight, double[][] errors,
            double[][][] weightGradients, double[][] biasGradients, double keepScale) {

        int lastLayer = hiddenSizes.length;
        double prediction = activations[lastLayer + 1][0];

        errors[lastLayer][0] = (prediction - truth) * sampleWeight;

        for (int layer = lastLayer; layer >= 0; layer--) {

            for (int prevNeuron = 0; prevNeuron < activations[layer].length; prevNeuron++) {
                for (int currNeuron = 0; currNeuron < errors[layer].length; currNeuron++) {
                    weightGradients[layer][prevNeuron][currNeuron] += activations[layer][prevNeuron]
                            * errors[layer][currNeuron];
                }
            }

            for (int currNeuron = 0; currNeuron < errors[layer].length; currNeuron++) {
                biasGradients[layer][currNeuron] += errors[layer][currNeuron];
            }

            // the errors of the layer before (the input has no errors)
            if (layer > 0) {
                for (int prevNeuron = 0; prevNeuron < activations[layer].length; prevNeuron++) {

                    double sum = 0.0;
                    for (int currNeuron = 0; currNeuron < errors[layer].length; currNeuron++) {
                        sum += weights[layer][prevNeuron][currNeuron] * errors[layer][currNeuron];
                    }

                    errors[layer - 1][prevNeuron] = activations[layer][prevNeuron] > 0.0 ? sum * keepScale : 0.0;
                }
            }
        }
    }

    /**
     * Updates the weights and biases with plain gradient descent, using the
     * averaged gradients of the batch. The L2 penalty is only added to the
     * weights, never to the bias.
     */
    private void updateWeights(double[][][] weightGradients, double[][] biasGradients, int batchSize,
            double learningRate, double l2Lambda) {

        for (int layer = 0; layer < weights.length; layer++) {

            for (int prevNeuron = 0; prevNeuron < weights[layer].length; prevNeuron++) {
                for (int currNeuron = 0; currNeuron < weights[layer][prevNeuron].length; currNeuron++) {

                    double gradient = weightGradients[layer][prevNeuron][currNeuron] / batchSize
                            + l2Lambda * weights[layer][prevNeuron][currNeuron];

                    weights[layer][prevNeuron][currNeuron] -= learningRate * gradient;
                }
            }

            for (int currNeuron = 0; currNeuron < bias[layer].length; currNeuron++) {

                double gradient = biasGradients[layer][currNeuron] / batchSize;

                bias[layer][currNeuron] -= learningRate * gradient;
            }
        }
    }

    /**
     * Calculates the Binary Cross entropy loss
     *
     * @param prediction predicted value
     * @param truthValue truth value
     * @return Loss between prediction and truth value.
     */
    private double calculateLoss(double prediction, double truthValue) {
        // returning the Binary Cross Entropy loss, Also avoiding log(0)
        double eps = 1e-12;
        double pred = Math.min(Math.max(prediction, eps), 1 - eps);
        return -(truthValue * Math.log(pred) + (1 - truthValue) * Math.log(1 - pred));
    }

    /**
     * Average (weighted) loss of a dataset, without changing the weights.
     */
    private double calculateAverageLoss(double[][] matrix, int[] truth, double[] sampleWeights) {

        double[][] activations = newActivations();
        double loss = 0.0;

        for (int sample = 0; sample < matrix.length; sample++) {
            double prediction = forwardProp(matrix[sample], activations);
            loss += sampleWeights[sample] * calculateLoss(prediction, truth[sample]);
        }

        return loss / matrix.length;
    }

    /**
     * Getting the probabilities of class 1 (change) for a batch of samples.
     * 
     * @param matrix features (standardised like the training data)
     * @return probability of each sample
     */
    public double[] predictProbabilities(double[][] matrix) {

        double[][] activations = newActivations();
        double[] probabilities = new double[matrix.length];

        for (int sample = 0; sample < matrix.length; sample++) {

            // sample check
            if (matrix[sample].length != inputSize) {
                throw new IllegalArgumentException("Sample " + sample + " has " + matrix[sample].length
                        + " features, but the model has " + inputSize);
            }

            probabilities[sample] = forwardProp(clip(matrix[sample]), activations);
        }

        return probabilities;
    }

    /**
     * Predicts a single sample (used for inference on a new pair).
     *
     * @param features  features of the pair
     * @param threshold probability from which the sample is a change
     * @return 0 = no change, 1 = change
     */
    public int predict(double[] features, double threshold) {

        return LogisticRegression.predict(predictProbabilities(new double[][] { features }), threshold)[0];
    }

    /**
     * For printing the network, for example: 267 -> 64 -> 32 -> 1
     */
    public String describe() {

        StringBuilder text = new StringBuilder(String.valueOf(inputSize));

        for (int layer = 0; layer < hiddenSizes.length + 1; layer++) {
            text.append(" -> ").append(neuronsInLayer(layer));
        }

        return text.toString();
    }

    // the array for the outputs of each layer, forwardProp fills it
    private double[][] newActivations() {

        int numLayers = hiddenSizes.length + 1;
        double[][] activations = new double[numLayers + 1][];

        for (int layer = 0; layer < numLayers; layer++) {
            activations[layer + 1] = new double[neuronsInLayer(layer)];
        }

        return activations;
    }

    /**
     * Clips every feature of a sample to [-featureClip, featureClip] (0 = no
     * clipping). Gives a new array, the original stays untouched.
     */
    private double[] clip(double[] sample) {

        if (featureClip <= 0.0) {
            return sample;
        }

        double[] clipped = new double[sample.length];

        for (int feature = 0; feature < sample.length; feature++) {
            clipped[feature] = Math.max(-featureClip, Math.min(featureClip, sample[feature]));
        }

        return clipped;
    }

    private double[][] clipAll(double[][] matrix) {

        if (featureClip <= 0.0) {
            return matrix;
        }

        double[][] clipped = new double[matrix.length][];

        for (int sample = 0; sample < matrix.length; sample++) {
            clipped[sample] = clip(matrix[sample]);
        }

        return clipped;
    }

    private static void clearGradients(double[][][] weightGradients, double[][] biasGradients) {

        for (int layer = 0; layer < weightGradients.length; layer++) {
            for (double[] row : weightGradients[layer]) {
                Arrays.fill(row, 0.0);
            }
            Arrays.fill(biasGradients[layer], 0.0);
        }
    }

    private static void shuffle(int[] array, Random random) {

        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    /**
     * Deep copy for the weights
     * @param source
     * @return
     */
    private static double[][][] copyOf(double[][][] source) {

        double[][][] copy = new double[source.length][][];

        for (int layer = 0; layer < source.length; layer++) {
            copy[layer] = new double[source[layer].length][];

            for (int row = 0; row < source[layer].length; row++) {
                copy[layer][row] = source[layer][row].clone();
            }
        }

        return copy;
    }

    /**
     * Deep copy method
     * 
     * @param source
     * @return
     */
    private static double[][] copyOf(double[][] source) {

        double[][] copy = new double[source.length][];

        for (int layer = 0; layer < source.length; layer++) {
            copy[layer] = source[layer].clone();
        }

        return copy;
    }

    private double relu(double zScore) {
        return Math.max(0, zScore);
    }
}