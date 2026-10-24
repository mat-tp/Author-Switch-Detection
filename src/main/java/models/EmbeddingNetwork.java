package models;

import data.TripletSampler;
import util.VectorMath;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import data.RetrievalData;

/**
 * Dense neural network model for learning normalized unit-vector sentence embeddings.
 *
 * <p>Maps single-sentence feature representations into an $L_2$-normalized embedding space
 * where sentences authored by the same writer cluster closely together, and sentences by
 * different writers are separated by a margin.</p>
 *
 * <p><strong>Architecture Pipeline:</strong>
 * <pre>
 * input &rarr; hidden layers (ReLU) &rarr; embeddingSize outputs z &rarr; e = z / ||z||_2
 * </pre>
 * </p>
 *
 * <p><strong>Unit Length Invariant:</strong> Normalizing outputs onto a unit hypersphere bounds
 * squared Euclidean distance ($\|e_a - e_b\|^2 = 2 - 2 \cos \theta$). This restricts distance to
 * direction angles and prevents weight explosions.</p>
 *
 * <p><strong>Triplet Loss Optimization (Schroff et al., 2015):</strong>
 * <pre>
 * loss(a, p, n) = max(0, margin + ||e_a - e_p||^2 - ||e_a - e_n||^2)
 * </pre>
 * <ul>
 *   <li>{@code a} (Anchor): Target sentence.</li>
 *   <li>{@code p} (Positive): Sentence from the <em>same</em> segment (guaranteed same author).</li>
 *   <li>{@code n} (Negative): Sentence from an <em>adjacent</em> segment (guaranteed different author).</li>
 * </ul>
 * Loss evaluates to zero once the positive sample is closer than the negative sample by at least the margin.
 * </p>
 *
 * <p><strong>Hard Negative Mining:</strong> When {@code hardCandidates > 1}, multiple negative candidate
 * sentences are drawn per triplet, selecting the candidate nearest to the anchor under current weights.</p>
 */
public class EmbeddingNetwork implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final long RANDOM_SEED = 42L;

    /** Minimum batch subset allocated per thread to avoid multi-threading synchronization overhead. */
    private static final int MIN_TRIPLETS_PER_THREAD = 16;

    /** Layer weights array indexed by {@code [layer][inputNeuron][outputNeuron]}. */
    double[][][] weights;

    /** Layer biases array indexed by {@code [layer][outputNeuron]}. */
    double[][] bias;

    int inputSize;
    int[] hiddenSizes;
    int embeddingSize;

    transient int threads = Math.max(1, Integer.getInteger("threads", Math.min(8, Runtime.getRuntime().availableProcessors())));

    /**
     * Constructs an untrained {@code EmbeddingNetwork} instance and initializes weights using He normal distribution.
     *
     * @param inputSize Feature dimension size for input sentences ($\ge 1$).
     * @param hiddenSizes Array specifying hidden layer neuron capacities (or {@code null}/empty for direct linear projection).
     * @param embeddingSize Dimension size of the output normalized embedding ($\ge 2$).
     * @throws IllegalArgumentException If {@code inputSize < 1} or {@code embeddingSize < 2}.
     */
    public EmbeddingNetwork(int inputSize, int[] hiddenSizes, int embeddingSize) {

        if (inputSize < 1 || embeddingSize < 2) {
            throw new IllegalArgumentException("inputSize must be >= 1 and embeddingSize >= 2");
        }

        this.inputSize = inputSize;
        this.hiddenSizes = hiddenSizes == null ? new int[0] : hiddenSizes.clone();
        this.embeddingSize = embeddingSize;

        initWeights();
    }

    /**
     * Returns the configured input feature vector size.
     *
     * @return Input feature dimension.
     */
    public int getInputSize() {
        return inputSize;
    }

    /**
     * Returns the output normalized unit embedding vector size.
     *
     * @return Output embedding dimension.
     */
    public int getEmbeddingSize() {
        return embeddingSize;
    }

    /**
     * Sets the maximum parallel thread count allocated for parallel forward/backward batch calculations.
     *
     * @param threads Thread count ($\ge 1$).
     */
    public void setThreads(int threads) {
        this.threads = Math.max(1, threads);
    }

    private int layerCount() {
        return hiddenSizes.length + 1;
    }

    private int layerInput(int layer) {
        return layer == 0 ? inputSize : hiddenSizes[layer - 1];
    }

    private int layerOutput(int layer) {
        return layer == hiddenSizes.length ? embeddingSize : hiddenSizes[layer];
    }

    /**
     * Initializes network parameters via He initialization ($\text{variance} = 2 / \text{fan\_in}$).
     */
    private void initWeights() {

        int layers = layerCount();
        weights = new double[layers][][];
        bias = new double[layers][];
        Random rand = new Random(RANDOM_SEED);

        for (int layer = 0; layer < layers; layer++) {
            int in = layerInput(layer);
            int out = layerOutput(layer);
            double scale = Math.sqrt(2.0 / in);

            weights[layer] = new double[in][out];
            bias[layer] = new double[out];
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    weights[layer][i][j] = rand.nextGaussian() * scale;
                }
            }
        }
    }

    /** Internal forward pass tracking activation vectors and norm factors needed for backpropagation. */
    private static class Pass {
        double[][] activations;
        double norm;
        double[] embedding;
    }

    /**
     * Evaluates full forward activation pass for a single sentence feature vector.
     */
    private Pass forward(float[] input) {

        int layers = layerCount();
        Pass pass = new Pass();
        pass.activations = new double[layers + 1][];

        double[] current = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            current[i] = input[i];
        }
        pass.activations[0] = current;

        for (int layer = 0; layer < layers; layer++) {

            int in = layerInput(layer);
            int out = layerOutput(layer);
            double[] scores = bias[layer].clone();

            for (int i = 0; i < in; i++) {
                double value = current[i];
                if (value == 0.0) {
                    continue; // Inactive ReLU output node
                }
                double[] row = weights[layer][i];
                for (int j = 0; j < out; j++) {
                    scores[j] += value * row[j];
                }
            }

            if (layer < layers - 1) {
                for (int j = 0; j < out; j++) {
                    if (scores[j] < 0.0) {
                        scores[j] = 0.0;
                    }
                }
            }
            pass.activations[layer + 1] = scores;
            current = scores;
        }

        double sum = 0.0;
        for (double v : current) {
            sum += v * v;
        }
        pass.norm = Math.sqrt(sum) + 1e-12;
        pass.embedding = new double[embeddingSize];
        for (int j = 0; j < embeddingSize; j++) {
            pass.embedding[j] = current[j] / pass.norm;
        }
        return pass;
    }

    /**
     * Computes unit-length embedding vector for a single sentence.
     *
     * @param input Raw/standardized sentence feature array.
     * @return $L_2$-normalized double embedding array of length {@link #embeddingSize}.
     */
    public double[] embed(float[] input) {
        return forward(input).embedding;
    }

    /**
     * Computes unit-length embedding vectors in parallel for multiple sentences.
     *
     * @param inputs Matrix of input sentence feature vectors.
     * @return Matrix of normalized double embedding arrays.
     */
    public double[][] embedAll(float[][] inputs) {

        double[][] result = new double[inputs.length][];
        IntStream.range(0, inputs.length).parallel().forEach(i -> result[i] = forward(inputs[i]).embedding);
        return result;
    }

    private double[][][] newGradients() {
        double[][][] g = new double[layerCount()][][];
        for (int layer = 0; layer < g.length; layer++) {
            g[layer] = new double[layerInput(layer)][layerOutput(layer)];
        }
        return g;
    }

    private double[][] newBiasGradients() {
        double[][] g = new double[layerCount()][];
        for (int layer = 0; layer < g.length; layer++) {
            g[layer] = new double[layerOutput(layer)];
        }
        return g;
    }

    /**
     * Computes backpropagation derivatives for a sentence pass given loss gradient with respect to embedding $e$.
     *
     * <p>Derivation: For $e = z / \|z\|$, $dz = (dE - (dE \cdot e) e) / \|z\|$.</p>
     */
    private void backProp(Pass pass, double[] dEmbedding, double[][][] weightGradients, double[][] biasGradients) {

        int layers = layerCount();

        double dot = 0.0;
        for (int j = 0; j < embeddingSize; j++) {
            dot += dEmbedding[j] * pass.embedding[j];
        }
        double[] delta = new double[embeddingSize];
        for (int j = 0; j < embeddingSize; j++) {
            delta[j] = (dEmbedding[j] - dot * pass.embedding[j]) / pass.norm;
        }

        for (int layer = layers - 1; layer >= 0; layer--) {

            int in = layerInput(layer);
            int out = layerOutput(layer);
            double[] input = pass.activations[layer];

            double[] previousDelta = layer > 0 ? new double[in] : null;

            for (int j = 0; j < out; j++) {
                biasGradients[layer][j] += delta[j];
            }
            for (int i = 0; i < in; i++) {
                double value = input[i];
                double[] row = weights[layer][i];
                double[] gradientRow = weightGradients[layer][i];
                double sum = 0.0;
                for (int j = 0; j < out; j++) {
                    gradientRow[j] += value * delta[j];
                    sum += row[j] * delta[j];
                }
                if (previousDelta != null) {
                    previousDelta[i] = input[i] > 0.0 ? sum : 0.0;
                }
            }
            delta = previousDelta;
        }
    }

    /**
     * Evaluates triplet loss value from three sentence passes: $\max(0, \text{margin} + \|a-p\|^2 - \|a-n\|^2)$.
     */
    private static double tripletLoss(Pass a, Pass p, Pass n, double margin) {

        double dp = 0.0, dn = 0.0;
        for (int j = 0; j < a.embedding.length; j++) {
            double x = a.embedding[j] - p.embedding[j];
            double y = a.embedding[j] - n.embedding[j];
            dp += x * x;
            dn += y * y;
        }
        return Math.max(0.0, margin + dp - dn);
    }

    /**
     * Evaluates forward/backward pass for a single triplet and accumulates parameter gradients.
     */
    private double tripletStep(float[] a, float[] p, float[] n, double margin, double[][][] weightGradients,
                               double[][] biasGradients) {

        Pass pa = forward(a);
        Pass pp = forward(p);
        Pass pn = forward(n);

        double loss = tripletLoss(pa, pp, pn, margin);
        if (loss <= 0.0) {
            return 0.0;
        }

        double[] dA = new double[embeddingSize];
        double[] dP = new double[embeddingSize];
        double[] dN = new double[embeddingSize];
        for (int j = 0; j < embeddingSize; j++) {
            dA[j] = 2.0 * (pn.embedding[j] - pp.embedding[j]);
            dP[j] = -2.0 * pa.embedding[j];
            dN[j] = 2.0 * pa.embedding[j];
        }

        backProp(pa, dA, weightGradients, biasGradients);
        backProp(pp, dP, weightGradients, biasGradients);
        backProp(pn, dN, weightGradients, biasGradients);
        return loss;
    }

    /**
     * Evaluates mean loss and rank accuracy ($\text{sim}(a,p) > \text{sim}(a,n)$) across test triplets.
     *
     * @param data Test retrieval dataset container.
     * @param triplets Matrix of triplet sentence indices.
     * @param margin Target loss margin.
     * @return Double array containing {@code [meanLoss, accuracyFraction]}.
     */
    public double[] evaluateTriplets(RetrievalData data, int[][] triplets, double margin) {

        if (triplets.length == 0) {
            return new double[] { 0.0, 0.0 };
        }

        double[] losses = new double[triplets.length];
        boolean[] correct = new boolean[triplets.length];

        IntStream.range(0, triplets.length).parallel().forEach(t -> {
            Pass a = forward(data.x[triplets[t][0]]);
            Pass p = forward(data.x[triplets[t][1]]);
            Pass n = forward(data.x[triplets[t][2]]);
            losses[t] = tripletLoss(a, p, n, margin);
            double sp = 0.0, sn = 0.0;
            for (int j = 0; j < embeddingSize; j++) {
                sp += a.embedding[j] * p.embedding[j];
                sn += a.embedding[j] * n.embedding[j];
            }
            correct[t] = sp > sn;
        });

        double loss = 0.0;
        int right = 0;
        for (int t = 0; t < triplets.length; t++) {
            loss += losses[t];
            if (correct[t]) {
                right++;
            }
        }
        return new double[] { loss / triplets.length, (double) right / triplets.length };
    }

    /**
     * Trains the network parameters using Adam optimization over randomized triplets per epoch.
     *
     * @param train Primary training sentence dataset.
     * @param validation Optional validation dataset for early stopping (or {@code null}).
     * @param epochs Maximum epoch iterations.
     * @param tripletsPerEpoch Number of unique triplets drawn per epoch.
     * @param batchSize Batch size for Adam optimizer updates.
     * @param learningRate Learning rate ($\alpha$, e.g., 0.001).
     * @param margin Triplet loss margin ($\text{margin} \in [0.5, 1.0]$).
     * @param l2Lambda Weight decay penalty parameter ($0.0$ = disabled).
     * @param sampling How triplets are drawn (see {@link TripletSampler.Policy}).
     * @param patience Maximum epoch count allowed without validation loss improvement ($0$ = disabled).
     * @param lossHistory Destination list logging epoch metric pairs {@code [trainLoss, valLoss]}.
     */
    public void train(RetrievalData train, RetrievalData validation, int epochs, int tripletsPerEpoch, int batchSize,
                      double learningRate, double margin, double l2Lambda, TripletSampler.Policy sampling,
                      int patience, java.util.List<double[]> lossHistory) {

        TripletSampler.Policy validationSampling = new TripletSampler.Policy(1, sampling.crossDocumentRate(),
                sampling.useAuthorCounts(), sampling.farPositiveRate());
        int[][] validationTriplets = validation == null ? null
                : TripletSampler.draw(validation, Math.max(500, tripletsPerEpoch / 4), validationSampling,
                RANDOM_SEED + 99);

        double[][][] mean = newGradients(), variance = newGradients();
        double[][] biasMean = newBiasGradients(), biasVariance = newBiasGradients();
        long adamStep = 0;

        double bestValidationLoss = Double.MAX_VALUE;
        double[][][] bestWeights = null;
        double[][] bestBias = null;
        int bestEpoch = 0, withoutImprovement = 0;

        for (int epoch = 1; epoch <= epochs; epoch++) {

            int[][] triplets = TripletSampler.draw(train, tripletsPerEpoch, sampling, RANDOM_SEED + epoch);
            if (triplets.length == 0) {
                System.out.println("  No triplets can be drawn (no sentence has both a positive and a negative).");
                return;
            }

            double loss = 0.0;
            int active = 0;

            for (int batchStart = 0; batchStart < triplets.length; batchStart += batchSize) {

                int size = Math.min(batchSize, triplets.length - batchStart);

                int parts = Math.max(1, Math.min(threads, size / MIN_TRIPLETS_PER_THREAD));

                double[][][][] partWeights = new double[parts][][][];
                double[][][] partBias = new double[parts][][];
                double[] partLoss = new double[parts];
                int[] partActive = new int[parts];

                final int start = batchStart;
                final int[][] tripletsOfEpoch = triplets;

                IntStream.range(0, parts).parallel().forEach(part -> {

                    double[][][] gradients = newGradients();
                    double[][] biasGradients = newBiasGradients();

                    int from = start + (int) ((long) size * part / parts);
                    int to = start + (int) ((long) size * (part + 1) / parts);

                    for (int k = from; k < to; k++) {
                        int[] triplet = tripletsOfEpoch[k];
                        int negative = hardest(train, triplet);
                        double l = tripletStep(train.x[triplet[0]], train.x[triplet[1]], train.x[negative], margin,
                                gradients, biasGradients);
                        partLoss[part] += l;
                        if (l > 0.0) {
                            partActive[part]++;
                        }
                    }
                    partWeights[part] = gradients;
                    partBias[part] = biasGradients;
                });

                for (int part = 1; part < parts; part++) {
                    add(partWeights[0], partBias[0], partWeights[part], partBias[part]);
                }
                for (int part = 0; part < parts; part++) {
                    loss += partLoss[part];
                    active += partActive[part];
                }

                adamStep++;
                adamUpdate(partWeights[0], partBias[0], size, learningRate, l2Lambda, mean, variance, biasMean,
                        biasVariance, adamStep);
            }

            loss /= triplets.length;

            if (validationTriplets == null) {
                System.out.println(String.format(java.util.Locale.ROOT,
                        "  Epoch %d / %d | Loss: %.5f | triplets still wrong or too close: %.1f%%", epoch, epochs, loss,
                        100.0 * active / triplets.length));
                if (lossHistory != null) {
                    lossHistory.add(new double[] { loss });
                }
                continue;
            }

            double[] v = evaluateTriplets(validation, validationTriplets, margin);
            System.out.println(String.format(java.util.Locale.ROOT,
                    "  Epoch %d / %d | Loss: %.5f | Validation loss: %.5f | validation triplets right: %.1f%%", epoch,
                    epochs, loss, v[0], 100.0 * v[1]));
            if (lossHistory != null) {
                lossHistory.add(new double[] { loss, v[0] });
            }

            if (v[0] < bestValidationLoss) {
                bestValidationLoss = v[0];
                bestEpoch = epoch;
                withoutImprovement = 0;
                bestWeights = copyOf(weights);
                bestBias = copyOf(bias);
            } else {
                withoutImprovement++;
                if (patience > 0 && withoutImprovement >= patience) {
                    System.out.println("  Early stopping: no improvement for " + patience + " epochs");
                    break;
                }
            }
        }

        if (bestWeights != null) {
            weights = bestWeights;
            bias = bestBias;
            System.out.println(String.format(java.util.Locale.ROOT,
                    "  Using the weights of epoch %d (validation loss %.5f)", bestEpoch, bestValidationLoss));
        }
    }

    /** Hard negative mining: selects the negative candidate closest to the anchor under current weights. */
    private int hardest(RetrievalData data, int[] triplet) {

        if (triplet.length == 3) {
            return triplet[2];
        }

        double[] anchor = forward(data.x[triplet[0]]).embedding;
        int best = triplet[2];
        double bestSimilarity = -2.0;
        for (int c = 2; c < triplet.length; c++) {
            double[] e = forward(data.x[triplet[c]]).embedding;
            double similarity = VectorMath.dot(anchor, e);
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                best = triplet[c];
            }
        }
        return best;
    }

    private static void add(double[][][] weightSum, double[][] biasSum, double[][][] weightPart, double[][] biasPart) {

        for (int layer = 0; layer < weightSum.length; layer++) {
            for (int i = 0; i < weightSum[layer].length; i++) {
                for (int j = 0; j < weightSum[layer][i].length; j++) {
                    weightSum[layer][i][j] += weightPart[layer][i][j];
                }
            }
            for (int j = 0; j < biasSum[layer].length; j++) {
                biasSum[layer][j] += biasPart[layer][j];
            }
        }
    }

    private void adamUpdate(double[][][] weightGradients, double[][] biasGradients, int batchSize, double learningRate,
                            double l2Lambda, double[][][] mean, double[][][] variance, double[][] biasMean, double[][] biasVariance,
                            long step) {

        final double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
        double correction1 = 1.0 - Math.pow(beta1, step);
        double correction2 = 1.0 - Math.pow(beta2, step);

        for (int layer = 0; layer < weights.length; layer++) {
            for (int i = 0; i < weights[layer].length; i++) {
                for (int j = 0; j < weights[layer][i].length; j++) {
                    double g = weightGradients[layer][i][j] / batchSize + l2Lambda * weights[layer][i][j];
                    mean[layer][i][j] = beta1 * mean[layer][i][j] + (1.0 - beta1) * g;
                    variance[layer][i][j] = beta2 * variance[layer][i][j] + (1.0 - beta2) * g * g;

                    weights[layer][i][j] -= learningRate * (mean[layer][i][j] / correction1)
                            / (Math.sqrt(variance[layer][i][j] / correction2) + epsilon);
                }
            }
            for (int j = 0; j < bias[layer].length; j++) {
                double g = biasGradients[layer][j] / batchSize;
                biasMean[layer][j] = beta1 * biasMean[layer][j] + (1.0 - beta1) * g;
                biasVariance[layer][j] = beta2 * biasVariance[layer][j] + (1.0 - beta2) * g * g;

                bias[layer][j] -= learningRate * (biasMean[layer][j] / correction1)
                        / (Math.sqrt(biasVariance[layer][j] / correction2) + epsilon);
            }
        }
    }

    private static double[][][] copyOf(double[][][] source) {
        double[][][] copy = new double[source.length][][];
        for (int l = 0; l < source.length; l++) {
            copy[l] = new double[source[l].length][];
            for (int i = 0; i < source[l].length; i++) {
                copy[l][i] = source[l][i].clone();
            }
        }
        return copy;
    }

    private static double[][] copyOf(double[][] source) {
        double[][] copy = new double[source.length][];
        for (int l = 0; l < source.length; l++) {
            copy[l] = source[l].clone();
        }
        return copy;
    }

    /**
     * Compares backpropagation analytical weight gradients against finite-difference numerical gradients.
     *
     * @param a Anchor sentence feature vector.
     * @param p Positive sentence feature vector.
     * @param n Negative sentence feature vector.
     * @param maxParams Maximum random parameters to evaluate.
     * @return Maximum relative error observed ($\approx 1\text{e}{-6}$ is normal, $\ge 1\text{e}{-2}$ indicates gradient bug).
     */
    public double maxGradientError(float[] a, float[] p, float[] n, int maxParams) {

        double margin = 4.0;
        double[][][] weightGradients = newGradients();
        double[][] biasGradients = newBiasGradients();
        tripletStep(a, p, n, margin, weightGradients, biasGradients);

        Random random = new Random(RANDOM_SEED);
        double worst = 0.0;
        double h = 1e-5;

        for (int check = 0; check < maxParams; check++) {

            int layer = random.nextInt(weights.length);
            int i = random.nextInt(weights[layer].length);
            int j = random.nextInt(weights[layer][i].length);

            double original = weights[layer][i][j];
            weights[layer][i][j] = original + h;
            double up = tripletLoss(forward(a), forward(p), forward(n), margin);
            weights[layer][i][j] = original - h;
            double down = tripletLoss(forward(a), forward(p), forward(n), margin);
            weights[layer][i][j] = original;

            double numeric = (up - down) / (2.0 * h);
            double analytic = weightGradients[layer][i][j];
            double error = Math.abs(numeric - analytic) / Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
            worst = Math.max(worst, error);
        }
        return worst;
    }

    @Override
    public String toString() {
        return "EmbeddingNetwork(" + inputSize + " -> " + Arrays.toString(hiddenSizes) + " -> " + embeddingSize + ")";
    }
}