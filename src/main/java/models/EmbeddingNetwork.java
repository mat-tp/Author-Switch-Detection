package models;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import data.RetrievalData;

/**
 * The retrieval model: a small neural network that turns the features of ONE sentence into an
 * embedding (a unit vector), such that sentences of the same author are close and sentences of
 * different authors are far apart. It is the same kind of network as NeuralNetwork (fully connected
 * layers with relu, the weights as plain arrays, one Adam step per batch), only the output is a
 * vector of embeddingSize numbers that is divided by its length instead of one sigmoid.
 *
 * input -> hidden layers (relu) -> embeddingSize numbers z -> e = z / |z|
 *
 * Why unit vectors: the squared distance of two of them is 2 - 2 cos, so the distance only measures
 * the angle and the network can not make everything "far" by making the numbers big.
 *
 * Training with the triplet loss (Schroff et al., 2015; the metric learning way of "put similar
 * together and further apart"):
 *
 * loss(a, p, n) = max(0, margin + |e_a - e_p|^2 - |e_a - e_n|^2)
 *
 * a = the anchor sentence, p = a sentence of the SAME segment (the same author, for sure),
 * n = a sentence of a NEXT segment (a different author, for sure: a change lies between). The
 * loss is 0 when the positive is closer than the negative by at least the margin, so only the
 * triplets that are still wrong or too close give a gradient.
 *
 * Mining (hardCandidates > 1): for every triplet, hardCandidates negatives are drawn and the one that is closest
 * to the anchor with the current weights is used (hard negative mining). Easy negatives teach nothing
 * after a few epochs, the hard ones are the sentences the network still confuses with the anchor.
 */
public class EmbeddingNetwork implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final long RANDOM_SEED = 42L;

    // smaller batches are calculated by one thread, the threads cost more than they save
    private static final int MIN_TRIPLETS_PER_THREAD = 16;

    /*
     * weights[layer][inputNeuron][outputNeuron], bias[layer][outputNeuron]
     * The last layer has no relu, its output z is divided by |z|.
     */
    double[][][] weights;
    double[][] bias;

    int inputSize;
    int[] hiddenSizes;
    int embeddingSize;

    transient int threads = Math.max(1, Integer.getInteger("threads", Math.min(8, Runtime.getRuntime().availableProcessors())));

    public EmbeddingNetwork(int inputSize, int[] hiddenSizes, int embeddingSize) {

        if (inputSize < 1 || embeddingSize < 2) {
            throw new IllegalArgumentException("inputSize must be >= 1 and embeddingSize >= 2");
        }

        this.inputSize = inputSize;
        this.hiddenSizes = hiddenSizes == null ? new int[0] : hiddenSizes.clone();
        this.embeddingSize = embeddingSize;

        initWeights();
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getEmbeddingSize() {
        return embeddingSize;
    }

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

    // He initialisation, the weights of a relu layer have the variance 2 / inputs
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

    // ------------------------------------------------------------------
    // Forward
    // ------------------------------------------------------------------

    /** What backProp needs from the forward pass of one sentence. */
    private static class Pass {
        double[][] activations; // [0] = input, [l + 1] = output of layer l after relu (the last one: z)
        double norm; // |z|
        double[] embedding; // z / |z|
    }

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
                    continue; // a switched off relu, adds nothing
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

    /** The embedding (length 1) of one sentence. */
    public double[] embed(float[] input) {
        return forward(input).embedding;
    }

    /** The embeddings of many sentences. */
    public double[][] embedAll(float[][] inputs) {

        double[][] result = new double[inputs.length][];
        IntStream.range(0, inputs.length).parallel().forEach(i -> result[i] = forward(inputs[i]).embedding);
        return result;
    }

    // ------------------------------------------------------------------
    // Backward
    // ------------------------------------------------------------------

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
     * Adds the gradient of one sentence to the sums, given the gradient of the loss with respect to
     * its embedding e.
     *
     * e = z / |z|  gives  dz = (dE - (dE . e) e) / |z|  (the part of dE that is along e does not change e)
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
                    // relu: the neuron passes the gradient only when it was on
                    previousDelta[i] = input[i] > 0.0 ? sum : 0.0;
                }
            }
            delta = previousDelta;
        }
    }

    /** loss of a triplet from the three passes: max(0, margin + |a-p|^2 - |a-n|^2) */
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

    /** the loss of one triplet and its gradient added to the sums (nothing is added when the loss is 0) */
    private double tripletStep(float[] a, float[] p, float[] n, double margin, double[][][] weightGradients,
            double[][] biasGradients) {

        Pass pa = forward(a);
        Pass pp = forward(p);
        Pass pn = forward(n);

        double loss = tripletLoss(pa, pp, pn, margin);
        if (loss <= 0.0) {
            return 0.0;
        }

        // d loss / d e_a = 2 (e_n - e_p), d / d e_p = -2 e_a + ... see the formula in the class comment
        // |a-p|^2 = 2 - 2 a.p  and  |a-n|^2 = 2 - 2 a.n  ->  loss = margin - 2 a.p + 2 a.n
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

    // ------------------------------------------------------------------
    // Triplets
    // ------------------------------------------------------------------

    /**
     * Draws triplets (anchor, positive, negative) as sentence numbers of the data.
     *
     * Only the facts that the labels give are used: the positive is another sentence of the anchor's
     * segment, the negative is a sentence of the segment BEFORE or AFTER the anchor's segment.
     * Sentences of far away segments, or of other documents, can have the same author (the dataset
     * does not tell), so they are no negatives, unless crossDocumentRate > 0 asks for it
     * (an ablation setting: for datasets where documents never share an author).
     *
     * @param count             number of triplets
     * @param candidates        negatives drawn per triplet (the closest to the anchor is kept by train(...))
     * @param crossDocumentRate share of the negatives that come from a random other document
     * @return int[count][2 + candidates] = {anchor, positive, negative 1, negative 2, ...}
     */
    public static int[][] drawTriplets(RetrievalData data, int count, int candidates, double crossDocumentRate,
            long seed) {

        Random random = new Random(seed);

        // the anchors: a segment with at least 2 sentences, and a next segment in the document
        int[] anchors = new int[data.size()];
        int nAnchors = 0;
        for (int s = 0; s < data.size(); s++) {
            int segment = data.segment[s];
            if (data.segmentLength(segment) < 2) {
                continue;
            }
            if (hasNeighbour(data, segment)) {
                anchors[nAnchors++] = s;
            }
        }

        if (nAnchors == 0) {
            return new int[0][];
        }

        int[][] triplets = new int[count][2 + Math.max(1, candidates)];
        for (int t = 0; t < count; t++) {

            int anchor = anchors[random.nextInt(nAnchors)];
            int segment = data.segment[anchor];

            int positive;
            do {
                positive = data.segStart[segment] + random.nextInt(data.segmentLength(segment));
            } while (positive == anchor);

            triplets[t][0] = anchor;
            triplets[t][1] = positive;

            for (int c = 0; c < Math.max(1, candidates); c++) {
                triplets[t][2 + c] = (crossDocumentRate > 0.0 && random.nextDouble() < crossDocumentRate
                        && data.documents() > 1) ? randomOtherDocumentSentence(data, anchor, random)
                                : randomNeighbourSentence(data, segment, random);
            }
        }
        return triplets;
    }

    private static boolean hasNeighbour(RetrievalData data, int segment) {

        int doc = data.segDoc[segment];
        boolean before = segment > 0 && data.segDoc[segment - 1] == doc;
        boolean after = segment + 1 < data.segments() && data.segDoc[segment + 1] == doc;
        return before || after;
    }

    private static int randomNeighbourSentence(RetrievalData data, int segment, Random random) {

        int doc = data.segDoc[segment];
        boolean before = segment > 0 && data.segDoc[segment - 1] == doc;
        boolean after = segment + 1 < data.segments() && data.segDoc[segment + 1] == doc;

        int neighbour;
        if (before && after) {
            neighbour = random.nextBoolean() ? segment - 1 : segment + 1;
        } else {
            neighbour = before ? segment - 1 : segment + 1;
        }
        return data.segStart[neighbour] + random.nextInt(data.segmentLength(neighbour));
    }

    private static int randomOtherDocumentSentence(RetrievalData data, int anchor, Random random) {

        int doc;
        do {
            doc = random.nextInt(data.documents());
        } while (doc == data.doc[anchor]);
        int size = data.docStart[doc + 1] - data.docStart[doc];
        return data.docStart[doc] + random.nextInt(size);
    }

    /** Mean triplet loss and the share of triplets with sim(a,p) > sim(a,n), for the first negative of every triplet. */
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

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    /**
     * Trains the network with Adam on freshly drawn triplets every epoch.
     *
     * @param train             the training sentences
     * @param validation        the validation sentences (fixed triplets, early stopping), may be null
     * @param epochs            maximum number of epochs
     * @param tripletsPerEpoch  triplets drawn per epoch
     * @param batchSize         triplets per Adam step
     * @param learningRate      0.001 is a good value for Adam
     * @param margin            the margin of the loss (squared distances of unit vectors are in [0, 4]); 0.5 to 1
     * @param l2Lambda          weight decay, 0 = none
     * @param candidates        negatives per triplet, the hardest is used (1 = random negative)
     * @param crossDocumentRate share of negatives from other documents (see drawTriplets)
     * @param patience          epochs without a better validation loss before it stops, 0 = never
     * @param lossHistory       receives {training loss, validation loss} per epoch (may be null)
     */
    public void train(RetrievalData train, RetrievalData validation, int epochs, int tripletsPerEpoch, int batchSize,
            double learningRate, double margin, double l2Lambda, int candidates, double crossDocumentRate,
            int patience, java.util.List<double[]> lossHistory) {

        int[][] validationTriplets = validation == null ? null
                : drawTriplets(validation, Math.max(500, tripletsPerEpoch / 4), 1, crossDocumentRate, RANDOM_SEED + 99);

        double[][][] mean = newGradients(), variance = newGradients();
        double[][] biasMean = newBiasGradients(), biasVariance = newBiasGradients();
        long adamStep = 0;

        double bestValidationLoss = Double.MAX_VALUE;
        double[][][] bestWeights = null;
        double[][] bestBias = null;
        int bestEpoch = 0, withoutImprovement = 0;

        for (int epoch = 1; epoch <= epochs; epoch++) {

            int[][] triplets = drawTriplets(train, tripletsPerEpoch, candidates, crossDocumentRate,
                    RANDOM_SEED + epoch);
            if (triplets.length == 0) {
                System.out.println("  No triplets can be drawn (no segment with 2 sentences next to another segment).");
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

                // adding the parts, always in the same order
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

    // the candidate negative that is closest to the anchor with the current weights (the only one: no work)
    private int hardest(RetrievalData data, int[] triplet) {

        if (triplet.length == 3) {
            return triplet[2];
        }

        double[] anchor = forward(data.x[triplet[0]]).embedding;
        int best = triplet[2];
        double bestSimilarity = -2.0;
        for (int c = 2; c < triplet.length; c++) {
            double[] e = forward(data.x[triplet[c]]).embedding;
            double similarity = 0.0;
            for (int j = 0; j < embeddingSize; j++) {
                similarity += anchor[j] * e[j];
            }
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
                    mean[layer][i][j] = beta1 * mean[layer][i][j] + (1 - beta1) * g;
                    variance[layer][i][j] = beta2 * variance[layer][i][j] + (1 - beta2) * g * g;
                    weights[layer][i][j] -= learningRate * (mean[layer][i][j] / correction1)
                            / (Math.sqrt(variance[layer][i][j] / correction2) + epsilon);
                }
            }
            for (int j = 0; j < bias[layer].length; j++) {
                double g = biasGradients[layer][j] / batchSize;
                biasMean[layer][j] = beta1 * biasMean[layer][j] + (1 - beta1) * g;
                biasVariance[layer][j] = beta2 * biasVariance[layer][j] + (1 - beta2) * g * g;
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

    // ------------------------------------------------------------------
    // Gradient check
    // ------------------------------------------------------------------

    /**
     * Compares the gradient of backProp with the numerical one (central differences) for a triplet, the
     * way NeuralNetwork.maxGradientError does. The margin is made large (4 is more than any loss
     * can reach) so the triplet is surely active.
     *
     * @return the largest relative error of the checked weights (about 1e-6 is right, 1e-2 is a bug)
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

            double numeric = (up - down) / (2 * h);
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
