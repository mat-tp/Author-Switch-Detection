package com.hyp.authorswitchdetection.ml;

import com.hyp.authorswitchdetection.service.TrainingStatus;
import data.Corpus;
import features.FeatureConfig;
import features.FeatureExtractor;
import models.DecisionTree;
import models.LogisticRegression;
import models.Metrics;
import models.ModelStore;
import models.NeuralNetwork;
import models.RetrievalPipeline;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * The training pipeline of the web app. It is the pipeline of the command line (Main.java),
 * with a small grid search on top: for every model a few settings are tried, and the one with the
 * best validation macro-F1 (the measure PAN 2026 ranks by) is kept.
 *
 * Everything that is not the grid is shared with Main, not copied: the data (data.Corpus), the features
 * (features.FeatureExtractor, settings from features.FeatureConfig), the pair vectors
 * (data.PairMatrixBuilder), the file format (models.ModelStore) and the retrieval model
 * (models.RetrievalPipeline). A model trained here is therefore the same kind of model as one trained
 * by Main, and a change in the features reaches both.
 *
 * Held-out test split: the "validation" split of PAN is divided into two halves (by problem ID, seeded,
 * see DatasetLoader.loadProblems): the first half ("validation") is used for every decision (grid search,
 * early stopping, the probability threshold); the second half ("test") is not touched until the very end
 * and gives the numbers on the Results page. If there is no test half, the validation numbers are
 * reported, and Outcome.evaluatedOnTest says so.
 *
 * The winning model is saved under app.trained-models.dir. MlModelLoader reads that folder, so a
 * freshly trained model is used for prediction immediately, no restart needed.
 */
@Component
public class RealTrainer {

    @Value("${app.dataset.root:dataset/mawsa26-pan-zenodo-DATA/}")
    private String datasetRoot;

    @Value("${app.trained-models.dir:data/models}")
    private String modelsDir;

    // at most this many training pairs (and per difficulty, this many validation / test pairs)
    @Value("${app.train.subset:300000}")
    private int subsetSize;

    /** One finished training run, ready to be turned into a ModelMetrics row. */
    public static class Outcome {
        public String chosenConfig;
        public String featureConfig;
        public double f1;               // F1 of the change class
        public double macroF1;          // the PAN measure
        public double precision;
        public double recall;
        public double balancedAccuracy;
        public double threshold;
        public boolean evaluatedOnTest; // true = the numbers are held-out test numbers
        public int nTestPairs;
        public String difficultyBreakdownCsv = "";
        public String curveXLabel;
        public String curveYLabel;
        public List<Double> curveX = new ArrayList<>();
        public List<Double> curveY = new ArrayList<>();
        public String curveY2Label;     // null = only one series
        public List<Double> curveY2 = new ArrayList<>();
        public List<String> gridResults = new ArrayList<>();
    }

    /** The two feature layouts of the grid: only the 4 measures per profile (compressed), or every bucket difference too (full). */
    private static final class Variant {
        final boolean full;
        final Corpus.Datasets data;

        Variant(boolean full, Corpus.Datasets data) {
            this.full = full;
            this.data = data;
        }

        String label() {
            return full ? "full" : "compressed";
        }
    }

    // ------------------------------------------------------------------
    // Data
    // ------------------------------------------------------------------

    private FeatureConfig baseConfig(boolean full) {
        FeatureConfig config = FeatureConfig.fromSystemProperties();
        config.fullBlockDifferences = full;
        config.validate();
        return config;
    }

    private Corpus loadCorpus(TrainingStatus status, boolean needFullWidth) {

        step(status, "Loading dataset (all difficulties)...", 5);

        int widest = new FeatureExtractor(null, baseConfig(needFullWidth)).pairVectorLength();
        Corpus corpus = Corpus.load(datasetRoot, false, subsetSize, FeatureConfig.Casing.PRESERVE, widest, 0.5);
        if (corpus == null) {
            throw new IllegalStateException("No training data found under " + datasetRoot
                    + " — check app.dataset.root and that the dataset folder is in place.");
        }
        return corpus;
    }

    private Variant build(Corpus corpus, boolean full, TrainingStatus status, int percent) {

        step(status, "Extracting " + (full ? "full" : "compressed") + " pair features...", percent);
        return new Variant(full, corpus.build(baseConfig(full)));
    }

    // ------------------------------------------------------------------
    // The three pair models
    // ------------------------------------------------------------------

    public Outcome trainDecisionTree(TrainingStatus status) {

        Corpus corpus = loadCorpus(status, false);

        /*
         * A simple, justifiable tree:
         * - only the compressed pair features. Each one has a name and a meaning
         *   ("how different are the two sentences in average word length"), so a
         *   split can be read and defended. The bucket differences can't be.
         * - a shallow tree (at most depth 5 = at most 5 questions per decision).
         * - the depth is picked on the VALIDATION split, and when a deeper tree is
         *   not clearly better the simpler one wins (Occam's razor): the shallowest
         *   depth within simplicityTolerance of the best validation macro-F1.
         */
        int[] maxDepths = { 2, 3, 4, 5 };
        final double simplicityTolerance = 0.005;

        Variant v = build(corpus, false, status, 20);
        Corpus.Datasets d = v.data;

        Outcome outcome = new Outcome();
        outcome.curveXLabel = "max depth";
        outcome.curveYLabel = "Validation macro-F1";

        DecisionTree[] trees = new DecisionTree[maxDepths.length];
        double[] validationScore = new double[maxDepths.length];
        double best = -1.0;

        step(status, "Searching the tree depth (on validation)...", 35);

        for (int i = 0; i < maxDepths.length; i++) {

            DecisionTree tree = new DecisionTree(d.train, d.trainY, maxDepths[i], 20, 5, 0.0);
            tree.setMaxBins(Integer.getInteger("tree.bins", 0));
            tree.makeBinaryTree();
            trees[i] = tree;

            validationScore[i] = validationMacroF1(tree::predictProbabilities, d);
            best = Math.max(best, validationScore[i]);

            outcome.gridResults.add("compressed features, depth=" + maxDepths[i] + " -> validation macro-F1="
                    + round(validationScore[i]));
            outcome.curveX.add((double) maxDepths[i]);
            outcome.curveY.add(validationScore[i]);

            step(status, "Tried " + (i + 1) + " / " + maxDepths.length + " depths...",
                    35 + (i + 1) * 30 / maxDepths.length);
        }

        int chosen = 0;
        while (validationScore[chosen] < best - simplicityTolerance) {
            chosen++;
        }
        DecisionTree tree = trees[chosen];

        // questions whose answer never changes the label are removed (lossless: same predictions, smaller tree)
        tree.pruneRedundantSplits();

        String config = "compressed features, max depth=" + maxDepths[chosen] + " (shallowest depth within "
                + simplicityTolerance + " validation macro-F1 of the best)";

        return finish(status, outcome, ModelStore.TREE, "decisionTree.ser", tree, d, tree::predictProbabilities, config);
    }

    public Outcome trainLogisticRegression(TrainingStatus status) {

        Corpus corpus = loadCorpus(status, true);

        double[] learningRates = { 0.1, 0.6 };
        int searchIterations = 800;
        int finalIterations = 4000;

        Outcome outcome = new Outcome();
        outcome.curveXLabel = "iteration";
        outcome.curveYLabel = "Training loss";

        List<Variant> variants = List.of(build(corpus, false, status, 12), build(corpus, true, status, 18));

        double best = -1.0;
        double bestLearningRate = learningRates[0];
        Variant bestVariant = variants.get(0);

        step(status, "Searching learning rate / feature set (on validation)...", 25);
        int tried = 0;
        int total = learningRates.length * variants.size();

        for (Variant v : variants) {
            for (double lr : learningRates) {

                double[] weightsBias = LogisticRegression.gradiantDescent(v.data.train, v.data.trainY,
                        searchIterations, lr, true);
                double score = validationMacroF1(m -> LogisticRegression.predictProbabilities(m, weightsBias), v.data);

                outcome.gridResults.add(v.label() + " features, lr=" + lr + " -> validation macro-F1=" + round(score));

                if (score > best) {
                    best = score;
                    bestLearningRate = lr;
                    bestVariant = v;
                }
                tried++;
                step(status, "Tried " + tried + " / " + total + " configurations...", 25 + tried * 25 / total);
            }
        }

        // final, longer run with the winning combination: this is the one that gets saved and charted
        step(status, "Final training run with the best configuration...", 55);
        Corpus.Datasets d = bestVariant.data;

        List<Double> lossHistory = new ArrayList<>();
        double[] weightsBias = LogisticRegression.gradiantDescent(d.train, d.trainY, finalIterations,
                bestLearningRate, true, lossHistory);

        // downsampling the loss history so the chart on the page stays light
        int recordEvery = Math.max(1, lossHistory.size() / 300);
        for (int i = 0; i < lossHistory.size(); i += recordEvery) {
            outcome.curveX.add((double) (i + 1));
            outcome.curveY.add(lossHistory.get(i));
        }

        String config = bestVariant.label() + " features, learning rate=" + bestLearningRate;
        return finish(status, outcome, ModelStore.LOGISTIC, "logisticRegression.ser", weightsBias, d,
                m -> LogisticRegression.predictProbabilities(m, weightsBias), config);
    }

    public Outcome trainNeuralNetwork(TrainingStatus status) {

        Corpus corpus = loadCorpus(status, true);

        int[][] hiddenSizeOptions = { { 64, 32 }, { 128, 64, 32 } };
        int searchEpochs = 8;
        int finalEpochs = 40;

        Outcome outcome = new Outcome();
        outcome.curveXLabel = "epoch";
        outcome.curveYLabel = "Training loss";
        outcome.curveY2Label = "Validation loss";

        List<Variant> variants = List.of(build(corpus, false, status, 12), build(corpus, true, status, 18));

        double best = -1.0;
        int[] bestHidden = hiddenSizeOptions[0];
        Variant bestVariant = variants.get(0);

        step(status, "Searching architecture / feature set (on validation)...", 25);
        int tried = 0;
        int total = hiddenSizeOptions.length * variants.size();

        for (Variant v : variants) {
            for (int[] hidden : hiddenSizeOptions) {

                NeuralNetwork network = new NeuralNetwork(v.data.train, v.data.trainY, hidden, 5.0);
                network.train(searchEpochs, 0.001, 64, 1e-2, 0.2, true, v.data.val, v.data.valY, 3);
                double score = validationMacroF1(network::predictProbabilities, v.data);

                outcome.gridResults.add(v.label() + " features, hidden=" + Arrays.toString(hidden)
                        + " -> validation macro-F1=" + round(score));

                if (score > best) {
                    best = score;
                    bestHidden = hidden;
                    bestVariant = v;
                }
                tried++;
                step(status, "Tried " + tried + " / " + total + " configurations...", 25 + tried * 25 / total);
            }
        }

        // final, longer run (more epochs, more patience) with the winning combination
        step(status, "Final training run with the best configuration...", 55);
        Corpus.Datasets d = bestVariant.data;

        NeuralNetwork network = new NeuralNetwork(d.train, d.trainY, bestHidden, 5.0);
        List<double[]> lossHistory = new ArrayList<>();
        // validation data is used for early stopping only (never the test split)
        network.train(finalEpochs, 0.001, 64, 1e-2, 0.2, true, d.val, d.valY, 8, lossHistory);

        for (int epoch = 0; epoch < lossHistory.size(); epoch++) {
            double[] row = lossHistory.get(epoch);
            outcome.curveX.add((double) (epoch + 1));
            outcome.curveY.add(row[0]);
            if (row.length > 1) {
                outcome.curveY2.add(row[1]);
            }
        }

        String config = bestVariant.label() + " features, hidden layers=" + Arrays.toString(bestHidden);
        return finish(status, outcome, ModelStore.NEURAL, "neuralNetwork.ser", network, d,
                network::predictProbabilities, config);
    }

    // ------------------------------------------------------------------
    // The retrieval model
    // ------------------------------------------------------------------

    /**
     * Trains the retrieval model (sentences embedded one by one, triplet loss) with the settings of
     * RetrievalPipeline.Settings, and reports the macro-F1 of the switch detection with it.
     */
    public Outcome trainRetrieval(TrainingStatus status) {

        step(status, "Loading the documents (all difficulties)...", 5);
        RetrievalPipeline.Documents documents = RetrievalPipeline.loadDocuments(datasetRoot, false);
        if (documents.train.isEmpty()) {
            throw new IllegalStateException("No training data found under " + datasetRoot
                    + " — check app.dataset.root and that the dataset folder is in place.");
        }

        step(status, "Extracting features and training the embedding network (triplet loss)...", 15);
        RetrievalPipeline.Settings settings = new RetrievalPipeline.Settings();
        RetrievalPipeline.Result r = RetrievalPipeline.run(settings, documents, false);

        step(status, "Saving model...", 95);
        saveModel("retrieval.ser", ModelStore.RETRIEVAL, r.model, r.threshold, r.extractor, r.description);

        Outcome outcome = new Outcome();
        outcome.chosenConfig = String.format(java.util.Locale.ROOT,
                "window=%d, retrieval P@1 %.3f MAP %.3f (raw features: P@1 %.3f MAP %.3f), switch AUC %.3f (raw %.3f) | %s",
                r.window, r.retrieval.precisionAt1, r.retrieval.map, r.rawBaseline.precisionAt1, r.rawBaseline.map,
                r.testAuc, r.rawAuc, r.description);
        outcome.featureConfig = r.extractor.getConfig().signature();
        outcome.precision = r.metrics[1];
        outcome.recall = r.metrics[2];
        outcome.f1 = r.metrics[3];
        outcome.balancedAccuracy = r.metrics[4];
        outcome.macroF1 = r.metrics[5];
        outcome.threshold = r.threshold;
        outcome.evaluatedOnTest = r.onTest;
        outcome.difficultyBreakdownCsv = r.breakdown;
        return outcome;
    }

    // ------------------------------------------------------------------
    // Shared steps
    // ------------------------------------------------------------------

    /** Validation macro-F1 at the best threshold (the number a grid search compares), 0 without validation data. */
    private double validationMacroF1(Function<double[][], double[]> probabilities, Corpus.Datasets d) {

        if (d.val == null) {
            return 0.0;
        }
        double[] p = probabilities.apply(d.val);
        double threshold = LogisticRegression.findBestThreshold(p, d.valY);
        return Metrics.calcMacroF1(LogisticRegression.predict(p, threshold), d.valY);
    }

    /**
     * The part that is the same for the three models: the threshold (validation only), the numbers
     * on the test split (once), the per-difficulty breakdown, saving.
     */
    private Outcome finish(TrainingStatus status, Outcome outcome, String type, String fileName, Object model,
            Corpus.Datasets d, Function<double[][], double[]> probabilities, String config) {

        double threshold = 0.5;
        if (d.val != null) {
            threshold = LogisticRegression.findBestThreshold(probabilities.apply(d.val), d.valY);
        }
        final double finalThreshold = threshold;

        step(status, "Evaluating with the chosen threshold on the held-out test split...", 80);
        double[] result = null;
        int nPairs = 0;
        if (d.eval != null) {
            // eval = test when there is a test half, else the validation half (d.evalIsTest says which)
            result = Metrics.evaluateQuietly(LogisticRegression.predict(probabilities.apply(d.eval), threshold), d.evalY);
            nPairs = d.evalY.length;
        }

        step(status, "Breaking the result down by difficulty (easy / medium / hard)...", 88);
        StringBuilder breakdown = new StringBuilder();
        for (Map.Entry<String, double[][]> e : d.evalByDifficulty.entrySet()) {
            int[] predictions = LogisticRegression.predict(probabilities.apply(e.getValue()), finalThreshold);
            if (breakdown.length() > 0) {
                breakdown.append(";");
            }
            breakdown.append(Metrics.formatBreakdownEntry(e.getKey(), predictions, d.evalYByDifficulty.get(e.getKey())));
        }

        step(status, "Saving model...", 95);
        saveModel(fileName, type, model, threshold, d.extractor, config);

        outcome.chosenConfig = config;
        outcome.featureConfig = d.extractor.getConfig().signature();
        outcome.threshold = threshold;
        outcome.difficultyBreakdownCsv = breakdown.toString();
        outcome.evaluatedOnTest = d.evalIsTest;
        outcome.nTestPairs = nPairs;
        if (result != null) {
            outcome.precision = result[1];
            outcome.recall = result[2];
            outcome.f1 = result[3];
            outcome.balancedAccuracy = result[4];
            outcome.macroF1 = result[5];
        }
        return outcome;
    }

    private void saveModel(String fileName, String type, Object model, double threshold, FeatureExtractor extractor,
            String description) {

        try {
            ModelStore.save(new File(modelsDir, fileName).getPath(), type, model, threshold, extractor, description);
        } catch (IOException e) {
            // a model that was not saved is not a finished training run: the page must not say "Done"
            throw new IllegalStateException("The model could not be saved to " + fileName + ": " + e.getMessage(), e);
        }
    }

    private void step(TrainingStatus status, String task, int percent) {
        status.currentTask = task;
        status.percent = percent;
        status.log.add("[" + percent + "%] " + task);
    }

    private static double round(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }
}
