package com.hyp.authorswitchdetection.ml;

import com.hyp.authorswitchdetection.service.TrainingStatus;
import data.DatasetLoader;
import data.SentencePair;
import features.FeatureExtractor;
import models.DecisionTree;
import models.LogisticRegression;
import models.Metrics;
import models.NeuralNetwork;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/*
    TODO: (1) Random Forest still has nothing to train against — models/RandomForest.java
    is empty in the standalone project too.
*/

/**
 * The real training pipeline, replacing TrainingService's old simulated
 * numbers. Same dataset format / feature pipeline as Main.java, generalised
 * to train on ALL 3 difficulties combined instead of just "easy", and with
 * a small grid search added on top: for every model, a few feature-set +
 * hyperparameter combinations are tried, and the one with the best
 * validation F1 is kept ("TODO: parameter learning for all the Models").
 *
 * Held-out test split: same idea as Main.java's USE_FULL_VALIDATION = false.
 * The "validation" PAN split is itself divided into two halves (by problem
 * ID, seeded, see DatasetLoader.loadSplitHalf): the first half ("validation")
 * is used for every grid-search decision and for choosing the probability
 * threshold; the second half ("test") is never touched until the very end,
 * and is what f1 / precision / recall / balancedAccuracy on the Outcome (and
 * therefore what shows up on the Results page) are computed from. If the
 * test half happens to be empty (e.g. too few problems for a difficulty),
 * this falls back to reporting validation numbers instead, and says so via
 * Outcome.evaluatedOnTest = false.
 *
 * The winning model is saved to disk under app.trained-models.dir, NOT back
 * into src/main/resources (that copy is the original pre-trained baseline)
 * MlModelLoader checks that folder first, so a freshly trained model is
 * used for prediction immediately, no restart or redeploy needed.
 */
@Component
public class RealTrainer {

    private static final String[] DIFFICULTIES = { "easy", "medium", "hard" };

    @Value("${app.dataset.root:dataset/mawsa26-pan-zenodo-DATA/}")
    private String datasetRoot;

    @Value("${app.trained-models.dir:data/models}")
    private String modelsDir;

    /**
     * One finished training run, ready to be turned into a ModelMetrics row.
     */
    public static class Outcome {
        public String chosenConfig;
        public double f1;
        public double precision;
        public double recall;
        public double balancedAccuracy;
        public double threshold;
        public boolean evaluatedOnTest; // true = f1/precision/recall/balancedAccuracy are held-out test numbers
        public int nTestPairs;
        public String difficultyBreakdownCsv = ""; // ";" joined per-difficulty summary, same split as above
        public String curveXLabel;
        public String curveYLabel;
        public List<Double> curveX = new ArrayList<>();
        public List<Double> curveY = new ArrayList<>();
        public String curveY2Label; // null = only one series
        public List<Double> curveY2 = new ArrayList<>();
        public List<String> gridResults = new ArrayList<>();
    }

    /** Precision/recall/F1/balancedAccuracy of a fitted model, plus which split they came from. */
    private static final class FinalMetrics {
        double[] result; // {accuracy, precision, recall, f1, balancedAccuracy}
        boolean onTest;
        int nPairs;
    }

    public Outcome trainDecisionTree(TrainingStatus status) {

        Dataset data = loadDataset(status);

        /*
         * A simple, justifiable tree:
         * - only the 11 compressed pair features. Each one has a name and a meaning
         *   ("how different are the two sentences in average word length"), so a
         *   split can be read and defended. The 256 n-gram bucket differences can't be.
         * - a shallow tree (at most depth 5 = at most 5 questions per decision).
         * - the depth is picked on the VALIDATION split, and when a deeper tree is
         *   not clearly better the simpler one wins (Occam's razor): the shallowest
         *   depth within simplicityTolerance of the best validation F1.
         */
        int[] maxDepths = { 2, 3, 4, 5 };
        final double simplicityTolerance = 0.005;
        final boolean fullFeatures = false;

        Outcome outcome = new Outcome();
        outcome.curveXLabel = "max depth";
        outcome.curveYLabel = "Validation F1";

        DecisionTree[] trees = new DecisionTree[maxDepths.length];
        double[] validationF1 = new double[maxDepths.length];
        double bestF1 = -1.0;

        step(status, "Searching the tree depth (on validation)...", 35);

        for (int d = 0; d < maxDepths.length; d++) {

            DecisionTree tree = new DecisionTree(data.trainCompressed, data.trainTruth, maxDepths[d], 20, 5, 0.0);
            tree.makeBinaryTree();
            trees[d] = tree;

            double[] result = data.validation.isEmpty() ? null : tree.evaluate(data.valCompressed, data.valTruth);
            validationF1[d] = result == null ? 0.0 : result[3];
            bestF1 = Math.max(bestF1, validationF1[d]);

            outcome.gridResults.add("compressed features, depth=" + maxDepths[d] + " -> validation F1="
                    + round(validationF1[d]));
            outcome.curveX.add((double) maxDepths[d]);
            outcome.curveY.add(validationF1[d]);

            step(status, "Tried " + (d + 1) + " / " + maxDepths.length + " depths...",
                    35 + (d + 1) * 30 / maxDepths.length);
        }

        // the shallowest tree that is (almost) as good as the best one
        int chosen = 0;
        while (validationF1[chosen] < bestF1 - simplicityTolerance) {
            chosen++;
        }

        int bestDepth = maxDepths[chosen];
        DecisionTree bestTree = trees[chosen];

        // questions whose answer never changes the label are removed (lossless: same predictions, smaller tree)
        bestTree.pruneRedundantSplits();
        boolean bestFullFeatures = fullFeatures;

        step(status, "Evaluating the chosen tree on the held-out test split...", 80);
        double[][] testMatrix = bestFullFeatures ? data.testFull : data.testCompressed;
        double[][] valMatrix = bestFullFeatures ? data.valFull : data.valCompressed;
        FinalMetrics finalMetrics = evaluateFinal(bestTree, testMatrix, data.testTruth, data.test.size(),
                valMatrix, data.valTruth, data.validation.size());

        step(status, "Breaking the result down by difficulty (easy / medium / hard)...", 88);
        FeatureExtractor bestExtractor = bestFullFeatures ? data.extractorFull : data.extractorCompressed;
        Map<String, List<SentencePair>> breakdownPairs = finalMetrics.onTest ? data.testByDifficulty
                : data.validationByDifficulty;
        String difficultyBreakdown = evaluateBreakdown(bestTree::predictAll,
                buildDifficultyMatrices(breakdownPairs, bestExtractor, bestFullFeatures),
                truthByDifficultyOf(breakdownPairs));

        step(status, "Saving model...", 95);
        saveModel("decisionTree.ser", bestFullFeatures, bestTree, bestExtractor);

        outcome.chosenConfig = "compressed (11 named pair) features, max depth=" + bestDepth
                + " (shallowest depth within " + simplicityTolerance + " validation F1 of the best)";
        outcome.threshold = 0.5; // a tree has no probability threshold to tune
        outcome.difficultyBreakdownCsv = difficultyBreakdown;
        applyFinalMetrics(outcome, finalMetrics);

        return outcome;
    }

    public Outcome trainLogisticRegression(TrainingStatus status) {

        Dataset data = loadDataset(status);

        double[] learningRates = { 0.1, 0.6 };
        boolean[] featureSets = { false, true };
        int searchIterations = 800;
        int finalIterations = 4000;

        Outcome outcome = new Outcome();
        outcome.curveXLabel = "iteration";
        outcome.curveYLabel = "Training loss";
        outcome.curveY2Label = null; // logistic regression has no separate validation loss tracked

        double bestF1 = -1.0;
        double bestLearningRate = learningRates[0];
        boolean bestFullFeatures = true;

        step(status, "Searching learning rate / feature set (on validation)...", 25);
        int tried = 0;
        int total = learningRates.length * featureSets.length;

        for (boolean fullFeatures : featureSets) {
            for (double lr : learningRates) {

                double[][] trainMatrix = fullFeatures ? data.trainFull : data.trainCompressed;
                double[][] valMatrix = fullFeatures ? data.valFull : data.valCompressed;

                double[] weightsBias = LogisticRegression.gradiantDescent(trainMatrix, data.trainTruth,
                        searchIterations, lr, true);

                double f1 = 0.0;
                if (!data.validation.isEmpty()) {
                    double[] probabilities = LogisticRegression.predictProbabilities(valMatrix, weightsBias);
                    double threshold = LogisticRegression.findBestThreshold(probabilities, data.valTruth);
                    f1 = Metrics.calcF1(LogisticRegression.predict(probabilities, threshold), data.valTruth);
                }

                String label = (fullFeatures ? "full" : "compressed") + " features, lr=" + lr;
                outcome.gridResults.add(label + " -> validation F1=" + round(f1));

                if (f1 > bestF1) {
                    bestF1 = f1;
                    bestLearningRate = lr;
                    bestFullFeatures = fullFeatures;
                }

                tried++;
                step(status, "Tried " + tried + " / " + total + " configurations...", 25 + tried * 25 / total);
            }
        }

        // final, longer run with the winning combination this is the one that
        // gets saved, and its loss history is what gets charted
        step(status, "Final training run with the best configuration...", 55);

        double[][] trainMatrix = bestFullFeatures ? data.trainFull : data.trainCompressed;
        double[][] valMatrix = bestFullFeatures ? data.valFull : data.valCompressed;
        double[][] testMatrix = bestFullFeatures ? data.testFull : data.testCompressed;
        FeatureExtractor extractor = bestFullFeatures ? data.extractorFull : data.extractorCompressed;

        List<Double> lossHistory = new ArrayList<>();
        double[] weightsBias = LogisticRegression.gradiantDescent(trainMatrix, data.trainTruth, finalIterations,
                bestLearningRate, true, lossHistory);

        // downsampling the loss history so the chart on the page stays light
        int recordEvery = Math.max(1, lossHistory.size() / 300);
        for (int i = 0; i < lossHistory.size(); i += recordEvery) {
            outcome.curveX.add((double) (i + 1));
            outcome.curveY.add(lossHistory.get(i));
        }

        // the decision threshold is chosen on the VALIDATION split only, then used
        // once, unchanged, on the untouched test split below
        double threshold = 0.5;
        if (!data.validation.isEmpty()) {
            double[] probabilities = LogisticRegression.predictProbabilities(valMatrix, weightsBias);
            threshold = LogisticRegression.findBestThreshold(probabilities, data.valTruth);
        }

        step(status, "Evaluating with the chosen threshold on the held-out test split...", 80);
        double finalThreshold = threshold;
        FinalMetrics finalMetrics = evaluateFinal(
                testMatrix == null ? null : LogisticRegression.predict(
                        LogisticRegression.predictProbabilities(testMatrix, weightsBias), finalThreshold),
                data.testTruth, data.test.size(),
                valMatrix == null ? null : LogisticRegression.predict(
                        LogisticRegression.predictProbabilities(valMatrix, weightsBias), finalThreshold),
                data.valTruth, data.validation.size());

        step(status, "Breaking the result down by difficulty (easy / medium / hard)...", 88);
        Map<String, List<SentencePair>> breakdownPairs = finalMetrics.onTest ? data.testByDifficulty
                : data.validationByDifficulty;
        String difficultyBreakdown = evaluateBreakdown(
                matrix -> LogisticRegression.predict(LogisticRegression.predictProbabilities(matrix, weightsBias), finalThreshold),
                buildDifficultyMatrices(breakdownPairs, extractor, bestFullFeatures),
                truthByDifficultyOf(breakdownPairs));

        step(status, "Saving model...", 95);
        saveModel("logisticRegression.ser", bestFullFeatures, weightsBias, threshold, extractor);

        outcome.chosenConfig = (bestFullFeatures ? "full" : "compressed") + " features, learning rate=" + bestLearningRate;
        outcome.threshold = finalThreshold;
        outcome.difficultyBreakdownCsv = difficultyBreakdown;
        applyFinalMetrics(outcome, finalMetrics);

        return outcome;
    }

    public Outcome trainNeuralNetwork(TrainingStatus status) {

        Dataset data = loadDataset(status);

        int[][] hiddenSizeOptions = { { 64, 32 }, { 128, 64, 32 } };
        boolean[] featureSets = { false, true };
        int searchEpochs = 8;
        int finalEpochs = 40;

        Outcome outcome = new Outcome();
        outcome.curveXLabel = "epoch";
        outcome.curveYLabel = "Training loss";
        outcome.curveY2Label = "Validation loss";

        double bestF1 = -1.0;
        int[] bestHiddenSizes = hiddenSizeOptions[0];
        boolean bestFullFeatures = true;

        step(status, "Searching architecture / feature set (on validation)...", 25);
        int tried = 0;
        int total = hiddenSizeOptions.length * featureSets.length;

        for (boolean fullFeatures : featureSets) {
            for (int[] hiddenSizes : hiddenSizeOptions) {

                double[][] trainMatrix = fullFeatures ? data.trainFull : data.trainCompressed;
                double[][] valMatrix = fullFeatures ? data.valFull : data.valCompressed;

                NeuralNetwork network = new NeuralNetwork(trainMatrix, data.trainTruth, hiddenSizes, 5.0);
                network.train(searchEpochs, 0.001, 64, 1e-2, 0.2, true, valMatrix, data.valTruth, 3);

                double f1 = 0.0;
                if (!data.validation.isEmpty()) {
                    double[] probabilities = network.predictProbabilities(valMatrix);
                    double threshold = LogisticRegression.findBestThreshold(probabilities, data.valTruth);
                    f1 = Metrics.calcF1(LogisticRegression.predict(probabilities, threshold), data.valTruth);
                }

                String label = (fullFeatures ? "full" : "compressed") + " features, hidden=" + java.util.Arrays.toString(hiddenSizes);
                outcome.gridResults.add(label + " -> validation F1=" + round(f1));

                if (f1 > bestF1) {
                    bestF1 = f1;
                    bestHiddenSizes = hiddenSizes;
                    bestFullFeatures = fullFeatures;
                }

                tried++;
                step(status, "Tried " + tried + " / " + total + " configurations...", 25 + tried * 25 / total);
            }
        }

        // final, longer run (more epochs, more patience) with the winning combination
        step(status, "Final training run with the best configuration...", 55);

        double[][] trainMatrix = bestFullFeatures ? data.trainFull : data.trainCompressed;
        double[][] valMatrix = bestFullFeatures ? data.valFull : data.valCompressed;
        double[][] testMatrix = bestFullFeatures ? data.testFull : data.testCompressed;
        FeatureExtractor extractor = bestFullFeatures ? data.extractorFull : data.extractorCompressed;

        NeuralNetwork network = new NeuralNetwork(trainMatrix, data.trainTruth, bestHiddenSizes, 5.0);
        List<double[]> lossHistory = new ArrayList<>();
        // validation data is used for early stopping only (never the test split)
        network.train(finalEpochs, 0.001, 64, 1e-2, 0.2, true, valMatrix, data.valTruth, 8, lossHistory);

        for (int epoch = 0; epoch < lossHistory.size(); epoch++) {
            double[] row = lossHistory.get(epoch);
            outcome.curveX.add((double) (epoch + 1));
            outcome.curveY.add(row[0]);
            if (row.length > 1) {
                outcome.curveY2.add(row[1]);
            }
        }

        // the decision threshold is chosen on the VALIDATION split only, then used
        // once, unchanged, on the untouched test split below
        double threshold = 0.5;
        if (!data.validation.isEmpty()) {
            double[] probabilities = network.predictProbabilities(valMatrix);
            threshold = LogisticRegression.findBestThreshold(probabilities, data.valTruth);
        }

        step(status, "Evaluating with the chosen threshold on the held-out test split...", 80);
        double finalThreshold = threshold;
        FinalMetrics finalMetrics = evaluateFinal(
                testMatrix == null ? null : LogisticRegression.predict(network.predictProbabilities(testMatrix), finalThreshold),
                data.testTruth, data.test.size(),
                valMatrix == null ? null : LogisticRegression.predict(network.predictProbabilities(valMatrix), finalThreshold),
                data.valTruth, data.validation.size());

        step(status, "Breaking the result down by difficulty (easy / medium / hard)...", 88);
        Map<String, List<SentencePair>> breakdownPairs = finalMetrics.onTest ? data.testByDifficulty
                : data.validationByDifficulty;
        String difficultyBreakdown = evaluateBreakdown(
                matrix -> LogisticRegression.predict(network.predictProbabilities(matrix), finalThreshold),
                buildDifficultyMatrices(breakdownPairs, extractor, bestFullFeatures),
                truthByDifficultyOf(breakdownPairs));

        step(status, "Saving model...", 95);
        saveModel("neuralNetwork.ser", bestFullFeatures, network, threshold, extractor);

        outcome.chosenConfig = (bestFullFeatures ? "full" : "compressed") + " features, hidden layers="
                + java.util.Arrays.toString(bestHiddenSizes);
        outcome.threshold = threshold;
        outcome.difficultyBreakdownCsv = difficultyBreakdown;
        applyFinalMetrics(outcome, finalMetrics);

        return outcome;
    }

    /**
     * Evaluates on the held-out test split when it is available and non-empty;
     * otherwise falls back to the validation split (and marks the result as such
     * via FinalMetrics.onTest = false), so a run still produces numbers even if
     * a dataset happens to have no test half.
     */
    private FinalMetrics evaluateFinal(DecisionTree bestTree, double[][] testMatrix, int[] testTruth, int nTest,
            double[][] valMatrix, int[] valTruth, int nVal) {

        FinalMetrics fm = new FinalMetrics();

        if (testMatrix != null && testMatrix.length > 0) {
            fm.result = bestTree.evaluate(testMatrix, testTruth);
            fm.onTest = true;
            fm.nPairs = nTest;
        } else if (valMatrix != null && valMatrix.length > 0) {
            fm.result = bestTree.evaluate(valMatrix, valTruth);
            fm.onTest = false;
            fm.nPairs = nVal;
        }

        return fm;
    }

    /** Same idea as the overload above, but for models evaluated via pre-computed 0/1 predictions. */
    private FinalMetrics evaluateFinal(int[] testPredictions, int[] testTruth, int nTest,
            int[] valPredictions, int[] valTruth, int nVal) {

        FinalMetrics fm = new FinalMetrics();

        if (testPredictions != null && testPredictions.length > 0) {
            fm.result = Metrics.evaluate(testPredictions, testTruth);
            fm.onTest = true;
            fm.nPairs = nTest;
        } else if (valPredictions != null && valPredictions.length > 0) {
            fm.result = Metrics.evaluate(valPredictions, valTruth);
            fm.onTest = false;
            fm.nPairs = nVal;
        }

        return fm;
    }

    private void applyFinalMetrics(Outcome outcome, FinalMetrics fm) {

        if (fm == null || fm.result == null) {
            outcome.evaluatedOnTest = false;
            outcome.nTestPairs = 0;
            return;
        }

        outcome.precision = fm.result[1];
        outcome.recall = fm.result[2];
        outcome.f1 = fm.result[3];
        outcome.balancedAccuracy = fm.result.length > 4 ? fm.result[4] : 0.0;
        outcome.evaluatedOnTest = fm.onTest;
        outcome.nTestPairs = fm.nPairs;
    }

    /**
     * Builds one feature matrix per difficulty (skipping any difficulty with no
     * pairs), using an ALREADY FITTED extractor — same rule as buildPairMatrix,
     * never refits anything.
     */
    private Map<String, double[][]> buildDifficultyMatrices(Map<String, List<SentencePair>> pairsByDifficulty,
            FeatureExtractor extractor, boolean fullFeatures) {

        Map<String, double[][]> matrices = new LinkedHashMap<>();
        for (String difficulty : DIFFICULTIES) {
            List<SentencePair> pairs = pairsByDifficulty.get(difficulty);
            if (pairs != null && !pairs.isEmpty()) {
                matrices.put(difficulty, buildPairMatrix(pairs, extractor, fullFeatures));
            }
        }
        return matrices;
    }

    private Map<String, int[]> truthByDifficultyOf(Map<String, List<SentencePair>> pairsByDifficulty) {

        Map<String, int[]> truths = new LinkedHashMap<>();
        for (String difficulty : DIFFICULTIES) {
            List<SentencePair> pairs = pairsByDifficulty.get(difficulty);
            if (pairs != null && !pairs.isEmpty()) {
                truths.put(difficulty, truthValuesOf(pairs));
            }
        }
        return truths;
    }

    /**
     * Evaluates a fitted model separately on each difficulty ("easy", "medium",
     * "hard"), using models.Metrics.formatBreakdownEntry — the SAME formatting
     * Main.java uses — and returns the ";" joined summary stored in
     * Outcome.difficultyBreakdownCsv / ModelMetrics.difficultyBreakdownCsv, so
     * the Results page shows exactly this breakdown.
     */
    private String evaluateBreakdown(Function<double[][], int[]> predictor, Map<String, double[][]> matricesByDifficulty,
            Map<String, int[]> truthByDifficulty) {

        StringBuilder breakdown = new StringBuilder();

        for (String difficulty : DIFFICULTIES) {

            double[][] matrix = matricesByDifficulty.get(difficulty);
            int[] truth = truthByDifficulty.get(difficulty);

            if (matrix == null || matrix.length == 0) {
                continue;
            }

            int[] predictions = predictor.apply(matrix);
            String entry = Metrics.formatBreakdownEntry(difficulty, predictions, truth);

            if (breakdown.length() > 0) {
                breakdown.append(";");
            }
            breakdown.append(entry);
        }

        return breakdown.toString();
    }

    // everything needed to train + validate + test, both feature-set variants built once
    private static class Dataset {
        List<SentencePair> train;
        List<SentencePair> validation;
        List<SentencePair> test;
        // same pairs as validation/test above, kept split out by difficulty too,
        // so the final evaluation can ALSO be broken down per difficulty
        Map<String, List<SentencePair>> validationByDifficulty = new LinkedHashMap<>();
        Map<String, List<SentencePair>> testByDifficulty = new LinkedHashMap<>();
        int[] trainTruth;
        int[] valTruth;
        int[] testTruth;
        double[][] trainCompressed;
        double[][] trainFull;
        double[][] valCompressed;
        double[][] valFull;
        double[][] testCompressed;
        double[][] testFull;
        FeatureExtractor extractorCompressed;
        FeatureExtractor extractorFull;
    }

    private Dataset loadDataset(TrainingStatus status) {

        step(status, "Loading dataset (all difficulties)...", 5);

        List<SentencePair> train = new ArrayList<>();
        List<SentencePair> validation = new ArrayList<>();
        List<SentencePair> test = new ArrayList<>();
        List<String> corpus = new ArrayList<>();
        Map<String, List<SentencePair>> validationByDifficulty = new LinkedHashMap<>();
        Map<String, List<SentencePair>> testByDifficulty = new LinkedHashMap<>();

        for (String difficulty : DIFFICULTIES) {
            train.addAll(DatasetLoader.loadSplit(datasetRoot, "train", difficulty));
            // the PAN "validation" split is itself divided by problem ID (seeded,
            // see DatasetLoader.loadSplitHalf): first half = validation (used for
            // every grid-search / threshold decision), second half = test (used
            // exactly once, at the very end, and never for any decision)
            List<SentencePair> validationForDifficulty = DatasetLoader.loadSplitHalf(datasetRoot, "validation",
                    difficulty, true);
            List<SentencePair> testForDifficulty = DatasetLoader.loadSplitHalf(datasetRoot, "validation", difficulty,
                    false);
            validation.addAll(validationForDifficulty);
            test.addAll(testForDifficulty);
            corpus.addAll(DatasetLoader.getAllSentences(datasetRoot, "train", difficulty));

            validationByDifficulty.put(difficulty, validationForDifficulty);
            testByDifficulty.put(difficulty, testForDifficulty);
        }

        if (train.isEmpty()) {
            throw new IllegalStateException("No training data found under " + datasetRoot
                    + " — check app.dataset.root and that the dataset folder is in place.");
        }

        Dataset data = new Dataset();
        data.train = train;
        data.validation = validation;
        data.test = test;
        data.validationByDifficulty = validationByDifficulty;
        data.testByDifficulty = testByDifficulty;
        data.trainTruth = truthValuesOf(train);
        data.valTruth = truthValuesOf(validation);
        data.testTruth = truthValuesOf(test);

        step(status, "Fitting feature extractor on the training corpus...", 10);

        // two separate extractors: the n-gram difference scaler is only fitted
        // when the full feature set is actually used, so keeping one extractor per
        // feature set avoids fitting it on features that end up unused
        data.extractorCompressed = new FeatureExtractor(corpus);
        data.extractorFull = new FeatureExtractor(corpus);

        step(status, "Extracting pair features (compressed)...", 15);
        buildMatrices(data, false);

        step(status, "Extracting pair features (full)...", 20);
        buildMatrices(data, true);

        return data;
    }

    private void buildMatrices(Dataset data, boolean fullFeatures) {

        FeatureExtractor extractor = fullFeatures ? data.extractorFull : data.extractorCompressed;

        double[][] sentenceFeatures1 = new double[data.train.size()][];
        double[][] sentenceFeatures2 = new double[data.train.size()][];
        List<double[]> trainingFeatures = new ArrayList<>();

        for (int i = 0; i < data.train.size(); i++) {
            sentenceFeatures1[i] = extractor.getFeatures(data.train.get(i).getSentence1());
            sentenceFeatures2[i] = extractor.getFeatures(data.train.get(i).getSentence2());
            trainingFeatures.add(sentenceFeatures1[i]);
            trainingFeatures.add(sentenceFeatures2[i]);
        }

        extractor.fitScaler(trainingFeatures);

        double[][] trainMatrix = new double[data.train.size()][];
        for (int i = 0; i < data.train.size(); i++) {
            double[] f1 = extractor.standardize(sentenceFeatures1[i]);
            double[] f2 = extractor.standardize(sentenceFeatures2[i]);
            trainMatrix[i] = fullFeatures ? data.train.get(i).getFullPairFeatures(f1, f2)
                    : data.train.get(i).getPairFeatures(f1, f2);
        }

        if (fullFeatures) {
            extractor.fitNGramDifferenceScaler(trainMatrix);
            for (int i = 0; i < trainMatrix.length; i++) {
                trainMatrix[i] = extractor.standardizeNGramDifferences(trainMatrix[i]);
            }
        }

        double[][] valMatrix = buildPairMatrix(data.validation, extractor, fullFeatures);
        double[][] testMatrix = buildPairMatrix(data.test, extractor, fullFeatures);

        if (fullFeatures) {
            data.trainFull = trainMatrix;
            data.valFull = valMatrix;
            data.testFull = testMatrix;
        } else {
            data.trainCompressed = trainMatrix;
            data.valCompressed = valMatrix;
            data.testCompressed = testMatrix;
        }
    }

    /**
     * Builds the feature matrix of a validation or test split using the
     * extractor's z-score parameters, which were fitted on the TRAINING data
     * only (see buildMatrices above) — same rule for both splits, so the test
     * split is transformed exactly like validation, never refitted.
     */
    private double[][] buildPairMatrix(List<SentencePair> pairs, FeatureExtractor extractor, boolean fullFeatures) {

        double[][] matrix = new double[pairs.size()][];
        for (int i = 0; i < pairs.size(); i++) {
            double[] f1 = extractor.standardize(extractor.getFeatures(pairs.get(i).getSentence1()));
            double[] f2 = extractor.standardize(extractor.getFeatures(pairs.get(i).getSentence2()));
            double[] pairFeatures = fullFeatures ? pairs.get(i).getFullPairFeatures(f1, f2)
                    : pairs.get(i).getPairFeatures(f1, f2);
            matrix[i] = fullFeatures ? extractor.standardizeNGramDifferences(pairFeatures) : pairFeatures;
        }
        return matrix;
    }

    private int[] truthValuesOf(List<SentencePair> pairs) {
        int[] truth = new int[pairs.size()];
        for (int i = 0; i < pairs.size(); i++) {
            truth[i] = pairs.get(i).getLabel();
        }
        return truth;
    }

    private void saveModel(String fileName, boolean fullFeatures, Object... toWrite) {

        File dir = new File(modelsDir);
        dir.mkdirs();

        try (FileOutputStream fileOut = new FileOutputStream(new File(dir, fileName));
                BufferedOutputStream bos = new BufferedOutputStream(fileOut);
                ObjectOutputStream oos = new ObjectOutputStream(bos)) {

            for (Object o : toWrite) {
                if (o instanceof Double d) {
                    oos.writeDouble(d);
                } else {
                    oos.writeObject(o);
                }
            }

            // written last, after the extractor — see MlModelLoader.readFullFeaturesFlag
            oos.writeBoolean(fullFeatures);

        } catch (IOException e) {
            System.err.println("Error saving model to " + fileName);
            System.err.println(e.getMessage());
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
