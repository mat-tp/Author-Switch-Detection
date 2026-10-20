import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import data.Corpus;
import data.PairMatrixBuilder;
import data.SentencePair;
import features.FeatureConfig;
import features.FeatureExtractor;
import models.DecisionTree;
import models.LogisticRegression;
import models.Metrics;
import models.ModelStore;
import models.NeuralNetwork;

import com.hyp.authorswitchdetection.model.ModelMetrics;
import com.hyp.authorswitchdetection.model.ModelMetricsIO;

/*TODO: (1) Importance(G)=F1_all − F1_without G
    That tells you much more directly whether the group contributes useful predictive information.
    -> done by Ablation.java (feature groups, macro-F1)
*/

/*TODO: (1) Parameter learning for all the Models */
/*  -> done by Ablation.java (parameter sweeps, validation ranked) */

/*

    From Previous Logistic Regression Run Not this one Actually:

    The model achieves 92.88% overall accuracy, substantially exceeding the 69.01% majority-class baseline.
    It achieves 89.13% precision, 87.72% recall, and an F1-score of 88.42%, indicating that it can identify
    author-switch instances while maintaining a relatively low number of false-positive predictions.

*/

public class Main {

    // Path to the MAWSA dataset.
    static final String DATASET_PATH = System.getProperty("dataset", "dataset/mawsa26-pan-zenodo-DATA/");

    /*
        Every difficulty level is combined for training/validation/test and the final evaluation is also broken down
        per difficulty (see evaluateByDifficulty) and not just reported combined.

     */
    static final String[] DIFFICULTIES = Corpus.DIFFICULTIES;

    // The models that can be trained
    // (the constants / helpers below are package-private so Ablation.java reuses
    // exactly the same data pipeline and baseline settings as Main)
    private static final String DECISION_TREE = "tree";
    private static final String LOGISTIC_REGRESSION = "logistic";
    private static final String NEURAL_NETWORK = "neural";
    private static final String ALL_MODELS = "all";

    /*
     * Which model to train. Using the CMD Commands:
     *
     * java -cp out Main all
     * java -cp out Main tree
     * java -cp out Main logistic
     * java -cp out Main neural
     *
     * The feature settings come from -D options, see features.FeatureConfig:
     * java -Dcasing=hybrid -Dchar.n=2,3,4 -Dchar.buckets=512 -Dpair=abs+mean -cp out Main all
     */
    private static final String MODEL_TYPE = ALL_MODELS;

    /*
     * false: the validation set is divided in two halves.
     * 1st half = validation (choosing the threshold), 2nd half = test (unused data).
     * true: the FULL validation set is used for validation and no test is done,
     * for when the real testing dataset is available.
     */
    static final boolean USE_FULL_VALIDATION = Boolean.getBoolean("full.validation");

    // The file the model is saved to (and loaded from for the inference), in the
    // SAME folder + on-disk format the Spring web app's MlModelLoader reads from
    // (app.trained-models.dir, default "data/models"), so a model trained here
    // from the command line is picked up for live prediction too, not just for
    // the metrics on the Results page.
    private static final String MODELS_DIR = System.getProperty("models.dir", "data/models/");

    // Where trained-run metrics are appended so they show up on the web app's
    // Results page (same file the Spring app's ModelMetricsStore reads/writes:
    // app.model-metrics.file, default "data/model-metrics.json").
    private static final String METRICS_FILE = System.getProperty("metrics.file", "data/model-metrics.json");

    // Data settings. At most this many training pairs (and per difficulty, this
    // many validation / test pairs) are used. For a quick trial run:
    // java -Dsubset=50000 -cp out Main tree
    // Default = no limit: ALL the pairs are used (the memory guard below still
    // shrinks everything if the heap is too small). Give more heap with -Xmx.
    static final int SUBSET_SIZE = Integer.getInteger("subset", Integer.MAX_VALUE);

    /*
     * The feature matrices (training + validation + test) may use at most this
     * part of the JVM's max heap (-Xmx). The rest is left for the raw text while
     * it is being loaded, for the models while they train, and for the garbage
     * collector. If the data wanted does not fit, ALL the sets are shrunk by the
     * same fraction (see PairMatrixBuilder.fractionThatFitsInMemory), so the run finishes instead
     * of ending with "java.lang.OutOfMemoryError: Java heap space".
     */
    static final double HEAP_FRACTION_FOR_MATRICES = 0.5;

    // Logistic Regression settings
    static final int N_ITERATIONS = Integer.getInteger("lr.iterations", 10000);
    static final double LEARNING_RATE = Double.parseDouble(System.getProperty("lr.rate", "0.6"));
    static final double LR_L2_LAMBDA = Double.parseDouble(System.getProperty("lr.l2", "0.0")); // 0 = no regularisation
    static final boolean USE_CLASS_WEIGHTS = true;

    /*
     * Neural Network settings (input -> hidden layers with ReLU -> 1 sigmoid
     * output). The validation data is used for early stopping, so the network
     * stops when the validation loss no longer improves.
     */
    static final int[] NN_HIDDEN_SIZES = { 128, 64, 32, 16, 8 };
    static final int NN_MAX_EPOCHS = Integer.getInteger("nn.epochs", 60);
    static final String NN_OPTIMIZER = System.getProperty("nn.optimizer", "adam"); // adam | sgd
    // Adam needs a small step, the plain mini batch gradient descent a bigger one
    static final double NN_LEARNING_RATE = Double.parseDouble(
            System.getProperty("nn.lr", NN_OPTIMIZER.equalsIgnoreCase("sgd") ? "0.05" : "0.001"));
    static final int NN_BATCH_SIZE = 64;
    static final double NN_L2_LAMBDA = Double.parseDouble(System.getProperty("nn.l2", "1e-2")); // 0 = no regularisation
    static final double NN_DROPOUT = 0.2; // chance a hidden neuron is switched off, 0 = no dropout
    static final double NN_FEATURE_CLIP = 5.0; // features are clipped to [-5, 5], 0 = no clipping
    static final int NN_PATIENCE = 8; // 0 = no early stopping
    static final boolean NN_USE_CLASS_WEIGHTS = true;

    // Decision Tree settings: maxDepth, minSamplesSplit, minSamplesLeaf,
    // minImpurityDecrease
    static final int TREE_MAX_DEPTH = 5;
    static final int TREE_MIN_SAMPLES_SPLIT = 20;
    static final int TREE_MIN_SAMPLES_LEAF = 5;
    static final double TREE_MIN_IMPURITY_DECREASE = 0.0;
    // 0 = exact CART split search, 16..255 = histogram split search (much faster on a big dataset)
    static final int TREE_MAX_BINS = Integer.getInteger("tree.bins", 0);

    public static void main(String[] args) {

        String modelType = args.length > 0 ? args[0].toLowerCase() : MODEL_TYPE;

        if (!modelType.equals(DECISION_TREE) && !modelType.equals(LOGISTIC_REGRESSION)
                && !modelType.equals(NEURAL_NETWORK) && !modelType.equals(ALL_MODELS)) {
            System.err.println("Unknown model: " + modelType + " (use '" + DECISION_TREE + "', '"
                    + LOGISTIC_REGRESSION + "', '" + NEURAL_NETWORK + "' or '" + ALL_MODELS + "')");
            return;
        }

        String[] modelsToRun = modelType.equals(ALL_MODELS)
                ? new String[] { DECISION_TREE, LOGISTIC_REGRESSION, NEURAL_NETWORK }
                : new String[] { modelType };

        System.out.println("Model: " + modelType);

        // How the features are computed (casing, n-gram orders, buckets, pair representation)
        FeatureConfig config = FeatureConfig.fromSystemProperties();
        System.out.println("Feature settings: " + config.signature());

        /*
         * The text of all the splits, loaded once. The width of the pair vector is only needed
         * for the memory guard, a probe extractor (without a corpus) knows it.
         */
        int pairFeatureCount = new FeatureExtractor(null, config).pairVectorLength();

        Corpus corpus = Corpus.load(DATASET_PATH, USE_FULL_VALIDATION, SUBSET_SIZE, config.casing, pairFeatureCount,
                HEAP_FRACTION_FOR_MATRICES);
        if (corpus == null) {
            return;
        }

        Corpus.Datasets d = corpus.build(config);
        corpus = null; // the raw text is not needed anymore, only the numbers

        FeatureExtractor extractor = d.extractor;

        printClassBalance("Training", d.trainY);
        printFeatureSummary("Training", d.train, extractor);

        if (d.val != null) {
            System.out.println();
            System.out.println("Number of validation pairs: " + d.val.length);
            printClassBalance("Validation", d.valY);
            printFeatureSummary("Validation", d.val, extractor);
        } else {
            System.out.println();
            System.out.println("Number of validation pairs: 0");
        }

        if (!USE_FULL_VALIDATION) {
            System.out.println("Number of test pairs: " + (d.test == null ? 0 : d.test.length));
            if (d.test != null) {
                printClassBalance("Test", d.testY);
            }
        }

        if (d.eval != null) {
            printBaselines(d.evalIsTest ? "Test" : "Validation", d.evalY);
        }

        /*
         * Training, evaluating and saving of the chosen model (or of all the models,
         * they all use the same features and the same train / validation / test data).
         */
        String sentence1 = "The brown fox jumps over the lazy dog.";
        String sentence2 = "The lazy dog finally woke up.";

        double[][] results = new double[modelsToRun.length][];

        for (int m = 0; m < modelsToRun.length; m++) {

            String model = modelsToRun[m];
            String modelFilePath = getModelFilePath(model);

            if (modelsToRun.length > 1) {
                System.out.println();
                System.out.println("==================== " + model + " ====================");
            }

            if (model.equals(DECISION_TREE)) {
                results[m] = trainDecisionTree(d, modelFilePath);

            } else if (model.equals(NEURAL_NETWORK)) {
                results[m] = trainNeuralNetwork(d, modelFilePath);

            } else {
                results[m] = trainLogisticRegression(d, modelFilePath);
            }

            /*
             * For Inference: reload the model that was just saved and predict a new pair.
             */
            predictNewPair(model, modelFilePath, sentence1, sentence2);
        }

        if (modelsToRun.length > 1) {
            printSummary(modelsToRun, results, d.evalIsTest);
        }
    }

    /**
     * Helper method
     * The file the model is saved to (and loaded from for the inference).
     */
    private static String getModelFilePath(String model) {

        if (model.equals(DECISION_TREE)) {
            return MODELS_DIR + "decisionTree.ser";
        }
        if (model.equals(NEURAL_NETWORK)) {
            return MODELS_DIR + "neuralNetwork.ser";
        }
        if (model.equals(LOGISTIC_REGRESSION)) {
            return MODELS_DIR + "logisticRegression.ser";
        }
        return "";
    }

    /**
     * Maps this file's short model names ("tree", "logistic", "neural") to the
     * names the web app's ClassifierRegistry / Results page use ("decision_tree",
     * "logistic_regression", "neural_network").
     */
    private static String webModelName(String model) {

        if (model.equals(DECISION_TREE)) {
            return "decision_tree";
        }
        if (model.equals(NEURAL_NETWORK)) {
            return "neural_network";
        }
        if (model.equals(LOGISTIC_REGRESSION)) {
            return "logistic_regression";
        }
        return model;
    }

    /**
     * Appends one row to the shared model-metrics.json file which will be accessed
     * on the web app's Results page. Every read there re parses the file, so no app restart is needed, whether the
     * web app was already running or is started after wards.
     *
     * @param model short model name ("tree" | "logistic" | "neural")
     * @param result {accuracy, precision, recall, f1, balancedAccuracy, macroF1} from Metrics.evaluate, or null
     * @param onTest true if result came from the held-out test split
     * @param nEvalPairs number of pairs result was computed on
     * @param nTrainPairs number of training pairs used
     * @param threshold decision threshold used
     * @param chosenConfig short human-readable description of the run's settings
     * @param difficultyBreakdown ";" joined per-difficulty summary (see evaluateByDifficulty), "" if none
     */
    private static void saveMetricsRow(String model, double[] result, boolean onTest, int nEvalPairs,
                                       int nTrainPairs, double threshold, String chosenConfig, String featureConfig,
                                       String difficultyBreakdown) {

        if (result == null) {
            System.out.println("(No test/validation data — skipping the results.html metrics row for " + model + ".)");
            return;
        }

        ModelMetrics m = new ModelMetrics();
        m.setModelName(webModelName(model));
        m.setVersionName(webModelName(model) + "_cli_" + (onTest ? "test" : "validation"));
        m.setFull(true);
        m.setNProblems(nTrainPairs);
        m.setPrecision(result[1]);
        m.setRecall(result[2]);
        m.setF1(result[3]);
        m.setBalancedAccuracy(result.length > 4 ? result[4] : 0.0);
        m.setMacroF1(result.length > 5 ? result[5] : 0.0);
        m.setDecisionThreshold(threshold);
        m.setEvaluatedOnTest(onTest);
        m.setNTestPairs(nEvalPairs);
        m.setDifficultyBreakdownCsv(difficultyBreakdown);
        m.setTrainedBy("cli");
        m.setFeatureConfig(featureConfig);
        m.setChosenConfig(chosenConfig + (onTest
                ? " — evaluated on the held-out test split (2nd half of validation, never used for training or thresholding)"
                : " — no test split was available; evaluated on the validation split instead"));

        ModelMetricsIO.append(METRICS_FILE, m);
        System.out.println("Metrics row appended to " + METRICS_FILE + " (visible on the web app's Results page).");
    }

    /**
     * Prints the results of all the models next to each other.
     *
     * @param models  the names of the models
     * @param results {accuracy, precision, recall, f1, balancedAccuracy, macroF1} of each
     *                model (null = no result)
     * @param onTest  true: the results are from the test data, false: from the
     *                validation data
     */
    private static void printSummary(String[] models, double[][] results, boolean onTest) {

        System.out.println();
        System.out.println("==================== Summary (" + (onTest ? "test" : "validation") + " data) "
                + "====================");
        System.out.println(String.format("  %-10s %9s %10s %8s %8s %9s %9s", "Model", "Accuracy", "Precision", "Recall",
                "F1(chg)", "Bal.Acc", "Macro-F1"));

        for (int m = 0; m < models.length; m++) {

            if (results[m] == null) {
                System.out.println(String.format("  %-10s %s", models[m], "no result"));
                continue;
            }

            // older result arrays (e.g. before balanced accuracy / macro-F1 were added) are shorter
            String balancedAccuracy = results[m].length > 4
                    ? String.format(java.util.Locale.ROOT, "%9.4f", results[m][4])
                    : String.format("%9s", "n/a");
            String macroF1 = results[m].length > 5
                    ? String.format(java.util.Locale.ROOT, "%9.4f", results[m][5])
                    : String.format("%9s", "n/a");

            System.out.println(String.format(java.util.Locale.ROOT, "  %-10s %9.4f %10.4f %8.4f %8.4f %s %s",
                    models[m], results[m][0], results[m][1], results[m][2], results[m][3], balancedAccuracy,
                    macroF1));
        }
    }

    /**
     * Evaluates a fitted model separately on each difficulty ("easy", "medium",
     * "hard"), printing a confusion matrix + metrics for each one (via
     * Metrics.formatBreakdownEntry), and returns the ";" joined summary that
     * gets stored in ModelMetrics.difficultyBreakdownCsv, so results.html can
     * show the exact same breakdown printed here.
     *
     * @param splitLabel               "Test" or "Validation", just for the console output
     * @param predictor                turns a feature matrix into 0/1 predictions
     * @param matricesByDifficulty     feature matrix per difficulty (missing/empty = skipped)
     * @param truthByDifficulty        truth values per difficulty
     * @return the per-difficulty breakdown, "" if no difficulty had any data
     */
    private static String evaluateByDifficulty(String splitLabel,
                                               java.util.function.Function<double[][], int[]> predictor,
                                               Map<String, double[][]> matricesByDifficulty, Map<String, int[]> truthByDifficulty) {

        // Break down for all the difficulties
        StringBuilder breakdown = new StringBuilder();

        for (String difficulty : DIFFICULTIES) {

            double[][] matrix = matricesByDifficulty.get(difficulty);
            int[] truth = truthByDifficulty.get(difficulty);

            if (matrix == null || matrix.length == 0) {
                continue;
            }

            System.out.println();
            System.out.println(splitLabel + " (" + difficulty + "):");

            int[] predictions = predictor.apply(matrix);
            String entry = Metrics.formatBreakdownEntry(difficulty, predictions, truth);

            if (breakdown.length() > 0) {
                breakdown.append(";");
            }
            breakdown.append(entry);
        }

        return breakdown.toString();
    }

    /**
     * The part that is the same for the three models: chooses the threshold on the validation
     * data (macro-F1), uses it ONCE on the test data, breaks the result down per difficulty,
     * saves the model and the metrics row.
     *
     * @param probabilities turns a pair feature matrix into the probabilities of a change
     * @return {accuracy, precision, recall, f1, balancedAccuracy, macroF1} of the test data (of the
     *         validation data when there is no test data), null if there is neither
     */
    private static double[] evaluateAndSave(String model, Object trainedModel, java.util.function.Function<double[][], double[]> probabilities,
            Corpus.Datasets d, String modelFilePath, String settings) {

        double threshold = 0.5;
        double[] result = null;

        if (d.val != null) {

            double[] validationProbabilities = probabilities.apply(d.val);

            System.out.println();
            System.out.println("Validation with the default threshold 0.5:");
            Metrics.evaluate(LogisticRegression.predict(validationProbabilities, 0.5), d.valY);

            // The threshold is chosen on the validation data only (best macro-F1, the PAN measure).
            threshold = LogisticRegression.findBestThreshold(validationProbabilities, d.valY);

            System.out.println();
            System.out.println("Validation with the best threshold " + threshold + ":");
            result = Metrics.evaluate(LogisticRegression.predict(validationProbabilities, threshold), d.valY);
        }

        // Test: used only once with the chosen threshold
        boolean onTest = d.test != null;
        final double finalThreshold = threshold;

        if (d.test != null) {

            double[] testProbabilities = probabilities.apply(d.test);

            System.out.println();
            System.out.println("Test with threshold " + threshold + ":");
            result = Metrics.evaluate(LogisticRegression.predict(testProbabilities, threshold), d.testY);
        }

        String difficultyBreakdown = evaluateByDifficulty(onTest ? "Test" : "Validation",
                matrix -> LogisticRegression.predict(probabilities.apply(matrix), finalThreshold),
                d.evalByDifficulty, d.evalYByDifficulty);

        /*
         * PAN scores every difficulty as its own dataset, so the threshold of a level is chosen on the validation
         * data of THAT level. The saved model keeps the pooled threshold (one number for the web app); these
         * per-level numbers are for the submission of one dataset: TiraRunner ... --threshold T
         */
        StringBuilder perLevel = new StringBuilder();
        java.util.Map<String, Double> levelThresholds = new java.util.LinkedHashMap<>();
        if (!d.valByDifficulty.isEmpty()) {
            System.out.println();
            System.out.println("Per level with the threshold chosen on that level's own validation data:");
            for (String difficulty : DIFFICULTIES) {
                double[][] levelVal = d.valByDifficulty.get(difficulty);
                double[][] levelEval = d.evalByDifficulty.get(difficulty);
                if (levelVal == null || levelVal.length == 0 || levelEval == null || levelEval.length == 0) {
                    continue;
                }
                double levelThreshold = LogisticRegression.findBestThreshold(probabilities.apply(levelVal),
                        d.valYByDifficulty.get(difficulty));
                double levelMacro = Metrics.evaluateQuietly(
                        LogisticRegression.predict(probabilities.apply(levelEval), levelThreshold),
                        d.evalYByDifficulty.get(difficulty))[5];
                System.out.println(String.format(java.util.Locale.ROOT,
                        "  %-6s threshold %.3f -> %s macro-F1 %.4f", difficulty, levelThreshold,
                        onTest ? "test" : "validation", levelMacro));
                perLevel.append(String.format(java.util.Locale.ROOT, " %s=%.3f(%.4f)", difficulty, levelThreshold,
                        levelMacro));
                levelThresholds.put(difficulty, levelThreshold);
            }
        }
        if (perLevel.length() > 0) {
            settings = settings + " | own threshold per level (macro-F1):" + perLevel;
        }
        // saved next to the model, for TiraRunner / the web app (models.LevelRouter); all three levels or none
        boolean levelsWritten = false;
        if (levelThresholds.size() == models.DifficultyDetector.LEVELS.length) {
            try {
                models.LevelRouter.writeLevels(new java.io.File(modelFilePath), levelThresholds,
                        "chosen on the first half of the validation split of each level; by Main");
                levelsWritten = true;
            } catch (IOException e) {
                System.err.println("Could not write the thresholds per level: " + e.getMessage());
            }
        }

        // Saving the model, into the same folder + format the web app's MlModelLoader reads from
        try {
            ModelStore.save(modelFilePath, model, trainedModel, threshold, d.extractor, settings);
            System.out.println();
            System.out.println("Model saved successfully to " + modelFilePath);
            if (levelsWritten) {
                System.out.println("Thresholds per level saved to " + models.LevelRouter.levelsFile(new java.io.File(modelFilePath))
                        + " (used with the difficulty detector: ./scripts/build.sh difficulty)");
            }

        } catch (IOException e) {
            System.err.println("Error " + e.getMessage());
            e.printStackTrace();
        }

        int nEvalPairs = onTest ? d.testY.length : (d.val != null ? d.valY.length : 0);
        saveMetricsRow(model, result, onTest, nEvalPairs, d.trainY.length, threshold, settings,
                d.extractor.getConfig().signature(), difficultyBreakdown);

        return result;
    }

    /**
     * Trains the decision tree, then evaluates it on the validation and test data
     * (combined AND broken down per difficulty) and saves it.
     *
     * @return {accuracy, precision, recall, f1, balancedAccuracy, macroF1} of the test data (of the validation
     *         data when there is no test data), null if there is neither
     */
    private static double[] trainDecisionTree(Corpus.Datasets d, String modelFilePath) {

        DecisionTree tree = new DecisionTree(d.train, d.trainY, TREE_MAX_DEPTH, TREE_MIN_SAMPLES_SPLIT,
                TREE_MIN_SAMPLES_LEAF, TREE_MIN_IMPURITY_DECREASE);
        tree.setMaxBins(TREE_MAX_BINS);
        tree.makeBinaryTree();

        System.out.println();
        System.out.println("Decision tree training complete.");
        System.out.println();
        System.out.println("Tree structure:");
        tree.printTree();
        System.out.println();

        // The probability of a leaf is the (class weighted) share of changes in it, so the threshold
        // can be chosen on the validation data like for the other two models.
        String settings = "max depth=" + TREE_MAX_DEPTH + ", minSamplesSplit=" + TREE_MIN_SAMPLES_SPLIT
                + ", minSamplesLeaf=" + TREE_MIN_SAMPLES_LEAF
                + ", splits=" + (TREE_MAX_BINS == 0 ? "exact" : "histogram(" + TREE_MAX_BINS + ")")
                + ", pair vector " + d.train[0].length;

        return evaluateAndSave(DECISION_TREE, tree, tree::predictProbabilities, d, modelFilePath, settings);
    }

    /**
     * Trains the logistic regression, chooses the threshold on the validation data,
     * evaluates it on the test data and saves it.
     *
     * @return {accuracy, precision, recall, f1, balancedAccuracy, macroF1} of the test data (of the validation
     *         data when there is no test data), null if there is neither
     */
    private static double[] trainLogisticRegression(Corpus.Datasets d, String modelFilePath) {

        System.out.println();
        System.out.println("Training Logistic Regression...");

        double[] weightsBias = LogisticRegression.gradiantDescent(d.train, d.trainY, N_ITERATIONS,
                LEARNING_RATE, USE_CLASS_WEIGHTS, LR_L2_LAMBDA, null);

        System.out.println();
        System.out.println("Logistic Regression training complete.");
        printTopWeights(weightsBias, d.extractor.pairFeatureNames());

        String settings = "iterations=" + N_ITERATIONS + ", learningRate=" + LEARNING_RATE
                + ", l2=" + LR_L2_LAMBDA + ", classWeights=" + USE_CLASS_WEIGHTS
                + ", pair vector " + d.train[0].length;

        return evaluateAndSave(LOGISTIC_REGRESSION, weightsBias,
                matrix -> LogisticRegression.predictProbabilities(matrix, weightsBias), d, modelFilePath, settings);
    }

    /**
     * Trains the neural network (early stopping on the validation data), chooses
     * the threshold on the validation data, evaluates it on the test data and saves
     * it.
     *
     * @return {accuracy, precision, recall, f1, balancedAccuracy, macroF1} of the test data (of the validation
     *         data when there is no test data), null if there is neither
     */
    private static double[] trainNeuralNetwork(Corpus.Datasets d, String modelFilePath) {

        System.out.println();
        System.out.println("Training Neural Network...");

        NeuralNetwork network = new NeuralNetwork(d.train, d.trainY, NN_HIDDEN_SIZES, NN_FEATURE_CLIP);
        network.setOptimizer(NN_OPTIMIZER);
        System.out.println("Architecture: " + network.describe() + " | optimizer: " + NN_OPTIMIZER
                + " | learning rate: " + NN_LEARNING_RATE);

        network.train(NN_MAX_EPOCHS, NN_LEARNING_RATE, NN_BATCH_SIZE, NN_L2_LAMBDA, NN_DROPOUT, NN_USE_CLASS_WEIGHTS,
                d.val, d.valY, NN_PATIENCE);

        System.out.println();
        System.out.println("Neural Network training complete.");

        // Training data: to see how far it is from the validation results
        // (a big gap = over-fitting)
        System.out.println();
        System.out.println("Training data with the default threshold 0.5:");
        Metrics.evaluate(LogisticRegression.predict(network.predictProbabilities(d.train), 0.5), d.trainY);

        String settings = "hidden=" + Arrays.toString(NN_HIDDEN_SIZES) + ", epochs=" + NN_MAX_EPOCHS
                + ", optimizer=" + NN_OPTIMIZER + ", learningRate=" + NN_LEARNING_RATE + ", l2=" + NN_L2_LAMBDA
                + ", dropout=" + NN_DROPOUT + ", pair vector " + d.train[0].length;

        return evaluateAndSave(NEURAL_NETWORK, network, network::predictProbabilities, d, modelFilePath, settings);
    }

    /**
     * Loads the saved model and predicts if there is a change between two
     * sentences.
     *
     * 0 = no author/style change
     * 1 = author/style change
     */
    private static void predictNewPair(String modelType, String modelFilePath, String sentence1, String sentence2) {

        // The label of a new pair is unknown, so -1 is only a placeholder.
        SentencePair testPair = new SentencePair(sentence1, sentence2, -1);

        ModelStore.Loaded loaded;

        try {
            loaded = ModelStore.load(modelFilePath);
            System.out.println("Model loaded from " + modelFilePath);

        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
            return; // nothing to predict with
        }

        double probability = loaded.probability(PairMatrixBuilder.getPairFeatures(testPair, loaded.extractor));
        int prediction = probability > loaded.threshold ? 1 : 0;

        System.out.println();
        System.out.println("Test sentence pair:");
        System.out.println("Sentence 1: " + sentence1);
        System.out.println("Sentence 2: " + sentence2);
        System.out.println(String.format(java.util.Locale.ROOT, "Prediction: %d (probability of a change %.3f, threshold %.3f)",
                prediction, probability, loaded.threshold));
    }

    // ---- The helpers below are kept with their old names, Ablation.java and the tests use them

    /** Builds the feature matrix of pairs, using the z-score parameters of the extractor (fitted on the training data). */
    static double[][] buildFeatureMatrix(List<SentencePair> pairs, FeatureExtractor extractor) {
        return PairMatrixBuilder.buildFeatureMatrix(pairs, extractor);
    }

    static double[][] buildTrainingMatrix(List<SentencePair> pairs, FeatureExtractor extractor) {
        return PairMatrixBuilder.buildTrainingMatrix(pairs, extractor);
    }

    static int[] getTruthValues(List<SentencePair> pairs) {
        return PairMatrixBuilder.getTruthValues(pairs);
    }

    /**
     * Prints the weights of the logistic regression with the largest size, by the NAME of the pair
     * feature (the whole list of weights is hundreds to thousands of numbers long).
     */
    private static void printTopWeights(double[] weightsBias, String[] names) {

        int nFeatures = weightsBias.length - 1;
        Integer[] order = new Integer[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(Math.abs(weightsBias[b]), Math.abs(weightsBias[a])));

        System.out.println("Bias: " + weightsBias[nFeatures]);
        System.out.println("Largest weights (the features are standardised, so the size can be compared):");
        for (int k = 0; k < Math.min(15, nFeatures); k++) {
            int i = order[k];
            System.out.println(String.format(java.util.Locale.ROOT, "  %-36s %9.4f", i < names.length ? names[i] : "feature " + i,
                    weightsBias[i]));
        }
    }

    /**
     * Prints the mean, standard deviation and the largest value of every feature,
     * to see if the scale of the features is reasonable (z-score differences
     * should be mostly between 0 and 3), and if training and validation look alike.
     */
    private static void printFeatureSummary(String name, double[][] matrix, FeatureExtractor extractor) {

        int nFeatures = matrix[0].length;
        // the bucket differences (full features) are summarised in one row
        int shownFeatures = Math.min(nFeatures, extractor.compressedPairCount());
        String[] names = extractor.pairFeatureNames();

        System.out.println(name + " feature summary (mean / std / max):");

        for (int feature = 0; feature < shownFeatures; feature++) {
            printFeatureRow(String.format("%-34s", names[feature]), matrix, feature, feature + 1);
        }

        if (nFeatures > shownFeatures) {
            printFeatureRow(String.format("features %d-%d (bucket differences, averaged)", shownFeatures,
                    nFeatures - 1), matrix, shownFeatures, nFeatures);
        }
    }

    /**
     * Prints the mean, standard deviation (both averaged over the columns) and the
     * largest value of the columns from (included) to (excluded).
     */
    private static void printFeatureRow(String label, double[][] matrix, int from, int to) {

        double meanSum = 0.0;
        double stdSum = 0.0;
        double max = Double.NEGATIVE_INFINITY;

        for (int feature = from; feature < to; feature++) {

            double sum = 0.0;

            for (double[] row : matrix) {
                sum += row[feature];
                max = Math.max(max, row[feature]);
            }

            double mean = sum / matrix.length;
            double squares = 0.0;

            for (double[] row : matrix) {
                squares += (row[feature] - mean) * (row[feature] - mean);
            }

            meanSum += mean;
            stdSum += Math.sqrt(squares / matrix.length);
        }

        System.out.println(String.format(java.util.Locale.ROOT, "  %s: %9.4f / %9.4f / %9.4f", label,
                meanSum / (to - from), stdSum / (to - from), max));
    }

    /**
     * Helper method
     * @param name The AuthorSwitch (based on the ones or zeros)
     * @param truthValues
     */
    private static void printClassBalance(String name, int[] truthValues) {

        int ones = 0;
        for (int truth : truthValues) {
            ones += truth;
        }

        System.out.println(String.format("%s: %d pairs, %d changes (%.2f%%)", name, truthValues.length, ones,
                100.0 * ones / truthValues.length));
    }

    /**
     * The macro-F1 that predicting a constant gets (the official PAN baselines): everything is "no change",
     * everything is "change". A model has to be clearly above both.
     */
    private static void printBaselines(String name, int[] truthValues) {

        int[] allZero = new int[truthValues.length];
        int[] allOne = new int[truthValues.length];
        Arrays.fill(allOne, 1);

        System.out.println(String.format(java.util.Locale.ROOT,
                "%s baselines (macro-F1): always 'no change' = %.4f | always 'change' = %.4f", name,
                Metrics.calcMacroF1(allZero, truthValues), Metrics.calcMacroF1(allOne, truthValues)));
    }
}
