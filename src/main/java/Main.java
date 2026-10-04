import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import data.DatasetLoader;
import data.SentencePair;
import features.FeatureExtractor;
import models.DecisionTree;
import models.LogisticRegression;
import models.Metrics;
import models.NeuralNetwork;

import com.hyp.authorswitchdetection.model.ModelMetrics;
import com.hyp.authorswitchdetection.model.ModelMetricsIO;

/*TODO: (1) Utility for saving the model And loading the model.*/
/*TODO: (1) Importance(G)=F1_all − F1_without G
    That tells you much more directly whether the group contributes useful predictive information. 
*/

/*TODO: (1) Parameter learning for all the Models */
/*TODO: (1) Difficulty levels: Provide the reporting based on EASY, MEDIUM and HARD */

/*TODO: (1) Seed variations (42, 123, 456) */

/* 
    
    From Previous Logistic Regression Run Not this one Actually:

    The model achieves 92.88% overall accuracy, substantially exceeding the 69.01% majority-class baseline. 
    It achieves 89.13% precision, 87.72% recall, and an F1-score of 88.42%, indicating that it can identify 
    author-switch instances while maintaining a relatively low number of false-positive predictions.

*/

public class Main {

    // Path to the MAWSA dataset.
    private static final String DATASET_PATH = "dataset/mawsa26-pan-zenodo-DATA/";

    /*
        Every difficulty level is combined for training/validation/test and the final evaluation is also broken down
        per difficulty (see evaluateByDifficulty) and not just reported combined.

     */
    private static final String[] DIFFICULTIES = { "easy", "medium", "hard" };

    // The models that can be trained
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
     */
    private static final String MODEL_TYPE = ALL_MODELS;

    /*
     * false: the validation set is divided in two halves.
     * 1st half = validation (choosing the threshold), 2nd half = test (unused data).
     * true: the FULL validation set is used for validation and no test is done,
     * for when the real testing dataset is available.
     */
    private static final boolean USE_FULL_VALIDATION = false;

    // TODO: parameter learning

    /*
     * false: 11 features per pair (9 z-score differences, n-gram cosine, overall
     * distance)
     * true: all features (the 11 above + 256 n-gram bucket differences), not
     * compressed. A saved model has to be trained again after changing this.
     */
    // false = only the 11 compressed features (much less memory and much faster
    // model training). The 256 n-gram bucket differences are left out, but the
    // 9 scalars (incl. TF-IDF) and the n-gram cosine are still calculated, since
    // those are part of the 11. Set to true again to bring the 267 back.
    private static final boolean USE_FULL_FEATURES = false;

    // The file the model is saved to (and loaded from for the inference), in the
    // SAME folder + on-disk format the Spring web app's MlModelLoader reads from
    // (app.trained-models.dir, default "data/models"), so a model trained here
    // from the command line is picked up for live prediction too, not just for
    // the metrics on the Results page.
    private static final String MODELS_DIR = "data/models/";

    // Where trained-run metrics are appended so they show up on the web app's
    // Results page (same file the Spring app's ModelMetricsStore reads/writes:
    // app.model-metrics.file, default "data/model-metrics.json").
    private static final String METRICS_FILE = "data/model-metrics.json";

    // Data settings. At most this many training pairs (and per difficulty, this
    // many validation / test pairs) are used. For a quick trial run:
    // java -Dsubset=50000 -cp out Main tree
    // Default = no limit: ALL the pairs are used (the memory guard below still
    // shrinks everything if the heap is too small). Give more heap with -Xmx.
    private static final int SUBSET_SIZE = Integer.getInteger("subset", Integer.MAX_VALUE);

    /*
     * The feature matrices (training + validation + test) may use at most this
     * part of the JVM's max heap (-Xmx). The rest is left for the raw text while
     * it is being loaded, for the models while they train, and for the garbage
     * collector. If the data wanted does not fit, ALL the sets are shrunk by the
     * same fraction (see fractionThatFitsInMemory), so the run finishes instead
     * of ending with "java.lang.OutOfMemoryError: Java heap space".
     */
    private static final double HEAP_FRACTION_FOR_MATRICES = 0.5;

    // How many training pairs are recomputed the slow (original) way at the end
    // of buildTrainingMatrix, to check the memory-friendly way gives the same numbers
    private static final int VERIFY_SAMPLE = 200;

    // Logistic Regression settings
    private static final int N_ITERATIONS = 10000;
    private static final double LEARNING_RATE = 0.6;
    private static final boolean USE_CLASS_WEIGHTS = true;

    /*
     * Neural Network settings (input -> hidden layers with ReLU -> 1 sigmoid
     * output). The validation data is used for early stopping, so the network
     * stops when the validation loss no longer improves.
     */
    private static final int[] NN_HIDDEN_SIZES = { 128, 64, 32, 16, 8 };
    private static final int NN_MAX_EPOCHS = 60;
    private static final double NN_LEARNING_RATE = 0.001; // learning rate of the plain mini batch gradient descent (no Adam)
    private static final int NN_BATCH_SIZE = 64;
    private static final double NN_L2_LAMBDA = 1e-2; // 0 = no regularisation
    private static final double NN_DROPOUT = 0.2; // chance a hidden neuron is switched off, 0 = no dropout
    private static final double NN_FEATURE_CLIP = 5.0; // features are clipped to [-5, 5], 0 = no clipping
    private static final int NN_PATIENCE = 8; // 0 = no early stopping
    private static final boolean NN_USE_CLASS_WEIGHTS = true;

    // Decision Tree settings: maxDepth, minSamplesSplit, minSamplesLeaf,
    // minImpurityDecrease
    private static final int TREE_MAX_DEPTH = 5;
    private static final int TREE_MIN_SAMPLES_SPLIT = 20;
    private static final int TREE_MIN_SAMPLES_LEAF = 5;
    private static final double TREE_MIN_IMPURITY_DECREASE = 0.0;

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

        /*
         * The extractor first: all it keeps from the corpus is the vocabulary / IDF
         * table, so the corpus text itself is dropped straight after.
         */
        List<String> trainingCorpus = DatasetLoader.getAllSentences(DATASET_PATH, "train");
        System.out.println("Training corpus size (all difficulties): " + trainingCorpus.size());

        FeatureExtractor extractor = new FeatureExtractor(trainingCorpus);
        trainingCorpus = null;

        /*
         * Load the pairs (only the text, the features come later) — ALL 3 difficulty
         * levels combined (same as the web app's RealTrainer), not just "easy".
         * Validation & test are loaded per difficulty (so each difficulty can be
         * evaluated separately below) AND combined (for the overall numbers).
         */
        List<SentencePair> dataset = new ArrayList<>();
        for (String difficulty : DIFFICULTIES) {
            dataset.addAll(DatasetLoader.loadSplit(DATASET_PATH, "train", difficulty));
        }

        System.out.println("Number of training pairs: " + dataset.size());
        if (dataset.isEmpty()) {
            System.err.println("No training data was loaded.");
            return;
        }

        Map<String, List<SentencePair>> validationByDifficulty = new LinkedHashMap<>();
        Map<String, List<SentencePair>> testByDifficulty = new LinkedHashMap<>();

        for (String difficulty : DIFFICULTIES) {

            List<SentencePair> val = USE_FULL_VALIDATION
                    ? DatasetLoader.loadSplit(DATASET_PATH, "validation", difficulty)
                    : DatasetLoader.loadSplitHalf(DATASET_PATH, "validation", difficulty, true);
            validationByDifficulty.put(difficulty, val);

            if (!USE_FULL_VALIDATION) {
                // the 2nd half of the validation set, used only once at the end
                testByDifficulty.put(difficulty,
                        DatasetLoader.loadSplitHalf(DATASET_PATH, "validation", difficulty, false));
            }
        }

        /*
         * How much of it fits in memory: the wanted sizes are SUBSET_SIZE at most,
         * and if that is more than the heap can hold, everything is shrunk by the
         * same fraction (the pairs are shuffled with a fixed seed first, so the
         * smaller sets are still a fair picture of the whole).
         */
        long wantedTrain = Math.min(dataset.size(), SUBSET_SIZE);
        long wantedEvaluation = 0;
        for (List<SentencePair> v : validationByDifficulty.values()) {
            wantedEvaluation += Math.min(v.size(), SUBSET_SIZE);
        }
        for (List<SentencePair> t : testByDifficulty.values()) {
            wantedEvaluation += Math.min(t.size(), SUBSET_SIZE);
        }

        int pairFeatureCount = FeatureExtractor.COMPRESSED_PAIR_COUNT;
        if (USE_FULL_FEATURES) {
            pairFeatureCount += extractor.getFeatures("x").length - FeatureExtractor.SCALAR_COUNT;
        }

        double fraction = fractionThatFitsInMemory(wantedTrain, wantedEvaluation, pairFeatureCount);

        dataset = takeSubset(dataset, (int) (wantedTrain * fraction));
        for (String difficulty : DIFFICULTIES) {
            List<SentencePair> val = validationByDifficulty.get(difficulty);
            validationByDifficulty.put(difficulty, takeSubset(val, (int) (Math.min(val.size(), SUBSET_SIZE) * fraction)));

            if (!USE_FULL_VALIDATION) {
                List<SentencePair> test = testByDifficulty.get(difficulty);
                testByDifficulty.put(difficulty, takeSubset(test, (int) (Math.min(test.size(), SUBSET_SIZE) * fraction)));
            }
        }

        System.out.println("Using " + dataset.size() + " training pairs"
                + (fraction < 1.0 ? " (shrunk to fit the heap; run with a bigger -Xmx to use all)" : " (all of them)") + ".");

        System.out.println();
        System.out.println("Extracting features for " + dataset.size() + " pairs...");

        double[][] featureMatrix = buildTrainingMatrix(dataset, extractor); // X
        int[] truthValues = getTruthValues(dataset); // y
        dataset = null; // the raw text is not needed anymore, only the numbers

        printClassBalance("Training", truthValues);
        printFeatureSummary("Training", featureMatrix);

        /*
         * Validation & test data, both not used for training (nor for the z-score
         * parameters). Every difficulty's matrix is built ONCE, and the combined one
         * is just those same rows in one array (no second copy of the numbers).
         */
        Map<String, double[][]> validationMatrixByDifficulty = new LinkedHashMap<>();
        Map<String, int[]> validationTruthByDifficulty = new LinkedHashMap<>();

        for (String difficulty : DIFFICULTIES) {
            List<SentencePair> pairs = validationByDifficulty.get(difficulty);
            if (pairs != null && !pairs.isEmpty()) {
                validationMatrixByDifficulty.put(difficulty, buildFeatureMatrix(pairs, extractor));
                validationTruthByDifficulty.put(difficulty, getTruthValues(pairs));
            }
        }
        validationByDifficulty = null;

        double[][] validationMatrix = null;
        int[] validationTruthValues = null;

        if (!validationMatrixByDifficulty.isEmpty()) {
            validationMatrix = joinMatrices(validationMatrixByDifficulty.values());
            validationTruthValues = joinTruth(validationTruthByDifficulty.values());

            System.out.println();
            System.out.println("Number of validation pairs: " + validationMatrix.length);
            printClassBalance("Validation", validationTruthValues);
            printFeatureSummary("Validation", validationMatrix);
        } else {
            System.out.println();
            System.out.println("Number of validation pairs: 0");
        }

        double[][] testMatrix = null;
        int[] testTruthValues = null;
        Map<String, double[][]> testMatrixByDifficulty = new LinkedHashMap<>();
        Map<String, int[]> testTruthByDifficulty = new LinkedHashMap<>();

        if (!USE_FULL_VALIDATION) {

            for (String difficulty : DIFFICULTIES) {
                List<SentencePair> pairs = testByDifficulty.get(difficulty);
                if (pairs != null && !pairs.isEmpty()) {
                    testMatrixByDifficulty.put(difficulty, buildFeatureMatrix(pairs, extractor));
                    testTruthByDifficulty.put(difficulty, getTruthValues(pairs));
                }
            }
            testByDifficulty = null;

            if (!testMatrixByDifficulty.isEmpty()) {
                testMatrix = joinMatrices(testMatrixByDifficulty.values());
                testTruthValues = joinTruth(testTruthByDifficulty.values());
            }

            System.out.println("Number of test pairs: " + (testMatrix == null ? 0 : testMatrix.length));

            if (testMatrix != null) {
                printClassBalance("Test", testTruthValues);
            }
        }

        // whichever split the FINAL, combined numbers below come from (test when
        // available, else validation) is also what the per-difficulty breakdown is
        // computed from, so the two are always directly comparable
        Map<String, double[][]> breakdownMatrixByDifficulty = testMatrix != null ? testMatrixByDifficulty
                : validationMatrixByDifficulty;
        Map<String, int[]> breakdownTruthByDifficulty = testMatrix != null ? testTruthByDifficulty
                : validationTruthByDifficulty;

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

                results[m] = trainDecisionTree(featureMatrix, truthValues, validationMatrix, validationTruthValues,
                        testMatrix, testTruthValues, extractor, modelFilePath, breakdownMatrixByDifficulty,
                        breakdownTruthByDifficulty);

            } else if (model.equals(NEURAL_NETWORK)) {

                results[m] = trainNeuralNetwork(featureMatrix, truthValues, validationMatrix, validationTruthValues,
                        testMatrix, testTruthValues, extractor, modelFilePath, breakdownMatrixByDifficulty,
                        breakdownTruthByDifficulty);

            } else {

                results[m] = trainLogisticRegression(featureMatrix, truthValues, validationMatrix,
                        validationTruthValues, testMatrix, testTruthValues, extractor, modelFilePath,
                        breakdownMatrixByDifficulty, breakdownTruthByDifficulty);
            }

            /*
             * For Inference: reload the model that was just saved and predict a new pair.
             */
            predictNewPair(model, modelFilePath, sentence1, sentence2);
        }

        if (modelsToRun.length > 1) {
            printSummary(modelsToRun, results, testMatrix != null);
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
     * @param result {accuracy, precision, recall, f1, balancedAccuracy} from Metrics.evaluate / DecisionTree.evaluate, or null
     * @param onTest true if result came from the held-out test split
     * @param nEvalPairs number of pairs result was computed on
     * @param nTrainPairs number of training pairs used
     * @param threshold decision threshold used (0.5 for the tree)
     * @param chosenConfig short human-readable description of the run's settings
     * @param difficultyBreakdown ";" joined per-difficulty summary (see evaluateByDifficulty), "" if none
     */
    private static void saveMetricsRow(String model, double[] result, boolean onTest, int nEvalPairs,
                                       int nTrainPairs, double threshold, String chosenConfig, String difficultyBreakdown) {

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
        m.setDecisionThreshold(threshold);
        m.setEvaluatedOnTest(onTest);
        m.setNTestPairs(nEvalPairs);
        m.setDifficultyBreakdownCsv(difficultyBreakdown);
        m.setTrainedBy("cli");
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
     * @param results {accuracy, precision, recall, f1, balancedAccuracy} of each
     *                model (null = no result)
     * @param onTest  true: the results are from the test data, false: from the
     *                validation data
     */
    private static void printSummary(String[] models, double[][] results, boolean onTest) {

        System.out.println();
        System.out.println("==================== Summary (" + (onTest ? "test" : "validation") + " data) "
                + "====================");
        System.out.println(String.format("  %-10s %9s %10s %8s %8s %9s", "Model", "Accuracy", "Precision", "Recall",
                "F1", "Bal.Acc"));

        for (int m = 0; m < models.length; m++) {

            if (results[m] == null) {
                System.out.println(String.format("  %-10s %s", models[m], "no result"));
                continue;
            }

            // older result arrays (e.g. before balanced accuracy was added) only have 4
            String balancedAccuracy = results[m].length > 4
                    ? String.format(java.util.Locale.ROOT, "%9.4f", results[m][4])
                    : String.format("%9s", "n/a");

            System.out.println(String.format(java.util.Locale.ROOT, "  %-10s %9.4f %10.4f %8.4f %8.4f %s", models[m],
                    results[m][0], results[m][1], results[m][2], results[m][3], balancedAccuracy));
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
     * Trains the decision tree, then evaluates it on the validation and test data
     * (combined AND broken down per difficulty) and saves it.
     *
     * @return {accuracy, precision, recall, f1} of the test data (of the validation
     *         data when there is no test data), null if there is neither
     */
    private static double[] trainDecisionTree(double[][] featureMatrix, int[] truthValues, double[][] validationMatrix,
                                              int[] validationTruthValues, double[][] testMatrix, int[] testTruthValues, FeatureExtractor extractor,
                                              String modelFilePath, Map<String, double[][]> breakdownMatrixByDifficulty,
                                              Map<String, int[]> breakdownTruthByDifficulty) {

        DecisionTree tree = new DecisionTree(featureMatrix, truthValues, TREE_MAX_DEPTH, TREE_MIN_SAMPLES_SPLIT,
                TREE_MIN_SAMPLES_LEAF, TREE_MIN_IMPURITY_DECREASE);
        tree.makeBinaryTree();

        double[] result = null;

        System.out.println();
        System.out.println("Decision tree training complete.");
        System.out.println();
        System.out.println("Tree structure:");
        tree.printTree();
        System.out.println();

        if (validationMatrix != null) {
            System.out.println("Validation:");
            result = tree.evaluate(validationMatrix, validationTruthValues);
        }

        boolean onTest = testMatrix != null;

        if (testMatrix != null) {
            System.out.println();
            System.out.println("Test:");
            result = tree.evaluate(testMatrix, testTruthValues);
        }

        String difficultyBreakdown = evaluateByDifficulty(onTest ? "Test" : "Validation", tree::predictAll,
                breakdownMatrixByDifficulty, breakdownTruthByDifficulty);

        // Saving decision tree model, into the same folder + format the web app's
        // MlModelLoader reads from (fullFeatures flag written last)
        new java.io.File(modelFilePath).getParentFile().mkdirs();

        try (FileOutputStream fileOut = new FileOutputStream(modelFilePath);
             BufferedOutputStream bos = new BufferedOutputStream(fileOut);
             ObjectOutputStream oos = new ObjectOutputStream(bos)) {

            oos.writeObject(tree);
            oos.writeObject(extractor);
            oos.writeBoolean(USE_FULL_FEATURES);
            System.out.println("Model saved successfully to " + modelFilePath);

        } catch (IOException e) {
            System.err.println("Error " + e.getMessage());
            e.printStackTrace();
        }

        int nEvalPairs = onTest ? testTruthValues.length : (validationMatrix != null ? validationTruthValues.length : 0);
        saveMetricsRow(DECISION_TREE, result, onTest, nEvalPairs, truthValues.length, 0.5,
                "max depth=" + TREE_MAX_DEPTH + ", minSamplesSplit=" + TREE_MIN_SAMPLES_SPLIT
                        + ", minSamplesLeaf=" + TREE_MIN_SAMPLES_LEAF
                        + ", " + (USE_FULL_FEATURES ? "full" : "compressed") + " features",
                difficultyBreakdown);

        return result;
    }

    /**
     * Trains the logistic regression, chooses the threshold on the validation data,
     * evaluates it on the test data and saves it.
     *
     * @return {accuracy, precision, recall, f1} of the test data (of the validation
     *         data when there is no test data), null if there is neither
     */
    private static double[] trainLogisticRegression(double[][] featureMatrix, int[] truthValues,
                                                    double[][] validationMatrix, int[] validationTruthValues, double[][] testMatrix, int[] testTruthValues,
                                                    FeatureExtractor extractor, String modelFilePath, Map<String, double[][]> breakdownMatrixByDifficulty,
                                                    Map<String, int[]> breakdownTruthByDifficulty) {

        System.out.println();
        System.out.println("Training Logistic Regression...");

        double[] weightsBias = LogisticRegression.gradiantDescent(featureMatrix, truthValues, N_ITERATIONS,
                LEARNING_RATE, USE_CLASS_WEIGHTS);

        System.out.println();
        System.out.println("Logistic Regression training complete.");
        System.out.println(
                "Weights: " + java.util.Arrays.toString(java.util.Arrays.copyOf(weightsBias, weightsBias.length - 1)));
        System.out.println("Bias: " + weightsBias[weightsBias.length - 1]);

        double threshold = 0.5;
        double[] result = null;

        if (validationMatrix != null) {

            double[] probabilities = LogisticRegression.predictProbabilities(validationMatrix, weightsBias);

            System.out.println();
            System.out.println("Validation with the default threshold 0.5:");
            Metrics.evaluate(LogisticRegression.predict(probabilities, 0.5), validationTruthValues);

            // The threshold is chosen on the validation data only.
            threshold = LogisticRegression.findBestThreshold(probabilities, validationTruthValues);

            System.out.println();
            System.out.println("Validation with the best threshold " + threshold + ":");
            result = Metrics.evaluate(LogisticRegression.predict(probabilities, threshold), validationTruthValues);
        }

        // Test: used only once with the chosen threshold
        boolean onTest = testMatrix != null;
        double finalThreshold = threshold;

        if (testMatrix != null) {

            double[] probabilities = LogisticRegression.predictProbabilities(testMatrix, weightsBias);

            System.out.println();
            System.out.println("Test with threshold " + threshold + ":");
            result = Metrics.evaluate(LogisticRegression.predict(probabilities, threshold), testTruthValues);
        }

        String difficultyBreakdown = evaluateByDifficulty(onTest ? "Test" : "Validation",
                matrix -> LogisticRegression.predict(LogisticRegression.predictProbabilities(matrix, weightsBias), finalThreshold),
                breakdownMatrixByDifficulty, breakdownTruthByDifficulty);

        // Saving the model, into the same folder + format the web app's
        // MlModelLoader reads from (fullFeatures flag written last)
        new java.io.File(modelFilePath).getParentFile().mkdirs();

        try (FileOutputStream fileOut = new FileOutputStream(modelFilePath);
             BufferedOutputStream bos = new BufferedOutputStream(fileOut);
             ObjectOutputStream oos = new ObjectOutputStream(bos)) {

            oos.writeObject(weightsBias);
            oos.writeDouble(threshold);
            oos.writeObject(extractor);
            oos.writeBoolean(USE_FULL_FEATURES);
            System.out.println();
            System.out.println("Model saved successfully to " + modelFilePath);

        } catch (IOException e) {
            System.err.println("Error " + e.getMessage());
            e.printStackTrace();
        }

        int nEvalPairs = onTest ? testTruthValues.length : (validationMatrix != null ? validationTruthValues.length : 0);
        saveMetricsRow(LOGISTIC_REGRESSION, result, onTest, nEvalPairs, truthValues.length, threshold,
                "iterations=" + N_ITERATIONS + ", learningRate=" + LEARNING_RATE
                        + ", classWeights=" + USE_CLASS_WEIGHTS
                        + ", " + (USE_FULL_FEATURES ? "full" : "compressed") + " features",
                difficultyBreakdown);

        return result;
    }

    /**
     * Trains the neural network (early stopping on the validation data), chooses
     * the threshold on the validation data, evaluates it on the test data and saves
     * it.
     *
     * @return {accuracy, precision, recall, f1} of the test data (of the validation
     *         data when there is no test data), null if there is neither
     */
    private static double[] trainNeuralNetwork(double[][] featureMatrix, int[] truthValues, double[][] validationMatrix,
                                               int[] validationTruthValues, double[][] testMatrix, int[] testTruthValues, FeatureExtractor extractor,
                                               String modelFilePath, Map<String, double[][]> breakdownMatrixByDifficulty,
                                               Map<String, int[]> breakdownTruthByDifficulty) {

        System.out.println();
        System.out.println("Training Neural Network...");

        NeuralNetwork network = new NeuralNetwork(featureMatrix, truthValues, NN_HIDDEN_SIZES, NN_FEATURE_CLIP);
        System.out.println("Architecture: " + network.describe());

        network.train(NN_MAX_EPOCHS, NN_LEARNING_RATE, NN_BATCH_SIZE, NN_L2_LAMBDA, NN_DROPOUT, NN_USE_CLASS_WEIGHTS,
                validationMatrix, validationTruthValues, NN_PATIENCE);

        System.out.println();
        System.out.println("Neural Network training complete.");

        // Training data: to see how far it is from the validation results
        // (a big gap = over-fitting)
        System.out.println();
        System.out.println("Training data with the default threshold 0.5:");
        Metrics.evaluate(LogisticRegression.predict(network.predictProbabilities(featureMatrix), 0.5), truthValues);

        double threshold = 0.5;
        double[] result = null;

        if (validationMatrix != null) {

            double[] probabilities = network.predictProbabilities(validationMatrix);

            System.out.println();
            System.out.println("Validation with the default threshold 0.5:");
            Metrics.evaluate(LogisticRegression.predict(probabilities, 0.5), validationTruthValues);

            // The threshold is chosen on the validation data only.
            threshold = LogisticRegression.findBestThreshold(probabilities, validationTruthValues);

            System.out.println();
            System.out.println("Validation with the best threshold " + threshold + ":");
            result = Metrics.evaluate(LogisticRegression.predict(probabilities, threshold), validationTruthValues);
        }

        // Test: used only once with the chosen threshold
        boolean onTest = testMatrix != null;
        double finalThreshold = threshold;

        if (testMatrix != null) {

            double[] probabilities = network.predictProbabilities(testMatrix);

            System.out.println();
            System.out.println("Test with threshold " + threshold + ":");
            result = Metrics.evaluate(LogisticRegression.predict(probabilities, threshold), testTruthValues);
        }

        String difficultyBreakdown = evaluateByDifficulty(onTest ? "Test" : "Validation",
                matrix -> LogisticRegression.predict(network.predictProbabilities(matrix), finalThreshold),
                breakdownMatrixByDifficulty, breakdownTruthByDifficulty);

        // Saving the model, into the same folder + format the web app's
        // MlModelLoader reads from (fullFeatures flag written last)
        new java.io.File(modelFilePath).getParentFile().mkdirs();

        try (FileOutputStream fileOut = new FileOutputStream(modelFilePath);
             BufferedOutputStream bos = new BufferedOutputStream(fileOut);
             ObjectOutputStream oos = new ObjectOutputStream(bos)) {

            oos.writeObject(network);
            oos.writeDouble(threshold);
            oos.writeObject(extractor);
            oos.writeBoolean(USE_FULL_FEATURES);
            System.out.println();
            System.out.println("Model saved successfully to " + modelFilePath);

        } catch (IOException e) {
            System.err.println("Error " + e.getMessage());
            e.printStackTrace();
        }

        int nEvalPairs = onTest ? testTruthValues.length : (validationMatrix != null ? validationTruthValues.length : 0);
        saveMetricsRow(NEURAL_NETWORK, result, onTest, nEvalPairs, truthValues.length, threshold,
                "hidden=" + java.util.Arrays.toString(NN_HIDDEN_SIZES) + ", epochs=" + NN_MAX_EPOCHS
                        + ", " + (USE_FULL_FEATURES ? "full" : "compressed") + " features",
                difficultyBreakdown);

        return result;
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

        int prediction;

        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(modelFilePath)))) {

            if (modelType.equals(DECISION_TREE)) {

                DecisionTree treeTest = (DecisionTree) ois.readObject();
                FeatureExtractor extractorTest = (FeatureExtractor) ois.readObject();
                System.out.println("Model loaded from " + modelFilePath);

                prediction = treeTest.predict(getPairFeatures(testPair, extractorTest));

            } else if (modelType.equals(NEURAL_NETWORK)) {

                NeuralNetwork networkTest = (NeuralNetwork) ois.readObject();
                double thresholdTest = ois.readDouble();
                FeatureExtractor extractorTest = (FeatureExtractor) ois.readObject();
                System.out.println("Model loaded from " + modelFilePath);

                prediction = networkTest.predict(getPairFeatures(testPair, extractorTest), thresholdTest);

            } else {

                double[] weightsBiasTest = (double[]) ois.readObject();
                double thresholdTest = ois.readDouble();
                FeatureExtractor extractorTest = (FeatureExtractor) ois.readObject();
                System.out.println("Model loaded from " + modelFilePath);

                prediction = LogisticRegression.predict(getPairFeatures(testPair, extractorTest), weightsBiasTest,
                        thresholdTest);
            }

        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
            return; // nothing to predict with
        }

        System.out.println();
        System.out.println("Test sentence pair:");
        System.out.println("Sentence 1: " + sentence1);
        System.out.println("Sentence 2: " + sentence2);
        System.out.println("Prediction: " + prediction);
    }

    /**
     * Pairwise representation of one pair, using the (LOADED) extractor.
     */
    private static double[] getPairFeatures(SentencePair pair, FeatureExtractor extractor) {

        double[] features1 = extractor.standardize(extractor.getFeatures(pair.getSentence1()));
        double[] features2 = extractor.standardize(extractor.getFeatures(pair.getSentence2()));

        double[] pairFeatures = makePairFeatures(pair, features1, features2);

        // same n-gram difference z-score parameters as the training data
        if (USE_FULL_FEATURES) {
            pairFeatures = extractor.standardizeNGramDifferences(pairFeatures);
        }

        return pairFeatures;
    }

    /**
     * Builds the feature matrix of pairs, using the z-score parameters of the extractor (fitted on the training data).
     */
    private static double[][] buildFeatureMatrix(List<SentencePair> pairs, FeatureExtractor extractor) {

        double[][] matrix = new double[pairs.size()][];

        for (int i = 0; i < pairs.size(); i++) {
            matrix[i] = getPairFeatures(pairs.get(i), extractor);
        }

        return matrix;
    }

    private static int[] getTruthValues(List<SentencePair> pairs) {

        int[] truthValues = new int[pairs.size()];

        for (int i = 0; i < pairs.size(); i++) {
            truthValues[i] = pairs.get(i).getLabel();
        }

        return truthValues;
    }

    /**
     * Takes at most limit number of pairs. The pairs are shuffled with a fixed seed first,
     * because the first pairs of the list come from only a few documents and are
     * not a good picture of the whole split. The chosen pairs are COPIED into a
     * new list, so the pairs that were not chosen (and the shuffled list) can be
     * garbage collected instead of staying alive behind a subList view.
     */
    private static List<SentencePair> takeSubset(List<SentencePair> pairs, int limit) {

        if (pairs.size() <= limit) {
            return pairs;
        }

        List<SentencePair> shuffled = new ArrayList<>(pairs);
        Collections.shuffle(shuffled, new Random(42L));

        return new ArrayList<>(shuffled.subList(0, limit));
    }

    /**
     * Helper method
     * Which fraction (0 to 1) of the wanted training / validation / test pairs the heap can hold as
     * feature matrices. 1.0 = everything fits.
     *
     * @param nTrain           training pairs wanted
     * @param nEvaluation      validation + test pairs wanted
     * @param pairFeatureCount numbers in one pair's feature vector (267 or 11)
     */
    private static double fractionThatFitsInMemory(long nTrain, long nEvaluation, int pairFeatureCount) {

        long rowBytes = 8L * pairFeatureCount + 16; // the doubles + the array header

        // while the training matrix is built each pair also keeps the raw scalars of
        // its two sentences (see buildTrainingMatrix), and its label
        long trainBytes = rowBytes + 2 * (8L * FeatureExtractor.SCALAR_COUNT + 16) + 4;

        long wanted = nTrain * trainBytes + nEvaluation * rowBytes;
        long budget = (long) (Runtime.getRuntime().maxMemory() * HEAP_FRACTION_FOR_MATRICES);

        System.out.println();
        System.out.println(String.format("Memory: max heap %d MB, the feature matrices may use %d MB.",
                Runtime.getRuntime().maxMemory() / (1024 * 1024), budget / (1024 * 1024)));
        System.out.println(String.format("        All the wanted data needs about %d MB (run with -Xmx%dm to use it all).",
                wanted / (1024 * 1024), (long) Math.ceil(wanted / HEAP_FRACTION_FOR_MATRICES / (1024 * 1024))));

        if (wanted <= budget) {
            return 1.0;
        }

        double fraction = (double) budget / wanted;
        System.out.println(String.format("        That does not fit: using %.1f%% of the training, validation and test "
                + "pairs, so the run does not run out of memory.", fraction * 100));

        return fraction;
    }

    /**
     * Feature matrix of the TRAINING pairs, the same numbers as
     * buildFeatureMatrix would give, but without ever holding the full feature
     * vectors of all the sentences (2 x 265 numbers per pair) next to the pair
     * matrix (267 numbers per pair), which more than doubled the memory needed.
     */
    static double[][] buildTrainingMatrix(List<SentencePair> pairs, FeatureExtractor extractor) {

        int scalarCount = FeatureExtractor.SCALAR_COUNT;
        int n = pairs.size();

        double[][] featureMatrix = new double[n][];
        // I need to make a 2D so that I can concatenate a 2D
        double[][] scalars1 = new double[n][];
        double[][] scalars2 = new double[n][];
        List<double[]> trainingScalars = new ArrayList<>(2 * n);

        for (int i = 0; i < n; i++) {

            double[] features1 = extractor.getFeatures(pairs.get(i).getSentence1());
            double[] features2 = extractor.getFeatures(pairs.get(i).getSentence2());

            scalars1[i] = java.util.Arrays.copyOf(features1, scalarCount);
            scalars2[i] = java.util.Arrays.copyOf(features2, scalarCount);
            trainingScalars.add(scalars1[i]);
            trainingScalars.add(scalars2[i]);

            // The n-gram part of a pair vector does not use the z-scores standardising.
            // The z-score slots ([0..8] and [10]) are filled in below.
            featureMatrix[i] = makePairFeatures(pairs.get(i), features1, features2);

            if ((i + 1) % 10000 == 0 || i == n - 1) {
                System.out.println("  Processed " + (i + 1) + " / " + n);
            }
        }

        // The z-score parameters are fitted on the TRAINING sentences only.
        extractor.fitScaler(trainingScalars);
        trainingScalars = null;

        for (int i = 0; i < n; i++) {

            double[] z1 = extractor.standardize(scalars1[i]);
            double[] z2 = extractor.standardize(scalars2[i]);

            // same arithmetic as SentencePair.getPairFeatures
            double squaredDistance = 0.0;
            for (int k = 0; k < scalarCount; k++) {
                double difference = z1[k] - z2[k];
                featureMatrix[i][k] = Math.abs(difference);
                squaredDistance += difference * difference;
            }
            featureMatrix[i][scalarCount + 1] = Math.sqrt(squaredDistance / scalarCount);

            scalars1[i] = null;
            scalars2[i] = null;
        }

        // The n-gram differences are standardised too (fitted on the TRAINING pairs
        // only)
        if (USE_FULL_FEATURES) {

            extractor.fitNGramDifferenceScaler(featureMatrix);

            for (int i = 0; i < featureMatrix.length; i++) {
                featureMatrix[i] = extractor.standardizeNGramDifferences(featureMatrix[i]);
            }
        }

        // verifying features exists
        for (int i = 0; i < featureMatrix.length; i++) {

            if (featureMatrix[i] == null) {
                throw new IllegalStateException("Feature vector at index " + i + " is null.");
            }
        }

        // The first pairs are done again the slow (original) way, they have to be identical
        int verified = Math.min(VERIFY_SAMPLE, n);
        for (int i = 0; i < verified; i++) {

            double[] expected = getPairFeatures(pairs.get(i), extractor);

            if (!java.util.Arrays.equals(expected, featureMatrix[i])) {
                throw new IllegalStateException("The memory-friendly feature matrix differs from "
                        + "getPairFeatures at pair " + i + ". Was SentencePair changed? Fix buildTrainingMatrix.");
            }
        }
        System.out.println("  Checked " + verified + " pairs against the original way: identical.");

        return featureMatrix;
    }

    /** The rows of all the matrices in one array (the rows are shared, not copied). */
    private static double[][] joinMatrices(java.util.Collection<double[][]> matrices) {

        int total = 0;
        for (double[][] matrix : matrices) {
            total += matrix.length;
        }

        double[][] joined = new double[total][];
        int next = 0;
        for (double[][] matrix : matrices) {
            System.arraycopy(matrix, 0, joined, next, matrix.length);
            next += matrix.length;
        }

        return joined;
    }

    private static int[] joinTruth(java.util.Collection<int[]> truths) {

        int total = 0;
        for (int[] truth : truths) {
            total += truth.length;
        }

        int[] joined = new int[total];
        int next = 0;
        for (int[] truth : truths) {
            System.arraycopy(truth, 0, joined, next, truth.length);
            next += truth.length;
        }

        return joined;
    }

    /**
     * Prints the mean, standard deviation and the largest value of every feature,
     * to see if the scale of the features is reasonable (z-score differences
     * should be mostly between 0 and 3), and if training and validation look alike.
     */
    private static void printFeatureSummary(String name, double[][] matrix) {

        int nFeatures = matrix[0].length;
        // the n-gram bucket differences (full features) are summarised in one row
        int shownFeatures = Math.min(nFeatures, FeatureExtractor.COMPRESSED_PAIR_COUNT);

        System.out.println(name + " feature summary (mean / std / max):");

        for (int feature = 0; feature < shownFeatures; feature++) {
            printFeatureRow(String.format("feature %2d", feature), matrix, feature, feature + 1);
        }

        if (nFeatures > shownFeatures) {
            printFeatureRow(String.format("features %d-%d (n-gram differences, averaged)", shownFeatures,
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
     * Helper method
     * Pairwise representation of the two sentences, compressed or with all the
     * features using all the FEATURES defined.
     * @param pair
     * @param features1
     * @param features2
     * @return
     */
    private static double[] makePairFeatures(SentencePair pair, double[] features1, double[] features2) {

        if (USE_FULL_FEATURES) {
            return pair.getFullPairFeatures(features1, features2);
        }

        return pair.getPairFeatures(features1, features2);
    }
}