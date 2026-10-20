package data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import features.FeatureConfig;
import features.FeatureExtractor;

/**
 * The text of a whole run: the training sentences, the training pairs and the validation / test
 * pairs of every difficulty, loaded ONCE.
 *
 * Why it is separate from the feature matrices: the ablation study builds the matrices many times
 * with different feature settings (casing, n-gram orders, bucket sizes, pair representation, ...)
 * from the same text, and Main builds them once. build(config) is the only place that does it,
 * with the protocol the project always had:
 * - the extractor (TF-IDF table) and the z-score parameters are fitted on the TRAINING data only
 * - validation (the 1st half of the validation split) chooses the threshold / stops the training
 * - test (the 2nd half, never used before) gives the reported numbers; with full validation
 * (USE_FULL_VALIDATION) there is no test and the validation numbers are reported
 */
public class Corpus {

    public static final String[] DIFFICULTIES = { "easy", "medium", "hard" };

    /**
     * The difficulty levels a run uses. PAN scores every level as its OWN dataset, so a run can be limited to one
     * of them with -Ddifficulty=easy|medium|hard (training, validation, test, thresholds: all of that level only).
     * The default, "all", trains on the three levels together and reports every level separately.
     */
    public static String[] active() {
        String wanted = System.getProperty("difficulty", "all").trim().toLowerCase();
        if (wanted.isEmpty() || wanted.equals("all")) {
            return DIFFICULTIES;
        }
        for (String d : DIFFICULTIES) {
            if (d.equals(wanted)) {
                return new String[] { d };
            }
        }
        throw new IllegalArgumentException("-Ddifficulty must be all, easy, medium or hard, not '" + wanted + "'");
    }

    public final List<String> trainingSentences;
    public List<SentencePair> train;
    public final Map<String, List<SentencePair>> validation = new LinkedHashMap<>();
    public final Map<String, List<SentencePair>> test = new LinkedHashMap<>();
    public final boolean fullValidation;
    public final FeatureConfig.Casing loadedCasing;

    /** true when the memory guard cut the data down */
    public double fractionUsed = 1.0;

    private Corpus(List<String> trainingSentences, boolean fullValidation, FeatureConfig.Casing loadedCasing) {
        this.trainingSentences = trainingSentences;
        this.fullValidation = fullValidation;
        this.loadedCasing = loadedCasing;
    }

    /**
     * Loads the text.
     *
     * @param datasetPath     the folder with easy/medium/hard
     * @param useFullValidation true = the whole validation split is validation and there is no test
     * @param subsetSize      at most this many pairs per split (and per difficulty for validation / test)
     * @param casing          the text is lower-cased while loading when this is LOWERCASE (the extractor
     *                        does the same by itself, so loading with PRESERVE and using LOWERCASE
     *                        gives the same features; ablation variants rely on that)
     * @param pairFeatureCount width of the pair vector (for the memory guard)
     * @param heapFraction    the part of the heap the matrices may use
     * @return the corpus, or null when there is no training data
     */
    public static Corpus load(String datasetPath, boolean useFullValidation, int subsetSize,
            FeatureConfig.Casing casing, int pairFeatureCount, double heapFraction) {

        List<String> trainingSentences = new ArrayList<>();
        for (String difficulty : active()) {
            trainingSentences.addAll(DatasetLoader.getAllSentences(datasetPath, "train", difficulty, casing));
        }
        System.out.println("Training corpus size (" + String.join(", ", active()) + "): " + trainingSentences.size());

        Corpus corpus = new Corpus(trainingSentences, useFullValidation, casing);

        List<SentencePair> dataset = new ArrayList<>();
        for (String difficulty : active()) {
            dataset.addAll(DatasetLoader.loadSplit(datasetPath, "train", difficulty, casing));
        }

        System.out.println("Number of training pairs: " + dataset.size());
        if (dataset.isEmpty()) {
            System.err.println("No training data was loaded.");
            return null;
        }

        for (String difficulty : active()) {

            corpus.validation.put(difficulty, useFullValidation
                    ? DatasetLoader.loadSplit(datasetPath, "validation", difficulty, casing)
                    : DatasetLoader.loadSplitHalf(datasetPath, "validation", difficulty, true, casing));

            if (!useFullValidation) {
                // the 2nd half of the validation set, used only once at the end
                corpus.test.put(difficulty,
                        DatasetLoader.loadSplitHalf(datasetPath, "validation", difficulty, false, casing));
            }
        }

        /*
         * How much of it fits in memory: the wanted sizes are subsetSize at most,
         * and if that is more than the heap can hold, everything is shrunk by the
         * same fraction (the pairs are shuffled with a fixed seed first, so the
         * smaller sets are still a fair picture of the whole).
         */
        long wantedTrain = Math.min(dataset.size(), subsetSize);
        long wantedEvaluation = 0;
        for (List<SentencePair> v : corpus.validation.values()) {
            wantedEvaluation += Math.min(v.size(), subsetSize);
        }
        for (List<SentencePair> t : corpus.test.values()) {
            wantedEvaluation += Math.min(t.size(), subsetSize);
        }

        double fraction = PairMatrixBuilder.fractionThatFitsInMemory(wantedTrain, wantedEvaluation, pairFeatureCount,
                heapFraction);
        corpus.fractionUsed = fraction;

        corpus.train = PairMatrixBuilder.takeSubset(dataset, (int) (wantedTrain * fraction));
        for (String difficulty : active()) {

            List<SentencePair> val = corpus.validation.get(difficulty);
            corpus.validation.put(difficulty,
                    PairMatrixBuilder.takeSubset(val, (int) (Math.min(val.size(), subsetSize) * fraction)));

            if (!useFullValidation) {
                List<SentencePair> testPairs = corpus.test.get(difficulty);
                corpus.test.put(difficulty, PairMatrixBuilder.takeSubset(testPairs,
                        (int) (Math.min(testPairs.size(), subsetSize) * fraction)));
            }
        }

        System.out.println("Using " + corpus.train.size() + " training pairs"
                + (fraction < 1.0 ? " (shrunk to fit the heap; run with a bigger -Xmx to use all)" : " (all of them)")
                + ".");

        return corpus;
    }

    /** The matrices of one feature setting. */
    public static class Datasets {

        public FeatureExtractor extractor;

        public double[][] train;
        public int[] trainY;

        public double[][] val; // null = no validation data
        public int[] valY;

        public double[][] test; // null = no test data (full validation)
        public int[] testY;

        // the split the final numbers come from: test when available, else validation
        public double[][] eval;
        public int[] evalY;
        public boolean evalIsTest;
        public Map<String, double[][]> evalByDifficulty = new LinkedHashMap<>();
        public Map<String, int[]> evalYByDifficulty = new LinkedHashMap<>();

        // the validation data of every difficulty on its own: the threshold of a level is chosen on its own part
        public Map<String, double[][]> valByDifficulty = new LinkedHashMap<>();
        public Map<String, int[]> valYByDifficulty = new LinkedHashMap<>();
    }

    /**
     * Fits the extractor on the training text and builds all the matrices.
     *
     * @param config how the features are computed
     * @return the matrices, with val / eval filled in as far as the data goes
     */
    public Datasets build(FeatureConfig config) {

        Datasets d = new Datasets();

        // the extractor first: all it keeps from the corpus is the vocabulary / IDF table
        d.extractor = new FeatureExtractor(trainingSentences, config);

        System.out.println();
        System.out.println("Features: " + d.extractor.describe());
        System.out.println("Extracting features for " + train.size() + " pairs...");

        d.train = PairMatrixBuilder.buildTrainingMatrix(train, d.extractor); // X
        d.trainY = PairMatrixBuilder.getTruthValues(train); // y

        /*
         * Validation & test data, both not used for training (nor for the z-score
         * parameters). Every difficulty's matrix is built ONCE, and the combined one
         * is just those same rows in one array (no second copy of the numbers).
         */
        Map<String, double[][]> validationMatrices = new LinkedHashMap<>();
        Map<String, int[]> validationTruth = new LinkedHashMap<>();
        Map<String, double[][]> testMatrices = new LinkedHashMap<>();
        Map<String, int[]> testTruth = new LinkedHashMap<>();

        for (String difficulty : active()) {

            List<SentencePair> val = validation.get(difficulty);
            if (val != null && !val.isEmpty()) {
                validationMatrices.put(difficulty, PairMatrixBuilder.buildFeatureMatrix(val, d.extractor));
                validationTruth.put(difficulty, PairMatrixBuilder.getTruthValues(val));
            }

            List<SentencePair> testPairs = test.get(difficulty);
            if (testPairs != null && !testPairs.isEmpty()) {
                testMatrices.put(difficulty, PairMatrixBuilder.buildFeatureMatrix(testPairs, d.extractor));
                testTruth.put(difficulty, PairMatrixBuilder.getTruthValues(testPairs));
            }
        }

        d.valByDifficulty = validationMatrices;
        d.valYByDifficulty = validationTruth;

        if (!validationMatrices.isEmpty()) {
            d.val = PairMatrixBuilder.joinMatrices(validationMatrices.values());
            d.valY = PairMatrixBuilder.joinTruth(validationTruth.values());
        }

        if (!testMatrices.isEmpty()) {
            d.test = PairMatrixBuilder.joinMatrices(testMatrices.values());
            d.testY = PairMatrixBuilder.joinTruth(testTruth.values());

            d.eval = d.test;
            d.evalY = d.testY;
            d.evalIsTest = true;
            d.evalByDifficulty = testMatrices;
            d.evalYByDifficulty = testTruth;

        } else if (d.val != null) {

            d.eval = d.val;
            d.evalY = d.valY;
            d.evalIsTest = false;
            d.evalByDifficulty = validationMatrices;
            d.evalYByDifficulty = validationTruth;
        }

        return d;
    }
}
