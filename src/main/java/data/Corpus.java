package data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import features.FeatureConfig;
import features.FeatureExtractor;

/**
 * Manages the raw text payload for a complete experimental run, caching training sentences,
 * training pairs, and validation/test pairs across all specified difficulty levels.
 *
 * <p><strong>Design Rationale:</strong> This dataset container is strictly separated from
 * the numeric feature matrices to optimize ablation studies. Ablation testing requires
 * building many different feature representations (varying casing, n-gram orders, bucket sizes)
 * from the exact same text. Caching the raw text allows {@link #build(FeatureConfig)} to
 * repeatedly generate different feature matrices without redundant disk I/O.</p>
 *
 * <p><strong>Evaluation Protocol:</strong>
 * <ul>
 *   <li><strong>Training:</strong> The feature extractor (TF-IDF tables) and z-score standardizations are fitted exclusively on this subset.</li>
 *   <li><strong>Validation:</strong> The first half of the validation split, used to dynamically select decision thresholds or trigger early stopping.</li>
 *   <li><strong>Test:</strong> The second half of the validation split (strictly held out) used to generate the final reported metrics. If {@code USE_FULL_VALIDATION} is active, this split is disabled and validation numbers are reported instead.</li>
 * </ul>
 * </p>
 */
public class Corpus {

    public static final String[] DIFFICULTIES = { "easy", "medium", "hard" };

    /**
     * Determines which difficulty levels are active for the current experimental run based on system properties.
     *
     * <p>PAN evaluates each difficulty level as an independent dataset. Runs can be restricted to a single
     * level using the JVM argument {@code -Ddifficulty=easy|medium|hard}. The default ({@code all}) trains
     * on the three levels combined but evaluates them separately.</p>
     *
     * @return Array of active difficulty string identifiers.
     * @throws IllegalArgumentException If the provided system property does not match valid difficulty tiers.
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

    /** Indicates the fraction of data retained when memory constraints force downsampling (1.0 = all data used). */
    public double fractionUsed = 1.0;

    private Corpus(List<String> trainingSentences, boolean fullValidation, FeatureConfig.Casing loadedCasing) {
        this.trainingSentences = trainingSentences;
        this.fullValidation = fullValidation;
        this.loadedCasing = loadedCasing;
    }

    /**
     * Loads the raw text dataset into memory and applies necessary memory-guard downsampling.
     *
     * @param datasetPath Base directory path containing easy/medium/hard dataset folders.
     * @param useFullValidation If {@code true}, allocates the entire validation split to validation and disables the test split.
     * @param subsetSize Maximum allowed pairs per split (and per difficulty level for evaluation splits).
     * @param casing Initial text casing applied during load. (Note: Loading with PRESERVE and extracting with LOWERCASE yields the same features as loading with LOWERCASE).
     * @param pairFeatureCount Projected width of the extracted pair feature vector, used for memory footprint estimation.
     * @param heapFraction Maximum fraction of available JVM heap memory the resulting matrices are allowed to occupy.
     * @return Fully populated {@link Corpus} instance, or {@code null} if no training data was found.
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
                corpus.test.put(difficulty,
                        DatasetLoader.loadSplitHalf(datasetPath, "validation", difficulty, false, casing));
            }
        }

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

    /**
     * Container holding the extracted numeric feature matrices and truth labels for a specific feature configuration.
     */
    public static class Datasets {

        public FeatureExtractor extractor;

        public double[][] train;
        public int[] trainY;

        public double[][] val;
        public int[] valY;

        public double[][] test;
        public int[] testY;

        public double[][] eval;
        public int[] evalY;
        public boolean evalIsTest;
        public Map<String, double[][]> evalByDifficulty = new LinkedHashMap<>();
        public Map<String, int[]> evalYByDifficulty = new LinkedHashMap<>();

        public Map<String, double[][]> valByDifficulty = new LinkedHashMap<>();
        public Map<String, int[]> valYByDifficulty = new LinkedHashMap<>();
    }

    /**
     * Initializes a feature extractor on the cached training text and generates numeric matrices for all data splits.
     *
     * @param config Feature configuration parameters specifying which features to extract.
     * @return Populated {@link Datasets} container containing feature matrices (X) and label arrays (y).
     */
    public Datasets build(FeatureConfig config) {

        Datasets d = new Datasets();

        d.extractor = new FeatureExtractor(trainingSentences, config);

        System.out.println();
        System.out.println("Features: " + d.extractor.describe());
        System.out.println("Extracting features for " + train.size() + " pairs...");

        d.train = PairMatrixBuilder.buildTrainingMatrix(train, d.extractor);
        d.trainY = PairMatrixBuilder.getTruthValues(train);

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