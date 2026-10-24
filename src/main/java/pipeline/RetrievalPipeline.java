package pipeline;

import data.TripletSampler;
import evaluation.ThresholdSelector;
import models.EmbeddingNetwork;
import models.RetrievalModel;
import evaluation.AuthorCountEvaluation;
import evaluation.Metrics;
import evaluation.RetrievalEvaluator;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import data.DatasetLoader;
import data.RetrievalData;
import features.FeatureConfig;
import features.FeatureExtractor;

/**
 * The training and the evaluation of the retrieval model in one call, used by the command line
 * (RetrievalMain, with its ablation) and by the web app (RealTrainer). See RetrievalMain for the settings.
 */
public class RetrievalPipeline {

    private RetrievalPipeline() {
        // utility class, not meant to be instantiated
    }

    /** Every setting of one run. */
    public static class Settings {
        public FeatureConfig features = FeatureConfig.fromSystemProperties();
        public int[] hidden = ints(System.getProperty("ret.hidden", "128"));
        public int dim = Integer.getInteger("ret.dim", 32);
        public int epochs = Integer.getInteger("ret.epochs", 30);
        public int triplets = Integer.getInteger("ret.triplets", 20000);
        public int batch = Integer.getInteger("ret.batch", 64);
        public double learningRate = Double.parseDouble(System.getProperty("ret.lr", "0.001"));
        public double margin = Double.parseDouble(System.getProperty("ret.margin", "0.5"));
        public double l2 = Double.parseDouble(System.getProperty("ret.l2", "1e-4"));
        public int candidates = Integer.getInteger("ret.candidates", 4);
        public double crossDocument = Double.parseDouble(System.getProperty("ret.crossdoc", "0"));
        public boolean useAuthorCounts = Boolean.parseBoolean(System.getProperty("ret.authorcounts", "true"));
        public double farPositiveRate = Double.parseDouble(System.getProperty("ret.farpositive", "0.2"));
        public int patience = Integer.getInteger("ret.patience", 5);
        public double profileScale = Double.parseDouble(System.getProperty("ret.scale", "4"));
        public int[] windows = ints(System.getProperty("ret.windows", "1,2,3"));
        // At most this many training sentences (whole documents). Every sentence is kept as ~660 numbers (2.6 KB), so all
        // ~600,000 sentences of the PAN training data need several GB; the training only draws 20,000 triplets per epoch, and
        // more than about 150,000 sentences adds little. Raise it with -Dret.maxsentences=... on a big machine.
        public int maxSentences = Integer.getInteger("ret.maxsentences", 150000);

        public Settings copy() {
            Settings s = new Settings();
            s.features = features.copy();
            s.hidden = hidden.clone();
            s.dim = dim;
            s.epochs = epochs;
            s.triplets = triplets;
            s.batch = batch;
            s.learningRate = learningRate;
            s.margin = margin;
            s.l2 = l2;
            s.candidates = candidates;
            s.crossDocument = crossDocument;
            s.useAuthorCounts = useAuthorCounts;
            s.farPositiveRate = farPositiveRate;
            s.patience = patience;
            s.profileScale = profileScale;
            s.windows = windows.clone();
            s.maxSentences = maxSentences;
            return s;
        }

        /** The triplet drawing rule these settings ask for. */
        public TripletSampler.Policy samplingPolicy() {
            return useAuthorCounts ? TripletSampler.Policy.withAuthorCounts(candidates, crossDocument, farPositiveRate)
                    : TripletSampler.Policy.adjacentSegments(candidates, crossDocument);
        }

        public String describe() {
            return String.format(Locale.ROOT,
                    "hidden=%s dim=%d margin=%.2f lr=%g l2=%g candidates=%d crossdoc=%.2f authorcounts=%b farpositive=%.2f scale=%.1f epochs<=%d triplets/epoch=%d | %s",
                    java.util.Arrays.toString(hidden), dim, margin, learningRate, l2, candidates, crossDocument,
                    useAuthorCounts, farPositiveRate, profileScale, epochs, triplets, features.signature());
        }
    }

    public static int[] ints(String text) {
        if (text == null || text.isBlank()) {
            return new int[0];
        }
        String[] parts = text.split(",");
        int[] result = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Integer.parseInt(parts[i].trim());
        }
        return result;
    }

    /** The documents of a run, loaded once (an ablation trains many models on the same text). */
    public static class Documents {
        public List<DatasetLoader.Problem> train = new ArrayList<>();
        public List<DatasetLoader.Problem> validation = new ArrayList<>();
        public List<DatasetLoader.Problem> test = new ArrayList<>(); // empty with full validation
    }

    public static Documents loadDocuments(String datasetPath, boolean useFullValidation) {

        Documents d = new Documents();
        for (String difficulty : data.Corpus.active()) {
            d.train.addAll(DatasetLoader.loadProblems(datasetPath, "train", difficulty, FeatureConfig.Casing.PRESERVE,
                    DatasetLoader.Part.ALL));
            if (useFullValidation) {
                d.validation.addAll(DatasetLoader.loadProblems(datasetPath, "validation", difficulty,
                        FeatureConfig.Casing.PRESERVE, DatasetLoader.Part.ALL));
            } else {
                d.validation.addAll(DatasetLoader.loadProblems(datasetPath, "validation", difficulty,
                        FeatureConfig.Casing.PRESERVE, DatasetLoader.Part.FIRST_HALF));
                d.test.addAll(DatasetLoader.loadProblems(datasetPath, "validation", difficulty,
                        FeatureConfig.Casing.PRESERVE, DatasetLoader.Part.SECOND_HALF));
            }
        }
        System.out.println("Documents: " + d.train.size() + " train, " + d.validation.size() + " validation, "
                + d.test.size() + " test");
        return d;
    }

    /** What one run measured. */
    public static class Result {
        public String description;
        public int window;
        public double threshold;
        public double validationAuc, testAuc;
        public double[] metrics; // Metrics.evaluate of the test (validation when no test) switch detection
        public double validationMacroF1;
        public RetrievalEvaluator.Report retrieval, rawBaseline, randomBaseline;
        public double rawAuc; // switch detection with the cosine of the raw feature vectors, window chosen the same way
        public double rawMacroF1;
        public AuthorCountEvaluation authorCount;
        public String breakdown = "";
        public boolean onTest;
        public int trainSentences;
        public RetrievalModel model;
        public FeatureExtractor extractor;
    }

    public static Result run(Settings s, Documents documents, boolean verbose) {

        Result r = new Result();
        r.description = s.describe();

        List<String> corpus = new ArrayList<>();
        for (DatasetLoader.Problem p : documents.train) {
            corpus.addAll(p.sentences);
        }

        // the extractor and the z-score parameters: the training sentences only
        FeatureExtractor extractor = new FeatureExtractor(corpus, s.features);
        RetrievalData train = RetrievalData.buildTraining(documents.train, extractor, s.profileScale, s.maxSentences);
        RetrievalData validation = RetrievalData.build(documents.validation, extractor, s.profileScale,
                Integer.MAX_VALUE, false);
        RetrievalData test = documents.test.isEmpty() ? null
                : RetrievalData.build(documents.test, extractor, s.profileScale, Integer.MAX_VALUE, false);
        r.trainSentences = train.size();

        if (verbose) {
            System.out.println();
            System.out.println("Retrieval model: " + s.describe());
            System.out.println("Training sentences: " + train.size() + " in " + train.segments() + " segments of "
                    + train.documents() + " documents, input size " + train.x[0].length);
        }

        EmbeddingNetwork network = new EmbeddingNetwork(train.x[0].length, s.hidden, s.dim);
        PrintStreamSwitch quiet = verbose ? null : PrintStreamSwitch.mute();
        try {
            network.train(train, validation, s.epochs, s.triplets, s.batch, s.learningRate, s.margin, s.l2,
                    s.samplingPolicy(), s.patience, null);
        } finally {
            if (quiet != null) {
                quiet.restore();
            }
        }

        // Switch detection: window and threshold from the validation documents only
        double[][] validationEmbeddings = network.embedAll(validation.x);
        double[][] validationRaw = RetrievalEvaluator.normalise(validation.x);
        int[] validationTruth = RetrievalEvaluator.switchTruth(validation, null);

        int bestWindow = s.windows.length > 0 ? s.windows[0] : 1;
        double bestAuc = -1.0;
        for (int w : s.windows) {
            double auc = RetrievalEvaluator.auc(
                    RetrievalEvaluator.switchScores(validation, validationEmbeddings, w, null), validationTruth);
            if (auc > bestAuc) {
                bestAuc = auc;
                bestWindow = w;
            }
        }
        r.window = bestWindow;
        r.validationAuc = bestAuc;

        double[] validationScores = RetrievalEvaluator.switchScores(validation, validationEmbeddings, bestWindow, null);
        r.threshold = ThresholdSelector.findBestThreshold(validationScores, validationTruth, true);
        r.validationMacroF1 = Metrics.calcMacroF1(ThresholdSelector.apply(validationScores, r.threshold),
                validationTruth);

        // the baseline: the cosine of the raw feature vectors, window and threshold chosen the same way
        int rawWindow = bestWindowFor(validation, validationRaw, validationTruth, s.windows);
        double[] rawValidationScores = RetrievalEvaluator.switchScores(validation, validationRaw, rawWindow, null);
        double rawThreshold = ThresholdSelector.findBestThreshold(rawValidationScores, validationTruth, true);

        // The report split: test when there is one
        RetrievalData eval = test != null ? test : validation;
        r.onTest = test != null;

        double[][] embeddings = network.embedAll(eval.x);
        double[][] raw = RetrievalEvaluator.normalise(eval.x);
        int[] truth = RetrievalEvaluator.switchTruth(eval, null);

        double[] scores = RetrievalEvaluator.switchScores(eval, embeddings, r.window, null);
        r.testAuc = RetrievalEvaluator.auc(scores, truth);
        r.metrics = quietEvaluate(ThresholdSelector.apply(scores, r.threshold), truth, verbose);

        double[] rawScores = RetrievalEvaluator.switchScores(eval, raw, rawWindow, null);
        r.rawAuc = RetrievalEvaluator.auc(rawScores, truth);
        r.rawMacroF1 = Metrics.calcMacroF1(ThresholdSelector.apply(rawScores, rawThreshold), truth);
        r.authorCount = RetrievalEvaluator.authorCount(eval, embeddings, r.window, r.threshold);

        r.retrieval = RetrievalEvaluator.evaluate(eval, embeddings, null);
        r.rawBaseline = RetrievalEvaluator.evaluate(eval, raw, null);
        r.randomBaseline = RetrievalEvaluator.evaluate(eval,
                RetrievalEvaluator.randomVectors(eval.size(), s.dim, 7L), null);

        // the per-difficulty lines print a confusion matrix each, only wanted for a normal run
        StringBuilder breakdown = new StringBuilder();
        for (String difficulty : data.Corpus.active()) {
            if (!verbose) {
                break;
            }
            int[] t = RetrievalEvaluator.switchTruth(eval, difficulty);
            if (t.length == 0) {
                continue;
            }
            int[] p = ThresholdSelector.apply(
                    RetrievalEvaluator.switchScores(eval, embeddings, r.window, difficulty), r.threshold);
            if (breakdown.length() > 0) {
                breakdown.append(";");
            }
            breakdown.append(Metrics.formatBreakdownEntry(difficulty, p, t));
        }
        r.breakdown = breakdown.toString();

        r.model = new RetrievalModel(network, s.profileScale, r.window);
        r.extractor = extractor;
        return r;
    }

    private static double[] quietEvaluate(int[] predictions, int[] truth, boolean verbose) {
        return verbose ? Metrics.evaluate(predictions, truth) : Metrics.evaluateQuietly(predictions, truth);
    }

    private static int bestWindowFor(RetrievalData data, double[][] vectors, int[] truth, int[] windows) {
        int best = windows.length > 0 ? windows[0] : 1;
        double bestAuc = -1.0;
        for (int w : windows) {
            double auc = RetrievalEvaluator.auc(RetrievalEvaluator.switchScores(data, vectors, w, null), truth);
            if (auc > bestAuc) {
                bestAuc = auc;
                best = w;
            }
        }
        return best;
    }

    /**
     * Silences System.out while a model trains inside the ablation (the epochs are not interesting 30 times).
     */
        private record PrintStreamSwitch(PrintStream original) {

        static PrintStreamSwitch mute() {
                PrintStreamSwitch s = new PrintStreamSwitch(System.out);
                System.setOut(new PrintStream(OutputStream.nullOutputStream()));
                return s;
            }

            void restore() {
                System.setOut(original);
            }
        }
}
