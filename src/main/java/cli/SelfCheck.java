package cli;

import evaluation.ThresholdSelector;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import data.RetrievalData;
import features.FeatureConfig;
import features.FeatureExtractor;
import features.HeuristicPosTagger;
import features.PosTagset;
import features.TextUtils;
import models.DecisionTree;
import models.DifficultyDetector;
import routing.LevelRouter;
import models.SoftmaxRegression;
import models.EmbeddingNetwork;
import models.LogisticRegression;
import analysis.DocumentSegmentation;
import evaluation.AuthorCountEvaluation;
import evaluation.Metrics;
import persistence.ModelStore;
import models.NeuralNetwork;
import evaluation.RetrievalEvaluator;

/**
 * Tests of the parts that every reported number depends on. No test library: each check prints PASS or
 * FAIL, the exit code is 1 when one failed.
 *
 * java -cp out SelfCheck
 *
 * - the gradients of the logistic regression, the neural network and the embedding network against numerical gradients
 * - macro-F1 and the AUC against hand-computed values
 * - the decision tree: exact and histogram split search give the same tree when the bins can hold every value
 * - the feature layout for every pair representation and n-gram order 1..5 (the widths and the names agree)
 * - casing, the hash, the POS tagger, the profile blocks
 * - save and load (ModelStore) give the same probabilities
 */
public class SelfCheck {

    private static int failed = 0, passed = 0;

    private static void check(String name, boolean ok, String detail) {
        if (ok) {
            passed++;
            System.out.println("PASS  " + name);
        } else {
            failed++;
            System.out.println("FAIL  " + name + "  " + detail);
        }
    }

    private static boolean near(double a, double b, double tolerance) {
        return Math.abs(a - b) <= tolerance;
    }

    public static void main(String[] args) throws Exception {

        metrics();
        authorCount();
        gradients();
        tree();
        features();
        retrievalMeasures();
        modelStore();
        difficulty();

        System.out.println();
        System.out.println(passed + " passed, " + failed + " failed");
        System.exit(failed == 0 ? 0 : 1);
    }

    private static void metrics() {

        // TP=1 FP=1 TN=2 FN=0: F1(change)=2/3, F1(same)=4/5, macro = 11/15
        int[] predictions = { 1, 1, 0, 0 };
        int[] truth = { 1, 0, 0, 0 };
        check("macro-F1 by hand (11/15)", near(Metrics.calcMacroF1(predictions, truth), 11.0 / 15.0, 1e-12),
                "got " + Metrics.calcMacroF1(predictions, truth));
        check("F1 of the change class by hand (2/3)", near(Metrics.calcF1(predictions, truth), 2.0 / 3.0, 1e-12), "");
        check("balanced accuracy by hand (1.0 recall, 2/3 specificity)",
                near(Metrics.calcBalancedAccuracy(predictions, truth), (1.0 + 2.0 / 3.0) / 2.0, 1e-12), "");

        // the PAN baselines: with 80% 'same' pairs, all-0 gives macro-F1 = (F1same + 0) / 2
        int[] truth2 = new int[100];
        for (int i = 0; i < 20; i++) {
            truth2[i] = 1;
        }
        int[] all0 = new int[100], all1 = new int[100];
        Arrays.fill(all1, 1);
        double same = 2.0 * 80 / (2.0 * 80 + 20);
        check("all-0 baseline macro-F1", near(Metrics.calcMacroF1(all0, truth2), same / 2.0, 1e-12), "");
        double change = 2.0 * 20 / (2.0 * 20 + 80);
        check("all-1 baseline macro-F1", near(Metrics.calcMacroF1(all1, truth2), change / 2.0, 1e-12), "");

        // the best threshold for macro-F1 on separable scores must separate them
        double[] probabilities = { 0.9, 0.8, 0.7, 0.2, 0.1, 0.05 };
        int[] t = { 1, 1, 1, 0, 0, 0 };
        double threshold = ThresholdSelector.findBestThreshold(probabilities, t);
        check("findBestThreshold separates separable scores",
                Metrics.calcMacroF1(ThresholdSelector.apply(probabilities, threshold), t) == 1.0,
                "threshold " + threshold);
    }

    private static double[][] randomMatrix(int rows, int columns, long seed) {
        Random random = new Random(seed);
        double[][] m = new double[rows][columns];
        for (double[] row : m) {
            for (int j = 0; j < columns; j++) {
                row[j] = random.nextGaussian();
            }
        }
        return m;
    }

    private static void authorCount() {

        // scores of 5 boundaries, threshold 0.5: changes after sentences 2 and 5 (strictly greater)
        double[] scores = { 0.1, 0.9, 0.5, 0.2, 0.7 };
        DocumentSegmentation segmentation = DocumentSegmentation.fromScores(scores, 0.5);
        check("segmentation: changes where the score is strictly above the threshold",
                java.util.Arrays.equals(segmentation.changeDecisions(), new int[] { 0, 1, 0, 0, 1 }), "");
        check("segmentation: switches + 1 segments", segmentation.estimatedAuthors() == 3, "got " + segmentation.estimatedAuthors());
        check("segmentation: segments cover all six sentences without gaps",
                segmentation.segments().get(0).firstSentence == 0 && segmentation.segments().get(0).endSentence == 2
                        && segmentation.segments().get(1).endSentence == 5
                        && segmentation.segments().get(2).endSentence == 6, "");
        check("segmentation: a single sentence is one segment",
                DocumentSegmentation.fromScores(new double[0], 0.5).estimatedAuthors() == 1, "");

        AuthorCountEvaluation evaluation = new AuthorCountEvaluation();
        evaluation.add(3, 3);
        evaluation.add(5, 4);
        evaluation.add(2, 4);
        evaluation.add(7, 0);
        check("author count: exact rate 1/3 and unknown counts are skipped",
                evaluation.documents() == 3 && near(evaluation.exactMatchRate(), 1.0 / 3.0, 1e-12), "");
        check("author count: mean absolute error 1, mean signed error -1/3",
                near(evaluation.meanAbsoluteError(), 1.0, 1e-12) && near(evaluation.meanSignedError(), -1.0 / 3.0, 1e-12), "");
    }

    private static void gradients() {

        double[][] x = randomMatrix(40, 6, 1);
        int[] y = new int[40];
        Random random = new Random(2);
        for (int i = 0; i < 40; i++) {
            y[i] = random.nextInt(4) == 0 ? 1 : 0;
        }

        double[] weights = { 0.3, -0.2, 0.1, 0.5, -0.4, 0.2 };
        double error = LogisticRegression.maxGradientError(x, y, weights, 0.1, true);
        check("logistic regression gradient = numerical gradient", error < 1e-5, "max relative error " + error);

        NeuralNetwork network = new NeuralNetwork(x, y, new int[] { 5, 4 }, 0.0);
        error = network.maxGradientError(x, y, 60);
        check("neural network backprop = numerical gradient", error < 1e-3, "max relative error " + error);

        EmbeddingNetwork embedding = new EmbeddingNetwork(6, new int[] { 5 }, 4);
        float[] a = new float[6], p = new float[6], n = new float[6];
        for (int i = 0; i < 6; i++) {
            a[i] = (float) x[0][i];
            p[i] = (float) x[1][i];
            n[i] = (float) x[2][i];
        }
        error = embedding.maxGradientError(a, p, n, 40);
        check("embedding network (triplet loss) gradient = numerical gradient", error < 1e-3,
                "max relative error " + error);

        double[] e = embedding.embed(a);
        double length = 0;
        for (double v : e) {
            length += v * v;
        }
        check("embeddings have length 1", near(length, 1.0, 1e-9), "|e|^2 = " + length);
    }

    private static void tree() {

        // integer features with 12 different values: 64 bins hold every value, so the histogram search
        // looks at exactly the same candidate splits as the exact search
        Random random = new Random(5);
        int n = 600;
        double[][] x = new double[n][5];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 5; j++) {
                x[i][j] = random.nextInt(12);
            }
            y[i] = (x[i][0] >= 8 || x[i][2] == 3) ? 1 : 0;
            if (random.nextInt(25) == 0) {
                y[i] = 1 - y[i];
            }
        }

        DecisionTree exact = new DecisionTree(x, y, 4, 20, 5, 0.0);
        exact.setVerbose(false);
        exact.setMaxBins(0);
        exact.makeBinaryTree();
        DecisionTree histogram = new DecisionTree(x, y, 4, 20, 5, 0.0);
        histogram.setVerbose(false);
        histogram.setMaxBins(64);
        histogram.makeBinaryTree();

        check("tree: histogram splits = exact splits when the bins hold every value",
                Arrays.equals(exact.predictAll(x), histogram.predictAll(x)), "predictions differ");

        double[] probabilities = exact.predictProbabilities(x);
        boolean inRange = true;
        for (double q : probabilities) {
            inRange &= q >= 0.0 && q <= 1.0;
        }
        check("tree: leaf probabilities are in [0, 1]", inRange, "");
        check("tree: learns a rule it can represent (training accuracy > 0.9)",
                1.0 - (double) countDifferent(exact.predictAll(x), y) / n > 0.9, "");

        int before = Arrays.hashCode(exact.predictAll(x));
        exact.pruneRedundantSplits();
        check("tree: pruning redundant splits does not change a prediction",
                before == Arrays.hashCode(exact.predictAll(x)), "");
    }

    private static int countDifferent(int[] a, int[] b) {
        int d = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] != b[i] ? 1 : 0;
        }
        return d;
    }

    private static final String[] CORPUS = {
            "The old man did not say a word, but he looked at me for a long time.",
            "I'm not sure what you mean, said Alice; \"it's quite impossible!\"",
            "However, nothing could be done about it - nothing at all...",
            "She walked quickly down the narrow street, and the rain began to fall.",
            "What a strange thing to say?! He laughed, and went away.",
            "We shall see what happens tomorrow, when the others arrive." };

    private static FeatureExtractor fitted(FeatureConfig config) {

        FeatureExtractor extractor = new FeatureExtractor(Arrays.asList(CORPUS), config);
        extractor.prepare();
        List<double[]> raw = new ArrayList<>();
        for (String s : CORPUS) {
            raw.add(extractor.getFeatures(s));
        }
        extractor.fitScaler(raw);

        if (config.fullBlockDifferences) {
            // the bucket differences are standardised with parameters fitted on training pairs
            double[][] pairs = new double[raw.size() - 1][];
            for (int i = 0; i < pairs.length; i++) {
                pairs[i] = new data.SentencePair(CORPUS[i], CORPUS[i + 1], 0).getUnscaledPairFeatures(extractor,
                        raw.get(i), raw.get(i + 1));
            }
            extractor.fitBlockDifferenceScaler(pairs);
        }
        return extractor;
    }

    private static void features() {

        check("63 scalar features, names and count agree",
                FeatureExtractor.SCALAR_COUNT == 63 && FeatureExtractor.SCALAR_NAMES.length == 63,
                "count " + FeatureExtractor.SCALAR_COUNT);

        // the hash is stable and inside the range
        int h1 = TextUtils.hash("abc", 0, 3), h2 = TextUtils.hash("abc", 0, 3), h3 = TextUtils.hash("abd", 0, 3);
        check("hash is deterministic and sees the last character", h1 == h2 && h1 != h3, "");
        check("hash of int values is deterministic",
                TextUtils.hash(new int[] { 1, 2, 3 }, 0, 3) == TextUtils.hash(new int[] { 1, 2, 3 }, 0, 3), "");

        // the tagger: one tag per token, every tag a real one
        List<String> tokens = TextUtils.tokenizeWithPunctuation("The old man walked slowly to the house .");
        int[] tags = new HeuristicPosTagger().tag(tokens);
        boolean valid = tags.length == tokens.size();
        for (int t : tags) {
            valid &= t >= 0 && t < PosTagset.COUNT;
        }
        check("POS tagger: one valid tag per token", valid, Arrays.toString(tags));
        check("POS tagger: The=det, slowly=adv, house=noun, full stop=punct",
                tags[0] == PosTagset.DET && tags[4] == PosTagset.ADV && tags[tags.length - 2] == PosTagset.NOUN
                        && tags[tags.length - 1] == PosTagset.PUNCT, Arrays.toString(tags));

        // casing: LOWERCASE must give the features of the lower-cased text
        FeatureConfig lower = new FeatureConfig();
        lower.casing = FeatureConfig.Casing.LOWERCASE;
        FeatureExtractor lowerExtractor = fitted(lower);
        check("casing=lowercase: 'Hello World' and 'hello world' have the same features",
                Arrays.equals(lowerExtractor.getFeatures("Hello World, I SAID."),
                        lowerExtractor.getFeatures("hello world, i said.")), "");
        FeatureExtractor preserve = fitted(new FeatureConfig());
        check("casing=preserve: 'Hello World' and 'hello world' differ",
                !Arrays.equals(preserve.getFeatures("Hello World, I SAID."),
                        preserve.getFeatures("hello world, i said.")), "");

        // the layout: every pair representation and every n-gram order
        boolean layoutOk = true;
        String detail = "";
        for (FeatureConfigCase c : cases()) {
            FeatureExtractor e = fitted(c.config);
            double[] a = e.getFeatures(CORPUS[0]), b = e.getFeatures(CORPUS[3]);
            double[] pair = e.pairVector(a, b);
            boolean ok = a.length == e.sentenceVectorLength() && pair.length == e.pairVectorLength()
                    && e.pairFeatureNames().length == pair.length;
            for (double v : pair) {
                ok &= !Double.isNaN(v) && !Double.isInfinite(v);
            }
            // the same sentence twice: every |difference| column is 0
            double[] same = e.pairVector(a, a);
            int[] absolute = e.absoluteDifferenceColumns(); // null: MEAN and RELATIVE have no |u - v| part
            if (absolute != null) {
                for (int col : absolute) {
                    ok &= same[col] == 0.0;
                }
            }
            if (!ok) {
                layoutOk = false;
                detail += c.name + " ";
            }
        }
        check("pair vector: widths, names, finite values, |u-u| = 0 (all representations, n = 1..5, full and compressed)",
                layoutOk, detail);

        // a profile is a distribution: it adds up to 1 (or 0 for a sentence without any n-gram)
        FeatureConfig all = new FeatureConfig();
        all.charNGramOrders = new int[] { 1, 2, 3, 4, 5 };
        all.posNGramOrders = new int[] { 1, 2, 3 };
        FeatureExtractor e = fitted(all);
        double[] v = e.getFeatures(CORPUS[1]);
        boolean sums = true;
        for (int block = 0; block < e.blockCount(); block++) {
            double sum = 0;
            for (int i = 0; i < e.blockSize(block); i++) {
                sum += v[e.blockOffset(block) + i];
            }
            if (e.blockIds()[block].equals("fw")) {
                // function words: the share of the tokens of the sentence that is a function word, so it is <= 1
                sums &= sum >= 0.0 && sum <= 1.0 + 1e-9;
            } else {
                sums &= near(sum, 1.0, 1e-9) || near(sum, 0.0, 1e-12);
            }
        }
        check("hashed profile blocks add up to 1 (function-word block: the share of function words, <= 1)", sums, "");

        // the retrieval input has the same length and the square root keeps a profile at length 1
        float[] input = e.embeddingInput(v, 4.0);
        check("embeddingInput has one number per feature", input.length == v.length, "");
    }

    private static class FeatureConfigCase {
        final String name;
        final FeatureConfig config;

        FeatureConfigCase(String name, FeatureConfig config) {
            this.name = name;
            this.config = config;
        }
    }

    private static List<FeatureConfigCase> cases() {

        List<FeatureConfigCase> cases = new ArrayList<>();
        for (String representation : new String[] { "ABSOLUTE", "SIGNED", "MEAN", "ABSOLUTE_AND_MEAN", "RELATIVE",
                "CONCAT", "ABSOLUTE_AND_PRODUCT" }) {
            for (boolean full : new boolean[] { false, true }) {
                FeatureConfig c = new FeatureConfig();
                c.pairRepresentation = representation;
                c.fullBlockDifferences = full;
                c.validate();
                cases.add(new FeatureConfigCase(representation + (full ? "/full" : ""), c));
            }
        }
        for (int n = 1; n <= 5; n++) {
            FeatureConfig c = new FeatureConfig();
            c.charNGramOrders = new int[] { n };
            c.posNGramOrders = new int[] { Math.min(n, 3) };
            c.charTyped = n % 2 == 0;
            c.validate();
            cases.add(new FeatureConfigCase("char n=" + n, c));
        }
        FeatureConfig none = new FeatureConfig();
        none.charNGramOrders = new int[0];
        none.posNGramOrders = new int[0];
        none.functionWordProfile = false;
        none.validate();
        cases.add(new FeatureConfigCase("no profiles", none));
        return cases;
    }

    private static void retrievalMeasures() {

        check("AUC of a perfect ranking is 1",
                RetrievalEvaluator.auc(new double[] { 0.1, 0.2, 0.8, 0.9 }, new int[] { 0, 0, 1, 1 }) == 1.0, "");
        check("AUC of a reversed ranking is 0",
                RetrievalEvaluator.auc(new double[] { 0.9, 0.8, 0.2, 0.1 }, new int[] { 0, 0, 1, 1 }) == 0.0, "");
        check("AUC of equal scores is 0.5",
                RetrievalEvaluator.auc(new double[] { 0.5, 0.5, 0.5, 0.5 }, new int[] { 0, 1, 0, 1 }) == 0.5, "");
        check("AUC by hand (3 of 4 pairs right = 0.75)",
                near(RetrievalEvaluator.auc(new double[] { 0.1, 0.6, 0.4, 0.9 }, new int[] { 0, 0, 1, 1 }), 0.75, 1e-12),
                "");
    }

    private static void modelStore() throws Exception {

        FeatureExtractor extractor = fitted(new FeatureConfig());
        double[] a = extractor.getFeatures(CORPUS[0]), b = extractor.getFeatures(CORPUS[1]);

        int width = extractor.pairVectorLength();
        double[][] x = randomMatrix(50, width, 9);
        int[] y = new int[50];
        for (int i = 0; i < 50; i++) {
            y[i] = x[i][0] > 0 ? 1 : 0;
        }
        double[] weightsBias = LogisticRegression.gradiantDescent(x, y, 200, 0.3, true);

        File file = File.createTempFile("selfcheck", ".ser");
        file.deleteOnExit();
        ModelStore.save(file.getPath(), ModelStore.LOGISTIC, weightsBias, 0.37, extractor, "self check");
        ModelStore.Loaded loaded = ModelStore.load(file.getPath());

        double[][] pair = { extractor.pairVector(a, b) };
        double[] expected = LogisticRegression.predictProbabilities(pair, weightsBias);
        double[] actual = loaded.probabilities(new double[][] { loaded.extractor.pairVector(
                loaded.extractor.getFeatures(CORPUS[0]), loaded.extractor.getFeatures(CORPUS[1])) });

        check("save + load: same probability, same threshold, same description",
                near(expected[0], actual[0], 1e-12) && loaded.threshold == 0.37 && "self check".equals(loaded.description),
                expected[0] + " vs " + actual[0]);

        // a file of the old format must be refused with a message, not read as a model
        File old = File.createTempFile("oldformat", ".ser");
        old.deleteOnExit();
        java.nio.file.Files.write(old.toPath(), new byte[] { (byte) 0xac, (byte) 0xed, 0, 5, 't', 0, 1, 'x' });
        boolean refused = false;
        try {
            ModelStore.load(old.getPath());
        } catch (java.io.IOException expectedException) {
            refused = true;
        }
        check("a file of the old format is refused", refused, "");

        // retrieval data: segments are the runs between the changes
        data.DatasetLoader.Problem problem = new data.DatasetLoader.Problem("1", "easy",
                Arrays.asList(CORPUS), new int[] { 0, 1, 0, 0, 1 });
        RetrievalData rd = RetrievalData.build(List.of(problem), extractor, 4.0, Integer.MAX_VALUE, false);
        check("retrieval data: segments {0,1} {2,3,4} {5}",
                rd.segments() == 3 && rd.segmentLength(0) == 2 && rd.segmentLength(1) == 3 && rd.segmentLength(2) == 1,
                "segments " + rd.segments());
    }

    private static void difficulty() throws Exception {

        // 1. gradient of the softmax regression against a numerical gradient
        Random random = new Random(5);
        int classes = 3, features = 4, n = 30;
        double[][] x = randomMatrix(n, features, 11);
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = random.nextInt(classes);
        }
        double[] classWeight = { 1.0, 2.0, 0.5 };
        SoftmaxRegression model = new SoftmaxRegression(classes, features);
        for (int c = 0; c < classes; c++) {
            for (int j = 0; j <= features; j++) {
                model.setWeight(c, j, random.nextGaussian() * 0.3);
            }
        }
        double[][] gradient = new double[classes][features + 1];
        model.lossAndGradient(x, y, classWeight, 0.01, gradient);
        double worst = 0.0;
        for (int c = 0; c < classes; c++) {
            for (int j = 0; j <= features; j++) {
                double w = model.getWeight(c, j), h = 1e-5;
                model.setWeight(c, j, w + h);
                double up = model.lossAndGradient(x, y, classWeight, 0.01, null);
                model.setWeight(c, j, w - h);
                double down = model.lossAndGradient(x, y, classWeight, 0.01, null);
                model.setWeight(c, j, w);
                double numeric = (up - down) / (2 * h);
                worst = Math.max(worst, Math.abs(numeric - gradient[c][j]) / Math.max(1e-8, Math.abs(numeric) + Math.abs(gradient[c][j])));
            }
        }
        check("softmax regression gradient = numerical gradient", worst < 1e-5, "max relative error " + worst);

        double[] p = model.probabilities(x[0]);
        check("softmax probabilities are positive and add up to 1", p[0] > 0 && p[1] > 0 && p[2] > 0 && near(p[0] + p[1] + p[2], 1.0, 1e-12), "");

        // 2. the detector: probabilities, determinism, save + load
        java.util.Map<String, List<List<String>>> documents = new java.util.LinkedHashMap<>();
        for (int level = 0; level < 3; level++) {
            List<List<String>> list = new ArrayList<>();
            for (int d = 0; d < 12; d++) {
                List<String> sentences = new ArrayList<>();
                for (int k = 0; k < 8 + level * 3; k++) {
                    sentences.add(CORPUS[(d + k * (level + 1)) % CORPUS.length]);
                }
                list.add(sentences);
            }
            documents.put(DifficultyDetector.LEVELS[level], list);
        }
        DifficultyDetector detector = DifficultyDetector.train(documents, 50, 0.05, 0.001);
        List<String> sample = Arrays.asList(CORPUS);
        double[] q1 = detector.probabilities(sample), q2 = detector.probabilities(sample);
        check("detector: 3 probabilities, add up to 1, same answer twice",
                q1.length == 3 && near(q1[0] + q1[1] + q1[2], 1.0, 1e-9) && Arrays.equals(q1, q2), Arrays.toString(q1));

        File detectorFile = new File(File.createTempFile("selfcheck", ".dir").getParentFile(), "selfcheck-router-" + System.nanoTime());
        detectorFile.mkdirs();
        File modelFile = new File(detectorFile, "m.ser");
        File saved = LevelRouter.detectorFile(modelFile);
        detector.save(saved.getPath());
        DifficultyDetector reloaded = DifficultyDetector.load(saved.getPath());
        check("detector: save + load gives the same probabilities", Arrays.equals(q1, reloaded.probabilities(sample)), "");

        // 3. the router never guesses: missing files = the pooled threshold and a reason
        File lonely = new File(detectorFile, "nothing" + System.nanoTime() + "/x.ser");
        LevelRouter missing = LevelRouter.forModelFile(lonely);
        LevelRouter.Decision none = missing.decide(sample, 0.42);
        check("router without files: not ready, pooled threshold, no level, a reason",
                !missing.ready() && missing.problem() != null && none.level == null && none.threshold == 0.42, "");

        // 4. the threshold file round trip and the router with everything there
        java.util.Map<String, Double> thresholds = new java.util.LinkedHashMap<>();
        thresholds.put("easy", 0.55);
        thresholds.put("medium", 0.31);
        thresholds.put("hard", 0.27);
        LevelRouter.writeLevels(modelFile, thresholds, "self check");
        java.util.Map<String, Double> back = LevelRouter.readLevels(LevelRouter.levelsFile(modelFile));
        check("thresholds per level: write + read", Math.abs(back.get("easy") - 0.55) < 1e-9 && Math.abs(back.get("hard") - 0.27) < 1e-9, back.toString());

        LevelRouter router = LevelRouter.forModelFile(modelFile);
        LevelRouter.Decision decision = router.decide(sample, 0.42);
        check("router with detector and thresholds: uses the threshold of the detected level",
                router.ready() && decision.level != null
                        && decision.threshold == thresholds.get(decision.level), String.valueOf(decision.level));

        List<double[]> many = new ArrayList<>();
        many.add(new double[] { 0.1, 0.8, 0.1 });
        many.add(new double[] { 0.2, 0.2, 0.6 });
        many.add(new double[] { 0.1, 0.7, 0.2 });
        check("dataset decision: level with the highest mean probability", "medium".equals(router.decideDataset(many, 0.42).level), "");
    }
}
