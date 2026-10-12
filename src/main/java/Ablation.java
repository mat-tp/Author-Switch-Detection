import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.Function;

import data.DatasetLoader;
import data.SentencePair;
import features.FeatureExtractor;
import features.FeatureGroup;
import models.DecisionTree;
import models.LogisticRegression;
import models.Metrics;
import models.NeuralNetwork;

/**
 * Automated ablation study.
 *
 * (1) FEATURE-GROUP ABLATION: for every model, train on all the features
 * (baseline), then retrain with ONE feature group removed at a time and report
 *
 * Importance(G) = F1_all - F1_without_G
 *
 * (a positive importance = the group carries useful information, around 0 = it
 * adds nothing, negative = removing it helped). Optionally also trains on ONE
 * group alone ("only G"), to show what a group can do by itself.
 *
 * (2) PARAMETER ABLATION: starting from the baseline settings in Main.java, ONE
 * hyperparameter is varied at a time (decision tree depth, min samples, ...;
 * logistic regression learning rate, iterations; neural network layers, L2,
 * dropout, ...), plus a depth x min-leaf grid for the decision tree.
 *
 * Methodology notes:
 * - The features are extracted ONCE. Every experiment just selects columns of
 * the pair matrix (and recomputes the overall distance from the kept columns, so
 * a removed group is really gone from it too).
 * - Thresholds (logistic / neural) are chosen on the validation data; results
 * are reported on the held-out test half (same protocol as Main). Parameters
 * are ranked by VALIDATION F1, never by the test F1, so the "best" setting is
 * not tuned on the test data.
 * - Seeds (-Dseeds=42,123,456): each seed trains on a different random
 * training subsample (-DseedFraction, default 0.8) and results are reported as
 * mean +- std. With one seed (default) all the training pairs are used.
 *
 * Usage (see scripts/ablation.sh):
 * java -cp out Ablation [all|features|params]
 *
 * Options (-D...):
 * models=tree,logistic,neural which models (default all three)
 * only=false skip the "only group G" runs
 * seeds=42,123,456 repeat with several training subsamples
 * seedFraction=0.8 training fraction per seed (only when several seeds)
 * quick=true fewer values per parameter sweep
 * subset=50000 as in Main: at most this many pairs per split
 * ablation.out=data/ablation/ output folder
 * ablation.nnEpochs=N / ablation.lrIterations=N override the baseline
 */
public class Ablation {

    static final String TREE = "tree";
    static final String LOGISTIC = "logistic";
    static final String NEURAL = "neural";

    static final String OUT_DIR = System.getProperty("ablation.out", "data/ablation/");
    static final boolean QUICK = Boolean.getBoolean("quick");
    static final boolean RUN_ONLY_GROUP = !"false".equalsIgnoreCase(System.getProperty("only", "true"));
    static final double SEED_FRACTION = Double.parseDouble(System.getProperty("seedFraction", "0.8"));

    private static final PrintStream NULL_STREAM = new PrintStream(OutputStream.nullOutputStream());

    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------

    /** All the matrices of a run. Feature selection returns a NEW Data with fewer columns. */
    static final class Data {
        double[][] train;
        int[] trainY;
        double[][] val;
        int[] valY;
        // the split the reported numbers come from: test when available, else validation
        double[][] eval;
        int[] evalY;
        boolean evalIsTest;
        Map<String, double[][]> evalByDifficulty = new LinkedHashMap<>();
        Map<String, int[]> evalYByDifficulty = new LinkedHashMap<>();

        Data project(Selection s) {

            if (s.isAll()) {
                return this;
            }

            Data p = new Data();
            p.train = s.apply(train);
            p.trainY = trainY;
            p.val = val == null ? null : s.apply(val);
            p.valY = valY;
            p.eval = s.apply(eval);
            p.evalY = evalY;
            p.evalIsTest = evalIsTest;
            for (Map.Entry<String, double[][]> e : evalByDifficulty.entrySet()) {
                p.evalByDifficulty.put(e.getKey(), s.apply(e.getValue()));
            }
            p.evalYByDifficulty = evalYByDifficulty;
            return p;
        }

        /** Same data, but only a random fraction of the TRAINING rows (rows are shared, not copied). */
        Data withTrainingSample(long seed, double fraction) {

            int n = trainY.length;
            int keep = Math.max(1, (int) Math.round(n * fraction));

            List<Integer> indices = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices, new Random(seed));

            double[][] x = new double[keep][];
            int[] y = new int[keep];
            for (int i = 0; i < keep; i++) {
                x[i] = train[indices.get(i)];
                y[i] = trainY[indices.get(i)];
            }

            Data p = new Data();
            p.train = x;
            p.trainY = y;
            p.val = val;
            p.valY = valY;
            p.eval = eval;
            p.evalY = evalY;
            p.evalIsTest = evalIsTest;
            p.evalByDifficulty = evalByDifficulty;
            p.evalYByDifficulty = evalYByDifficulty;
            return p;
        }
    }

    /**
     * Which columns of the pair vector are kept.
     *
     * Pair vector layout (see SentencePair): [0..S-1] |z-score difference| of each
     * scalar feature, [S] n-gram cosine, [S+1] overall distance, [S+2..] n-gram
     * bucket differences (full features only).
     */
    static final class Selection {

        static final int S = FeatureExtractor.SCALAR_COUNT;

        final String label;
        final boolean[] keepScalar = new boolean[S];
        boolean keepNGram;

        private Selection(String label) {
            this.label = label;
        }

        static Selection all() {
            Selection s = new Selection("ALL");
            Arrays.fill(s.keepScalar, true);
            s.keepNGram = true;
            return s;
        }

        static Selection without(FeatureGroup group) {
            Selection s = all();
            return s.remove("ALL - " + group.name(), group);
        }

        static Selection only(FeatureGroup group) {
            Selection s = new Selection("ONLY " + group.name());
            for (int slot : group.scalarSlots()) {
                s.keepScalar[slot] = true;
            }
            s.keepNGram = group.ownsNGramColumns();
            return s;
        }

        private Selection remove(String newLabel, FeatureGroup group) {
            Selection s = new Selection(newLabel);
            System.arraycopy(keepScalar, 0, s.keepScalar, 0, S);
            s.keepNGram = keepNGram;
            for (int slot : group.scalarSlots()) {
                s.keepScalar[slot] = false;
            }
            if (group.ownsNGramColumns()) {
                s.keepNGram = false;
            }
            return s;
        }

        boolean isAll() {
            if (!keepNGram) {
                return false;
            }
            for (boolean k : keepScalar) {
                if (!k) {
                    return false;
                }
            }
            return true;
        }

        int keptScalars() {
            int n = 0;
            for (boolean k : keepScalar) {
                if (k) {
                    n++;
                }
            }
            return n;
        }

        /** The overall distance is only a separate feature when it is not just one column again. */
        boolean keepDistance() {
            return keptScalars() >= 2;
        }

        double[][] apply(double[][] matrix) {

            int[] kept = new int[keptScalars()];
            int next = 0;
            for (int i = 0; i < S; i++) {
                if (keepScalar[i]) {
                    kept[next++] = i;
                }
            }

            boolean distance = keepDistance();
            double[][] out = new double[matrix.length][];

            for (int r = 0; r < matrix.length; r++) {

                double[] row = matrix[r];
                int buckets = keepNGram ? row.length - (S + 2) : 0;
                double[] projected = new double[kept.length + (keepNGram ? 1 : 0) + (distance ? 1 : 0) + buckets];
                int c = 0;

                double squared = 0.0;
                for (int k : kept) {
                    projected[c++] = row[k];
                    squared += row[k] * row[k];
                }
                if (keepNGram) {
                    projected[c++] = row[S];
                }
                if (distance) {
                    // the same formula as SentencePair.getPairFeatures, over the KEPT features only
                    projected[c++] = Math.sqrt(squared / kept.length);
                }
                for (int b = 0; b < buckets; b++) {
                    projected[c++] = row[S + 2 + b];
                }
                out[r] = projected;
            }

            return out;
        }
    }

    // ---------------------------------------------------------------------
    // Hyperparameters
    // ---------------------------------------------------------------------

    /** All the hyperparameters of the 3 models. Defaults = the baseline in Main.java. */
    static final class Config {

        int treeDepth = Main.TREE_MAX_DEPTH;
        int treeMinSplit = Main.TREE_MIN_SAMPLES_SPLIT;
        int treeMinLeaf = Main.TREE_MIN_SAMPLES_LEAF;
        double treeMinImpurity = Main.TREE_MIN_IMPURITY_DECREASE;
        boolean treeClassWeights = true;

        int lrIterations = Integer.getInteger("ablation.lrIterations", Main.N_ITERATIONS);
        double lrRate = Main.LEARNING_RATE;
        boolean lrClassWeights = Main.USE_CLASS_WEIGHTS;

        int[] nnHidden = Main.NN_HIDDEN_SIZES.clone();
        int nnEpochs = Integer.getInteger("ablation.nnEpochs", Main.NN_MAX_EPOCHS);
        double nnRate = Main.NN_LEARNING_RATE;
        int nnBatch = Main.NN_BATCH_SIZE;
        double nnL2 = Main.NN_L2_LAMBDA;
        double nnDropout = Main.NN_DROPOUT;
        double nnClip = Main.NN_FEATURE_CLIP;
        int nnPatience = Main.NN_PATIENCE;
        boolean nnClassWeights = Main.NN_USE_CLASS_WEIGHTS;

        Config copy() {
            Config c = new Config();
            c.treeDepth = treeDepth;
            c.treeMinSplit = treeMinSplit;
            c.treeMinLeaf = treeMinLeaf;
            c.treeMinImpurity = treeMinImpurity;
            c.treeClassWeights = treeClassWeights;
            c.lrIterations = lrIterations;
            c.lrRate = lrRate;
            c.lrClassWeights = lrClassWeights;
            c.nnHidden = nnHidden.clone();
            c.nnEpochs = nnEpochs;
            c.nnRate = nnRate;
            c.nnBatch = nnBatch;
            c.nnL2 = nnL2;
            c.nnDropout = nnDropout;
            c.nnClip = nnClip;
            c.nnPatience = nnPatience;
            c.nnClassWeights = nnClassWeights;
            return c;
        }

        /** Complete description (used as part of the cache key). */
        String signature(String model) {
            switch (model) {
                case TREE:
                    return "d" + treeDepth + "/s" + treeMinSplit + "/l" + treeMinLeaf + "/i" + treeMinImpurity
                            + "/w" + treeClassWeights;
                case LOGISTIC:
                    return "it" + lrIterations + "/lr" + lrRate + "/w" + lrClassWeights;
                default:
                    return "h" + Arrays.toString(nnHidden) + "/e" + nnEpochs + "/lr" + nnRate + "/b" + nnBatch
                            + "/l2" + nnL2 + "/dr" + nnDropout + "/c" + nnClip + "/p" + nnPatience + "/w"
                            + nnClassWeights;
            }
        }
    }

    /** One hyperparameter and the values it is tried with. */
    static final class Sweep {

        final String model;
        final String name;
        final String[] values;
        final BiConsumer<Config, String> setter;
        final Function<Config, String> getter; // the baseline value, to mark it in the report

        Sweep(String model, String name, String[] values, BiConsumer<Config, String> setter,
                Function<Config, String> getter) {
            this.model = model;
            this.name = name;
            this.setter = setter;
            this.getter = getter;
            this.values = withBaseline(model, QUICK ? thin(values) : values, setter, getter);
        }

        /**
         * The baseline value is ALWAYS part of the sweep (otherwise there is nothing
         * to compare the other values with). It is added when it is missing.
         */
        private static String[] withBaseline(String model, String[] values, BiConsumer<Config, String> setter,
                Function<Config, String> getter) {

            Config baseline = new Config();
            String baselineSignature = baseline.signature(model);

            for (String v : values) {
                Config c = baseline.copy();
                setter.accept(c, v);
                if (c.signature(model).equals(baselineSignature)) {
                    return values;
                }
            }

            List<String> list = new ArrayList<>(Arrays.asList(values));
            list.add(getter.apply(baseline));

            // keep numeric values in ascending order (layer layouts / booleans stay as they are)
            try {
                list.sort(Comparator.comparingDouble(Double::parseDouble));
            } catch (NumberFormatException notNumeric) {
                // leave the order
            }
            return list.toArray(new String[0]);
        }

        /** Quick mode: first, middle and last value only. */
        private static String[] thin(String[] v) {
            if (v.length <= 3) {
                return v;
            }
            return new String[] { v[0], v[v.length / 2], v[v.length - 1] };
        }
    }

    static List<Sweep> sweeps() {

        List<Sweep> s = new ArrayList<>();

        // ---- Decision tree
        s.add(new Sweep(TREE, "max_depth", new String[] { "1", "2", "3", "4", "5", "6", "8", "10", "15", "20" },
                (c, v) -> c.treeDepth = Integer.parseInt(v), c -> String.valueOf(c.treeDepth)));
        s.add(new Sweep(TREE, "min_samples_split", new String[] { "2", "10", "20", "50", "100", "500" },
                (c, v) -> c.treeMinSplit = Integer.parseInt(v), c -> String.valueOf(c.treeMinSplit)));
        s.add(new Sweep(TREE, "min_samples_leaf", new String[] { "1", "2", "5", "10", "25", "100" },
                (c, v) -> c.treeMinLeaf = Integer.parseInt(v), c -> String.valueOf(c.treeMinLeaf)));
        s.add(new Sweep(TREE, "min_impurity_decrease", new String[] { "0.0", "1.0E-5", "1.0E-4", "1.0E-3", "1.0E-2" },
                (c, v) -> c.treeMinImpurity = Double.parseDouble(v), c -> String.valueOf(c.treeMinImpurity)));
        s.add(new Sweep(TREE, "class_weights", new String[] { "true", "false" },
                (c, v) -> c.treeClassWeights = Boolean.parseBoolean(v), c -> String.valueOf(c.treeClassWeights)));

        // ---- Logistic regression
        s.add(new Sweep(LOGISTIC, "learning_rate", new String[] { "0.01", "0.05", "0.1", "0.3", "0.6", "1.0" },
                (c, v) -> c.lrRate = Double.parseDouble(v), c -> String.valueOf(c.lrRate)));
        s.add(new Sweep(LOGISTIC, "iterations", new String[] { "100", "500", "1000", "2500", "5000", "10000" },
                (c, v) -> c.lrIterations = Integer.parseInt(v), c -> String.valueOf(c.lrIterations)));
        s.add(new Sweep(LOGISTIC, "class_weights", new String[] { "true", "false" },
                (c, v) -> c.lrClassWeights = Boolean.parseBoolean(v), c -> String.valueOf(c.lrClassWeights)));

        // ---- Neural network
        s.add(new Sweep(NEURAL, "hidden_layers", new String[] { "16", "32-16", "64-32", "128-64-32", "128-64-32-16-8" },
                (c, v) -> c.nnHidden = parseHidden(v), c -> joinHidden(c.nnHidden)));
        s.add(new Sweep(NEURAL, "l2_lambda", new String[] { "0.0", "1.0E-4", "1.0E-3", "1.0E-2", "1.0E-1" },
                (c, v) -> c.nnL2 = Double.parseDouble(v), c -> String.valueOf(c.nnL2)));
        s.add(new Sweep(NEURAL, "dropout", new String[] { "0.0", "0.1", "0.2", "0.3", "0.5" },
                (c, v) -> c.nnDropout = Double.parseDouble(v), c -> String.valueOf(c.nnDropout)));
        s.add(new Sweep(NEURAL, "learning_rate", new String[] { "1.0E-4", "5.0E-4", "1.0E-3", "5.0E-3", "1.0E-2" },
                (c, v) -> c.nnRate = Double.parseDouble(v), c -> String.valueOf(c.nnRate)));
        s.add(new Sweep(NEURAL, "batch_size", new String[] { "32", "64", "128", "256" },
                (c, v) -> c.nnBatch = Integer.parseInt(v), c -> String.valueOf(c.nnBatch)));
        s.add(new Sweep(NEURAL, "feature_clip", new String[] { "0.0", "3.0", "5.0" },
                (c, v) -> c.nnClip = Double.parseDouble(v), c -> String.valueOf(c.nnClip)));
        s.add(new Sweep(NEURAL, "patience", new String[] { "0", "3", "8" },
                (c, v) -> c.nnPatience = Integer.parseInt(v), c -> String.valueOf(c.nnPatience)));

        return s;
    }

    static int[] parseHidden(String v) {
        String[] parts = v.split("-");
        int[] sizes = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            sizes[i] = Integer.parseInt(parts[i]);
        }
        return sizes;
    }

    static String joinHidden(int[] sizes) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sizes.length; i++) {
            sb.append(i == 0 ? "" : "-").append(sizes[i]);
        }
        return sb.toString();
    }

    // ---------------------------------------------------------------------
    // Training one configuration
    // ---------------------------------------------------------------------

    /** Result of ONE training run. */
    static final class RunResult {
        double threshold = 0.5;
        double[] val; // {accuracy, precision, recall, f1, balancedAccuracy}, null = no validation data
        double[] eval;
        double trainF1;
        Map<String, Double> evalF1ByDifficulty = new LinkedHashMap<>();
        double seconds;
        int nFeatures;
    }

    /** Result of a run over all the seeds (mean and std). */
    static final class Agg {
        int seeds;
        int nFeatures;
        double trainF1;
        double valF1, valF1Std;
        double[] evalMean = new double[5];
        double[] evalStd = new double[5];
        Map<String, Double> f1ByDifficulty = new LinkedHashMap<>();
        double threshold;
        double seconds;
        String error; // set when the run failed (the study carries on)

        double evalF1() {
            return evalMean[3];
        }
    }

    /** Turns a feature matrix into 0/1 predictions. */
    interface Predictor {
        int[] predict(double[][] matrix);
    }

    static RunResult runOnce(String model, Config c, Data d) {

        PrintStream original = System.out;
        System.setOut(NULL_STREAM); // the models print their training progress

        try {
            long start = System.nanoTime();
            RunResult r = new RunResult();
            r.nFeatures = d.train[0].length;

            Predictor predictor;

            if (model.equals(TREE)) {

                DecisionTree tree = new DecisionTree(d.train, d.trainY, c.treeDepth, c.treeMinSplit, c.treeMinLeaf,
                        c.treeMinImpurity);
                tree.setVerbose(false);
                tree.setUseClassWeights(c.treeClassWeights);
                tree.makeBinaryTree();
                predictor = tree::predictAll;

            } else if (model.equals(LOGISTIC)) {

                double[] weights = LogisticRegression.gradiantDescent(d.train, d.trainY, c.lrIterations, c.lrRate,
                        c.lrClassWeights);
                Function<double[][], double[]> probabilities = m -> LogisticRegression.predictProbabilities(m, weights);
                r.threshold = chooseThreshold(probabilities, d);
                final double threshold = r.threshold;
                predictor = m -> LogisticRegression.predict(probabilities.apply(m), threshold);

            } else {

                NeuralNetwork network = new NeuralNetwork(d.train, d.trainY, c.nnHidden, c.nnClip);
                network.train(c.nnEpochs, c.nnRate, c.nnBatch, c.nnL2, c.nnDropout, c.nnClassWeights, d.val, d.valY,
                        c.nnPatience);
                Function<double[][], double[]> probabilities = network::predictProbabilities;
                r.threshold = chooseThreshold(probabilities, d);
                final double threshold = r.threshold;
                predictor = m -> LogisticRegression.predict(probabilities.apply(m), threshold);
            }

            r.trainF1 = Metrics.evaluateQuietly(predictor.predict(d.train), d.trainY)[3];
            r.val = d.val == null ? null : Metrics.evaluateQuietly(predictor.predict(d.val), d.valY);
            r.eval = Metrics.evaluateQuietly(predictor.predict(d.eval), d.evalY);

            for (Map.Entry<String, double[][]> e : d.evalByDifficulty.entrySet()) {
                int[] truth = d.evalYByDifficulty.get(e.getKey());
                r.evalF1ByDifficulty.put(e.getKey(),
                        Metrics.evaluateQuietly(predictor.predict(e.getValue()), truth)[3]);
            }

            r.seconds = (System.nanoTime() - start) / 1e9;
            return r;

        } finally {
            System.setOut(original);
        }
    }

    /** The threshold is chosen on the VALIDATION data only (same as Main). */
    private static double chooseThreshold(Function<double[][], double[]> probabilities, Data d) {
        if (d.val == null) {
            return 0.5;
        }
        return LogisticRegression.findBestThreshold(probabilities.apply(d.val), d.valY);
    }

    // run cache: the same (model, configuration, feature selection) is never trained twice
    private static final Map<String, Agg> CACHE = new HashMap<>();

    static List<Long> seeds() {
        List<Long> seeds = new ArrayList<>();
        for (String part : System.getProperty("seeds", "42").split(",")) {
            if (!part.isBlank()) {
                seeds.add(Long.parseLong(part.trim()));
            }
        }
        return seeds;
    }

    /** Trains one experiment (over all the seeds) and averages. A failed run is recorded, not thrown. */
    static Agg run(String model, Config config, Selection selection, Data fullData) {

        String key = model + "|" + config.signature(model) + "|" + selection.label;
        Agg cached = CACHE.get(key);
        if (cached != null) {
            return cached;
        }

        List<Long> seeds = seeds();
        List<RunResult> results = new ArrayList<>();
        Agg agg = new Agg();

        try {
            Data data = fullData.project(selection);

            for (long seed : seeds) {
                Data runData = seeds.size() > 1 ? data.withTrainingSample(seed, SEED_FRACTION) : data;
                results.add(runOnce(model, config, runData));
            }
            average(results, agg);

        } catch (RuntimeException | OutOfMemoryError e) {
            agg.error = e.getClass().getSimpleName() + ": " + e.getMessage();
            System.err.println("   ! run failed (" + agg.error + "), continuing");
        }

        agg.seeds = seeds.size();
        CACHE.put(key, agg);
        return agg;
    }

    private static void average(List<RunResult> results, Agg agg) {

        int n = results.size();
        agg.nFeatures = results.get(0).nFeatures;

        double[] valF1 = new double[n];
        double[][] eval = new double[5][n];

        for (int i = 0; i < n; i++) {
            RunResult r = results.get(i);
            valF1[i] = r.val == null ? Double.NaN : r.val[3];
            for (int k = 0; k < 5; k++) {
                eval[k][i] = r.eval[k];
            }
            agg.trainF1 += r.trainF1 / n;
            agg.threshold += r.threshold / n;
            agg.seconds += r.seconds;
        }

        agg.valF1 = mean(valF1);
        agg.valF1Std = std(valF1);
        for (int k = 0; k < 5; k++) {
            agg.evalMean[k] = mean(eval[k]);
            agg.evalStd[k] = std(eval[k]);
        }

        for (String difficulty : results.get(0).evalF1ByDifficulty.keySet()) {
            double sum = 0.0;
            for (RunResult r : results) {
                sum += r.evalF1ByDifficulty.get(difficulty);
            }
            agg.f1ByDifficulty.put(difficulty, sum / n);
        }
    }

    static double mean(double[] v) {
        double s = 0.0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    /** Sample standard deviation, 0 for a single run. */
    static double std(double[] v) {
        if (v.length < 2) {
            return 0.0;
        }
        double m = mean(v);
        double s = 0.0;
        for (double x : v) {
            s += (x - m) * (x - m);
        }
        return Math.sqrt(s / (v.length - 1));
    }

    // ---------------------------------------------------------------------
    // Experiments + results
    // ---------------------------------------------------------------------

    /** One line of the results. */
    static final class Row {
        String kind; // feature | param | grid
        String model;
        String experiment; // e.g. "ALL - POS", "max_depth=3"
        String group = ""; // feature group (feature rows)
        String param = ""; // parameter name (param rows)
        String value = ""; // parameter value
        boolean isBaseline;
        Agg agg;
    }

    static final List<Row> ROWS = new ArrayList<>();
    private static BufferedWriter csv;

    private static void record(Row row) throws IOException {
        ROWS.add(row);
        writeCsvRow(row);
    }

    static String f(double v) {
        return Double.isNaN(v) ? "" : String.format(Locale.ROOT, "%.4f", v);
    }

    private static void writeCsvHeader() throws IOException {
        csv.write("kind,model,experiment,group,param,value,baseline,n_features,seeds,train_f1,val_f1,val_f1_std,"
                + "eval_f1,eval_f1_std,accuracy,precision,recall,balanced_accuracy,f1_easy,f1_medium,f1_hard,"
                + "threshold,seconds,error");
        csv.newLine();
    }

    private static void writeCsvRow(Row r) throws IOException {
        Agg a = r.agg;
        csv.write(String.join(",", r.kind, r.model, quote(r.experiment), r.group, r.param, quote(r.value),
                String.valueOf(r.isBaseline), String.valueOf(a.nFeatures), String.valueOf(a.seeds), f(a.trainF1),
                f(a.valF1), f(a.valF1Std), f(a.evalMean[3]), f(a.evalStd[3]), f(a.evalMean[0]), f(a.evalMean[1]),
                f(a.evalMean[2]), f(a.evalMean[4]), f(a.f1ByDifficulty.getOrDefault("easy", Double.NaN)),
                f(a.f1ByDifficulty.getOrDefault("medium", Double.NaN)),
                f(a.f1ByDifficulty.getOrDefault("hard", Double.NaN)), f(a.threshold),
                String.format(Locale.ROOT, "%.1f", a.seconds), quote(a.error == null ? "" : a.error)));
        csv.newLine();
        csv.flush(); // a long run keeps everything finished so far even if it is interrupted
    }

    private static String quote(String s) {
        return "\"" + s.replace("\"", "'") + "\"";
    }

    private static int done = 0;
    private static int total = 0;

    private static void progress(String model, String what, Agg a) {
        done++;
        System.out.println(String.format(Locale.ROOT, "[%d/%d] %-8s %-34s eval F1=%s%s  (%.0fs)", done, total, model,
                what, f(a.evalF1()), a.error != null ? "  FAILED" : "", a.seconds));
    }

    static void featureAblation(String model, Data data) throws IOException {

        Config baseline = new Config();

        Row base = new Row();
        base.kind = "feature";
        base.model = model;
        base.experiment = "ALL";
        base.group = "ALL";
        base.isBaseline = true;
        base.agg = run(model, baseline, Selection.all(), data);
        record(base);
        progress(model, "ALL features", base.agg);

        for (FeatureGroup group : FeatureGroup.values()) {
            Row r = new Row();
            r.kind = "feature";
            r.model = model;
            r.experiment = "ALL - " + group.name();
            r.group = group.name();
            r.agg = run(model, baseline, Selection.without(group), data);
            record(r);
            progress(model, r.experiment, r.agg);
        }

        if (RUN_ONLY_GROUP) {
            for (FeatureGroup group : FeatureGroup.values()) {
                Row r = new Row();
                r.kind = "feature";
                r.model = model;
                r.experiment = "ONLY " + group.name();
                r.group = group.name();
                r.agg = run(model, baseline, Selection.only(group), data);
                record(r);
                progress(model, r.experiment, r.agg);
            }
        }
    }

    static final String[] GRID_DEPTHS = { "2", "3", "5", "8", "12" };
    static final String[] GRID_LEAVES = { "1", "5", "25", "100" };

    static void parameterAblation(String model, Data data) throws IOException {

        Config baseline = new Config();
        Selection all = Selection.all();

        for (Sweep sweep : sweeps()) {

            if (!sweep.model.equals(model)) {
                continue;
            }

            for (String value : sweep.values) {
                Config c = baseline.copy();
                sweep.setter.accept(c, value);

                Row r = new Row();
                r.kind = "param";
                r.model = model;
                r.param = sweep.name;
                r.value = value;
                r.experiment = sweep.name + "=" + value;
                r.isBaseline = c.signature(model).equals(baseline.signature(model));
                r.agg = run(model, c, all, data);
                record(r);
                progress(model, r.experiment, r.agg);
            }
        }

        // the 2D grid: depth x min leaf (depth alone can be misleading, as a bigger
        // leaf also limits the tree)
        if (model.equals(TREE)) {
            for (String depth : QUICK ? new String[] { "3", "8" } : GRID_DEPTHS) {
                for (String leaf : QUICK ? new String[] { "5", "100" } : GRID_LEAVES) {
                    Config c = baseline.copy();
                    c.treeDepth = Integer.parseInt(depth);
                    c.treeMinLeaf = Integer.parseInt(leaf);

                    Row r = new Row();
                    r.kind = "grid";
                    r.model = model;
                    r.param = "max_depth x min_samples_leaf";
                    r.value = depth + "x" + leaf;
                    r.experiment = "depth=" + depth + ", leaf=" + leaf;
                    r.agg = run(model, c, all, data);
                    record(r);
                    progress(model, r.experiment, r.agg);
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Entry point
    // ---------------------------------------------------------------------

    /** Every scalar slot has to belong to exactly one group, or a new feature would silently be ignored. */
    static void checkGroupCoverage() {

        int[] owners = new int[FeatureExtractor.SCALAR_COUNT];

        for (FeatureGroup group : FeatureGroup.values()) {
            for (int slot : group.scalarSlots()) {
                owners[slot]++;
            }
        }

        for (int slot = 0; slot < owners.length; slot++) {
            if (owners[slot] != 1) {
                throw new IllegalStateException("Scalar feature " + slot + " belongs to " + owners[slot]
                        + " feature groups (must be exactly 1). Update features.FeatureGroup.");
            }
        }
    }

    public static void main(String[] args) throws IOException {

        String mode = args.length > 0 ? args[0].toLowerCase() : "all";
        if (!mode.equals("all") && !mode.equals("features") && !mode.equals("params")) {
            System.err.println("Unknown mode: " + mode + " (use all | features | params)");
            return;
        }

        Data data = loadData();
        if (data == null) {
            return;
        }

        run(data, mode);
    }

    /** The whole study on already prepared data (also used by the tests). */
    static void run(Data data, String mode) throws IOException {

        checkGroupCoverage();

        List<String> models = new ArrayList<>();
        for (String m : System.getProperty("models", "tree,logistic,neural").split(",")) {
            models.add(m.trim().toLowerCase());
        }

        boolean doFeatures = mode.equals("all") || mode.equals("features");
        boolean doParams = mode.equals("all") || mode.equals("params");

        // count the runs, for the progress output
        total = 0;
        done = 0;
        for (String model : models) {
            if (doFeatures) {
                total += 1 + FeatureGroup.values().length * (RUN_ONLY_GROUP ? 2 : 1);
            }
            if (doParams) {
                for (Sweep s : sweeps()) {
                    if (s.model.equals(model)) {
                        total += s.values.length;
                    }
                }
                if (model.equals(TREE)) {
                    total += (QUICK ? 4 : GRID_DEPTHS.length * GRID_LEAVES.length);
                }
            }
        }

        Files.createDirectories(Paths.get(OUT_DIR));
        ROWS.clear();
        CACHE.clear();

        System.out.println();
        System.out.println("ABLATION STUDY: " + mode + " | models " + models + " | seeds " + seeds()
                + " | up to " + total + " runs (repeated runs are cached)");
        System.out.println("Train pairs: " + data.trainY.length + " | " + (data.evalIsTest ? "test" : "validation")
                + " pairs: " + data.evalY.length + " | pair features: " + data.train[0].length);
        System.out.println();

        long start = System.nanoTime();

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(OUT_DIR, "ablation_results.csv"),
                StandardCharsets.UTF_8)) {

            csv = writer;
            writeCsvHeader();

            for (String model : models) {
                if (doFeatures) {
                    featureAblation(model, data);
                }
                if (doParams) {
                    parameterAblation(model, data);
                }
            }
        }

        double minutes = (System.nanoTime() - start) / 60e9;
        AblationReport.write(ROWS, data, OUT_DIR, minutes);

        System.out.println();
        System.out.println(String.format(Locale.ROOT, "Done in %.1f min. Files in %s:", minutes, OUT_DIR));
        System.out.println("  ablation_results.csv   every run (one row each)");
        System.out.println("  ablation_report.md     importance ranking + parameter tables + best settings");
        System.out.println("  ablation_tables.tex    LaTeX tables for the manuscript");
    }

    // ---------------------------------------------------------------------
    // Loading the data (same pipeline and protocol as Main.main)
    // ---------------------------------------------------------------------

    static Data loadData() {

        List<String> trainingCorpus = DatasetLoader.getAllSentences(Main.DATASET_PATH, "train");
        System.out.println("Training corpus size (all difficulties): " + trainingCorpus.size());

        FeatureExtractor extractor = new FeatureExtractor(trainingCorpus);
        trainingCorpus = null;

        List<SentencePair> dataset = new ArrayList<>();
        for (String difficulty : Main.DIFFICULTIES) {
            dataset.addAll(DatasetLoader.loadSplit(Main.DATASET_PATH, "train", difficulty));
        }
        if (dataset.isEmpty()) {
            System.err.println("No training data was loaded.");
            return null;
        }

        Map<String, List<SentencePair>> validationByDifficulty = new LinkedHashMap<>();
        Map<String, List<SentencePair>> testByDifficulty = new LinkedHashMap<>();

        for (String difficulty : Main.DIFFICULTIES) {
            validationByDifficulty.put(difficulty, Main.USE_FULL_VALIDATION
                    ? DatasetLoader.loadSplit(Main.DATASET_PATH, "validation", difficulty)
                    : DatasetLoader.loadSplitHalf(Main.DATASET_PATH, "validation", difficulty, true));

            if (!Main.USE_FULL_VALIDATION) {
                testByDifficulty.put(difficulty,
                        DatasetLoader.loadSplitHalf(Main.DATASET_PATH, "validation", difficulty, false));
            }
        }

        long wantedTrain = Math.min(dataset.size(), Main.SUBSET_SIZE);
        long wantedEvaluation = 0;
        for (List<SentencePair> v : validationByDifficulty.values()) {
            wantedEvaluation += Math.min(v.size(), Main.SUBSET_SIZE);
        }
        for (List<SentencePair> t : testByDifficulty.values()) {
            wantedEvaluation += Math.min(t.size(), Main.SUBSET_SIZE);
        }

        int pairFeatureCount = FeatureExtractor.COMPRESSED_PAIR_COUNT;
        if (Main.USE_FULL_FEATURES) {
            pairFeatureCount += extractor.getFeatures("x").length - FeatureExtractor.SCALAR_COUNT;
        }

        double fraction = Main.fractionThatFitsInMemory(wantedTrain, wantedEvaluation, pairFeatureCount);
        if (Main.USE_FULL_FEATURES) {
            // every experiment makes a projected copy of the matrices next to the original
            fraction *= 0.5;
        }

        dataset = Main.takeSubset(dataset, (int) (wantedTrain * fraction));

        Data data = new Data();
        System.out.println("Extracting features for " + dataset.size() + " pairs...");
        data.train = Main.buildTrainingMatrix(dataset, extractor);
        data.trainY = Main.getTruthValues(dataset);
        dataset = null;

        Map<String, double[][]> validationMatrices = new LinkedHashMap<>();
        Map<String, int[]> validationTruth = new LinkedHashMap<>();
        Map<String, double[][]> testMatrices = new LinkedHashMap<>();
        Map<String, int[]> testTruth = new LinkedHashMap<>();

        for (String difficulty : Main.DIFFICULTIES) {

            List<SentencePair> val = Main.takeSubset(validationByDifficulty.get(difficulty),
                    (int) (Math.min(validationByDifficulty.get(difficulty).size(), Main.SUBSET_SIZE) * fraction));
            if (!val.isEmpty()) {
                validationMatrices.put(difficulty, Main.buildFeatureMatrix(val, extractor));
                validationTruth.put(difficulty, Main.getTruthValues(val));
            }

            if (!Main.USE_FULL_VALIDATION) {
                List<SentencePair> test = Main.takeSubset(testByDifficulty.get(difficulty),
                        (int) (Math.min(testByDifficulty.get(difficulty).size(), Main.SUBSET_SIZE) * fraction));
                if (!test.isEmpty()) {
                    testMatrices.put(difficulty, Main.buildFeatureMatrix(test, extractor));
                    testTruth.put(difficulty, Main.getTruthValues(test));
                }
            }
        }

        if (!validationMatrices.isEmpty()) {
            data.val = Main.joinMatrices(validationMatrices.values());
            data.valY = Main.joinTruth(validationTruth.values());
        }

        if (!testMatrices.isEmpty()) {
            data.eval = Main.joinMatrices(testMatrices.values());
            data.evalY = Main.joinTruth(testTruth.values());
            data.evalIsTest = true;
            data.evalByDifficulty = testMatrices;
            data.evalYByDifficulty = testTruth;
        } else if (data.val != null) {
            data.eval = data.val;
            data.evalY = data.valY;
            data.evalIsTest = false;
            data.evalByDifficulty = validationMatrices;
            data.evalYByDifficulty = validationTruth;
        } else {
            System.err.println("No validation or test data was loaded.");
            return null;
        }

        return data;
    }
}
