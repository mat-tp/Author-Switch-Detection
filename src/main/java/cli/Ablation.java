package cli;

import pipeline.ModelDefaults;
import evaluation.LevelEvaluation;
import util.Statistics;
import evaluation.ThresholdSelector;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
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
import java.util.function.Consumer;
import java.util.function.Function;

import data.Corpus;
import features.FeatureConfig;
import features.FeatureExtractor;
import features.FeatureGroup;
import models.DecisionTree;
import models.LogisticRegression;
import evaluation.Metrics;
import models.NeuralNetwork;

/**
 * Automated ablation study.
 *
 * (1) FEATURE-GROUP ABLATION: for every model, train on all the features
 * (baseline), then retrain with ONE feature group removed at a time and report
 *
 * Importance(G) = MacroF1_all - MacroF1_without_G
 *
 * (a positive importance = the group carries useful information, around 0 = it
 * adds nothing, negative = removing it helped). Optionally also trains on ONE
 * group alone ("only G"), to show what a group can do by itself.
 *
 * (2) PARAMETER ABLATION: starting from the baseline settings in Main.java, ONE
 * hyperparameter is varied at a time (decision tree depth, min samples, ...;
 * logistic regression learning rate, iterations, L2; neural network layers, L2,
 * dropout, optimiser ...), plus a depth x min-leaf grid for the decision tree.
 *
 * (3) PIPELINE ABLATION: the features themselves are computed again for every setting
 * of the feature pipeline (features.FeatureConfig) and the three models are trained on them:
 * - casing: preserve / lower-case / hybrid
 * - character n-gram order n = 1..5, combinations of orders, no character profile
 * - number of hash buckets (and n x buckets grid with -Dpipeline.grid=true)
 * - typed (affix + punctuation) n-grams
 * - POS n-gram orders, number of POS buckets, POS source (heuristic / CoreNLP)
 * - function-word profile on / off
 * - how the two sentences of a pair become one vector (7 pair representations)
 * - the cut of the z-scores (off / 3 / 5 / 10)
 * - compressed (profile measures) vs full (every bucket difference) pair features
 *
 * Methodology notes:
 * - For (1) and (2) the features are extracted ONCE. Every experiment just selects columns of
 * the pair matrix (and recomputes the overall distance from the kept columns, so
 * a removed group is really gone from it too).
 * - Thresholds are chosen on the validation data by macro-F1 (the PAN measure); results
 * are reported on the held-out test half (same protocol as Main). Parameters
 * are ranked by VALIDATION macro-F1, never by the test score, so the "best" setting is
 * not tuned on the test data.
 * - Seeds (-Dseeds=42,123,456): each seed trains on a different random
 * training subsample (-DseedFraction, default 0.8) and results are reported as
 * mean +- std. With one seed (default) all the training pairs are used.
 *
 * Usage (see scripts/ablation.sh):
 * java -cp out Ablation [all|features|params|pipeline]
 *
 * Options (-D...):
 * models=tree,logistic,neural which models (default all three)
 * only=false skip the "only group G" runs
 * seeds=42,123,456 repeat with several training subsamples
 * seedFraction=0.8 training fraction per seed (only when several seeds)
 * quick=true fewer values per parameter sweep / pipeline family
 * pipeline.grid=true also the char n x bucket grid
 * pipeline.models=logistic,tree models for the pipeline study (default: the same as models)
 * subset=50000 as in Main: at most this many pairs per split
 * ablation.out=data/ablation/ output folder
 * ablation.nnEpochs=N / ablation.lrIterations=N override the baseline
 * (the feature settings of the baseline: -Dcasing, -Dchar.n, ... see FeatureConfig)
 */
public class Ablation {

    static final String TREE = "tree";
    static final String LOGISTIC = "logistic";
    static final String NEURAL = "neural";

    static final String OUT_DIR = System.getProperty("ablation.out", "data/ablation/");
    static final boolean QUICK = Boolean.getBoolean("quick");
    static final boolean RUN_ONLY_GROUP = !"false".equalsIgnoreCase(System.getProperty("only", "true"));
    static final boolean PIPELINE_GRID = Boolean.getBoolean("pipeline.grid");
    static final double SEED_FRACTION = Double.parseDouble(System.getProperty("seedFraction", "0.8"));

    // the indices of the arrays Metrics.evaluateQuietly gives
    static final int ACCURACY = 0, PRECISION = 1, RECALL = 2, F1_CHANGE = 3, BALANCED = 4, MACRO_F1 = 5;
    static final int N_METRICS = 6;

    private static final PrintStream NULL_STREAM = new PrintStream(OutputStream.nullOutputStream());

    /** All the matrices of a run. Feature selection returns a NEW Data with fewer columns. */
    static final class Data {
        FeatureExtractor extractor;
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
        // the validation data of every level on its own (to choose the threshold of a level)
        Map<String, double[][]> valByDifficulty = new LinkedHashMap<>();
        Map<String, int[]> valYByDifficulty = new LinkedHashMap<>();

        static Data of(Corpus.Datasets d) {
            Data data = new Data();
            data.extractor = d.extractor;
            data.train = d.train;
            data.trainY = d.trainY;
            data.val = d.val;
            data.valY = d.valY;
            data.eval = d.eval;
            data.evalY = d.evalY;
            data.evalIsTest = d.evalIsTest;
            data.evalByDifficulty = d.evalByDifficulty;
            data.evalYByDifficulty = d.evalYByDifficulty;
            data.valByDifficulty = d.valByDifficulty;
            data.valYByDifficulty = d.valYByDifficulty;
            return data;
        }

        Data project(Selection s) {

            if (s.isAll()) {
                return this;
            }

            Data p = new Data();
            p.extractor = extractor;
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
            for (Map.Entry<String, double[][]> e : valByDifficulty.entrySet()) {
                p.valByDifficulty.put(e.getKey(), s.apply(e.getValue()));
            }
            p.valYByDifficulty = valYByDifficulty;
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
            p.extractor = extractor;
            p.train = x;
            p.trainY = y;
            p.val = val;
            p.valY = valY;
            p.eval = eval;
            p.evalY = evalY;
            p.evalIsTest = evalIsTest;
            p.evalByDifficulty = evalByDifficulty;
            p.evalYByDifficulty = evalYByDifficulty;
            p.valByDifficulty = valByDifficulty;
            p.valYByDifficulty = valYByDifficulty;
            return p;
        }
    }

    /**
     * Which columns of the pair vector are kept.
     *
     * The layout comes from the extractor (FeatureExtractor.pairColumns): the scalar part in the
     * pair representation, 4 measures per profile block, the overall distance and, for the full
     * features, the bucket differences. A group owns its scalar columns and its blocks.
     *
     * The overall distance is the RMS of the z-score differences of ALL scalar features, so it
     * would carry a removed group in through the back door. It is therefore calculated again
     * from the KEPT difference columns (or dropped when the representation has no difference
     * columns, MEAN and RELATIVE, or fewer than two are kept).
     */
    static final class Selection {

        final String label;
        final FeatureExtractor extractor;
        final boolean[] keep; // per pair vector column (the distance column is handled separately)
        final boolean all;

        private Selection(String label, FeatureExtractor extractor, boolean[] keep, boolean all) {
            this.label = label;
            this.extractor = extractor;
            this.keep = keep;
            this.all = all;
        }

        static Selection all(FeatureExtractor extractor) {
            boolean[] keep = new boolean[extractor.pairVectorLength()];
            Arrays.fill(keep, true);
            return new Selection("ALL", extractor, keep, true);
        }

        static Selection without(FeatureGroup group, FeatureExtractor extractor) {

            boolean[] keep = new boolean[extractor.pairVectorLength()];
            Arrays.fill(keep, true);
            for (int column : extractor.pairColumns(group)) {
                keep[column] = false;
            }
            return new Selection("ALL - " + group.name(), extractor, keep, false);
        }

        static Selection only(FeatureGroup group, FeatureExtractor extractor) {

            boolean[] keep = new boolean[extractor.pairVectorLength()];
            for (int column : extractor.pairColumns(group)) {
                keep[column] = true;
            }
            return new Selection("ONLY " + group.name(), extractor, keep, false);
        }

        boolean isAll() {
            return all;
        }

        /** The difference columns that are kept (the overall distance is calculated from them). */
        private int[] keptDifferenceColumns() {

            int[] differences = extractor.absoluteDifferenceColumns();
            if (differences == null) {
                return new int[0];
            }

            int count = 0;
            for (int column : differences) {
                if (keep[column]) {
                    count++;
                }
            }

            int[] kept = new int[count];
            int next = 0;
            for (int column : differences) {
                if (keep[column]) {
                    kept[next++] = column;
                }
            }
            return kept;
        }

        double[][] apply(double[][] matrix) {

            int distanceColumn = extractor.distanceColumn();
            int[] differences = keptDifferenceColumns();
            boolean distance = differences.length >= 2;

            int kept = 0;
            for (int c = 0; c < keep.length; c++) {
                if (keep[c] && c != distanceColumn) {
                    kept++;
                }
            }

            double[][] out = new double[matrix.length][];

            for (int r = 0; r < matrix.length; r++) {

                double[] row = matrix[r];
                double[] projected = new double[kept + (distance ? 1 : 0)];
                int c = 0;

                for (int column = 0; column < keep.length; column++) {
                    if (keep[column] && column != distanceColumn) {
                        projected[c++] = row[column];
                    }
                }

                if (distance) {
                    // the same formula as FeatureExtractor.writeScalarPart, over the KEPT features only
                    double squared = 0.0;
                    for (int column : differences) {
                        squared += row[column] * row[column];
                    }
                    projected[c] = Math.sqrt(squared / differences.length);
                }
                out[r] = projected;
            }

            return out;
        }
    }

    /** All the hyperparameters of the 3 models. Defaults = the baseline in Main.java. */
    static final class Config {

        int treeDepth = ModelDefaults.TREE_MAX_DEPTH;
        int treeMinSplit = ModelDefaults.TREE_MIN_SAMPLES_SPLIT;
        int treeMinLeaf = ModelDefaults.TREE_MIN_SAMPLES_LEAF;
        double treeMinImpurity = ModelDefaults.TREE_MIN_IMPURITY_DECREASE;
        boolean treeClassWeights = true;
        int treeBins = ModelDefaults.TREE_MAX_BINS;

        int lrIterations = Integer.getInteger("ablation.lrIterations", ModelDefaults.LOGISTIC_ITERATIONS);
        double lrRate = ModelDefaults.LOGISTIC_LEARNING_RATE;
        double lrL2 = ModelDefaults.LOGISTIC_L2_PENALTY;
        boolean lrClassWeights = ModelDefaults.LOGISTIC_CLASS_WEIGHTS;

        int[] nnHidden = ModelDefaults.NEURAL_HIDDEN_LAYERS.clone();
        int nnEpochs = Integer.getInteger("ablation.nnEpochs", ModelDefaults.NEURAL_MAX_EPOCHS);
        String nnOptimizer = ModelDefaults.NEURAL_OPTIMIZER;
        double nnRate = ModelDefaults.NEURAL_LEARNING_RATE;
        int nnBatch = ModelDefaults.NEURAL_BATCH_SIZE;
        double nnL2 = ModelDefaults.NEURAL_L2_PENALTY;
        double nnDropout = ModelDefaults.NEURAL_DROPOUT;
        double nnClip = ModelDefaults.NEURAL_FEATURE_CLIP;
        int nnPatience = ModelDefaults.NEURAL_PATIENCE;
        boolean nnClassWeights = ModelDefaults.NEURAL_CLASS_WEIGHTS;

        Config copy() {
            Config c = new Config();
            c.treeDepth = treeDepth;
            c.treeMinSplit = treeMinSplit;
            c.treeMinLeaf = treeMinLeaf;
            c.treeMinImpurity = treeMinImpurity;
            c.treeClassWeights = treeClassWeights;
            c.treeBins = treeBins;
            c.lrIterations = lrIterations;
            c.lrRate = lrRate;
            c.lrL2 = lrL2;
            c.lrClassWeights = lrClassWeights;
            c.nnHidden = nnHidden.clone();
            c.nnEpochs = nnEpochs;
            c.nnOptimizer = nnOptimizer;
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
                            + "/w" + treeClassWeights + "/b" + treeBins;
                case LOGISTIC:
                    return "it" + lrIterations + "/lr" + lrRate + "/l2" + lrL2 + "/w" + lrClassWeights;
                default:
                    return "h" + Arrays.toString(nnHidden) + "/e" + nnEpochs + "/" + nnOptimizer + "/lr" + nnRate
                            + "/b" + nnBatch + "/l2" + nnL2 + "/dr" + nnDropout + "/c" + nnClip + "/p" + nnPatience
                            + "/w" + nnClassWeights;
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

        // Decision tree
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
        // 0 = exact CART, the others are the histogram split search
        s.add(new Sweep(TREE, "split_bins", new String[] { "0", "16", "32", "64", "128", "255" },
                (c, v) -> c.treeBins = Integer.parseInt(v), c -> String.valueOf(c.treeBins)));

        // Logistic regression
        s.add(new Sweep(LOGISTIC, "learning_rate", new String[] { "0.01", "0.05", "0.1", "0.3", "0.6", "1.0" },
                (c, v) -> c.lrRate = Double.parseDouble(v), c -> String.valueOf(c.lrRate)));
        s.add(new Sweep(LOGISTIC, "iterations", new String[] { "100", "500", "1000", "2500", "5000", "10000" },
                (c, v) -> c.lrIterations = Integer.parseInt(v), c -> String.valueOf(c.lrIterations)));
        s.add(new Sweep(LOGISTIC, "l2_lambda", new String[] { "0.0", "1.0E-4", "1.0E-3", "1.0E-2", "1.0E-1" },
                (c, v) -> c.lrL2 = Double.parseDouble(v), c -> String.valueOf(c.lrL2)));
        s.add(new Sweep(LOGISTIC, "class_weights", new String[] { "true", "false" },
                (c, v) -> c.lrClassWeights = Boolean.parseBoolean(v), c -> String.valueOf(c.lrClassWeights)));

        // Neural network
        s.add(new Sweep(NEURAL, "hidden_layers", new String[] { "16", "32-16", "64-32", "128-64-32", "128-64-32-16-8" },
                (c, v) -> c.nnHidden = parseHidden(v), c -> joinHidden(c.nnHidden)));
        s.add(new Sweep(NEURAL, "l2_lambda", new String[] { "0.0", "1.0E-4", "1.0E-3", "1.0E-2", "1.0E-1" },
                (c, v) -> c.nnL2 = Double.parseDouble(v), c -> String.valueOf(c.nnL2)));
        s.add(new Sweep(NEURAL, "dropout", new String[] { "0.0", "0.1", "0.2", "0.3", "0.5" },
                (c, v) -> c.nnDropout = Double.parseDouble(v), c -> String.valueOf(c.nnDropout)));
        s.add(new Sweep(NEURAL, "learning_rate", new String[] { "1.0E-4", "5.0E-4", "1.0E-3", "5.0E-3", "1.0E-2" },
                (c, v) -> c.nnRate = Double.parseDouble(v), c -> String.valueOf(c.nnRate)));
        // the optimiser has its own usual step size, so choosing sgd also sets 0.05 (plain mini-batch gradient descent)
        s.add(new Sweep(NEURAL, "optimizer", new String[] { "adam", "sgd" }, (c, v) -> {
            c.nnOptimizer = v;
            c.nnRate = v.equals("sgd") ? 0.05 : 0.001;
        }, c -> c.nnOptimizer));
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

    /** Result of ONE training run. */
    static final class RunResult {
        double threshold = 0.5;
        double[] val; // {accuracy, precision, recall, f1, balancedAccuracy, macroF1}, null = no validation data
        double[] eval;
        double trainMacro;
        // macro-F1 of each level with the ONE threshold of the pooled validation data (how the study used to report)
        Map<String, Double> evalMacroByDifficulty = new LinkedHashMap<>();
        // macro-F1 of each level with the threshold chosen on THAT level's validation data (PAN scores every level
        // as its own dataset, so this is the number that corresponds to a submission)
        Map<String, Double> evalMacroTunedByDifficulty = new LinkedHashMap<>();
        Map<String, Double> thresholdByDifficulty = new LinkedHashMap<>();
        double seconds;
        int nFeatures;
    }

    /** Result of a run over all the seeds (mean and std). */
    static final class Agg {
        int seeds;
        int nFeatures;
        double trainMacro;
        double valMacro, valMacroStd;
        double[] evalMean = new double[N_METRICS];
        double[] evalStd = new double[N_METRICS];
        // per level, one threshold for all levels (global) and one per level (tuned, the headline per-level number)
        Map<String, Double> macroByDifficultyGlobal = new LinkedHashMap<>();
        Map<String, Double> macroByDifficulty = new LinkedHashMap<>();
        Map<String, Double> thresholdByDifficulty = new LinkedHashMap<>();
        double threshold;
        double seconds;
        String error; // set when the run failed (the study carries on)

        /** The measure the study ranks by: macro-F1 of the evaluation split (PAN 2026's measure). */
        double score() {
            return evalMean[MACRO_F1];
        }

        double scoreStd() {
            return evalStd[MACRO_F1];
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

            Function<double[][], double[]> probabilities;

            if (model.equals(TREE)) {

                DecisionTree tree = new DecisionTree(d.train, d.trainY, c.treeDepth, c.treeMinSplit, c.treeMinLeaf,
                        c.treeMinImpurity);
                tree.setVerbose(false);
                tree.setUseClassWeights(c.treeClassWeights);
                tree.setMaxBins(c.treeBins);
                tree.makeBinaryTree();
                probabilities = tree::predictProbabilities;

            } else if (model.equals(LOGISTIC)) {

                double[] weights = LogisticRegression.gradiantDescent(d.train, d.trainY, c.lrIterations, c.lrRate,
                        c.lrClassWeights, c.lrL2, null);
                probabilities = m -> LogisticRegression.predictProbabilities(m, weights);

            } else {

                NeuralNetwork network = new NeuralNetwork(d.train, d.trainY, c.nnHidden, c.nnClip);
                network.setOptimizer(c.nnOptimizer);
                network.train(c.nnEpochs, c.nnRate, c.nnBatch, c.nnL2, c.nnDropout, c.nnClassWeights, d.val, d.valY,
                        c.nnPatience);
                probabilities = network::predictProbabilities;
            }

            // all three models: the threshold is chosen on the VALIDATION data only (same as Main)
            r.threshold = LevelEvaluation.chooseThreshold(probabilities, d.val, d.valY);
            final double threshold = r.threshold;
            Predictor predictor = m -> ThresholdSelector.apply(probabilities.apply(m), threshold);

            r.trainMacro = Metrics.evaluateQuietly(predictor.predict(d.train), d.trainY)[MACRO_F1];
            r.val = d.val == null ? null : Metrics.evaluateQuietly(predictor.predict(d.val), d.valY);
            r.eval = Metrics.evaluateQuietly(predictor.predict(d.eval), d.evalY);

            for (Map.Entry<String, double[][]> e : d.evalByDifficulty.entrySet()) {
                int[] truth = d.evalYByDifficulty.get(e.getKey());
                r.evalMacroByDifficulty.put(e.getKey(),
                        Metrics.evaluateQuietly(predictor.predict(e.getValue()), truth)[MACRO_F1]);
            }

            // the threshold of each level, chosen on the validation data of that level only
            Map<String, Double> levelThresholds = LevelEvaluation.chooseLevelThresholds(probabilities,
                    d.valByDifficulty, d.valYByDifficulty, d.evalByDifficulty.keySet());
            for (Map.Entry<String, double[][]> e : d.evalByDifficulty.entrySet()) {
                String level = e.getKey();
                double levelThreshold = levelThresholds.getOrDefault(level, r.threshold);
                r.thresholdByDifficulty.put(level, levelThreshold);
                r.evalMacroTunedByDifficulty.put(level, LevelEvaluation.macroF1(probabilities, levelThreshold,
                        e.getValue(), d.evalYByDifficulty.get(level)));
            }

            r.seconds = (System.nanoTime() - start) / 1e9;
            return r;

        } finally {
            System.setOut(original);
        }
    }

    // run cache: the same (model, configuration, feature pipeline, feature selection) is never trained twice
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

        String key = fullData.extractor.getConfig().signature() + "|" + model + "|" + config.signature(model) + "|"
                + selection.label;
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

        double[] valMacro = new double[n];
        double[][] eval = new double[N_METRICS][n];

        for (int i = 0; i < n; i++) {
            RunResult r = results.get(i);
            valMacro[i] = r.val == null ? Double.NaN : r.val[MACRO_F1];
            for (int k = 0; k < N_METRICS; k++) {
                eval[k][i] = r.eval[k];
            }
            agg.trainMacro += r.trainMacro / n;
            agg.threshold += r.threshold / n;
            agg.seconds += r.seconds;
        }

        agg.valMacro = Statistics.mean(valMacro);
        agg.valMacroStd = Statistics.sampleStandardDeviation(valMacro);
        for (int k = 0; k < N_METRICS; k++) {
            agg.evalMean[k] = Statistics.mean(eval[k]);
            agg.evalStd[k] = Statistics.sampleStandardDeviation(eval[k]);
        }

        for (String difficulty : results.get(0).evalMacroByDifficulty.keySet()) {
            double sumGlobal = 0.0, sumTuned = 0.0, sumThreshold = 0.0;
            for (RunResult r : results) {
                sumGlobal += r.evalMacroByDifficulty.get(difficulty);
                sumTuned += r.evalMacroTunedByDifficulty.get(difficulty);
                sumThreshold += r.thresholdByDifficulty.get(difficulty);
            }
            agg.macroByDifficultyGlobal.put(difficulty, sumGlobal / n);
            agg.macroByDifficulty.put(difficulty, sumTuned / n);
            agg.thresholdByDifficulty.put(difficulty, sumThreshold / n);
        }
    }

    /** One line of the results. */
    static final class Row {
        String kind; // feature | param | grid | pipeline
        String model;
        String experiment; // e.g. "ALL - POS_RATIOS", "max_depth=3", "char n=4"
        String group = ""; // feature group (feature rows) / pipeline family (pipeline rows)
        String param = ""; // parameter name (param rows)
        String value = ""; // parameter value
        String pipeline = ""; // FeatureConfig.signature() the row was computed with
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
        csv.write("kind,model,experiment,group,param,value,pipeline,baseline,n_features,seeds,train_macro_f1,"
                + "val_macro_f1,val_macro_f1_std,eval_macro_f1,eval_macro_f1_std,eval_f1_change,accuracy,precision,"
                + "recall,balanced_accuracy,macro_f1_easy,macro_f1_medium,macro_f1_hard,threshold,seconds,error,"
                + "macro_f1_easy_global_thr,macro_f1_medium_global_thr,macro_f1_hard_global_thr,"
                + "thr_easy,thr_medium,thr_hard");
        csv.newLine();
    }

    private static void writeCsvRow(Row r) throws IOException {
        Agg a = r.agg;
        csv.write(String.join(",", r.kind, r.model, quote(r.experiment), r.group, r.param, quote(r.value),
                quote(r.pipeline), String.valueOf(r.isBaseline), String.valueOf(a.nFeatures),
                String.valueOf(a.seeds), f(a.trainMacro), f(a.valMacro), f(a.valMacroStd), f(a.score()),
                f(a.scoreStd()), f(a.evalMean[F1_CHANGE]), f(a.evalMean[ACCURACY]), f(a.evalMean[PRECISION]),
                f(a.evalMean[RECALL]), f(a.evalMean[BALANCED]),
                f(a.macroByDifficulty.getOrDefault("easy", Double.NaN)),
                f(a.macroByDifficulty.getOrDefault("medium", Double.NaN)),
                f(a.macroByDifficulty.getOrDefault("hard", Double.NaN)), f(a.threshold),
                String.format(Locale.ROOT, "%.1f", a.seconds), quote(a.error == null ? "" : a.error),
                f(a.macroByDifficultyGlobal.getOrDefault("easy", Double.NaN)),
                f(a.macroByDifficultyGlobal.getOrDefault("medium", Double.NaN)),
                f(a.macroByDifficultyGlobal.getOrDefault("hard", Double.NaN)),
                f(a.thresholdByDifficulty.getOrDefault("easy", Double.NaN)),
                f(a.thresholdByDifficulty.getOrDefault("medium", Double.NaN)),
                f(a.thresholdByDifficulty.getOrDefault("hard", Double.NaN))));
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
        System.out.println(String.format(Locale.ROOT, "[%d/%d] %-8s %-40s eval macro-F1=%s%s  (%.0fs)", done, total,
                model, what, f(a.score()), a.error != null ? "  FAILED" : "", a.seconds));
    }

    static void featureAblation(String model, Data data) throws IOException {

        Config baseline = new Config();
        FeatureExtractor extractor = data.extractor;
        String pipeline = extractor.getConfig().signature();

        Row base = new Row();
        base.kind = "feature";
        base.model = model;
        base.experiment = "ALL";
        base.group = "ALL";
        base.pipeline = pipeline;
        base.isBaseline = true;
        base.agg = run(model, baseline, Selection.all(extractor), data);
        record(base);
        progress(model, "ALL features", base.agg);

        for (FeatureGroup group : groupsPresent(extractor)) {
            Row r = new Row();
            r.kind = "feature";
            r.model = model;
            r.experiment = "ALL - " + group.name();
            r.group = group.name();
            r.pipeline = pipeline;
            r.agg = run(model, baseline, Selection.without(group, extractor), data);
            record(r);
            progress(model, r.experiment, r.agg);
        }

        if (RUN_ONLY_GROUP) {
            for (FeatureGroup group : groupsPresent(extractor)) {
                Row r = new Row();
                r.kind = "feature";
                r.model = model;
                r.experiment = "ONLY " + group.name();
                r.group = group.name();
                r.pipeline = pipeline;
                r.agg = run(model, baseline, Selection.only(group, extractor), data);
                record(r);
                progress(model, r.experiment, r.agg);
            }
        }
    }

    /** The groups that have something in this pipeline (a profile group is empty when its profile is switched off). */
    static List<FeatureGroup> groupsPresent(FeatureExtractor extractor) {

        List<FeatureGroup> groups = new ArrayList<>();
        for (FeatureGroup group : FeatureGroup.values()) {
            if (extractor.pairColumns(group).length > 0) {
                groups.add(group);
            }
        }
        return groups;
    }

    static final String[] GRID_DEPTHS = { "2", "3", "5", "8", "12" };
    static final String[] GRID_LEAVES = { "1", "5", "25", "100" };

    static void parameterAblation(String model, Data data) throws IOException {

        Config baseline = new Config();
        Selection all = Selection.all(data.extractor);
        String pipeline = data.extractor.getConfig().signature();

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
                r.pipeline = pipeline;
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
                    r.pipeline = pipeline;
                    r.experiment = "depth=" + depth + ", leaf=" + leaf;
                    r.agg = run(model, c, all, data);
                    record(r);
                    progress(model, r.experiment, r.agg);
                }
            }
        }
    }

    /** One setting of the feature pipeline. */
    static final class Variant {

        final String family; // casing | char_n | char_buckets | char_typed | pos_n | pos_buckets | pos_tagger | function_words | pair_representation | block_differences | char_grid
        final String label; // e.g. "n=4"
        final Consumer<FeatureConfig> change;

        Variant(String family, String label, Consumer<FeatureConfig> change) {
            this.family = family;
            this.label = label;
            this.change = change;
        }

        FeatureConfig apply(FeatureConfig base) {
            FeatureConfig c = base.copy();
            change.accept(c);
            c.validate();
            return c;
        }
    }

    private static int[] orders(int... n) {
        return n;
    }

    /**
     * All the pipeline variants. Each one changes the baseline config (from the -D options) in ONE
     * place, the baseline itself is added to every family so each table has its reference row.
     */
    static List<Variant> variants(FeatureConfig base) {

        List<Variant> v = new ArrayList<>();

        // Casing (the text is used as written / lower-cased / hybrid)
        for (FeatureConfig.Casing casing : FeatureConfig.Casing.values()) {
            v.add(new Variant("casing", casing.name().toLowerCase(), c -> c.casing = casing));
        }

        // Character n-gram order: single orders 1..5, combinations, none
        for (int n = 1; n <= 5; n++) {
            final int order = n;
            v.add(new Variant("char_n", "n=" + n, c -> c.charNGramOrders = orders(order)));
        }
        v.add(new Variant("char_n", "n=1,2", c -> c.charNGramOrders = orders(1, 2)));
        v.add(new Variant("char_n", "n=2,3", c -> c.charNGramOrders = orders(2, 3)));
        v.add(new Variant("char_n", "n=3,4", c -> c.charNGramOrders = orders(3, 4)));
        v.add(new Variant("char_n", "n=3,4,5", c -> c.charNGramOrders = orders(3, 4, 5)));
        v.add(new Variant("char_n", "n=1..5", c -> c.charNGramOrders = orders(1, 2, 3, 4, 5)));
        v.add(new Variant("char_n", "none", c -> c.charNGramOrders = orders()));

        // Number of buckets of the character profile (n as in the baseline)
        for (int buckets : new int[] { 16, 32, 64, 128, 256, 512, 1024, 2048 }) {
            v.add(new Variant("char_buckets", "buckets=" + buckets, c -> c.charBuckets = buckets));
        }

        // Typed n-grams (affix + punctuation only, Sapkota et al. 2015)
        v.add(new Variant("char_typed", "all n-grams", c -> c.charTyped = false));
        v.add(new Variant("char_typed", "typed (affix+punct)", c -> c.charTyped = true));

        // POS n-grams
        v.add(new Variant("pos_n", "none", c -> c.posNGramOrders = orders()));
        for (int n = 1; n <= 3; n++) {
            final int order = n;
            v.add(new Variant("pos_n", "n=" + n, c -> c.posNGramOrders = orders(order)));
        }
        v.add(new Variant("pos_n", "n=1,2", c -> c.posNGramOrders = orders(1, 2)));
        v.add(new Variant("pos_n", "n=1,2,3", c -> c.posNGramOrders = orders(1, 2, 3)));
        for (int buckets : new int[] { 32, 64, 128, 256, 512 }) {
            v.add(new Variant("pos_buckets", "buckets=" + buckets, c -> c.posBuckets = buckets));
        }
        v.add(new Variant("pos_tagger", "heuristic", c -> c.posTagger = "heuristic"));
        if (corenlpAvailable()) {
            v.add(new Variant("pos_tagger", "corenlp", c -> c.posTagger = "corenlp"));
        }

        // Function word profile
        v.add(new Variant("function_words", "off", c -> c.functionWordProfile = false));
        v.add(new Variant("function_words", "on", c -> c.functionWordProfile = true));

        // How two sentences become one vector
        for (data.SentencePair.Representation representation : data.SentencePair.Representation.values()) {
            v.add(new Variant("pair_representation", representation.name().toLowerCase(),
                    c -> c.pairRepresentation = representation.name()));
        }

        // The cut of the z-scores of the scalar features
        for (double clip : new double[] { 0.0, 3.0, 5.0, 10.0 }) {
            v.add(new Variant("z_clip", clip == 0.0 ? "off" : "+-" + (int) clip, c -> c.zClip = clip));
        }

        // 4 numbers per profile vs every bucket difference
        v.add(new Variant("block_differences", "compressed (4 measures)", c -> c.fullBlockDifferences = false));
        v.add(new Variant("block_differences", "full (every bucket)", c -> c.fullBlockDifferences = true));

        // N x buckets grid of the character profile
        if (PIPELINE_GRID) {
            for (int n = 1; n <= 5; n++) {
                for (int buckets : new int[] { 64, 128, 256, 512, 1024 }) {
                    final int order = n;
                    final int size = buckets;
                    v.add(new Variant("char_grid", "n=" + n + " x " + buckets, c -> {
                        c.charNGramOrders = orders(order);
                        c.charBuckets = size;
                    }));
                }
            }
        }

        if (QUICK) {
            // fewer values per family: the first, the middle, the last and the baseline are kept
            List<Variant> thinned = new ArrayList<>();
            Map<String, List<Variant>> byFamily = new LinkedHashMap<>();
            for (Variant variant : v) {
                byFamily.computeIfAbsent(variant.family, k -> new ArrayList<>()).add(variant);
            }
            for (List<Variant> family : byFamily.values()) {
                for (int i = 0; i < family.size(); i++) {
                    boolean keep = family.size() <= 4 || i == 0 || i == family.size() - 1 || i == family.size() / 2
                            || family.get(i).apply(base).signature().equals(base.signature());
                    if (keep) {
                        thinned.add(family.get(i));
                    }
                }
            }
            return thinned;
        }

        return v;
    }

    private static boolean corenlpAvailable() {
        try {
            Class.forName("edu.stanford.nlp.pipeline.StanfordCoreNLP");
            return true;
        } catch (ClassNotFoundException | LinkageError e) {
            return false;
        }
    }

    static void pipelineAblation(List<String> models, Corpus corpus, FeatureConfig base, Data baseData)
            throws IOException {

        Config hyper = new Config();
        List<Variant> variants = variants(base);

        for (Variant variant : variants) {

            FeatureConfig config = variant.apply(base);
            boolean isBaseline = config.signature().equals(base.signature());

            Data data = null;

            // the matrices of this variant (the baseline's are already there)
            if (isBaseline) {
                data = baseData;
            } else if (!fitsInMemory(corpus, config)) {
                System.out.println("   ! skipped " + variant.family + " " + variant.label
                        + ": the feature matrices would not fit in the heap (give more -Xmx)");
                done += models.size();
                continue;
            } else {
                PrintStream original = System.out;
                System.setOut(NULL_STREAM);
                try {
                    data = Data.of(corpus.build(config));
                } catch (RuntimeException | OutOfMemoryError e) {
                    System.setOut(original);
                    System.err.println("   ! feature extraction failed for " + variant.family + " " + variant.label
                            + " (" + e + "), continuing");
                    done += models.size();
                    continue;
                } finally {
                    System.setOut(original);
                }
            }

            for (String model : models) {
                Row r = new Row();
                r.kind = "pipeline";
                r.model = model;
                r.group = variant.family;
                r.param = variant.family;
                r.value = variant.label;
                r.experiment = variant.family + ": " + variant.label;
                r.pipeline = config.signature();
                r.isBaseline = isBaseline;
                r.agg = run(model, hyper, Selection.all(data.extractor), data);
                r.agg.nFeatures = Math.max(r.agg.nFeatures, data.train[0].length);
                record(r);
                progress(model, r.experiment, r.agg);
            }

            data = null; // the next variant gets the memory
            System.gc();
        }
    }

    /** Would the matrices of this config fit? (The baseline fitted, the full features may not.) */
    private static boolean fitsInMemory(Corpus corpus, FeatureConfig config) {

        int width = new FeatureExtractor(null, config).pairVectorLength();
        long rows = corpus.train.size();
        for (List<?> list : corpus.validation.values()) {
            rows += list.size();
        }
        for (List<?> list : corpus.test.values()) {
            rows += list.size();
        }

        long need = rows * (8L * width + 16 + 2 * (8L * FeatureExtractor.SCALAR_COUNT + 16));
        long budget = (long) (Runtime.getRuntime().maxMemory() * 0.45);

        return need <= budget;
    }

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
                throw new IllegalStateException("Scalar feature " + FeatureExtractor.SCALAR_NAMES[slot] + " belongs to "
                        + owners[slot] + " feature groups (must be exactly 1). Update features.FeatureGroup.");
            }
        }
    }

    public static void main(String[] args) throws IOException {

        String mode = args.length > 0 ? args[0].toLowerCase() : "all";
        if (!mode.equals("all") && !mode.equals("features") && !mode.equals("params") && !mode.equals("pipeline")) {
            System.err.println("Unknown mode: " + mode + " (use all | features | params | pipeline)");
            return;
        }

        checkGroupCoverage();

        FeatureConfig base = FeatureConfig.fromSystemProperties();
        System.out.println("Baseline feature settings: " + base.signature());

        int pairFeatureCount = new FeatureExtractor(null, base).pairVectorLength();

        // the text is loaded as written: the casing variants lower-case inside the extractor
        double heapFraction = mode.equals("pipeline") || mode.equals("all") ? Main.HEAP_FRACTION_FOR_MATRICES * 0.8
                : Main.HEAP_FRACTION_FOR_MATRICES * 0.5;
        Corpus corpus = Corpus.load(Main.DATASET_PATH, Main.USE_FULL_VALIDATION, Main.SUBSET_SIZE,
                FeatureConfig.Casing.PRESERVE, pairFeatureCount, heapFraction);
        if (corpus == null) {
            return;
        }

        Corpus.Datasets built = corpus.build(base);
        if (built.eval == null) {
            System.err.println("No validation or test data was loaded.");
            return;
        }

        run(corpus, base, Data.of(built), mode);
    }

    /** The whole study on already prepared data. */
    static void run(Corpus corpus, FeatureConfig base, Data data, String mode) throws IOException {

        List<String> models = new ArrayList<>();
        for (String m : System.getProperty("models", "tree,logistic,neural").split(",")) {
            models.add(m.trim().toLowerCase());
        }

        List<String> pipelineModels = new ArrayList<>();
        for (String m : System.getProperty("pipeline.models", String.join(",", models)).split(",")) {
            pipelineModels.add(m.trim().toLowerCase());
        }

        boolean doFeatures = mode.equals("all") || mode.equals("features");
        boolean doParams = mode.equals("all") || mode.equals("params");
        boolean doPipeline = mode.equals("all") || mode.equals("pipeline");

        // count the runs, for the progress output
        total = 0;
        done = 0;
        for (String model : models) {
            if (doFeatures) {
                total += 1 + groupsPresent(data.extractor).size() * (RUN_ONLY_GROUP ? 2 : 1);
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
        if (doPipeline) {
            total += variants(base).size() * pipelineModels.size();
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

            if (doPipeline) {
                pipelineAblation(pipelineModels, corpus, base, data);
            }
        }

        double minutes = (System.nanoTime() - start) / 60e9;
        AblationReport.write(ROWS, data, base, OUT_DIR, minutes);

        System.out.println();
        System.out.println(String.format(Locale.ROOT, "Done in %.1f min. Files in %s:", minutes, OUT_DIR));
        System.out.println("  ablation_results.csv   every run (one row each)");
        System.out.println("  ablation_report.md     importance ranking + parameter tables + pipeline tables + best settings");
        System.out.println("  ablation_tables.tex    LaTeX tables for the manuscript");
    }
}
