package cli;

import evaluation.ThresholdSelector;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import data.DatasetLoader;
import data.DatasetLoader.Problem;
import features.FeatureConfig;
import models.DifficultyDetector;
import routing.LevelRouter;
import evaluation.Metrics;
import persistence.ModelStore;

/**
 * The difficulty level (easy / medium / hard) is NOT given when the software is used: it is detected from the text of
 * every document, and the threshold of the detected level is used. This tool makes the three things that needs, using the
 * trained models as they are (nothing is retrained):
 *
 * java -Ddataset=dataset/mawsa26-pan-zenodo-DATA/ -cp out DifficultyMain train       the detector -> data/models/difficulty.ser
 * java -Ddataset=...                              -cp out DifficultyMain calibrate    the threshold of every level of every
 *                                                                                      saved model -> data/models/NAME.ser.levels
 * java -Ddataset=...                              -cp out DifficultyMain evaluate     what the detector costs: pooled threshold
 *                                                                                      against the true level against the detected level
 * java -Ddataset=...                              -cp out DifficultyMain all          the three, one after the other (default)
 *
 * An optional second argument limits calibrate / evaluate to one model: neural | logistic | tree | retrieval.
 *
 * Protocol (the same as everywhere in the project): the detector is trained on the TRAIN split; the thresholds are
 * chosen on the first half of the validation split of each level; evaluate reports on the second half, which nothing
 * has used.
 *
 * Options (-D): dataset | models.dir | diff.maxdocs 20000 (training documents per level) | diff.iterations 1500 |
 * diff.rate 0.05 | diff.l2 0.001
 */
public class DifficultyMain {

    static final String DATASET = System.getProperty("dataset", "dataset/mawsa26-pan-zenodo-DATA/");
    static final String MODELS_DIR = System.getProperty("models.dir", "data/models/");
    static final String[] LEVELS = DifficultyDetector.LEVELS;
    static final String[][] KNOWN_MODELS = { { "neural", "neuralNetwork.ser" }, { "logistic", "logisticRegression.ser" },
            { "tree", "decisionTree.ser" }, { "retrieval", "retrieval.ser" } };

    public static void main(String[] args) throws IOException, ClassNotFoundException {

        String mode = args.length > 0 ? args[0] : "all";
        String model = args.length > 1 ? args[1] : "all";

        switch (mode) {
            case "train":
                train();
                break;
            case "calibrate":
                calibrate(model);
                break;
            case "evaluate":
                evaluate(model);
                break;
            case "all":
                train();
                calibrate(model);
                evaluate(model);
                break;
            default:
                System.err.println("usage: DifficultyMain [train|calibrate|evaluate|all] [neural|logistic|tree|retrieval]");
                System.exit(1);
        }
    }

    private static List<Problem> load(String split, String level, DatasetLoader.Part part) {
        List<Problem> problems = DatasetLoader.loadProblems(DATASET, split, level, FeatureConfig.Casing.PRESERVE, part);
        if (problems.isEmpty()) {
            throw new IllegalStateException("no " + split + " documents of the level '" + level + "' under " + DATASET
                    + " (the detector needs all three levels)");
        }
        return problems;
    }

    private static List<List<String>> sentences(List<Problem> problems) {
        List<List<String>> result = new ArrayList<>();
        for (Problem p : problems) {
            result.add(p.sentences);
        }
        return result;
    }

    static void train() throws IOException {

        int maxDocs = Integer.getInteger("diff.maxdocs", 20000);
        int iterations = Integer.getInteger("diff.iterations", 1500);
        double rate = Double.parseDouble(System.getProperty("diff.rate", "0.05"));
        double l2 = Double.parseDouble(System.getProperty("diff.l2", "0.001"));

        Map<String, List<List<String>>> training = new LinkedHashMap<>();
        for (String level : LEVELS) {
            List<Problem> problems = load("train", level, DatasetLoader.Part.ALL);
            if (problems.size() > maxDocs) {
                List<Problem> shuffled = new ArrayList<>(problems);
                Collections.shuffle(shuffled, new Random(42L));
                problems = shuffled.subList(0, maxDocs);
            }
            training.put(level, sentences(problems));
            System.out.println("Training documents " + level + ": " + problems.size());
        }

        DifficultyDetector detector = DifficultyDetector.train(training, iterations, rate, l2);
        System.out.println("Detector: " + detector.description);

        System.out.println();
        System.out.println("How well the level is recognised from the text of a document (rows = true level, columns = detected):");
        report("train (sample)", detector, sample(training, 3000));
        Map<String, List<List<String>>> first = new LinkedHashMap<>(), second = new LinkedHashMap<>();
        for (String level : LEVELS) {
            first.put(level, sentences(load("validation", level, DatasetLoader.Part.FIRST_HALF)));
            second.put(level, sentences(load("validation", level, DatasetLoader.Part.SECOND_HALF)));
        }
        report("validation, first half", detector, first);
        report("validation, second half (not used for anything)", detector, second);

        System.out.println();
        System.out.println("Mean of the document numbers per level (second half of validation): how the levels differ");
        System.out.print(String.format(Locale.ROOT, "%-30s", "number"));
        for (String level : LEVELS) {
            System.out.print(String.format(Locale.ROOT, "%12s", level));
        }
        System.out.println();
        double[][] means = new double[LEVELS.length][DifficultyDetector.FEATURE_NAMES.length];
        for (int l = 0; l < LEVELS.length; l++) {
            List<List<String>> docs = second.get(LEVELS[l]);
            for (List<String> doc : docs) {
                double[] v = detector.describe(doc);
                for (int j = 0; j < v.length; j++) {
                    means[l][j] += v[j] / docs.size();
                }
            }
        }
        for (int j = 0; j < DifficultyDetector.FEATURE_NAMES.length; j++) {
            System.out.print(String.format(Locale.ROOT, "%-30s", DifficultyDetector.FEATURE_NAMES[j]));
            for (int l = 0; l < LEVELS.length; l++) {
                System.out.print(String.format(Locale.ROOT, "%12.4f", means[l][j]));
            }
            System.out.println();
        }

        String path = MODELS_DIR + "difficulty.ser";
        detector.save(path);
        System.out.println();
        System.out.println("Detector saved to " + path);
    }

    private static Map<String, List<List<String>>> sample(Map<String, List<List<String>>> all, int perLevel) {
        Map<String, List<List<String>>> result = new LinkedHashMap<>();
        for (Map.Entry<String, List<List<String>>> e : all.entrySet()) {
            result.put(e.getKey(), e.getValue().subList(0, Math.min(perLevel, e.getValue().size())));
        }
        return result;
    }

    /** Accuracy and the confusion matrix on documents of known level. */
    private static void report(String name, DifficultyDetector detector, Map<String, List<List<String>>> documents) {

        int[][] confusion = new int[LEVELS.length][LEVELS.length];
        int right = 0, total = 0;
        for (int l = 0; l < LEVELS.length; l++) {
            List<List<String>> docs = documents.get(LEVELS[l]);
            int[] detected = new int[docs.size()];
            IntStream.range(0, docs.size()).parallel()
                    .forEach(i -> detected[i] = DifficultyDetector.argMax(detector.probabilities(docs.get(i))));
            for (int d : detected) {
                confusion[l][d]++;
                total++;
            }
            right += confusion[l][l];
        }
        System.out.println();
        System.out.println(String.format(Locale.ROOT, "%s: accuracy %.4f (%d documents)", name,
                total == 0 ? 0.0 : (double) right / total, total));
        System.out.println(String.format(Locale.ROOT, "  %-8s %8s %8s %8s", "", LEVELS[0], LEVELS[1], LEVELS[2]));
        for (int l = 0; l < LEVELS.length; l++) {
            System.out.println(String.format(Locale.ROOT, "  %-8s %8d %8d %8d", LEVELS[l], confusion[l][0], confusion[l][1],
                    confusion[l][2]));
        }
    }

    /** The probabilities of a model for one document, its truth and the level it was in. */
    private static final class Scored {
        double[] probabilities;
        int[] truth;
        List<String> sentences;
    }

    private static List<Scored> score(ModelStore.Loaded model, List<Problem> problems) {

        Scored[] scored = new Scored[problems.size()];
        IntStream.range(0, scored.length).parallel().forEach(i -> {
            Scored s = new Scored();
            s.sentences = problems.get(i).sentences;
            s.truth = problems.get(i).changes;
            s.probabilities = model.changeScores(s.sentences);
            scored[i] = s;
        });
        List<Scored> list = new ArrayList<>();
        for (Scored s : scored) {
            if (s.probabilities.length == s.truth.length) { // a document whose truth does not line up is skipped
                list.add(s);
            }
        }
        return list;
    }

    private static List<File> modelFiles(String which) {

        List<File> files = new ArrayList<>();
        for (String[] k : KNOWN_MODELS) {
            if (which.equals("all") || which.equalsIgnoreCase(k[0])) {
                File f = new File(MODELS_DIR, k[1]);
                if (f.isFile()) {
                    files.add(f);
                } else if (!which.equals("all")) {
                    throw new IllegalStateException("model file not found: " + f);
                }
            }
        }
        if (files.isEmpty()) {
            throw new IllegalStateException("no trained model found in " + MODELS_DIR);
        }
        return files;
    }

    static void calibrate(String which) throws IOException, ClassNotFoundException {

        for (File file : modelFiles(which)) {

            ModelStore.Loaded model = ModelStore.load(file.getPath());
            System.out.println();
            System.out.println("Calibrating " + file.getName() + " (type " + model.type + ", pooled threshold " + model.threshold + ")");

            Map<String, Double> thresholds = new LinkedHashMap<>();
            for (String level : LEVELS) {
                List<Scored> scored = score(model, load("validation", level, DatasetLoader.Part.FIRST_HALF));
                double[] probabilities = flatten(scored);
                int[] truth = flattenTruth(scored);
                double threshold = ThresholdSelector.findBestThreshold(probabilities, truth);
                thresholds.put(level, threshold);
                System.out.println(String.format(Locale.ROOT,
                        "  %-6s threshold %.4f -> validation (first half) macro-F1 %.4f  [the pooled threshold gives %.4f]",
                        level, threshold, Metrics.calcMacroF1(ThresholdSelector.apply(probabilities, threshold), truth),
                        Metrics.calcMacroF1(ThresholdSelector.apply(probabilities, model.threshold), truth)));
            }
            LevelRouter.writeLevels(file, thresholds, "chosen on the first half of the validation split of each level; by DifficultyMain calibrate");
            System.out.println("  written to " + LevelRouter.levelsFile(file));
        }
    }

    private static double[] flatten(List<Scored> scored) {
        int n = 0;
        for (Scored s : scored) {
            n += s.probabilities.length;
        }
        double[] all = new double[n];
        int k = 0;
        for (Scored s : scored) {
            for (double p : s.probabilities) {
                all[k++] = p;
            }
        }
        return all;
    }

    private static int[] flattenTruth(List<Scored> scored) {
        int n = 0;
        for (Scored s : scored) {
            n += s.truth.length;
        }
        int[] all = new int[n];
        int k = 0;
        for (Scored s : scored) {
            for (int t : s.truth) {
                all[k++] = t;
            }
        }
        return all;
    }

    static void evaluate(String which) throws IOException, ClassNotFoundException {

        DifficultyDetector detector = DifficultyDetector.load(MODELS_DIR + "difficulty.ser");

        Map<String, List<Problem>> test = new LinkedHashMap<>();
        for (String level : LEVELS) {
            test.put(level, load("validation", level, DatasetLoader.Part.SECOND_HALF));
        }

        for (File file : modelFiles(which)) {

            ModelStore.Loaded model = ModelStore.load(file.getPath());
            File levelsFile = LevelRouter.levelsFile(file);
            if (!levelsFile.isFile()) {
                System.out.println("Skipping " + file.getName() + ": no " + levelsFile.getName() + " (run calibrate first)");
                continue;
            }
            Map<String, Double> levelThresholds = LevelRouter.readLevels(levelsFile);

            System.out.println();
            System.out.println("==================== " + file.getName() + " on the second half of validation ====================");
            System.out.println("macro-F1 per true level. 'pooled' = the one threshold of the model (" + model.threshold
                    + "), 'true level' = the threshold of the level the document is really in (needs the level to be given), "
                    + "'detected' = the threshold of the level the detector says (nothing is given), "
                    + "'folder vote' = one detected level for the whole level folder.");
            System.out.println(String.format(Locale.ROOT, "%-8s %7s %10s %10s %12s %10s %12s", "level", "pairs", "all-0", "pooled",
                    "true level", "detected", "folder vote"));

            Map<String, int[][]> predictions = new LinkedHashMap<>(); // scheme -> per level, flat
            String[] schemes = { "pooled", "true", "detected", "vote" };
            for (String scheme : schemes) {
                predictions.put(scheme, new int[LEVELS.length][]);
            }
            int[][] truthByLevel = new int[LEVELS.length][];
            int[][] confusion = new int[LEVELS.length][LEVELS.length];

            for (int l = 0; l < LEVELS.length; l++) {

                List<Scored> scored = score(model, test.get(LEVELS[l]));
                int[] truth = flattenTruth(scored);
                truthByLevel[l] = truth;

                // the detector's probabilities for each document
                double[][] detectorProbabilities = new double[scored.size()][];
                final List<Scored> scoredFinal = scored;
                IntStream.range(0, scored.size()).parallel()
                        .forEach(i -> detectorProbabilities[i] = detector.probabilities(scoredFinal.get(i).sentences));

                double[] meanProbabilities = new double[LEVELS.length];
                for (double[] p : detectorProbabilities) {
                    for (int k = 0; k < LEVELS.length; k++) {
                        meanProbabilities[k] += p[k] / Math.max(1, detectorProbabilities.length);
                    }
                }
                double voteThreshold = levelThresholds.get(LEVELS[DifficultyDetector.argMax(meanProbabilities)]);

                List<int[]> pooled = new ArrayList<>(), exact = new ArrayList<>(), detected = new ArrayList<>(),
                        vote = new ArrayList<>();
                for (int i = 0; i < scored.size(); i++) {
                    double[] p = scored.get(i).probabilities;
                    int d = DifficultyDetector.argMax(detectorProbabilities[i]);
                    confusion[l][d]++;
                    pooled.add(ThresholdSelector.apply(p, model.threshold));
                    exact.add(ThresholdSelector.apply(p, levelThresholds.get(LEVELS[l])));
                    detected.add(ThresholdSelector.apply(p, levelThresholds.get(LEVELS[d])));
                    vote.add(ThresholdSelector.apply(p, voteThreshold));
                }
                predictions.get("pooled")[l] = join(pooled);
                predictions.get("true")[l] = join(exact);
                predictions.get("detected")[l] = join(detected);
                predictions.get("vote")[l] = join(vote);

                System.out.println(String.format(Locale.ROOT, "%-8s %7d %10.4f %10.4f %12.4f %10.4f %12.4f", LEVELS[l],
                        truth.length, Metrics.calcMacroF1(new int[truth.length], truth),
                        Metrics.calcMacroF1(predictions.get("pooled")[l], truth),
                        Metrics.calcMacroF1(predictions.get("true")[l], truth),
                        Metrics.calcMacroF1(predictions.get("detected")[l], truth),
                        Metrics.calcMacroF1(predictions.get("vote")[l], truth)));
            }

            int[] allTruth = concat(truthByLevel);
            System.out.println(String.format(Locale.ROOT, "%-8s %7d %10.4f %10.4f %12.4f %10.4f %12.4f", "all", allTruth.length,
                    Metrics.calcMacroF1(new int[allTruth.length], allTruth),
                    Metrics.calcMacroF1(concat(predictions.get("pooled")), allTruth),
                    Metrics.calcMacroF1(concat(predictions.get("true")), allTruth),
                    Metrics.calcMacroF1(concat(predictions.get("detected")), allTruth),
                    Metrics.calcMacroF1(concat(predictions.get("vote")), allTruth)));

            int right = 0, total = 0;
            for (int l = 0; l < LEVELS.length; l++) {
                for (int d = 0; d < LEVELS.length; d++) {
                    total += confusion[l][d];
                }
                right += confusion[l][l];
            }
            System.out.println(String.format(Locale.ROOT,
                    "Detector on these documents: accuracy %.4f. Rows true level, columns detected (easy/medium/hard): "
                            + "easy %s | medium %s | hard %s",
                    total == 0 ? 0.0 : (double) right / total, java.util.Arrays.toString(confusion[0]),
                    java.util.Arrays.toString(confusion[1]), java.util.Arrays.toString(confusion[2])));
        }
    }

    private static int[] join(List<int[]> parts) {
        return concat(parts.toArray(new int[0][]));
    }

    private static int[] concat(int[][] parts) {
        int n = 0;
        for (int[] p : parts) {
            n += p.length;
        }
        int[] all = new int[n];
        int k = 0;
        for (int[] p : parts) {
            for (int v : p) {
                all[k++] = v;
            }
        }
        return all;
    }
}
