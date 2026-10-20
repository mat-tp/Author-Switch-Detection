import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import data.DatasetLoader;
import features.FeatureConfig;
import features.FeatureExtractor;
import models.DifficultyDetector;
import models.LevelRouter;
import models.ModelStore;
import models.RetrievalModel;

/**
 * The software that is submitted to TIRA: reads the problems of the input folder, predicts the author
 * switches with a TRAINED model, writes one solution file per problem.
 *
 * java -cp out TiraRunner -i INPUT_DIR -o OUTPUT_DIR [--model neural|logistic|tree|retrieval|PATH]
 *                         [--threshold T] [--difficulty auto|auto-dataset|easy|medium|hard|none]
 *
 * The difficulty level (easy / medium / hard) does NOT have to be given: by default (--difficulty auto) the level of
 * every document is detected from its text (models.DifficultyDetector) and the threshold of that level is used
 * (models.LevelRouter, files difficulty.ser and NAME.ser.levels next to the model, made by DifficultyMain). When these
 * files are missing the runner says so and uses the single threshold saved in the model.
 *
 * input : problem-N.txt (one sentence per line), in INPUT_DIR or in sub-folders of it (easy / medium / hard)
 * output: solution-problem-N.json = {"changes": [0, 1, 0, ...]}, one value per pair of neighbouring
 *         sentences, in the same sub-folder of OUTPUT_DIR as the problem was in INPUT_DIR
 *
 * Rules this runner keeps, because the first submission broke them (it returned the constant 1, so the
 * software on TIRA did not match the paper):
 * - it never makes up an answer: no model file, a model of an older version, a problem it can not read
 *   = it stops with an error and a non-zero exit code. There is no default prediction.
 * - the model is the one saved by Main / RetrievalMain (models.ModelStore), with the feature settings,
 *   z-score parameters and threshold it was trained with. The threshold was chosen on validation data.
 * - when the whole run is one constant (every pair 0, or every pair 1) it prints a warning: that is what
 *   a broken deployment looks like, on real data a working model does not do that.
 *
 * The model is chosen by --model, or the environment variable MODEL; without either, the first of
 * neural, logistic, tree, retrieval that exists in -Dmodels.dir (default data/models/) is used, and the
 * name is printed. The sandbox could not run Docker / TIRA: the command line behaviour is tested
 * (see scripts/tira_selftest.sh), the container build is not.
 */
public class TiraRunner {

    /** --threshold T replaces the threshold saved in the model (for one PAN dataset: the threshold of that level). */
    private static Double thresholdOverride = null;

    /**
     * How the threshold is chosen when --threshold is not given:
     * auto          (default) the difficulty of every document is detected from its text and the threshold of that level is used
     * auto-dataset  one level for the whole input folder: the level most documents look like
     * easy | medium | hard   that level, without detection
     * none          the single threshold saved in the model
     */
    private static String difficultyMode = "auto";

    private static final String MODELS_DIR = System.getProperty("models.dir", "data/models/");

    public static void main(String[] args) {

        try {
            System.exit(run(args));
        } catch (Exception e) {
            System.err.println("ERROR: " + e.getMessage());
            System.exit(1);
        }
    }

    static int run(String[] args) throws IOException, ClassNotFoundException {

        String input = null, output = null, model = System.getenv("MODEL");
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-i":
                case "--input":
                    input = args[++i];
                    break;
                case "-o":
                case "--output":
                    output = args[++i];
                    break;
                case "-m":
                case "--model":
                    model = args[++i];
                    break;
                case "--threshold":
                    thresholdOverride = Double.valueOf(args[++i]);
                    break;
                case "--difficulty":
                    difficultyMode = args[++i].trim().toLowerCase(java.util.Locale.ROOT);
                    break;
                default:
                    throw new IllegalArgumentException("unknown argument " + args[i]
                            + " (usage: -i INPUT_DIR -o OUTPUT_DIR [--model neural|logistic|tree|retrieval|PATH] [--threshold T] [--difficulty auto|auto-dataset|easy|medium|hard|none])");
            }
        }
        if (input == null || output == null) {
            throw new IllegalArgumentException("usage: -i INPUT_DIR -o OUTPUT_DIR [--model neural|logistic|tree|retrieval|PATH] [--threshold T] [--difficulty auto|auto-dataset|easy|medium|hard|none]");
        }

        if (!java.util.Arrays.asList("auto", "auto-dataset", "easy", "medium", "hard", "none").contains(difficultyMode)) {
            throw new IllegalArgumentException("--difficulty must be auto, auto-dataset, easy, medium, hard or none, not '"
                    + difficultyMode + "'");
        }

        Path inputDir = Paths.get(input);
        Path outputDir = Paths.get(output);
        if (!Files.isDirectory(inputDir)) {
            throw new IOException("input folder does not exist: " + inputDir);
        }

        // ---- the model: no model, no output
        Path modelPath = resolveModel(model);
        ModelStore.Loaded loaded = ModelStore.load(modelPath.toString()); // fails with "retrain the model" for an old file
        System.out.println("Model: " + loaded.type + " (" + modelPath + ")");
        System.out.println("Trained with: " + loaded.description);
        LevelRouter router = LevelRouter.forModelFile(modelPath.toFile());
        boolean routed = thresholdOverride == null && !difficultyMode.equals("none");
        if (thresholdOverride != null) {
            System.out.println("Threshold: " + thresholdOverride + " (from --threshold; the model has " + loaded.threshold + ")");
        } else if (!routed) {
            System.out.println("Threshold: " + loaded.threshold + " (the one saved in the model)");
        } else if (!router.ready()) {
            System.err.println("WARNING: the difficulty cannot be detected, so the single threshold saved in the model ("
                    + loaded.threshold + ") is used for every document. Reason: " + router.problem());
            routed = false;
        } else {
            System.out.println("Threshold: chosen by difficulty (" + difficultyMode + "); pooled threshold of the model: "
                    + loaded.threshold);
        }

        // ---- the problems
        List<Path> problems = new ArrayList<>();
        try (Stream<Path> files = Files.walk(inputDir)) {
            files.filter(Files::isRegularFile).filter(p -> {
                String name = p.getFileName().toString();
                return name.startsWith("problem-") && name.endsWith(".txt");
            }).forEach(problems::add);
        }
        Collections.sort(problems);
        if (problems.isEmpty()) {
            throw new IOException("no problem-*.txt file found under " + inputDir);
        }

        // the sentences of every document are read once
        List<List<String>> documents = new ArrayList<>();
        for (Path problem : problems) {
            documents.add(DatasetLoader.extractSentences(Files.readString(problem, StandardCharsets.UTF_8),
                    FeatureConfig.Casing.PRESERVE));
        }

        // one level for the whole folder
        Double folderThreshold = null;
        if (routed && difficultyMode.equals("auto-dataset")) {
            List<double[]> probabilities = new ArrayList<>();
            for (List<String> sentences : documents) {
                probabilities.add(router.detector().probabilities(sentences));
            }
            LevelRouter.Decision decision = router.decideDataset(probabilities, loaded.threshold);
            folderThreshold = decision.threshold;
            System.out.println("Detected for the whole folder: " + decision.level + " (mean probabilities easy/medium/hard "
                    + formatProbabilities(decision.probabilities) + ") -> threshold " + decision.threshold);
        } else if (routed && !difficultyMode.equals("auto")) {
            folderThreshold = router.thresholdOf(difficultyMode, loaded.threshold);
            System.out.println("Level given by --difficulty: " + difficultyMode + " -> threshold " + folderThreshold);
        }

        long pairs = 0, switches = 0;
        int[] detectedCount = new int[DifficultyDetector.LEVELS.length];
        for (int d = 0; d < problems.size(); d++) {

            Path problem = problems.get(d);
            List<String> sentences = documents.get(d);

            double threshold;
            if (thresholdOverride != null) {
                threshold = thresholdOverride;
            } else if (folderThreshold != null) {
                threshold = folderThreshold;
            } else if (routed) {
                LevelRouter.Decision decision = router.decide(sentences, loaded.threshold);
                threshold = decision.threshold;
                if (decision.level != null) {
                    detectedCount[java.util.Arrays.asList(DifficultyDetector.LEVELS).indexOf(decision.level)]++;
                }
            } else {
                threshold = loaded.threshold;
            }

            int[] changes = apply(probabilities(loaded, sentences), threshold);
            pairs += changes.length;
            for (int c : changes) {
                switches += c;
            }

            String id = problem.getFileName().toString().replace("problem-", "").replace(".txt", "");
            Path relative = inputDir.relativize(problem.getParent());
            Path outFolder = outputDir.resolve(relative);
            Files.createDirectories(outFolder);

            StringBuilder json = new StringBuilder("{\"changes\": [");
            for (int i = 0; i < changes.length; i++) {
                json.append(i == 0 ? "" : ", ").append(changes[i]);
            }
            json.append("]}\n");
            Files.writeString(outFolder.resolve("solution-problem-" + id + ".json"), json.toString(),
                    StandardCharsets.UTF_8);
        }

        if (routed && difficultyMode.equals("auto")) {
            System.out.println("Detected difficulty of the documents: easy " + detectedCount[0] + ", medium "
                    + detectedCount[1] + ", hard " + detectedCount[2]);
        }

        System.out.println(String.format(java.util.Locale.ROOT, "Wrote %d solution files, %d pairs, %.1f%% predicted as a change.",
                problems.size(), pairs, pairs == 0 ? 0.0 : 100.0 * switches / pairs));

        if (pairs > 50 && (switches == 0 || switches == pairs)) {
            System.err.println("WARNING: every pair got the same prediction (" + (switches == 0 ? 0 : 1)
                    + "). A trained model does not do that on real data: check that the model, the input and the "
                    + "feature files (dataset/word-processing/...) are the ones the model was trained with.");
        }
        return 0;
    }

    private static String formatProbabilities(double[] p) {
        return String.format(java.util.Locale.ROOT, "%.2f/%.2f/%.2f", p[0], p[1], p[2]);
    }

    /** The probability of a change for every two neighbouring sentences (n - 1 numbers). */
    static double[] probabilities(ModelStore.Loaded loaded, List<String> sentences) {

        int n = Math.max(0, sentences.size() - 1);
        if (n == 0) {
            return new double[0];
        }

        FeatureExtractor extractor = loaded.extractor;

        if (ModelStore.RETRIEVAL.equals(loaded.type)) {
            RetrievalModel retrieval = (RetrievalModel) loaded.model;
            return retrieval.changeScores(retrieval.embed(extractor, sentences));
        }

        List<double[]> raw = new ArrayList<>();
        for (String sentence : sentences) {
            raw.add(extractor.getFeatures(sentence));
        }
        double[][] matrix = new double[n][];
        for (int i = 0; i < n; i++) {
            matrix[i] = extractor.pairVector(raw.get(i), raw.get(i + 1));
        }
        return loaded.probabilities(matrix);
    }

    /** 0/1 for every pair; '>' like models.LogisticRegression.predict, the function every reported number used. */
    static int[] apply(double[] probabilities, double threshold) {

        int[] changes = new int[probabilities.length];
        for (int i = 0; i < changes.length; i++) {
            changes[i] = probabilities[i] > threshold ? 1 : 0;
        }
        return changes;
    }

    private static Path resolveModel(String model) throws IOException {

        String[][] known = { { "neural", "neuralNetwork.ser" }, { "logistic", "logisticRegression.ser" },
                { "tree", "decisionTree.ser" }, { "retrieval", "retrieval.ser" } };

        if (model != null && !model.isBlank()) {
            for (String[] k : known) {
                if (k[0].equalsIgnoreCase(model)) {
                    return existing(Paths.get(MODELS_DIR, k[1]));
                }
            }
            return existing(Paths.get(model));
        }

        for (String[] k : known) {
            Path p = Paths.get(MODELS_DIR, k[1]);
            if (Files.isRegularFile(p)) {
                System.out.println("No --model given: using " + k[0]);
                return p;
            }
        }
        throw new IOException("no trained model found in " + MODELS_DIR + " (train one with Main, or pass --model PATH). "
                + "There is no default prediction.");
    }

    private static Path existing(Path p) throws IOException {
        if (!Files.isRegularFile(p)) {
            throw new IOException("model file not found: " + p);
        }
        return p;
    }
}
