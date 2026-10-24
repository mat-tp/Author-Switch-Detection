package routing;

import models.DifficultyDetector;
import cli.DifficultyMain;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Dynamic decision threshold router that selects classification thresholds for documents
 * based on their predicted difficulty level without requiring manual configuration.
 *
 * <p>Operates alongside two model-associated files located in the model directory:
 * <ul>
 *   <li>{@code difficulty.ser}: Serialized {@link DifficultyDetector} instance (trained via {@code DifficultyMain train}).</li>
 *   <li>{@code NAME.ser.levels}: Calibrated threshold mapping file for model {@code NAME.ser} (generated via {@code DifficultyMain calibrate}), containing key-value pairs such as {@code easy=0.525}.</li>
 * </ul>
 * </p>
 *
 * <p><strong>Fallback Behavior:</strong> If either resource file is missing or unreadable,
 * {@link #decide(List, double)} gracefully falls back to the model's default pooled threshold
 * and populates explanatory diagnostic context in {@link Decision#source}.</p>
 */
public final class LevelRouter {

    /**
     * Immutable container holding threshold selection results and level probabilities for a document or dataset.
     */
    public static final class Decision {

        /** Predicted level tier ({@code "easy"}, {@code "medium"}, or {@code "hard"}); {@code null} if level prediction is unavailable. */
        public final String level;

        /** Array of level probabilities corresponding to {@link DifficultyDetector#LEVELS}; {@code null} if unavailable. */
        public final double[] probabilities;

        /** Target decision threshold selected for evaluation. */
        public final double threshold;

        /** Descriptive summary explaining how or where the threshold was determined. */
        public final String source;

        Decision(String level, double[] probabilities, double threshold, String source) {
            this.level = level;
            this.probabilities = probabilities;
            this.threshold = threshold;
            this.source = source;
        }
    }

    private final DifficultyDetector detector;
    private final Map<String, Double> thresholds;
    private final String problem;

    /**
     * Private constructor to instantiate an initialized {@link LevelRouter}.
     */
    private LevelRouter(DifficultyDetector detector, Map<String, Double> thresholds, String problem) {
        this.detector = detector;
        this.thresholds = thresholds;
        this.problem = problem;
    }

    private static final Map<String, LevelRouter> CACHE = new HashMap<>();
    private static final Map<String, String> STAMPS = new HashMap<>();

    /**
     * Resolves the associated {@code difficulty.ser} file location for a given model file.
     *
     * @param modelFile Target model file handle.
     * @return File handle pointing to {@code difficulty.ser} in the model directory.
     */
    public static File detectorFile(File modelFile) {
        File folder = modelFile.getAbsoluteFile().getParentFile();
        return new File(folder, "difficulty.ser");
    }

    /**
     * Resolves the associated level threshold file location for a given model file.
     *
     * @param modelFile Target model file handle.
     * @return File handle pointing to {@code <modelFile>.levels}.
     */
    public static File levelsFile(File modelFile) {
        return new File(modelFile.getPath() + ".levels");
    }

    /**
     * Factory method retrieving or creating a cached {@link LevelRouter} instance for a specified model file.
     *
     * @param modelFile Target model file handle.
     * @return Cached or freshly loaded {@link LevelRouter} instance (never {@code null}).
     */
    public static synchronized LevelRouter forModelFile(File modelFile) {

        File detectorFile = detectorFile(modelFile);
        File levelsFile = levelsFile(modelFile);
        String key = modelFile.getAbsolutePath();
        String stamp = detectorFile.lastModified() + "/" + detectorFile.length() + "|" + levelsFile.lastModified() + "/"
                + levelsFile.length();

        LevelRouter cached = CACHE.get(key);
        if (cached != null && stamp.equals(STAMPS.get(key))) {
            return cached;
        }

        DifficultyDetector detector = null;
        Map<String, Double> thresholds = null;
        StringBuilder problem = new StringBuilder();

        if (!detectorFile.isFile()) {
            problem.append("no difficulty detector (").append(detectorFile.getPath())
                    .append("): train it with DifficultyMain train");
        } else {
            try {
                detector = DifficultyDetector.load(detectorFile.getPath());
            } catch (IOException e) {
                problem.append(e.getMessage());
            }
        }

        if (!levelsFile.isFile()) {
            if (problem.length() > 0) {
                problem.append("; ");
            }
            problem.append("no thresholds per level (").append(levelsFile.getPath())
                    .append("): create them with DifficultyMain calibrate");
        } else {
            try {
                thresholds = readLevels(levelsFile);
            } catch (IOException | RuntimeException e) {
                if (problem.length() > 0) {
                    problem.append("; ");
                }
                problem.append("could not read ").append(levelsFile.getPath()).append(": ").append(e.getMessage());
            }
        }

        LevelRouter router = new LevelRouter(detector, thresholds, problem.length() == 0 ? null : problem.toString());
        CACHE.put(key, router);
        STAMPS.put(key, stamp);
        return router;
    }

    /**
     * Checks whether both the difficulty detector and complete level threshold mappings are loaded.
     *
     * @return {@code true} if router is fully operational; {@code false} otherwise.
     */
    public boolean ready() {
        return detector != null && thresholds != null && thresholds.size() == DifficultyDetector.LEVELS.length;
    }

    /**
     * Provides diagnostic details if the router cannot be fully initialized.
     *
     * @return Error message description string, or {@code null} if {@link #ready()} is {@code true}.
     */
    public String problem() {
        if (ready()) {
            return null;
        }
        return problem != null ? problem : "the thresholds per level are incomplete";
    }

    /**
     * Returns the underlying difficulty detector instance.
     *
     * @return Loaded {@link DifficultyDetector} instance, or {@code null} if missing.
     */
    public DifficultyDetector detector() {
        return detector;
    }

    /**
     * Retrieves the specific decision threshold for a given difficulty level string.
     *
     * @param level Target difficulty tier string (e.g., {@code "easy"}).
     * @param pooled Fallback threshold returned if level mappings are absent.
     * @return Calibrated double threshold value.
     */
    public double thresholdOf(String level, double pooled) {
        Double t = thresholds == null ? null : thresholds.get(level);
        return t == null ? pooled : t;
    }

    /**
     * Predicts document difficulty and resolves its corresponding decision threshold.
     *
     * @param sentences List of sentence strings comprising the document.
     * @param pooled Fallback pooled threshold value.
     * @return Populate {@link Decision} record containing selected threshold and level metadata.
     */
    public Decision decide(List<String> sentences, double pooled) {

        if (!ready()) {
            return new Decision(null, null, pooled, "pooled threshold of the model (" + problem() + ")");
        }
        double[] p = detector.probabilities(sentences);
        String level = DifficultyDetector.LEVELS[DifficultyDetector.argMax(p)];
        return new Decision(level, p, thresholds.get(level), "threshold of the detected level '" + level + "'");
    }

    /**
     * Evaluates a dataset-wide decision threshold based on the highest mean level probability across all documents.
     *
     * @param documentProbabilities List of per-document level probability arrays.
     * @param pooled Fallback pooled threshold value.
     * @return Populate {@link Decision} record containing selected threshold and aggregated level metadata.
     */
    public Decision decideDataset(List<double[]> documentProbabilities, double pooled) {

        if (!ready() || documentProbabilities.isEmpty()) {
            return new Decision(null, null, pooled, "pooled threshold of the model (" + problem() + ")");
        }
        double[] mean = new double[DifficultyDetector.LEVELS.length];
        for (double[] p : documentProbabilities) {
            for (int i = 0; i < mean.length; i++) {
                mean[i] += p[i] / documentProbabilities.size();
            }
        }
        String level = DifficultyDetector.LEVELS[DifficultyDetector.argMax(mean)];
        return new Decision(level, mean, thresholds.get(level),
                "threshold of the level '" + level + "' that most documents of the dataset look like");
    }

    /**
     * Parses key-value level thresholds from a specified plain text configuration file.
     *
     * @param file Target text file handle.
     * @return Map linking level string keys to double threshold values.
     * @throws IOException If file reading fails.
     */
    public static Map<String, Double> readLevels(File file) throws IOException {

        Map<String, Double> map = new LinkedHashMap<>();
        for (String line : Files.readAllLines(file.toPath(), StandardCharsets.UTF_8)) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#") || !line.contains("=")) {
                continue;
            }
            String[] parts = line.split("=", 2);
            map.put(parts[0].trim().toLowerCase(Locale.ROOT), Double.valueOf(parts[1].trim()));
        }
        return map;
    }

    /**
     * Writes dynamic level thresholds to a text file associated with a target model file.
     *
     * @param modelFile Target model file handle.
     * @param thresholds Map of level threshold values.
     * @param comment Additional header comment string to include in output file.
     * @throws IOException If file creation or write stream operation fails.
     */
    public static void writeLevels(File modelFile, Map<String, Double> thresholds, String comment) throws IOException {

        StringBuilder text = new StringBuilder();
        text.append("# Threshold of each difficulty level for ").append(modelFile.getName()).append('\n');
        text.append("# ").append(comment).append('\n');
        for (String level : DifficultyDetector.LEVELS) {
            if (thresholds.containsKey(level)) {
                text.append(level).append('=').append(String.format(Locale.ROOT, "%.4f", thresholds.get(level)))
                        .append('\n');
            }
        }
        Files.writeString(levelsFile(modelFile).toPath(), text.toString(), StandardCharsets.UTF_8);
    }
}