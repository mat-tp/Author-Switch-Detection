package models;

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
 * Chooses the decision threshold of a document by the difficulty level it looks like, so the level never has to be
 * given by hand.
 *
 * Two files belong to it, both in the folder of the model:
 * - difficulty.ser        the DifficultyDetector (DifficultyMain train)
 * - NAME.ser.levels       the threshold of every level for the model NAME.ser (DifficultyMain calibrate), a text file
 *                         with lines like "easy=0.525"
 *
 * Without them the router does not guess: decide() returns the pooled threshold the model was saved with and says why,
 * so the caller can print it.
 */
public final class LevelRouter {

    /** The result for one document (or one dataset). */
    public static final class Decision {
        /** easy / medium / hard, null when the level is not known */
        public final String level;
        /** probability of easy, medium, hard (null when the level is not known) */
        public final double[] probabilities;
        public final double threshold;
        /** one short sentence: where the threshold comes from */
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

    private LevelRouter(DifficultyDetector detector, Map<String, Double> thresholds, String problem) {
        this.detector = detector;
        this.thresholds = thresholds;
        this.problem = problem;
    }

    // ------------------------------------------------------------------
    // Loading (cached: the web app asks on every prediction)
    // ------------------------------------------------------------------

    private static final Map<String, LevelRouter> CACHE = new HashMap<>();
    private static final Map<String, String> STAMPS = new HashMap<>();

    public static File detectorFile(File modelFile) {
        File folder = modelFile.getAbsoluteFile().getParentFile();
        return new File(folder, "difficulty.ser");
    }

    public static File levelsFile(File modelFile) {
        return new File(modelFile.getPath() + ".levels");
    }

    /** The router of a model file. Never null; check ready() / problem(). */
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

    // ------------------------------------------------------------------
    // Use
    // ------------------------------------------------------------------

    /** true when both the detector and the thresholds are there */
    public boolean ready() {
        return detector != null && thresholds != null && thresholds.size() == DifficultyDetector.LEVELS.length;
    }

    /** why the router can not be used (null = it can) */
    public String problem() {
        if (ready()) {
            return null;
        }
        return problem != null ? problem : "the thresholds per level are incomplete";
    }

    public DifficultyDetector detector() {
        return detector;
    }

    /** The threshold of a level, or `pooled` when there is none. */
    public double thresholdOf(String level, double pooled) {
        Double t = thresholds == null ? null : thresholds.get(level);
        return t == null ? pooled : t;
    }

    /** The level of one document and the threshold that belongs to it. */
    public Decision decide(List<String> sentences, double pooled) {

        if (!ready()) {
            return new Decision(null, null, pooled, "pooled threshold of the model (" + problem() + ")");
        }
        double[] p = detector.probabilities(sentences);
        String level = DifficultyDetector.LEVELS[DifficultyDetector.argMax(p)];
        return new Decision(level, p, thresholds.get(level), "threshold of the detected level '" + level + "'");
    }

    /** One level for a whole dataset: the level with the highest mean probability over its documents. */
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

    // ------------------------------------------------------------------
    // The text file with the thresholds
    // ------------------------------------------------------------------

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
