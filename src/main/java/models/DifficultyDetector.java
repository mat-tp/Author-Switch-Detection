package models;

import util.Statistics;
import cli.DifficultyMain;
import java.io.IOException;
import java.io.InvalidClassException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import features.FunctionWords;

/**
 * Heuristic detector classifying document text difficulty into {@code easy}, {@code medium}, or {@code hard} categories.
 *
 * <p>Enables automatic threshold selection in downstream routing pipelines (e.g., {@code LevelRouter})
 * without requiring explicit dataset tier metadata.</p>
 *
 * <p><strong>Feature Extraction Summary:</strong>
 * Evaluates intra-document topic drift and adjacent-sentence stylistic variation across 16 core features:
 * <ul>
 *   <li><strong>Size Metrics:</strong> Sentence count, sentence length mean, and standard deviation.</li>
 *   <li><strong>Topic Continuity:</strong> Cosine similarity across content words (excluding function words) for neighbouring sentences (mean, std, 10th/50th/90th percentiles), document centroid similarity, pairwise average similarity, and vocabulary repetition rates.</li>
 *   <li><strong>Style Continuity:</strong> Euclidean distance between neighbouring sentences across a standardized 12-dimensional style vector (length, punctuation, capitalization, digits, quotes), variance, 90th percentile, and centroid distance.</li>
 * </ul>
 * Features and their squared terms are passed to a {@link SoftmaxRegression} classifier to predict category probabilities.</p>
 *
 * <p><strong>Self-Contained Model:</strong> This class is fully self-contained and operates independently
 * of {@code FeatureExtractor}, allowing direct serialization to disk (e.g., {@code data/models/difficulty.ser}).</p>
 */
public class DifficultyDetector implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Supported document difficulty classification levels. */
    public static final String[] LEVELS = { "easy", "medium", "hard" };

    /** Descriptive string names corresponding to the 16 extracted document features. */
    public static final String[] FEATURE_NAMES = { "log_sentences", "mean_log_words", "std_log_words",
            "topic_cos_neighbour_mean", "topic_cos_neighbour_std", "topic_cos_neighbour_p10", "topic_cos_neighbour_p50",
            "topic_cos_neighbour_p90", "topic_cos_to_centroid_mean", "topic_cos_all_pairs_mean",
            "content_type_token_ratio", "content_words_repeated_share", "style_dist_neighbour_mean",
            "style_dist_neighbour_std", "style_dist_neighbour_p90", "style_dist_to_centroid_mean" };

    private static final int TOPIC_BUCKETS = 1024;
    private static final int STYLE_SIZE = 12;
    private static final int MAX_PAIR_SENTENCES = 150;

    // Standardization parameters for sentence style vectors (fitted on training sentences)
    private final double[] styleMeans;
    private final double[] styleStds;

    // Standardization parameters for overall document feature rows (fitted on training documents)
    private final double[] documentMeans;
    private final double[] documentStds;

    private final SoftmaxRegression classifier;

    /** Concise single-line summary of training configuration and parameters. */
    public final String description;

    /**
     * Private constructor to instantiate an initialized {@link DifficultyDetector}.
     */
    private DifficultyDetector(double[] styleMeans, double[] styleStds, double[] documentMeans, double[] documentStds,
                               SoftmaxRegression classifier, String description) {
        this.styleMeans = styleMeans;
        this.styleStds = styleStds;
        this.documentMeans = documentMeans;
        this.documentStds = documentStds;
        this.classifier = classifier;
        this.description = description;
    }

    /**
     * Trains a new {@code DifficultyDetector} instance on labeled document corpora.
     *
     * @param documentsByLevel Map associating level keys ({@code "easy"}, {@code "medium"}, {@code "hard"}) with lists of document sentence lists.
     * @param iterations Softmax regression gradient descent iterations.
     * @param rate Learning rate parameter.
     * @param l2 L2 regularization coefficient.
     * @return Trained {@link DifficultyDetector} instance.
     * @throws IllegalArgumentException If any level key lacks training documents.
     */
    public static DifficultyDetector train(Map<String, List<List<String>>> documentsByLevel, int iterations,
                                           double rate, double l2) {

        // 1. Fit sentence-level style feature scaler
        double[] sum = new double[STYLE_SIZE], sumSquares = new double[STYLE_SIZE];
        long count = 0;
        for (List<List<String>> documents : documentsByLevel.values()) {
            for (List<String> document : documents) {
                for (String sentence : document) {
                    double[] v = styleVector(sentence);
                    for (int j = 0; j < STYLE_SIZE; j++) {
                        sum[j] += v[j];
                        sumSquares[j] += v[j] * v[j];
                    }
                    count++;
                }
            }
        }
        double[] styleMeans = new double[STYLE_SIZE], styleStds = new double[STYLE_SIZE];
        for (int j = 0; j < STYLE_SIZE; j++) {
            styleMeans[j] = sum[j] / Math.max(1, count);
            styleStds[j] = Math.sqrt(Math.max(1e-12, sumSquares[j] / Math.max(1, count) - styleMeans[j] * styleMeans[j]));
        }

        // 2. Extract document-level feature rows
        List<double[]> rows = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int level = 0; level < LEVELS.length; level++) {
            List<List<String>> documents = documentsByLevel.get(LEVELS[level]);
            if (documents == null || documents.isEmpty()) {
                throw new IllegalArgumentException("no training documents for the level " + LEVELS[level]);
            }
            double[][] numbers = new double[documents.size()][];
            java.util.stream.IntStream.range(0, documents.size()).parallel()
                    .forEach(i -> numbers[i] = documentNumbers(documents.get(i), styleMeans, styleStds));
            for (double[] row : numbers) {
                rows.add(row);
                labels.add(level);
            }
        }

        // 3. Standardize document feature vectors and fit softmax regression
        int features = FEATURE_NAMES.length;
        double[] means = Statistics.columnMeans(rows, 0, features);
        double[] stds = Statistics.columnStandardDeviations(rows, 0, means);
        for (int j = 0; j < features; j++) {
            stds[j] = Math.max(1e-9, stds[j]);
        }

        double[][] x = new double[rows.size()][];
        int[] y = new int[rows.size()];
        for (int i = 0; i < x.length; i++) {
            x[i] = expand(standardise(rows.get(i), means, stds));
            y[i] = labels.get(i);
        }

        SoftmaxRegression classifier = SoftmaxRegression.train(x, y, LEVELS.length, iterations, rate, l2, true);

        String description = String.format(java.util.Locale.ROOT,
                "softmax regression on %d document numbers and their squares, %d training documents, iterations=%d, "
                        + "learningRate=%s, l2=%s",
                features, x.length, iterations, rate, l2);
        return new DifficultyDetector(styleMeans, styleStds, means, stds, classifier, description);
    }

    /**
     * Evaluates class probability distribution across {@code [easy, medium, hard]} for a given document.
     *
     * @param sentences List of sentence strings comprising the document.
     * @return Double array of probabilities corresponding to {@link #LEVELS} order.
     */
    public double[] probabilities(List<String> sentences) {

        double[] numbers = documentNumbers(sentences, styleMeans, styleStds);
        return classifier.probabilities(expand(standardise(numbers, documentMeans, documentStds)));
    }

    /**
     * Classifies a document into its most probable difficulty tier string name.
     *
     * @param sentences List of sentence strings comprising the document.
     * @return String name of the predicted level ({@code "easy"}, {@code "medium"}, or {@code "hard"}).
     */
    public String detect(List<String> sentences) {
        return LEVELS[argMax(probabilities(sentences))];
    }

    /**
     * Helper method to find the index of the maximum value in a probability array.
     *
     * @param p Array of double probability values.
     * @return 0-based index of maximum value.
     */
    public static int argMax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Extracts unstandardized raw feature values for reporting and diagnostics.
     *
     * @param sentences List of sentence strings comprising the document.
     * @return Array of raw extracted document feature values.
     */
    public double[] describe(List<String> sentences) {
        return documentNumbers(sentences, styleMeans, styleStds);
    }

    /** Standardizes feature values into z-scores, clipping extreme outliers at [-5.0, 5.0]. */
    private static double[] standardise(double[] numbers, double[] means, double[] stds) {
        double[] z = new double[numbers.length];
        for (int j = 0; j < z.length; j++) {
            z[j] = Math.max(-5.0, Math.min(5.0, (numbers[j] - means[j]) / stds[j]));
        }
        return z;
    }

    /** Polynomial feature expansion appending squared feature values to allow non-linear decision boundaries. */
    private static double[] expand(double[] z) {
        double[] x = new double[z.length * 2];
        for (int j = 0; j < z.length; j++) {
            x[j] = z[j];
            x[z.length + j] = z[j] * z[j];
        }
        return x;
    }

    /** Sparse topic representation storing sorted non-zero bucket indices and square-rooted frequency values. */
    private static final class Sparse {
        int[] index;
        double[] value;
    }

    /**
     * Computes the 16 raw document feature values from input sentences.
     */
    static double[] documentNumbers(List<String> sentences, double[] styleMeans, double[] styleStds) {

        int n = sentences.size();
        double[] out = new double[FEATURE_NAMES.length];
        out[0] = Math.log(1 + n);
        if (n < 2) {
            return out;
        }

        // Sentence length metrics
        double[] logWords = new double[n];
        for (int i = 0; i < n; i++) {
            logWords[i] = Math.log(1 + countWords(sentences.get(i)));
        }
        out[1] = Statistics.mean(logWords);
        out[2] = Statistics.populationStandardDeviation(logWords);

        // Topic feature extraction
        Sparse[] topic = new Sparse[n];
        double[] centroid = new double[TOPIC_BUCKETS];
        int contentTokens = 0;
        java.util.HashMap<String, Integer> sentencesOfWord = new java.util.HashMap<>();
        for (int i = 0; i < n; i++) {
            List<String> words = contentWords(sentences.get(i));
            contentTokens += words.size();
            for (String word : new java.util.HashSet<>(words)) {
                sentencesOfWord.merge(word, 1, Integer::sum);
            }
            topic[i] = topicVector(words);
            for (int k = 0; k < topic[i].index.length; k++) {
                centroid[topic[i].index[k]] += topic[i].value[k];
            }
        }
        double centroidNorm = 0.0;
        for (double v : centroid) {
            centroidNorm += v * v;
        }
        centroidNorm = Math.sqrt(centroidNorm);

        double[] neighbour = new double[n - 1];
        for (int i = 0; i + 1 < n; i++) {
            neighbour[i] = cosine(topic[i], topic[i + 1]);
        }
        out[3] = Statistics.mean(neighbour);
        out[4] = Statistics.populationStandardDeviation(neighbour);
        out[5] = Statistics.percentile(neighbour, 0.10);
        out[6] = Statistics.percentile(neighbour, 0.50);
        out[7] = Statistics.percentile(neighbour, 0.90);

        double toCentroid = 0.0;
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int k = 0; k < topic[i].index.length; k++) {
                dot += topic[i].value[k] * centroid[topic[i].index[k]];
            }
            toCentroid += centroidNorm == 0.0 ? 0.0 : dot / centroidNorm;
        }
        out[8] = toCentroid / n;

        int m = Math.min(n, MAX_PAIR_SENTENCES);
        double allPairs = 0.0;
        long pairs = 0;
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                allPairs += cosine(topic[i], topic[j]);
                pairs++;
            }
        }
        out[9] = pairs == 0 ? 0.0 : allPairs / pairs;

        out[10] = contentTokens == 0 ? 0.0 : (double) sentencesOfWord.size() / contentTokens;
        int repeated = 0;
        for (int count : sentencesOfWord.values()) {
            if (count >= 2) {
                repeated++;
            }
        }
        out[11] = sentencesOfWord.isEmpty() ? 0.0 : (double) repeated / sentencesOfWord.size();

        // Style feature extraction
        double[][] style = new double[n][];
        double[] styleCentroid = new double[STYLE_SIZE];
        for (int i = 0; i < n; i++) {
            double[] v = styleVector(sentences.get(i));
            for (int j = 0; j < STYLE_SIZE; j++) {
                v[j] = (v[j] - styleMeans[j]) / styleStds[j];
                styleCentroid[j] += v[j] / n;
            }
            style[i] = v;
        }
        double[] styleNeighbour = new double[n - 1];
        for (int i = 0; i + 1 < n; i++) {
            double d = 0.0;
            for (int j = 0; j < STYLE_SIZE; j++) {
                d += Math.abs(style[i][j] - style[i + 1][j]);
            }
            styleNeighbour[i] = d / STYLE_SIZE;
        }
        out[12] = Statistics.mean(styleNeighbour);
        out[13] = Statistics.populationStandardDeviation(styleNeighbour);
        out[14] = Statistics.percentile(styleNeighbour, 0.90);

        double toStyleCentroid = 0.0;
        for (int i = 0; i < n; i++) {
            double d = 0.0;
            for (int j = 0; j < STYLE_SIZE; j++) {
                d += Math.abs(style[i][j] - styleCentroid[j]);
            }
            toStyleCentroid += d / STYLE_SIZE;
        }
        out[15] = toStyleCentroid / n;
        return out;
    }

    /** Helper method counting whitespace-delimited words in a sentence. */
    private static int countWords(String sentence) {
        int words = 0;
        boolean inside = false;
        for (int i = 0; i < sentence.length(); i++) {
            boolean space = Character.isWhitespace(sentence.charAt(i));
            if (!space && !inside) {
                words++;
            }
            inside = !space;
        }
        return words;
    }

    /** Extracts lowercase content words ($\ge 3$ characters) excluding closed-class function words. */
    private static List<String> contentWords(String sentence) {

        List<String> words = new ArrayList<>();
        StringBuilder word = new StringBuilder();
        for (int i = 0; i <= sentence.length(); i++) {
            char c = i < sentence.length() ? sentence.charAt(i) : ' ';
            if (Character.isLetter(c)) {
                word.append(Character.toLowerCase(c));
            } else {
                if (word.length() >= 3 && !FunctionWords.contains(word.toString())) {
                    words.add(word.toString());
                }
                word.setLength(0);
            }
        }
        return words;
    }

    /** Hashes content words into a normalized sparse topic distribution vector. */
    private static Sparse topicVector(List<String> words) {

        java.util.TreeMap<Integer, Double> counts = new java.util.TreeMap<>();
        for (String word : words) {
            int h = word.hashCode() * 0x9E3779B1;
            int bucket = ((h ^ (h >>> 15)) & 0x7fffffff) % TOPIC_BUCKETS;
            counts.merge(bucket, 1.0, Double::sum);
        }
        Sparse s = new Sparse();
        s.index = new int[counts.size()];
        s.value = new double[counts.size()];
        double norm = 0.0;
        int k = 0;
        for (Map.Entry<Integer, Double> e : counts.entrySet()) {
            s.index[k] = e.getKey();
            s.value[k] = Math.sqrt(e.getValue());
            norm += s.value[k] * s.value[k];
            k++;
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < k; i++) {
            s.value[i] = norm == 0.0 ? 0.0 : s.value[i] / norm;
        }
        return s;
    }

    /** Cosine similarity of two unit-length sparse topic vectors (the sparse counterpart of {@code VectorMath.dot}). */
    private static double cosine(Sparse a, Sparse b) {

        double dot = 0.0;
        int i = 0, j = 0;
        while (i < a.index.length && j < b.index.length) {
            if (a.index[i] == b.index[j]) {
                dot += a.value[i] * b.value[j];
                i++;
                j++;
            } else if (a.index[i] < b.index[j]) {
                i++;
            } else {
                j++;
            }
        }
        return dot;
    }

    /** Computes a 12-dimensional sentence-level style feature vector. */
    static double[] styleVector(String s) {

        int chars = Math.max(1, s.length());
        int words = Math.max(1, countWords(s));
        int letters = 0, upper = 0, digits = 0, punctuation = 0, commas = 0, quotes = 0, apostrophes = 0,
                excitement = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isLetter(c)) {
                letters++;
                if (Character.isUpperCase(c)) {
                    upper++;
                }
            } else if (Character.isDigit(c)) {
                digits++;
            } else if (!Character.isWhitespace(c)) {
                punctuation++;
                if (c == ',') {
                    commas++;
                } else if (c == '"' || c == '“' || c == '”') {
                    quotes++;
                } else if (c == '\'' || c == '’') {
                    apostrophes++;
                } else if (c == '!' || c == '?') {
                    excitement++;
                }
            }
        }
        String trimmed = s.trim();
        char last = trimmed.isEmpty() ? ' ' : trimmed.charAt(trimmed.length() - 1);
        char first = trimmed.isEmpty() ? ' ' : trimmed.charAt(0);

        return new double[] { Math.log(1 + words), Math.log(1 + chars), (double) letters / words,
                (double) commas / words, 100.0 * quotes / chars, 100.0 * apostrophes / chars,
                letters == 0 ? 0.0 : (double) upper / letters, (double) digits / chars, (double) punctuation / chars,
                (double) excitement / words, (last == '.' || last == '!' || last == '?' || last == '"'
                || last == '”') ? 1.0 : 0.0,
                (Character.isUpperCase(first) || first == '"' || first == '“') ? 1.0 : 0.0 };
    }

    /**
     * Serializes the current detector instance to disk at the specified path.
     *
     * @param path Target file destination path string.
     * @throws IOException If file creation or write stream operation fails.
     */
    public void save(String path) throws IOException {
        Path p = Paths.get(path);
        if (p.getParent() != null) {
            Files.createDirectories(p.getParent());
        }
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(p))) {
            out.writeObject(this);
        }
    }

    /**
     * Deserializes a stored {@code DifficultyDetector} instance from disk.
     *
     * @param path Source model file path string.
     * @return Deserialized {@link DifficultyDetector} instance.
     * @throws IOException If file access fails or serialized version mismatches current class layout.
     */
    public static DifficultyDetector load(String path) throws IOException {
        try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(Paths.get(path)))) {
            return (DifficultyDetector) in.readObject();
        } catch (InvalidClassException e) {
            throw new IOException("the difficulty detector " + path + " is from another version of the code: train it again "
                    + "(DifficultyMain train)", e);
        } catch (ClassNotFoundException | ClassCastException e) {
            throw new IOException(path + " is not a difficulty detector: " + e.getMessage(), e);
        }
    }
}