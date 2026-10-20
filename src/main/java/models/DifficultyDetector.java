package models;

import java.io.IOException;
import java.io.InvalidClassException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import features.FunctionWords;

/**
 * Tells from the TEXT of a document alone whether it looks like an easy, medium or hard document, so nobody has
 * to say which PAN dataset is being processed (the level is needed to choose the decision threshold, see LevelRouter).
 *
 * What differs between the levels is how much the topic changes inside a document (easy: a lot, hard: not at all) and
 * how different neighbouring sentences look. So the detector turns a document into 16 numbers:
 * - size: number of sentences, mean and spread of the sentence length
 * - topic: cosine similarity of the content words (function words left out) of neighbouring sentences (mean, std, 10th /
 *   50th / 90th percentile), of every sentence to the document centroid, of all pairs; how much of the content
 *   vocabulary repeats between sentences
 * - style: distance between neighbouring sentences in a small standardised style vector (length, punctuation, capitals,
 *   digits, quotes ...), its spread, its 90th percentile and the distance to the document centroid
 * and a softmax regression (on the numbers and their squares) turns them into the probability of each level.
 *
 * Everything the detector needs is inside this object (it does not use the FeatureExtractor), so it can be saved to
 * data/models/difficulty.ser and used on its own.
 *
 * The labels come from the folder the training documents are in (easy / medium / hard). How well a level can be
 * recognised from the text is an empirical question: DifficultyMain prints the accuracy and the confusion matrix.
 */
public class DifficultyDetector implements Serializable {

    private static final long serialVersionUID = 1L;

    public static final String[] LEVELS = { "easy", "medium", "hard" };

    public static final String[] FEATURE_NAMES = { "log_sentences", "mean_log_words", "std_log_words",
            "topic_cos_neighbour_mean", "topic_cos_neighbour_std", "topic_cos_neighbour_p10", "topic_cos_neighbour_p50",
            "topic_cos_neighbour_p90", "topic_cos_to_centroid_mean", "topic_cos_all_pairs_mean",
            "content_type_token_ratio", "content_words_repeated_share", "style_dist_neighbour_mean",
            "style_dist_neighbour_std", "style_dist_neighbour_p90", "style_dist_to_centroid_mean" };

    private static final int TOPIC_BUCKETS = 1024;
    private static final int STYLE_SIZE = 12;
    private static final int MAX_PAIR_SENTENCES = 150;

    // standardisation of the style vector of a sentence (training sentences)
    private final double[] styleMeans;
    private final double[] styleStds;

    // standardisation of the document numbers (training documents)
    private final double[] documentMeans;
    private final double[] documentStds;

    private final SoftmaxRegression classifier;

    /** one line: how the detector was trained (shown by the tools) */
    public final String description;

    private DifficultyDetector(double[] styleMeans, double[] styleStds, double[] documentMeans, double[] documentStds,
            SoftmaxRegression classifier, String description) {
        this.styleMeans = styleMeans;
        this.styleStds = styleStds;
        this.documentMeans = documentMeans;
        this.documentStds = documentStds;
        this.classifier = classifier;
        this.description = description;
    }

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    /**
     * @param documentsByLevel the training documents (each one a list of sentences) of every level; the
     *                         map must hold the keys "easy", "medium" and "hard"
     */
    public static DifficultyDetector train(Map<String, List<List<String>>> documentsByLevel, int iterations,
            double rate, double l2) {

        // 1. the style scaler, from all the training sentences
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

        // 2. the numbers of every document
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

        // 3. standardise them and fit the classifier
        int features = FEATURE_NAMES.length;
        double[] means = new double[features], stds = new double[features];
        for (double[] row : rows) {
            for (int j = 0; j < features; j++) {
                means[j] += row[j] / rows.size();
            }
        }
        for (double[] row : rows) {
            for (int j = 0; j < features; j++) {
                stds[j] += (row[j] - means[j]) * (row[j] - means[j]) / rows.size();
            }
        }
        for (int j = 0; j < features; j++) {
            stds[j] = Math.max(1e-9, Math.sqrt(stds[j]));
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

    // ------------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------------

    /** The probabilities of easy, medium, hard (in the order of LEVELS) for one document. */
    public double[] probabilities(List<String> sentences) {

        double[] numbers = documentNumbers(sentences, styleMeans, styleStds);
        return classifier.probabilities(expand(standardise(numbers, documentMeans, documentStds)));
    }

    /** The most probable level. */
    public String detect(List<String> sentences) {
        return LEVELS[argMax(probabilities(sentences))];
    }

    public static int argMax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }

    /** The raw (not standardised) document numbers, for the report of how the levels differ. */
    public double[] describe(List<String> sentences) {
        return documentNumbers(sentences, styleMeans, styleStds);
    }

    private static double[] standardise(double[] numbers, double[] means, double[] stds) {
        double[] z = new double[numbers.length];
        for (int j = 0; j < z.length; j++) {
            // clipped, like the scalar features of the models: one odd document must not dominate
            z[j] = Math.max(-5.0, Math.min(5.0, (numbers[j] - means[j]) / stds[j]));
        }
        return z;
    }

    /** The numbers and their squares: a little non-linearity for a linear classifier. */
    private static double[] expand(double[] z) {
        double[] x = new double[z.length * 2];
        for (int j = 0; j < z.length; j++) {
            x[j] = z[j];
            x[z.length + j] = z[j] * z[j];
        }
        return x;
    }

    // ------------------------------------------------------------------
    // The numbers of a document
    // ------------------------------------------------------------------

    /** A topic vector: the content words of a sentence hashed to buckets (sorted indices and their values). */
    private static final class Sparse {
        int[] index;
        double[] value;
    }

    static double[] documentNumbers(List<String> sentences, double[] styleMeans, double[] styleStds) {

        int n = sentences.size();
        double[] out = new double[FEATURE_NAMES.length];
        out[0] = Math.log(1 + n);
        if (n < 2) {
            return out;
        }

        // sentence length
        double[] logWords = new double[n];
        for (int i = 0; i < n; i++) {
            logWords[i] = Math.log(1 + countWords(sentences.get(i)));
        }
        out[1] = mean(logWords);
        out[2] = std(logWords);

        // topic vectors
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
        out[3] = mean(neighbour);
        out[4] = std(neighbour);
        out[5] = percentile(neighbour, 0.10);
        out[6] = percentile(neighbour, 0.50);
        out[7] = percentile(neighbour, 0.90);

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

        // style vectors
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
        out[12] = mean(styleNeighbour);
        out[13] = std(styleNeighbour);
        out[14] = percentile(styleNeighbour, 0.90);

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

    /** Lower-case words of letters (at least 3), without the function words. */
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
            s.value[k] = Math.sqrt(e.getValue()); // square root of the count: one repeated word does not dominate
            norm += s.value[k] * s.value[k];
            k++;
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < k; i++) {
            s.value[i] = norm == 0.0 ? 0.0 : s.value[i] / norm;
        }
        return s;
    }

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
        return dot; // both are unit vectors (or empty)
    }

    /** 12 simple, cheap numbers about how a sentence is written (standardised with the training sentences). */
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

    // ------------------------------------------------------------------
    // Small statistics
    // ------------------------------------------------------------------

    private static double mean(double[] v) {
        double s = 0.0;
        for (double x : v) {
            s += x;
        }
        return v.length == 0 ? 0.0 : s / v.length;
    }

    private static double std(double[] v) {
        double m = mean(v);
        double s = 0.0;
        for (double x : v) {
            s += (x - m) * (x - m);
        }
        return v.length == 0 ? 0.0 : Math.sqrt(s / v.length);
    }

    private static double percentile(double[] v, double p) {
        double[] sorted = v.clone();
        Arrays.sort(sorted);
        if (sorted.length == 0) {
            return 0.0;
        }
        return sorted[(int) Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1) + 0.5))];
    }

    // ------------------------------------------------------------------
    // Saving
    // ------------------------------------------------------------------

    public void save(String path) throws IOException {
        Path p = Paths.get(path);
        if (p.getParent() != null) {
            Files.createDirectories(p.getParent());
        }
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(p))) {
            out.writeObject(this);
        }
    }

    /** Loads a detector saved by this class (the user's own file; a file of an older version is refused with a message). */
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
