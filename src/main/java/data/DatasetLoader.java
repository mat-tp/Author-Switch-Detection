package data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import features.FeatureConfig;
import features.TextUtils;

public class DatasetLoader {

    /** Fixed seed, so the validation and testing halves are the same on every run. */
    private static final long SPLIT_SEED = 42L;

    private static final java.util.regex.Pattern AUTHORS_ENTRY =
            java.util.regex.Pattern.compile("\"authors\"\\s*:\\s*(\\d+)");

    /** Which problems (documents) of a split are loaded. */
    public enum Part {
        ALL, FIRST_HALF, SECOND_HALF
    }

    /**
     * ONE problem (document) of the dataset: its sentences in order and the change
     * label of every boundary (changes[i] = 1 when the author changes between
     * sentence i and sentence i + 1).
     *
     * Used where the order of the sentences matters: the retrieval model (the
     * segments of a document), the TIRA run and the per-document evaluation.
     */
    public static class Problem {

        public final String id;
        public final String difficulty;
        public final List<String> sentences;
        public final int[] changes;

        /** Number of authors from the truth file, 0 when the file does not say. */
        public final int authors;

        public Problem(String id, String difficulty, List<String> sentences, int[] changes) {
            this(id, difficulty, sentences, changes, 0);
        }

        public Problem(String id, String difficulty, List<String> sentences, int[] changes, int authors) {
            this.id = id;
            this.difficulty = difficulty;
            this.sentences = sentences;
            this.changes = changes;
            this.authors = authors;
        }

        /**
         * Segment number of every sentence: it goes up by one at every change. Two sentences with the
         * same segment number were written by the same author without another author in between.
         * The truth only says how many authors there are, not which segment belongs to which author, so two
         * different segments can still have the same author; {@link SegmentAuthorship} says when the
         * truth settles that.
         */
        public int[] segments() {

            int[] segments = new int[sentences.size()];
            for (int i = 1; i < segments.length; i++) {
                segments[i] = segments[i - 1] + (changes[i - 1] == 1 ? 1 : 0);
            }
            return segments;
        }
    }

    /**
     * Loads all sentence pairs from a specified split and difficulty (the text as it is written).
     *
     * @param basePath   base dataset directory
     * @param splitType  train or validation
     * @param difficulty easy, medium, or hard
     * @return list of sentence pairs
     */
    public static List<SentencePair> loadSplit(String basePath, String splitType, String difficulty) {
        return loadSplit(basePath, splitType, difficulty, FeatureConfig.Casing.PRESERVE);
    }

    /**
     * Loads all sentence pairs from a specified split and difficulty.
     *
     * @param casing LOWERCASE lower-cases the text while it is loaded. PRESERVE and
     *               HYBRID keep it (HYBRID lower-cases only inside the profile features,
     *               see FeatureConfig.Casing)
     */
    public static List<SentencePair> loadSplit(String basePath, String splitType, String difficulty,
            FeatureConfig.Casing casing) {

        List<SentencePair> dataset = new ArrayList<>();

        for (Problem problem : loadProblems(basePath, splitType, difficulty, casing, Part.ALL)) {
            addPairs(problem, dataset);
        }

        return dataset;
    }

    /**
     * Loading all sentences
     *
     * @param basePath   basiline path to the project
     * @param splitType  The splitting portion of the dataset (Train/ Val)
     * @param difficulty Difficulty based on the dataset(EASY, MEDIUM, HARD).
     * @return A List of all the sentences from the specified path, splitType and
     *         difficulty
     */
    public static List<String> getAllSentences(String basePath, String splitType, String difficulty) {
        return getAllSentences(basePath, splitType, difficulty, FeatureConfig.Casing.PRESERVE);
    }

    public static List<String> getAllSentences(String basePath, String splitType, String difficulty,
            FeatureConfig.Casing casing) {

        List<String> sentences = new ArrayList<>();
        Path dirPath = Paths.get(basePath, difficulty, splitType);

        if (!Files.exists(dirPath)) {
            System.err.println("Directory does not exist: " + dirPath);
            return sentences;
        }

        for (String problemId : listProblemIds(dirPath)) {

            try {
                String problem = Files.readString(dirPath.resolve("problem-" + problemId + ".txt"));
                sentences.addAll(extractSentences(problem, casing));

            } catch (IOException e) {
                System.err.println("Error reading problem " + problemId);
                System.err.println(e.getMessage());
            }
        }

        return sentences;
    }

    /**
     * Overloading the getAllSentences methods to return all the sentences based on
     * the split type instead
     *
     * @param basePath  basiline path to the project
     * @param splitType The splitting portion of the dataset (Train/ Val)
     * @return A List of all the sentences from the specified path, splitType and
     *         difficulty
     */
    public static List<String> getAllSentences(String basePath, String splitType) {
        return getAllSentences(basePath, splitType, FeatureConfig.Casing.PRESERVE);
    }

    public static List<String> getAllSentences(String basePath, String splitType, FeatureConfig.Casing casing) {

        // difining the difficulties instead.
        String[] difficulties = { "easy", "medium", "hard" };

        List<String> allSentences = new ArrayList<>();

        for (String difficulty : difficulties) {

            List<String> sentences = getAllSentences(basePath, splitType, difficulty, casing);
            allSentences.addAll(sentences);
        }

        return allSentences;

    }

    /**
     * Loads one half of a split. The problems (documents) are divided into two
     * halves, NOT the pairs, so pairs of the same document never end up in both
     * halves. The problems are shuffled using a fixed seed, so the halves are
     * exactly the same on every run.
     *
     * Used to divide the validation split into a validation half and a testing
     * half, since the real testing dataset is not available.
     *
     * @param basePath   base dataset directory
     * @param splitType  train or validation
     * @param difficulty easy, medium, or hard
     * @param firstHalf  true returns the first half (validation), false returns
     *                   the second half (testing)
     * @return list of sentence pairs from the chosen half
     */
    public static List<SentencePair> loadSplitHalf(String basePath, String splitType, String difficulty,
            boolean firstHalf) {
        return loadSplitHalf(basePath, splitType, difficulty, firstHalf, FeatureConfig.Casing.PRESERVE);
    }

    public static List<SentencePair> loadSplitHalf(String basePath, String splitType, String difficulty,
            boolean firstHalf, FeatureConfig.Casing casing) {

        List<SentencePair> dataset = new ArrayList<>();

        Part part = firstHalf ? Part.FIRST_HALF : Part.SECOND_HALF;
        for (Problem problem : loadProblems(basePath, splitType, difficulty, casing, part)) {
            addPairs(problem, dataset);
        }

        return dataset;
    }

    /**
     * Loads the problems (documents) of a split, with their sentences in order.
     *
     * @param basePath   base dataset directory
     * @param splitType  train or validation
     * @param difficulty easy, medium, or hard
     * @param casing     see loadSplit
     * @param part       all the problems, or one half of them (the same fixed shuffle as
     *                   loadSplitHalf, so the halves of both methods are the same documents)
     * @return the problems, sorted by id inside the part
     */
    public static List<Problem> loadProblems(String basePath, String splitType, String difficulty,
            FeatureConfig.Casing casing, Part part) {

        List<Problem> problems = new ArrayList<>();
        Path dirPath = Paths.get(basePath, difficulty, splitType);

        if (!Files.exists(dirPath)) {
            System.err.println("Directory does not exist: " + dirPath);
            return problems;
        }

        List<String> problemIds = listProblemIds(dirPath);

        if (part != Part.ALL) {

            // the order of Files.list is not guaranteed, that is why the ids are sorted first
            Collections.shuffle(problemIds, new Random(SPLIT_SEED));

            int halfSize = (problemIds.size() + 1) / 2;
            problemIds = part == Part.FIRST_HALF ? new ArrayList<>(problemIds.subList(0, halfSize))
                    : new ArrayList<>(problemIds.subList(halfSize, problemIds.size()));
            Collections.sort(problemIds);
        }

        for (String problemId : problemIds) {

            Problem problem = loadProblem(dirPath, problemId, difficulty, casing);
            if (problem != null) {
                problems.add(problem);
            }
        }

        return problems;
    }

    /** The ids of the problems in a directory (the number in problem-ID.txt), sorted. */
    private static List<String> listProblemIds(Path dirPath) {

        List<String> problemIds = new ArrayList<>();

        try (java.util.stream.Stream<Path> files = Files.list(dirPath)) {

            files.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().startsWith("problem-"))
                    .filter(path -> path.getFileName().toString().endsWith(".txt"))
                    .forEach(problemPath -> {
                        String fileName = problemPath.getFileName().toString();
                        problemIds.add(fileName.replace("problem-", "").replace(".txt", ""));
                    });

        } catch (IOException e) {

            System.err.println("Error listing directory: " + dirPath);
            System.err.println(e.getMessage());
        }

        Collections.sort(problemIds);
        return problemIds;
    }

    /**
     * Loads ONE problem, null when its truth file is missing, unreadable or does not
     * match the sentences.
     */
    private static Problem loadProblem(Path dirPath, String problemId, String difficulty,
            FeatureConfig.Casing casing) {

        Path problemPath = dirPath.resolve("problem-" + problemId + ".txt");
        Path truthPath = dirPath.resolve("truth-problem-" + problemId + ".json");

        if (!Files.exists(truthPath)) {
            System.err.println("Truth file does not exist: " + truthPath);
            return null;
        }

        try {

            List<String> sentences = extractSentences(Files.readString(problemPath), casing);
            String truth = Files.readString(truthPath);
            int[] changes = extractChanges(truth);

            // Check that the number of boundaries matches.
            if (sentences.size() - 1 != changes.length) {
                System.err.println("Sentence/change mismatch " + "for problem " + problemId);
                return null;
            }

            return new Problem(problemId, difficulty, sentences, changes, extractAuthors(truth));

        } catch (IOException e) {

            System.err.println("Error reading problem: " + problemPath);
            System.err.println(e.getMessage());
            return null;
        }
    }

    private static void addPairs(Problem problem, List<SentencePair> dataset) {

        for (int i = 0; i < problem.sentences.size() - 1; i++) {
            dataset.add(new SentencePair(problem.sentences.get(i), problem.sentences.get(i + 1), problem.changes[i]));
        }
    }

    /**
     * Extracts sentences from a problem file (one sentence per line).
     *
     * @param problem problem text
     * @param casing  the casing policy, see FeatureConfig.Casing
     * @return list of sentences
     */
    public static List<String> extractSentences(String problem, FeatureConfig.Casing casing) {

        List<String> sentences = new ArrayList<>();

        String[] lines = problem.split("\\r\\n|\\r|\\n");

        for (String line : lines) {
            String sentence = line.trim();

            if (!sentence.isEmpty()) {
                sentences.add(TextUtils.applyCasing(sentence, casing));
            }
        }

        return sentences;
    }

    /**
     * Extracts the changes array from the truth JSON.
     *
     * @param truth truth JSON
     * @return change labels
     */
    public static int[] extractChanges(String truth) {

        int changesKeyIndex = truth.indexOf("\"changes\""); // search changes

        if (changesKeyIndex == -1) {
            throw new IllegalArgumentException("Could not find 'changes' key");
        }

        int start = truth.indexOf("[", changesKeyIndex);
        int end = truth.indexOf("]", start);

        if (start == -1 || end == -1) {
            throw new IllegalArgumentException("Could not find changes array");
        }

        String changesString = truth.substring(start + 1, end).trim();

        // a document of one sentence has no boundaries
        if (changesString.isEmpty()) {
            return new int[0];
        }

        String[] values = changesString.split(",");

        int[] changes = new int[values.length];

        for (int i = 0; i < values.length; i++) {
            changes[i] = Integer.parseInt(values[i].trim());
        }

        return changes;
    }

    /**
     * Reads the number of authors from the truth JSON.
     *
     * @param truth the content of a truth file, for example {@code {"authors": 4, "changes": [0, 1, 1]}}
     * @return the number of authors, 0 when the file has no usable 'authors' entry
     */
    public static int extractAuthors(String truth) {

        java.util.regex.Matcher matcher = AUTHORS_ENTRY.matcher(truth);
        return matcher.find() ? Integer.parseInt(matcher.group(1)) : 0;
    }
}
