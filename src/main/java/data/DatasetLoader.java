package data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DatasetLoader {

    // fixed seed, so the validation / testing halves are the same on every run
    private static final long SPLIT_SEED = 42L;

    /**
     * Loads all sentence pairs from a specified split and difficulty.
     *
     * @param basePath   base dataset directory
     * @param splitType  train or validation
     * @param difficulty easy, medium, or hard
     * @return list of sentence pairs
     */
    public static List<SentencePair> loadSplit(String basePath, String splitType, String difficulty) {

        List<SentencePair> dataset = new ArrayList<>();
        Path dirPath = Paths.get(basePath, difficulty, splitType);
        if (!Files.exists(dirPath)) {
            System.err.println("Directory does not exist: " + dirPath);
            return dataset;
        }

        try {

            Files.list(dirPath).filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().startsWith("problem-"))
                    .filter(path -> path.getFileName().toString().endsWith(".txt"))
                    .forEach(problemPath -> {

                        try {

                            // getting problem ID.
                            String fileName = problemPath.getFileName().toString();
                            String problemId = fileName.replace("problem-", "").replace(".txt", "");

                            // finding it's matching truth value
                            Path truthPath = dirPath.resolve("truth-problem-" + problemId + ".json");

                            if (!Files.exists(truthPath)) {
                                System.err.println("Truth file does not exist: " + truthPath);
                                return;
                            }

                            String problem = Files.readString(problemPath); // reading a sentences
                            String truth = Files.readString(truthPath); // reading a sentences truth values

                            List<String> sentences = extractSentences(problem);
                            int[] changes = extractChanges(truth);

                            // Check that the number of boundaries matches.
                            if (sentences.size() - 1 != changes.length) {
                                System.err.println("Sentence/change mismatch " + "for problem " + problemId);
                                return;
                            }

                            // creating pairs
                            for (int i = 0; i < sentences.size() - 1; i++) {
                                SentencePair pair = new SentencePair(sentences.get(i), sentences.get(i + 1),
                                        changes[i]);
                                dataset.add(pair);
                            }

                        } catch (IOException e) {

                            System.err.println("Error reading problem: " + problemPath);
                            System.err.println(e.getMessage());
                        }
                    });

        } catch (IOException e) {

            System.err.println("Error listing directory: " + dirPath);
            System.err.println(e.getMessage());
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

        List<String> sentences = new ArrayList<>();
        Path dirPath = Paths.get(basePath, difficulty, splitType);

        if (!Files.exists(dirPath)) {
            System.err.println("Directory does not exist: " + dirPath);
            return sentences;
        }

        try {
            Files.list(dirPath).filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().startsWith("problem-"))
                    .filter(path -> path.getFileName().toString().endsWith(".txt"))
                    .forEach(problemPath -> {

                        try {
                            String problem = Files.readString(problemPath);
                            List<String> problemSentences = extractSentences(problem);
                            sentences.addAll(problemSentences);
                        } catch (IOException e) {
                            System.err.println("Error reading problem " + problemPath);
                            System.err.println(e.getMessage());
                        }
                    });
        } catch (IOException e) {
            System.err.println("Error listing directory: " + dirPath);
            System.err.println(e.getMessage());
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

        // difining the difficulties instead.
        String[] difficulties = { "easy", "medium", "hard" };

        List<String> allSentences = new ArrayList<>();

        for (String difficulty : difficulties) {

            List<String> sentences = getAllSentences(basePath, splitType, difficulty);
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

        List<SentencePair> dataset = new ArrayList<>();
        Path dirPath = Paths.get(basePath, difficulty, splitType);

        if (!Files.exists(dirPath)) {
            System.err.println("Directory does not exist: " + dirPath);
            return dataset;
        }

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
            return dataset;
        }

        // sorting first, because the order of Files.list is not guaranteed
        Collections.sort(problemIds);
        Collections.shuffle(problemIds, new Random(SPLIT_SEED));

        int halfSize = (problemIds.size() + 1) / 2;
        List<String> chosenIds;

        if (firstHalf) {
            chosenIds = problemIds.subList(0, halfSize);
        } else {
            chosenIds = problemIds.subList(halfSize, problemIds.size());
        }

        for (String problemId : chosenIds) {
            loadProblemPairs(dirPath, problemId, dataset);
        }

        return dataset;
    }

    /**
     * Loads the pairs of ONE problem and adds them to the dataset.
     *
     * @param dirPath   directory of the split
     * @param problemId problem ID (the number in problem-ID.txt)
     * @param dataset   list that the pairs are added to
     */
    private static void loadProblemPairs(Path dirPath, String problemId, List<SentencePair> dataset) {

        Path problemPath = dirPath.resolve("problem-" + problemId + ".txt");
        Path truthPath = dirPath.resolve("truth-problem-" + problemId + ".json");

        if (!Files.exists(truthPath)) {
            System.err.println("Truth file does not exist: " + truthPath);
            return;
        }

        try {

            List<String> sentences = extractSentences(Files.readString(problemPath));
            int[] changes = extractChanges(Files.readString(truthPath));

            // Check that the number of boundaries matches.
            if (sentences.size() - 1 != changes.length) {
                System.err.println("Sentence/change mismatch " + "for problem " + problemId);
                return;
            }

            for (int i = 0; i < sentences.size() - 1; i++) {
                dataset.add(new SentencePair(sentences.get(i), sentences.get(i + 1), changes[i]));
            }

        } catch (IOException e) {

            System.err.println("Error reading problem: " + problemPath);
            System.err.println(e.getMessage());
        }
    }

    /**
     * Extracts sentences from a problem file.
     *
     * @param problem problem text
     * @return list of sentences
     */
    private static List<String> extractSentences(String problem) {

        List<String> sentences = new ArrayList<>();

        String[] lines = problem.split("\\r\\n|\\r|\\n");

        for (String line : lines) {
            String sentence = line.trim();

            if (!sentence.isEmpty()) {
                sentences.add(sentence);
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

        String changesString = truth.substring(start + 1, end);
        String[] values = changesString.split(",");

        int[] changes = new int[values.length];

        for (int i = 0; i < values.length; i++) {
            changes[i] = Integer.parseInt(values[i].trim());
        }

        return changes;
    }
}