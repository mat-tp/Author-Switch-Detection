package data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import features.FeatureExtractor;

/**
 * Turns lists of sentence pairs into feature matrices.
 *
 * The one place where the data pipeline lives, so the command line (Main), the ablation study
 * (Ablation), the retrieval model and the web app (RealTrainer) build their matrices in
 * EXACTLY the same way. The helpers used to sit in Main.java; they are public static here and
 * Main only calls them.
 *
 * The extraction is parallel: the pairs are cut into chunks of CHUNK pairs, each chunk is
 * calculated sequentially by one thread (the second sentence of a pair is the first sentence of the
 * next pair, so its features are used twice instead of calculated twice), and the result does not
 * depend on the number of threads.
 */
public class PairMatrixBuilder {

    private static final int CHUNK = 256;

    // How many training pairs are recomputed the slow (original) way at the end
    // of buildTrainingMatrix, to check the memory-friendly way gives the same numbers
    private static final int VERIFY_SAMPLE = 200;

    /**
     * Feature matrix of the TRAINING pairs. Fits the z-score parameters of the extractor on the
     * training sentences (and, for the full features, of the bucket differences), so this has to be
     * done BEFORE any validation or test matrix is built.
     *
     * The full feature vectors of all the sentences (hundreds to thousands of numbers each) are never
     * kept: only the scalars of each sentence, and the pair vector.
     */
    public static double[][] buildTrainingMatrix(List<SentencePair> pairs, FeatureExtractor extractor) {

        extractor.prepare();

        int scalarCount = FeatureExtractor.SCALAR_COUNT;
        int n = pairs.size();
        int width = extractor.pairVectorLength();

        double[][] featureMatrix = new double[n][];
        double[][] scalars1 = new double[n][];
        double[][] scalars2 = new double[n][];

        AtomicInteger processed = new AtomicInteger();
        int nChunks = (n + CHUNK - 1) / CHUNK;

        IntStream.range(0, nChunks).parallel().forEach(chunk -> {

            int from = chunk * CHUNK;
            int to = Math.min(from + CHUNK, n);

            String previousSentence = null;
            double[] previousFeatures = null;

            for (int i = from; i < to; i++) {

                SentencePair pair = pairs.get(i);

                double[] features1 = pair.getSentence1().equals(previousSentence) ? previousFeatures
                        : extractor.getFeatures(pair.getSentence1());
                double[] features2 = extractor.getFeatures(pair.getSentence2());

                // the same array when the sentence was already there, it is a second use of the same numbers
                scalars1[i] = features1 == previousFeatures && i > from ? scalars2[i - 1]
                        : java.util.Arrays.copyOf(features1, scalarCount);
                scalars2[i] = java.util.Arrays.copyOf(features2, scalarCount);

                // The profile part of a pair vector does not use the z-scores. The scalar columns
                // and the overall distance are filled in below, once the z-score parameters are fitted.
                featureMatrix[i] = new double[width];
                extractor.writeProfilePart(features1, features2, featureMatrix[i]);

                previousSentence = pair.getSentence2();
                previousFeatures = features2;

                int count = processed.incrementAndGet();
                if (count % 10000 == 0 || count == n) {
                    System.out.println("  Processed " + count + " / " + n);
                }
            }
        });

        // The z-score parameters are fitted on the TRAINING sentences only (each distinct array once).
        List<double[]> trainingScalars = new ArrayList<>(n + 1);
        for (int i = 0; i < n; i++) {
            if (i == 0 || scalars1[i] != scalars2[i - 1]) {
                trainingScalars.add(scalars1[i]);
            }
            trainingScalars.add(scalars2[i]);
        }
        extractor.fitScaler(trainingScalars);
        trainingScalars = null;

        IntStream.range(0, n).parallel().forEach(i -> {

            double[] z1 = extractor.standardize(scalars1[i]);
            double[] z2 = extractor.standardize(scalars2[i]);

            extractor.writeScalarPart(scalars1[i], scalars2[i], z1, z2, featureMatrix[i]);
        });

        // The bucket differences are standardised too (fitted on the TRAINING pairs only)
        if (extractor.getConfig().fullBlockDifferences) {

            extractor.fitBlockDifferenceScaler(featureMatrix);

            IntStream.range(0, n).parallel().forEach(i -> {
                featureMatrix[i] = extractor.standardizeBlockDifferences(featureMatrix[i]);
            });
        }

        // verifying features exists
        for (int i = 0; i < featureMatrix.length; i++) {

            if (featureMatrix[i] == null) {
                throw new IllegalStateException("Feature vector at index " + i + " is null.");
            }
        }

        // The first pairs are done again the slow (original) way, they have to be identical
        int verified = Math.min(VERIFY_SAMPLE, n);
        for (int i = 0; i < verified; i++) {

            double[] expected = getPairFeatures(pairs.get(i), extractor);

            if (!java.util.Arrays.equals(expected, featureMatrix[i])) {
                throw new IllegalStateException("The memory-friendly feature matrix differs from "
                        + "getPairFeatures at pair " + i + ". Was SentencePair changed? Fix buildTrainingMatrix.");
            }
        }
        System.out.println("  Checked " + verified + " pairs against the original way: identical.");

        return featureMatrix;
    }

    /**
     * Builds the feature matrix of pairs, using the z-score parameters of the extractor (fitted on the
     * training data).
     */
    public static double[][] buildFeatureMatrix(List<SentencePair> pairs, FeatureExtractor extractor) {

        extractor.prepare();

        int n = pairs.size();
        double[][] matrix = new double[n][];
        int nChunks = (n + CHUNK - 1) / CHUNK;

        IntStream.range(0, nChunks).parallel().forEach(chunk -> {

            int from = chunk * CHUNK;
            int to = Math.min(from + CHUNK, n);

            String previousSentence = null;
            double[] previousFeatures = null;

            for (int i = from; i < to; i++) {

                SentencePair pair = pairs.get(i);

                double[] features1 = pair.getSentence1().equals(previousSentence) ? previousFeatures
                        : extractor.getFeatures(pair.getSentence1());
                double[] features2 = extractor.getFeatures(pair.getSentence2());

                matrix[i] = extractor.pairVector(features1, features2);

                previousSentence = pair.getSentence2();
                previousFeatures = features2;
            }
        });

        return matrix;
    }

    /**
     * Pairwise representation of one pair, using the (LOADED or fitted) extractor.
     */
    public static double[] getPairFeatures(SentencePair pair, FeatureExtractor extractor) {

        double[] features1 = extractor.getFeatures(pair.getSentence1());
        double[] features2 = extractor.getFeatures(pair.getSentence2());

        return pair.getPairFeatures(extractor, features1, features2);
    }

    /**
     * The raw feature vectors (FeatureExtractor.getFeatures) of a list of sentences.
     * (Used by the retrieval model, which embeds single sentences, not pairs.)
     */
    public static double[][] buildSentenceMatrix(List<String> sentences, FeatureExtractor extractor) {

        extractor.prepare();

        double[][] matrix = new double[sentences.size()][];
        IntStream.range(0, matrix.length).parallel().forEach(i -> matrix[i] = extractor.getFeatures(sentences.get(i)));

        return matrix;
    }

    public static int[] getTruthValues(List<SentencePair> pairs) {

        int[] truthValues = new int[pairs.size()];

        for (int i = 0; i < pairs.size(); i++) {
            truthValues[i] = pairs.get(i).getLabel();
        }

        return truthValues;
    }

    /**
     * Takes at most limit number of pairs. The pairs are shuffled with a fixed seed first,
     * because the first pairs of the list come from only a few documents and are
     * not a good picture of the whole split. The chosen pairs are COPIED into a
     * new list, so the pairs that were not chosen (and the shuffled list) can be
     * garbage collected instead of staying alive behind a subList view.
     */
    public static List<SentencePair> takeSubset(List<SentencePair> pairs, int limit) {

        if (pairs.size() <= limit) {
            return pairs;
        }

        List<SentencePair> shuffled = new ArrayList<>(pairs);
        Collections.shuffle(shuffled, new Random(42L));

        return new ArrayList<>(shuffled.subList(0, limit));
    }

    /**
     * Which fraction (0 to 1) of the wanted training / validation / test pairs the heap can hold as
     * feature matrices. 1.0 = everything fits.
     *
     * @param nTrain           training pairs wanted
     * @param nEvaluation      validation + test pairs wanted
     * @param pairFeatureCount numbers in one pair's feature vector
     * @param heapFraction     the part of the heap the matrices may use
     */
    public static double fractionThatFitsInMemory(long nTrain, long nEvaluation, int pairFeatureCount,
            double heapFraction) {

        long rowBytes = 8L * pairFeatureCount + 16; // the doubles + the array header

        // while the training matrix is built each pair also keeps the raw scalars of
        // its sentences (see buildTrainingMatrix), and its label
        long trainBytes = rowBytes + 2 * (8L * FeatureExtractor.SCALAR_COUNT + 16) + 4;

        long wanted = nTrain * trainBytes + nEvaluation * rowBytes;
        long budget = (long) (Runtime.getRuntime().maxMemory() * heapFraction);

        System.out.println();
        System.out.println(String.format("Memory: max heap %d MB, the feature matrices may use %d MB.",
                Runtime.getRuntime().maxMemory() / (1024 * 1024), budget / (1024 * 1024)));
        System.out.println(String.format("        All the wanted data needs about %d MB (run with -Xmx%dm to use it all).",
                wanted / (1024 * 1024), (long) Math.ceil(wanted / heapFraction / (1024 * 1024))));

        if (wanted <= budget) {
            return 1.0;
        }

        double fraction = (double) budget / wanted;
        System.out.println(String.format("        That does not fit: using %.1f%% of the training, validation and test "
                + "pairs, so the run does not run out of memory.", fraction * 100));

        return fraction;
    }

    /** The rows of all the matrices in one array (the rows are shared, not copied). */
    public static double[][] joinMatrices(Collection<double[][]> matrices) {

        int total = 0;
        for (double[][] matrix : matrices) {
            total += matrix.length;
        }

        double[][] joined = new double[total][];
        int next = 0;
        for (double[][] matrix : matrices) {
            System.arraycopy(matrix, 0, joined, next, matrix.length);
            next += matrix.length;
        }

        return joined;
    }

    public static int[] joinTruth(Collection<int[]> truths) {

        int total = 0;
        for (int[] truth : truths) {
            total += truth.length;
        }

        int[] joined = new int[total];
        int next = 0;
        for (int[] truth : truths) {
            System.arraycopy(truth, 0, joined, next, truth.length);
            next += truth.length;
        }

        return joined;
    }
}
