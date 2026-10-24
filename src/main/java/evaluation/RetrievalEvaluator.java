package evaluation;

import analysis.DocumentSegmentation;
import util.BoundaryScores;
import util.VectorMath;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import data.RetrievalData;

/**
 * Evaluates the performance of the retrieval model using generated embeddings (or baseline raw/random feature vectors).
 *
 * <p>This benchmarking suite evaluates the model across three primary dimensions:</p>
 * <ul>
 *   <li><strong>In-Document Retrieval:</strong> Treats every sentence as a query and ranks all other sentences in the same document by cosine similarity.
 *       Matches are deemed relevant only if they reside in the query's own segment (guaranteed same author).
 *       Calculates conservative Information Retrieval (IR) metrics: Precision@1, Recall@5, Mean Reciprocal Rank (MRR), and Mean Average Precision (MAP).
 *       (Note: Sentences from non-adjacent segments by the same returning author are conservatively treated as non-relevant because the truth file does not definitively link them without clustering).</li>
 *   <li><strong>Triplet Accuracy:</strong> Verifies the Triplet Loss training objective by measuring how often a sentence is stylistically closer to sentences in its own segment (positive mates) than to sentences in an adjacent segment (negatives). A score of 0.5 represents chance.</li>
 *   <li><strong>Switch Detection & Author Counting:</strong> Evaluates author boundary detection by calculating the Area Under the ROC Curve (AUC) for boundary change scores, providing a threshold-free measure of the model's ability to identify stylistic shifts.</li>
 * </ul>
 */
public class RetrievalEvaluator {

    private RetrievalEvaluator() {
        // utility class, not meant to be instantiated
    }

    /**
     * Container for Information Retrieval and Triplet Accuracy evaluation results.
     */
    public static class Report {

        public int queries;
        public double precisionAt1, recallAt5, mrr, map;
        public double tripletAccuracy;
        public int triplets;

        @Override
        public String toString() {
            return String.format(java.util.Locale.ROOT,
                    "P@1 %.4f | Recall@5 %.4f | MRR %.4f | MAP %.4f | triplet accuracy %.4f (%d queries, %d triplets)",
                    precisionAt1, recallAt5, mrr, map, tripletAccuracy, queries, triplets);
        }
    }

    /**
     * Standardizes raw feature vectors by dividing each vector by its L2 norm, serving as a baseline measure.
     *
     * @param x Matrix containing one feature row per sentence.
     * @return Matrix of standardized unit vectors.
     */
    public static double[][] normalise(float[][] x) {

        double[][] result = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            result[i] = VectorMath.unitVector(x[i]);
        }
        return result;
    }

    /**
     * Generates random unit vectors to establish the chance-level baseline for all retrieval measures.
     *
     * @param count Number of random vectors to generate.
     * @param size  Dimensionality length of each vector.
     * @param seed  Initialization seed for the random number generator.
     * @return Matrix of random unit vectors.
     */
    public static double[][] randomVectors(int count, int size, long seed) {

        Random random = new Random(seed);
        double[][] result = new double[count][size];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = random.nextGaussian();
            }
            VectorMath.normaliseInPlace(result[i]);
        }
        return result;
    }

    /**
     * Evaluates in-document sentence retrieval and local triplet accuracy.
     *
     * @param data       The dataset containing ground-truth documents and segments.
     * @param vectors    One unit vector per sentence in the dataset (network embeddings or baselines).
     * @param difficulty Optional difficulty filter; if specified, only evaluates documents of this tier (null for all).
     * @return A populated {@link Report} containing MAP, MRR, P@1, Recall@5, and triplet accuracy metrics.
     */
    public static Report evaluate(RetrievalData data, double[][] vectors, String difficulty) {

        Report report = new Report();
        double p1 = 0, r5 = 0, rr = 0, ap = 0;
        int queries = 0;

        double tripletsRight = 0;
        int triplets = 0;

        for (int d = 0; d < data.documents(); d++) {

            if (difficulty != null && !difficulty.equals(data.difficulty[d])) {
                continue;
            }

            int from = data.docStart[d], to = data.docStart[d + 1];
            int n = to - from;
            Integer[] order = new Integer[n];
            double[] similarity = new double[n];

            for (int q = from; q < to; q++) {

                int segment = data.segment[q];
                int relevant = data.segmentLength(segment) - 1;
                if (relevant < 1) {
                    continue; // Skip queries lacking relevant intra-segment mates
                }

                for (int j = 0; j < n; j++) {
                    order[j] = j;
                    similarity[j] = (from + j == q) ? Double.NEGATIVE_INFINITY : VectorMath.dot(vectors[q], vectors[from + j]);
                }
                final double[] sim = similarity;
                java.util.Arrays.sort(order, (x, y) -> Double.compare(sim[y], sim[x]));

                // The query itself is ranked last (-infinity), leaving a ranking of n - 1 entries
                int found = 0;
                double precisionSum = 0.0;
                double firstRank = 0.0;
                for (int rank = 1; rank <= n - 1; rank++) {

                    int sentence = from + order[rank - 1];
                    if (data.segment[sentence] == segment) {
                        found++;
                        precisionSum += (double) found / rank;
                        if (firstRank == 0.0) {
                            firstRank = rank;
                        }
                        if (rank == 1) {
                            p1++;
                        }
                    }
                    if (rank == 5) {
                        r5 += (double) found / relevant;
                    }
                }
                if (n - 1 < 5) {
                    r5 += (double) found / relevant;
                }
                rr += firstRank == 0.0 ? 0.0 : 1.0 / firstRank;
                ap += precisionSum / relevant;
                queries++;

                // Triplet accuracy: mean similarity to the own segment vs mean similarity to adjacent segments
                int before = segment - 1, after = segment + 1;
                boolean hasBefore = before >= 0 && data.segDoc[before] == d;
                boolean hasAfter = after < data.segments() && data.segDoc[after] == d;
                for (int neighbour : new int[] { before, after }) {
                    if ((neighbour == before && !hasBefore) || (neighbour == after && !hasAfter)) {
                        continue;
                    }
                    double sumMate = 0.0;
                    int mates = 0;
                    for (int s = data.segStart[segment]; s < data.segStart[segment + 1]; s++) {
                        if (s != q) {
                            sumMate += VectorMath.dot(vectors[q], vectors[s]);
                            mates++;
                        }
                    }
                    double sumOther = 0.0;
                    int others = 0;
                    for (int s = data.segStart[neighbour]; s < data.segStart[neighbour + 1]; s++) {
                        sumOther += VectorMath.dot(vectors[q], vectors[s]);
                        others++;
                    }

                    if (mates > 0 && others > 0) {
                        tripletsRight += (sumMate / mates > sumOther / others) ? 1.0 : 0.0;
                        triplets++;
                    }
                }
            }
        }

        report.queries = queries;
        if (queries > 0) {
            report.precisionAt1 = p1 / queries;
            report.recallAt5 = r5 / queries;
            report.mrr = rr / queries;
            report.map = ap / queries;
        }
        report.triplets = triplets;
        report.tripletAccuracy = triplets == 0 ? 0.0 : tripletsRight / triplets;
        return report;
    }

    /**
     * Calculates windowed change scores between adjacent sentences across all requested documents.
     *
     * <p>The score represents "a change between sentence i and i + 1", computed as:
     * {@code (1 - cosine(mean_embedding_left, mean_embedding_right)) / 2}.
     * This produces a value in [0, 1] mirroring a probability score.</p>
     *
     * @param data       The dataset containing ground-truth documents.
     * @param vectors    The generated sentence vectors.
     * @param window     Number of sentences left and right of the boundary to average.
     * @param difficulty Optional difficulty filter; if specified, only evaluates documents of this tier.
     * @return Array of sequential boundary change scores spanning all eligible document boundaries.
     */
    public static double[] switchScores(RetrievalData data, double[][] vectors, int window, String difficulty) {

        List<Double> scores = new ArrayList<>();

        for (int d = 0; d < data.documents(); d++) {

            if (difficulty != null && !difficulty.equals(data.difficulty[d])) {
                continue;
            }
            for (double score : BoundaryScores.forDocument(vectors, data.docStart[d], data.docStart[d + 1], window)) {
                scores.add(score);
            }
        }

        double[] result = new double[scores.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = scores.get(i);
        }
        return result;
    }

    /**
     * Compares the model's estimated author count for each document against the dataset ground truth.
     *
     * @param data       The dataset containing ground-truth author counts.
     * @param vectors    The generated sentence vectors.
     * @param window     Number of sentences left and right of the boundary to evaluate.
     * @param threshold  The decision threshold for defining an author switch boundary.
     * @return An {@link AuthorCountEvaluation} tracking exact matches and mean errors. Documents lacking a known author count are skipped.
     */
    public static AuthorCountEvaluation authorCount(RetrievalData data, double[][] vectors, int window,
                                                    double threshold) {

        AuthorCountEvaluation evaluation = new AuthorCountEvaluation();
        for (int d = 0; d < data.documents(); d++) {
            double[] scores = BoundaryScores.forDocument(vectors, data.docStart[d], data.docStart[d + 1], window);
            evaluation.add(DocumentSegmentation.fromScores(scores, threshold).estimatedAuthors(), data.authors[d]);
        }
        return evaluation;
    }

    /**
     * Aggregates the ground-truth change labels across documents to align chronologically with {@link #switchScores}.
     *
     * @param data       The dataset containing ground-truth change labels.
     * @param difficulty Optional difficulty filter; if specified, only returns labels for documents of this tier.
     * @return Array of binary integers (1 for a true author switch, 0 otherwise).
     */
    public static int[] switchTruth(RetrievalData data, String difficulty) {

        List<Integer> labels = new ArrayList<>();
        for (int d = 0; d < data.documents(); d++) {
            if (difficulty != null && !difficulty.equals(data.difficulty[d])) {
                continue;
            }
            for (int c : data.changes[d]) {
                labels.add(c);
            }
        }
        int[] result = new int[labels.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = labels.get(i);
        }
        return result;
    }

    /**
     * Computes the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) for switch detection.
     *
     * <p>Represents the statistical probability that a randomly chosen true author change scores strictly
     * higher than a randomly chosen continuation boundary (ties count as half).</p>
     *
     * @param scores Array of predicted change boundary scores.
     * @param truth  Array of ground-truth boundary labels (1 = switch, 0 = same author).
     * @return AUC double value where 1.0 is perfect prediction and 0.5 is random chance.
     */
    public static double auc(double[] scores, int[] truth) {

        int n = scores.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));

        double positives = 0, negatives = 0, rankSum = 0;
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && scores[order[j + 1]] == scores[order[i]]) {
                j++;
            }
            double averageRank = (i + j) / 2.0 + 1.0;
            for (int k = i; k <= j; k++) {
                if (truth[order[k]] == 1) {
                    positives++;
                    rankSum += averageRank;
                } else {
                    negatives++;
                }
            }
            i = j + 1;
        }
        if (positives == 0 || negatives == 0) {
            return 0.5;
        }
        return (rankSum - positives * (positives + 1) / 2.0) / (positives * negatives);
    }
}