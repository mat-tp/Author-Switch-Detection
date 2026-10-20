package models;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import data.RetrievalData;

/**
 * The measures of the retrieval model, on embeddings (or on any vectors: the baselines use the
 * raw feature vectors and random vectors, to show what the learned embedding adds).
 *
 * 1. Retrieval inside a document. Every sentence is a query, the other sentences of the SAME document
 * are ranked by cosine similarity, the relevant ones are those of the query's own segment (same author
 * for sure). Reported: P@1 (is the best match in the own segment), Recall@5, MRR and MAP. These
 * numbers are conservative: a sentence of another segment may have the same author too (a document
 * with 2 authors alternates), but the labels do not say so, so it counts as not relevant.
 * Queries that have no other sentence in their segment are skipped.
 *
 * 2. Triplet accuracy: for a sentence with a segment mate and a neighbour segment, how often the mate is
 * closer than the sentence of the neighbour segment (the exact thing the network was trained for, 0.5 = chance).
 *
 * 3. Switch detection: the distance between the sentences left of a position and the sentences right
 * of it (window = how many) is the score of "a change here". Measured by the area under the ROC curve
 * (threshold free) and, with a threshold, by the macro-F1 that PAN uses.
 */
public class RetrievalEvaluator {

    private RetrievalEvaluator() {
        // utility class, not meant to be instantiated
    }

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

    // ------------------------------------------------------------------
    // Vectors
    // ------------------------------------------------------------------

    /** The same vectors, each divided by its length (the baseline "cosine of the raw features"). */
    public static double[][] normalise(float[][] x) {

        double[][] result = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            double sum = 0.0;
            for (float v : x[i]) {
                sum += (double) v * v;
            }
            double norm = Math.sqrt(sum) + 1e-12;
            result[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                result[i][j] = x[i][j] / norm;
            }
        }
        return result;
    }

    /** Random unit vectors: the chance level of every measure. */
    public static double[][] randomVectors(int count, int size, long seed) {

        Random random = new Random(seed);
        double[][] result = new double[count][size];
        for (int i = 0; i < count; i++) {
            double sum = 0.0;
            for (int j = 0; j < size; j++) {
                result[i][j] = random.nextGaussian();
                sum += result[i][j] * result[i][j];
            }
            double norm = Math.sqrt(sum);
            for (int j = 0; j < size; j++) {
                result[i][j] /= norm;
            }
        }
        return result;
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // ------------------------------------------------------------------
    // 1. + 2. Retrieval inside documents and triplet accuracy
    // ------------------------------------------------------------------

    /**
     * @param data       the documents
     * @param vectors    one unit vector per sentence of data (embeddings or baselines)
     * @param difficulty only the documents of this difficulty, null = all
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
                    continue;
                }

                for (int j = 0; j < n; j++) {
                    order[j] = j;
                    similarity[j] = (from + j == q) ? Double.NEGATIVE_INFINITY : dot(vectors[q], vectors[from + j]);
                }
                final double[] sim = similarity;
                java.util.Arrays.sort(order, (x, y) -> Double.compare(sim[y], sim[x]));

                // the query itself is last (-infinity), the ranking has n - 1 entries
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

                // triplet accuracy: the mean similarity to the own segment (without the query) against the mean
                // similarity to the segment before / after (all sentences, not a sample: exact and repeatable)
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
                            sumMate += dot(vectors[q], vectors[s]);
                            mates++;
                        }
                    }
                    double sumOther = 0.0;
                    int others = 0;
                    for (int s = data.segStart[neighbour]; s < data.segStart[neighbour + 1]; s++) {
                        sumOther += dot(vectors[q], vectors[s]);
                        others++;
                    }
                    // the mean similarity to the own segment against the mean similarity to the neighbour segment
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

    // ------------------------------------------------------------------
    // 3. Switch detection
    // ------------------------------------------------------------------

    /**
     * The score of "a change between sentence i and i + 1" for every pair of every document (documents one after
     * the other, the same order as the labels in changes[][]): 1 - cosine of the mean embedding of the
     * `window` sentences up to i and the mean of the `window` sentences from i + 1 (cut at the document borders),
     * divided by 2 so it lies in [0, 1] and can be used like a probability.
     */
    public static double[] switchScores(RetrievalData data, double[][] vectors, int window, String difficulty) {

        List<Double> scores = new ArrayList<>();

        for (int d = 0; d < data.documents(); d++) {

            if (difficulty != null && !difficulty.equals(data.difficulty[d])) {
                continue;
            }
            int from = data.docStart[d], to = data.docStart[d + 1];
            int size = vectors[from].length;

            for (int i = from; i + 1 < to; i++) {

                double[] left = new double[size], right = new double[size];
                for (int k = Math.max(from, i - window + 1); k <= i; k++) {
                    for (int j = 0; j < size; j++) {
                        left[j] += vectors[k][j];
                    }
                }
                for (int k = i + 1; k <= Math.min(to - 1, i + window); k++) {
                    for (int j = 0; j < size; j++) {
                        right[j] += vectors[k][j];
                    }
                }
                double cosine = dot(left, right) / (Math.sqrt(dot(left, left) * dot(right, right)) + 1e-12);
                scores.add((1.0 - cosine) / 2.0);
            }
        }

        double[] result = new double[scores.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = scores.get(i);
        }
        return result;
    }

    /** The change labels of the documents, the same order as switchScores. */
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

    /** Area under the ROC curve = the chance that a change scores higher than a no-change (ties count half). */
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
