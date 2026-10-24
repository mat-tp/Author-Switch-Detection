package util;

/**
 * Scores the boundaries between neighbouring sentences from their embeddings.
 *
 * <p>The score of the boundary between sentence i and sentence i + 1 compares the sentences just before
 * the boundary with the sentences just after it. Both sides are summed over a window of sentences, cut at
 * the borders of the document, and the score is the cosine distance of the two sums scaled to the range
 * 0 (same style) to 1 (opposite style), so that it can be thresholded like a probability.</p>
 *
 * <p>The retrieval model and the evaluator share this one implementation.</p>
 */
public final class BoundaryScores {

    private BoundaryScores() {
        // utility class, not meant to be instantiated
    }

    /**
     * Scores every boundary of one document.
     *
     * @param embeddings one embedding per sentence, in the order of the text
     * @param first      index of the first sentence of the document
     * @param end        index after the last sentence of the document
     * @param window     how many sentences on each side of a boundary are summed
     * @return one score per boundary, {@code end - first - 1} numbers (none for a single sentence)
     */
    public static double[] forDocument(double[][] embeddings, int first, int end, int window) {

        int boundaries = Math.max(0, end - first - 1);
        double[] scores = new double[boundaries];
        if (boundaries == 0) {
            return scores;
        }
        int width = embeddings[first].length;

        for (int boundary = 0; boundary < boundaries; boundary++) {
            int sentence = first + boundary;
            double[] before = VectorMath.sumRows(embeddings, Math.max(first, sentence - window + 1), sentence, width);
            double[] after = VectorMath.sumRows(embeddings, sentence + 1, Math.min(end - 1, sentence + window), width);
            scores[boundary] = VectorMath.scaledCosineDistance(before, after);
        }
        return scores;
    }
}
