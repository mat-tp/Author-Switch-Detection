package analysis;

import evaluation.ThresholdSelector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Cuts a document into distinct runs of sentences at every position where the change score
 * exceeds the decision threshold, deriving the estimated number of authors directly from the segment count.
 *
 * <p>The estimate is deliberately simple: a document with {@code k} detected author changes has {@code k + 1}
 * segments, and that number is reported as the estimated author count. No clustering of the segments takes part.
 * Consequently, the estimate is an upper bound on the true count whenever an author returns later in the text
 * (e.g., a document with twelve segments written by four authors), and a lower bound whenever a real change
 * is missed.</p>
 *
 * <p>The segmentation itself is exact in the sense that every boundary is a thresholded change score of the model,
 * ensuring that the highlighted splits and the reported author count strictly agree.</p>
 */
public final class DocumentSegmentation {

    /**
     * A contiguous run of sentences without a detected author change.
     */
    public static final class Segment {

        /** Index of the first sentence of the segment. */
        public final int firstSentence;

        /** Index strictly one past the last sentence of the segment. */
        public final int endSentence;

        Segment(int firstSentence, int endSentence) {
            this.firstSentence = firstSentence;
            this.endSentence = endSentence;
        }

        /**
         * Calculates the total number of sentences contained within this segment.
         *
         * @return Sentence count for the segment.
         */
        public int length() {
            return endSentence - firstSentence;
        }
    }

    private final List<Segment> segments;
    private final int[] changeDecisions;

    private DocumentSegmentation(List<Segment> segments, int[] changeDecisions) {
        this.segments = Collections.unmodifiableList(segments);
        this.changeDecisions = changeDecisions;
    }

    /**
     * Segments a document based on the stylistic change scores of its sentence boundaries.
     *
     * @param changeScores Array of scores per boundary between neighbouring sentences (length = {@code sentences - 1}).
     * @param threshold Decision boundary; a change is predicted where the score is strictly greater than this value.
     * @return Fully initialized {@link DocumentSegmentation} instance mapping the document splits.
     */
    public static DocumentSegmentation fromScores(double[] changeScores, double threshold) {
        int[] decisions = ThresholdSelector.apply(changeScores, threshold);
        List<Segment> segments = new ArrayList<>();
        int first = 0;
        for (int boundary = 0; boundary < decisions.length; boundary++) {
            if (decisions[boundary] == 1) {
                segments.add(new Segment(first, boundary + 1));
                first = boundary + 1;
            }
        }
        segments.add(new Segment(first, decisions.length + 1));
        return new DocumentSegmentation(segments, decisions);
    }

    /**
     * Retrieves the sequential list of document segments.
     *
     * @return Unmodifiable list of {@link Segment} objects in reading order.
     */
    public List<Segment> segments() {
        return segments;
    }

    /**
     * Retrieves the binary boundary decisions used to generate the segments.
     *
     * @return Integer array containing {@code 1} per boundary where a change is predicted, and {@code 0} elsewhere.
     */
    public int[] changeDecisions() {
        return changeDecisions.clone();
    }

    /**
     * Calculates the total number of predicted author transition boundaries.
     *
     * @return Integer count of predicted author changes.
     */
    public int changeCount() {
        return segments.size() - 1;
    }

    /**
     * Estimates the total number of distinct authors based on the generated segmentation.
     *
     * @return The segment count, acting as an upper-bound estimate of the true author count.
     */
    public int estimatedAuthors() {
        return segments.size();
    }

    /**
     * Maps the specific sentence indices belonging to each segment to support frontend visualization views.
     *
     * @return A list containing a list of sentence indices for every segment in chronological order.
     */
    public List<List<Integer>> sentenceIndicesPerSegment() {
        List<List<Integer>> result = new ArrayList<>();
        for (Segment segment : segments) {
            List<Integer> indices = new ArrayList<>(segment.length());
            for (int i = segment.firstSentence; i < segment.endSentence; i++) {
                indices.add(i);
            }
            result.add(indices);
        }
        return result;
    }
}