package data;

/**
 * Interprets the ground-truth file of a document to determine authorship relationships between segments.
 *
 * <p>A segment is defined as a contiguous run of sentences without an author change. The ground-truth
 * data provides two absolute facts: where the author changes occur (the {@code changes} array) and
 * the total number of distinct authors who wrote the document (the {@code authors} count).
 * From these two facts, specific pairs of segments can be determined with certainty, while others remain ambiguous:</p>
 *
 * <ul>
 *   <li>A single segment is entirely written by one author.</li>
 *   <li>Neighbouring segments have different authors, as a change boundary explicitly lies between them.</li>
 *   <li>When the total number of authors equals the number of segments, every segment has a unique author.</li>
 *   <li>When there are exactly two authors, they must alternate: segments an even distance apart share an author,
 *       and segments an odd distance apart do not.</li>
 *   <li>With three or more authors and fewer authors than segments, an author can return after a gap
 *       (e.g., A B C A). In this scenario, no non-adjacent pairs can be determined with absolute certainty.</li>
 * </ul>
 *
 * <p>The retrieval model is exclusively trained using these guaranteed pairs; introducing speculative pairs
 * would introduce noisy or incorrect labels into the network.</p>
 */
public final class SegmentAuthorship {

    /**
     * Represents the logical relationship between two segments of a single document.
     */
    public enum Relation {
        /** The same author unequivocally wrote both segments. */
        SAME_AUTHOR,

        /** Different authors unequivocally wrote the segments. */
        DIFFERENT_AUTHOR,

        /** The ground-truth data does not provide enough information to settle the relationship. */
        UNKNOWN
    }

    private SegmentAuthorship() {
        // Utility class, not meant to be instantiated
    }

    /**
     * Decides how two segments of the same document relate, relying strictly on guaranteed truths from the dataset.
     *
     * @param firstSegment Position of the first segment in the document (0-indexed).
     * @param secondSegment Position of the second segment in the document (0-indexed).
     * @param segmentCount Total number of segments in the document.
     * @param authors Number of distinct authors provided by the truth file (0 if unknown).
     * @return The deterministic {@link Relation} between the two segments.
     */
    public static Relation between(int firstSegment, int secondSegment, int segmentCount, int authors) {

        if (firstSegment == secondSegment) {
            return Relation.SAME_AUTHOR;
        }

        int distance = Math.abs(firstSegment - secondSegment);
        if (distance == 1) {
            return Relation.DIFFERENT_AUTHOR;
        }

        if (authors <= 0 || authors > segmentCount) {
            return Relation.UNKNOWN;
        }

        if (authors == segmentCount) {
            return Relation.DIFFERENT_AUTHOR;
        }

        if (authors == 2) {
            return distance % 2 == 0 ? Relation.SAME_AUTHOR : Relation.DIFFERENT_AUTHOR;
        }

        return Relation.UNKNOWN;
    }
}