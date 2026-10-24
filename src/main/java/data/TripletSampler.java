package data;

import java.util.Random;

import data.SegmentAuthorship.Relation;

/**
 * Draws the training triplets of the retrieval model, consisting of an anchor sentence, a positive sentence
 * written by the same author, and several candidate negatives written by other authors. The training step
 * keeps the candidate that is closest to the anchor, which serves as the hard negative.
 *
 * <p>Two distinct policies dictate which sentences qualify as positive or negative matches. The original
 * adjacent segments rule dictates that positives originate from the anchor's own segment and negatives from
 * an immediately neighbouring segment. Alternatively, the author counts rule utilises the truth file data via
 * {@link SegmentAuthorship} to source negatives from any segment definitively written by another author, and
 * positives from any segment guaranteed to share the same author. Pairs that cannot be definitively resolved
 * by the truth file are strictly ignored.</p>
 */
public final class TripletSampler {

    /**
     * Defines the configuration parameters for drawing training triplets.
     *
     * @param candidates         Number of negative candidates drawn per triplet, allowing the training loop to select the hardest one.
     * @param crossDocumentRate  Proportion of negative candidates sourced from entirely different documents.
     * @param useAuthorCounts    Flag indicating whether to utilise the total author count from the truth file rather than strictly adjacent segments.
     * @param farPositiveRate    Probability that a positive match is drawn from a non-adjacent segment of the same author when author counts are utilised.
     */
    public record Policy(int candidates, double crossDocumentRate, boolean useAuthorCounts, double farPositiveRate) {

        /**
         * Instantiates a policy relying on the original rule where positives are drawn from the anchor's own segment and negatives from a neighbouring segment.
         */
        public static Policy adjacentSegments(int candidates, double crossDocumentRate) {
            return new Policy(candidates, crossDocumentRate, false, 0.0);
        }

        /**
         * Instantiates a policy that utilises the ground-truth author count to identify positive and negative candidates across the entire document.
         */
        public static Policy withAuthorCounts(int candidates, double crossDocumentRate, double farPositiveRate) {
            return new Policy(candidates, crossDocumentRate, true, farPositiveRate);
        }
    }

    private TripletSampler() {
    }

    /**
     * Generates batches of sentence index triplets for neural network training.
     *
     * @param data   The container holding document sentences alongside their respective segment and author mappings.
     * @param count  The desired total number of triplets to generate.
     * @param policy The configuration policy governing how positive and negative sentences are selected.
     * @param seed   The initialization seed for the random number generator to ensure reproducibility.
     * @return A two-dimensional array containing {@code count} rows formatted as {@code [anchor, positive, negative candidates...]}, or an empty array if no valid pairs exist.
     */
    public static int[][] draw(RetrievalData data, int count, Policy policy, long seed) {
        return policy.useAuthorCounts() ? drawWithAuthorCounts(data, count, policy, seed)
                : drawAdjacent(data, count, policy, seed);
    }

    /**
     * Executes the triplet generation strategy that utilises dataset-wide author counts to find global positive and negative candidates.
     *
     * @param data   The data container representing the loaded documents.
     * @param count  The required number of triplets to generate.
     * @param policy The sampling policy defining candidate counts and cross-document rates.
     * @param seed   The random initialization seed.
     * @return Multi-dimensional array of drawn triplet indices.
     */
    private static int[][] drawWithAuthorCounts(RetrievalData data, int count, Policy policy, long seed) {

        Random random = new Random(seed);
        int candidates = Math.max(1, policy.candidates());

        int[][] otherAuthor = new int[data.segments()][];
        int[][] sameAuthor = new int[data.segments()][];
        for (int segment = 0; segment < data.segments(); segment++) {
            classifySegments(data, segment, otherAuthor, sameAuthor);
        }

        int[] anchors = new int[data.size()];
        int anchorCount = 0;
        for (int sentence = 0; sentence < data.size(); sentence++) {
            int segment = data.segment[sentence];
            boolean hasNegative = otherAuthor[segment].length > 0;
            boolean hasPositive = data.segmentLength(segment) >= 2 || sameAuthor[segment].length > 0;
            if (hasNegative && hasPositive) {
                anchors[anchorCount++] = sentence;
            }
        }
        if (anchorCount == 0) {
            return new int[0][];
        }

        int[][] triplets = new int[count][2 + candidates];
        for (int t = 0; t < count; t++) {

            int anchor = anchors[random.nextInt(anchorCount)];
            int segment = data.segment[anchor];

            triplets[t][0] = anchor;
            triplets[t][1] = drawPositive(data, anchor, sameAuthor[segment], policy.farPositiveRate(), random);

            for (int c = 0; c < candidates; c++) {
                boolean otherDocument = policy.crossDocumentRate() > 0.0 && data.documents() > 1
                        && random.nextDouble() < policy.crossDocumentRate();
                triplets[t][2 + c] = otherDocument ? randomSentenceOfOtherDocument(data, anchor, random)
                        : randomSentenceOfSegment(data, otherAuthor[segment], random);
            }
        }
        return triplets;
    }

    /**
     * Categorizes all other segments within the same document as definitively sharing the same author
     * or belonging to a different author relative to the target segment.
     *
     * @param data        The data container representing the loaded documents.
     * @param segment     The target segment index being analyzed.
     * @param otherAuthor Output array to populate with segment indices confirmed to be by a different author.
     * @param sameAuthor  Output array to populate with segment indices confirmed to be by the same author.
     */
    private static void classifySegments(RetrievalData data, int segment, int[][] otherAuthor, int[][] sameAuthor) {

        int document = data.segDoc[segment];
        int first = data.docFirstSegment[document];
        int segmentCount = data.segmentsInDocument(document);
        int authors = data.authors[document];
        int position = segment - first;

        int[] different = new int[segmentCount];
        int[] same = new int[segmentCount];
        int differentCount = 0;
        int sameCount = 0;
        for (int other = 0; other < segmentCount; other++) {
            if (other == position) {
                continue;
            }
            Relation relation = SegmentAuthorship.between(position, other, segmentCount, authors);
            if (relation == Relation.DIFFERENT_AUTHOR) {
                different[differentCount++] = first + other;
            } else if (relation == Relation.SAME_AUTHOR) {
                same[sameCount++] = first + other;
            }
        }
        otherAuthor[segment] = java.util.Arrays.copyOf(different, differentCount);
        sameAuthor[segment] = java.util.Arrays.copyOf(same, sameCount);
    }

    /**
     * Selects a positive match for the anchor sentence, stochastically choosing between a sentence in the
     * same segment or a non-adjacent segment by the same author based on the far positive rate.
     *
     * @param data               The data container representing the loaded documents.
     * @param anchor             The globally indexed anchor sentence.
     * @param sameAuthorSegments Array of non-adjacent segment indices confirmed to be by the same author.
     * @param farPositiveRate    Probability threshold for drawing from a non-adjacent segment.
     * @param random             Random generator instance.
     * @return The global sentence index of the selected positive match.
     */
    private static int drawPositive(RetrievalData data, int anchor, int[] sameAuthorSegments, double farPositiveRate,
                                    Random random) {

        int segment = data.segment[anchor];
        boolean nearAvailable = data.segmentLength(segment) >= 2;
        boolean farAvailable = sameAuthorSegments.length > 0;
        boolean useFar = farAvailable && (!nearAvailable || random.nextDouble() < farPositiveRate);

        if (useFar) {
            return randomSentenceOfSegment(data, sameAuthorSegments, random);
        }
        int positive;
        do {
            positive = data.segStart[segment] + random.nextInt(data.segmentLength(segment));
        } while (positive == anchor);
        return positive;
    }

    /**
     * Selects a random sentence from a given array of candidate segment indices.
     *
     * @param data     The data container representing the loaded documents.
     * @param segments Array of valid segment indices to sample from.
     * @param random   Random generator instance.
     * @return The global sentence index of the randomly selected sentence.
     */
    private static int randomSentenceOfSegment(RetrievalData data, int[] segments, Random random) {
        int segment = segments[random.nextInt(segments.length)];
        return data.segStart[segment] + random.nextInt(data.segmentLength(segment));
    }

    /**
     * Executes the baseline triplet generation strategy, restricting positive matches to the anchor's own segment
     * and negative candidates to immediately adjacent segments.
     *
     * @param data   The data container representing the loaded documents.
     * @param count  The required number of triplets to generate.
     * @param policy The sampling policy defining candidate counts and cross-document rates.
     * @param seed   The random initialization seed.
     * @return Multi-dimensional array of drawn triplet indices.
     */
    private static int[][] drawAdjacent(RetrievalData data, int count, Policy policy, long seed) {

        Random random = new Random(seed);

        int[] anchors = new int[data.size()];
        int anchorCount = 0;
        for (int sentence = 0; sentence < data.size(); sentence++) {
            int segment = data.segment[sentence];
            if (data.segmentLength(segment) >= 2 && hasNeighbour(data, segment)) {
                anchors[anchorCount++] = sentence;
            }
        }
        if (anchorCount == 0) {
            return new int[0][];
        }

        int candidates = Math.max(1, policy.candidates());
        int[][] triplets = new int[count][2 + candidates];
        for (int t = 0; t < count; t++) {

            int anchor = anchors[random.nextInt(anchorCount)];
            int segment = data.segment[anchor];

            int positive;
            do {
                positive = data.segStart[segment] + random.nextInt(data.segmentLength(segment));
            } while (positive == anchor);

            triplets[t][0] = anchor;
            triplets[t][1] = positive;

            for (int c = 0; c < candidates; c++) {
                boolean otherDocument = policy.crossDocumentRate() > 0.0 && random.nextDouble() < policy.crossDocumentRate()
                        && data.documents() > 1;
                triplets[t][2 + c] = otherDocument ? randomSentenceOfOtherDocument(data, anchor, random)
                        : randomNeighbourSentence(data, segment, random);
            }
        }
        return triplets;
    }

    /**
     * Evaluates whether a given segment has at least one adjacent segment within the same document.
     *
     * @param data    The data container representing the loaded documents.
     * @param segment The index of the segment to evaluate.
     * @return {@code true} if an adjacent segment exists in the same document, {@code false} otherwise.
     */
    private static boolean hasNeighbour(RetrievalData data, int segment) {
        int document = data.segDoc[segment];
        boolean before = segment > 0 && data.segDoc[segment - 1] == document;
        boolean after = segment + 1 < data.segments() && data.segDoc[segment + 1] == document;
        return before || after;
    }

    /**
     * Selects a random sentence from an immediately preceding or succeeding segment within the same document.
     *
     * @param data    The data container representing the loaded documents.
     * @param segment The reference segment index.
     * @param random  Random generator instance.
     * @return The global sentence index of a sentence from an adjacent segment.
     */
    private static int randomNeighbourSentence(RetrievalData data, int segment, Random random) {

        int document = data.segDoc[segment];
        boolean before = segment > 0 && data.segDoc[segment - 1] == document;
        boolean after = segment + 1 < data.segments() && data.segDoc[segment + 1] == document;

        int neighbour;
        if (before && after) {
            neighbour = random.nextBoolean() ? segment - 1 : segment + 1;
        } else {
            neighbour = before ? segment - 1 : segment + 1;
        }
        return data.segStart[neighbour] + random.nextInt(data.segmentLength(neighbour));
    }

    /**
     * Selects a random sentence from an entirely different document to serve as a cross-document negative candidate.
     *
     * @param data   The data container representing the loaded documents.
     * @param anchor The global index of the current anchor sentence.
     * @param random Random generator instance.
     * @return The global sentence index of a sentence from a different document.
     */
    private static int randomSentenceOfOtherDocument(RetrievalData data, int anchor, Random random) {

        int document;
        do {
            document = random.nextInt(data.documents());
        } while (document == data.doc[anchor]);
        int size = data.docStart[document + 1] - data.docStart[document];
        return data.docStart[document] + random.nextInt(size);
    }
}