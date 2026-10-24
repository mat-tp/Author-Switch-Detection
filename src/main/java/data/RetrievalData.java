package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import features.FeatureExtractor;

/**
 * Encapsulates sentence-level data and document metadata for retrieval models operating
 * on individual sentences rather than sentence pairs.
 *
 * <p>Manages dense network embedding inputs for each sentence alongside mapping arrays
 * linking sentences to their parent documents and contiguous segments.</p>
 *
 * <p><strong>Segment Properties:</strong> A segment represents a contiguous run of sentences
 * authored by a single writer without an authorial change (see {@code DatasetLoader.Problem.segments}).
 * Sentences are stored sequentially document-by-document and segment-by-segment in contiguous array blocks.</p>
 *
 * <p><strong>Authorship Invariants:</strong>
 * <ul>
 *   <li>Two sentences within the <em>same</em> segment share the same author.</li>
 *   <li>Two sentences in <em>adjacent</em> segments belong to different authors.</li>
 *   <li>Non-adjacent segments may or may not share an author; training and evaluation protocols exclusively leverage adjacent/intra-segment relations.</li>
 * </ul>
 * </p>
 */
public class RetrievalData {

    /** Network input feature vectors for every sentence generated via {@link FeatureExtractor#embeddingInput(double[], double)}. */
    public final float[][] x;

    /** Parent document index corresponding to each sentence. */
    public final int[] doc;

    /** Global contiguous segment index corresponding to each sentence. */
    public final int[] segment;

    /** Starting sentence index for each document; final entry stores total sentence count. */
    public final int[] docStart;

    /** Starting sentence index for each segment; final entry stores total sentence count. */
    public final int[] segStart;

    /** Parent document index corresponding to each segment. */
    public final int[] segDoc;

    /** Author change indicator labels for each document (length = sentences - 1). */
    public final int[][] changes;

    /** Number of authors of each document from the truth file, 0 when it is not known. */
    public final int[] authors;

    /** Global index of the first segment of each document; the final entry stores the total segment count. */
    public final int[] docFirstSegment;

    /** Categorical difficulty level string (e.g., easy, medium, hard) for each document. */
    public final String[] difficulty;

    /** Original sentence text retained for reporting and examples ({@code null} if omitted). */
    public final String[] text;

    /**
     * Private constructor to instantiate immutable {@link RetrievalData} containers.
     */
    private RetrievalData(float[][] x, int[] doc, int[] segment, int[] docStart, int[] segStart, int[] segDoc,
                          int[][] changes, int[] authors, int[] docFirstSegment, String[] difficulty, String[] text) {
        this.x = x;
        this.doc = doc;
        this.segment = segment;
        this.docStart = docStart;
        this.segStart = segStart;
        this.segDoc = segDoc;
        this.changes = changes;
        this.authors = authors;
        this.docFirstSegment = docFirstSegment;
        this.difficulty = difficulty;
        this.text = text;
    }

    /**
     * Returns the total number of processed sentences in the dataset.
     *
     * @return Integer sentence count.
     */
    public int size() {
        return x.length;
    }

    /**
     * Returns the total number of documents in the dataset.
     *
     * @return Integer document count.
     */
    public int documents() {
        return docStart.length - 1;
    }

    /**
     * Returns the total number of segments across all documents.
     *
     * @return Integer segment count.
     */
    public int segments() {
        return segStart.length - 1;
    }

    /**
     * Returns the position of a segment inside its own document.
     *
     * @param segment Global segment index.
     * @return Position starting at 0 for the first segment of the document.
     */
    public int segmentPositionInDocument(int segment) {
        return segment - docFirstSegment[segDoc[segment]];
    }

    /**
     * Returns how many segments a document has.
     *
     * @param document Document index.
     * @return Number of segments.
     */
    public int segmentsInDocument(int document) {
        return docFirstSegment[document + 1] - docFirstSegment[document];
    }

    /**
     * Calculates the number of sentences contained within a specific segment.
     *
     * @param segment Target global segment index.
     * @return Integer sentence count for the segment.
     */
    public int segmentLength(int segment) {
        return segStart[segment + 1] - segStart[segment];
    }

    /**
     * Fits feature normalization parameters on training document sentences and returns a populated dataset.
     *
     * @param problems List of training documents.
     * @param extractor Unfitted feature extractor instance to fit.
     * @param profileScale Weighting factor for profile distributions (see {@link FeatureExtractor#embeddingInput(double[], double)}).
     * @param maxSentences Upper boundary on sentence count to process (selected via deterministic seed).
     * @return Fully initialized and scaled {@link RetrievalData} container.
     */
    public static RetrievalData buildTraining(List<DatasetLoader.Problem> problems, FeatureExtractor extractor,
                                              double profileScale, int maxSentences) {

        List<DatasetLoader.Problem> chosen = limit(problems, maxSentences);
        extractor.prepare();

        // Pass 1: Extract scalar features to fit z-score normalization parameters
        List<List<String>> sentences = new ArrayList<>();
        for (DatasetLoader.Problem p : chosen) {
            sentences.add(p.sentences);
        }

        int total = 0;
        for (DatasetLoader.Problem p : chosen) {
            total += p.sentences.size();
        }

        double[][] scalars = new double[total][];
        int[] offsets = new int[chosen.size()];
        int next = 0;
        for (int d = 0; d < chosen.size(); d++) {
            offsets[d] = next;
            next += chosen.get(d).sentences.size();
        }

        IntStream.range(0, chosen.size()).parallel().forEach(d -> {
            List<String> doc = chosen.get(d).sentences;
            for (int i = 0; i < doc.size(); i++) {
                scalars[offsets[d] + i] = java.util.Arrays.copyOf(extractor.getFeatures(doc.get(i)),
                        FeatureExtractor.SCALAR_COUNT);
            }
        });

        List<double[]> scalarList = new ArrayList<>(total);
        Collections.addAll(scalarList, scalars);
        extractor.fitScaler(scalarList);

        return build(chosen, extractor, profileScale, Integer.MAX_VALUE, true);
    }

    /**
     * Constructs a retrieval dataset from documents using a pre-fitted feature extractor.
     *
     * @param problems List of target documents.
     * @param extractor Pre-fitted feature extractor instance.
     * @param profileScale Weighting factor for profile distributions.
     * @param maxSentences Upper boundary on total sentence count.
     * @param keepText {@code true} to retain raw text strings for reporting.
     * @return Initialized {@link RetrievalData} container.
     */
    public static RetrievalData build(List<DatasetLoader.Problem> problems, FeatureExtractor extractor,
                                      double profileScale, int maxSentences, boolean keepText) {

        List<DatasetLoader.Problem> chosen = limit(problems, maxSentences);
        extractor.prepare();

        int nDocs = chosen.size();
        int[] docStart = new int[nDocs + 1];
        for (int d = 0; d < nDocs; d++) {
            docStart[d + 1] = docStart[d] + chosen.get(d).sentences.size();
        }
        int total = docStart[nDocs];

        float[][] x = new float[total][];
        int[] doc = new int[total];
        int[] segment = new int[total];
        String[] text = keepText ? new String[total] : null;

        // Process contiguous global segment allocations
        int[] segmentsBefore = new int[nDocs + 1];
        int[][] localSegments = new int[nDocs][];
        for (int d = 0; d < nDocs; d++) {
            localSegments[d] = chosen.get(d).segments();
            int count = localSegments[d].length == 0 ? 0 : localSegments[d][localSegments[d].length - 1] + 1;
            segmentsBefore[d + 1] = segmentsBefore[d] + count;
        }

        int nSegments = segmentsBefore[nDocs];
        int[] segStart = new int[nSegments + 1];
        int[] segDoc = new int[nSegments];
        int[][] changes = new int[nDocs][];
        int[] authors = new int[nDocs];
        String[] difficulty = new String[nDocs];

        for (int d = 0; d < nDocs; d++) {
            changes[d] = chosen.get(d).changes;
            authors[d] = chosen.get(d).authors;
            difficulty[d] = chosen.get(d).difficulty;
            for (int i = 0; i < localSegments[d].length; i++) {
                int global = segmentsBefore[d] + localSegments[d][i];
                int sentence = docStart[d] + i;
                doc[sentence] = d;
                segment[sentence] = global;
                segDoc[global] = d;
                if (i == 0 || localSegments[d][i] != localSegments[d][i - 1]) {
                    segStart[global] = sentence;
                }
            }
        }
        segStart[nSegments] = total;

        IntStream.range(0, nDocs).parallel().forEach(d -> {
            List<String> sentences = chosen.get(d).sentences;
            for (int i = 0; i < sentences.size(); i++) {
                x[docStart[d] + i] = extractor.embeddingInput(extractor.getFeatures(sentences.get(i)), profileScale);
                if (text != null) {
                    text[docStart[d] + i] = sentences.get(i);
                }
            }
        });

        return new RetrievalData(x, doc, segment, docStart, segStart, segDoc, changes, authors, segmentsBefore,
                difficulty, text);
    }

    /**
     * Filters documents to fit within a specified maximum sentence capacity using a deterministic shuffle.
     *
     * @param problems Input list of documents.
     * @param maxSentences Maximum sentence allowance.
     * @return Subset of selected documents.
     */
    private static List<DatasetLoader.Problem> limit(List<DatasetLoader.Problem> problems, int maxSentences) {

        long total = 0;
        for (DatasetLoader.Problem p : problems) {
            total += p.sentences.size();
        }
        if (total <= maxSentences) {
            return problems;
        }

        List<DatasetLoader.Problem> shuffled = new ArrayList<>(problems);
        Collections.shuffle(shuffled, new Random(42L));

        List<DatasetLoader.Problem> chosen = new ArrayList<>();
        long used = 0;
        for (DatasetLoader.Problem p : shuffled) {
            if (used + p.sentences.size() > maxSentences) {
                continue;
            }
            chosen.add(p);
            used += p.sentences.size();
        }
        return chosen;
    }
}