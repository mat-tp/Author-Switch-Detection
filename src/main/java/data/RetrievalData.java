package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import features.FeatureExtractor;

/**
 * The sentences of a list of documents for the retrieval model, ONE sentence at a time (no pairs):
 * the network input of every sentence, and which document and which segment it belongs to.
 *
 * A segment is a run of sentences without a change between them (DatasetLoader.Problem.segments).
 * The sentences are stored document after document, segment after segment, so the sentences of a
 * segment (or document) are one block of the arrays.
 *
 * What is known about authorship: two sentences of the SAME segment have the same author, two sentences of
 * NEXT-TO-EACH-OTHER segments have different authors. Two segments that are not next to each other can
 * have the same author (a document with two authors alternates) or not, the dataset does not say. The
 * training and the evaluation therefore only use the first two facts.
 */
public class RetrievalData {

    /** the network input of every sentence (FeatureExtractor.embeddingInput) */
    public final float[][] x;
    /** document of every sentence */
    public final int[] doc;
    /** global segment number of every sentence */
    public final int[] segment;
    /** first sentence of every document; one more entry at the end = number of sentences */
    public final int[] docStart;
    /** first sentence of every segment; one more entry at the end */
    public final int[] segStart;
    /** document of every segment */
    public final int[] segDoc;
    /** the change labels of every document (length = sentences - 1) */
    public final int[][] changes;
    /** easy / medium / hard of every document */
    public final String[] difficulty;
    /** the text, for the examples in the reports (null when not kept) */
    public final String[] text;

    private RetrievalData(float[][] x, int[] doc, int[] segment, int[] docStart, int[] segStart, int[] segDoc,
            int[][] changes, String[] difficulty, String[] text) {
        this.x = x;
        this.doc = doc;
        this.segment = segment;
        this.docStart = docStart;
        this.segStart = segStart;
        this.segDoc = segDoc;
        this.changes = changes;
        this.difficulty = difficulty;
        this.text = text;
    }

    public int size() {
        return x.length;
    }

    public int documents() {
        return docStart.length - 1;
    }

    public int segments() {
        return segStart.length - 1;
    }

    /** Number of sentences of a segment. */
    public int segmentLength(int segment) {
        return segStart[segment + 1] - segStart[segment];
    }

    /**
     * Fits the z-score parameters of the extractor on the sentences of the training documents
     * (each sentence once) - the pair pipeline fits the same scaler on the same sentences - and
     * returns the training data.
     *
     * @param problems      the training documents
     * @param extractor     not fitted yet; fitted by this call
     * @param profileScale  see FeatureExtractor.embeddingInput
     * @param maxSentences  at most this many sentences (whole documents, chosen with a fixed seed)
     */
    public static RetrievalData buildTraining(List<DatasetLoader.Problem> problems, FeatureExtractor extractor,
            double profileScale, int maxSentences) {

        List<DatasetLoader.Problem> chosen = limit(problems, maxSentences);
        extractor.prepare();

        // pass 1: only the scalars, for the z-score parameters
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
     * The data of documents, with the extractor already fitted (validation, test, and the
     * documents of the web app).
     *
     * @param keepText true = keep the sentences (for examples)
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

        // segments: global numbers, contiguous
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
        String[] difficulty = new String[nDocs];

        for (int d = 0; d < nDocs; d++) {
            changes[d] = chosen.get(d).changes;
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

        return new RetrievalData(x, doc, segment, docStart, segStart, segDoc, changes, difficulty, text);
    }

    /** Whole documents, at most maxSentences sentences (a fixed shuffle first, so it is a fair picture). */
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
