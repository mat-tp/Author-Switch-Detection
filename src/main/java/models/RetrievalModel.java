package models;

import util.BoundaryScores;
import util.VectorMath;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import features.FeatureExtractor;

/**
 * What is saved for the retrieval model (ModelStore type "retrieval"): the embedding network and the
 * two settings that belong to it. The fitted extractor is saved next to it by ModelStore, as for every model.
 *
 * A document goes in as a list of sentences. Each sentence is embedded on its own (position in the document
 * plays no role), so the same sentence has the same embedding in any document, and the model can be used
 * for the two jobs: retrieval (similarity between sentences) and switch detection (distance between neighbours).
 */
public class RetrievalModel implements Serializable {

    private static final long serialVersionUID = 1L;

    public final EmbeddingNetwork network;

    /** the factor of the profile part of the input, see FeatureExtractor.embeddingInput */
    public final double profileScale;

    /** how many sentences left and right of a position are compared, chosen on the validation data */
    public final int window;

    public RetrievalModel(EmbeddingNetwork network, double profileScale, int window) {
        this.network = network;
        this.profileScale = profileScale;
        this.window = window;
    }

    /** The network input of every sentence (z-scores need the fitted extractor). */
    public float[][] inputs(FeatureExtractor extractor, List<String> sentences) {

        float[][] x = new float[sentences.size()][];
        for (int i = 0; i < x.length; i++) {
            x[i] = extractor.embeddingInput(extractor.getFeatures(sentences.get(i)), profileScale);
        }
        return x;
    }

    /** The embeddings (unit vectors) of the sentences. */
    public double[][] embed(FeatureExtractor extractor, List<String> sentences) {
        return network.embedAll(inputs(extractor, sentences));
    }

    /**
     * Cosine similarity of every two sentences.
     *
     * @param embeddings unit-length embeddings, one per sentence
     * @return a symmetric matrix of size sentences x sentences
     */
    public static double[][] similarityMatrix(double[][] embeddings) {

        int n = embeddings.length;
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double similarity = VectorMath.dot(embeddings[i], embeddings[j]);
                matrix[i][j] = similarity;
                matrix[j][i] = similarity;
            }
        }
        return matrix;
    }

    /**
     * The score of a change between every two neighbouring sentences: 0 to 1, high means different style.
     * See {@link BoundaryScores} for the definition, which the evaluator uses as well.
     *
     * @param embeddings the embeddings of the sentences of one document
     * @return one score per boundary, {@code sentences - 1} numbers
     */
    public double[] changeScores(double[][] embeddings) {
        return BoundaryScores.forDocument(embeddings, 0, embeddings.length, window);
    }

    /** 0 / 1 per pair of neighbours, with the threshold chosen on the validation data. */
    public int[] predict(FeatureExtractor extractor, List<String> sentences, double threshold) {

        double[] scores = changeScores(embed(extractor, sentences));
        int[] result = new int[scores.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = scores[i] > threshold ? 1 : 0;
        }
        return result;
    }

    /** The sentences that are most similar to sentence `query`: {index, similarity}, best first. */
    public static List<double[]> nearest(double[][] similarity, int query, int k) {

        List<double[]> all = new ArrayList<>();
        for (int j = 0; j < similarity.length; j++) {
            if (j != query) {
                all.add(new double[] { j, similarity[query][j] });
            }
        }
        all.sort((a, b) -> Double.compare(b[1], a[1]));
        return all.subList(0, Math.min(k, all.size()));
    }
}
