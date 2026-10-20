package models;

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

    /** Cosine similarity of every two sentences, sentences.size() x sentences.size(). */
    public static double[][] similarityMatrix(double[][] embeddings) {

        int n = embeddings.length;
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < embeddings[i].length; k++) {
                    sum += embeddings[i][k] * embeddings[j][k];
                }
                matrix[i][j] = sum;
                matrix[j][i] = sum;
            }
        }
        return matrix;
    }

    /**
     * The score of a change between every two neighbouring sentences (sentences.size() - 1 numbers, 0 to 1,
     * high = different style), computed like RetrievalEvaluator.switchScores for one document.
     */
    public double[] changeScores(double[][] embeddings) {

        int n = embeddings.length;
        double[] scores = new double[Math.max(0, n - 1)];
        if (n < 2) {
            return scores;
        }
        int size = embeddings[0].length;

        for (int i = 0; i + 1 < n; i++) {
            double[] left = new double[size], right = new double[size];
            for (int k = Math.max(0, i - window + 1); k <= i; k++) {
                for (int j = 0; j < size; j++) {
                    left[j] += embeddings[k][j];
                }
            }
            for (int k = i + 1; k <= Math.min(n - 1, i + window); k++) {
                for (int j = 0; j < size; j++) {
                    right[j] += embeddings[k][j];
                }
            }
            double dot = 0.0, a = 0.0, b = 0.0;
            for (int j = 0; j < size; j++) {
                dot += left[j] * right[j];
                a += left[j] * left[j];
                b += right[j] * right[j];
            }
            scores[i] = (1.0 - dot / (Math.sqrt(a * b) + 1e-12)) / 2.0;
        }
        return scores;
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
