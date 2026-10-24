package com.hyp.authorswitchdetection.service;

import cli.RetrievalMain;
import com.hyp.authorswitchdetection.ml.MlModelLoader;
import persistence.ModelStore;
import models.RetrievalModel;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Locale;

/**
 * The Retrieve page: the sentences are embedded one by one by the retrieval model, the page shows which sentences
 * are written most alike (a heatmap of the cosine similarities, and the nearest neighbours of every sentence)
 * and where the neighbouring sentences are most unlike (the change scores used for the switch detection).
 */
@Service
public class RetrievalService {

    /** More sentences would make a heatmap nobody can read. */
    public static final int MAX_SENTENCES = 60;

    private final MlModelLoader modelLoader;

    public RetrievalService(MlModelLoader modelLoader) {
        this.modelLoader = modelLoader;
    }

    public RetrievalResult analyze(List<String> allSentences, int neighbours) {

        RetrievalResult result = new RetrievalResult();

        ModelStore.Loaded loaded = modelLoader.retrieval();
        if (loaded == null) {
            String problem = modelLoader.problem(MlModelLoader.RETRIEVAL);
            result.error = "The retrieval model can not be used: " + (problem != null ? problem
                    : "it has not been trained yet. Train it on the Train page (Retrieval), or run RetrievalMain.");
            return result;
        }

        List<String> sentences = allSentences;
        if (sentences.size() > MAX_SENTENCES) {
            sentences = allSentences.subList(0, MAX_SENTENCES);
            result.truncated = true;
        }

        RetrievalModel model = (RetrievalModel) loaded.model;
        result.modelDescription = loaded.description;
        result.nSentences = sentences.size();
        result.threshold = loaded.threshold;
        result.window = model.window;

        double[][] embeddings = model.embed(loaded.extractor, sentences);
        double[][] similarity = RetrievalModel.similarityMatrix(embeddings);
        result.similarity = similarity;

        // the colour scale: from the smallest to the largest similarity between two DIFFERENT sentences
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (int i = 0; i < similarity.length; i++) {
            for (int j = 0; j < similarity.length; j++) {
                if (i != j) {
                    min = Math.min(min, similarity[i][j]);
                    max = Math.max(max, similarity[i][j]);
                }
            }
        }
        if (min > max) { // a single sentence
            min = 0.0;
            max = 1.0;
        }
        result.minSimilarity = min;
        result.maxSimilarity = max;

        result.cellColor = new String[similarity.length][similarity.length];
        for (int i = 0; i < similarity.length; i++) {
            for (int j = 0; j < similarity.length; j++) {
                double t = i == j ? 1.0 : (max - min < 1e-9 ? 0.5 : (similarity[i][j] - min) / (max - min));
                // light = unlike, dark blue = alike
                result.cellColor[i][j] = String.format(Locale.ROOT, "hsl(215, 70%%, %d%%)",
                        (int) Math.round(96 - 56 * t));
            }
        }

        result.changeScore = model.changeScores(embeddings);
        result.predictedSwitch = new boolean[result.changeScore.length];
        for (int i = 0; i < result.changeScore.length; i++) {
            result.predictedSwitch[i] = result.changeScore[i] > loaded.threshold;
        }

        for (int i = 0; i < sentences.size(); i++) {
            RetrievalResult.SentenceView view = new RetrievalResult.SentenceView();
            view.index = i + 1;
            view.text = sentences.get(i);
            for (double[] near : RetrievalModel.nearest(similarity, i, neighbours)) {
                RetrievalResult.Neighbour n = new RetrievalResult.Neighbour();
                n.index = (int) near[0] + 1;
                n.similarity = near[1];
                n.text = sentences.get((int) near[0]);
                view.nearest.add(n);
            }
            result.sentences.add(view);
        }
        return result;
    }
}
