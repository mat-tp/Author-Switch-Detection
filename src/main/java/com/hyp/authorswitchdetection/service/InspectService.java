package com.hyp.authorswitchdetection.service;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import com.hyp.authorswitchdetection.ml.PairFeatureUtil;
import data.SentencePair;
import features.FeatureExtractor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

/**
 * Shows the features exactly as the trained model computes them.
 *
 * The features are defined relative to the training data (TF-IDF uses the
 * training corpus, z-scores use the training mean / std), so they can only be
 * computed with the FeatureExtractor that was saved together with a model.
 * Nothing is reimplemented here, this is the same call chain
 * PairFeatureUtil.pairFeatures uses for a prediction.
 */
@Service
public class InspectService {

    private final MlModelLoader modelLoader;

    public InspectService(MlModelLoader modelLoader) {
        this.modelLoader = modelLoader;
    }

    /** The fitted extractor that was saved with the given model, if that model is available. */
    public Optional<FeatureExtractor> extractorFor(String modelName) {

        if (modelName == null) {
            return Optional.empty();
        }

        switch (modelName) {
            case "decision_tree":
                return Optional.ofNullable(modelLoader.decisionTree()).map(MlModelLoader.DecisionTreeModel::extractor);
            case "logistic_regression":
                return Optional.ofNullable(modelLoader.logisticRegression())
                        .map(MlModelLoader.LogisticRegressionModel::extractor);
            case "neural_network":
                return Optional.ofNullable(modelLoader.neuralNetwork())
                        .map(MlModelLoader.NeuralNetworkModel::extractor);
            default:
                return Optional.empty();
        }
    }

    public InspectResult inspect(List<String> sentences, String modelName, FeatureExtractor extractor) {

        InspectResult result = new InspectResult();
        result.modelName = modelName;
        result.nSentences = sentences.size();
        result.nPairs = Math.max(0, sentences.size() - 1);
        result.sentenceFeatureNames = PairFeatureUtil.SENTENCE_FEATURE_NAMES;
        result.pairFeatureNames = PairFeatureUtil.PAIR_FEATURE_NAMES;

        // extracting every sentence once (raw, then z-score)
        int n = sentences.size();
        double[][] raw = new double[n][];
        double[][] z = new double[n][];

        for (int i = 0; i < n; i++) {
            raw[i] = extractor.getFeatures(sentences.get(i));
            z[i] = extractor.standardize(raw[i]);
        }

        for (int i = 0; i < n; i++) {

            InspectResult.SentenceView view = new InspectResult.SentenceView();
            view.index = i + 1;
            view.text = sentences.get(i);
            view.raw = java.util.Arrays.copyOfRange(raw[i], 0, FeatureExtractor.SCALAR_COUNT);
            view.zScore = java.util.Arrays.copyOfRange(z[i], 0, FeatureExtractor.SCALAR_COUNT);

            int used = 0;
            for (int b = FeatureExtractor.SCALAR_COUNT; b < raw[i].length; b++) {
                if (raw[i][b] > 0) used++;
            }
            view.nGramBucketsUsed = used;

            // the 11 pair features between this sentence and the next one
            // (label -1 = unknown, same trick PairFeatureUtil uses)
            if (i < n - 1) {
                SentencePair pair = new SentencePair(sentences.get(i), sentences.get(i + 1), -1);
                view.toNext = pair.getPairFeatures(z[i], z[i + 1]);
            }

            result.sentences.add(view);
        }

        return result;
    }
}
