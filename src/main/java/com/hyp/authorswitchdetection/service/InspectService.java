package com.hyp.authorswitchdetection.service;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import com.hyp.authorswitchdetection.ml.PairFeatureUtil;
import features.FeatureConfig;
import features.FeatureExtractor;
import persistence.ModelStore;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Shows the features exactly as the trained model computes them.
 *
 * The features are defined relative to the training data (TF-IDF uses the training corpus, z-scores use
 * the training mean / std), so they are computed with the FeatureExtractor that was saved together with a
 * model. Nothing is reimplemented here, this is the same call chain a prediction uses.
 *
 * When no model is available the page still works: the features are then computed with
 * an extractor fitted on the pasted text itself (the sentences are the "training data"), and the page
 * says so, because the z-scores then only tell how a sentence differs from the rest of THIS text.
 */
@Service
public class InspectService {

    private final MlModelLoader modelLoader;

    public InspectService(MlModelLoader modelLoader) {
        this.modelLoader = modelLoader;
    }

    /** The model type (tree, ...) of a model name from the classifier registry. */
    private static String typeOf(String modelName) {

        if (modelName == null) {
            return null;
        }
        switch (modelName) {
            case "decision_tree":
                return MlModelLoader.TREE;
            case "logistic_regression":
                return MlModelLoader.LOGISTIC;
            case "neural_network":
                return MlModelLoader.NEURAL;
            case "retrieval_embedding":
                return MlModelLoader.RETRIEVAL;
            default:
                return null;
        }
    }

    /** The fitted extractor that was saved with the given model, if that model is available. */
    public Optional<FeatureExtractor> extractorFor(String modelName) {

        String type = typeOf(modelName);
        ModelStore.Loaded loaded = type == null ? null : modelLoader.get(type);
        return Optional.ofNullable(loaded).map(l -> l.extractor);
    }

    /** Why the model of this name can not be used although a file exists, null if there is no problem. */
    public String problemWith(String modelName) {
        String type = typeOf(modelName);
        return type == null ? null : modelLoader.problem(type);
    }

    /** An extractor fitted on the pasted sentences themselves (default feature settings), for use without a model. */
    public FeatureExtractor extractorFromText(List<String> sentences) {

        FeatureExtractor extractor = new FeatureExtractor(sentences, new FeatureConfig());
        extractor.prepare();

        List<double[]> raw = new ArrayList<>();
        for (String sentence : sentences) {
            raw.add(extractor.getFeatures(sentence));
        }
        extractor.fitScaler(raw);
        return extractor;
    }

    public InspectResult inspect(List<String> sentences, String modelName, FeatureExtractor extractor) {

        InspectResult result = new InspectResult();
        result.modelName = modelName;
        result.nSentences = sentences.size();
        result.nPairs = Math.max(0, sentences.size() - 1);
        result.sentenceFeatureNames = PairFeatureUtil.SENTENCE_FEATURE_NAMES;
        result.pairFeatureNames = PairFeatureUtil.pairFeatureNames(extractor);
        result.featureConfig = extractor.getConfig().signature();
        result.featureDescription = extractor.describe();
        result.sentenceVectorLength = extractor.sentenceVectorLength();
        result.profileBuckets = extractor.sentenceVectorLength() - FeatureExtractor.SCALAR_COUNT;

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

            // the pair vector between this sentence and the next one, exactly what a model is given
            if (i < n - 1) {
                view.toNext = extractor.pairVector(raw[i], raw[i + 1]);
            }

            result.sentences.add(view);
        }

        return result;
    }
}
