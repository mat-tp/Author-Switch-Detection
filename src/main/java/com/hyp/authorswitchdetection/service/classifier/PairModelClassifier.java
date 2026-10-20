package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import com.hyp.authorswitchdetection.ml.PairFeatureUtil;
import models.ModelStore;

import java.util.List;

/**
 * What the three pair models (decision tree, logistic regression, neural network) have in common: the loaded
 * model comes from the MlModelLoader on every call (holding the loader, not a snapshot, so a reload after
 * a training run is used on the very next prediction), and the probability is computed by
 * ModelStore.Loaded for the type of model.
 */
abstract class PairModelClassifier implements AuthorSwitchClassifier {

    protected final MlModelLoader modelLoader;
    private final String type;

    PairModelClassifier(MlModelLoader modelLoader, String type) {
        this.modelLoader = modelLoader;
        this.type = type;
    }

    protected ModelStore.Loaded model() {
        return modelLoader.get(type);
    }

    @Override
    public boolean isImplemented() { return model() != null; }

    @Override
    public String unavailableReason() { return modelLoader.problem(type); }

    @Override
    public double threshold() {
        ModelStore.Loaded model = model();
        return model != null ? model.threshold : 0.5;
    }

    @Override
    public String describeTraining() {
        ModelStore.Loaded model = model();
        return model != null ? model.description : "";
    }

    @Override
    public models.LevelRouter.Decision decideThreshold(List<String> sentences) {
        java.io.File file = modelLoader.fileOf(type);
        ModelStore.Loaded model = model();
        if (file == null || model == null) {
            return null;
        }
        models.LevelRouter router = models.LevelRouter.forModelFile(file);
        return router.ready() ? router.decide(sentences, model.threshold) : null;
    }

    @Override
    public String levelRouterProblem() {
        java.io.File file = modelLoader.fileOf(type);
        if (file == null) {
            return "";
        }
        String problem = models.LevelRouter.forModelFile(file).problem();
        return problem == null ? "" : problem;
    }

    private ModelStore.Loaded require() {
        ModelStore.Loaded model = model();
        if (model == null) {
            throw new UnsupportedOperationException(getDisplayName() + " model is not available.");
        }
        return model;
    }

    @Override
    public double predictSwitchProbability(String a, String b) {

        ModelStore.Loaded model = require();
        return model.probability(PairFeatureUtil.pairFeatures(model.extractor, a, b));
    }

    @Override
    public double[] switchProbabilities(List<String> sentences) {

        ModelStore.Loaded model = require();
        double[][] matrix = PairFeatureUtil.neighbourPairFeatures(model.extractor, sentences);
        return matrix.length == 0 ? new double[0] : model.probabilities(matrix);
    }
}
