package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import persistence.ModelStore;
import routing.LevelRouter;

import java.io.File;
import java.util.List;

/**
 * What every trained model (decision tree, logistic regression, neural network and the retrieval model) has in
 * common as a classifier: the loaded model comes from the {@link MlModelLoader} on every call, not from a snapshot,
 * so a reload after a training run is used on the very next prediction. The scores come from
 * {@link ModelStore.Loaded#changeScores}, the one scoring path of the whole project.
 */
abstract class StoredModelClassifier implements AuthorSwitchClassifier {

    protected final MlModelLoader modelLoader;
    private final String type;

    StoredModelClassifier(MlModelLoader modelLoader, String type) {
        this.modelLoader = modelLoader;
        this.type = type;
    }

    protected ModelStore.Loaded model() {
        return modelLoader.get(type);
    }

    @Override
    public boolean isImplemented() {
        return model() != null;
    }

    @Override
    public String unavailableReason() {
        return modelLoader.problem(type);
    }

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
    public LevelRouter.Decision decideThreshold(List<String> sentences) {
        File file = modelLoader.fileOf(type);
        ModelStore.Loaded model = model();
        if (file == null || model == null) {
            return null;
        }
        LevelRouter router = LevelRouter.forModelFile(file);
        return router.ready() ? router.decide(sentences, model.threshold) : null;
    }

    @Override
    public String levelRouterProblem() {
        File file = modelLoader.fileOf(type);
        if (file == null) {
            return "";
        }
        String problem = LevelRouter.forModelFile(file).problem();
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
    public double predictSwitchProbability(String sentenceA, String sentenceB) {
        return require().changeScores(List.of(sentenceA, sentenceB))[0];
    }

    @Override
    public double[] switchProbabilities(List<String> sentences) {
        return require().changeScores(sentences);
    }
}
