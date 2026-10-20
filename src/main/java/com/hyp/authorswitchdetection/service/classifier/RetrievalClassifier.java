package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import models.ModelStore;
import models.RetrievalModel;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * The retrieval model (models.RetrievalModel): every sentence is embedded on its own and a change is
 * a large distance between the sentences left and right of a position. It is used like the other models for
 * the switch detection on the Predict page, and by RetrievalService for the similarity view.
 */
@Component
public class RetrievalClassifier implements AuthorSwitchClassifier {

    private final MlModelLoader modelLoader;

    public RetrievalClassifier(MlModelLoader modelLoader) {
        this.modelLoader = modelLoader;
    }

    @Override
    public String getName() { return "retrieval_embedding"; }

    @Override
    public String getDisplayName() { return "Retrieval (embedding)"; }

    @Override
    public boolean isImplemented() { return modelLoader.retrieval() != null; }

    @Override
    public String unavailableReason() { return modelLoader.problem(MlModelLoader.RETRIEVAL); }

    @Override
    public double threshold() {
        ModelStore.Loaded model = modelLoader.retrieval();
        return model != null ? model.threshold : 0.5;
    }

    @Override
    public String describeTraining() {
        ModelStore.Loaded model = modelLoader.retrieval();
        return model != null ? model.description : "";
    }

    @Override
    public double predictSwitchProbability(String a, String b) {
        return switchProbabilities(List.of(a, b))[0];
    }

    @Override
    public double[] switchProbabilities(List<String> sentences) {

        ModelStore.Loaded loaded = modelLoader.retrieval();
        if (loaded == null) {
            throw new UnsupportedOperationException("The retrieval model is not available.");
        }
        RetrievalModel model = (RetrievalModel) loaded.model;
        return model.changeScores(model.embed(loaded.extractor, sentences));
    }
}
