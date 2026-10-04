package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import com.hyp.authorswitchdetection.ml.PairFeatureUtil;
import models.LogisticRegression;
import org.springframework.stereotype.Component;

/*
    Was a stub ("replace this with your real logistic regression model"), now
    backed by the actual weights + bias trained by RealTrainer (or, before
    anything has been trained from the UI, the bundled baseline) — see
    MlModelLoader.
*/
@Component
public class LogisticRegressionClassifier implements AuthorSwitchClassifier {

    // holding the loader, not a snapshot of its model — reload() after a fresh
    // training run should be picked up on the very next prediction
    private final MlModelLoader modelLoader;

    public LogisticRegressionClassifier(MlModelLoader modelLoader) {
        this.modelLoader = modelLoader;
    }

    @Override
    public String getName() { return "logistic_regression"; }

    @Override
    public String getDisplayName() { return "Logistic Regression"; }

    @Override
    public boolean isImplemented() { return modelLoader.logisticRegression() != null; }

    /** Threshold chosen during training (best F1 of class 1 on the validation split). */
    public double trainedThreshold() {
        MlModelLoader.LogisticRegressionModel model = modelLoader.logisticRegression();
        return model != null ? model.threshold() : 0.5;
    }

    @Override
    public double predictSwitchProbability(String a, String b) {

        MlModelLoader.LogisticRegressionModel model = modelLoader.logisticRegression();
        if (model == null) {
            throw new UnsupportedOperationException("Logistic Regression model is not available.");
        }

        double[] pairFeatures = PairFeatureUtil.pairFeatures(model.extractor(), a, b, model.fullFeatures());

        // single-sample batch, same trick LogisticRegression.predict(features, ...) uses
        return LogisticRegression.predictProbabilities(new double[][]{pairFeatures}, model.weightsBias())[0];
    }
}
