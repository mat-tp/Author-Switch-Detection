package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import com.hyp.authorswitchdetection.ml.PairFeatureUtil;
import org.springframework.stereotype.Component;

/*
    Was a stub ("replace this with your real neural network model"), now
    backed by the actual network trained by RealTrainer (or, before anything
    has been trained from the UI, the bundled baseline) — see MlModelLoader.
*/
@Component
public class NeuralNetworkClassifier implements AuthorSwitchClassifier {

    // holding the loader, not a snapshot of its model — reload() after a fresh
    // training run should be picked up on the very next prediction
    private final MlModelLoader modelLoader;

    public NeuralNetworkClassifier(MlModelLoader modelLoader) {
        this.modelLoader = modelLoader;
    }

    @Override
    public String getName() { return "neural_network"; }

    @Override
    public String getDisplayName() { return "Neural Network"; }

    @Override
    public boolean isImplemented() { return modelLoader.neuralNetwork() != null; }

    /** Threshold chosen during training (best F1 of class 1 on the validation split). */
    public double trainedThreshold() {
        MlModelLoader.NeuralNetworkModel model = modelLoader.neuralNetwork();
        return model != null ? model.threshold() : 0.5;
    }

    @Override
    public double predictSwitchProbability(String a, String b) {

        MlModelLoader.NeuralNetworkModel model = modelLoader.neuralNetwork();
        if (model == null) {
            throw new UnsupportedOperationException("Neural Network model is not available.");
        }

        double[] pairFeatures = PairFeatureUtil.pairFeatures(model.extractor(), a, b, model.fullFeatures());

        // single-sample batch, matching how Main.predictNewPair calls it
        return model.network().predictProbabilities(new double[][]{pairFeatures})[0];
    }
}
