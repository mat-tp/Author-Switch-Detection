package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import org.springframework.stereotype.Component;

/** The neural network trained by Main / RealTrainer, see MlModelLoader. */
@Component
public class NeuralNetworkClassifier extends StoredModelClassifier {

    public NeuralNetworkClassifier(MlModelLoader modelLoader) {
        super(modelLoader, MlModelLoader.NEURAL);
    }

    @Override
    public String getName() { return "neural_network"; }

    @Override
    public String getDisplayName() { return "Neural Network"; }
}
