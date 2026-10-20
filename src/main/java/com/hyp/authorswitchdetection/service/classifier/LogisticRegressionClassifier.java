package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import org.springframework.stereotype.Component;

/** The logistic regression trained by Main / RealTrainer, see MlModelLoader. */
@Component
public class LogisticRegressionClassifier extends PairModelClassifier {

    public LogisticRegressionClassifier(MlModelLoader modelLoader) {
        super(modelLoader, MlModelLoader.LOGISTIC);
    }

    @Override
    public String getName() { return "logistic_regression"; }

    @Override
    public String getDisplayName() { return "Logistic Regression"; }
}
