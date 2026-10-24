package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import org.springframework.stereotype.Component;

/**
 * The retrieval model ({@link models.RetrievalModel}): every sentence is embedded on its own and a change is a
 * large distance between the sentences left and right of a position. It is used like the other models for the
 * switch detection on the Predict page, and by the Analyse and Retrieve pages for the similarity view and the
 * author segments.
 */
@Component
public class RetrievalClassifier extends StoredModelClassifier {

    public RetrievalClassifier(MlModelLoader modelLoader) {
        super(modelLoader, MlModelLoader.RETRIEVAL);
    }

    @Override
    public String getName() {
        return "retrieval_embedding";
    }

    @Override
    public String getDisplayName() {
        return "Retrieval (embedding)";
    }
}
