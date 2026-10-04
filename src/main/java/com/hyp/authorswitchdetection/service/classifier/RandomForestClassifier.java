package com.hyp.authorswitchdetection.service.classifier;

import org.springframework.stereotype.Component;

/** TODO: replace this stub with your real random forest model (if you get to it). */
@Component
public class RandomForestClassifier implements AuthorSwitchClassifier {

    @Override
    public String getName() { return "random_forest"; }

    @Override
    public String getDisplayName() { return "Random Forest"; }

    @Override
    public boolean isImplemented() { return false; }

    @Override
    public double predictSwitchProbability(String a, String b) {
        throw new UnsupportedOperationException("Random Forest model is not trained/implemented yet.");
    }
}
