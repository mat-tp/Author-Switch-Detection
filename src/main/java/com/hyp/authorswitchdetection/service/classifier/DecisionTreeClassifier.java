package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import com.hyp.authorswitchdetection.ml.PairFeatureUtil;
import models.DecisionTree;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/*
    Was a stub ("replace this with your real decision tree model"), now backed
    by the actual decision tree trained by RealTrainer (or, before anything has
    been trained from the UI, the bundled baseline) — see MlModelLoader.
*/
@Component
public class DecisionTreeClassifier implements AuthorSwitchClassifier {

    // holding the loader, not a snapshot of its model — reload() after a fresh
    // training run should be picked up on the very next prediction
    private final MlModelLoader modelLoader;

    public DecisionTreeClassifier(MlModelLoader modelLoader) {
        this.modelLoader = modelLoader;
    }

    @Override
    public String getName() { return "decision_tree"; }

    @Override
    public String getDisplayName() { return "Decision Tree"; }

    @Override
    public boolean isImplemented() { return modelLoader.decisionTree() != null; }

    @Override
    public double predictSwitchProbability(String a, String b) {

        MlModelLoader.DecisionTreeModel model = modelLoader.decisionTree();
        if (model == null) {
            throw new UnsupportedOperationException("Decision Tree model is not available.");
        }

        double[] pairFeatures = PairFeatureUtil.pairFeatures(model.extractor(), a, b, model.fullFeatures());

        // a decision tree gives a hard 0/1 label, not a probability (see hasProbability)
        return model.tree().predict(pairFeatures) == 1 ? 1.0 : 0.0;
    }

    /** A tree ends in a class label, so there is no probability to compare against a threshold. */
    @Override
    public boolean hasProbability() { return false; }

    /**
     * The questions the tree asked for this pair, e.g.
     * "diff_avg_word_length = 0.412 > 0.318", ending with the label reached.
     */
    @Override
    public List<String> explain(String a, String b) {

        MlModelLoader.DecisionTreeModel model = modelLoader.decisionTree();
        if (model == null) {
            return List.of();
        }

        double[] pairFeatures = PairFeatureUtil.pairFeatures(model.extractor(), a, b, model.fullFeatures());
        DecisionTree.Explanation explanation = model.tree().explain(pairFeatures);

        List<String> lines = new ArrayList<>();
        for (DecisionTree.DecisionStep step : explanation.steps) {
            lines.add(String.format(Locale.ROOT, "%s = %.3f %s %.3f",
                    PairFeatureUtil.pairFeatureName(step.featureIndex), step.value,
                    step.wentLeft ? "<=" : ">", step.threshold));
        }
        lines.add(explanation.label == 1 ? "\u2192 author switch" : "\u2192 same author");
        return lines;
    }

    /** The whole tree as if / else rules, so the model can be read (and defended) as a whole. */
    @Override
    public List<String> describeModel() {

        MlModelLoader.DecisionTreeModel model = modelLoader.decisionTree();
        if (model == null) {
            return List.of();
        }

        String[] names = new String[model.fullFeatures() ? 11 + 256 : 11];
        for (int i = 0; i < names.length; i++) {
            names[i] = PairFeatureUtil.pairFeatureName(i);
        }
        return model.tree().toRules(names);
    }
}
