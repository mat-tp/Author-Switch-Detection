package com.hyp.authorswitchdetection.service.classifier;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import com.hyp.authorswitchdetection.ml.PairFeatureUtil;
import models.DecisionTree;
import models.ModelStore;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The decision tree trained by Main / RealTrainer, see MlModelLoader.
 *
 * The tree has leaf probabilities (the share of change pairs that reached the leaf in training), so it
 * gives a probability and a threshold like the other models. The reasons shown with a prediction are
 * the questions the tree asked.
 */
@Component
public class DecisionTreeClassifier extends PairModelClassifier {

    public DecisionTreeClassifier(MlModelLoader modelLoader) {
        super(modelLoader, MlModelLoader.TREE);
    }

    @Override
    public String getName() { return "decision_tree"; }

    @Override
    public String getDisplayName() { return "Decision Tree"; }

    /**
     * The questions the tree asked for this pair, e.g.
     * "diff_avg_word_length = 0.412 > 0.318", ending with the label reached.
     */
    @Override
    public List<String> explain(String a, String b) {

        ModelStore.Loaded model = model();
        if (model == null) {
            return List.of();
        }

        double[] pairFeatures = PairFeatureUtil.pairFeatures(model.extractor, a, b);
        String[] names = model.extractor.pairFeatureNames();
        DecisionTree.Explanation explanation = ((DecisionTree) model.model).explain(pairFeatures);

        List<String> lines = new ArrayList<>();
        for (DecisionTree.DecisionStep step : explanation.steps) {
            lines.add(String.format(Locale.ROOT, "%s = %.3f %s %.3f", names[step.featureIndex], step.value,
                    step.wentLeft ? "<=" : ">", step.threshold));
        }
        lines.add(explanation.label == 1 ? "→ author switch" : "→ same author");
        return lines;
    }

    /** The whole tree as if / else rules, so the model can be read (and defended) as a whole. */
    @Override
    public List<String> describeModel() {

        ModelStore.Loaded model = model();
        if (model == null) {
            return List.of();
        }
        return ((DecisionTree) model.model).toRules(model.extractor.pairFeatureNames());
    }
}
