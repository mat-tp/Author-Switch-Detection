package com.hyp.authorswitchdetection.service.classifier;

import java.util.List;

/**
 * Contract every model (decision tree, logistic regression, neural network, retrieval embedding) implements.
 * Given the sentences of a document, return for every two neighbouring sentences how likely
 * an author switch is between them.
 *
 * The feature vector of a pair is always made by the FeatureExtractor that was saved with the
 * model (see PairFeatureUtil), so a model never sees features it was not trained on.
 */
public interface AuthorSwitchClassifier {

    /** Machine-readable key, e.g. "logistic_regression". Shown in the Train / Predict dropdowns. */
    String getName();

    /** Human-readable label, e.g. "Logistic Regression". */
    String getDisplayName();

    /** True once a trained model of this kind could be loaded. */
    boolean isImplemented();

    /**
     * Why the model can not be used although a file exists (a model saved before the features were
     * extended): the message to show. null = no problem, or nothing was trained yet.
     */
    default String unavailableReason() { return null; }

    /**
     * Probability (0..1) that sentence B starts a new author. For a model that
     * only gives a hard label (see {@link #hasProbability()}) this is 0.0 or 1.0.
     */
    double predictSwitchProbability(String sentenceA, String sentenceB);

    /**
     * The probabilities for every two neighbouring sentences of a document (n - 1 numbers). The
     * default asks pair by pair; the models override it to extract every sentence only once.
     */
    default double[] switchProbabilities(List<String> sentences) {

        double[] probabilities = new double[Math.max(0, sentences.size() - 1)];
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = predictSwitchProbability(sentences.get(i), sentences.get(i + 1));
        }
        return probabilities;
    }

    /**
     * The difficulty level of this document, detected from its text, and the threshold of that level (routing.LevelRouter).
     * null = not available (no detector or no thresholds per level): the single threshold() is used.
     */
    default routing.LevelRouter.Decision decideThreshold(List<String> sentences) { return null; }

    /** Why decideThreshold gives null (empty = no reason to show). */
    default String levelRouterProblem() { return ""; }

    /** The decision threshold chosen on the validation data when the model was trained (saved inside the model file). */
    default double threshold() { return 0.5; }

    /**
     * True when the output is a real probability that is compared against a
     * tuned threshold. A decision tree gives a hard 0/1 label instead: it has
     * no probability and no threshold.
     */
    default boolean hasProbability() { return true; }

    /** One line about the model: the settings it was trained with (empty if not known). */
    default String describeTraining() { return ""; }

    /** Human-readable reasons behind the prediction of one pair (empty if the model can't explain itself). */
    default List<String> explain(String sentenceA, String sentenceB) { return List.of(); }

    /** The whole model as readable rules (empty if the model isn't rule based). */
    default List<String> describeModel() { return List.of(); }
}
