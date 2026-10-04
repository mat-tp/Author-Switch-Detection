package com.hyp.authorswitchdetection.service.classifier;

import java.util.List;

/**
 * Contract every model (logistic regression, decision tree, random forest,
 * neural network, ...) implements. Given two consecutive sentences, return
 * whether an author switch happened between them.
 *
 * Every model turns the pair of raw sentences into the SAME feature vector it
 * was trained on (see PairFeatureUtil): a sentence becomes 9 scalar features
 * (+ 256 hashed character-trigram buckets), a pair becomes the absolute
 * z-score differences of the scalars, the n-gram cosine similarity and the
 * overall (RMS) z-score distance.
 */
public interface AuthorSwitchClassifier {

    /** Machine-readable key, e.g. "logistic_regression". Shown in the Train / Predict dropdowns. */
    String getName();

    /** Human-readable label, e.g. "Logistic Regression". */
    String getDisplayName();

    /** True once this classifier has real, trained parameters (not just a stub). */
    boolean isImplemented();

    /**
     * Probability (0..1) that sentence B starts a new author. For a model that
     * only gives a hard label (see {@link #hasProbability()}) this is 0.0 or 1.0.
     */
    double predictSwitchProbability(String sentenceA, String sentenceB);

    /**
     * True when the output is a real probability that is compared against a
     * tuned threshold. A decision tree gives a hard 0/1 label instead: it has
     * no probability and no threshold.
     */
    default boolean hasProbability() { return true; }

    /** Human-readable reasons behind the prediction of one pair (empty if the model can't explain itself). */
    default List<String> explain(String sentenceA, String sentenceB) { return List.of(); }

    /** The whole model as readable rules (empty if the model isn't rule based). */
    default List<String> describeModel() { return List.of(); }
}
