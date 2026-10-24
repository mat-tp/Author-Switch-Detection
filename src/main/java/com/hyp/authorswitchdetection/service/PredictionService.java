package com.hyp.authorswitchdetection.service;

import analysis.DocumentSegmentation;
import com.hyp.authorswitchdetection.service.classifier.AuthorSwitchClassifier;
import com.hyp.authorswitchdetection.service.classifier.ClassifierRegistry;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

/**
 * Drives the Prediction page.
 *
 * A prediction is made by a trained model or not at all: when the model was not trained yet, or was saved by
 * an older version of the code, the result says so (PredictionResult.error). There is no stand-in
 * that makes up numbers, because a page that shows random "predictions" next to the name of a model
 * can be mistaken for a result.
 */
@Service
public class PredictionService {

    private final ClassifierRegistry classifierRegistry;

    public PredictionService(ClassifierRegistry classifierRegistry) {
        this.classifierRegistry = classifierRegistry;
    }

    /**
     * @param sentences one sentence per entry
     * @param modelName which model to use
     * @param truth     optional truth values (one 0/1 per sentence pair), null if not given
     */
    public PredictionResult predict(List<String> sentences, String modelName, int[] truth) {

        PredictionResult result = new PredictionResult();
        result.nSentences = sentences.size();
        result.nPairs = Math.max(0, sentences.size() - 1);
        result.modelName = modelName;

        Optional<AuthorSwitchClassifier> found = modelName == null ? Optional.empty()
                : classifierRegistry.get(modelName);

        if (found.isEmpty()) {
            result.error = "Choose a model first.";
            return result;
        }

        AuthorSwitchClassifier classifier = found.get();

        if (!classifier.isImplemented()) {
            String reason = classifier.unavailableReason();
            result.error = classifier.getDisplayName() + " can not be used: "
                    + (reason != null ? reason : "it has not been trained yet. Train it on the Train page "
                            + "(or with Main / RetrievalMain), then come back.");
            return result;
        }

        result.hasThreshold = classifier.hasProbability();
        result.threshold = classifier.threshold();
        result.modelDescription = classifier.describeTraining();

        // the truth has to line up with the pairs, otherwise metrics would be wrong
        if (truth != null) {
            boolean onlyZeroOne = true;
            for (int t : truth) {
                if (t != 0 && t != 1) onlyZeroOne = false;
            }

            if (!onlyZeroOne) {
                result.truthError = "The truth file may only contain 0 (same author) and 1 (author switch).";
            } else if (truth.length == result.nPairs) {
                result.hasTruth = true;
            } else {
                result.truthError = "The truth file has " + truth.length + " values, but the text has "
                        + result.nSentences + " sentences (" + result.nPairs + " pairs). "
                        + "They must match, so no metrics were computed.";
            }
        }

        result.modelRules = classifier.describeModel();

        // the difficulty level is not asked for: it is detected from the text and the threshold of that level is used
        if (result.hasThreshold) {
            routing.LevelRouter.Decision decision = classifier.decideThreshold(sentences);
            if (decision != null && decision.level != null) {
                result.threshold = decision.threshold;
                result.detectedLevel = decision.level;
                result.levelProbabilities = String.format(java.util.Locale.ROOT, "easy %.2f | medium %.2f | hard %.2f",
                        decision.probabilities[0], decision.probabilities[1], decision.probabilities[2]);
                result.thresholdNote = decision.source;
            } else {
                String problem = classifier.levelRouterProblem();
                result.thresholdNote = problem.isEmpty() ? "" : "single threshold of the model; difficulty not detected: " + problem;
            }
        }

        double[] probabilities = classifier.switchProbabilities(sentences);

        int[] predicted = new int[result.nPairs];

        for (int i = 0; i < result.nPairs; i++) {

            PredictionResult.Pair pair = new PredictionResult.Pair();
            pair.index = i;
            pair.sentenceA = sentences.get(i);
            pair.sentenceB = sentences.get(i + 1);

            // '>' like the command line: models.LogisticRegression.predict
            pair.isSwitch = probabilities[i] > result.threshold;
            pair.probability = Math.round(probabilities[i] * 1000.0) / 1000.0;
            pair.reasons = classifier.explain(pair.sentenceA, pair.sentenceB);

            predicted[i] = pair.isSwitch ? 1 : 0;

            if (result.hasTruth) {
                pair.truth = truth[i];
                pair.outcome = pair.isSwitch ? (truth[i] == 1 ? "TP" : "FP") : (truth[i] == 1 ? "FN" : "TN");
            }

            if (pair.isSwitch) {
                result.nSwitches++;
            }
            result.pairs.add(pair);
        }

        // a segment is not an author (the same author can come back), see DocumentSegmentation
        DocumentSegmentation segmentation = DocumentSegmentation.fromScores(probabilities, result.threshold);
        result.authorSegments = segmentation.sentenceIndicesPerSegment();
        result.nSegments = segmentation.estimatedAuthors();

        if (result.hasTruth && result.nPairs > 0) {
            result.metrics = EvaluationMetrics.of(predicted, truth);
        }

        return result;
    }
}
