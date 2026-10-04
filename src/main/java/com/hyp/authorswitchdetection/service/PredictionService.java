package com.hyp.authorswitchdetection.service;

import com.hyp.authorswitchdetection.model.ModelMetrics;
import com.hyp.authorswitchdetection.service.classifier.AuthorSwitchClassifier;
import com.hyp.authorswitchdetection.service.classifier.ClassifierRegistry;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/*
    TODO: (1) Random Forest still falls back to the random baseline below,
    since hyProjectTP never actually trained one.
*/

/**
 * Drives the Prediction page.
 */
@Service
public class PredictionService {

    private final TrainingService trainingService;
    private final ClassifierRegistry classifierRegistry;

    public PredictionService(TrainingService trainingService, ClassifierRegistry classifierRegistry) {
        this.trainingService = trainingService;
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

        Optional<AuthorSwitchClassifier> classifier = modelName == null
                ? Optional.empty()
                : classifierRegistry.get(modelName).filter(AuthorSwitchClassifier::isImplemented);

        result.mode = classifier.isPresent() ? "model" : "random";

        // only models with a probability have a threshold (tuned on validation).
        // A decision tree outputs a class label, so there is nothing to tune.
        boolean withProbability = classifier.map(AuthorSwitchClassifier::hasProbability).orElse(true);
        result.hasThreshold = withProbability;

        ModelMetrics trained = modelName == null ? null : trainingService.latestFor(modelName);
        result.threshold = (withProbability && trained != null) ? trained.getDecisionThreshold() : 0.5;

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

        result.modelRules = classifier.map(AuthorSwitchClassifier::describeModel).orElse(List.of());

        Random rnd = new Random();
        int[] predicted = new int[result.nPairs];
        List<Integer> segmentStartIdx = new ArrayList<>();
        segmentStartIdx.add(0);

        for (int i = 0; i < result.nPairs; i++) {

            PredictionResult.Pair pair = new PredictionResult.Pair();
            pair.index = i;
            pair.sentenceA = sentences.get(i);
            pair.sentenceB = sentences.get(i + 1);

            if (classifier.isPresent()) {

                double p = classifier.get().predictSwitchProbability(pair.sentenceA, pair.sentenceB);
                pair.isSwitch = p >= result.threshold;
                pair.probability = withProbability ? Math.round(p * 1000.0) / 1000.0 : null;
                pair.reasons = classifier.get().explain(pair.sentenceA, pair.sentenceB);

            } else {

                // stand-by random predictions when nothing is trained
                double p = Math.max(0.0, Math.min(1.0, rnd.nextGaussian() * 0.2 + 0.3));
                pair.isSwitch = p >= result.threshold;
                pair.probability = Math.round(p * 1000.0) / 1000.0;
            }

            predicted[i] = pair.isSwitch ? 1 : 0;

            if (result.hasTruth) {
                pair.truth = truth[i];
                pair.outcome = pair.isSwitch ? (truth[i] == 1 ? "TP" : "FP") : (truth[i] == 1 ? "FN" : "TN");
            }

            if (pair.isSwitch) {
                result.nSwitches++;
                segmentStartIdx.add(i + 1);
            }
            result.pairs.add(pair);
        }

        // cutting the text at every predicted switch. A segment is not an author:
        // the same author can come back (A B A), so this is only "switches + 1"
        result.authorSegments = buildSegments(segmentStartIdx, sentences.size());
        result.nSegments = result.authorSegments.size();

        if (result.hasTruth && result.nPairs > 0) {
            result.metrics = EvaluationMetrics.of(predicted, truth);
        }

        return result;
    }

    private List<List<Integer>> buildSegments(List<Integer> starts, int nSentences) {
        List<List<Integer>> segments = new ArrayList<>();
        for (int s = 0; s < starts.size(); s++) {
            int from = starts.get(s);
            int to = (s + 1 < starts.size()) ? starts.get(s + 1) : nSentences;
            List<Integer> seg = new ArrayList<>();
            for (int i = from; i < to; i++) seg.add(i);
            segments.add(seg);
        }
        return segments;
    }
}
