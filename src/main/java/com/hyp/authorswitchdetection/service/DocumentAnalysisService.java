package com.hyp.authorswitchdetection.service;

import analysis.DocumentSegmentation;
import com.hyp.authorswitchdetection.ml.MlModelLoader;
import features.FeatureExtractor;
import models.DifficultyDetector;
import persistence.ModelStore;
import routing.LevelRouter;
import org.springframework.stereotype.Service;
import util.Statistics;

import java.io.File;
import java.util.List;

/**
 * Builds the three views of the Analyse page for one document.
 *
 * <ul>
 *   <li><strong>Difficulty:</strong> predicted by the {@link DifficultyDetector} from document-level statistics;
 *       it does not use the retrieval model.</li>
 *   <li><strong>Raw features:</strong> the scalar stylometric features of every sentence, extracted without any
 *       trained model, so they are the same whichever model is installed.</li>
 *   <li><strong>Splits and author estimate:</strong> the {@code RetrievalModel} change scores are compared
 *       with the decision threshold (of the detected difficulty level when the router is calibrated, otherwise
 *       the pooled threshold of the model); every exceeded boundary starts a new segment and the number of
 *       segments is the estimated number of authors, see {@link DocumentSegmentation}.</li>
 * </ul>
 */
@Service
public class DocumentAnalysisService {

    /** Rows of the per-sentence feature table; the summary over all sentences is always complete. */
    public static final int MAX_FEATURE_ROWS = 300;

    private final MlModelLoader modelLoader;

    public DocumentAnalysisService(MlModelLoader modelLoader) {
        this.modelLoader = modelLoader;
    }

    /**
     * Analyses a document.
     *
     * @param sentences the sentences of the document in reading order
     * @return the analysis; unavailable parts explain themselves
     */
    public DocumentAnalysis analyse(List<String> sentences) {

        DocumentAnalysis analysis = new DocumentAnalysis();
        analysis.sentenceCount = sentences.size();

        File retrievalFile = modelLoader.fileOf(MlModelLoader.RETRIEVAL);
        LevelRouter router = LevelRouter.forModelFile(retrievalFile);

        describeFeatures(analysis, sentences);
        describeDifficulty(analysis, router, sentences);
        describeSegmentation(analysis, router, sentences);
        return analysis;
    }

    private void describeFeatures(DocumentAnalysis analysis, List<String> sentences) {

        int featureCount = FeatureExtractor.SCALAR_COUNT;
        for (int f = 0; f < featureCount; f++) {
            analysis.featureNames.add(FeatureExtractor.SCALAR_NAMES[f]);
        }

        double[][] values = new double[sentences.size()][];
        for (int i = 0; i < sentences.size(); i++) {
            double[] row = new SentenceFeatures(sentences.get(i), i, sentences.size()).toVector();
            values[i] = java.util.Arrays.copyOf(row, featureCount);
        }

        for (int f = 0; f < featureCount; f++) {
            double[] column = new double[values.length];
            double minimum = Double.MAX_VALUE;
            double maximum = -Double.MAX_VALUE;
            for (int i = 0; i < values.length; i++) {
                column[i] = values[i][f];
                minimum = Math.min(minimum, column[i]);
                maximum = Math.max(maximum, column[i]);
            }
            DocumentAnalysis.FeatureSummary summary = new DocumentAnalysis.FeatureSummary();
            summary.name = FeatureExtractor.SCALAR_NAMES[f];
            summary.mean = Statistics.mean(column);
            summary.standardDeviation = Statistics.populationStandardDeviation(column);
            summary.minimum = minimum;
            summary.maximum = maximum;
            analysis.featureSummaries.add(summary);
        }

        int rows = Math.min(values.length, MAX_FEATURE_ROWS);
        for (int i = 0; i < rows; i++) {
            analysis.featureRows.add(values[i]);
        }
        analysis.truncatedFeatures = values.length > rows;
    }

    private void describeDifficulty(DocumentAnalysis analysis, LevelRouter router, List<String> sentences) {

        DifficultyDetector detector = router.detector();
        if (detector == null) {
            analysis.difficultyProblem = router.problem() != null ? router.problem()
                    : "no difficulty detector is installed";
            return;
        }
        double[] probabilities = detector.probabilities(sentences);
        analysis.difficultyProbabilities = probabilities;
        analysis.difficultyLevels = DifficultyDetector.LEVELS;
        analysis.difficultyLevel = DifficultyDetector.LEVELS[DifficultyDetector.argMax(probabilities)];
    }

    private void describeSegmentation(DocumentAnalysis analysis, LevelRouter router, List<String> sentences) {

        ModelStore.Loaded loaded = modelLoader.retrieval();
        if (loaded == null) {
            String problem = modelLoader.problem(MlModelLoader.RETRIEVAL);
            analysis.segmentationProblem = "The retrieval model can not be used: "
                    + (problem != null ? problem : "it has not been trained yet. Train it on the Train page or with RetrievalMain.");
            return;
        }
        if (sentences.size() < 2) {
            analysis.segmentationProblem = "At least two sentences are needed to look for an author change.";
            return;
        }

        models.RetrievalModel model = (models.RetrievalModel) loaded.model;
        double[] scores = loaded.changeScores(sentences);

        LevelRouter.Decision decision = router.ready() ? router.decide(sentences, loaded.threshold) : null;
        analysis.threshold = decision != null && decision.level != null ? decision.threshold : loaded.threshold;
        analysis.thresholdSource = decision != null && decision.level != null ? decision.source
                : "pooled threshold of the model, chosen on validation data"
                        + (router.problem() != null ? " (" + router.problem() + ")" : "");
        analysis.modelDescription = loaded.description;
        analysis.window = model.window;
        analysis.changeScores = scores;

        DocumentSegmentation segmentation = DocumentSegmentation.fromScores(scores, analysis.threshold);
        analysis.predictedChanges = segmentation.changeCount();
        analysis.estimatedAuthors = segmentation.estimatedAuthors();

        int[] decisions = segmentation.changeDecisions();
        int segmentNumber = 0;
        for (DocumentSegmentation.Segment segment : segmentation.segments()) {
            DocumentAnalysis.SegmentView view = new DocumentAnalysis.SegmentView();
            view.number = ++segmentNumber;
            view.firstSentence = segment.firstSentence + 1;
            view.lastSentence = segment.endSentence;
            view.sentenceCount = segment.length();
            for (int i = segment.firstSentence; i < segment.endSentence; i++) {
                view.wordCount += sentences.get(i).trim().split("\\s+").length;

                DocumentAnalysis.SentenceView sentence = new DocumentAnalysis.SentenceView();
                sentence.index = i + 1;
                sentence.text = sentences.get(i);
                sentence.segment = view.number;
                if (i < scores.length) {
                    sentence.scoreToNext = scores[i];
                    sentence.splitAfter = decisions[i] == 1;
                }
                analysis.sentences.add(sentence);
            }
            analysis.segments.add(view);
        }
    }
}
