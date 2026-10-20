package com.hyp.authorswitchdetection.service;

import com.hyp.authorswitchdetection.ml.MlModelLoader;
import com.hyp.authorswitchdetection.ml.RealTrainer;
import com.hyp.authorswitchdetection.model.ModelMetrics;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Drives the page for Training the  Model.
 */
@Service
public class TrainingService {

    private final ModelMetricsStore metricsStore;
    private final RealTrainer realTrainer;
    private final MlModelLoader modelLoader;
    private final TrainingStatus status = new TrainingStatus();

    public TrainingService(ModelMetricsStore metricsStore, RealTrainer realTrainer, MlModelLoader modelLoader) {
        this.metricsStore = metricsStore;
        this.realTrainer = realTrainer;
        this.modelLoader = modelLoader;
    }

    public TrainingStatus getStatus() {
        return status;
    }

    public boolean isAnyModelTrained() {
        return metricsStore.count() > 0;
    }

    public List<ModelMetrics> allMetricsByF1Desc() {
        return metricsStore.findAllByOrderByF1Desc();
    }

    /**
     * One row per model and source (web / command line): the latest run, best macro-F1 first. Every older run
     * stays in the file (see history), it is only not shown twice in the comparison.
     */
    public List<ModelMetrics> latestRuns() {

        java.util.Map<String, ModelMetrics> latest = new java.util.LinkedHashMap<>();
        for (ModelMetrics m : metricsStore.findAllByOrderByF1Desc()) {
            String key = m.getModelName() + "|" + m.getTrainedBy();
            ModelMetrics current = latest.get(key);
            if (current == null || m.getTrainedAt().isAfter(current.getTrainedAt())) {
                latest.put(key, m);
            }
        }
        List<ModelMetrics> rows = new java.util.ArrayList<>(latest.values());
        rows.sort(java.util.Comparator.comparingDouble(ModelMetrics::getMacroF1).reversed());
        return rows;
    }

    /** All the runs of one model, newest first. */
    public List<ModelMetrics> history(String modelName) {

        List<ModelMetrics> rows = new java.util.ArrayList<>();
        for (ModelMetrics m : metricsStore.findAllByOrderByF1Desc()) {
            if (m.getModelName().equals(modelName)) {
                rows.add(m);
            }
        }
        rows.sort(java.util.Comparator.comparing(ModelMetrics::getTrainedAt).reversed());
        return rows;
    }

    public ModelMetrics latestFor(String modelName) {
        return metricsStore.findFirstByModelNameOrderByTrainedAtDesc(modelName).orElse(null);
    }

    public synchronized void startTraining(String modelName) {
        if (status.running) return;
        status.reset();
        status.running = true;
        Thread t = new Thread(() -> runTraining(modelName));
        t.setDaemon(true);
        t.start();
    }

    private void runTraining(String modelName) {
        try {
            RealTrainer.Outcome outcome;

            // the model name from the classifier registry ("decision_tree", ...)
            // decides which real training routine to run
            if (modelName.equals("decision_tree")) {
                outcome = realTrainer.trainDecisionTree(status);
            } else if (modelName.equals("logistic_regression")) {
                outcome = realTrainer.trainLogisticRegression(status);
            } else if (modelName.equals("neural_network")) {
                outcome = realTrainer.trainNeuralNetwork(status);
            } else if (modelName.equals("retrieval_embedding")) {
                outcome = realTrainer.trainRetrieval(status);
            } else {
                throw new IllegalArgumentException(modelName + " has no training routine");
            }

            metricsStore.save(toModelMetrics(modelName, outcome));

            // the classifiers hold onto MlModelLoader, not a snapshot of the model, so
            // this reload is picked up on the very next prediction
            modelLoader.reload();

            status.currentTask = "Done.";
            status.percent = 100;
            status.log.add("[100%] Done. Chosen configuration: " + outcome.chosenConfig);
            status.done = true;

        } catch (Exception e) {
            status.error = e.getMessage() != null ? e.getMessage() : e.toString();
        } finally {
            status.running = false;
        }
    }

    private ModelMetrics toModelMetrics(String modelName, RealTrainer.Outcome outcome) {

        ModelMetrics m = new ModelMetrics();
        m.setModelName(modelName);
        m.setVersionName(modelName);
        m.setFull(true); // always the full dataset now, see class comment above
        m.setF1(outcome.f1);
        m.setMacroF1(outcome.macroF1);
        m.setFeatureConfig(outcome.featureConfig);
        m.setPrecision(outcome.precision);
        m.setRecall(outcome.recall);
        m.setBalancedAccuracy(outcome.balancedAccuracy);
        m.setDecisionThreshold(outcome.threshold);
        m.setEvaluatedOnTest(outcome.evaluatedOnTest);
        m.setNTestPairs(outcome.nTestPairs);
        m.setDifficultyBreakdownCsv(outcome.difficultyBreakdownCsv);
        m.setTrainedBy("web");
        m.setChosenConfig(outcome.chosenConfig);
        m.setGridResultsCsv(String.join(";", outcome.gridResults));
        m.setCurveXLabel(outcome.curveXLabel);
        m.setCurveYLabel(outcome.curveYLabel);
        m.setCurveXCsv(joinDoubles(outcome.curveX));
        m.setCurveYCsv(joinDoubles(outcome.curveY));
        m.setCurveY2Label(outcome.curveY2Label);
        m.setCurveY2Csv(joinDoubles(outcome.curveY2));
        return m;
    }

    private String joinDoubles(List<Double> values) {

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(values.get(i));
        }
        return sb.toString();
    }
}
