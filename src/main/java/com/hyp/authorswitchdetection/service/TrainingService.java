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
            } else {
                throw new IllegalArgumentException(modelName + " has no training routine yet (Random Forest was "
                        + "never implemented in hyProjectTP either)");
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
            status.error = e.getMessage();
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
