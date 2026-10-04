package com.hyp.authorswitchdetection.model;

import java.time.LocalDateTime;

/**
 * One row per training run. Kept intentionally close to what the Train /
 * Results pages already expect (f1, precision, recall, decisionThreshold).
 * Plain POJO, no database — see ModelMetricsStore, which reads/writes these
 * to a JSON file.
 */
public class ModelMetrics {

    private Long id;

    private String modelName;       // e.g. "logistic_regression"
    private String versionName;     // e.g. "logistic_regression_sample_tr_50"

    private boolean isFull = true;
    private Integer nProblems;

    private double f1;
    private double precision;
    private double recall;
    private double balancedAccuracy;
    private double decisionThreshold;

    // true when f1/precision/recall/balancedAccuracy above come from the
    // held-out test split (never used for training, validation or threshold
    // selection); false means they are validation-split numbers, e.g. because
    // no test split was available for that run
    private boolean evaluatedOnTest;
    private Integer nTestPairs;

    // per-difficulty breakdown of the SAME evaluation (test when available,
    // else validation): "easy:n=..,accuracy=..,precision=..,recall=..,f1=..,
    // balancedAccuracy=..;medium:...;hard:...", empty when not computed
    private String difficultyBreakdownCsv;

    // "web" (Train page) or "cli" (standalone Main.java / build.sh) —
    // purely informational, shown on the Results page
    private String trainedBy = "web";

    private Double testF1Mean;
    private Double testF1Std;
    private Double trainF1Mean;
    private Double trainF1Std;

    private LocalDateTime trainedAt = LocalDateTime.now();

    // --- feature-set / hyperparameter search + training curve, from RealTrainer ---
    private String chosenConfig;       // e.g. "full features, learning rate=0.6"
    private String gridResultsCsv;     // every combination tried, one per line, ";" joined
    private String curveXLabel;        // e.g. "iteration", "epoch", "max depth"
    private String curveYLabel;        // e.g. "Training loss", "Validation F1"
    private String curveXCsv;          // "," joined
    private String curveYCsv;          // "," joined
    private String curveY2Label;       // null = only one series on the chart
    private String curveY2Csv;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getVersionName() { return versionName; }
    public void setVersionName(String versionName) { this.versionName = versionName; }

    public boolean isFull() { return isFull; }
    public void setFull(boolean full) { isFull = full; }

    public Integer getNProblems() { return nProblems; }
    public void setNProblems(Integer nProblems) { this.nProblems = nProblems; }

    public double getF1() { return f1; }
    public void setF1(double f1) { this.f1 = f1; }

    public double getPrecision() { return precision; }
    public void setPrecision(double precision) { this.precision = precision; }

    public double getRecall() { return recall; }
    public void setRecall(double recall) { this.recall = recall; }

    public double getBalancedAccuracy() { return balancedAccuracy; }
    public void setBalancedAccuracy(double balancedAccuracy) { this.balancedAccuracy = balancedAccuracy; }

    public boolean isEvaluatedOnTest() { return evaluatedOnTest; }
    public void setEvaluatedOnTest(boolean evaluatedOnTest) { this.evaluatedOnTest = evaluatedOnTest; }

    public Integer getNTestPairs() { return nTestPairs; }
    public void setNTestPairs(Integer nTestPairs) { this.nTestPairs = nTestPairs; }

    public String getDifficultyBreakdownCsv() { return difficultyBreakdownCsv; }
    public void setDifficultyBreakdownCsv(String difficultyBreakdownCsv) { this.difficultyBreakdownCsv = difficultyBreakdownCsv; }

    public String getTrainedBy() { return trainedBy; }
    public void setTrainedBy(String trainedBy) { this.trainedBy = trainedBy; }

    public double getDecisionThreshold() { return decisionThreshold; }
    public void setDecisionThreshold(double decisionThreshold) { this.decisionThreshold = decisionThreshold; }

    public Double getTestF1Mean() { return testF1Mean; }
    public void setTestF1Mean(Double testF1Mean) { this.testF1Mean = testF1Mean; }

    public Double getTestF1Std() { return testF1Std; }
    public void setTestF1Std(Double testF1Std) { this.testF1Std = testF1Std; }

    public Double getTrainF1Mean() { return trainF1Mean; }
    public void setTrainF1Mean(Double trainF1Mean) { this.trainF1Mean = trainF1Mean; }

    public Double getTrainF1Std() { return trainF1Std; }
    public void setTrainF1Std(Double trainF1Std) { this.trainF1Std = trainF1Std; }

    public LocalDateTime getTrainedAt() { return trainedAt; }
    public void setTrainedAt(LocalDateTime trainedAt) { this.trainedAt = trainedAt; }

    public String getChosenConfig() { return chosenConfig; }
    public void setChosenConfig(String chosenConfig) { this.chosenConfig = chosenConfig; }

    public String getGridResultsCsv() { return gridResultsCsv; }
    public void setGridResultsCsv(String gridResultsCsv) { this.gridResultsCsv = gridResultsCsv; }

    /** Grid results with one configuration per line, for display in the web app. */
    public String getGridResultsDisplay() {
        return gridResultsCsv == null ? "" : gridResultsCsv.replace(";", "\n");
    }

    public String getCurveXLabel() { return curveXLabel; }
    public void setCurveXLabel(String curveXLabel) { this.curveXLabel = curveXLabel; }

    public String getCurveYLabel() { return curveYLabel; }
    public void setCurveYLabel(String curveYLabel) { this.curveYLabel = curveYLabel; }

    public String getCurveXCsv() { return curveXCsv; }
    public void setCurveXCsv(String curveXCsv) { this.curveXCsv = curveXCsv; }

    public String getCurveYCsv() { return curveYCsv; }
    public void setCurveYCsv(String curveYCsv) { this.curveYCsv = curveYCsv; }

    public String getCurveY2Label() { return curveY2Label; }
    public void setCurveY2Label(String curveY2Label) { this.curveY2Label = curveY2Label; }

    public String getCurveY2Csv() { return curveY2Csv; }
    public void setCurveY2Csv(String curveY2Csv) { this.curveY2Csv = curveY2Csv; }
}