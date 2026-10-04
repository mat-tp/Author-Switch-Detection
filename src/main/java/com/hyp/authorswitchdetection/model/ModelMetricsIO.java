package com.hyp.authorswitchdetection.model;

import com.hyp.authorswitchdetection.service.JsonLine;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads and writes the "data/model-metrics.json" JSON-lines file that backs
 * the Results page.
 *
 * Deliberately Spring-free (no annotations, no injected config) so it can be
 * called from two completely different places:
 *   - com.hyp.authorswitchdetection.service.ModelMetricsStore, the Spring
 *     @Service used by the web app (Train / Results pages), and
 *   - the standalone command-line Main.java (no Spring container at all),
 *     so a model trained with "./scripts/build.sh" shows up on the Results
 *     page exactly like one trained from the "Train" page in the browser,
 *     without needing Maven or a running server to produce it.
 *
 * Every read re-parses the file from disk (it is a handful of KB at most),
 * so a run appended by a *different* JVM process — e.g. Main.java run from
 * the terminal while the web app is already up — is picked up immediately,
 * no restart required.
 */
public final class ModelMetricsIO {

    private ModelMetricsIO() {
        // utility class, not meant to be instantiated
    }

    /** Reads every training run currently on disk. Empty list if the file doesn't exist yet. */
    public static synchronized List<ModelMetrics> loadAll(String filePath) {

        List<ModelMetrics> metrics = new ArrayList<>();
        File file = new File(filePath);
        if (!file.exists()) {
            return metrics;
        }

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {

                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                metrics.add(parseMetrics(line));
            }

        } catch (IOException e) {
            System.err.println("Error reading model metrics file: " + filePath);
            System.err.println(e.getMessage());
        }

        return metrics;
    }

    /**
     * Appends one training run to disk (never edits or removes existing
     * rows) and assigns it the next id, computed from whatever is already on
     * disk so two different processes writing at different times still get
     * increasing ids.
     */
    public static synchronized ModelMetrics append(String filePath, ModelMetrics m) {

        long nextId = 1;
        for (ModelMetrics existing : loadAll(filePath)) {
            if (existing.getId() != null && existing.getId() >= nextId) {
                nextId = existing.getId() + 1;
            }
        }
        m.setId(nextId);

        File file = new File(filePath);
        File parent = file.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(
                new OutputStreamWriter(new FileOutputStream(file, true), StandardCharsets.UTF_8))) {

            writer.println(toJsonLine(m));

        } catch (IOException e) {
            System.err.println("Error writing model metrics file: " + filePath);
            System.err.println(e.getMessage());
        }

        return m;
    }

    private static String toJsonLine(ModelMetrics m) {

        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"id\":").append(m.getId()).append(",");
        sb.append("\"modelName\":\"").append(JsonLine.escape(m.getModelName())).append("\",");
        sb.append("\"versionName\":\"").append(JsonLine.escape(m.getVersionName())).append("\",");
        sb.append("\"isFull\":").append(m.isFull()).append(",");
        sb.append("\"nProblems\":").append(m.getNProblems()).append(",");
        sb.append("\"f1\":").append(m.getF1()).append(",");
        sb.append("\"precision\":").append(m.getPrecision()).append(",");
        sb.append("\"recall\":").append(m.getRecall()).append(",");
        sb.append("\"balancedAccuracy\":").append(m.getBalancedAccuracy()).append(",");
        sb.append("\"decisionThreshold\":").append(m.getDecisionThreshold()).append(",");
        sb.append("\"evaluatedOnTest\":").append(m.isEvaluatedOnTest()).append(",");
        sb.append("\"nTestPairs\":").append(m.getNTestPairs()).append(",");
        sb.append("\"difficultyBreakdownCsv\":\"").append(JsonLine.escape(nullToEmpty(m.getDifficultyBreakdownCsv()))).append("\",");
        sb.append("\"testF1Mean\":").append(m.getTestF1Mean()).append(",");
        sb.append("\"testF1Std\":").append(m.getTestF1Std()).append(",");
        sb.append("\"trainF1Mean\":").append(m.getTrainF1Mean()).append(",");
        sb.append("\"trainF1Std\":").append(m.getTrainF1Std()).append(",");
        sb.append("\"trainedAt\":\"").append(m.getTrainedAt()).append("\",");
        sb.append("\"trainedBy\":\"").append(JsonLine.escape(nullToEmpty(m.getTrainedBy()))).append("\",");
        sb.append("\"chosenConfig\":\"").append(JsonLine.escape(nullToEmpty(m.getChosenConfig()))).append("\",");
        sb.append("\"gridResultsCsv\":\"").append(JsonLine.escape(nullToEmpty(m.getGridResultsCsv()))).append("\",");
        sb.append("\"curveXLabel\":\"").append(JsonLine.escape(nullToEmpty(m.getCurveXLabel()))).append("\",");
        sb.append("\"curveYLabel\":\"").append(JsonLine.escape(nullToEmpty(m.getCurveYLabel()))).append("\",");
        sb.append("\"curveXCsv\":\"").append(JsonLine.escape(nullToEmpty(m.getCurveXCsv()))).append("\",");
        sb.append("\"curveYCsv\":\"").append(JsonLine.escape(nullToEmpty(m.getCurveYCsv()))).append("\",");
        sb.append("\"curveY2Label\":\"").append(JsonLine.escape(nullToEmpty(m.getCurveY2Label()))).append("\",");
        sb.append("\"curveY2Csv\":\"").append(JsonLine.escape(nullToEmpty(m.getCurveY2Csv()))).append("\"");
        sb.append("}");
        return sb.toString();
    }

    private static ModelMetrics parseMetrics(String line) {

        ModelMetrics m = new ModelMetrics();
        m.setId(JsonLine.extractLong(line, "id"));
        m.setModelName(JsonLine.extractString(line, "modelName"));
        m.setVersionName(JsonLine.extractString(line, "versionName"));
        m.setFull(JsonLine.extractBoolean(line, "isFull", true));
        m.setNProblems(JsonLine.extractInteger(line, "nProblems"));
        m.setF1(JsonLine.extractDouble(line, "f1"));
        m.setPrecision(JsonLine.extractDouble(line, "precision"));
        m.setRecall(JsonLine.extractDouble(line, "recall"));
        Double balancedAccuracy = JsonLine.extractDouble(line, "balancedAccuracy");
        m.setBalancedAccuracy(balancedAccuracy == null ? 0.0 : balancedAccuracy);
        m.setDecisionThreshold(JsonLine.extractDouble(line, "decisionThreshold"));
        m.setEvaluatedOnTest(JsonLine.extractBoolean(line, "evaluatedOnTest", false));
        m.setNTestPairs(JsonLine.extractInteger(line, "nTestPairs"));
        m.setDifficultyBreakdownCsv(JsonLine.extractString(line, "difficultyBreakdownCsv"));
        m.setTestF1Mean(JsonLine.extractDouble(line, "testF1Mean"));
        m.setTestF1Std(JsonLine.extractDouble(line, "testF1Std"));
        m.setTrainF1Mean(JsonLine.extractDouble(line, "trainF1Mean"));
        m.setTrainF1Std(JsonLine.extractDouble(line, "trainF1Std"));
        m.setTrainedAt(LocalDateTime.parse(JsonLine.extractString(line, "trainedAt")));
        m.setTrainedBy(JsonLine.extractString(line, "trainedBy"));
        m.setChosenConfig(JsonLine.extractString(line, "chosenConfig"));
        m.setGridResultsCsv(JsonLine.extractString(line, "gridResultsCsv"));
        m.setCurveXLabel(JsonLine.extractString(line, "curveXLabel"));
        m.setCurveYLabel(JsonLine.extractString(line, "curveYLabel"));
        m.setCurveXCsv(JsonLine.extractString(line, "curveXCsv"));
        m.setCurveYCsv(JsonLine.extractString(line, "curveYCsv"));
        m.setCurveY2Label(JsonLine.extractString(line, "curveY2Label"));
        m.setCurveY2Csv(JsonLine.extractString(line, "curveY2Csv"));
        return m;
    }

    private static String nullToEmpty(String s) {
        return s == null ? "" : s;
    }
}
