package com.hyp.authorswitchdetection.ml;

import models.ModelStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Loading the trained models for prediction, through models.ModelStore (the same code and the same
 * file format the command line uses, so a model trained with Main / RetrievalMain is picked up here).
 *
 * A model file that was saved by an older version of the code (another feature set) is NOT
 * used: problem(type) returns the reason, and the pages show it as a banner ("retrain the model").
 * There is no bundled baseline and no random fallback any more: a prediction is always made by a
 * trained model, or it is not made.
 */
@Component
public class MlModelLoader {

    private static final Logger log = LoggerFactory.getLogger(MlModelLoader.class);

    public static final String TREE = ModelStore.TREE;
    public static final String LOGISTIC = ModelStore.LOGISTIC;
    public static final String NEURAL = ModelStore.NEURAL;
    public static final String RETRIEVAL = ModelStore.RETRIEVAL;

    /** The file names are the ones Main.java / RetrievalMain.java write. */
    private static final Map<String, String> FILES = new LinkedHashMap<>();
    static {
        FILES.put(TREE, "decisionTree.ser");
        FILES.put(LOGISTIC, "logisticRegression.ser");
        FILES.put(NEURAL, "neuralNetwork.ser");
        FILES.put(RETRIEVAL, "retrieval.ser");
    }

    @Value("${app.trained-models.dir:data/models}")
    private String trainedModelsDir;

    // not final: reload() swaps these after a new model has been trained
    private volatile Map<String, ModelStore.Loaded> loaded = new LinkedHashMap<>();
    private volatile Map<String, String> problems = new LinkedHashMap<>();

    /** Loads when first asked (@Value is not set yet in the constructor), and again when a model file changed. */
    private volatile boolean read = false;
    private volatile String stamp = "";

    /** The size and time of change of every model file: a model trained by Main while the app runs is noticed. */
    private String currentStamp() {
        StringBuilder sb = new StringBuilder();
        for (String file : FILES.values()) {
            File f = new File(trainedModelsDir, file);
            sb.append(f.isFile() ? f.lastModified() + ":" + f.length() : "-").append('|');
        }
        return sb.toString();
    }

    /** Re-reads all the models, called again by TrainingService once a run finishes. */
    public synchronized void reload() {

        Map<String, ModelStore.Loaded> newLoaded = new LinkedHashMap<>();
        Map<String, String> newProblems = new LinkedHashMap<>();

        for (Map.Entry<String, String> entry : FILES.entrySet()) {

            File file = new File(trainedModelsDir, entry.getValue());
            if (!file.isFile()) {
                continue; // nothing trained yet, that is not a problem, just a model that is not there
            }

            try {
                newLoaded.put(entry.getKey(), ModelStore.load(file.getPath()));
                log.info("Loaded {} model from {}", entry.getKey(), file.getPath());

            } catch (Exception | LinkageError e) {

                // e.g. "retrain the model" for a file of the old format
                String reason = e.getMessage();
                if (reason == null || e instanceof java.io.EOFException || e instanceof java.io.StreamCorruptedException
                        || e instanceof ClassNotFoundException || e instanceof LinkageError) {
                    reason = "the model file is damaged or was saved by another version of the code - retrain the model";
                }
                newProblems.put(entry.getKey(), reason);
                log.warn("Could not load the {} model ({}): {}", entry.getKey(), file.getPath(), reason);
            }
        }

        this.loaded = newLoaded;
        this.problems = newProblems;
        this.stamp = currentStamp();
        this.read = true;
    }

    private void ensureRead() {
        if (!read || !stamp.equals(currentStamp())) {
            reload();
        }
    }

    /** The loaded model of this type (tree, logistic, neural, retrieval), or null. */
    public ModelStore.Loaded get(String type) {
        ensureRead();
        return loaded.get(type);
    }

    public ModelStore.Loaded decisionTree() {
        return get(TREE);
    }

    public ModelStore.Loaded logisticRegression() {
        return get(LOGISTIC);
    }

    public ModelStore.Loaded neuralNetwork() {
        return get(NEURAL);
    }

    public ModelStore.Loaded retrieval() {
        return get(RETRIEVAL);
    }

    /** Why a model file exists but could not be used (stale, corrupt), null when it loaded or there is no file. */
    public String problem(String type) {
        ensureRead();
        return problems.get(type);
    }

    /** All the problems, type -> message; shown as a banner on the pages. */
    public Map<String, String> allProblems() {
        ensureRead();
        return Map.copyOf(problems);
    }

    /** The file a model of this type is read from. */
    public File fileOf(String type) {
        return new File(trainedModelsDir, FILES.get(type));
    }
}
