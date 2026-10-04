package com.hyp.authorswitchdetection.ml;

import features.FeatureExtractor;
import models.DecisionTree;
import models.NeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;

/*
    TODO: (1) Random Forest has no .ser file yet
*/

/**
 * Loading the models used for prediction.
 */
@Component
public class MlModelLoader {

    private static final Logger log = LoggerFactory.getLogger(MlModelLoader.class);

    @Value("${app.trained-models.dir:data/models}")
    private String trainedModelsDir;

    public record DecisionTreeModel(DecisionTree tree, FeatureExtractor extractor, boolean fullFeatures) {
    }

    public record LogisticRegressionModel(double[] weightsBias, double threshold, FeatureExtractor extractor,
            boolean fullFeatures) {
    }

    public record NeuralNetworkModel(NeuralNetwork network, double threshold, FeatureExtractor extractor,
            boolean fullFeatures) {
    }

    // not final: reload() will swap these out once a new model has been trained
    private volatile DecisionTreeModel decisionTree;
    private volatile LogisticRegressionModel logisticRegression;
    private volatile NeuralNetworkModel neuralNetwork;

    public MlModelLoader() {
        reload();
    }

    /** Re-reads all 3 models, called again by TrainingService once a run finishes. */
    public void reload() {
        this.decisionTree = loadDecisionTree("decisionTree.ser");
        this.logisticRegression = loadLogisticRegression("logisticRegression.ser");
        this.neuralNetwork = loadNeuralNetwork("neuralNetwork.ser");
    }

    public DecisionTreeModel decisionTree() {
        return decisionTree;
    }

    public LogisticRegressionModel logisticRegression() {
        return logisticRegression;
    }

    public NeuralNetworkModel neuralNetwork() {
        return neuralNetwork;
    }

    /**
     * Loading the decision tree + its extractor, same order Main.trainDecisionTree
     * (and RealTrainer.trainDecisionTree) write them in.
     */
    private DecisionTreeModel loadDecisionTree(String fileName) {

        try (ObjectInputStream ois = openObjectStream(fileName)) {

            DecisionTree tree = (DecisionTree) ois.readObject();
            FeatureExtractor extractor = (FeatureExtractor) ois.readObject();
            boolean fullFeatures = readFullFeaturesFlag(ois);

            // drop the questions whose answer never changes the label (lossless, same predictions)
            int removed = tree.pruneRedundantSplits();

            log.info("Loaded decision tree model from {} ({} redundant splits removed, {} left)", describeSource(fileName),
                    removed, tree.countSplits());
            return new DecisionTreeModel(tree, extractor, fullFeatures);

        } catch (Exception e) {

            log.warn("Could not load decision tree model ({}): {}", fileName, e.getMessage());
            return null;
        }
    }

    /**
     * Loading the logistic regression weights + bias, the chosen threshold, then
     * the extractor, same order Main.trainLogisticRegression writes them in.
     */
    private LogisticRegressionModel loadLogisticRegression(String fileName) {

        try (ObjectInputStream ois = openObjectStream(fileName)) {

            double[] weightsBias = (double[]) ois.readObject();
            double threshold = ois.readDouble();
            FeatureExtractor extractor = (FeatureExtractor) ois.readObject();
            boolean fullFeatures = readFullFeaturesFlag(ois);

            log.info("Loaded logistic regression model from {}", describeSource(fileName));
            return new LogisticRegressionModel(weightsBias, threshold, extractor, fullFeatures);

        } catch (Exception e) {

            log.warn("Could not load logistic regression model ({}): {}", fileName, e.getMessage());
            return null;
        }
    }

    /**
     * Loading the network, the chosen threshold, then the extractor, same order
     * Main.trainNeuralNetwork writes them in.
     */
    private NeuralNetworkModel loadNeuralNetwork(String fileName) {

        try (ObjectInputStream ois = openObjectStream(fileName)) {

            NeuralNetwork network = (NeuralNetwork) ois.readObject();
            double threshold = ois.readDouble();
            FeatureExtractor extractor = (FeatureExtractor) ois.readObject();
            boolean fullFeatures = readFullFeaturesFlag(ois);

            log.info("Loaded neural network model from {}", describeSource(fileName));
            return new NeuralNetworkModel(network, threshold, extractor, fullFeatures);

        } catch (Exception e) {

            log.warn("Could not load neural network model ({}): {}", fileName, e.getMessage());
            return null;
        }
    }

    /**
     * Reads a boolean flags for which feature set the model actually uses.
     */
    private boolean readFullFeaturesFlag(ObjectInputStream ois) {
        try {
            return ois.readBoolean();
        } catch (IOException endOfOldFile) {
            return true;
        }
    }

    // a freshly trained model on disk wins over the bundled baseline on the classpath
    private ObjectInputStream openObjectStream(String fileName) throws IOException {

        File localFile = new File(trainedModelsDir, fileName);
        if (localFile.isFile()) {
            return new ObjectInputStream(new BufferedInputStream(new FileInputStream(localFile)));
        }

        InputStream in = getClass().getClassLoader().getResourceAsStream("ml-models/" + fileName);
        if (in == null) {
            throw new IOException("Not found in " + trainedModelsDir + " or on the classpath: " + fileName);
        }

        return new ObjectInputStream(new BufferedInputStream(in));
    }

    private String describeSource(String fileName) {
        File localFile = new File(trainedModelsDir, fileName);
        return localFile.isFile() ? localFile.getPath() : "ml-models/" + fileName + " (bundled baseline)";
    }
}
