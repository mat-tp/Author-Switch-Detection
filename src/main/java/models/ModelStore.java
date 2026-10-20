package models;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OptionalDataException;
import java.io.StreamCorruptedException;

import features.FeatureExtractor;

/**
 * Saving and loading of the trained models, ONE format for the command line (Main), the
 * web app (RealTrainer / MlModelLoader), the ablation study and the TIRA runner.
 *
 * The file is a Java object stream (like the first version) with a small header in front:
 *
 * UTF "ASD-MODEL" | int format | UTF type | UTF description | object model | double threshold | object extractor
 *
 * type: "tree", "logistic", "neural" (or "retrieval", see RetrievalMain).
 * The extractor is the FITTED one: it carries the feature settings (casing, n-gram orders,
 * bucket sizes, pair representation), the z-score parameters and the TF-IDF table of the
 * training data, so a loaded model computes exactly the features it was trained on.
 *
 * Why a header: the models saved by the first version had no header and another feature
 * layout; opening one of those used to fail somewhere inside the web app, or worse,
 * give wrong predictions. Now it fails here, with the sentence "retrain the model".
 */
public final class ModelStore {

    public static final String TREE = "tree";
    public static final String LOGISTIC = "logistic";
    public static final String NEURAL = "neural";
    public static final String RETRIEVAL = "retrieval";

    private static final String MAGIC = "ASD-MODEL";
    private static final int FORMAT = 3;

    private ModelStore() {
        // utility class, not meant to be instantiated
    }

    /** A model that was loaded from a file, everything needed to predict with it. */
    public static class Loaded {

        public final String type;
        public final String description;
        public final Object model; // DecisionTree, double[] (weights + bias), NeuralNetwork, EmbeddingNetwork
        public final double threshold; // probability (or, for the retrieval model, distance) from which a pair is a change
        public final FeatureExtractor extractor;

        Loaded(String type, String description, Object model, double threshold, FeatureExtractor extractor) {
            this.type = type;
            this.description = description;
            this.model = model;
            this.threshold = threshold;
            this.extractor = extractor;
        }

        /** Probability of a change for every row of a (standardised) pair feature matrix. */
        public double[] probabilities(double[][] pairMatrix) {

            switch (type) {
                case TREE:
                    return ((DecisionTree) model).predictProbabilities(pairMatrix);
                case LOGISTIC:
                    return LogisticRegression.predictProbabilities(pairMatrix, (double[]) model);
                case NEURAL:
                    return ((NeuralNetwork) model).predictProbabilities(pairMatrix);
                default:
                    throw new UnsupportedOperationException(
                            "a " + type + " model does not score pair feature vectors");
            }
        }

        /** Probability of a change for ONE pair feature vector. */
        public double probability(double[] pairFeatures) {
            return probabilities(new double[][] { pairFeatures })[0];
        }

        /** 0 = no change, 1 = change, with the threshold that was chosen on the validation data. */
        public int predict(double[] pairFeatures) {
            return probability(pairFeatures) > threshold ? 1 : 0;
        }
    }

    /**
     * Saves a trained model together with its fitted extractor.
     *
     * @param path        the file (the folder is created)
     * @param type        TREE, LOGISTIC, NEURAL or RETRIEVAL
     * @param model       the trained model object
     * @param threshold   the decision threshold chosen on the validation data (0.5 when none)
     * @param extractor   the extractor that was fitted on the training data
     * @param description one line: the settings of the run (shown by the web app)
     */
    public static void save(String path, String type, Object model, double threshold, FeatureExtractor extractor,
            String description) throws IOException {

        File parent = new File(path).getAbsoluteFile().getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }

        try (FileOutputStream fileOut = new FileOutputStream(path);
                BufferedOutputStream bos = new BufferedOutputStream(fileOut);
                ObjectOutputStream oos = new ObjectOutputStream(bos)) {

            oos.writeUTF(MAGIC);
            oos.writeInt(FORMAT);
            oos.writeUTF(type);
            oos.writeUTF(description == null ? "" : description);
            oos.writeObject(model);
            oos.writeDouble(threshold);
            oos.writeObject(extractor);
        }
    }

    public static Loaded load(String path) throws IOException, ClassNotFoundException {

        try (InputStream in = new FileInputStream(path)) {
            return load(in);
        }
    }

    /**
     * Loads a model.
     *
     * @throws IOException with a readable message when the file was not saved by this version
     *                     (no header, another format) or when the features it was trained with
     *                     no longer exist in the code - the model has to be trained again then
     */
    public static Loaded load(InputStream stream) throws IOException, ClassNotFoundException {

        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(stream))) {

            String magic;
            try {
                magic = ois.readUTF();
            } catch (OptionalDataException | StreamCorruptedException notOurs) {
                throw new IOException("this model file has the old format (saved before the feature set was "
                        + "extended) - retrain the model");
            }

            if (!MAGIC.equals(magic)) {
                throw new IOException("not a model file of this project");
            }

            int format = ois.readInt();
            if (format != FORMAT) {
                throw new IOException("model file format " + format + " is not supported (expected " + FORMAT
                        + ") - retrain the model");
            }

            String type = ois.readUTF();
            String description = ois.readUTF();

            Object model;
            FeatureExtractor extractor;
            double threshold;

            try {
                model = ois.readObject();
                threshold = ois.readDouble();
                extractor = (FeatureExtractor) ois.readObject();

            } catch (java.io.InvalidClassException changed) {
                throw new IOException("the model was saved by another version of the code (" + changed.getMessage()
                        + ") - retrain the model");
            }

            try {
                extractor.requireCompatible();
            } catch (IllegalStateException incompatible) {
                throw new IOException(incompatible.getMessage());
            }

            return new Loaded(type, description, model, threshold, extractor);
        }
    }
}
