/**
 *
 * @author Tshephang Matlala
 * @version 01
 */

package models;

import java.io.Serializable;
import java.util.Arrays;

public class DecisionTree implements Serializable {

    /**
     * This class creates a Decision Tree implementation.
     */
    public static class Node implements Serializable {

        /**
         * Internal node contains a feature index and threshold used to split samples.
         */
        private static final long serialVersionUID = 1L;

        int featureIndex;
        double threshold;
        Node left;
        Node right;
        int predictedClassLabel; // majority class

        /**
         * Internal node constructor.
         *
         * @param featureIndex feature used for splitting
         * @param threshold    threshold used for splitting
         */
        public Node(int featureIndex, double threshold) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
        }

        /**
         * Internal node constructor with children.
         *
         * @param featureIndex feature used for splitting
         * @param threshold    threshold used for splitting
         * @param left         left child
         * @param right        right child
         */
        public Node(int featureIndex, double threshold, Node left, Node right) {

            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }

        /**
         * Leaf node constructor.
         *
         * @param predictedClassLabel majority class
         */
        public Node(int predictedClassLabel) {
            this.predictedClassLabel = predictedClassLabel;
        }

        /**
         * Checks whether this node is a leaf node.
         *
         * @return true if this node has no children
         */
        public boolean isLeafNode() {
            return left == null && right == null;
        }
    }

    private static final long serialVersionUID = 1L;

    // Global Parameters
    // Default values of the pre-pruning parameters.
    public static final int MAX_DEPTH = 5;
    public static final int MIN_SAMPLES_SPLIT = 20;
    public static final int MIN_SAMPLES_LEAF = 5;
    public static final double MIN_IMPURITY_DECREASE = 0.0;

    /*
     * TODO: Feature learning to see which performs better.
     */

    // Root node of the decision tree
    private Node root;

    // Attributes of the model
    // The training data is only needed while the tree is being built, so it is
    // transient (same as NeuralNetwork): otherwise the saved .ser contains the
    // WHOLE training matrix (GBs for a big run) and the web app can't load it.
    transient double[][] featureMatrix; // X
    transient int[] truthValues; // Y
    int numFeatures;

    // Parameters used by this tree
    private int maxDepth = MAX_DEPTH; // no node is split at or below this depth
    private int minSamplesSplit = MIN_SAMPLES_SPLIT; // a node needs at least this many samples to be split
    private int minSamplesLeaf = MIN_SAMPLES_LEAF; // every child of a split keeps at least this many samples
    private double minImpurityDecrease = MIN_IMPURITY_DECREASE; // a split must improve the impurity by more than this

    // Class imbalance handling, the weight of one sample for each class
    private boolean useClassWeights = true;
    private double zerosWeight = 1.0;
    private double onesWeight = 1.0;

    // Debugging the decision tree
    private boolean verbose = true;

    // Tolerance threshold
    double epsilon = 1e-9;

    /**
     * Constructor for the decision tree.
     *
     * Uses the default pre pruning parameters.
     *
     * @param featureMatrix feature matrix
     * @param truthValues   truth values for each sample
     */
    public DecisionTree(double[][] featureMatrix, int[] truthValues) {

        validateTrainingData(featureMatrix, truthValues);

        this.featureMatrix = featureMatrix;
        this.truthValues = truthValues;
        this.numFeatures = featureMatrix[0].length;
    }

    /**
     * Constructor for the decision tree with pre-pruning parameters.
     * To allow for feature learning (which depth is perfom)
     *
     * @param featureMatrix feature matrix (X-values)
     * @param truthValues truth values for each sample (y-values)
     * @param maxDepth            maximum depth of the tree
     * @param minSamplesSplit     minimum number of samples a node needs to be split
     * @param minSamplesLeaf      minimum number of samples in each child of a split
     * @param minImpurityDecrease minimum impurity improvement needed to accept a split
     * 
     */
    public DecisionTree(double[][] featureMatrix, int[] truthValues, int maxDepth, int minSamplesSplit,
            int minSamplesLeaf, double minImpurityDecrease) {

        this(featureMatrix, truthValues);

        if (maxDepth < 1) {
            throw new IllegalArgumentException("maxDepth must be at least 1");
        }

        if (minSamplesSplit < 2) {
            throw new IllegalArgumentException("minSamplesSplit must be at least 2");
        }

        if (minSamplesLeaf < 1) {
            throw new IllegalArgumentException("minSamplesLeaf must be at least 1");
        }

        if (minImpurityDecrease < 0.0) {
            throw new IllegalArgumentException("minImpurityDecrease must not be negative");
        }

        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.minImpurityDecrease = minImpurityDecrease;
    }

    /**
     * Ensure that the training data is good, so there's no bad input preventing
     * producing a wrong tree.
     *
     * @param featureMatrix feature matrix
     * @param truthValues   truth values for each sample
     */
    private void validateTrainingData(double[][] featureMatrix, int[] truthValues) {

        // checking null values And ensure all the features are valid not null.
        if (featureMatrix == null || truthValues == null) {
            throw new IllegalArgumentException("featureMatrix and truthValues must not be null");
        }

        if (featureMatrix.length == 0) {
            throw new IllegalArgumentException("featureMatrix is empty");
        }

        if (featureMatrix.length != truthValues.length) {
            throw new IllegalArgumentException("featureMatrix and truthValues must have the same length");
        }

        if (featureMatrix[0] == null) {
            throw new IllegalArgumentException("Sample 0 is null");
        }

        int expectedFeatures = featureMatrix[0].length;

        for (int i = 0; i < featureMatrix.length; i++) {

            if (featureMatrix[i] == null || featureMatrix[i].length != expectedFeatures) {
                throw new IllegalArgumentException("Sample " + i + " is null or has a different number of features");
            }

            for (int j = 0; j < expectedFeatures; j++) {

                if (Double.isNaN(featureMatrix[i][j]) || Double.isInfinite(featureMatrix[i][j])) {
                    throw new IllegalArgumentException("Sample " + i + ", feature " + j + " is NaN or infinite");
                }
            }

            if (truthValues[i] != 0 && truthValues[i] != 1) {
                throw new IllegalArgumentException("Sample " + i + " has an unexpected truth value: " + truthValues[i]);
            }
        }
    }

    /**
     * Turns the class weighting for imbalanced data, given the imbalanceness in the
     * authorship data.
     *
     * @param useClassWeights true to weight each class inversely to its frequency
     */
    public void setUseClassWeights(boolean useClassWeights) {
        this.useClassWeights = useClassWeights;
    }

    /**
     * Turns the printing of the tree building steps on or off.
     *
     * @param verbose true to print every node while the tree is built
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * Calculates the weight of one sample of each class.
     *
     * "Balanced" weighting: weight = total samples / (2 * samples of that class),
     * so both classes carry the same total weight, no matter how rare a style
     * change is.
     * Without weighting, every sample has a weight of 1.
     */
    private void calcClassWeights() {

        int onesCount = 0;
        int zerosCount = 0;

        for (int i = 0; i < truthValues.length; i++) {

            if (truthValues[i] == 1) {
                onesCount++;

            } else {
                zerosCount++;
            }
        }

        // Weighting needs both classes to be present
        if (useClassWeights && onesCount > 0 && zerosCount > 0) {

            int totalCount = onesCount + zerosCount;

            zerosWeight = (double) totalCount / (2.0 * zerosCount);
            onesWeight = (double) totalCount / (2.0 * onesCount);

        } else {

            zerosWeight = 1.0;
            onesWeight = 1.0;
        }

        if (verbose) {
            System.out.println("Class counts: zeros=" + zerosCount + ", ones=" + onesCount
                    + " | weights: zeros=" + zerosWeight + ", ones=" + onesWeight);
        }
    }

    /**
     * Calculates the impurity based on the truth values of the complete dataset.
     *
     * True uses Entropy False uses Gini Index.
     *
     * @param isUsingEntropyt true for Entropy, false for Gini Index
     * @return impurity score
     */
    private double calcImpurity(boolean isUsingEntropyt) {

        // Test edge cases
        if (truthValues.length == 0) {
            throw new IllegalStateException("Cannot calculate impurity for empty truth values");
        }

        // Counts
        int onesCount = 0;
        int zerosCount = 0;

        for (int i = 0; i < truthValues.length; i++) {

            if (truthValues[i] == 1) {
                onesCount++;

            } else if (truthValues[i] == 0) {
                zerosCount++;

            } else {
                System.err.println("Unexpected truth value is found: " + truthValues[i]);
            }
        }

        int totalCount = onesCount + zerosCount;

        if (totalCount == 0) {
            throw new IllegalStateException("No valid truth values were found");
        }

        // Calculate proportions
        double ones_prop = (double) onesCount / totalCount;
        double zeros_prop = (double) zerosCount / totalCount;

        // Simple check
        if (Math.abs((ones_prop + zeros_prop) - 1.0) > epsilon) {

            System.err.println("Ones and zeros: " + (ones_prop + zeros_prop));
        }

        // Entropy
        if (isUsingEntropyt) {

            double entropy = 0.0;

            // Checkin
            if (ones_prop > 0) {
                entropy -= ones_prop * Math.log(ones_prop);
            }

            if (zeros_prop > 0) {
                entropy -= zeros_prop * Math.log(zeros_prop);
            }

            return entropy;

        } else {

            // Gini Index
            return 1.0 - Math.pow(ones_prop, 2) - Math.pow(zeros_prop, 2);
        }
    }

    /**
     * Overloaded method for calculating the impurity of the complete dataset using
     * Gini Index.
     *
     * @return impurity score
     */
    private double calcImpurity() {
        return calcImpurity(false);
    }

    /**
     * Calculates the impurity for a subset of samples.
     *
     * The samples array contains indices referring to
     * the original truthValues array.
     *
     * @param samples         sample indices
     * @param isUsingEntropyt true for Entropy, false for Gini Index
     * @return impurity score
     */
    private double calcImpurity(int[] samples, boolean isUsingEntropyt) {

        // Test edge cases
        if (samples == null || samples.length == 0) {
            throw new IllegalStateException("Cannot calculate impurity for empty samples");
        }

        // Counts
        int onesCount = 0;
        int zerosCount = 0;

        for (int i = 0; i < samples.length; i++) {

            int sample = samples[i];

            if (truthValues[sample] == 1) {
                onesCount++;

            } else if (truthValues[sample] == 0) {
                zerosCount++;

            } else {
                System.err.println("Unexpected truth value is found: " + truthValues[sample]);
            }
        }

        int totalCount = onesCount + zerosCount;

        if (totalCount == 0) {
            throw new IllegalStateException("No valid truth values were found");
        }

        // Calculate proportions
        double ones_prop = (double) onesCount / totalCount;
        double zeros_prop = (double) zerosCount / totalCount;

        // Entropy
        if (isUsingEntropyt) {

            double entropy = 0.0;

            if (ones_prop > 0) {
                entropy -= ones_prop * Math.log(ones_prop);
            }

            if (zeros_prop > 0) {
                entropy -= zeros_prop * Math.log(zeros_prop);
            }

            return entropy;

        } else {

            // Gini Index
            return 1.0 - Math.pow(ones_prop, 2) - Math.pow(zeros_prop, 2);
        }
    }

    /**
     * Overloaded method for calculating the impurity
     * of a subset using Gini Index.
     *
     * @param samples sample indices
     * @return impurity score
     */
    private double calcImpurity(int[] samples) {
        return calcImpurity(samples, false);
    }

    /**
     * Splits the dataset based on the current feature and threshold.
     *
     * split_array[i] == 0 means the sample goes LEFT.
     * split_array[i] == 1 means the sample goes RIGHT.
     *
     * @param samples      sample indices
     * @param featureIndex feature used for splitting
     * @param threshold    threshold used for splitting
     * @return array containing left/right split flags
     */
    public int[] splitDataset(int[] samples, int featureIndex, double threshold) {

        int[] split_array = new int[samples.length];

        int leftCount = 0;
        int rightCount = 0;

        for (int i = 0; i < samples.length; i++) {

            int sample = samples[i];

            if (featureMatrix[sample][featureIndex] <= threshold) {

                // LEFT
                split_array[i] = 0;
                leftCount++;

            } else {

                // RIGHT
                split_array[i] = 1;
                rightCount++;
            }
        }

        return split_array;
    }

    /**
     * Extracts the sample indices matching a given split flag.
     *
     * @param samples     sample indices that were split [[],[],[],...]
     * @param split_array 0/1 flags from splitDataset [0,1,0,...]
     * @param sideFlag    0 for left, 1 for right
     * @return subset of samples matching flag
     */
    private int[] filterByFlag(int[] samples, int[] split_array, int sideFlag) {

        if (samples.length != split_array.length) {
            throw new IllegalArgumentException("samples and split_array must have the same length");
        }

        // determining the required size
        int count = 0;

        for (int i = 0; i < split_array.length; i++) {

            if (split_array[i] == sideFlag) {
                count++;
            }
        }

        // Create result array
        int[] result = new int[count];

        int pos = 0;

        for (int i = 0; i < split_array.length; i++) {

            if (split_array[i] == sideFlag) {
                result[pos] = samples[i];
                pos++;
            }
        }

        return result;
    }

    /**
     * Calculates the majority class among the given samples.
     * Each sample counts with its class weight, so a rare class is not outvoted.
     *
     * @param samples sample indices
     * @return majority class, either 0 or 1
     */
    private int calcMajorityClassLabel(int[] samples) {

        if (samples == null || samples.length == 0) {
            throw new IllegalStateException("Cannot calculate majority class from empty samples");
        }

        int onesCount = 0;
        int zerosCount = 0;

        for (int i = 0; i < samples.length; i++) {

            int sample = samples[i];

            if (truthValues[sample] == 1) {
                onesCount++;

            } else if (truthValues[sample] == 0) {
                zerosCount++;
            }
        }

        // weighted counts
        double ones_weight = onesCount * onesWeight;
        double zeros_weight = zerosCount * zerosWeight;

        // in case of a tie, class 0 is selected.
        if (ones_weight > zeros_weight) {
            return 1;
        }

        return 0;
    }

    /**
     * Calculates the Gini Index from weighted class counts.
     *
     * @param zeros_weight total weight of the class 0 samples
     * @param ones_weight  total weight of the class 1 samples
     * @return impurity score
     */
    private double calcWeightedImpurity(double zeros_weight, double ones_weight) {

        double total_weight = zeros_weight + ones_weight;

        if (total_weight <= 0.0) {
            return 0.0;
        }

        double ones_prop = ones_weight / total_weight;
        double zeros_prop = zeros_weight / total_weight;

        return 1.0 - Math.pow(ones_prop, 2) - Math.pow(zeros_prop, 2);
    }

    /**
     * Searches all featureMatrix and candidate thresholds to find the split that
     * minimizes weighted impurity.
     *
     * For each feature the samples are sorted once and swept from left to right,
     * keeping running class weights.
     * This gives every possible split in O(n log n) per feature instead of O(n^2).
     * The class weights handle imbalanced data
     *
     * @param samples sample indices to consider
     * @return {featureIndex, threshold}, or null if no useful split was found
     */
    private double[] findBestSplit(int[] samples) {

        int num_samples = samples.length;
        int num_featureMatrix = featureMatrix[0].length;

        // Class weights of this node
        double total_zeros_weight = 0.0;
        double total_ones_weight = 0.0;

        for (int i = 0; i < num_samples; i++) {

            if (truthValues[samples[i]] == 1) {
                total_ones_weight += onesWeight;

            } else {
                total_zeros_weight += zerosWeight;
            }
        }

        double total_weight = total_zeros_weight + total_ones_weight;

        // Impurity before splitting
        double parent_impurity = calcWeightedImpurity(total_zeros_weight, total_ones_weight);

        double best_impurity = Double.MAX_VALUE; // Should be 1, as the highest the impurity can be is 1.
        int best_featureIndex = -1;
        double best_threshold = 0.0;

        // positions of the samples, sorted by the current feature
        Integer[] order = new Integer[num_samples];

        for (int featureIndex = 0; featureIndex < num_featureMatrix; featureIndex++) {

            // each sample is assigned an index
            for (int i = 0; i < num_samples; i++) {
                order[i] = i;
            }

            final int current_feature = featureIndex;
            Arrays.sort(order, (a, b) -> Double.compare(
                    featureMatrix[samples[a]][current_feature],
                    featureMatrix[samples[b]][current_feature]));

            double left_zeros_weight = 0.0;
            double left_ones_weight = 0.0;

            // Try every sample value as a candidate threshold
            // (the samples are sorted, so every gap between two neighbouring values is
            // tried in one sweep)
            for (int k = 0; k < num_samples - 1; k++) {

                // Split the samples: everything up to position k goes LEFT
                int sample = samples[order[k]];

                if (truthValues[sample] == 1) {
                    left_ones_weight += onesWeight;

                } else {
                    left_zeros_weight += zerosWeight;
                }

                double value = featureMatrix[sample][featureIndex];
                double next_value = featureMatrix[samples[order[k + 1]]][featureIndex];

                // skipping invalid splits
                // (equal values can not be separated)
                if (value == next_value) {
                    continue;
                }

                // Extract left and right samples
                int left_count = k + 1;
                int right_count = num_samples - left_count;

                // skipping invalid splits (a leaf smaller than minSamplesLeaf)
                if (left_count < minSamplesLeaf || right_count < minSamplesLeaf) {
                    continue;
                }

                // Calculate child impurities
                double right_zeros_weight = total_zeros_weight - left_zeros_weight;
                double right_ones_weight = total_ones_weight - left_ones_weight;

                double left_weight = left_zeros_weight + left_ones_weight;
                double right_weight = right_zeros_weight + right_ones_weight;

                // Calculate weighted impurity
                double weighted_impurity = (left_weight / total_weight)
                        * calcWeightedImpurity(left_zeros_weight, left_ones_weight)
                        + (right_weight / total_weight)
                                * calcWeightedImpurity(right_zeros_weight, right_ones_weight);

                // updating the best splits
                if (weighted_impurity < best_impurity) {

                    // threshold between the two values, so it generalises better than a sample
                    // value
                    double candidate_threshold = (value + next_value) / 2.0;

                    // rounding can push the middle up to next_value, which would break the "<="
                    // rule
                    if (candidate_threshold >= next_value) {
                        candidate_threshold = value;
                    }

                    best_impurity = weighted_impurity;
                    best_featureIndex = featureIndex;
                    best_threshold = candidate_threshold;
                }
            }
        }

        // No valid split found
        if (best_featureIndex == -1) {
            return null;
        }

        // Split does not improve impurity (by more than minImpurityDecrease)
        if (parent_impurity - best_impurity <= minImpurityDecrease + epsilon) {
            return null;
        }

        return new double[] { best_featureIndex, best_threshold };
    }

    /**
     * Recursively creates a binary decision tree for the given samples.
     *
     * @param samples sample indices belonging to this node
     * @param depth   current depth of the tree
     * @return root node of this subtree
     */
    private Node makeBinaryTree(int[] samples, int depth) {

        // Defining leaf nodes conditions:
        // 1. stopping if the maximum depth has been reached
        if (depth >= maxDepth) {
            logNode(depth, "Leaf (max depth), samples=" + samples.length);
            return new Node(calcMajorityClassLabel(samples)); // leaf node
        }

        // 2. stopping if there are too few samples
        if (samples.length < minSamplesSplit) {
            logNode(depth, "Leaf (too few samples), samples=" + samples.length);
            return new Node(calcMajorityClassLabel(samples)); // leaf node
        }

        // 3. stopping if the node is pure
        if (calcImpurity(samples) < epsilon) {
            logNode(depth, "Leaf (pure node), samples=" + samples.length);
            return new Node(calcMajorityClassLabel(samples));
        }

        double[] split = findBestSplit(samples);

        // 4. stopping if no split keeps minSamplesLeaf and improves the impurity by
        // minImpurityDecrease
        if (split == null) { // no useful split
            logNode(depth, "Leaf (no useful split), samples=" + samples.length);
            return new Node(calcMajorityClassLabel(samples));
        }

        // split dictionary
        int featureIndex = (int) split[0];
        double threshold = split[1];

        logNode(depth, "Split on feature " + featureIndex + " <= " + threshold + ", samples=" + samples.length);
        int[] split_array = splitDataset(samples, featureIndex, threshold);

        // Extract left and right samples
        int[] left_samples = filterByFlag(samples, split_array, 0);
        int[] right_samples = filterByFlag(samples, split_array, 1);

        // Create internal node
        Node node = new Node(featureIndex, threshold);

        // Recursively create left child
        node.left = makeBinaryTree(left_samples, depth + 1);

        // Recursively create right child
        node.right = makeBinaryTree(right_samples, depth + 1);

        return node;
    }

    /**
     * Prints one line of the tree building steps, if verbose is on.
     *
     * @param depth   current depth of the tree
     * @param message text to print
     */
    private void logNode(int depth, String message) {

        if (verbose) {
            System.out.println("  ".repeat(depth) + "[depth " + depth + "] " + message);
        }
    }

    /**
     * Public entry point for creating the
     * binary decision tree.
     */
    public void makeBinaryTree() {

        // Calculate the class weights, for the imbalanced classes
        calcClassWeights();

        // Create an array containing every sample index
        int[] all_samples = new int[truthValues.length];

        for (int i = 0; i < all_samples.length; i++) {
            all_samples[i] = i;
        }

        // Start building the tree at depth 0
        root = makeBinaryTree(all_samples, 0);
    }

    /**
     * Prints the structure of the trained tree
     * to standard output.
     */
    public void printTree() {
        printTree(root, 0);
    }

    /**
     * Recursively prints a node and its children,
     * indented by depth.
     *
     * @param node  current node
     * @param depth current depth (for indentation)
     */
    private void printTree(Node node, int depth) {

        if (node == null) {
            return;
        }

        String indent = "  ".repeat(depth);

        if (node.isLeafNode()) {

            System.out.println(indent + "Leaf -> class " + node.predictedClassLabel);

        } else {
            System.out.println(indent + "Feature[" + node.featureIndex + "] <= " + node.threshold);

            printTree(node.left, depth + 1);
            printTree(node.right, depth + 1);
        }
    }

    public int predict(double[] sample) {

        if (root == null) {
            throw new IllegalStateException("The tree is not trained. Call makeBinaryTree() first");
        }

        if (sample == null || sample.length != numFeatures) {
            throw new IllegalArgumentException("sample must have " + numFeatures + " features");
        }

        return predict(sample, root);
    }

    public int predict(double[] sample, Node node) {

        if (node.isLeafNode()) {
            return node.predictedClassLabel;
        }

        if (sample[node.featureIndex] <= node.threshold) {
            // System.out.println("Left Node index: " + node.featureIndex);
            return predict(sample, node.left);

        } else {
            // System.out.println("Right Node index: " + node.featureIndex);
            return predict(sample, node.right);
        }
    }

    /**
     * Removes the splits that can never change a prediction: a node whose two
     * children are leaves with the SAME label asks a question whose answer
     * doesn't matter. It is replaced by a single leaf. Repeats upwards, so a
     * whole branch that always ends in the same label collapses too.
     *
     * Lossless: predict() gives exactly the same label for every sample before
     * and after. The tree just gets smaller and easier to read.
     *
     * @return number of splits removed
     */
    public int pruneRedundantSplits() {

        int before = countSplits();
        root = pruneRedundant(root);
        return before - countSplits();
    }

    private Node pruneRedundant(Node node) {

        if (node == null || node.isLeafNode()) {
            return node;
        }

        node.left = pruneRedundant(node.left);
        node.right = pruneRedundant(node.right);

        if (node.left.isLeafNode() && node.right.isLeafNode()
                && node.left.predictedClassLabel == node.right.predictedClassLabel) {
            return new Node(node.left.predictedClassLabel);
        }

        return node;
    }

    /**
     * @return number of yes/no questions (internal nodes) in the tree
     */
    public int countSplits() {
        return countSplits(root);
    }

    private int countSplits(Node node) {
        return (node == null || node.isLeafNode()) ? 0 : 1 + countSplits(node.left) + countSplits(node.right);
    }

    /**
     * One yes/no question the tree asked while classifying a sample.
     */
    public static class DecisionStep {

        public final int featureIndex;
        public final double threshold;
        public final double value; // the sample's value for this feature
        public final boolean wentLeft; // true when value <= threshold

        public DecisionStep(int featureIndex, double threshold, double value, boolean wentLeft) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.value = value;
            this.wentLeft = wentLeft;
        }
    }

    /**
     * The path a sample took through the tree, and the label at the end of it.
     */
    public static class Explanation {

        public final java.util.List<DecisionStep> steps;
        public final int label;

        public Explanation(java.util.List<DecisionStep> steps, int label) {
            this.steps = steps;
            this.label = label;
        }
    }

    /**
     * Classifies a sample AND records every question asked on the way down.
     * Gives exactly the same label as predict(sample); this is what makes each
     * prediction justifiable: "feature X was above Y, so ...".
     *
     * @param sample feature vector
     * @return the path taken and the predicted label
     */
    public Explanation explain(double[] sample) {

        if (root == null) {
            throw new IllegalStateException("The tree is not trained. Call makeBinaryTree() first");
        }

        if (sample == null || sample.length != numFeatures) {
            throw new IllegalArgumentException("sample must have " + numFeatures + " features");
        }

        java.util.List<DecisionStep> steps = new java.util.ArrayList<>();
        Node node = root;

        while (!node.isLeafNode()) {

            double value = sample[node.featureIndex];
            boolean goLeft = value <= node.threshold;
            steps.add(new DecisionStep(node.featureIndex, node.threshold, value, goLeft));
            node = goLeft ? node.left : node.right;
        }

        return new Explanation(steps, node.predictedClassLabel);
    }

    /**
     * The whole tree as readable if / else lines, one per node.
     *
     * @param featureNames name of each feature, by index (a missing name falls
     *                     back to "feature[i]")
     * @return the rules, indented by depth
     */
    public java.util.List<String> toRules(String[] featureNames) {

        java.util.List<String> lines = new java.util.ArrayList<>();
        collectRules(root, 0, featureNames, lines);
        return lines;
    }

    private void collectRules(Node node, int depth, String[] names, java.util.List<String> lines) {

        if (node == null) {
            return;
        }

        String indent = "    ".repeat(depth);

        if (node.isLeafNode()) {
            lines.add(indent + "-> " + (node.predictedClassLabel == 1 ? "AUTHOR SWITCH" : "same author"));
            return;
        }

        String name = (names != null && node.featureIndex < names.length) ? names[node.featureIndex]
                : "feature[" + node.featureIndex + "]";
        String threshold = String.format(java.util.Locale.ROOT, "%.3f", node.threshold);

        lines.add(indent + "if " + name + " <= " + threshold + ":");
        collectRules(node.left, depth + 1, names, lines);
        lines.add(indent + "else:  (" + name + " > " + threshold + ")");
        collectRules(node.right, depth + 1, names, lines);
    }

    /**
     * Predicts the label of every sample in a feature matrix.
     *
     * @param samples feature matrix
     * @return predicted label for each row
     */
    public int[] predictAll(double[][] samples) {

        int[] predictions = new int[samples.length];

        for (int i = 0; i < samples.length; i++) {
            predictions[i] = predict(samples[i]);
        }

        return predictions;
    }

    /**
     * Evaluates the trained tree on a labelled dataset (e.g. the validation split).
     *
     * Style changes are usually the minority class, so accuracy alone can look good
     * while the tree never detects a change. This prints the confusion matrix, the
     * precision,
     * recall and F1 of class 1 (change), and the accuracy of always predicting the
     * majority class.
     *
     * @param testMatrix      feature matrix of the dataset
     * @param testTruthValues truth values of the dataset
     * @return {accuracy, precision, recall, f1, balancedAccuracy}
     */
    public double[] evaluate(double[][] testMatrix, int[] testTruthValues) {

        if (testMatrix == null || testTruthValues == null || testMatrix.length == 0
                || testMatrix.length != testTruthValues.length) {
            throw new IllegalArgumentException("testMatrix and testTruthValues must be non-empty and the same length");
        }

        int[] predictions = predictAll(testMatrix);

        // shared with every other model, so accuracy/precision/recall/F1/balanced
        // accuracy are computed the same way everywhere (see models.Metrics)
        return Metrics.evaluate(predictions, testTruthValues);
    }
}