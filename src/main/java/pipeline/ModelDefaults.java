package pipeline;

/**
 * The hyper-parameters of the three pair models, in one place. The command-line trainer, the web trainer and the
 * ablation study all start from these values, so they cannot drift apart.
 *
 * <h2>Why these values</h2>
 *
 * <p>Each sentence pair is described by a vector of about 60 to 130 numbers (63 scalar features per sentence, see
 * {@code FeatureExtractor}; the length of the pair vector depends on the pair representation, for example
 * {@code ABSOLUTE} or {@code CONCAT}). Many of those features are strongly correlated, for example the different
 * word-length and word-count measures. The defaults below are chosen for that situation:</p>
 *
 * <ul>
 *   <li><strong>Decision tree:</strong> a depth of 15 lets the tree combine many features before it must decide,
 *       where a depth of 5 can ask only five questions about 63 features. {@link #TREE_MIN_SAMPLES_SPLIT} and
 *       {@link #TREE_MIN_SAMPLES_LEAF} keep very small leaves from forming. The web trainer compares the depths in
 *       {@link #TREE_DEPTH_GRID} on validation data and takes the shallowest one within
 *       {@link #TREE_SIMPLICITY_TOLERANCE} of the best.</li>
 *   <li><strong>Logistic regression:</strong> a step of 0.1 with a small L2 penalty. On stand-in data the old step of 0.6
 *       made the loss jump (0.70 at the start, 2.6 after 2000 iterations), and 0.1 lowered it steadily to 0.61.
 *       The penalty keeps the weights of correlated features from growing in opposite directions.</li>
 *   <li><strong>Neural network:</strong> a funnel of 128, 64 and 32 units with dropout 0.3, and early stopping on the
 *       validation loss, so a larger network does not simply train longer.</li>
 * </ul>
 *
 * <h2>What has been measured and what has not</h2>
 *
 * <p>These values are reasoned starting points. They have <em>not</em> been compared with the previous defaults on
 * the real PAN data. The earlier neural-network sweeps (single seed, validation macro-F1, noise about 0.002) point
 * partly the other way, so run the comparison before reporting numbers:</p>
 *
 * <table>
 *   <caption>Earlier neural network sweeps (validation macro-F1)</caption>
 *   <tr><th>Setting</th><th>Result of the sweep</th><th>Value used here</th></tr>
 *   <tr><td>L2 penalty</td><td>0: 0.7379, 1e-4: 0.7340, 1e-3: 0.7291, 1e-2: 0.7201</td>
 *       <td>1e-4 (the sweep preferred 0, a difference above the noise)</td></tr>
 *   <tr><td>Hidden layers</td><td>16; 32-16; 64-32; 128-64-32: 0.7210, 0.7209, 0.7206, 0.7203 (all equal)</td>
 *       <td>128-64-32 (no gain was measured, and training is about ten times slower than 32-16)</td></tr>
 *   <tr><td>Dropout</td><td>0: 0.7223, 0.1: 0.7206, 0.2: 0.7201, 0.3: 0.7192 (within noise)</td>
 *       <td>0.3</td></tr>
 * </table>
 *
 * <p>The tree depth and the logistic regression settings were never part of a parameter sweep, so no earlier
 * measurement exists for them. Because these defaults are also the baseline of the ablation study, results of an
 * ablation run with the new defaults are not comparable with the earlier 120-run results.</p>
 */
public final class ModelDefaults {

    private ModelDefaults() {
        // constants only
    }

    /** Deepest tree the decision tree is allowed to grow to capture multi-feature interactions. */
    public static final int TREE_MAX_DEPTH = 15;
    /** Fewest samples a node needs before it is eligible for a split. */
    public static final int TREE_MIN_SAMPLES_SPLIT = 10;
    /** Fewest samples a leaf may hold, preventing micro-clusters. */
    public static final int TREE_MIN_SAMPLES_LEAF = 2;
    /** A split must lower the impurity by at least this much. */
    public static final double TREE_MIN_IMPURITY_DECREASE = 0.0;
    /** 0 searches every split exactly, 16 to 255 uses histograms (faster on full training sets). */
    public static final int TREE_MAX_BINS = Integer.getInteger("tree.bins", 0);
    /** Depths the web trainer compares on validation data. */
    public static final int[] TREE_DEPTH_GRID = { 5, 10, 15, 20 };
    /** The web trainer takes the shallowest depth within this macro-F1 of the best one. */
    public static final double TREE_SIMPLICITY_TOLERANCE = 0.005;

    /** Gradient-descent iterations; training stops earlier when the loss stalls. */
    public static final int LOGISTIC_ITERATIONS = Integer.getInteger("lr.iterations", 2000);
    /** Gradient-descent step size. */
    public static final double LOGISTIC_LEARNING_RATE = Double.parseDouble(System.getProperty("lr.rate", "0.1"));
    /** L2 penalty to handle collinear features. */
    public static final double LOGISTIC_L2_PENALTY = Double.parseDouble(System.getProperty("lr.l2", "1e-4"));
    /** Weight the rare class 'change' up, so that it is not ignored. */
    public static final boolean LOGISTIC_CLASS_WEIGHTS = true;
    /** Learning rates the web trainer compares. */
    public static final double[] LOGISTIC_LEARNING_RATE_GRID = { 0.01, 0.1, 1.0 };
    /** Iterations of each web search run. */
    public static final int LOGISTIC_SEARCH_ITERATIONS = 800;
    /** Iterations of the final web run. */
    public static final int LOGISTIC_FINAL_ITERATIONS = 4000;

    /** Hidden layer widths, input to output funnel. */
    public static final int[] NEURAL_HIDDEN_LAYERS = { 128, 64, 32 };
    /** Maximum training epochs. */
    public static final int NEURAL_MAX_EPOCHS = Integer.getInteger("nn.epochs", 100);
    /** Optimizer selection (adam or sgd). */
    public static final String NEURAL_OPTIMIZER = System.getProperty("nn.optimizer", "adam");
    /** Default Adam learning rate. */
    public static final double NEURAL_LEARNING_RATE = Double.parseDouble(
            System.getProperty("nn.lr", NEURAL_OPTIMIZER.equalsIgnoreCase("sgd") ? "0.05" : "0.001"));
    /** Samples per network update. */
    public static final int NEURAL_BATCH_SIZE = 64;
    /** Standard weight decay penalty. */
    public static final double NEURAL_L2_PENALTY = Double.parseDouble(System.getProperty("nn.l2", "1e-4"));
    /** Regularization dropout chance for hidden neurons. */
    public static final double NEURAL_DROPOUT = 0.3;
    /** Inputs are clipped to plus or minus this value to prevent extreme outliers; 0 for no clipping. */
    public static final double NEURAL_FEATURE_CLIP = 5.0;
    /** Epochs without a better validation loss before training stops (early stopping). */
    public static final int NEURAL_PATIENCE = 10;
    /** Weight the rare class 'change' up. */
    public static final boolean NEURAL_CLASS_WEIGHTS = true;
    /** Architectures the web trainer compares on validation data. */
    public static final int[][] NEURAL_HIDDEN_LAYERS_GRID = { { 64, 32 }, { 128, 64, 32 }, { 256, 128, 64 } };
    /** Epochs of each web search run. */
    public static final int NEURAL_SEARCH_EPOCHS = 15;
    /** Patience of each web search run. */
    public static final int NEURAL_SEARCH_PATIENCE = 4;
    /** Epochs of the final web run. */
    public static final int NEURAL_FINAL_EPOCHS = 80;
}