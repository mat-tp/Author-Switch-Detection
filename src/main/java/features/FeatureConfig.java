package features;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Centralized configuration class managing feature computation parameters for sentences.
 *
 * <p>Instances are serialized alongside model files inside {@code FeatureExtractor}, ensuring that
 * loaded models compute features identically to their training state. It also allows
 * automated ablation studies ({@code Ablation.java}) to systematically vary individual parameters.</p>
 *
 * <p><strong>System Property Options ({@code -D} flags):</strong></p>
 * <pre>
 * java -Dcasing=hybrid -Dchar.n=2,3,4 -Dchar.buckets=512 -Dpair=concat -cp out Main
 * </pre>
 *
 * <table border="1">
 *   <caption>Available Configuration Options</caption>
 *   <tr><th>Option</th><th>Values</th><th>Default</th></tr>
 *   <tr><td>{@code casing}</td><td>{@code preserve | lower | hybrid}</td><td>{@code preserve}</td></tr>
 *   <tr><td>{@code char.n}</td><td>Comma-separated range {@code 1..5}, {@code ""} = none</td><td>{@code 3}</td></tr>
 *   <tr><td>{@code char.buckets}</td><td>Hash buckets per character profile</td><td>{@code 256}</td></tr>
 *   <tr><td>{@code char.typed}</td><td>{@code true} = affix + punctuation n-grams only</td><td>{@code false}</td></tr>
 *   <tr><td>{@code pos.n}</td><td>Comma-separated range {@code 1..5}, {@code ""} = none</td><td>{@code 2}</td></tr>
 *   <tr><td>{@code pos.buckets}</td><td>Hash buckets per POS profile</td><td>{@code 128}</td></tr>
 *   <tr><td>{@code pos.tagger}</td><td>{@code heuristic | corenlp}</td><td>{@code heuristic}</td></tr>
 *   <tr><td>{@code fw.profile}</td><td>{@code true} = function-word frequency profile</td><td>{@code true}</td></tr>
 *   <tr><td>{@code full}</td><td>{@code true} = include raw bucket differences</td><td>{@code false}</td></tr>
 *   <tr><td>{@code pair}</td><td>See {@code SentencePair.Representation}</td><td>{@code ABSOLUTE}</td></tr>
 *   <tr><td>{@code z.clip}</td><td>Scalar z-score threshold boundary (0 = off)</td><td>{@code 5.0}</td></tr>
 * </table>
 *
 * <p><em>Note: Default values reflect preliminary literature baselines (see {@code docs/RESEARCH.md});
 * final configurations are evaluated via ablation experiments on PAN benchmark datasets.</em></p>
 */
public class FeatureConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Defines casing treatment during feature extraction.
     */
    public enum Casing {
        /** Text remains untouched. Preserves capitalization signals and ALL-CAPS usage differences across authors. */
        PRESERVE,

        /** Text is completely lower-cased prior to feature generation, yielding denser profiles with lower topic noise. */
        LOWERCASE,

        /** Scalar features inspect original casing; hashed profiles (char/POS n-grams, function words) inspect lower-cased text. */
        HYBRID
    }

    // Configuration Settings (Public fields retained for model compatibility)

    /** Casing strategy applied during extraction. */
    public Casing casing = Casing.PRESERVE;

    /** N-gram order lengths for character profiles; each order operates in its own block. */
    public int[] charNGramOrders = { 3 };

    /** Total hash bucket count allocated per character n-gram profile. */
    public int charBuckets = 256;

    /** When set to {@code true}, restricts character n-grams to affixes and punctuation (Sapkota et al., 2015). */
    public boolean charTyped = false;

    /** N-gram order lengths for POS tag profiles. */
    public int[] posNGramOrders = { 2 };

    /** Total hash bucket count allocated per POS n-gram profile. */
    public int posBuckets = 128;

    /** Selected POS tagger implementation: {@code "heuristic"} (built-in) or {@code "corenlp"} (Stanford CoreNLP). */
    public String posTagger = "heuristic";

    /** Enables the unhashed function-word frequency profile (1 bucket per word). */
    public boolean functionWordProfile = true;

    /**
     * Toggles detailed vector outputs for profile blocks.
     * <ul>
     *   <li>{@code false}: Profile yields 4 summary metrics (cosine, Jensen-Shannon, total variation, support overlap).</li>
     *   <li>{@code true}: Appends raw bucket absolute differences, expanding vector width.</li>
     * </ul>
     */
    public boolean fullBlockDifferences = false;

    /** Pair representation transformation method (refers to {@code data.SentencePair.Representation}). */
    public String pairRepresentation = "ABSOLUTE";

    /**
     * Maximum z-score threshold boundary applied to scalar features (0 = disabled).
     * <p>Prevents extreme outliers in low-variance features (e.g., ellipses, ALL-CAPS words)
     * from distorting linear models.</p>
     */
    public double zClip = 5.0;

    // Construction

    /**
     * Instantiates a default {@code FeatureConfig} and overrides properties using System {@code -D} flags.
     *
     * @return Validated {@link FeatureConfig} instance.
     */
    public static FeatureConfig fromSystemProperties() {

        FeatureConfig config = new FeatureConfig();

        config.casing = parseCasing(System.getProperty("casing", config.casing.name()));
        config.charNGramOrders = parseOrders(System.getProperty("char.n"), config.charNGramOrders);
        config.charBuckets = Integer.getInteger("char.buckets", config.charBuckets);
        config.charTyped = Boolean.parseBoolean(System.getProperty("char.typed", String.valueOf(config.charTyped)));
        config.posNGramOrders = parseOrders(System.getProperty("pos.n"), config.posNGramOrders);
        config.posBuckets = Integer.getInteger("pos.buckets", config.posBuckets);
        config.posTagger = System.getProperty("pos.tagger", config.posTagger).toLowerCase();
        config.functionWordProfile = Boolean
                .parseBoolean(System.getProperty("fw.profile", String.valueOf(config.functionWordProfile)));
        config.fullBlockDifferences = Boolean
                .parseBoolean(System.getProperty("full", String.valueOf(config.fullBlockDifferences)));
        config.pairRepresentation = System.getProperty("pair", config.pairRepresentation).toUpperCase();
        config.zClip = Double.parseDouble(System.getProperty("z.clip", String.valueOf(config.zClip)));

        config.validate();
        return config;
    }

    /**
     * Creates a deep copy of this configuration instance.
     *
     * @return A cloned {@link FeatureConfig} object.
     */
    public FeatureConfig copy() {

        FeatureConfig c = new FeatureConfig();
        c.casing = casing;
        c.charNGramOrders = charNGramOrders.clone();
        c.charBuckets = charBuckets;
        c.charTyped = charTyped;
        c.posNGramOrders = posNGramOrders.clone();
        c.posBuckets = posBuckets;
        c.posTagger = posTagger;
        c.functionWordProfile = functionWordProfile;
        c.fullBlockDifferences = fullBlockDifferences;
        c.pairRepresentation = pairRepresentation;
        c.zClip = zClip;
        return c;
    }

    /**
     * Validates configuration settings to fail fast prior to running execution loops.
     *
     * @throws IllegalArgumentException If any option setting falls outside valid bounds.
     */
    public void validate() {

        for (int n : charNGramOrders) {
            if (n < 1 || n > 5) {
                throw new IllegalArgumentException("character n-gram order must be 1..5, got " + n);
            }
        }
        for (int n : posNGramOrders) {
            if (n < 1 || n > 5) {
                throw new IllegalArgumentException("POS n-gram order must be 1..5, got " + n);
            }
        }
        if (zClip < 0.0) {
            throw new IllegalArgumentException("z.clip can not be negative (0 = off)");
        }
        if (charBuckets < 2 || posBuckets < 2) {
            throw new IllegalArgumentException("a profile needs at least 2 buckets");
        }
        if (!posTagger.equals("heuristic") && !posTagger.equals("corenlp")) {
            throw new IllegalArgumentException("pos.tagger must be heuristic or corenlp, got " + posTagger);
        }
        // Normalize name (e.g., abs+mean -> ABSOLUTE_AND_MEAN); validates representation validity early
        pairRepresentation = data.SentencePair.Representation.parse(pairRepresentation).name();
    }

    // Parsing helpers

    /**
     * Parses casing option strings, supporting shorthand inputs (e.g., "lower").
     */
    private static Casing parseCasing(String text) {

        String value = text.trim().toUpperCase();

        if (value.startsWith("LOWER")) {
            return Casing.LOWERCASE;
        }
        return Casing.valueOf(value);
    }

    /**
     * Parses comma-separated integer n-gram order lists (e.g., "2,3,4" -> {2, 3, 4}).
     */
    static int[] parseOrders(String text, int[] fallback) {

        if (text == null) {
            return fallback.clone();
        }
        if (text.isBlank() || text.equalsIgnoreCase("none")) {
            return new int[0];
        }

        String[] parts = text.split(",");
        int[] orders = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            orders[i] = Integer.parseInt(parts[i].trim());
        }
        return orders;
    }

    /**
     * Generates a compact string signature representing current settings for logging,
     * metrics tracking, and ablation caching.
     *
     * @return Compact configuration identifier string.
     */
    public String signature() {

        return "case=" + casing.name().toLowerCase()
                + "|char=" + Arrays.toString(charNGramOrders) + "x" + charBuckets + (charTyped ? "typed" : "")
                + "|pos=" + Arrays.toString(posNGramOrders) + "x" + posBuckets + "(" + posTagger + ")"
                + "|fw=" + functionWordProfile
                + "|full=" + fullBlockDifferences
                + "|pair=" + pairRepresentation
                + "|zclip=" + zClip;
    }

    @Override
    public String toString() {
        return signature();
    }
}