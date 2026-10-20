package features;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Every setting that changes HOW the features of a sentence are computed, in
 * one place. It is saved together with the model (inside the FeatureExtractor),
 * so a loaded model always computes its features exactly the way it was trained,
 * and the ablation study (Ablation.java) can vary one setting at a time.
 *
 * Settings come from -D options (same style as Main.java / Ablation.java), for
 * example:
 *
 * java -Dcasing=hybrid -Dchar.n=2,3,4 -Dchar.buckets=512 -Dpair=concat -cp out Main
 *
 * | option         | values                                   | default  |
 * | -------------- | ---------------------------------------- | -------- |
 * | casing         | preserve | lower | hybrid                | preserve |
 * | char.n         | any of 1..5, comma separated, "" = none  | 3        |
 * | char.buckets   | hash buckets of EVERY character profile  | 256      |
 * | char.typed     | true = only affix + punctuation n-grams  | false    |
 * | pos.n          | any of 1..5, comma separated, "" = none  | 2        |
 * | pos.buckets    | hash buckets of EVERY POS profile        | 128      |
 * | pos.tagger     | heuristic | corenlp                      | heuristic|
 * | fw.profile     | true = function-word frequency profile   | true     |
 * | full           | true = also give the raw bucket diffs    | false    |
 * | pair           | see SentencePair.Representation          | absolute |
 * | z.clip         | scalar z-scores are cut at +-this (0=off)| 5        |
 *
 * The defaults are provisional: they are the settings the literature points to
 * (character 3-grams, see docs/RESEARCH.md), the ablation study decides the
 * final ones on the real PAN data.
 */
public class FeatureConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * What happens with the letter case of the text.
     *
     * PRESERVE: the text is used exactly as written. "I" and "i" differ, capitals
     * and ALL-CAPS words are visible to every feature (authors differ in
     * these habits).
     * LOWERCASE: the text is lower-cased before ANY feature is computed. Fewer
     * distinct n-grams (denser profiles, less topic noise from names), but every
     * capital-letter signal is lost.
     * HYBRID: the scalar features (uppercase ratio, ALL-CAPS words, ...) see the
     * original text, the hashed profiles (character n-grams, POS n-grams,
     * function words) see the lower-cased text.
     */
    public enum Casing {
        PRESERVE, LOWERCASE, HYBRID
    }

    // ---- The settings (public, plain fields, the same way the models keep theirs)

    public Casing casing = Casing.PRESERVE;

    /** Orders n of the hashed character n-gram profiles, each one is its own block. */
    public int[] charNGramOrders = { 3 };
    public int charBuckets = 256;

    /** true = only the affix and punctuation n-grams (Sapkota et al., 2015), they carry less topic. */
    public boolean charTyped = false;

    /** Orders n of the hashed POS-tag n-gram profiles. */
    public int[] posNGramOrders = { 2 };
    public int posBuckets = 128;

    /** "heuristic" (built in, no libraries) or "corenlp" (Stanford CoreNLP, needs the CoreNLP jars). */
    public String posTagger = "heuristic";

    /** Function-word frequency profile (one bucket per function word, NOT hashed). */
    public boolean functionWordProfile = true;

    /**
     * false: a profile only gives 4 numbers to the pair vector (cosine, Jensen-Shannon distance,
     * total variation, support overlap).
     * true: the absolute difference of every bucket is added too (much wider vector).
     */
    public boolean fullBlockDifferences = false;

    /** How the two sentences of a pair become one vector, see data.SentencePair.Representation. */
    public String pairRepresentation = "ABSOLUTE";

    /**
     * The z-scores of the scalar features are cut at +-zClip (0 = no clipping). A rare feature
     * (ellipses, repeated marks, ALL-CAPS words) has a tiny standard deviation on the training
     * sentences, so one sentence that uses it gets a z-score of 50 or 150 and the difference of the pair
     * then dominates a linear model. The clip keeps the order of the values and removes the outliers.
     */
    public double zClip = 5.0;

    // ---- Construction

    /** The default settings, then every -D option on top. */
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

    /** Fails early, with a readable message, instead of an ArrayIndexOutOfBounds in the middle of a long run. */
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
        // normalised name (abs+mean -> ABSOLUTE_AND_MEAN), fails here when it is not a representation
        pairRepresentation = data.SentencePair.Representation.parse(pairRepresentation).name();
    }

    // ---- Parsing helpers

    private static Casing parseCasing(String text) {

        String value = text.trim().toUpperCase();

        // short forms, so -Dcasing=lower works
        if (value.startsWith("LOWER")) {
            return Casing.LOWERCASE;
        }
        return Casing.valueOf(value);
    }

    /** "2,3,4" -> {2,3,4}, "" -> {} (the profile is switched off), null -> the fallback. */
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

    /** Short text for logs, the metrics rows and the ablation cache key. */
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
