package data;

import features.FeatureExtractor;

/**
 * Defines strategies for combining two independent sentence feature vectors into a single
 * pairwise representation vector for authorship verification classification.
 *
 * <p><strong>Representation Strategies:</strong> Let {@code u, v} be the z-scored scalar features
 * of sentence 1 and sentence 2 (or raw values for {@code RELATIVE}). Hashed profile features
 * (character n-grams, POS n-grams, function words) are strictly processed through profile
 * divergence measures (Cosine, Jensen-Shannon, Total Variation), whereas scalar features
 * are combined using one of the following enumerated strategies:</p>
 *
 * <table border="1">
 *   <caption>Scalar Pair Representation Strategies</caption>
 *   <tr><th>Representation</th><th>Scalar Part of Pair Vector</th><th>Width</th><th>Symmetric</th></tr>
 *   <tr><td>{@link Representation#ABSOLUTE}</td><td>{@code |u - v|}</td><td>S</td><td>yes</td></tr>
 *   <tr><td>{@link Representation#SIGNED}</td><td>{@code u - v}</td><td>S</td><td>no</td></tr>
 *   <tr><td>{@link Representation#MEAN}</td><td>{@code (u + v) / 2}</td><td>S</td><td>yes</td></tr>
 *   <tr><td>{@link Representation#ABSOLUTE_AND_MEAN}</td><td>{@code [|u - v|, (u + v) / 2]}</td><td>2S</td><td>yes</td></tr>
 *   <tr><td>{@link Representation#RELATIVE}</td><td>{@code |u - v| / (|u| + |v| + eps)}</td><td>S</td><td>yes</td></tr>
 *   <tr><td>{@link Representation#CONCAT}</td><td>{@code [u, v, |u - v|]} (SBERT)</td><td>3S</td><td>no</td></tr>
 *   <tr><td>{@link Representation#ABSOLUTE_AND_PRODUCT}</td><td>{@code [|u - v|, u * v]}</td><td>2S</td><td>yes</td></tr>
 * </table>
 *
 * <p><em>Design Note:</em> {@code SIGNED} preserves the direction of change (A&rarr;B vs B&rarr;A).
 * {@code ABSOLUTE} alone obscures scale (e.g., a distance of 1 is the same for [1, 2] and [101, 102]),
 * which is why {@code MEAN} or {@code RELATIVE} are offered as complements. {@code CONCAT} and
 * {@code ABSOLUTE_AND_PRODUCT} are standard architectures adapted from sentence-pair models
 * (e.g., Reimers and Gurevych, 2019).</p>
 */
public class SentencePair {

    /**
     * Enumerates the available mathematical strategies for combining scalar features of two sentences.
     */
    public enum Representation {
        ABSOLUTE(1, new String[] { "abs" }, 0),
        SIGNED(1, new String[] { "signed" }, -1),
        MEAN(1, new String[] { "mean" }, -1),
        ABSOLUTE_AND_MEAN(2, new String[] { "abs", "mean" }, 0),
        RELATIVE(1, new String[] { "rel" }, -1),
        CONCAT(3, new String[] { "u", "v", "abs" }, 2),
        ABSOLUTE_AND_PRODUCT(2, new String[] { "abs", "prod" }, 0);

        private final int multiplier;
        private final String[] prefixes;
        private final int absolutePart;

        Representation(int multiplier, String[] prefixes, int absolutePart) {
            this.multiplier = multiplier;
            this.prefixes = prefixes;
            this.absolutePart = absolutePart;
        }

        /**
         * Returns how many times the scalar feature count is multiplied to form the pair vector.
         *
         * @return Integer multiplier for vector width allocation.
         */
        public int multiplier() {
            return multiplier;
        }

        /**
         * Returns the descriptive name prefixes for the generated vector parts.
         *
         * @return Array of string prefixes (e.g., {@code ["abs_log_word_count", ...]}).
         */
        public String[] columnPrefixes() {
            return prefixes.clone();
        }

        /**
         * Identifies which partition holds the absolute difference {@code |u - v|}.
         *
         * @return 0-based index of the absolute part, or {@code -1} if absent.
         */
        public int absolutePart() {
            return absolutePart;
        }

        /**
         * Parses a string configuration value into its corresponding {@code Representation} enum.
         *
         * @param text Configuration string (e.g., from {@code -Dpair=abs+mean}).
         * @return Parsed {@link Representation} value.
         */
        public static Representation parse(String text) {

            String value = text.trim().toUpperCase().replace('+', '_').replace('-', '_');

            switch (value) {
                case "ABS":
                    return ABSOLUTE;
                case "DIFF":
                    return SIGNED;
                case "ABS_MEAN":
                case "ABSOLUTE_MEAN":
                    return ABSOLUTE_AND_MEAN;
                case "REL":
                    return RELATIVE;
                case "ABS_PRODUCT":
                case "ABS_PROD":
                case "ABSOLUTE_PRODUCT":
                    return ABSOLUTE_AND_PRODUCT;
                default:
                    return valueOf(value);
            }
        }
    }

    public static final double EPSILON = 1e-8;

    private String sentence1;
    private String sentence2;
    private int label;

    /**
     * Constructs a pair wrapper for two consecutive sentences alongside their ground-truth relationship.
     *
     * @param sentence1 The first chronological sentence string.
     * @param sentence2 The second chronological sentence string.
     * @param label Integer ground-truth author change label (1 = switch, 0 = same author).
     */
    public SentencePair(String sentence1, String sentence2, int label) {

        this.sentence1 = sentence1;
        this.sentence2 = sentence2;
        this.label = label;
    }

    /**
     * Retrieves the first sentence of the pair.
     *
     * @return First sentence string.
     */
    public String getSentence1() {
        return sentence1;
    }

    /**
     * Retrieves the second sentence of the pair.
     *
     * @return Second sentence string.
     */
    public String getSentence2() {
        return sentence2;
    }

    /**
     * Retrieves the ground-truth transition label for the pair.
     *
     * @return Integer label (1 for an author change, 0 otherwise).
     */
    public int getLabel() {
        return label;
    }

    /**
     * Populates a destination array with the calculated scalar portion of a pairwise vector
     * according to the specified representation strategy.
     *
     * <p>Sub-components follow each other sequentially. For a scalar count {@code S}:
     * part 0 occupies indices {@code [0, S)}, part 1 occupies {@code [S, 2S)}, etc.</p>
     *
     * @param representation Strategy defining how to combine the scalar vectors.
     * @param raw1 Raw scalar features of sentence 1 (required by {@link Representation#RELATIVE}).
     * @param raw2 Raw scalar features of sentence 2.
     * @param z1 Z-score standardized features of sentence 1.
     * @param z2 Z-score standardized features of sentence 2.
     * @param scalarCount Total number of scalar features (dimension {@code S}).
     * @param out Destination array where the computed pairwise scalar block is written.
     */
    public static void writeScalarPart(Representation representation, double[] raw1, double[] raw2, double[] z1,
                                       double[] z2, int scalarCount, double[] out) {

        for (int i = 0; i < scalarCount; i++) {

            double u = z1[i];
            double v = z2[i];
            double abs = Math.abs(u - v);

            switch (representation) {
                case ABSOLUTE:
                    out[i] = abs;
                    break;
                case SIGNED:
                    out[i] = u - v;
                    break;
                case MEAN:
                    out[i] = (u + v) / 2.0;
                    break;
                case ABSOLUTE_AND_MEAN:
                    out[i] = abs;
                    out[scalarCount + i] = (u + v) / 2.0;
                    break;
                case RELATIVE:
                    out[i] = Math.abs(raw1[i] - raw2[i]) / (Math.abs(raw1[i]) + Math.abs(raw2[i]) + EPSILON);
                    break;
                case CONCAT:
                    out[i] = u;
                    out[scalarCount + i] = v;
                    out[2 * scalarCount + i] = abs;
                    break;
                case ABSOLUTE_AND_PRODUCT:
                    out[i] = abs;
                    out[scalarCount + i] = u * v;
                    break;
                default:
                    throw new IllegalArgumentException("Unknown representation " + representation);
            }
        }
    }

    /**
     * Computes the complete pairwise feature vector for two sentences, matching the input format
     * expected by trained classification models.
     *
     * <p>The generated vector concatenates the profile divergence measures, the designated scalar
     * representation block, and any full-feature bucket differences.</p>
     *
     * @param extractor Fully fitted feature extractor instance.
     * @param features1 Raw extracted features of the first sentence.
     * @param features2 Raw extracted features of the second sentence.
     * @return Complete combined pairwise feature vector as a double array.
     */
    public double[] getPairFeatures(FeatureExtractor extractor, double[] features1, double[] features2) {

        checkLengths(extractor, features1, features2);
        return extractor.pairVector(features1, features2);
    }

    /**
     * Similar to {@link #getPairFeatures(FeatureExtractor, double[], double[])}, but omits
     * standardization on the full-feature bucket differences.
     *
     * <p>Used internally during the training matrix construction phase, as bucket difference scalers
     * must be fitted collectively across all training pairs before they can be applied.</p>
     *
     * @param extractor Partially fitted feature extractor instance.
     * @param features1 Raw extracted features of the first sentence.
     * @param features2 Raw extracted features of the second sentence.
     * @return Pairwise feature vector with unscaled profile bucket differences.
     */
    public double[] getUnscaledPairFeatures(FeatureExtractor extractor, double[] features1, double[] features2) {

        checkLengths(extractor, features1, features2);

        double[] z1 = extractor.standardize(features1);
        double[] z2 = extractor.standardize(features2);

        double[] out = new double[extractor.pairVectorLength()];
        extractor.writeProfilePart(features1, features2, out);
        extractor.writeScalarPart(features1, features2, z1, z2, out);

        return out;
    }

    private static void checkLengths(FeatureExtractor extractor, double[] features1, double[] features2) {

        if (features1.length != features2.length) {
            throw new IllegalArgumentException("features1 and features2 must have same length");
        }
        if (features1.length != extractor.sentenceVectorLength()) {
            throw new IllegalArgumentException("features must come from FeatureExtractor.getFeatures (expected "
                    + extractor.sentenceVectorLength() + " values, got " + features1.length + ")");
        }
    }
}