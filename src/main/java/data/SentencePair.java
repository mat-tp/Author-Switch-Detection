package data;

import features.FeatureExtractor;

/*
    How two neighbouring sentences become ONE vector (the representation of a PAIR).

    u, v = the z-scored scalar features of sentence 1 and 2 (raw values for RELATIVE).
    The hashed profiles (character n-grams, POS n-grams, function words) always go in through
    profile measures (cosine, Jensen-Shannon, total variation, support overlap), see
    FeatureExtractor.profileMeasures.

    | representation       | scalar part of the pair vector | width | symmetric |
    | -------------------- | ------------------------------ | ----- | --------- |
    | ABSOLUTE             | |u - v|                         | S     | yes       |
    | SIGNED               | u - v                          | S     | no        |
    | MEAN                 | (u + v) / 2                    | S     | yes       |
    | ABSOLUTE_AND_MEAN    | [|u - v|, (u + v) / 2]         | 2S    | yes       |
    | RELATIVE             | |u - v| / (|u| + |v| + eps)     | S     | yes       |
    | CONCAT               | [u, v, |u - v|]  (SBERT)       | 3S    | no        |
    | ABSOLUTE_AND_PRODUCT | [|u - v|, u * v]               | 2S    | yes       |

    Why these: SIGNED keeps the direction of the change (A->B is not B->A), ABSOLUTE does not.
    The mean is added because |u - v| is the same for (1, 2) and (101, 102): it says HOW MUCH they
    differ but not WHERE on the scale. RELATIVE divides that away. CONCAT and ABSOLUTE_AND_PRODUCT
    come from sentence-pair models (Reimers and Gurevych, 2019: |u - v| is the most important part of the
    concatenation and u * v does not help on top of it), they give the classifier the raw sentences
    too. The ablation study (Ablation.java, pair.*) reports them one by one.
 */
public class SentencePair {

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

        /** How many times the scalar features appear in the pair vector. */
        public int multiplier() {
            return multiplier;
        }

        /** Name prefixes of the parts, in order (abs_log_word_count, ...). */
        public String[] columnPrefixes() {
            return prefixes.clone();
        }

        /** Which part holds |u - v| (-1 when there is none). */
        public int absolutePart() {
            return absolutePart;
        }

        /** Parses a value of -Dpair=, short forms allowed (abs, signed, abs+mean, concat, ...). */
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
     * Constructor for a pair of consecutive sentences.
     *
     * @param sentence1 first sentence
     * @param sentence2 second sentence
     * @param label     ground-truth change label
     */
    public SentencePair(String sentence1, String sentence2, int label) {

        this.sentence1 = sentence1;
        this.sentence2 = sentence2;
        this.label = label;
    }

    /**
     * Returns the first sentence.
     *
     * @return first sentence
     */
    public String getSentence1() {
        return sentence1;
    }

    /**
     * Returns the second sentence.
     *
     * @return second sentence
     */
    public String getSentence2() {
        return sentence2;
    }

    /**
     * Returns the ground-truth label.
     *
     * @return label
     */
    public int getLabel() {
        return label;
    }

    /**
     * Writes the scalar part of a pair vector in the given representation.
     * The parts follow each other (S = scalarCount): part 0 at [0, S), part 1 at [S, 2S), ...
     *
     * @param representation how the two sentences are combined
     * @param raw1           raw scalar features of sentence 1 (used by RELATIVE)
     * @param raw2           raw scalar features of sentence 2
     * @param z1             z-scores of sentence 1
     * @param z2             z-scores of sentence 2
     * @param scalarCount    S
     * @param out            the pair vector
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
                    // on the raw values: |a - b| / (|a| + |b|) is 0 (same) to 1 (opposite or one is 0)
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
     * Computing pairwise features for neighbouring sentences, as the model gets them:
     * the pair representation of the scalar features, the profile measures and the overall
     * distance (and the standardised bucket differences, when the extractor makes the full features).
     * The extractor has to be fitted.
     *
     * @param extractor the fitted feature extractor of the model
     * @param features1 raw features of the 1st sentence (FeatureExtractor.getFeatures)
     * @param features2 raw features of the 2nd sentence
     * @return array of features for the pair
     */
    public double[] getPairFeatures(FeatureExtractor extractor, double[] features1, double[] features2) {

        checkLengths(extractor, features1, features2);
        return extractor.pairVector(features1, features2);
    }

    /**
     * Same as getPairFeatures, but the bucket differences of the full features are NOT standardised yet
     * (used while the training pairs are built: the scaler of those columns can only be fitted
     * on all the training pairs, see FeatureExtractor.fitBlockDifferenceScaler). Without the
     * full features it is the same as getPairFeatures.
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
