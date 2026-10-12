package features;

/**
 * The feature groups of the project, used by the automated ablation study
 * (Ablation.java).
 *
 * A group is a set of SCALAR feature slots (see FeatureExtractor.getFeatures)
 * and, for the character n-gram group, also the n-gram cosine similarity and
 * the hashed bucket differences of the pair vector.
 *
 * WHEN A NEW FEATURE GROUP IS ADDED (punctuation marks, word lengths, ...):
 * give it its own offset/count constants in FeatureExtractor, then add ONE line
 * here. The ablation study then includes it automatically.
 */
public enum FeatureGroup {

    LENGTH("Sentence length (words, chars)", slots(0, 1)),
    FREQUENT_WORDS("Frequent-word ratio", slots(2)),
    PUNCTUATION("Punctuation density", slots(3)),
    ORTHOGRAPHY("Digit + uppercase ratio", slots(4, 5)),
    WORD_LENGTH("Average word length", slots(6)),
    TFIDF("Mean TF-IDF", slots(7)),
    EMOTION("NRC emotion", slots(8)),
    LEXICAL_DIVERSITY("Type-token ratio", slots(9)),
    PUNCTUATION_VARIETY("Punctuation variety", slots(10)),
    FLESCH_READING_EASE("Flesch Reading Ease", slots(11)),
    GUNNING_FOG("Gunning Fog Index", slots(12)),
    FLESCH_KINCAID("Flesch-Kincaid Grade Level", slots(13)),
    // POS ratios are not implemented yet (see FeatureExtractor.POS_COUNT); once they are:
    // POS("POS ratios", range(FeatureExtractor.POS_OFFSET, FeatureExtractor.POS_COUNT)),
    CHAR_NGRAM("Char 3-gram profile (cosine + buckets)", slots());

    private final String description;
    private final int[] scalarSlots;

    FeatureGroup(String description, int[] scalarSlots) {
        this.description = description;
        this.scalarSlots = scalarSlots;
    }

    private static int[] slots(int... slots) {
        return slots;
    }

    /** `count` consecutive scalar slots starting at `offset`. */
    private static int[] range(int offset, int count) {
        int[] slots = new int[count];
        for (int i = 0; i < count; i++) {
            slots[i] = offset + i;
        }
        return slots;
    }

    /**
     * Scalar feature slots (indices below FeatureExtractor.SCALAR_COUNT) in this
     * group.
     */
    public int[] scalarSlots() {
        return scalarSlots.clone();
    }

    /**
     * True for the group that also owns the n-gram cosine and the bucket
     * differences.
     */
    public boolean ownsNGramColumns() {
        return this == CHAR_NGRAM;
    }

    public String description() {
        return description;
    }
}
