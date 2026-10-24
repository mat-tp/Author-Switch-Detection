package features;

import java.util.ArrayList;
import java.util.List;

/**
 * Feature groups utilized by the automated ablation study ({@code Ablation.java}).
 *
 * <p>A feature group represents a logical collection of scalar features (referenced by name,
 * ensuring index shifts in {@code FeatureExtractor.SCALAR_NAMES} will not offset group column allocations)
 * as well as profile blocks owned by specific groups (e.g., hashed character n-grams, hashed POS n-grams,
 * and function-word profiles with their corresponding pair distance measures).</p>
 *
 * <p><strong>Adding a New Feature:</strong>
 * <ol>
 *   <li>Add the feature name to {@code FeatureExtractor.SCALAR_NAMES}.</li>
 *   <li>Associate the name with an existing group or create a single-line new group entry here.</li>
 * </ol>
 * The ablation pipeline automatically incorporates updated entries.</p>
 *
 * <p>Groups are grouped into higher-level families ({@code lexical}, {@code punctuation},
 * {@code orthography}, {@code syntax}, {@code readability}, {@code character}) for coarse ablation reporting.</p>
 */
public enum FeatureGroup {

    LENGTH("Sentence length (words, chars)", "length",
            names("log_word_count", "log_char_count")),

    FREQUENT_WORDS("Frequent-word ratio + word rarity", "lexical",
            names("frequent_word_ratio", "avg_word_freq_class")),

    PUNCTUATION("Punctuation density + variety", "punctuation",
            names("punctuation_density", "punctuation_variety")),

    PUNCTUATION_MARKS("Per-mark punctuation rates", "punctuation",
            names("comma_rate", "period_rate", "semicolon_rate", "colon_rate", "exclamation_rate",
                    "question_rate", "quote_rate", "apostrophe_rate", "bracket_rate", "dash_rate",
                    "ellipsis_rate", "repeated_punct_rate")),

    ORTHOGRAPHY("Digit, uppercase, ALL-CAPS, capitalised, non-ASCII, elongation", "orthography",
            names("digit_ratio", "uppercase_ratio", "all_caps_word_ratio", "capitalized_word_ratio",
                    "non_ascii_ratio", "elongation_ratio")),

    LAYOUT("Sentence start/end, dialogue, spacing", "orthography",
            names("starts_with_capital", "ends_with_terminal", "starts_with_dialogue",
                    "spacing_anomaly_rate", "starts_with_conjunction")),

    WORD_LENGTH("Word length mean, spread and distribution", "lexical",
            names("avg_word_length", "word_length_std", "wl_1_2", "wl_3_4", "wl_5_6", "wl_7_8",
                    "wl_9_10", "wl_11_plus")),

    TFIDF("Mean TF-IDF", "lexical", names("mean_tfidf")),

    EMOTION("NRC emotion", "lexical", names("emotion_score")),

    LEXICAL_DIVERSITY("Vocabulary richness (TTR, hapax, Simpson, Yule's K)", "lexical",
            names("type_token_ratio", "hapax_ratio", "simpson_d", "yule_k")),

    FLESCH_READING_EASE("Flesch Reading Ease", "readability", names("flesch_reading_ease")),

    GUNNING_FOG("Gunning Fog Index", "readability", names("gunning_fog")),

    FLESCH_KINCAID("Flesch-Kincaid Grade Level", "readability", names("flesch_kincaid_grade")),

    CONTRACTIONS("Contraction ratio", "orthography", names("contraction_ratio")),

    FUNCTION_WORDS("Function-word ratio + profile", "lexical", names("function_word_ratio")),

    POS_RATIOS("POS tag shares + entropy", "syntax",
            names("pos_noun", "pos_propn", "pos_verb", "pos_aux", "pos_adj", "pos_adv", "pos_pron",
                    "pos_det", "pos_adp", "pos_conj", "pos_num", "pos_part", "pos_intj", "pos_punct",
                    "pos_entropy")),

    POS_NGRAM("POS n-gram profile (measures + buckets)", "syntax", names()),

    CHAR_NGRAM("Character n-gram profile (measures + buckets)", "character", names());

    private final String description;
    private final String family;
    private final int[] scalarSlots;

    FeatureGroup(String description, String family, String[] scalarNames) {
        this.description = description;
        this.family = family;
        this.scalarSlots = new int[scalarNames.length];
        for (int i = 0; i < scalarNames.length; i++) {
            this.scalarSlots[i] = FeatureExtractor.indexOf(scalarNames[i]);
        }
    }

    /**
     * Helper varargs utility to construct a string array of feature names.
     */
    private static String[] names(String... names) {
        return names;
    }

    /**
     * Returns a copy of the scalar feature indices (below {@code FeatureExtractor.SCALAR_COUNT})
     * assigned to this group.
     *
     * @return Array of integer scalar feature indices.
     */
    public int[] scalarSlots() {
        return scalarSlots.clone();
    }

    /**
     * Checks if this group owns a specific profile block ID (e.g., {@code "char3"}, {@code "pos2"}, {@code "fw"}).
     * <p>When owned, pair distance measures and bucket differences for the profile block are tethered to this group during ablation.</p>
     *
     * @param blockId The identifier string of the profile block.
     * @return {@code true} if this group owns the specified block, {@code false} otherwise.
     */
    public boolean ownsBlock(String blockId) {

        switch (this) {
            case CHAR_NGRAM:
                return blockId.startsWith("char");
            case POS_NGRAM:
                return blockId.startsWith("pos");
            case FUNCTION_WORDS:
                return blockId.equals("fw");
            default:
                return false;
        }
    }

    /**
     * Returns a human-readable description of the feature group.
     *
     * @return String description.
     */
    public String description() {
        return description;
    }

    /**
     * Returns the feature family category (e.g., {@code lexical}, {@code punctuation}, {@code orthography}, {@code syntax}, {@code readability}, {@code character}, {@code length}).
     *
     * @return Feature family identifier string.
     */
    public String family() {
        return family;
    }

    /**
     * Retrieves all feature groups associated with a given family category.
     *
     * @param family Target feature family string.
     * @return List of matching {@link FeatureGroup} instances.
     */
    public static List<FeatureGroup> ofFamily(String family) {

        List<FeatureGroup> groups = new ArrayList<>();
        for (FeatureGroup group : values()) {
            if (group.family.equals(family)) {
                groups.add(group);
            }
        }
        return groups;
    }

    /**
     * Collects all unique feature family names in order of their first appearance.
     *
     * @return Ordered list of distinct feature family name strings.
     */
    public static List<String> families() {

        List<String> families = new ArrayList<>();
        for (FeatureGroup group : values()) {
            if (!families.contains(group.family)) {
                families.add(group.family);
            }
        }
        return families;
    }
}