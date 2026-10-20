package features;

import java.util.ArrayList;
import java.util.List;

/**
 * The feature groups of the project, used by the automated ablation study
 * (Ablation.java).
 *
 * A group is a set of SCALAR features (listed BY NAME, so a new feature in the middle of
 * FeatureExtractor.SCALAR_NAMES can not move a group onto the wrong column) and,
 * for the profile groups, the profile blocks the group owns (the hashed character n-grams,
 * the hashed POS n-grams, the function word profile) with their pair measures
 * (cosine, Jensen-Shannon, ...).
 *
 * WHEN A NEW FEATURE IS ADDED: add its name to FeatureExtractor.SCALAR_NAMES, then add the
 * name to a group here (or a new group with ONE line). The ablation study then includes it
 * automatically.
 *
 * The families group the groups for the coarse ablation table of the paper (lexical,
 * punctuation, orthography, syntax, readability, character profile).
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

    private static String[] names(String... names) {
        return names;
    }

    /**
     * Scalar feature slots (indices below FeatureExtractor.SCALAR_COUNT) in this
     * group.
     */
    public int[] scalarSlots() {
        return scalarSlots.clone();
    }

    /**
     * True when this group owns the profile block with that id ("char3", "pos2", "fw"):
     * its pair measures and (full features) its bucket differences are removed/kept with the group.
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

    public String description() {
        return description;
    }

    /** lexical, punctuation, orthography, syntax, readability, character, length */
    public String family() {
        return family;
    }

    /** The groups of one family. */
    public static List<FeatureGroup> ofFamily(String family) {

        List<FeatureGroup> groups = new ArrayList<>();
        for (FeatureGroup group : values()) {
            if (group.family.equals(family)) {
                groups.add(group);
            }
        }
        return groups;
    }

    /** The family names, in the order they first appear. */
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
