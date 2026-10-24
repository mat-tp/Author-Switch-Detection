package features;

import java.util.HashMap;
import java.util.Map;

/**
 * A fixed list of English function words (closed-class words: determiners, pronouns,
 * prepositions, conjunctions, auxiliaries, common adverbs, and particles).
 *
 * <p><strong>Stylometric Context:</strong> Function words are used almost unconsciously
 * by authors and carry little topic-specific weight, making them classic markers for
 * stylometric analysis (e.g., Mosteller &amp; Wallace on the Federalist Papers, Burrows' Delta).
 * Unlike frequency-based datasets (such as {@code google-10000}), this collection exclusively
 * targets grammatical function words rather than topic-driven content words.</p>
 *
 * <p><strong>Model Consistency Warning:</strong> The list is built-in and its array order is
 * <strong>strictly fixed</strong>. A word's position in {@link #WORDS} directly defines its feature
 * index/bucket in the function-word profile. Changing the order or contents of this list will
 * invalidate all previously trained models and require a complete re-train.</p>
 */
public final class FunctionWords {

    /**
     * Private constructor to prevent instantiation of this utility class.
     */
    private FunctionWords() {
        // Utility class, not meant to be instantiated
    }

    private static final String[] RAW = {
            // Articles / Determiners
            "the", "a", "an", "this", "that", "these", "those", "some", "any", "no", "every", "each", "all",
            "both", "either", "neither", "another", "other", "such", "what", "which", "whose", "much", "many",
            "more", "most", "less", "least", "few", "several", "enough",
            // Pronouns
            "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "he", "him", "his", "himself",
            "she", "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves", "who", "whom", "one", "someone", "anyone",
            "everyone", "nobody", "something", "anything", "everything", "nothing",
            // Prepositions
            "of", "in", "on", "at", "by", "for", "with", "from", "to", "into", "onto", "over", "under", "about",
            "above", "below", "after", "before", "between", "through", "during", "without", "within", "against",
            "among", "upon", "across", "behind", "beyond", "near", "toward", "towards", "until", "till", "since",
            "off", "out", "up", "down", "around", "along", "past", "per", "than", "like", "via",
            // Conjunctions
            "and", "or", "but", "nor", "so", "yet", "because", "although", "though", "while", "whereas", "if",
            "unless", "whether", "as", "once", "when", "where", "whenever", "wherever", "however", "therefore",
            "thus", "hence", "moreover", "otherwise",
            // Auxiliaries and Modals
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "done", "doing", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
            "ought",
            // Adverbs / Particles used like function words
            "not", "n't", "very", "too", "also", "just", "only", "even", "still", "already", "again", "ever",
            "never", "always", "often", "sometimes", "usually", "then", "now", "here", "there", "how", "why",
            "quite", "rather", "almost", "really", "perhaps", "maybe", "yes", "oh", "well", "however", "else",
            "own", "same", "back", "away", "together",
            // Contraction fragments ('s omitted due to ambiguity between possessive and "is")
            "'ll", "'re", "'ve", "'d", "'m"
    };

    /**
     * Deduplicated array of function words in fixed index order.
     * <p>Duplicate entries in {@link #RAW} retain their first occurrence to preserve bucket indexing.</p>
     */
    public static final String[] WORDS = new java.util.LinkedHashSet<>(java.util.Arrays.asList(RAW))
            .toArray(new String[0]);

    /** Total count of unique function words in the feature profile. */
    public static final int COUNT = WORDS.length;

    /** Internal lookup map for $O(1)$ index retrieval. */
    private static final Map<String, Integer> INDEX = new HashMap<>();

    static {
        for (int i = 0; i < WORDS.length; i++) {
            INDEX.put(WORDS[i], i);
        }
    }

    /**
     * Looks up the feature bucket index for a given lowercase word.
     *
     * @param lowerCaseWord A lower-cased token to query.
     * @return The 0-based bucket index in the function-word profile, or {@code -1} if it is not a function word.
     */
    public static int indexOf(String lowerCaseWord) {
        Integer index = INDEX.get(lowerCaseWord);
        return index == null ? -1 : index;
    }

    /**
     * Checks whether the given lowercase word is present in the function word lexicon.
     *
     * @param lowerCaseWord A lower-cased token to query.
     * @return {@code true} if the word is a recognized function word, {@code false} otherwise.
     */
    public static boolean contains(String lowerCaseWord) {
        return INDEX.containsKey(lowerCaseWord);
    }
}