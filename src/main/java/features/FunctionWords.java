package features;

import java.util.HashMap;
import java.util.Map;

/**
 * A fixed list of English function words (closed-class words: determiners,
 * pronouns, prepositions, conjunctions, auxiliaries, common adverbs and
 * particles).
 *
 * Why: function words are used almost unconsciously and carry little topic, so
 * they are the classic stylometric marker (Mosteller and Wallace on the
 * Federalist Papers, Burrows' Delta). The frequent-word list in the dataset folder
 * (google-10000) is a different thing: it also contains content words.
 *
 * The list is built into the code (no file to lose), the order is FIXED: the
 * position of a word in WORDS is its bucket in the function-word profile, so
 * changing the order or the list means every saved model has to be trained again.
 */
public final class FunctionWords {

    private FunctionWords() {
        // utility class, not meant to be instantiated
    }

    private static final String[] RAW = {
            // articles / determiners
            "the", "a", "an", "this", "that", "these", "those", "some", "any", "no", "every", "each", "all",
            "both", "either", "neither", "another", "other", "such", "what", "which", "whose", "much", "many",
            "more", "most", "less", "least", "few", "several", "enough",
            // pronouns
            "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "he", "him", "his", "himself",
            "she", "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves", "who", "whom", "one", "someone", "anyone",
            "everyone", "nobody", "something", "anything", "everything", "nothing",
            // prepositions
            "of", "in", "on", "at", "by", "for", "with", "from", "to", "into", "onto", "over", "under", "about",
            "above", "below", "after", "before", "between", "through", "during", "without", "within", "against",
            "among", "upon", "across", "behind", "beyond", "near", "toward", "towards", "until", "till", "since",
            "off", "out", "up", "down", "around", "along", "past", "per", "than", "like", "via",
            // conjunctions
            "and", "or", "but", "nor", "so", "yet", "because", "although", "though", "while", "whereas", "if",
            "unless", "whether", "as", "once", "when", "where", "whenever", "wherever", "however", "therefore",
            "thus", "hence", "moreover", "otherwise",
            // auxiliaries and modals
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "done", "doing", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
            "ought",
            // adverbs / particles that are used like function words
            "not", "n't", "very", "too", "also", "just", "only", "even", "still", "already", "again", "ever",
            "never", "always", "often", "sometimes", "usually", "then", "now", "here", "there", "how", "why",
            "quite", "rather", "almost", "really", "perhaps", "maybe", "yes", "oh", "well", "however", "else",
            "own", "same", "back", "away", "together",
            // contraction pieces ('s is left out: possessive and "is" cannot be told apart)
            "'ll", "'re", "'ve", "'d", "'m"
    };

    // a word that is listed twice keeps its FIRST position (the profile has no duplicate buckets)
    public static final String[] WORDS = new java.util.LinkedHashSet<>(java.util.Arrays.asList(RAW))
            .toArray(new String[0]);

    public static final int COUNT = WORDS.length;

    private static final Map<String, Integer> INDEX = new HashMap<>();

    static {
        for (int i = 0; i < WORDS.length; i++) {
            INDEX.put(WORDS[i], i);
        }
    }

    /**
     * @param lowerCaseWord a lower-cased token
     * @return its bucket in the profile, -1 if it is not a function word
     */
    public static int indexOf(String lowerCaseWord) {
        Integer index = INDEX.get(lowerCaseWord);
        return index == null ? -1 : index;
    }

    public static boolean contains(String lowerCaseWord) {
        return INDEX.containsKey(lowerCaseWord);
    }
}
