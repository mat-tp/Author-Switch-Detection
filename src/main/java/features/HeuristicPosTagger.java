package features;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A lightweight, rule-based Part-of-Speech tagger designed for fast execution.
 *
 * <p><strong>Rule Precedence Order (First matching rule wins):</strong>
 * <ol>
 *   <li>Punctuation and numbers</li>
 *   <li>Contractions (e.g., "didn't" as auxiliary, "it's" starting with a pronoun)</li>
 *   <li>Closed-class lexicons (determiners, prepositions, pronouns, auxiliaries, conjunctions, particles, interjections) and frequent short word lists</li>
 *   <li>Capitalized words inside sentences (flagged as proper nouns)</li>
 *   <li>Suffix rules ({@code -ly}, {@code -ing}, {@code -ed}, {@code -tion}, {@code -ous}, etc.)</li>
 *   <li>Contextual rules based on the previous tag (e.g., words after determiners mapped to nouns)</li>
 *   <li>Default fallback tag: Noun (most frequent open class)</li>
 * </ol>
 * </p>
 *
 * <p>Designed to provide a fast, deterministic signal for stylistic features (e.g., frequency of pronouns, verbs, adverbs per sentence).
 * Agreement with a trained tagger is documented in {@code docs/RESEARCH.md}.
 * Set system property {@code -Dpos.tagger=corenlp} to use Stanford CoreNLP instead.</p>

 * <p><strong>Thread Safety:</strong> This class is stateless and safe for concurrent use across multiple threads.</p>
 */
public class HeuristicPosTagger implements PosTagger {

    /**
     * Converts a space-separated word list into a set for quick $O(1)$ membership checks.
     *
     * @param words Space-delimited string of lexicon words.
     * @return Set containing the tokenized words.
     */
    private static Set<String> setOf(String words) {
        Set<String> set = new HashSet<>();
        for (String w : words.trim().split("\\s+")) {
            set.add(w);
        }
        return set;
    }

    // Common determiners such as "the", "a", "this", and "those".
    private static final Set<String> DET = setOf(
            "the a an this that these those some any no every each all both either neither another such what "
                    + "which whose whatever whichever much many more most less least few several enough");

    // Pronouns, including personal, possessive, and reflexive forms.
    private static final Set<String> PRON = setOf(
            "i me my mine myself you your yours yourself yourselves he him his himself she her hers herself it its "
                    + "itself we us our ours ourselves they them their theirs themselves who whom whoever someone "
                    + "anyone everyone nobody somebody anybody everybody something anything everything nothing "
                    + "there one's");

    // Prepositions: words expressing place, time, direction, or relationships.
    private static final Set<String> ADP = setOf(
            "of in on at by for with from into onto over under about above below after before between through "
                    + "during without within against among upon across behind beyond near toward towards until till "
                    + "since off out up down around along past per than like via as despite inside outside "
                    + "throughout beside besides beneath underneath unlike amid");

    // Coordinating and subordinating conjunctions connecting words, phrases, or clauses.
    private static final Set<String> CONJ = setOf(
            "and or but nor so yet because although though while whereas if unless whether once when where whenever "
                    + "wherever however therefore thus hence moreover otherwise plus");

    // Auxiliary and modal verbs, such as "is", "have", "will", and "can".
    private static final Set<String> AUX = setOf(
            "am is are was were be been being have has had having do does did will would shall should can could "
                    + "may might must ought cannot 'll 'd 'm 're 've 's");

    // Frequent adverbs, including negation, frequency, time, and manner words.
    private static final Set<String> ADV = setOf(
            "not n't very too also just only even still already again ever never always often sometimes usually "
                    + "then now here how why quite rather almost really perhaps maybe soon today tomorrow yesterday "
                    + "tonight away back together indeed certainly probably actually finally suddenly simply nearly "
                    + "hardly exactly completely instead anyway anymore else far further farther forward home "
                    + "ago once twice later earlier before afterward afterwards forever nowhere somewhere anywhere "
                    + "everywhere");

    // Particles handled specially by the tagger, especially infinitive "to".
    private static final Set<String> PART = setOf("to 's");

    // Interjections and conversational expressions such as "oh", "wow", and "hello".
    private static final Set<String> INTJ = setOf(
            "oh ah ahh aah uh um umm hmm hm huh wow yes yeah yep yup nope okay ok hey hi hello bye goodbye please "
                    + "thanks lol haha hahaha lmao omg ugh oops ouch yay damn alas");

    // Number words that map to a numeric tag.
    private static final Set<String> CARDINAL = setOf(
            "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                    + "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred "
                    + "thousand million billion dozen");

    // Base forms of frequent verbs; inflections (-s, -ed, -ing) are matched via stem stripping.
    private static final Set<String> VERB_BASE = setOf(
            "say go get make know think take see come want look use find give tell work call try ask need feel "
                    + "become leave put mean keep let begin seem help talk turn start show hear play run move like "
                    + "live believe hold bring happen write provide sit stand lose pay meet include continue set "
                    + "learn change lead understand watch follow stop create speak read allow add spend grow open "
                    + "walk win offer remember love consider appear buy wait serve die send expect build stay fall "
                    + "cut reach kill remain suggest raise pass sell require report decide pull return explain "
                    + "hope develop carry break receive agree support hit produce eat cover catch draw choose "
                    + "cause point listen realize place close involve increase thank wish smile laugh cry shout "
                    + "whisper answer reply nod shake drop push throw fly drive ride sleep wake dream save "
                    + "stare glance rush climb jump touch hug kiss fight forget imagine fix mind enjoy wonder "
                    + "knock pick tear ring sing swim hide shoot lie lay rise rest");

    // Irregular verb forms that suffix stripping cannot resolve.
    private static final Set<String> VERB_IRREGULAR = setOf(
            "said went gone got gotten made knew known thought took taken saw seen came gave given told found "
                    + "felt became left kept meant began begun held brought wrote written sat stood lost paid met "
                    + "ran spoke spoken grew grown built fell fallen sold sent spent won chose chosen drew drawn "
                    + "broke broken ate eaten caught fought forgot forgotten hid hidden shook shot slept woke "
                    + "wore worn threw thrown flew flown drove driven rode ridden sang sung swam rang rose risen "
                    + "lay laid stole stolen swore tore torn dug fed led read heard");

    // Common adjectives matched prior to suffix and context rules.
    private static final Set<String> ADJECTIVE = setOf(
            "good new first last long great little own other old right big high different small large next early "
                    + "young important few public bad same able real sure best better free full whole true hard "
                    + "easy black white red blue green yellow dark light happy sad poor rich strong weak hot cold "
                    + "nice beautiful pretty ugly tall short deep wide thin thick heavy quiet loud soft clean "
                    + "dirty empty ready afraid angry tired sick dead alive late open possible special main "
                    + "common clear close simple low military political social national human local recent "
                    + "certain similar available various entire final single particular major second third "
                    + "worse worst less lesser greater greatest bigger biggest smaller smallest older oldest "
                    + "younger youngest larger largest higher highest lower lowest stronger strongest "
                    + "grey gray brown golden silver wild safe sweet bright broken lonely lovely proud");

    // Common "-ing" nouns that are exceptions to standard verbal gerund rules.
    private static final Set<String> NOUN_ING = setOf(
            "thing morning evening king ring spring string wing ceiling building feeling wedding painting "
                    + "meeting beginning ending darling sibling bearing earring clothing");

    // Suffixes typical for nouns (e.g., "-tion", "-ness").
    private static final String[] NOUN_SUFFIX = { "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship",
            "hood", "ism", "ist", "age", "ure", "dom", "ery", "ology" };

    // Suffixes typical for adjectives (e.g., "-ous", "-ful").
    private static final String[] ADJ_SUFFIX = { "ous", "ful", "ive", "able", "ible", "less", "ish", "ical", "ic",
            "ary", "ent", "ant" };

    /**
     * Assigns a coarse part-of-speech tag ID to each input token.
     *
     * @param tokens List of token strings in sentence order.
     * @return Array of integer tag representations matching input token array order and length.
     */
    @Override
    public int[] tag(List<String> tokens) {

        int n = tokens.size();
        int[] tags = new int[n];

        // Determine the index of the first actual word (skipping leading punctuation/dashes)
        int firstWord = 0;
        while (firstWord < n && !hasLetterOrDigit(tokens.get(firstWord))) {
            firstWord++;
        }

        // Process tokens left-to-right to maintain context from previous tags
        for (int i = 0; i < n; i++) {

            int previous = i > 0 ? tags[i - 1] : -1;
            String previousWord = i > 0 ? tokens.get(i - 1).toLowerCase() : "";
            tags[i] = tagToken(tokens.get(i), i == firstWord, previous, previousWord);
        }

        return tags;
    }

    /**
     * Applies deterministic heuristic rules to tag a single token.
     * Order of evaluation is strict (first matching rule applies).
     */
    private int tagToken(String original, boolean sentenceStart, int previous, String previousWord) {

        // Rule 1: Punctuation and numeric checks
        if (!hasLetterOrDigit(original)) {
            return PosTagset.PUNCT;
        }

        if (isNumber(original)) {
            return PosTagset.NUM;
        }

        String word = original.toLowerCase().replace('’', '\'');

        // Rule 2: Contraction analysis via apostrophe detection
        int apostrophe = word.indexOf('\'');
        if (apostrophe > 0) {
            return tagContraction(original, word, apostrophe, sentenceStart);
        }

        // Rule 3: Known closed-class lexicons and frequent word lists
        if (word.equals("to")) {
            return PosTagset.PART;
        }
        if (INTJ.contains(word)) {
            return PosTagset.INTJ;
        }
        if (AUX.contains(word)) {
            return PosTagset.AUX;
        }
        if (DET.contains(word)) {
            return PosTagset.DET;
        }
        if (PRON.contains(word)) {
            return PosTagset.PRON;
        }
        if (CONJ.contains(word)) {
            return PosTagset.CONJ;
        }
        if (ADP.contains(word)) {
            return PosTagset.ADP;
        }
        if (ADV.contains(word)) {
            return PosTagset.ADV;
        }
        if (CARDINAL.contains(word)) {
            return PosTagset.NUM;
        }
        if (ADJECTIVE.contains(word)) {
            return PosTagset.ADJ;
        }
        if (VERB_IRREGULAR.contains(word)) {
            return PosTagset.VERB;
        }

        // Inflected verb forms ("looks", "walked", "running")
        // Post-determiner or post-adjective position overrides tag to NOUN ("the walks")
        if (isVerbForm(word)) {
            return previous == PosTagset.DET || previous == PosTagset.ADJ ? PosTagset.NOUN : PosTagset.VERB;
        }
        if (VERB_BASE.contains(word)) {
            return previous == PosTagset.DET || previous == PosTagset.ADJ ? PosTagset.NOUN : PosTagset.VERB;
        }

        // Rule 4: Proper Noun heuristic for mid-sentence capitalization
        boolean capitalised = Character.isUpperCase(original.charAt(0));
        if (capitalised && !sentenceStart) {
            return PosTagset.PROPN;
        }

        // Rule 5: Morphological suffix evaluation
        int fromSuffix = tagBySuffix(word, previous);
        if (fromSuffix >= 0) {
            return fromSuffix;
        }

        // Rule 6: Contextual tag association based on previous token tag
        if (previous == PosTagset.AUX || previous == PosTagset.PART) {
            return PosTagset.VERB; // "will stay", "to stay"
        }
        if (previous == PosTagset.DET || previous == PosTagset.ADJ || previous == PosTagset.ADP) {
            return PosTagset.NOUN;
        }
        if (previous == PosTagset.PRON && isSubjectPronoun(previousWord)) {
            return PosTagset.VERB; // "I stay", "they stay"
        }

        // Rule 7: Fallback default
        return PosTagset.NOUN;
    }

    /**
     * Resolves apostrophe-based tokens (contractions and possessives).
     */
    private int tagContraction(String original, String word, int apostrophe, boolean sentenceStart) {

        // Negation auxiliaries ("didn't", "couldn't", "won't")
        if (word.endsWith("n't")) {
            return PosTagset.AUX;
        }

        String head = word.substring(0, apostrophe);
        String tail = word.substring(apostrophe);

        // Pronoun head contractions ("it's", "he's", "i'm", "you're")
        if (PRON.contains(head) || head.equals("that") || head.equals("there") || head.equals("what")
                || head.equals("who") || head.equals("where") || head.equals("how") || head.equals("let")) {
            return head.equals("let") ? PosTagset.VERB : PosTagset.PRON;
        }

        // Non-standard contractions fallback
        if (!tail.equals("'s") && !tail.equals("'")) {
            return PosTagset.OTHER;
        }

        // Possessives ("John's", "dogs'")
        boolean capitalised = Character.isUpperCase(original.charAt(0));
        return capitalised && !sentenceStart ? PosTagset.PROPN : PosTagset.NOUN;
    }

    /**
     * Evaluates word endings for part-of-speech classification.
     *
     * @return Integer POS tag ID if suffix matches, otherwise -1.
     */
    private int tagBySuffix(String word, int previous) {

        int length = word.length();

        if (length > 4 && word.endsWith("ly")) {
            return PosTagset.ADV;
        }

        if (length > 4 && word.endsWith("ing")) {
            if (NOUN_ING.contains(word)) {
                return PosTagset.NOUN;
            }
            // "the running water": participle between determiner and noun acts as adjective
            return previous == PosTagset.DET ? PosTagset.ADJ : PosTagset.VERB;
        }

        if (length > 3 && word.endsWith("ed")) {
            return previous == PosTagset.DET ? PosTagset.ADJ : PosTagset.VERB;
        }

        // Match against noun suffixes
        for (String suffix : NOUN_SUFFIX) {
            if (length > suffix.length() + 2 && word.endsWith(suffix)) {
                return PosTagset.NOUN;
            }
        }

        // Match against adjective suffixes
        for (String suffix : ADJ_SUFFIX) {
            if (length > suffix.length() + 2 && word.endsWith(suffix)) {
                return PosTagset.ADJ;
            }
        }

        return -1;
    }

    /**
     * Determines whether a token matches inflected patterns (-s, -ed, -ing, -es)
     * derived from base verbs in {@link #VERB_BASE}.
     */
    private boolean isVerbForm(String word) {

        int length = word.length();

        if (length > 3 && word.endsWith("ing")) {
            String stem = word.substring(0, length - 3);
            return VERB_BASE.contains(stem) || VERB_BASE.contains(stem + "e")
                    || (stem.length() > 2 && stem.charAt(stem.length() - 1) == stem.charAt(stem.length() - 2)
                    && VERB_BASE.contains(stem.substring(0, stem.length() - 1)));
        }

        if (length > 2 && word.endsWith("ed")) {
            String stem = word.substring(0, length - 2);
            return VERB_BASE.contains(stem) || VERB_BASE.contains(stem + "e")
                    || (stem.length() > 2 && stem.charAt(stem.length() - 1) == stem.charAt(stem.length() - 2)
                    && VERB_BASE.contains(stem.substring(0, stem.length() - 1)))
                    || (stem.endsWith("i") && VERB_BASE.contains(stem.substring(0, stem.length() - 1) + "y"));
        }

        if (length > 2 && word.endsWith("es")) {
            String stem = word.substring(0, length - 2);
            if (VERB_BASE.contains(stem) || VERB_BASE.contains(stem + "e")) {
                return true;
            }
            if (stem.endsWith("i") && VERB_BASE.contains(stem.substring(0, stem.length() - 1) + "y")) {
                return true;
            }
        }

        if (length > 1 && word.endsWith("s")) {
            return VERB_BASE.contains(word.substring(0, length - 1));
        }

        return false;
    }

    // Helper: checks if word is a subject pronoun preceding a verb ("they run").
    private static boolean isSubjectPronoun(String word) {
        return word.equals("i") || word.equals("you") || word.equals("we") || word.equals("they")
                || word.equals("he") || word.equals("she") || word.equals("it");
    }

    // Helper: verifies whether a token contains alphanumeric characters.
    private static boolean hasLetterOrDigit(String token) {

        for (int i = 0; i < token.length(); i++) {
            if (Character.isLetterOrDigit(token.charAt(i))) {
                return true;
            }
        }
        return false;
    }

    /**
     * Evaluates whether a token represents a numeric value (including common symbols/separators).
     */
    private static boolean isNumber(String token) {

        boolean digit = false;

        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            if (Character.isDigit(c)) {
                digit = true;
            } else if (c != '.' && c != ',' && c != '-' && c != '/' && c != ':' && c != '%' && c != '$') {
                return false;
            }
        }

        return digit;
    }
}