package features;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TextUtils {

    // Both cleaning methods are to be considered in the implementation of the
    // features.

    /**
     * Cleans a sentence
     *
     * Not lowercesing as this could affect the performance of the model. For
     * example, "I" and "i" are different words and have different meanings.
     * Considering digits also as they could be important for some features (e.g.,
     * dates, numbers, etc.)
     *
     * @param sentence the string to clean
     * @return the cleaned string
     */
    public static String cleanSentence(String sentence) {
        if (sentence == null) {
            return "";
        }

        return sentence.replaceAll("[^a-zA-Z0-9']", "");

    }

    /**
     * Clean Sentence by removing all non-alphabetic characters and converting to
     * lowercase.
     *
     * @param sentence
     * @return
     */
    public static String cleanSentenceLowercase(String sentence) {
        if (sentence == null) {
            return "";
        }

        return sentence.replaceAll("[^a-zA-Z]", "").toLowerCase();
    }

    /**
     * Applies the casing policy of the configuration to a text.
     *
     * Used by the DatasetLoader (so the text is already in its final casing when it
     * is loaded) and again by the FeatureExtractor (so a text that comes from
     * somewhere else, a typed sentence in the web app or a TIRA problem file, is
     * treated exactly the same way). Lower-casing twice changes nothing.
     *
     * @param text   the text
     * @param casing PRESERVE keeps it, LOWERCASE lower-cases it
     * @return the text in the wanted casing
     */
    public static String applyCasing(String text, FeatureConfig.Casing casing) {

        if (text == null) {
            return "";
        }

        if (casing == FeatureConfig.Casing.LOWERCASE) {
            return text.toLowerCase(Locale.ROOT);
        }

        return text;
    }

    /**
     * Tokenises a string into a list of words.
     *
     * @param sentence the string to tokenize
     * @return a list of tokens
     */
    public static List<String> tokenize(String sentence) {

        List<String> tokens = new ArrayList<>();

        if (sentence == null || sentence.isBlank()) {
            return tokens;
        }

        for (String raw : sentence.split("\\s+")) {
            String cleaned = cleanSentence(raw);
            if (!cleaned.isEmpty()) {
                tokens.add(cleaned);
            }
        }

        return tokens;
    }

    /*
     * A word (letters, with inner apostrophes and hyphens: don't, well-known), a number (1,000.50),
     * an ellipsis, or ANY other single visible character (every punctuation mark is its own token).
     */
    private static final Pattern TOKEN = Pattern.compile(
            "\\p{L}[\\p{L}\\p{N}]*(?:['’]\\p{L}+)*(?:-\\p{L}[\\p{L}\\p{N}]*)*" // word
                    + "|\\p{N}+(?:[.,]\\p{N}+)*" // number
                    + "|\\.{2,}|…" // ellipsis
                    + "|\\S"); // any other mark

    /**
     * Tokenises a sentence like a POS tagger needs it: the words stay whole and
     * every punctuation mark becomes its own token.
     *
     * "She didn't say much; she never did." gives
     * [She, didn't, say, much, ;, she, never, did, .]
     *
     * @param sentence the sentence
     * @return the tokens, the case is not changed
     */
    public static List<String> tokenizeWithPunctuation(String sentence) {

        List<String> tokens = new ArrayList<>();

        if (sentence == null || sentence.isBlank()) {
            return tokens;
        }

        Matcher matcher = TOKEN.matcher(sentence);
        while (matcher.find()) {
            tokens.add(matcher.group());
        }

        return tokens;
    }

    /**
     * The method to check if a character belongs in an char[] array.
     *
     * @param targetChar char that we are searching for
     * @param charArray  characters of interest
     * @return true if it exists, else false
     */
    public static boolean isCharInChars(char targetChar, char[] charArray) {

        for (char c : charArray) {
            if (targetChar == c) {
                return true;
            }
        }
        return false;
    }

    /**
     * Hash of text[from, to) for the hashed n-gram profiles.
     *
     * Not String.hashCode(): that one is a plain polynomial, so for short n-grams
     * neighbouring grams ("the", "thf") land in neighbouring buckets and the low
     * bits are poorly mixed, which is exactly what floorMod(hash, buckets) uses.
     * FNV-1a followed by a murmur3 style finaliser mixes all the bits, so the
     * grams are spread evenly over the buckets, whatever the number of buckets is.
     *
     * @param text the text
     * @param from first character (included)
     * @param to   last character (excluded)
     * @return a well mixed 32 bit hash
     */
    public static int hash(CharSequence text, int from, int to) {

        int h = 0x811C9DC5; // FNV offset basis

        for (int i = from; i < to; i++) {
            h ^= text.charAt(i);
            h *= 0x01000193; // FNV prime
        }

        return mix(h);
    }

    /** Same, for a sequence of small integers (POS tag n-grams). */
    public static int hash(int[] values, int from, int to) {

        int h = 0x811C9DC5;

        for (int i = from; i < to; i++) {
            h ^= values[i] + 1; // +1: a tag 0 changes the hash too
            h *= 0x01000193;
        }

        return mix(h);
    }

    private static int mix(int h) {

        h ^= h >>> 16;
        h *= 0x85EBCA6B;
        h ^= h >>> 13;
        h *= 0xC2B2AE35;
        h ^= h >>> 16;

        return h;
    }

    /**
     * Helper method to return syllable count of a word. This is still an
     * approximation (English spelling has no exact rule), but it handles the
     * two cases that the first version got wrong most often: a silent final "e"
     * ("make" is one syllable, not two) and "y" as a vowel ("rhythm", "happy").
     *
     * @param word the word to count syllables for
     * @return the number of syllables in the word (at least 1 for a word)
     */
    public static int countSyllables(String word) {

        if (word == null || word.isEmpty()) {
            return 0;
        }

        String lower = word.toLowerCase(Locale.ROOT);
        int count = 0;
        boolean vowel = false;

        for (int i = 0; i < lower.length(); i++) {
            char c = lower.charAt(i);
            // y counts as a vowel unless it starts the word (yes, you)
            boolean isVowel = isVowel(c) || (c == 'y' && i > 0);

            // Count a syllable if the current character is a vowel and the previous
            // character was not a vowel (a group of vowels is ONE syllable)
            if (isVowel && !vowel) {
                count++;
            }
            vowel = isVowel;
        }

        // silent final e: make, like, time (but not "table", "little": consonant + le is a syllable)
        if (count > 1 && lower.endsWith("e") && !lower.endsWith("le") && !lower.endsWith("ee")
                && lower.length() > 2 && !isVowel(lower.charAt(lower.length() - 2))) {
            count--;
        }

        // "-ed" that is not pronounced (walked, jumped), but wanted/needed are 2 syllables
        if (count > 1 && lower.endsWith("ed") && !lower.endsWith("ted") && !lower.endsWith("ded")
                && lower.length() > 3 && !isVowel(lower.charAt(lower.length() - 3))) {
            count--;
        }

        return Math.max(count, 1);
    }

    private static boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
}
