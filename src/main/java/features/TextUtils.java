package features;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utility class providing string cleaning, tokenisation, hashing, and syllable
 * counting capabilities.
 */
public class TextUtils {

    /**
     * Cleans a sentence by stripping characters that are not alphanumeric or single
     * quotes.
     * Casing is intentionally preserved to maintain model performance and feature
     * integrity.
     *
     * @param sentence the input string to clean
     * @return the cleaned string, or an empty string if the input is null
     */
    public static String cleanSentence(String sentence) {
        if (sentence == null) {
            return "";
        }

        return sentence.replaceAll("[^a-zA-Z0-9']", "");
    }

    /**
     * Cleans a sentence by removing non-alphabetic characters and converting the
     * result to lowercase.
     *
     * @param sentence the input string to clean
     * @return the cleaned lowercase string, or an empty string if the input is null
     */
    public static String cleanSentenceLowercase(String sentence) {
        if (sentence == null) {
            return "";
        }

        return sentence.replaceAll("[^a-zA-Z]", "").toLowerCase();
    }

    /**
     * Applies the configured casing policy to a given text string.
     *
     * @param text   the input text
     * @param casing the casing rule to apply (PRESERVE or LOWERCASE)
     * @return the text formatted according to the specified casing policy
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
     * Tokenises a string into a list of words using basic whitespace splitting and
     * character cleaning.
     *
     * @param sentence the input string to tokenise
     * @return a list of cleaned word tokens
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
     * Regular expression pattern used to extract tokens for part-of-speech taggers.
     * Extracts individual words, formatted numbers, ellipses, and individual
     * punctuation marks.
     */
    private static final Pattern TOKEN = Pattern.compile(
            "\\p{L}[\\p{L}\\p{N}]*(?:['’]\\p{L}+)*(?:-\\p{L}[\\p{L}\\p{N}]*)*" // Matches Unicode words, contractions,
                                                                               // and hyphenated terms
                    + "|\\p{N}+(?:[.,]\\p{N}+)*" // Matches integers, decimals, and formatted numbers
                    + "|\\.{2,}|…" // Matches multi-period ellipses and Unicode ellipsis characters
                    + "|\\S"); // Matches any single remaining non-whitespace character or symbol

    /**
     * Tokenises a sentence into word structures and standalone punctuation marks
     * for linguistic processing.
     *
     * @param sentence the input sentence to tokenise
     * @return a list of tokens preserving character casing and punctuation
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
     * Evaluates whether a target character exists within a given character array.
     *
     * @param targetChar the character to search for
     * @param charArray  the array of characters to search within
     * @return true if the character is found; false otherwise
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
     * Computes a 32-bit FNV-1a and Murmur3-style mixed hash for a character
     * sequence slice.
     *
     * @param text the source character sequence
     * @param from the inclusive starting index
     * @param to   the exclusive ending index
     * @return a mixed 32-bit hash value
     */
    public static int hash(CharSequence text, int from, int to) {
        int h = 0x811C9DC5; // Initial FNV offset basis

        for (int i = from; i < to; i++) {
            h ^= text.charAt(i);
            h *= 0x01000193; // FNV prime multiplier
        }

        return mix(h);
    }

    /**
     * Computes a 32-bit mixed hash for a slice of an integer array, such as
     * part-of-speech tag n-grams.
     *
     * @param values the source integer array
     * @param from   the inclusive starting index
     * @param to     the exclusive ending index
     * @return a mixed 32-bit hash value
     */
    public static int hash(int[] values, int from, int to) {
        int h = 0x811C9DC5;

        for (int i = from; i < to; i++) {
            h ^= values[i] + 1; // Increment prevents zero values from remaining unhashed
            h *= 0x01000193;
        }

        return mix(h);
    }

    /**
     * Bit-mixing function based on Murmur3 bit-finalisers to distribute hash bits
     * evenly.
     *
     * @param h the intermediate hash state
     * @return the thoroughly mixed integer hash
     */
    private static int mix(int h) {
        h ^= h >>> 16;
        h *= 0x85EBCA6B;
        h ^= h >>> 13;
        h *= 0xC2B2AE35;
        h ^= h >>> 16;

        return h;
    }

    /**
     * Estimates the syllable count of a given word based on vowel cluster
     * heuristics.
     *
     * @param word the word to evaluate
     * @return the estimated number of syllables, returning at least 1 for valid
     *         inputs
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
            // Treat 'y' as a vowel unless it appears at the start of the word
            boolean isVowel = isVowel(c) || (c == 'y' && i > 0);

            // Increment syllable count at the start of each contiguous vowel sequence
            if (isVowel && !vowel) {
                count++;
            }
            vowel = isVowel;
        }

        // Adjust for unpronounced silent 'e' endings
        if (count > 1 && lower.endsWith("e") && !lower.endsWith("le") && !lower.endsWith("ee")
                && lower.length() > 2 && !isVowel(lower.charAt(lower.length() - 2))) {
            count--;
        }

        // Adjust for unpronounced '-ed' past-tense suffixes
        if (count > 1 && lower.endsWith("ed") && !lower.endsWith("ted") && !lower.endsWith("ded")
                && lower.length() > 3 && !isVowel(lower.charAt(lower.length() - 3))) {
            count--;
        }

        return Math.max(count, 1);
    }

    /**
     * Checks whether a character is an English vowel.
     *
     * @param c the character to check
     * @return true if the character is 'a', 'e', 'i', 'o', or 'u'; false otherwise
     */
    private static boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
}