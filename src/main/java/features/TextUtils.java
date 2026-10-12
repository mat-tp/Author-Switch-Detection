package features;

import java.util.ArrayList;
import java.util.List;

public class TextUtils {

    // TODO: Consider both uppercase and lowercase letters when cleaning a sentence

    /**
     * Cleans a sentence by removing all non-alphabetic characters and converting to
     * lowercase.
     *
     * @param sentence the string to clean
     * @return the cleaned string
     */
    public static String cleanSentence(String sentence) {
        if (sentence == null) {
            return "";
        }

        return sentence.toLowerCase().replaceAll("[^a-z']", "");

    }

    /**
     * Tokenises a string into a list of words.
     *
     * @param sentence the string to tokenise
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
     * Helper method to return syllable count of a word. This is a naive
     * implementation and may not be accurate for all words.
     * 
     * @param word the word to count syllables for
     * @return the number of syllables in the word
     */
    public static int countSyllables(String word) {
        // This is a very basic implementation and may not be accurate for all words
        int count = 0;
        boolean vowel = false;

        for (int i = 0; i < word.length(); i++) {
            char c = Character.toLowerCase(word.charAt(i));
            boolean isVowel = isVowel(c);

            // Count a syllable if the current character is a vowel and the previous
            // character was not a vowel
            // This is not the most approach but works for some simple cases.
            if (isVowel && !vowel) {
                count++;
            }
            vowel = isVowel;
        }

        return count;
    }

    private static boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
}
