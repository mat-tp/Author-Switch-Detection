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

}
