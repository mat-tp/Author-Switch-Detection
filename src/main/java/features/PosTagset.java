package features;

/**
 * The coarse part-of-speech tags used by the project.
 *
 * Why coarse: the Penn Treebank has 36 word tags, but a SINGLE sentence has only
 * about 15 tokens. 36 ratios per sentence are almost all 0, which gives
 * sparse and noisy pair differences. Folding the Penn tags into 15 classes keeps the
 * grammatical information (noun/verb/pronoun/... habits) and still has
 * something to count in a short sentence.
 *
 * Both taggers (the built-in heuristic one and Stanford CoreNLP) are mapped to
 * this same set, so they can be exchanged without changing any feature.
 */
public final class PosTagset {

    private PosTagset() {
        // utility class, not meant to be instantiated
    }

    public static final int NOUN = 0;
    public static final int PROPN = 1;
    public static final int VERB = 2;
    public static final int AUX = 3;
    public static final int ADJ = 4;
    public static final int ADV = 5;
    public static final int PRON = 6;
    public static final int DET = 7;
    public static final int ADP = 8;
    public static final int CONJ = 9;
    public static final int NUM = 10;
    public static final int PART = 11;
    public static final int INTJ = 12;
    public static final int PUNCT = 13;
    public static final int OTHER = 14;

    public static final String[] NAMES = { "noun", "propn", "verb", "aux", "adj", "adv", "pron", "det", "adp",
            "conj", "num", "part", "intj", "punct", "other" };

    public static final int COUNT = NAMES.length;

    /**
     * Folds a Penn Treebank tag (what CoreNLP and NLTK produce) into the coarse set.
     *
     * @param penn a Penn Treebank tag, e.g. "NNS", "VBD", "-LRB-", "."
     * @return the coarse tag index
     */
    public static int fromPenn(String penn) {

        if (penn == null || penn.isEmpty()) {
            return OTHER;
        }

        switch (penn) {
            case "NN":
            case "NNS":
                return NOUN;
            case "NNP":
            case "NNPS":
                return PROPN;
            case "VB":
            case "VBD":
            case "VBG":
            case "VBN":
            case "VBP":
            case "VBZ":
                return VERB;
            case "MD":
                return AUX;
            case "JJ":
            case "JJR":
            case "JJS":
                return ADJ;
            case "RB":
            case "RBR":
            case "RBS":
            case "WRB":
                return ADV;
            case "PRP":
            case "PRP$":
            case "WP":
            case "WP$":
            case "EX":
                return PRON;
            case "DT":
            case "PDT":
            case "WDT":
                return DET;
            case "IN":
                return ADP;
            case "CC":
                return CONJ;
            case "CD":
                return NUM;
            case "RP":
            case "TO":
            case "POS":
                return PART;
            case "UH":
                return INTJ;
            default:
                break;
        }

        // CoreNLP/Penn punctuation tags: . , : `` '' -LRB- -RRB- $ # HYPH NFP
        if (!Character.isLetter(penn.charAt(0)) || penn.equals("HYPH") || penn.equals("NFP")) {
            return PUNCT;
        }

        return OTHER; // FW, SYM, LS, ...
    }
}
