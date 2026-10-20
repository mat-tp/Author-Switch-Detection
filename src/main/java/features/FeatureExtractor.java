package features;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.io.FileNotFoundException;
import java.util.*;

import data.SentencePair;

/**
 * Features extraction for sentences and sentence pairs, used by the
 *
 * Most of the features are based on the paper as cited below:
 * Reference:
 *
 * @article{article,
 *                   author = {Oloo, Vivian and Wanzare, Lilian and Otieno,
 *                   Calvins},
 *                   year = {2022},
 *                   month = {11},
 *                   pages = {295-313},
 *                   title = {An Optimal Feature Set for Stylometry-based Style
 *                   Change detection at Document and Sentence Level},
 *                   journal = {International Journal of Scientific Research in
 *                   Computer Science, Engineering and Information Technology},
 *                   doi = {10.32628/CSEIT228617}
 *                   }
 *
 *                   The feature families that were added later (function words,
 *                   per mark punctuation, word length distribution, vocabulary richness,
 *                   contractions, POS ratios and POS n-grams, character n-grams of
 *                   order 1 to 5) follow the systems of the earlier PAN style change
 *                   detection tasks, see docs/RESEARCH.md for every source.
 *
 *                   Stanford CoreNLP: To be considered for higher POS Tagging
 *                   accuracy (-Dpos.tagger=corenlp), but it is slow and requires a lot of memory.
 *
 */
public class FeatureExtractor implements Serializable {

    private static final long serialVersionUID = 2L;

    // A saved model that was fitted with another layout can not be used (see requireCompatible)
    private static final int LAYOUT_VERSION = 2;

    /*
     * Paths for the files used to get the features (Function words And Word-Emotion
     * which is associated with SentimentAnalysis)
     */
    String frequentWordsFilePath = "dataset/word-processing/frequentWords/google-10000-english.txt";
    String sentimentAnalysisFilePath = "dataset/word-processing/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt";

    /*
     * Puctuations of Interest:
     * - , comma
     * - ; semicolon
     * - : colon
     * - ! exclamation
     * - ? question
     * - . period
     * - " straight double quote
     * - ' straight apostrophe
     * - ( left parenthesis
     * - ) right parenthesis
     * - [ left bracket
     * - ] right bracket
     * - { left brace
     * - } right brace
     * - - hyphen
     * - -- double hyphen
     * - – en dash
     * - — em dash
     * - … ellipsis character
     */
    private static final char[] PUNCTUATION = { '!', ';', ',', '.', '?', '\'', '"', ':', '(', ')', '-', '—', '–', '…',
            '[', ']', '{', '}' };

    /*
     * Scalar feature layout. This is the SINGLE source of truth: getFeatures,
     * FeatureGroup (ablation), PairFeatureUtil (names) and the web pages all read
     * it, so a new feature is added here (+ one line in FeatureGroup) and nothing
     * else.
     *
     * The first 14 are the original features (same order as before), the rest are
     * the feature families added for the PAN study (see docs/RESEARCH.md).
     */
    public static final String[] SCALAR_NAMES = {
            "log_word_count", // 0
            "log_char_count", // 1
            "frequent_word_ratio", // 2
            "punctuation_density", // 3
            "digit_ratio", // 4
            "uppercase_ratio", // 5
            "avg_word_length", // 6
            "mean_tfidf", // 7
            "emotion_score", // 8
            "type_token_ratio", // 9
            "punctuation_variety", // 10
            "flesch_reading_ease", // 11
            "gunning_fog", // 12
            "flesch_kincaid_grade", // 13

            // word length distribution
            "word_length_std", "wl_1_2", "wl_3_4", "wl_5_6", "wl_7_8", "wl_9_10", "wl_11_plus",

            // vocabulary richness / word rarity / function words
            "hapax_ratio", "simpson_d", "yule_k", "avg_word_freq_class", "function_word_ratio",

            // punctuation, one rate (per word) for every mark
            "comma_rate", "period_rate", "semicolon_rate", "colon_rate", "exclamation_rate", "question_rate",
            "quote_rate", "apostrophe_rate", "bracket_rate", "dash_rate", "ellipsis_rate", "repeated_punct_rate",

            // orthography / layout / dialogue habits
            "all_caps_word_ratio", "capitalized_word_ratio", "starts_with_capital", "ends_with_terminal",
            "starts_with_dialogue", "non_ascii_ratio", "spacing_anomaly_rate", "elongation_ratio",
            "starts_with_conjunction",

            // contractions (didn't, it's, I'll)
            "contraction_ratio",

            // syntax: share of every coarse POS tag, and how evenly they are spread
            "pos_noun", "pos_propn", "pos_verb", "pos_aux", "pos_adj", "pos_adv", "pos_pron", "pos_det",
            "pos_adp", "pos_conj", "pos_num", "pos_part", "pos_intj", "pos_punct", "pos_entropy"
    };

    /** Index of a scalar feature by name (used by FeatureGroup and the self checks). */
    public static int indexOf(String name) {

        for (int i = 0; i < SCALAR_NAMES.length; i++) {
            if (SCALAR_NAMES[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Unknown scalar feature: " + name);
    }

    // readability slots (FRES, GFI, FKGL)
    public static final int READABILITY_OFFSET = 11;
    public static final int READABILITY_COUNT = 3;

    // number of scalar features before the profile blocks
    public static final int SCALAR_COUNT = SCALAR_NAMES.length;

    // the slots written in getFeatures (looked up by name once, so a reordering of SCALAR_NAMES can not
    // silently write a value into the wrong feature)
    private static final int I_LOG_WORDS = indexOf("log_word_count");
    private static final int I_LOG_CHARS = indexOf("log_char_count");
    private static final int I_FREQUENT = indexOf("frequent_word_ratio");
    private static final int I_PUNCT_DENSITY = indexOf("punctuation_density");
    private static final int I_DIGIT = indexOf("digit_ratio");
    private static final int I_UPPER = indexOf("uppercase_ratio");
    private static final int I_AVG_WORD = indexOf("avg_word_length");
    private static final int I_TFIDF = indexOf("mean_tfidf");
    private static final int I_EMOTION = indexOf("emotion_score");
    private static final int I_TTR = indexOf("type_token_ratio");
    private static final int I_PUNCT_VARIETY = indexOf("punctuation_variety");
    private static final int I_FRES = indexOf("flesch_reading_ease");
    private static final int I_FOG = indexOf("gunning_fog");
    private static final int I_FKGL = indexOf("flesch_kincaid_grade");
    private static final int I_WL_STD = indexOf("word_length_std");
    private static final int I_WL_FIRST = indexOf("wl_1_2");
    private static final int I_HAPAX = indexOf("hapax_ratio");
    private static final int I_SIMPSON = indexOf("simpson_d");
    private static final int I_YULE = indexOf("yule_k");
    private static final int I_FREQ_CLASS = indexOf("avg_word_freq_class");
    private static final int I_FUNCTION = indexOf("function_word_ratio");
    private static final int I_COMMA = indexOf("comma_rate");
    private static final int I_PERIOD = indexOf("period_rate");
    private static final int I_SEMICOLON = indexOf("semicolon_rate");
    private static final int I_COLON = indexOf("colon_rate");
    private static final int I_EXCLAMATION = indexOf("exclamation_rate");
    private static final int I_QUESTION = indexOf("question_rate");
    private static final int I_QUOTE = indexOf("quote_rate");
    private static final int I_APOSTROPHE = indexOf("apostrophe_rate");
    private static final int I_BRACKET = indexOf("bracket_rate");
    private static final int I_DASH = indexOf("dash_rate");
    private static final int I_ELLIPSIS = indexOf("ellipsis_rate");
    private static final int I_REPEATED = indexOf("repeated_punct_rate");
    private static final int I_ALL_CAPS = indexOf("all_caps_word_ratio");
    private static final int I_CAPITALIZED = indexOf("capitalized_word_ratio");
    private static final int I_STARTS_CAPITAL = indexOf("starts_with_capital");
    private static final int I_ENDS_TERMINAL = indexOf("ends_with_terminal");
    private static final int I_STARTS_DIALOGUE = indexOf("starts_with_dialogue");
    private static final int I_NON_ASCII = indexOf("non_ascii_ratio");
    private static final int I_SPACING = indexOf("spacing_anomaly_rate");
    private static final int I_ELONGATION = indexOf("elongation_ratio");
    private static final int I_STARTS_CONJ = indexOf("starts_with_conjunction");
    private static final int I_CONTRACTION = indexOf("contraction_ratio");
    private static final int I_POS_FIRST = indexOf("pos_noun");
    private static final int I_POS_ENTROPY = indexOf("pos_entropy");

    // How many numbers every hashed / function-word profile gives to the pair vector
    // (see profileMeasures)
    public static final int MEASURES_PER_BLOCK = 4;
    public static final String[] MEASURE_NAMES = { "cosine", "jsd", "tv", "jaccard" };

    // z-score parameters of the profile bucket differences of the FULL pair
    // features (the columns after the compressed ones), fitted on the training
    // pairs
    private double[] blockDifferenceMeans;
    private double[] blockDifferenceStds;

    // z-score parameters of the scalar features is fitted on the training sentences
    private double[] scalarMeans;
    private double[] scalarStds;

    // word -> rank in the frequent word list (1 = most frequent)
    private Map<String, Integer> frequentWordRank;

    // This is based on the file (NCR-Emotion-Lexicon)
    enum WordEmotion {
        ANGER,
        ANTICIPATION,
        DISGUST,
        FEAR,
        JOY,
        NEGATIVE,
        POSITIVE,
        SADNESS,
        SURPRISE,
        TRUST
    }

    private final Map<String, int[]> wordEmotionData = new HashMap<>();
    // true once loadWordEmotionData() was tried, so a model loaded on another machine does not try the file again
    private boolean emotionLoadAttempted = false;

    // training data information for features
    private final List<String> vocabulary;
    private final Map<String, Integer> documentFrequency;
    private final int corpusSize;

    // How the features are computed (casing, n-gram orders, buckets, pair representation)
    private final FeatureConfig config;

    // the profile blocks that follow the scalars: id ("char3", "pos2", "fw"), where they start and how wide they are
    private final String[] blockIds;
    private final int[] blockOffsets;
    private final int[] blockSizes;

    private final SentencePair.Representation representation;

    // not saved with the model, created again from config.posTagger
    private transient PosTagger posTagger;

    // true once prepare() is done (not saved: a loaded extractor prepares itself again)
    private transient volatile boolean prepared = false;

    /**
     * Providing the Constructor with the Training data for calculating the term
     * frequency, IDF-TF. Uses the settings from the -D options (see FeatureConfig).
     *
     * @param trainingCorpus
     */
    public FeatureExtractor(List<String> trainingCorpus) {
        this(trainingCorpus, FeatureConfig.fromSystemProperties());
    }

    /**
     * Same, with the settings given.
     *
     * @param trainingCorpus the training sentences (only the vocabulary / IDF table is kept)
     * @param config         how to compute the features
     */
    public FeatureExtractor(List<String> trainingCorpus, FeatureConfig config) {

        config.validate();
        this.config = config.copy();
        this.representation = SentencePair.Representation.parse(this.config.pairRepresentation);

        // the layout of the profile blocks
        List<String> ids = new ArrayList<>();
        List<Integer> sizes = new ArrayList<>();
        for (int n : this.config.charNGramOrders) {
            ids.add("char" + n);
            sizes.add(this.config.charBuckets);
        }
        for (int n : this.config.posNGramOrders) {
            ids.add("pos" + n);
            sizes.add(this.config.posBuckets);
        }
        if (this.config.functionWordProfile) {
            ids.add("fw");
            sizes.add(FunctionWords.COUNT);
        }

        this.blockIds = ids.toArray(new String[0]);
        this.blockSizes = new int[ids.size()];
        this.blockOffsets = new int[ids.size()];
        int offset = SCALAR_COUNT;
        for (int b = 0; b < blockIds.length; b++) {
            blockSizes[b] = sizes.get(b);
            blockOffsets[b] = offset;
            offset += blockSizes[b];
        }

        // Initialise to null references
        if (trainingCorpus == null) {
            this.corpusSize = 0;
            this.documentFrequency = new HashMap<>();
            this.vocabulary = new ArrayList<>();
            return;
        }

        this.corpusSize = trainingCorpus.size();
        this.documentFrequency = new HashMap<>();

        // Build vocabulary and document frequency map
        for (String sentence : trainingCorpus) {

            // the unique words of the sentence, in the casing of the scalar features
            Set<String> uniqueWords = new HashSet<>(TextUtils.tokenize(scalarText(sentence)));

            for (String word : uniqueWords) {
                // aggregate each word in the unique word List
                documentFrequency.merge(word, 1, Integer::sum);
            }
        }

        // Update the vocab and ordering by sorting
        this.vocabulary = new ArrayList<>(documentFrequency.keySet());
        Collections.sort(this.vocabulary);
    }

    public FeatureConfig getConfig() {
        return config.copy();
    }

    // ---------------------------------------------------------------------
    // Layout of the vectors (sentence and pair)
    // ---------------------------------------------------------------------

    /** Number of profile blocks after the scalars of a sentence vector. */
    public int blockCount() {
        return blockIds.length;
    }

    /** Ids of the blocks, in order: "char3", "pos2", "fw", ... */
    public String[] blockIds() {
        return blockIds.clone();
    }

    public int blockOffset(int block) {
        return blockOffsets[block];
    }

    public int blockSize(int block) {
        return blockSizes[block];
    }

    /** Length of the vector of ONE sentence (getFeatures): the scalars, then every block. */
    public int sentenceVectorLength() {
        return SCALAR_COUNT + totalBlockWidth();
    }

    private int totalBlockWidth() {
        int total = 0;
        for (int size : blockSizes) {
            total += size;
        }
        return total;
    }

    public SentencePair.Representation representation() {
        return representation;
    }

    /** Width of the scalar part of the pair vector (depends on the pair representation). */
    public int scalarPairWidth() {
        return SCALAR_COUNT * representation.multiplier();
    }

    /** Column of the first profile measure in the pair vector. */
    public int measureOffset() {
        return scalarPairWidth();
    }

    /** Column of the overall distance. */
    public int distanceColumn() {
        return scalarPairWidth() + blockIds.length * MEASURES_PER_BLOCK;
    }

    /**
     * Number of columns of the COMPRESSED pair vector: the scalar part, 4 measures per
     * profile block and the overall distance.
     */
    public int compressedPairCount() {
        return distanceColumn() + 1;
    }

    /** Number of columns of the FULL pair vector (the compressed ones + every bucket difference). */
    public int fullPairCount() {
        return compressedPairCount() + totalBlockWidth();
    }

    /** The width of the pair vector this extractor makes (compressed, or full when the config says so). */
    public int pairVectorLength() {
        return config.fullBlockDifferences ? fullPairCount() : compressedPairCount();
    }

    /** Names of the columns of the pair vector (compressed part, then the bucket differences if full). */
    public String[] pairFeatureNames() {

        String[] names = new String[pairVectorLength()];
        int c = 0;

        for (String prefix : representation.columnPrefixes()) {
            for (int i = 0; i < SCALAR_COUNT; i++) {
                names[c++] = prefix + "_" + SCALAR_NAMES[i];
            }
        }
        for (String id : blockIds) {
            for (String measure : MEASURE_NAMES) {
                names[c++] = id + "_" + measure;
            }
        }
        names[c++] = "overall_distance";

        if (config.fullBlockDifferences) {
            for (int b = 0; b < blockIds.length; b++) {
                for (int k = 0; k < blockSizes[b]; k++) {
                    names[c++] = blockIds[b] + "_diff_" + k;
                }
            }
        }

        return names;
    }

    /**
     * The pair vector columns that belong to a feature group (used by the ablation study to
     * remove or keep a group): the scalar columns of the group in every part of the
     * pair representation, the measures of the profile blocks the group owns and
     * (full features) their bucket differences.
     */
    public int[] pairColumns(FeatureGroup group) {

        List<Integer> columns = new ArrayList<>();

        for (int slot : group.scalarSlots()) {
            for (int part = 0; part < representation.multiplier(); part++) {
                columns.add(part * SCALAR_COUNT + slot);
            }
        }

        int fullOffset = compressedPairCount();
        for (int b = 0; b < blockIds.length; b++) {

            if (group.ownsBlock(blockIds[b])) {
                for (int m = 0; m < MEASURES_PER_BLOCK; m++) {
                    columns.add(measureOffset() + b * MEASURES_PER_BLOCK + m);
                }
                if (config.fullBlockDifferences) {
                    for (int k = 0; k < blockSizes[b]; k++) {
                        columns.add(fullOffset + k);
                    }
                }
            }
            fullOffset += blockSizes[b];
        }

        int[] result = new int[columns.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = columns.get(i);
        }
        return result;
    }

    /**
     * The columns that hold |z-score difference| of every scalar feature, the overall distance is
     * calculated from them again when a group is removed. null when the pair representation
     * has no such part (MEAN, RELATIVE).
     */
    public int[] absoluteDifferenceColumns() {

        int part = representation.absolutePart();
        if (part < 0) {
            return null;
        }

        int[] columns = new int[SCALAR_COUNT];
        for (int i = 0; i < SCALAR_COUNT; i++) {
            columns[i] = part * SCALAR_COUNT + i;
        }
        return columns;
    }

    // ---------------------------------------------------------------------
    // Lexicons
    // ---------------------------------------------------------------------

    /**
     * Loads the frequent word list (the order of the file = the frequency rank).
     */
    public synchronized void loadFrequentWords() {
        // O(1) for add, search & remove
        frequentWordRank = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(frequentWordsFilePath))) {

            String line;
            int rank = 1;
            while ((line = reader.readLine()) != null) {
                String word = line.trim().toLowerCase();
                if (!word.isEmpty() && !frequentWordRank.containsKey(word)) {
                    frequentWordRank.put(word, rank++);
                }
            }

        } catch (FileNotFoundException e) {
            System.err.println("WARNING: File not found: " + frequentWordsFilePath
                    + " (frequent_word_ratio and avg_word_freq_class will be 0)");

        } catch (IOException e) {
            System.err.println("Error reading file: " + frequentWordsFilePath);
        }
    }

    /**
     * Loads NRC-Emotion data to the memory for extracting features further.
     *
     */
    public synchronized void loadWordEmotionData() {

        emotionLoadAttempted = true;

        /*
         * Reading from the path, where every line has 3 parts (word emotion
         * binaryValue)
         */
        try (BufferedReader reader = new BufferedReader(new FileReader(sentimentAnalysisFilePath))) {

            String line;

            while ((line = reader.readLine()) != null) {

                String[] parts = line.split("\\s+");

                // Expecting 3 parts: word, emotion, association
                if (parts.length != 3) {
                    continue; // Skipping malformed lines
                }

                String word = parts[0].toLowerCase();
                String emotion = parts[1].toUpperCase();
                int association = Integer.parseInt(parts[2]);

                WordEmotion wordEmotion;

                try {
                    wordEmotion = WordEmotion.valueOf(emotion);

                } catch (IllegalArgumentException e) {
                    continue; // Skipping unknown emotions
                }

                // scores are all the binaryValues to one word.
                int[] scores = wordEmotionData.get(word);

                if (scores == null) {

                    scores = new int[WordEmotion.values().length];
                    wordEmotionData.put(word, scores);
                }

                scores[wordEmotion.ordinal()] = association;

            }

        } catch (FileNotFoundException e) {
            System.err.println("WARNING: File not found: " + sentimentAnalysisFilePath
                    + " (emotion_score will be 0)");

        } catch (IOException e) {
            System.err.println("Error reading file: " + sentimentAnalysisFilePath);
        }
    }

    /**
     * Loads the lexicons once, before the (possibly many threaded) feature extraction starts.
     * After the model is trained they are part of the saved extractor, so a model
     * on another machine (the web app, the TIRA run) does not need the dataset folder.
     */
    public void prepare() {

        // the common case (every call of getFeatures) does not take the lock
        if (prepared) {
            return;
        }

        synchronized (this) {

            if (prepared) {
                return;
            }

            if (frequentWordRank == null) {
                loadFrequentWords();
            }
            if (wordEmotionData.isEmpty() && !emotionLoadAttempted) {
                loadWordEmotionData();
            }
            if (posTagger == null) {
                posTagger = PosTagger.create(config.posTagger);
            }

            prepared = true;
        }
    }

    /**
     * Getting the emotion scores for each word in a sentence
     *
     * @param sentence Current sentence
     * @return scores values for each emotion
     */
    private double[] getEmotionScores(String[] tokens) {

        double[] emotionScores = new double[WordEmotion.values().length];

        for (String token : tokens) {

            // getting the words only as the file(NCR-Word-emotion) has words
            int[] scores = wordEmotionData.get(cleanLower(token));

            if (scores != null) {

                for (int i = 0; i < scores.length; i++) {
                    emotionScores[i] += scores[i];
                }
            }
        }

        // we are expecting something like : [2,0,5,....]
        return emotionScores;
    }

    /**
     * Converting the emotion scores array into a single value, so that I can work
     * with it better
     *
     * @param emotionScores array of scores of emotions
     * @return double average value of the emotion scores array
     */
    private double getAverageEmotionScore(double[] emotionScores) {

        // get the ave
        double totalScore = 0.0;
        int count = 0;

        for (double score : emotionScores) {
            if (score > 0) {
                totalScore += score;
                count++;
            }
        }

        return count > 0 ? totalScore / count : 0.0;
    }

    /** Lower-case token with only its letters and apostrophes (no regex, this runs for every token). */
    private static String cleanLower(String token) {

        StringBuilder sb = null;

        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            boolean keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '\'';

            if (!keep && sb == null) {
                sb = new StringBuilder(token.length());
                sb.append(token, 0, i);
            }
            if (keep && sb != null) {
                sb.append(c);
            }
        }

        return (sb == null ? token : sb.toString()).toLowerCase(Locale.ROOT);
    }

    // ---------------------------------------------------------------------
    // Casing
    // ---------------------------------------------------------------------

    /** The text the SCALAR features see (original unless the casing is LOWERCASE). */
    private String scalarText(String sentence) {

        String text = sentence == null ? "" : sentence.trim();
        return config.casing == FeatureConfig.Casing.LOWERCASE ? TextUtils.applyCasing(text, config.casing) : text;
    }

    /** The text the profiles (character n-grams, function words) see (lower-case unless PRESERVE). */
    private String profileText(String sentence) {

        String text = sentence == null ? "" : sentence.trim();
        return config.casing == FeatureConfig.Casing.PRESERVE ? text : text.toLowerCase(Locale.ROOT);
    }

    // ---------------------------------------------------------------------
    // Profiles
    // ---------------------------------------------------------------------

    /**
     * Hashed Character N-Gram Implementation
     * To automatically handle unsceen words in n-grams, which becomes predictable
     *
     * @param text    the text
     * @param n       length of the n-grams (1 to 5)
     * @param buckets number of hash buckets (the length of the profile)
     * @param typed   true = only the affix and punctuation n-grams are counted
     *                (Sapkota et al., 2015: a character n-gram that is in the middle of a word
     *                mostly carries the topic, the one at the edge of a word (prefix, suffix)
     *                or with a punctuation mark carries the style)
     * @return relative frequency of every bucket (the buckets add up to 1, or all 0 when the text
     *         is shorter than n)
     */
    double[] computeCharNGrams(String text, int n, int buckets, boolean typed) {

        double[] nGramVector = new double[buckets];
        int total = 0;

        /*
         * Instead of relying on another corpus to extract the N-Gram, am relying on
         * hashes
         * For easy computation.
         */
        for (int i = 0; i + n <= text.length(); i++) {

            if (typed && !isAffixOrPunctuation(text, i, i + n)) {
                continue;
            }

            int bucket = Math.floorMod(TextUtils.hash(text, i, i + n), buckets);
            nGramVector[bucket]++;
            total++;
        }

        // normalise the feature vector
        if (total > 0) {
            for (int b = 0; b < buckets; b++) {
                nGramVector[b] /= total;
            }
        }

        return nGramVector;
    }

    /**
     * The n-gram text[from, to) is a PUNCTUATION n-gram (it has a mark that is not a letter, digit or
     * space) or an AFFIX n-gram (it starts at the beginning of a word, ends at the end of a word,
     * or has a space in it). The other ones (inside a word) are left out by the typed profile.
     */
    private static boolean isAffixOrPunctuation(String text, int from, int to) {

        for (int i = from; i < to; i++) {
            char c = text.charAt(i);
            if (!Character.isLetterOrDigit(c) && !Character.isWhitespace(c)) {
                return true; // punctuation (beginning, middle or end)
            }
            if (Character.isWhitespace(c)) {
                return true; // space prefix / space suffix
            }
        }

        boolean wordStart = from == 0 || !Character.isLetterOrDigit(text.charAt(from - 1));
        boolean wordEnd = to == text.length() || !Character.isLetterOrDigit(text.charAt(to));

        return wordStart || wordEnd;
    }

    /**
     * Hashed n-gram profile of the POS tags of a sentence.
     *
     * For n = 1 every tag has its OWN bucket (the tag distribution, no collisions). For n > 1 the sentence
     * is padded with a start and an end marker, so "how does a sentence begin / end" is part
     * of the profile.
     */
    double[] computePosNGrams(int[] tags, int n, int buckets) {

        double[] profile = new double[buckets];

        if (n == 1) {
            for (int tag : tags) {
                if (tag < buckets) {
                    profile[tag]++;
                }
            }
            return normalise(profile, tags.length);
        }

        int[] padded = new int[tags.length + 2];
        padded[0] = PosTagset.COUNT; // start of the sentence
        System.arraycopy(tags, 0, padded, 1, tags.length);
        padded[padded.length - 1] = PosTagset.COUNT + 1; // end of the sentence

        int total = 0;
        for (int i = 0; i + n <= padded.length; i++) {
            profile[Math.floorMod(TextUtils.hash(padded, i, i + n), buckets)]++;
            total++;
        }

        return normalise(profile, total);
    }

    private static double[] normalise(double[] profile, int total) {

        if (total > 0) {
            for (int i = 0; i < profile.length; i++) {
                profile[i] /= total;
            }
        }
        return profile;
    }

    /**
     * 4 numbers that say how alike two profiles are:
     * cosine (1 = same direction), Jensen-Shannon distance (0 = same, 1 = nothing in common),
     * total variation (half the L1 distance, the Delta of Burrows with relative frequencies) and the
     * overlap of the buckets that are used (Jaccard of the supports).
     *
     * A profile without a single n-gram (a sentence shorter than n) can not be compared:
     * the similarities are 0 and the distances 1, like the cosine always was.
     */
    static void profileMeasures(double[] a, double[] b, int offset, int size, double[] out, int outPos) {

        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        double totalVariation = 0.0;
        double divergence = 0.0;
        int inBoth = 0;
        int inEither = 0;
        double sumA = 0.0;
        double sumB = 0.0;

        for (int i = offset; i < offset + size; i++) {

            double p = a[i];
            double q = b[i];

            dot += p * q;
            normA += p * p;
            normB += q * q;
            sumA += p;
            sumB += q;
            totalVariation += Math.abs(p - q);

            double m = (p + q) / 2.0;
            if (p > 0.0) {
                divergence += 0.5 * p * log2(p / m);
            }
            if (q > 0.0) {
                divergence += 0.5 * q * log2(q / m);
            }

            if (p > 0.0 || q > 0.0) {
                inEither++;
                if (p > 0.0 && q > 0.0) {
                    inBoth++;
                }
            }
        }

        if (sumA == 0.0 || sumB == 0.0) {
            out[outPos] = 0.0;
            out[outPos + 1] = 1.0;
            out[outPos + 2] = 1.0;
            out[outPos + 3] = 0.0;
            return;
        }

        out[outPos] = dot / (Math.sqrt(normA) * Math.sqrt(normB));
        out[outPos + 1] = Math.sqrt(Math.max(0.0, Math.min(1.0, divergence))); // distance, 0..1
        out[outPos + 2] = Math.min(1.0, totalVariation / 2.0);
        out[outPos + 3] = inEither == 0 ? 0.0 : (double) inBoth / inEither;
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }

    // ---------------------------------------------------------------------
    // Scalar features
    // ---------------------------------------------------------------------

    /**
     * Type-token ratio (TTR) is a measure of lexical diversity in a text.
     * It is calculated as the ratio of the number of unique words (types)
     * to the total number of words (tokens) in a given text.
     *
     *
     * @param words the lower-cased words of the sentence
     * @return 1.0 means all words are unique, 0.0 means all words are the same
     */
    private double calcTypeTokenRatio(List<String> words) {

        // checking the repetion of words in a sentence.
        double typeTokenRatio = 0.0;
        if (words.isEmpty()) {
            return typeTokenRatio; // Return 0 if the sentence has no words
        } else {
            // getting the unique words in a sentence and dividing by the total number of
            // words
            typeTokenRatio = (double) new HashSet<>(words).size() / words.size();
        }

        return typeTokenRatio;

    }

    /**
     * Calculating the punctuation variety in a sentence.
     *
     * @param sentence
     * @return
     */
    private double punctuationVariety(String sentence) {

        if (sentence == null || sentence.isEmpty()) {
            return 0.0;
        }

        Set<Character> uniquePunctuation = new HashSet<>();

        for (int i = 0; i < sentence.length(); i++) {
            char ch = sentence.charAt(i);

            if (TextUtils.isCharInChars(ch, PUNCTUATION)) {
                uniquePunctuation.add(ch);
            }
        }

        // fraction of the punctuation marks of interest used in this sentence (0 to 1)
        return (double) uniquePunctuation.size() / PUNCTUATION.length;
    }

    /**
     * Calculates the Flesch Reading Ease Score (FRES).
     *
     * Formula:
     * FRES = 206.835 - (1.015 * ASL) - (84.6 * ASW)
     *
     * ASL = Average sentence length in words.
     * ASW = Average syllables per word.
     *
     * @param words      the words of the sentence
     * @param syllables  total syllables of the words
     * @return Flesch Reading Ease score
     */
    private double calcFleschReadingEaseScore(int words, int syllables) {

        if (words == 0) {
            return 0.0;
        }

        // Processing one sentence at a time
        double averageWordsPerSentence = words;
        double averageSyllablesPerWord = (double) syllables / words;

        return 206.835
                - (1.015 * averageWordsPerSentence)
                - (84.6 * averageSyllablesPerWord);
    }

    /**
     * Gunning Fog Index (GFI) is a readability test that estimates the years of
     * formal education needed to understand a text on the first reading.
     *
     * Formula:
     * Fog = 0.4 * (ASL + 100 * (ComplexWords / TotalWords))
     *
     * ASL = Average sentence length in words.
     * Complex words = Words containing more than three syllables.
     *
     *
     * Reference:
     * https://www.researchgate.net/publication/392521274_Readability_Analysis_with_the_Gunning_Fog_Index_Guilford_College_Writing_Handbook
     *
     * @param words        the words of the sentence
     * @param complexWords words with 3 or more syllables
     * @return
     */
    private double calcGunningFogIndex(int words, int complexWords) {

        if (words == 0) {
            return 0.0;
        }

        double averageWordsPerSentence = words; // Since we are processing one sentence at a time
        double percentageComplexWords = ((double) complexWords / words) * 100;

        return 0.4 * (averageWordsPerSentence + percentageComplexWords);
    }

    /**
     * Flench Kincaid Grade Level (FKGL) is a readability test that indicates the
     * U.S. school grade level required to comprehend a text.
     *
     * Refence: https://arxiv.org/html/2508.19221v1
     *
     * @param words     the words of the sentence
     * @param syllables total syllables of the words
     * @return
     */
    private double calcFleschKincaidGradeLevel(int words, int syllables) {

        if (words == 0) {
            return 0.0;
        }

        double averageWordsPerSentence = words; // Since we are processing one sentence at a time
        double averageSyllablesPerWord = (double) syllables / words;

        return (0.39 * averageWordsPerSentence) + (11.8 * averageSyllablesPerWord) - 15.59;
    }

    private static final Set<String> SENTENCE_STARTERS = new HashSet<>(
            Arrays.asList("and", "but", "so", "or", "because", "yet", "then", "though", "although"));

    /**
     * Getting the features from each sentence
     * This defines the features used in the project All together
     *
     * @param sentence Current sentence that am extracting features from.
     * @return array of features for the given sentence: the SCALAR_COUNT scalar features (names:
     *         SCALAR_NAMES), then the profile blocks (see blockIds()): hashed character n-grams,
     *         hashed POS n-grams, function word frequencies. A profile is sparse for a short
     *         sentence, that is why the pair vector only gets a few numbers
     *         (profileMeasures) out of every profile, unless the full features are asked for.
     *
     */
    public double[] getFeatures(String sentence) {

        prepare();

        // the casing policy: the scalar features and the profiles can see different texts (HYBRID)
        String text = scalarText(sentence);
        String profile = profileText(sentence);

        // typographic apostrophes (don’t) are normalised, so they are one word like don't
        String wordText = text.replace('’', '\'');

        String[] tokens = wordText.isEmpty() ? new String[0] : wordText.split("\\s+");

        double[] scalarFeatures = new double[SCALAR_COUNT];

        int nWords = tokens.length;
        int nChars = text.length();

        double charDensity = Math.max(nChars, 1);
        double wordDensity = Math.max(nWords, 1);

        // counting frequent words (of the google list) and the emotion words
        int frequentCount = 0;

        for (String token : tokens) {
            if (frequentWordRank != null && frequentWordRank.containsKey(cleanLower(token))) {
                frequentCount++;
            }
        }

        // counting puctions, uppercase (first words in a sentence or repr nouns),
        // letters and digits, every mark of its own, and the layout habits
        int punctuationCount = 0;
        int digitsCount = 0;
        int letterCount = 0;
        int upperCaseCount = 0;
        int commas = 0, periods = 0, semicolons = 0, colons = 0, exclamations = 0, questions = 0;
        int quotes = 0, apostrophes = 0, brackets = 0, dashes = 0, ellipses = 0, repeated = 0;
        int nonAscii = 0, spacingAnomalies = 0;

        for (int i = 0; i < nChars; i++) {

            char ch = wordText.charAt(i);

            if (ch > 127) {
                nonAscii++;
            }

            if (Character.isDigit(ch)) { // digit count
                digitsCount++;
            } else if (Character.isLetter(ch)) { // letters

                letterCount++;

                /*
                 * upper case can be useful for identifying:
                 * (1) First words of the sentence
                 * (2) Nouns and etc.
                 */
                if (Character.isUpperCase(ch)) {
                    upperCaseCount++;
                }

            } else if (TextUtils.isCharInChars(ch, PUNCTUATION)) {
                punctuationCount++;
            }

            char next = i + 1 < nChars ? wordText.charAt(i + 1) : ' ';
            char previous = i > 0 ? wordText.charAt(i - 1) : ' ';

            switch (ch) {
                case ',':
                    commas++;
                    break;
                case '.': {
                    // a run of dots is ONE ellipsis, not several periods
                    if (previous == '.') {
                        break; // already counted at the start of the run
                    }
                    int run = 1;
                    while (i + run < nChars && wordText.charAt(i + run) == '.') {
                        run++;
                    }
                    if (run >= 2) {
                        ellipses++;
                    } else {
                        periods++;
                    }
                    break;
                }
                case '…':
                    ellipses++;
                    break;
                case ';':
                    semicolons++;
                    break;
                case ':':
                    colons++;
                    break;
                case '!':
                    exclamations++;
                    break;
                case '?':
                    questions++;
                    break;
                case '"':
                case '“':
                case '”':
                case '«':
                case '»':
                    quotes++;
                    break;
                case '\'':
                case '‘':
                    apostrophes++;
                    break;
                case '(':
                case ')':
                case '[':
                case ']':
                case '{':
                case '}':
                    brackets++;
                    break;
                case '-':
                case '–':
                case '—':
                case '―':
                    dashes++;
                    break;
                default:
                    break;
            }

            // repeated marks (!!, ??, ,,) = the start of a run of the same mark
            if ((ch == '!' || ch == '?' || ch == ',' || ch == ';' || ch == ':') && next == ch && previous != ch) {
                repeated++;
            }

            // spacing: a double space, a space before a mark, no space after a mark that is followed by a letter
            if (ch == ' ' && next == ' ') {
                spacingAnomalies++;
            }
            if (ch == ' ' && (next == ',' || next == '.' || next == ';' || next == ':' || next == '!' || next == '?')) {
                spacingAnomalies++;
            }
            if ((ch == ',' || ch == ';' || ch == '!' || ch == '?') && Character.isLetter(next)) {
                spacingAnomalies++;
            }
        }

        /*
         * It is worth considering typos in text from person to person
         * Such as Incorrect Formatting (ALL-UPPER caps words, extra spacings, and so
         * on)
         */

        // the words: letters, digits and inner apostrophes, in the casing of the scalar features
        List<String> words = TextUtils.tokenize(wordText);
        int nW = words.size();
        double wDensity = Math.max(nW, 1);

        // Sentiment Scores = Average of the Scores
        // All words from + file are equal to 1, All from - are equal to -1, else = 0.
        /*
         * These values are from a sentiment files, which will be used to determine the
         * sentiment score
         * They are represented using binary scores (0/1)
         *
         * I process these values to give me a single sentence sentiment score instead.
         */

        // Word Emotion Scores
        double[] emotionScores = getEmotionScores(tokens);
        // using the average emotion score instead to reduce feature space
        double averageEmotionScore = getAverageEmotionScore(emotionScores);

        // ---- one pass over the words: lengths, vocabulary, function words, syllables, capitals
        List<String> lowerWords = new ArrayList<>(nW);
        Map<String, Integer> wordCounts = new HashMap<>();
        double[] functionWordCounts = config.functionWordProfile ? new double[FunctionWords.COUNT] : null;
        int functionWordTokens = 0;
        int[] lengthBins = new int[6];
        double lengthSum = 0.0;
        double lengthSquares = 0.0;
        int syllables = 0;
        int complexWords = 0;
        int allCaps = 0;
        int capitalised = 0;
        int contractions = 0;
        int elongated = 0;
        double freqClassSum = 0.0;

        for (int w = 0; w < nW; w++) {

            String word = words.get(w);
            String lower = word.toLowerCase(Locale.ROOT);
            lowerWords.add(lower);
            wordCounts.merge(lower, 1, Integer::sum);

            // length in letters and digits (the apostrophes are not part of the length)
            int length = 0;
            int upperLetters = 0;
            int letters = 0;
            int run = 1;
            boolean hasElongation = false;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (c != '\'') {
                    length++;
                }
                if (Character.isLetter(c)) {
                    letters++;
                    if (Character.isUpperCase(c)) {
                        upperLetters++;
                    }
                }
                if (i > 0 && Character.toLowerCase(c) == Character.toLowerCase(word.charAt(i - 1))) {
                    run++;
                    if (run >= 3 && Character.isLetter(c)) {
                        hasElongation = true;
                    }
                } else {
                    run = 1;
                }
            }

            lengthSum += length;
            lengthSquares += (double) length * length;
            lengthBins[length <= 2 ? 0 : length <= 4 ? 1 : length <= 6 ? 2 : length <= 8 ? 3 : length <= 10 ? 4 : 5]++;

            if (letters >= 2 && upperLetters == letters) {
                allCaps++;
            }
            if (w > 0 && Character.isUpperCase(word.charAt(0))) {
                capitalised++;
            }
            if (hasElongation) {
                elongated++;
            }

            int apostrophe = word.indexOf('\'');
            if (apostrophe > 0 && apostrophe < word.length() - 1) {
                contractions++;
            }

            // readability: syllables of the word without its apostrophes
            int wordSyllables = TextUtils.countSyllables(word.replace("'", ""));
            syllables += wordSyllables;

            // complex words are defined as words with 3 or more syllables
            // there are exclusions to be considered. except for proper nouns, familiar
            // jargon and compound words.
            if (wordSyllables >= 3) {
                complexWords++;
            }

            // word rarity: log2 of the rank in the frequent word list (a word that is not in the list is rarer than all of them)
            if (frequentWordRank != null) {
                Integer rank = frequentWordRank.get(lower);
                freqClassSum += log2(1.0 + (rank == null ? frequentWordRank.size() + 1 : rank));
            }

            if (addFunctionWord(lower, functionWordCounts)) {
                functionWordTokens++;
            }
        }

        // vocabulary richness of the sentence
        int hapax = 0;
        double repeatPairs = 0.0; // sum of n_i * (n_i - 1)
        double yuleSum = 0.0; // sum of i^2 * V_i, over the words that are used i times
        for (int count : wordCounts.values()) {
            if (count == 1) {
                hapax++;
            }
            repeatPairs += (double) count * (count - 1);
            yuleSum += (double) count * count;
        }

        double meanLength = nW == 0 ? 0.0 : lengthSum / nW;
        double lengthVariance = nW == 0 ? 0.0 : Math.max(0.0, lengthSquares / nW - meanLength * meanLength);

        // ---- the tokens and tags of the syntax features
        List<String> posTokens = TextUtils.tokenizeWithPunctuation(wordText);
        int[] tags = posTagger.tag(posTokens);

        // ---- the scalar features
        scalarFeatures[I_LOG_WORDS] = Math.log1p(nWords); // sentence length (words)
        scalarFeatures[I_LOG_CHARS] = Math.log1p(nChars); // sentence length (chars)
        scalarFeatures[I_FREQUENT] = frequentCount / wordDensity; // frequent-word ratio
        scalarFeatures[I_PUNCT_DENSITY] = punctuationCount / charDensity; // punctuation density
        scalarFeatures[I_DIGIT] = digitsCount / charDensity; // digit ratio
        scalarFeatures[I_UPPER] = upperCaseCount / charDensity; // uppercase ratio
        scalarFeatures[I_AVG_WORD] = letterCount / wordDensity; // average word length
        scalarFeatures[I_TFIDF] = calcTFIDFScore(words); // mean TF-IDF
        scalarFeatures[I_EMOTION] = averageEmotionScore / wordDensity; // emotion
        scalarFeatures[I_TTR] = calcTypeTokenRatio(lowerWords); // type-token ratio
        scalarFeatures[I_PUNCT_VARIETY] = punctuationVariety(wordText); // punctuation variety
        scalarFeatures[I_FRES] = calcFleschReadingEaseScore(nW, syllables);
        scalarFeatures[I_FOG] = calcGunningFogIndex(nW, complexWords);
        scalarFeatures[I_FKGL] = calcFleschKincaidGradeLevel(nW, syllables);

        scalarFeatures[I_WL_STD] = Math.sqrt(lengthVariance);
        for (int bin = 0; bin < lengthBins.length; bin++) {
            scalarFeatures[I_WL_FIRST + bin] = lengthBins[bin] / wDensity;
        }

        scalarFeatures[I_HAPAX] = hapax / wDensity;
        scalarFeatures[I_SIMPSON] = nW < 2 ? 0.0 : 1.0 - repeatPairs / ((double) nW * (nW - 1));
        scalarFeatures[I_YULE] = nW == 0 ? 0.0 : 10000.0 * (yuleSum - nW) / ((double) nW * nW);
        scalarFeatures[I_FREQ_CLASS] = freqClassSum / wDensity;
        scalarFeatures[I_FUNCTION] = functionWordTokens / wDensity;

        scalarFeatures[I_COMMA] = commas / wordDensity;
        scalarFeatures[I_PERIOD] = periods / wordDensity;
        scalarFeatures[I_SEMICOLON] = semicolons / wordDensity;
        scalarFeatures[I_COLON] = colons / wordDensity;
        scalarFeatures[I_EXCLAMATION] = exclamations / wordDensity;
        scalarFeatures[I_QUESTION] = questions / wordDensity;
        scalarFeatures[I_QUOTE] = quotes / wordDensity;
        scalarFeatures[I_APOSTROPHE] = apostrophes / wordDensity;
        scalarFeatures[I_BRACKET] = brackets / wordDensity;
        scalarFeatures[I_DASH] = dashes / wordDensity;
        scalarFeatures[I_ELLIPSIS] = ellipses / wordDensity;
        scalarFeatures[I_REPEATED] = repeated / wordDensity;

        scalarFeatures[I_ALL_CAPS] = allCaps / wDensity;
        scalarFeatures[I_CAPITALIZED] = capitalised / Math.max(nW - 1, 1);
        scalarFeatures[I_STARTS_CAPITAL] = startsWithCapital(wordText) ? 1.0 : 0.0;
        scalarFeatures[I_ENDS_TERMINAL] = endsWithTerminal(wordText) ? 1.0 : 0.0;
        scalarFeatures[I_STARTS_DIALOGUE] = startsWithDialogue(wordText) ? 1.0 : 0.0;
        scalarFeatures[I_NON_ASCII] = nonAscii / charDensity;
        scalarFeatures[I_SPACING] = spacingAnomalies / charDensity;
        scalarFeatures[I_ELONGATION] = elongated / wDensity;
        scalarFeatures[I_STARTS_CONJ] = nW > 0 && SENTENCE_STARTERS.contains(lowerWords.get(0)) ? 1.0 : 0.0;

        scalarFeatures[I_CONTRACTION] = contractions / wDensity;

        // syntax: the share of each coarse tag (OTHER is not a feature) and the entropy of the tag distribution
        double[] tagShare = new double[PosTagset.COUNT];
        for (int tag : tags) {
            tagShare[tag]++;
        }
        double entropy = 0.0;
        for (int tag = 0; tag < PosTagset.COUNT; tag++) {
            double share = tags.length == 0 ? 0.0 : tagShare[tag] / tags.length;
            if (tag < PosTagset.OTHER) {
                scalarFeatures[I_POS_FIRST + tag] = share;
            }
            if (share > 0.0) {
                entropy -= share * log2(share);
            }
        }
        // divided by the largest possible entropy, so the feature is between 0 and 1
        scalarFeatures[I_POS_ENTROPY] = entropy / log2(PosTagset.COUNT);

        // ---- the profile blocks
        double[] features = new double[sentenceVectorLength()];
        System.arraycopy(scalarFeatures, 0, features, 0, SCALAR_COUNT);

        for (int b = 0; b < blockIds.length; b++) {

            double[] block;
            String id = blockIds[b];

            if (id.startsWith("char")) {
                block = computeCharNGrams(profile, id.charAt(4) - '0', blockSizes[b], config.charTyped);

            } else if (id.startsWith("pos")) {
                block = computePosNGrams(tags, id.charAt(3) - '0', blockSizes[b]);

            } else { // fw
                block = normalise(functionWordCounts, nW);
            }

            System.arraycopy(block, 0, features, blockOffsets[b], blockSizes[b]);
        }

        return features;

    }

    /**
     * Counts a word in the function word profile.
     * Contractions are split: "didn't" is "did" + "n't", "I'll" is "i" + "'ll".
     *
     * @return true when the word contains at least one function word
     */
    private static boolean addFunctionWord(String lower, double[] counts) {

        int direct = FunctionWords.indexOf(lower);
        if (direct >= 0) {
            if (counts != null) {
                counts[direct]++;
            }
            return true;
        }

        int apostrophe = lower.indexOf('\'');
        if (apostrophe <= 0) {
            return false;
        }

        String head;
        String clitic;

        if (lower.endsWith("n't")) {
            head = lower.substring(0, lower.length() - 3);
            clitic = "n't";
            // can't, won't, shan't
            if (head.equals("ca")) {
                head = "can";
            } else if (head.equals("wo")) {
                head = "will";
            } else if (head.equals("sha")) {
                head = "shall";
            }
        } else {
            head = lower.substring(0, apostrophe);
            clitic = lower.substring(apostrophe);
        }

        boolean any = false;
        int headIndex = FunctionWords.indexOf(head);
        if (headIndex >= 0) {
            if (counts != null) {
                counts[headIndex]++;
            }
            any = true;
        }
        int cliticIndex = FunctionWords.indexOf(clitic);
        if (cliticIndex >= 0) {
            if (counts != null) {
                counts[cliticIndex]++;
            }
            any = true;
        }

        return any;
    }

    private static boolean startsWithCapital(String text) {

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isLetter(c)) {
                return Character.isUpperCase(c);
            }
            if (Character.isDigit(c)) {
                return false;
            }
        }
        return false;
    }

    /** The sentence ends with . ! ? or an ellipsis (a closing quote or bracket after it does not count). */
    private static boolean endsWithTerminal(String text) {

        int end = text.length() - 1;
        while (end >= 0) {
            char c = text.charAt(end);
            if (c == '"' || c == '\'' || c == '”' || c == '’' || c == ')' || c == ']'
                    || Character.isWhitespace(c)) {
                end--;
            } else {
                break;
            }
        }

        if (end < 0) {
            return false;
        }

        char last = text.charAt(end);
        return last == '.' || last == '!' || last == '?' || last == '…';
    }

    /** The sentence starts with a quotation mark or a dash (a line of dialogue). */
    private static boolean startsWithDialogue(String text) {

        if (text.isEmpty()) {
            return false;
        }

        char c = text.charAt(0);
        return c == '"' || c == '“' || c == '‘' || c == '\'' || c == '-' || c == '–' || c == '—'
                || c == '«';
    }

    /**
     * Inverse Document Frequency
     * Measuring how rare(Increases) or common(Reduces) a words occurs.
     *
     * @param term Word that I would like to get the IDF from.
     * @return the IDF value
     */
    private double calculateIDF(String term) {
        int docFreq = documentFrequency.getOrDefault(term, 0);

        // Adding 1 to avoid division by zero And to smooth
        return Math.log((double) (corpusSize + 1) / (docFreq + 1)) + 1;
    }

    private double[] calcTFIDF(List<String> words) {

        // precomputing the term frequency for the sentence
        // for efficiency.
        Map<String, Integer> termCounts = new HashMap<>();
        for (String word : words) {
            termCounts.merge(word, 1, Integer::sum);
        }

        double[] tfidf = new double[words.size()];
        for (int i = 0; i < words.size(); i++) {
            String term = words.get(i);
            double tf = (double) termCounts.get(term) / words.size(); // O(1) now
            double idf = calculateIDF(term);
            tfidf[i] = tf * idf;
        }
        return tfidf;
    }

    /**
     * Averaged TF-IDF score for a sentence
     *
     * @param words the words of the sentence
     * @return
     */
    private double calcTFIDFScore(List<String> words) {

        double[] tfidfScores = calcTFIDF(words);
        double totalScore = 0.0;
        for (double score : tfidfScores) {
            totalScore += score;
        }

        // a sentence without tokens has no scores, dividing by 0 would give NaN
        if (tfidfScores.length == 0) {
            return 0.0;
        }

        return totalScore / tfidfScores.length; // Average TF-IDF score
    }

    // TODO: Refactor the calculation methods.

    /**
     * Absolute difference between 2 vectors
     *
     * @param features1 features from 1st sentence
     * @param features2 features from 2nd sentence
     * @return array of distances
     */
    public double[] calcAbsDifference(double[] features1, double[] features2) {

        if (features1.length != features2.length) {
            throw new IllegalArgumentException("features1 and features2 must have same length");
        }

        double[] absDifferenceArray = new double[features1.length];

        for (int i = 0; i < features1.length; i++) {
            double distance = Math.abs(features1[i] - features2[i]);
            absDifferenceArray[i] = distance;
        }

        return absDifferenceArray;
    }

    // ---------------------------------------------------------------------
    // z-scores
    // ---------------------------------------------------------------------

    /**
     * Fits the z-score parameters (mean and standard deviation) of the scalar
     * features.
     *
     * Use TRAINING sentences only, so nothing from the validation/test data leaks
     * into the model.
     * The profile blocks are not standardised, they are compared using the profile
     * measures (cosine, ...) instead.
     *
     * @param trainingFeatures feature vectors (from getFeatures) of the training
     *                         sentences
     */
    public void fitScaler(List<double[]> trainingFeatures) {

        if (trainingFeatures == null || trainingFeatures.isEmpty()) {
            throw new IllegalArgumentException("Cannot fit the scaler without training features");
        }

        scalarMeans = new double[SCALAR_COUNT];
        scalarStds = new double[SCALAR_COUNT];

        int count = trainingFeatures.size();

        // means
        for (double[] features : trainingFeatures) {
            for (int i = 0; i < SCALAR_COUNT; i++) {
                scalarMeans[i] += features[i];
            }
        }

        for (int i = 0; i < SCALAR_COUNT; i++) {
            scalarMeans[i] /= count;
        }

        // standard deviations
        for (double[] features : trainingFeatures) {
            for (int i = 0; i < SCALAR_COUNT; i++) {
                double difference = features[i] - scalarMeans[i];
                scalarStds[i] += difference * difference;
            }
        }

        for (int i = 0; i < SCALAR_COUNT; i++) {
            scalarStds[i] = Math.sqrt(scalarStds[i] / count);

            // a constant feature has no spread, avoiding a division by zero
            if (scalarStds[i] < 1e-12) {
                scalarStds[i] = 1.0;
            }
        }
    }

    public boolean isFitted() {
        return scalarMeans != null && scalarStds != null;
    }

    /**
     * Fails with a readable message when this extractor was fitted (and saved with
     * a model) for another feature layout than the code now produces,
     * e.g. a model trained before a feature was added. Such a model has to be
     * retrained.
     */
    public void requireCompatible() {

        if (scalarMeans != null && scalarMeans.length != SCALAR_COUNT) {
            throw new IllegalStateException("model was trained with " + scalarMeans.length
                    + " scalar features but the code now has " + SCALAR_COUNT + " - retrain the model");
        }
        if (blockIds == null || config == null) {
            throw new IllegalStateException("model was trained with the old feature layout - retrain the model");
        }
    }

    /**
     * Z-score of the scalar features: (value - mean) / standard deviation, cut at
     * +-config.zClip (see FeatureConfig.zClip). The profile blocks are copied unchanged.
     *
     * @param features feature vector from getFeatures
     * @return copy of the feature vector, with the scalar features standardised
     */
    public double[] standardize(double[] features) {

        if (scalarMeans == null || scalarStds == null) {
            throw new IllegalStateException("The scaler is not fitted. Call fitScaler(...) first");
        }

        double[] standardized = features.clone();

        double clip = config.zClip;

        for (int i = 0; i < SCALAR_COUNT; i++) {
            double z = (features[i] - scalarMeans[i]) / scalarStds[i];
            standardized[i] = clip > 0.0 ? Math.max(-clip, Math.min(clip, z)) : z;
        }

        return standardized;
    }

    // ---------------------------------------------------------------------
    // The input of the retrieval model (one sentence, no pair)
    // ---------------------------------------------------------------------

    /**
     * The vector of ONE sentence that the embedding network of the retrieval model gets:
     * the z-scores of the scalar features (cut like in the pair features, see FeatureConfig.zClip)
     * followed by the square root of every profile bucket times profileScale.
     *
     * Why the square root: a profile is a distribution (its buckets add up to 1), the square root
     * (the Hellinger transform) makes the squared Euclidean distance between two profiles the
     * Hellinger distance, which is a proper distance for distributions, and it gives the rare buckets
     * a weight that is not lost next to the scalars. Every profile then has the length 1, and
     * profileScale sets how loud the profiles are next to the (about 63) z-scores.
     *
     * @param raw          the vector from getFeatures
     * @param profileScale the factor of the profile part (3 to 6 works, see the retrieval ablation)
     * @return the network input, same length as getFeatures gives; the scaler has to be fitted
     */
    public float[] embeddingInput(double[] raw, double profileScale) {

        double[] z = standardize(raw);
        float[] input = new float[raw.length];

        for (int i = 0; i < SCALAR_COUNT; i++) {
            input[i] = (float) z[i];
        }
        for (int i = SCALAR_COUNT; i < raw.length; i++) {
            input[i] = (float) (Math.sqrt(Math.max(0.0, raw[i])) * profileScale);
        }

        return input;
    }

    // ---------------------------------------------------------------------
    // The pair vector
    // ---------------------------------------------------------------------

    /**
     * Writes the part of the pair vector that does NOT need the z-score parameters: the 4 measures of
     * every profile block and (full features) the absolute difference of every bucket.
     * (The columns of the scalar part and the overall distance are written by writeScalarPart.)
     *
     * @param features1 raw features of the first sentence (getFeatures)
     * @param features2 raw features of the second sentence
     * @param out       the pair vector, pairVectorLength() long
     */
    public void writeProfilePart(double[] features1, double[] features2, double[] out) {

        for (int b = 0; b < blockIds.length; b++) {
            profileMeasures(features1, features2, blockOffsets[b], blockSizes[b], out,
                    measureOffset() + b * MEASURES_PER_BLOCK);
        }

        if (config.fullBlockDifferences) {

            int c = compressedPairCount();
            for (int b = 0; b < blockIds.length; b++) {
                for (int k = 0; k < blockSizes[b]; k++) {
                    out[c++] = Math.abs(features1[blockOffsets[b] + k] - features2[blockOffsets[b] + k]);
                }
            }
        }
    }

    /**
     * Writes the scalar part of the pair vector (according to the pair representation) and the overall
     * distance.
     *
     * @param raw1 the RAW scalars of sentence 1 (getFeatures, only the first SCALAR_COUNT are used)
     * @param raw2 the RAW scalars of sentence 2
     * @param z1   the z-scores of sentence 1 (standardize)
     * @param z2   the z-scores of sentence 2
     * @param out  the pair vector, pairVectorLength() long
     */
    public void writeScalarPart(double[] raw1, double[] raw2, double[] z1, double[] z2, double[] out) {

        SentencePair.writeScalarPart(representation, raw1, raw2, z1, z2, SCALAR_COUNT, out);

        double squaredDistance = 0.0;
        for (int i = 0; i < SCALAR_COUNT; i++) {
            double difference = z1[i] - z2[i];
            squaredDistance += difference * difference;
        }
        out[distanceColumn()] = Math.sqrt(squaredDistance / SCALAR_COUNT);
    }

    /**
     * The complete pair vector of two sentences, from their raw features. Standardises
     * both (the scaler has to be fitted) and, for the full features, standardises the bucket differences too
     * (fitBlockDifferenceScaler has to be called). This is what every model gets at prediction time.
     */
    public double[] pairVector(double[] raw1, double[] raw2) {

        double[] z1 = standardize(raw1);
        double[] z2 = standardize(raw2);

        double[] out = new double[pairVectorLength()];
        writeProfilePart(raw1, raw2, out);
        writeScalarPart(raw1, raw2, z1, z2, out);

        return config.fullBlockDifferences ? standardizeBlockDifferences(out) : out;
    }

    /**
     * Fits the z-score parameters (mean and standard deviation) of the profile bucket
     * differences in the FULL pair features. The differences are small numbers (around 0.01), so
     * without this the models would almost ignore them.
     *
     * Use TRAINING pairs only, so nothing from the validation/test data leaks into
     * the model.
     *
     * @param fullPairFeatures full pair feature vectors of the training pairs
     */
    public void fitBlockDifferenceScaler(double[][] fullPairFeatures) {

        if (fullPairFeatures == null || fullPairFeatures.length == 0) {
            throw new IllegalArgumentException("Cannot fit the scaler without training pair features");
        }

        int first = compressedPairCount();
        int nColumns = fullPairFeatures[0].length - first;

        if (nColumns <= 0) {
            throw new IllegalArgumentException("fullPairFeatures must be full pair vectors (config full=true)");
        }

        blockDifferenceMeans = new double[nColumns];
        blockDifferenceStds = new double[nColumns];

        int count = fullPairFeatures.length;

        // means
        for (double[] pairFeatures : fullPairFeatures) {
            for (int i = 0; i < nColumns; i++) {
                blockDifferenceMeans[i] += pairFeatures[first + i];
            }
        }

        for (int i = 0; i < nColumns; i++) {
            blockDifferenceMeans[i] /= count;
        }

        // standard deviations
        for (double[] pairFeatures : fullPairFeatures) {
            for (int i = 0; i < nColumns; i++) {
                double difference = pairFeatures[first + i] - blockDifferenceMeans[i];
                blockDifferenceStds[i] += difference * difference;
            }
        }

        for (int i = 0; i < nColumns; i++) {
            blockDifferenceStds[i] = Math.sqrt(blockDifferenceStds[i] / count);

            // a bucket that never differs has no spread, avoiding a division by zero
            if (blockDifferenceStds[i] < 1e-12) {
                blockDifferenceStds[i] = 1.0;
            }
        }
    }

    /**
     * Z-score of the bucket differences of a FULL pair feature vector.
     * The compressed features (the first compressedPairCount()) are copied unchanged.
     *
     * @param fullPairFeatures full pair feature vector
     * @return copy of the vector, with the bucket differences standardised
     */
    public double[] standardizeBlockDifferences(double[] fullPairFeatures) {

        if (blockDifferenceMeans == null || blockDifferenceStds == null) {
            throw new IllegalStateException(
                    "The bucket difference scaler is not fitted. Call fitBlockDifferenceScaler(...) first");
        }

        int first = compressedPairCount();

        if (fullPairFeatures.length != first + blockDifferenceMeans.length) {
            throw new IllegalArgumentException("Expected " + (first + blockDifferenceMeans.length)
                    + " features, but got " + fullPairFeatures.length);
        }

        double[] standardized = fullPairFeatures.clone();

        for (int i = 0; i < blockDifferenceMeans.length; i++) {
            standardized[first + i] = (fullPairFeatures[first + i] - blockDifferenceMeans[i]) / blockDifferenceStds[i];
        }

        return standardized;
    }

    /** One line for the logs and the metrics rows. */
    public String describe() {
        return config.signature() + " -> " + SCALAR_COUNT + " scalars + " + blockIds.length + " profile blocks ("
                + Arrays.toString(blockIds) + "), pair vector " + pairVectorLength();
    }
}
