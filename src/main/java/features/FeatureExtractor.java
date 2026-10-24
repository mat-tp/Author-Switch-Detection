package features;

import util.DistributionDistances;
import util.InformationTheory;
import util.Statistics;
import util.VectorMath;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.io.FileNotFoundException;
import java.util.*;

import data.SentencePair;

/**
 * Main feature extraction engine for individual sentences and sentence pairs.
 *
 * <p>Computes scalar stylometric features (e.g., readability scores, punctuation rates,
 * character/word statistics, emotion scores) and profile block distributions
 * (character n-grams, POS tag n-grams, function word profiles).</p>
 *
 * <p><strong>Tagging Note:</strong> Integrates with {@link PosTagger} (either heuristic or
 * Stanford CoreNLP based on configuration).</p>
 */
public class FeatureExtractor implements Serializable {

    private static final long serialVersionUID = 2L;

    // Saved models fitted with a different layout version cannot be loaded.
    private static final int LAYOUT_VERSION = 2;

    /** Path to the Google 10,000 frequent English words dataset. */
    String frequentWordsFilePath = "dataset/word-processing/frequentWords/google-10000-english.txt";

    /** Path to the NRC Emotion Lexicon word-level dataset. */
    String sentimentAnalysisFilePath = "dataset/word-processing/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt";

    /** Punctuation marks tracked for variety and density calculation. */
    private static final char[] PUNCTUATION = { '!', ';', ',', '.', '?', '\'', '"', ':', '(', ')', '-', '—', '–', '…',
            '[', ']', '{', '}' };

    /**
     * Array of all scalar feature names in strict layout order.
     * The first 14 elements represent core legacy features; remaining elements represent extended stylometric features.
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

            // Word length distribution
            "word_length_std", "wl_1_2", "wl_3_4", "wl_5_6", "wl_7_8", "wl_9_10", "wl_11_plus",

            // Vocabulary richness / word rarity / function words
            "hapax_ratio", "simpson_d", "yule_k", "avg_word_freq_class", "function_word_ratio",

            // Punctuation rates per word
            "comma_rate", "period_rate", "semicolon_rate", "colon_rate", "exclamation_rate", "question_rate",
            "quote_rate", "apostrophe_rate", "bracket_rate", "dash_rate", "ellipsis_rate", "repeated_punct_rate",

            // Orthography / layout / dialogue habits
            "all_caps_word_ratio", "capitalized_word_ratio", "starts_with_capital", "ends_with_terminal",
            "starts_with_dialogue", "non_ascii_ratio", "spacing_anomaly_rate", "elongation_ratio",
            "starts_with_conjunction",

            // Contractions
            "contraction_ratio",

            // Syntax: POS tag shares and entropy
            "pos_noun", "pos_propn", "pos_verb", "pos_aux", "pos_adj", "pos_adv", "pos_pron", "pos_det",
            "pos_adp", "pos_conj", "pos_num", "pos_part", "pos_intj", "pos_punct", "pos_entropy"
    };

    /**
     * Finds the index slot of a scalar feature by its string identifier.
     *
     * @param name Name of the scalar feature.
     * @return 0-based array index within {@link #SCALAR_NAMES}.
     * @throws IllegalArgumentException If feature name is not found.
     */
    public static int indexOf(String name) {

        for (int i = 0; i < SCALAR_NAMES.length; i++) {
            if (SCALAR_NAMES[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Unknown scalar feature: " + name);
    }

    // Readability feature offset and count
    public static final int READABILITY_OFFSET = 11;
    public static final int READABILITY_COUNT = 3;

    // Total count of scalar features before profile blocks
    public static final int SCALAR_COUNT = SCALAR_NAMES.length;

    // Pre-calculated feature index lookups
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

    /** Number of comparison measures produced per profile block (cosine, JSD, TV, Jaccard). */
    public static final int MEASURES_PER_BLOCK = 4;
    public static final String[] MEASURE_NAMES = { "cosine", "jsd", "tv", "jaccard" };

    // Standardisation parameters fitted on training pairs and sentences
    private double[] blockDifferenceMeans;
    private double[] blockDifferenceStds;
    private double[] scalarMeans;
    private double[] scalarStds;

    // Word rank mapping for frequent word list (1 = most frequent)
    private Map<String, Integer> frequentWordRank;

    /** Supported emotion categories from the NRC Emotion Lexicon. */
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
    private boolean emotionLoadAttempted = false;

    // Corpus training metrics for IDF and TF-IDF calculation
    private final List<String> vocabulary;
    private final Map<String, Integer> documentFrequency;
    private final int corpusSize;

    // Feature computation options
    private final FeatureConfig config;

    // Profile block metadata structures
    private final String[] blockIds;
    private final int[] blockOffsets;
    private final int[] blockSizes;

    private final SentencePair.Representation representation;

    private transient PosTagger posTagger;
    private transient volatile boolean prepared = false;

    /**
     * Constructs a feature extractor initialized with training corpus text and default system properties.
     *
     * @param trainingCorpus List of training sentence strings used for IDF computation.
     */
    public FeatureExtractor(List<String> trainingCorpus) {
        this(trainingCorpus, FeatureConfig.fromSystemProperties());
    }

    /**
     * Constructs a feature extractor initialized with training corpus text and a specific configuration.
     *
     * @param trainingCorpus List of training sentence strings.
     * @param config Feature configuration settings.
     */
    public FeatureExtractor(List<String> trainingCorpus, FeatureConfig config) {

        config.validate();
        this.config = config.copy();
        this.representation = SentencePair.Representation.parse(this.config.pairRepresentation);

        // Layout initialization for profile blocks
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

        if (trainingCorpus == null) {
            this.corpusSize = 0;
            this.documentFrequency = new HashMap<>();
            this.vocabulary = new ArrayList<>();
            return;
        }

        this.corpusSize = trainingCorpus.size();
        this.documentFrequency = new HashMap<>();

        // Process vocabulary and document frequency counters
        for (String sentence : trainingCorpus) {

            Set<String> uniqueWords = new HashSet<>(TextUtils.tokenize(scalarText(sentence)));

            for (String word : uniqueWords) {
                documentFrequency.merge(word, 1, Integer::sum);
            }
        }

        this.vocabulary = new ArrayList<>(documentFrequency.keySet());
        Collections.sort(this.vocabulary);
    }

    /**
     * Gets a copy of the active feature configuration.
     *
     * @return Cloned {@link FeatureConfig} instance.
     */
    public FeatureConfig getConfig() {
        return config.copy();
    }

    /** Returns the total number of profile blocks appending scalar features. */
    public int blockCount() {
        return blockIds.length;
    }

    /** Returns a cloned array of profile block identifiers. */
    public String[] blockIds() {
        return blockIds.clone();
    }

    /** Gets the starting column index for a given profile block. */
    public int blockOffset(int block) {
        return blockOffsets[block];
    }

    /** Gets the bucket width for a given profile block. */
    public int blockSize(int block) {
        return blockSizes[block];
    }

    /**
     * Total vector size for a single sentence (scalar count + all profile block widths).
     *
     * @return Total integer length.
     */
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

    /** Returns the current sentence pair representation model. */
    public SentencePair.Representation representation() {
        return representation;
    }

    /** Returns the total column width allocated to scalar features in pair vectors. */
    public int scalarPairWidth() {
        return SCALAR_COUNT * representation.multiplier();
    }

    /** Starting column position for profile distance measures in pair vectors. */
    public int measureOffset() {
        return scalarPairWidth();
    }

    /** Column index storing the overall calculated feature distance. */
    public int distanceColumn() {
        return scalarPairWidth() + blockIds.length * MEASURES_PER_BLOCK;
    }

    /** Total column count of the compressed pair feature vector. */
    public int compressedPairCount() {
        return distanceColumn() + 1;
    }

    /** Total column count of the full pair feature vector including raw bucket differences. */
    public int fullPairCount() {
        return compressedPairCount() + totalBlockWidth();
    }

    /** Returns the actual width of pair vectors generated by this extractor based on settings. */
    public int pairVectorLength() {
        return config.fullBlockDifferences ? fullPairCount() : compressedPairCount();
    }

    /**
     * Generates descriptive column names for all feature dimensions in pair vectors.
     *
     * @return Array of column header strings.
     */
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
     * Finds column indices corresponding to a specific {@link FeatureGroup} for ablation testing.
     *
     * @param group Target feature group.
     * @return Array of column indices assigned to the group.
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
     * Retrieves column indices corresponding to absolute z-score differences of scalar features.
     *
     * @return Array of column indices, or {@code null} if representation lacks absolute differences.
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

    /** Reads and indexes the frequent words file to determine word rank. */
    public synchronized void loadFrequentWords() {
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

    /** Reads and indexes the NRC Emotion Lexicon dataset. */
    public synchronized void loadWordEmotionData() {

        emotionLoadAttempted = true;

        try (BufferedReader reader = new BufferedReader(new FileReader(sentimentAnalysisFilePath))) {

            String line;

            while ((line = reader.readLine()) != null) {

                String[] parts = line.split("\\s+");

                if (parts.length != 3) {
                    continue;
                }

                String word = parts[0].toLowerCase();
                String emotion = parts[1].toUpperCase();
                int association = Integer.parseInt(parts[2]);

                WordEmotion wordEmotion;

                try {
                    wordEmotion = WordEmotion.valueOf(emotion);

                } catch (IllegalArgumentException e) {
                    continue;
                }

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
     * Prepares resources (lexicons, POS tagger) once before feature extraction starts.
     */
    public void prepare() {

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
     * Extracts raw NRC emotion scores across tokens in a sentence.
     *
     * @param tokens Array of sentence tokens.
     * @return Cumulative scores array for each emotion category.
     */
    private double[] getEmotionScores(String[] tokens) {

        double[] emotionScores = new double[WordEmotion.values().length];

        for (String token : tokens) {
            int[] scores = wordEmotionData.get(cleanLower(token));

            if (scores != null) {
                for (int i = 0; i < scores.length; i++) {
                    emotionScores[i] += scores[i];
                }
            }
        }

        return emotionScores;
    }

    /**
     * Calculates average non-zero emotion score across emotional dimensions.
     *
     * @param emotionScores Array of accumulated emotion scores.
     * @return Mean score across active emotion categories.
     */
    private double getAverageEmotionScore(double[] emotionScores) {

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

    /**
     * Strips non-alphabetic characters (preserving inner apostrophes) and converts token to lowercase.
     *
     * @param token Input word token.
     * @return Lowercase string containing letters and apostrophes only.
     */
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

    /**
     * Resolves text input casing strategy for scalar feature calculation.
     *
     * @param sentence Raw input sentence string.
     * @return Sentence string adjusted for scalar feature processing.
     */
    private String scalarText(String sentence) {
        String text = sentence == null ? "" : sentence.trim();
        return config.casing == FeatureConfig.Casing.LOWERCASE ? TextUtils.applyCasing(text, config.casing) : text;
    }

    /**
     * Resolves text input casing strategy for profile block processing.
     *
     * @param sentence Raw input sentence string.
     * @return Sentence string adjusted for profile block generation.
     */
    private String profileText(String sentence) {
        String text = sentence == null ? "" : sentence.trim();
        return config.casing == FeatureConfig.Casing.PRESERVE ? text : text.toLowerCase(Locale.ROOT);
    }

    /**
     * Calculates hashed character n-gram frequencies.
     *
     * @param text Input text string.
     * @param n N-gram length (1 to 5).
     * @param buckets Number of hash buckets.
     * @param typed {@code true} to restrict counts to affix and punctuation n-grams.
     * @return Normalized relative frequency array across buckets.
     */
    double[] computeCharNGrams(String text, int n, int buckets, boolean typed) {

        double[] nGramVector = new double[buckets];
        int total = 0;

        for (int i = 0; i + n <= text.length(); i++) {

            if (typed && !isAffixOrPunctuation(text, i, i + n)) {
                continue;
            }

            int bucket = Math.floorMod(TextUtils.hash(text, i, i + n), buckets);
            nGramVector[bucket]++;
            total++;
        }

        if (total > 0) {
            for (int b = 0; b < buckets; b++) {
                nGramVector[b] /= total;
            }
        }

        return nGramVector;
    }

    /**
     * Determines whether a substring n-gram corresponds to an affix or punctuation boundary.
     *
     * @param text Complete sentence text.
     * @param from Substring start index.
     * @param to Substring end index.
     * @return {@code true} if sub-sequence touches whitespace, punctuation, or word boundaries.
     */
    private static boolean isAffixOrPunctuation(String text, int from, int to) {

        for (int i = from; i < to; i++) {
            char c = text.charAt(i);
            if (!Character.isLetterOrDigit(c) && !Character.isWhitespace(c)) {
                return true;
            }
            if (Character.isWhitespace(c)) {
                return true;
            }
        }

        boolean wordStart = from == 0 || !Character.isLetterOrDigit(text.charAt(from - 1));
        boolean wordEnd = to == text.length() || !Character.isLetterOrDigit(text.charAt(to));

        return wordStart || wordEnd;
    }

    /**
     * Computes hashed n-gram profiles over sentence POS tags.
     *
     * @param tags Array of integer POS tag values.
     * @param n N-gram order.
     * @param buckets Number of hash buckets.
     * @return Normalized POS tag profile array.
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
        padded[0] = PosTagset.COUNT;
        System.arraycopy(tags, 0, padded, 1, tags.length);
        padded[padded.length - 1] = PosTagset.COUNT + 1;

        int total = 0;
        for (int i = 0; i + n <= padded.length; i++) {
            profile[Math.floorMod(TextUtils.hash(padded, i, i + n), buckets)]++;
            total++;
        }

        return normalise(profile, total);
    }

    /**
     * Normalizes a raw count array into a relative frequency distribution vector.
     *
     * @param profile Raw profile vector.
     * @param total Total token occurrence count.
     * @return Normalized distribution array.
     */
    private static double[] normalise(double[] profile, int total) {

        if (total > 0) {
            for (int i = 0; i < profile.length; i++) {
                profile[i] /= total;
            }
        }
        return profile;
    }

    /**
     * Calculates Type-Token Ratio (TTR) measuring lexical diversity.
     *
     * @param words Lowercase word token list.
     * @return Double ratio (unique words / total words).
     */
    private double calcTypeTokenRatio(List<String> words) {

        if (words.isEmpty()) {
            return 0.0;
        }
        return (double) new HashSet<>(words).size() / words.size();
    }

    /**
     * Calculates punctuation variety as a fraction of tracked punctuation marks present.
     *
     * @param sentence Target sentence text.
     * @return Proportion of distinct tracked punctuation marks used.
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

        return (double) uniquePunctuation.size() / PUNCTUATION.length;
    }

    /**
     * Calculates Flesch Reading Ease Score (FRES).
     *
     * @param words Total sentence word count.
     * @param syllables Total syllable count across words.
     * @return Double readability score.
     */
    private double calcFleschReadingEaseScore(int words, int syllables) {

        if (words == 0) {
            return 0.0;
        }

        double averageWordsPerSentence = words;
        double averageSyllablesPerWord = (double) syllables / words;

        return 206.835 - (1.015 * averageWordsPerSentence) - (84.6 * averageSyllablesPerWord);
    }

    /**
     * Calculates Gunning Fog Index (GFI) estimating formal education years required.
     *
     * @param words Total sentence word count.
     * @param complexWords Count of complex words ($\ge 3$ syllables).
     * @return Double index score.
     */
    private double calcGunningFogIndex(int words, int complexWords) {

        if (words == 0) {
            return 0.0;
        }

        double averageWordsPerSentence = words;
        double percentageComplexWords = ((double) complexWords / words) * 100;

        return 0.4 * (averageWordsPerSentence + percentageComplexWords);
    }

    /**
     * Calculates Flesch-Kincaid Grade Level (FKGL) indicating required school grade level.
     *
     * @param words Total sentence word count.
     * @param syllables Total syllable count across words.
     * @return Double grade level score.
     */
    private double calcFleschKincaidGradeLevel(int words, int syllables) {

        if (words == 0) {
            return 0.0;
        }

        double averageWordsPerSentence = words;
        double averageSyllablesPerWord = (double) syllables / words;

        return (0.39 * averageWordsPerSentence) + (11.8 * averageSyllablesPerWord) - 15.59;
    }

    private static final Set<String> SENTENCE_STARTERS = new HashSet<>(
            Arrays.asList("and", "but", "so", "or", "because", "yet", "then", "though", "although"));

    /**
     * Extracts full feature vector (scalar metrics followed by profile blocks) for a given sentence.
     *
     * @param sentence Target input sentence string.
     * @return Array of double feature values.
     */
    public double[] getFeatures(String sentence) {

        prepare();

        String text = scalarText(sentence);
        String profile = profileText(sentence);

        String wordText = text.replace('’', '\'');

        String[] tokens = wordText.isEmpty() ? new String[0] : wordText.split("\\s+");

        double[] scalarFeatures = new double[SCALAR_COUNT];

        int nWords = tokens.length;
        int nChars = text.length();

        double charDensity = Math.max(nChars, 1);
        double wordDensity = Math.max(nWords, 1);

        int frequentCount = 0;

        for (String token : tokens) {
            if (frequentWordRank != null && frequentWordRank.containsKey(cleanLower(token))) {
                frequentCount++;
            }
        }

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

            if (Character.isDigit(ch)) {
                digitsCount++;
            } else if (Character.isLetter(ch)) {

                letterCount++;

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
                    if (previous == '.') {
                        break;
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

            if ((ch == '!' || ch == '?' || ch == ',' || ch == ';' || ch == ':') && next == ch && previous != ch) {
                repeated++;
            }

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

        List<String> words = TextUtils.tokenize(wordText);
        int nW = words.size();
        double wDensity = Math.max(nW, 1);

        double[] emotionScores = getEmotionScores(tokens);
        double averageEmotionScore = getAverageEmotionScore(emotionScores);

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

            int wordSyllables = TextUtils.countSyllables(word.replace("'", ""));
            syllables += wordSyllables;

            if (wordSyllables >= 3) {
                complexWords++;
            }

            if (frequentWordRank != null) {
                Integer rank = frequentWordRank.get(lower);
                freqClassSum += InformationTheory.log2(1.0 + (rank == null ? frequentWordRank.size() + 1 : rank));
            }

            if (addFunctionWord(lower, functionWordCounts)) {
                functionWordTokens++;
            }
        }

        int hapax = 0;
        double repeatPairs = 0.0;
        double yuleSum = 0.0;
        for (int count : wordCounts.values()) {
            if (count == 1) {
                hapax++;
            }
            repeatPairs += (double) count * (count - 1);
            yuleSum += (double) count * count;
        }

        double meanLength = nW == 0 ? 0.0 : lengthSum / nW;
        double lengthVariance = nW == 0 ? 0.0 : Math.max(0.0, lengthSquares / nW - meanLength * meanLength);

        List<String> posTokens = TextUtils.tokenizeWithPunctuation(wordText);
        int[] tags = posTagger.tag(posTokens);

        scalarFeatures[I_LOG_WORDS] = Math.log1p(nWords);
        scalarFeatures[I_LOG_CHARS] = Math.log1p(nChars);
        scalarFeatures[I_FREQUENT] = frequentCount / wordDensity;
        scalarFeatures[I_PUNCT_DENSITY] = punctuationCount / charDensity;
        scalarFeatures[I_DIGIT] = digitsCount / charDensity;
        scalarFeatures[I_UPPER] = upperCaseCount / charDensity;
        scalarFeatures[I_AVG_WORD] = letterCount / wordDensity;
        scalarFeatures[I_TFIDF] = calcTFIDFScore(words);
        scalarFeatures[I_EMOTION] = averageEmotionScore / wordDensity;
        scalarFeatures[I_TTR] = calcTypeTokenRatio(lowerWords);
        scalarFeatures[I_PUNCT_VARIETY] = punctuationVariety(wordText);
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
        scalarFeatures[I_CAPITALIZED] = (double) capitalised / Math.max(nW - 1, 1);
        scalarFeatures[I_STARTS_CAPITAL] = startsWithCapital(wordText) ? 1.0 : 0.0;
        scalarFeatures[I_ENDS_TERMINAL] = endsWithTerminal(wordText) ? 1.0 : 0.0;
        scalarFeatures[I_STARTS_DIALOGUE] = startsWithDialogue(wordText) ? 1.0 : 0.0;
        scalarFeatures[I_NON_ASCII] = nonAscii / charDensity;
        scalarFeatures[I_SPACING] = spacingAnomalies / charDensity;
        scalarFeatures[I_ELONGATION] = elongated / wDensity;
        scalarFeatures[I_STARTS_CONJ] = nW > 0 && SENTENCE_STARTERS.contains(lowerWords.get(0)) ? 1.0 : 0.0;

        scalarFeatures[I_CONTRACTION] = contractions / wDensity;

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
                entropy -= share * InformationTheory.log2(share);
            }
        }
        scalarFeatures[I_POS_ENTROPY] = entropy / InformationTheory.log2(PosTagset.COUNT);

        double[] features = new double[sentenceVectorLength()];
        System.arraycopy(scalarFeatures, 0, features, 0, SCALAR_COUNT);

        for (int b = 0; b < blockIds.length; b++) {

            double[] block;
            String id = blockIds[b];

            if (id.startsWith("char")) {
                block = computeCharNGrams(profile, id.charAt(4) - '0', blockSizes[b], config.charTyped);

            } else if (id.startsWith("pos")) {
                block = computePosNGrams(tags, id.charAt(3) - '0', blockSizes[b]);

            } else {
                block = normalise(functionWordCounts, nW);
            }

            System.arraycopy(block, 0, features, blockOffsets[b], blockSizes[b]);
        }

        return features;
    }

    /**
     * Checks and increments occurrences in the function word counts profile.
     * Splits contractions into base word and clitic pieces where applicable.
     *
     * @param lower Lowercase word token.
     * @param counts Destination count array for function words.
     * @return {@code true} if word contains at least one function word piece.
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

    /**
     * Checks if sentence starts with a capital letter.
     *
     * @param text Sentence text string.
     * @return {@code true} if first alphabetic character is uppercase.
     */
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

    /**
     * Checks if sentence ends with a terminal punctuation mark.
     *
     * @param text Sentence text string.
     * @return {@code true} if trailing character sequence ends in terminal mark.
     */
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

    /**
     * Checks if sentence starts with dialogue markers or quotation characters.
     *
     * @param text Sentence text string.
     * @return {@code true} if leading character is a dialogue or quotation mark.
     */
    private static boolean startsWithDialogue(String text) {

        if (text.isEmpty()) {
            return false;
        }

        char c = text.charAt(0);
        return c == '"' || c == '“' || c == '‘' || c == '\'' || c == '-' || c == '–' || c == '—'
                || c == '«';
    }

    /**
     * Calculates Inverse Document Frequency (IDF) score for a term over the corpus.
     *
     * @param term Target word string.
     * @return Smoothed double IDF value.
     */
    private double calculateIDF(String term) {
        int docFreq = documentFrequency.getOrDefault(term, 0);
        return Math.log((double) (corpusSize + 1) / (docFreq + 1)) + 1;
    }

    /**
     * Calculates array of individual TF-IDF weights for each word in a list.
     *
     * @param words Input word token list.
     * @return Array of double TF-IDF values.
     */
    private double[] calcTFIDF(List<String> words) {

        Map<String, Integer> termCounts = new HashMap<>();
        for (String word : words) {
            termCounts.merge(word, 1, Integer::sum);
        }

        double[] tfidf = new double[words.size()];
        for (int i = 0; i < words.size(); i++) {
            String term = words.get(i);
            double tf = (double) termCounts.get(term) / words.size();
            double idf = calculateIDF(term);
            tfidf[i] = tf * idf;
        }
        return tfidf;
    }

    /**
     * Calculates average TF-IDF score for a sentence.
     *
     * @param words Input word token list.
     * @return Mean double TF-IDF score across sentence tokens.
     */
    private double calcTFIDFScore(List<String> words) {

        double[] tfidfScores = calcTFIDF(words);
        double totalScore = 0.0;
        for (double score : tfidfScores) {
            totalScore += score;
        }

        if (tfidfScores.length == 0) {
            return 0.0;
        }

        return totalScore / tfidfScores.length;
    }

    /**
     * Calculates absolute point-by-point differences between two feature vectors.
     *
     * @param features1 First feature array.
     * @param features2 Second feature array.
     * @return Array of absolute differences.
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

    /**
     * The pair vectors of all neighbouring sentences of a document. Every sentence is extracted once,
     * although it takes part in two pairs.
     *
     * @param sentences the sentences of one document, in order
     * @return one pair vector per boundary, {@code sentences - 1} rows
     */
    public double[][] neighbourPairVectors(List<String> sentences) {

        List<double[]> raw = new ArrayList<>(sentences.size());
        for (String sentence : sentences) {
            raw.add(getFeatures(sentence));
        }

        double[][] matrix = new double[Math.max(0, sentences.size() - 1)][];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = pairVector(raw.get(i), raw.get(i + 1));
        }
        return matrix;
    }

    /**
     * Fits standard deviation and mean parameters across training sentence scalar features.
     *
     * @param trainingFeatures List of feature vectors extracted from training data.
     */
    public void fitScaler(List<double[]> trainingFeatures) {

        if (trainingFeatures == null || trainingFeatures.isEmpty()) {
            throw new IllegalArgumentException("Cannot fit the scaler without training features");
        }

        scalarMeans = Statistics.columnMeans(trainingFeatures, 0, SCALAR_COUNT);
        scalarStds = Statistics.columnStandardDeviations(trainingFeatures, 0, scalarMeans);
        Statistics.replaceBelow(scalarStds, 1e-12, 1.0);
    }

    /** Checks whether the scalar standardizer has been fitted. */
    public boolean isFitted() {
        return scalarMeans != null && scalarStds != null;
    }

    /**
     * Ensures loaded model properties match current class scalar configurations.
     *
     * @throws IllegalStateException If configured feature sizes mismatch stored model values.
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
     * Standardizes scalar feature values to z-scores based on fitted dataset statistics.
     *
     * @param features Raw extracted feature array.
     * @return Cloned array with standardized scalar components.
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

    /**
     * Generates floating-point embedding input vectors suitable for deep neural models.
     *
     * @param raw Raw feature vector.
     * @param profileScale Weight factor for profile distributions.
     * @return Standardized float array representation.
     */
    public float[] embeddingInput(double[] raw, double profileScale) {

        double[] z = standardize(raw);
        float[] input = new float[raw.length];

        for (int i = 0; i < SCALAR_COUNT; i++) {
            input[i] = (float) z[i];
        }
        // Indices SCALAR_COUNT to raw.length keep ALL raw bucket values:
        for (int i = SCALAR_COUNT; i < raw.length; i++) {
            input[i] = (float) (Math.sqrt(Math.max(0.0, raw[i])) * profileScale);
        }

        return input;
    }

    /**
     * Populates profile distance metrics and optional bucket differences into a pair vector.
     *
     * @param features1 Raw features of sentence 1.
     * @param features2 Raw features of sentence 2.
     * @param out Target output pair array.
     */
    public void writeProfilePart(double[] features1, double[] features2, double[] out) {

        for (int b = 0; b < blockIds.length; b++) {
            DistributionDistances.compare(features1, features2, blockOffsets[b], blockSizes[b], out,
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
     * Populates scalar feature representation transforms and Euclidean distance metrics into a pair vector.
     *
     * @param raw1 Raw scalar features of sentence 1.
     * @param raw2 Raw scalar features of sentence 2.
     * @param z1 Standardized z-score array of sentence 1.
     * @param z2 Standardized z-score array of sentence 2.
     * @param out Target output pair array.
     */
    public void writeScalarPart(double[] raw1, double[] raw2, double[] z1, double[] z2, double[] out) {

        SentencePair.writeScalarPart(representation, raw1, raw2, z1, z2, SCALAR_COUNT, out);

        out[distanceColumn()] = VectorMath.rootMeanSquareDifference(z1, z2, SCALAR_COUNT);
    }

    /**
     * Constructs the complete pair feature vector comparing two sentence feature arrays.
     *
     * @param raw1 Raw features of sentence 1.
     * @param raw2 Raw features of sentence 2.
     * @return Complete pair feature vector.
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
     * Fits z-score normalization parameters over raw profile block differences in full pair vectors.
     *
     * @param fullPairFeatures Matrix of full pair feature vectors.
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

        List<double[]> rows = Arrays.asList(fullPairFeatures);
        blockDifferenceMeans = Statistics.columnMeans(rows, first, nColumns);
        blockDifferenceStds = Statistics.columnStandardDeviations(rows, first, blockDifferenceMeans);
        Statistics.replaceBelow(blockDifferenceStds, 1e-12, 1.0);
    }

    /**
     * Standardizes raw profile bucket differences in a full pair feature vector.
     *
     * @param fullPairFeatures Full pair feature vector.
     * @return Standardized full pair vector.
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

    /** Returns a concise string summary describing configuration layout dimensions. */
    public String describe() {
        return config.signature() + " -> " + SCALAR_COUNT + " scalars + " + blockIds.length + " profile blocks ("
                + Arrays.toString(blockIds) + "), pair vector " + pairVectorLength();
    }
}