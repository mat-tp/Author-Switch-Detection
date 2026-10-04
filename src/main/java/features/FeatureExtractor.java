package features;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.io.FileNotFoundException;
import java.util.*;

public class FeatureExtractor implements Serializable {

    private static final long serialVersionUID = 1L;
    // Paths for the files used to get the features (Function words And Word-Emotion
    // which is associated with SentimentAnalysis)
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
    private static final char[] PUNCTUATION = { '!', ';', ',', '.', '?', '\'', '"' };
    // bucket for holding hashed values for sentences
    private static final int NGRAM_BUCKETS = 256;
    // number of scalar features before the n-gram block
    public static final int SCALAR_COUNT = 9;

    // number of features in the compressed pair vector (see
    // SentencePair.getPairFeatures)
    public static final int COMPRESSED_PAIR_COUNT = SCALAR_COUNT + 2;

    // z-score parameters of the n-gram bucket differences of the FULL pair
    // features (the columns after the compressed ones), fitted on the training
    // pairs
    private double[] nGramDifferenceMeans;
    private double[] nGramDifferenceStds;

    // z-score parameters of the scalar features is fitted on the training sentences
    private double[] scalarMeans;
    private double[] scalarStds;

    private Set<String> frequentWords;

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

    // training data information for features
    private final List<String> vocabulary;
    private final Map<String, Integer> documentFrequency;
    private final int corpusSize;

    /**
     * Providing the Constructor with the Training data for calculating the term
     * frequency, IDF-TF
     * 
     * @param trainingCorpus
     */
    public FeatureExtractor(List<String> trainingCorpus) {

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

            // getting the unique words all lower-cased to reduce the size
            Set<String> uniqueWords = new HashSet<>(TextUtils.tokenize(sentence));

            for (String word : uniqueWords) {
                // aggregate each word in the unique word List
                documentFrequency.merge(word, 1, Integer::sum);
            }
        }

        // Update the vocab and ordering by sorting
        this.vocabulary = new ArrayList<>(documentFrequency.keySet());
        Collections.sort(this.vocabulary);
    }

    /*
     * Extracting function words based on the external function words list
     */

    public void loadFrequentWords() {
        // O(1) for add, search & remove
        frequentWords = new HashSet<String>();

        try (BufferedReader reader = new BufferedReader(new FileReader(frequentWordsFilePath))) {

            String line;
            while ((line = reader.readLine()) != null) {
                frequentWords.add(line.trim().toLowerCase());
            }

        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + frequentWordsFilePath);

        } catch (IOException e) {
            System.err.println("Error reading file: " + frequentWordsFilePath);
        }
    }

    /**
     * Loads NRC-Emotion data to the memory for extracting features further.
     * 
     */
    public void loadWordEmotionData() {

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
            System.err.println("File not found: " + sentimentAnalysisFilePath);

        } catch (IOException e) {
            System.err.println("Error reading file: " + sentimentAnalysisFilePath);
        }
    }

    /**
     * Getting the emotion scores for each word in a sentence
     * 
     * @param sentence Current sentence
     * @return scores values for each emotion
     */
    private double[] getEmotionScores(String sentence) {

        // load it to memory
        if (wordEmotionData.isEmpty()) {
            loadWordEmotionData();
        }

        double[] emotionScores = new double[WordEmotion.values().length];
        String[] tokens = sentence.split("\\s+");

        for (String token : tokens) {

            // getting the words only as the file(NCR-Word-emotion) has words
            String cleanedToken = token.toLowerCase().replaceAll("[^a-zA-Z']", "");
            int[] scores = wordEmotionData.get(cleanedToken);

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

    /**
     * Hashed N-Gram Implementation
     * To automatically handle unsceen words in n-grams, which becomes predictable
     * 
     * @param
     * @return
     */
    double[] computeCharNGrams(String sentence, int buckets) {
        // sentence = sentence.toLowerCase(); // lower case a sentence
        double[] nGramVector = new double[buckets];
        int total = 0;

        /*
         * Instead of relying on another corpus to extract the N-Gram, am relying on
         * hashes
         * For easy computation.
         */
        for (int i = 0; i + 3 <= sentence.length(); i++) {
            int hashCode = Math.floorMod(sentence.substring(i, i + 3).hashCode(), buckets);
            nGramVector[hashCode]++;
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
     * Getting the features from each sentence
     * This defines the features used in the project All together
     * 
     * @param sentence Current sentence that am extracting features from.
     * @return array of features for the given sentence
     * 
     *         Breif of the features:
     *         features[0]: sentence length in words (log-scaled)
     *         features[1]: sentence length in characters (log-scaled)
     *         features[2]: frequent-word ratio
     *         features[3]: punctuation density
     *         features[4]: digit ratio
     *         features[5]: uppercase ratio
     *         features[6]: average word length (letters per word)
     *         features[7]: mean TF-IDF score
     *         features[8]: emotion score (length-normalised)
     *         features[9..264] - hashed character 3-gram frequencies (256 buckets)
     *         - This is likely to create sparse vector especially for short
     *         sentences,
     *         that is why I will be handling it separately.
     * 
     */
    public double[] getFeatures(String sentence) {

        if (frequentWords == null) {
            loadFrequentWords();
        }

        // removing extra spaces
        sentence = sentence.trim();

        String[] tokens;
        if (sentence.isEmpty()) {
            tokens = new String[0];
        } else {
            tokens = sentence.split("\\s+");
        }

        double[] scalarFeatures = new double[SCALAR_COUNT]; // 9 features now...

        int nWords = tokens.length;
        int nChars = sentence.length();

        double charDensity = Math.max(nChars, 1);
        double wordDensity = Math.max(nWords, 1);

        // counting frequent words.
        int frequentCount = 0;

        for (String token : tokens) {

            String cleanedToken = token.toLowerCase().replaceAll("[^a-zA-Z']", "");
            if (frequentWords.contains(cleanedToken)) {
                frequentCount++;
            }
        }

        // counting puctions, uppercase (first words in a sentence or repr nouns),
        // letters and digits
        int punctuationCount = 0;
        int digitsCount = 0;
        int letterCount = 0;
        int upperCaseCount = 0;

        for (int i = 0; i < nChars; i++) {

            char ch = sentence.charAt(i);

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
        }

        /*
         * It is worth considering typos in text from person to person
         * Such as Incorrect Formatting (ALL-UPPER caps words, extra spacings, and so
         * on)
         * This will not be used full in my case since all the text is cleaned properly
         * and
         * UPPER-CAPS words could be representing Abreviations instead.
         */

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
        double[] emotionScores = getEmotionScores(sentence);
        // using the average emotion score instead to reduce feature space
        double averageEmotionScore = getAverageEmotionScore(emotionScores);

        scalarFeatures[0] = Math.log1p(nWords); // sentence length (words)
        scalarFeatures[1] = Math.log1p(nChars); // sentence length (chars)
        scalarFeatures[2] = frequentCount / wordDensity; // frequent-word ratio
        scalarFeatures[3] = punctuationCount / charDensity; // punctuation density
        scalarFeatures[4] = digitsCount / charDensity; // digit ratio
        scalarFeatures[5] = upperCaseCount / charDensity; // uppercase ratio
        scalarFeatures[6] = letterCount / wordDensity; // average word length
        scalarFeatures[7] = calcTFIDFScore(sentence); // mean TF-IDF
        scalarFeatures[8] = averageEmotionScore / wordDensity; // emotion

        double[] nGramFeatures = computeCharNGrams(sentence, NGRAM_BUCKETS);
        double[] features = new double[scalarFeatures.length + nGramFeatures.length];
        // copy both into the new array
        System.arraycopy(scalarFeatures, 0, features, 0, scalarFeatures.length);
        System.arraycopy(nGramFeatures, 0, features, scalarFeatures.length, nGramFeatures.length);

        return features;

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

    /**
     * Calculating the Term-Frequency, Based on the Document(Sentences)
     * 
     * @param term      Word of Interest
     * @param sentences Document
     * @return Term-Frequency Value
     */
    private double calculateTF(String term, List<String> sentences) {

        // Term-frequency: count of word in the doc over total number of words in doc
        int termCount = 0;
        if (sentences == null || sentences.isEmpty()) {
            return 0.0; // Return 0 if sentences list is null or empty
        }

        for (String token : sentences) {
            if (token.equals(term)) {
                termCount++;
            }
        }

        return (double) termCount / sentences.size(); // Term frequency
    }

    // private double[] calcTFIDF(String sentence) {

    // List<String> words = TextUtils.tokenize(sentence);

    // double[] tfidf = new double[words.size()];

    // for (int i = 0; i < words.size(); i++) {

    // String term = words.get(i);
    // double tf = calculateTF(term, words);
    // double idf = calculateIDF(term);

    // tfidf[i] = tf * idf;

    // }

    // return tfidf;
    // }

    private double[] calcTFIDF(String sentence) {

        // precomputing the term frequency for the sentence
        // for efficiency.
        List<String> words = TextUtils.tokenize(sentence);

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
     * @param sentence
     * @return
     */
    private double calcTFIDFScore(String sentence) {

        double[] tfidfScores = calcTFIDF(sentence);
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

    /**
     * Fits the z-score parameters (mean and standard deviation) of the scalar
     * features.
     *
     * Use TRAINING sentences only, so nothing from the validation/test data leaks
     * into the model.
     * The n-gram block is not standardised, it is compared using cosine similarity
     * instead.
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

    /**
     * Z-score of the scalar features: (value - mean) / standard deviation.
     * The n-gram block is copied unchanged, since it is compared using cosine
     * similarity.
     *
     * @param features feature vector from getFeatures
     * @return copy of the feature vector, with the scalar features standardised
     */
    public double[] standardize(double[] features) {

        if (scalarMeans == null || scalarStds == null) {
            throw new IllegalStateException("The scaler is not fitted. Call fitScaler(...) first");
        }

        double[] standardized = features.clone();

        for (int i = 0; i < SCALAR_COUNT; i++) {
            standardized[i] = (features[i] - scalarMeans[i]) / scalarStds[i];
        }

        return standardized;
    }

    /**
     * Fits the z-score parameters (mean and standard deviation) of the n-gram
     * bucket differences in the FULL pair features
     * (SentencePair.getFullPairFeatures).
     * The differences are small numbers (around 0.01), so without this the models
     * would almost ignore them.
     *
     * Use TRAINING pairs only, so nothing from the validation/test data leaks into
     * the model.
     *
     * @param fullPairFeatures full pair feature vectors of the training pairs
     */
    public void fitNGramDifferenceScaler(double[][] fullPairFeatures) {

        if (fullPairFeatures == null || fullPairFeatures.length == 0) {
            throw new IllegalArgumentException("Cannot fit the scaler without training pair features");
        }

        int nColumns = fullPairFeatures[0].length - COMPRESSED_PAIR_COUNT;

        if (nColumns <= 0) {
            throw new IllegalArgumentException("fullPairFeatures must come from SentencePair.getFullPairFeatures");
        }

        nGramDifferenceMeans = new double[nColumns];
        nGramDifferenceStds = new double[nColumns];

        int count = fullPairFeatures.length;

        // means
        for (double[] pairFeatures : fullPairFeatures) {
            for (int i = 0; i < nColumns; i++) {
                nGramDifferenceMeans[i] += pairFeatures[COMPRESSED_PAIR_COUNT + i];
            }
        }

        for (int i = 0; i < nColumns; i++) {
            nGramDifferenceMeans[i] /= count;
        }

        // standard deviations
        for (double[] pairFeatures : fullPairFeatures) {
            for (int i = 0; i < nColumns; i++) {
                double difference = pairFeatures[COMPRESSED_PAIR_COUNT + i] - nGramDifferenceMeans[i];
                nGramDifferenceStds[i] += difference * difference;
            }
        }

        for (int i = 0; i < nColumns; i++) {
            nGramDifferenceStds[i] = Math.sqrt(nGramDifferenceStds[i] / count);

            // a bucket that never differs has no spread, avoiding a division by zero
            if (nGramDifferenceStds[i] < 1e-12) {
                nGramDifferenceStds[i] = 1.0;
            }
        }
    }

    /**
     * Z-score of the n-gram bucket differences of a FULL pair feature vector.
     * The compressed features (the first COMPRESSED_PAIR_COUNT) are copied
     * unchanged.
     *
     * @param fullPairFeatures full pair feature vector
     * @return copy of the vector, with the n-gram differences standardised
     */
    public double[] standardizeNGramDifferences(double[] fullPairFeatures) {

        if (nGramDifferenceMeans == null || nGramDifferenceStds == null) {
            throw new IllegalStateException(
                    "The n-gram difference scaler is not fitted. Call fitNGramDifferenceScaler(...) first");
        }

        if (fullPairFeatures.length != COMPRESSED_PAIR_COUNT + nGramDifferenceMeans.length) {
            throw new IllegalArgumentException("Expected " + (COMPRESSED_PAIR_COUNT + nGramDifferenceMeans.length)
                    + " features, but got " + fullPairFeatures.length);
        }

        double[] standardized = fullPairFeatures.clone();

        for (int i = 0; i < nGramDifferenceMeans.length; i++) {
            standardized[COMPRESSED_PAIR_COUNT + i] = (fullPairFeatures[COMPRESSED_PAIR_COUNT + i]
                    - nGramDifferenceMeans[i]) / nGramDifferenceStds[i];
        }

        return standardized;
    }

    /**
     * Cosine similarity of the hashed character n-gram profiles of two sentences.
     * 1.0 means an identical profile, 0.0 means nothing in common.
     * A sentence too short to have any n-gram gives 0.0.
     *
     * @param features1 features from 1st sentence
     * @param features2 features from 2nd sentence
     * @return cosine similarity between 0 and 1
     */
    public static double calcNGramCosine(double[] features1, double[] features2) {

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = SCALAR_COUNT; i < features1.length; i++) {
            dotProduct += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }

        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
}