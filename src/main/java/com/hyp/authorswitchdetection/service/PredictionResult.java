package com.hyp.authorswitchdetection.service;

import java.util.ArrayList;
import java.util.List;

public class PredictionResult {

    public static class Pair {
        public int index;
        public String sentenceA;
        public String sentenceB;
        public boolean isSwitch;
        public Double probability;               // null for a model without probabilities (decision tree)
        public Integer truth;                    // null when no truth file was uploaded
        public String outcome;                   // TP / FP / TN / FN, null without truth
        public List<String> reasons = new ArrayList<>(); // why the model decided this (decision tree)

        public boolean isCorrect() {
            return truth != null && (truth == 1) == isSwitch;
        }
    }

    public int nSentences;
    public int nPairs;
    public String modelName;
    public String modelDescription;  // the settings the model was trained with
    public String error;             // why no prediction was made (no model, model out of date, ...)
    public boolean hasThreshold;     // false for a decision tree: it outputs a class label, not a probability
    public double threshold;
    public String detectedLevel;          // easy / medium / hard as detected from the text, null = not detected
    public String levelProbabilities;     // "easy 0.12 | medium 0.30 | hard 0.58"
    public String thresholdNote;          // where the threshold comes from, or why the level could not be detected
    public List<Pair> pairs = new ArrayList<>();
    public int nSwitches;
    public int nSegments;            // nSwitches + 1: the text cut at every predicted switch
    public List<List<Integer>> authorSegments = new ArrayList<>(); // sentence indices per segment

    public List<String> modelRules = new ArrayList<>(); // the whole model as rules (decision tree)

    public boolean hasTruth;         // truth file uploaded and usable
    public String truthError;        // why the truth file could not be used
    public EvaluationMetrics metrics;
}
