package com.hyp.authorswitchdetection.service;

import java.util.ArrayList;
import java.util.List;

/** What the Retrieve page shows: the similarity of every two sentences and the nearest neighbours of each. */
public class RetrievalResult {

    public static class Neighbour {
        public int index;          // 1-based
        public double similarity;
        public String text;
    }

    public static class SentenceView {
        public int index;          // 1-based
        public String text;
        public List<Neighbour> nearest = new ArrayList<>();
    }

    public String error;
    public String modelDescription;
    public int nSentences;
    public boolean truncated;          // more sentences were pasted than the heatmap shows

    public double[][] similarity;      // cosine similarity, -1..1
    public String[][] cellColor;       // CSS colour of every cell (darker = more similar)
    public double minSimilarity, maxSimilarity;   // of the off-diagonal cells: the colour scale

    public double[] changeScore;       // between sentence i and i + 1, 0..1
    public boolean[] predictedSwitch;
    public double threshold;
    public int window;

    public List<SentenceView> sentences = new ArrayList<>();
}
