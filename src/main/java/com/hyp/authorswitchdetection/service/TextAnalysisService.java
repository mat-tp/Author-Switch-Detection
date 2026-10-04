package com.hyp.authorswitchdetection.service;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Sentence splitting shared by the Inspect and Predict pages. Sentences are
 * assumed to be newline-separated, as the UI tells the user ("one sentence per
 * line"), the same way the dataset's problem-N.txt files are read.
 */
@Service
public class TextAnalysisService {

    public List<String> splitSentences(String text) {
        List<String> sentences = new ArrayList<>();
        if (text == null) return sentences;
        for (String line : text.split("\\R")) {
            String s = line.trim();
            if (!s.isEmpty()) sentences.add(s);
        }
        return sentences;
    }
}
