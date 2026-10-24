package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.DocumentAnalysis;
import com.hyp.authorswitchdetection.service.DocumentAnalysisService;
import com.hyp.authorswitchdetection.service.TextAnalysisService;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * The Analyse page: one document in, its difficulty, raw stylometric features, estimated number of authors and
 * the exact positions of the predicted author changes out.
 */
@Controller
@RequestMapping("/analyse")
public class AnalyseController {

    private final TextAnalysisService textAnalysisService;
    private final DocumentAnalysisService documentAnalysisService;
    private final CurrentUserService currentUserService;

    public AnalyseController(TextAnalysisService textAnalysisService,
                             DocumentAnalysisService documentAnalysisService,
                             CurrentUserService currentUserService) {
        this.textAnalysisService = textAnalysisService;
        this.documentAnalysisService = documentAnalysisService;
        this.currentUserService = currentUserService;
    }

    @GetMapping
    public String form(Model model, HttpSession session) {
        addCurrentUser(model, session);
        return "analyse";
    }

    @PostMapping
    public String analyse(@RequestParam(required = false) String text,
                          @RequestParam(required = false) MultipartFile file,
                          Model model, HttpSession session) throws IOException {

        addCurrentUser(model, session);

        String inputText = file != null && !file.isEmpty()
                ? new String(file.getBytes(), StandardCharsets.UTF_8)
                : (text == null ? "" : text);
        model.addAttribute("inputText", inputText);

        List<String> sentences = textAnalysisService.splitSentences(inputText);
        if (sentences.isEmpty()) {
            model.addAttribute("error", "No text to analyse. Paste some text (one sentence per line) or upload a .txt file.");
            return "analyse";
        }

        DocumentAnalysis analysis = documentAnalysisService.analyse(sentences);
        model.addAttribute("analysis", analysis);
        return "analyse";
    }

    private void addCurrentUser(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
    }
}
