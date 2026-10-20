package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.RetrievalResult;
import com.hyp.authorswitchdetection.service.RetrievalService;
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

/** The retrieval model: which sentences of a text are written alike. */
@Controller
@RequestMapping("/retrieve")
public class RetrieveController {

    private final TextAnalysisService textAnalysisService;
    private final RetrievalService retrievalService;
    private final CurrentUserService currentUserService;

    public RetrieveController(TextAnalysisService textAnalysisService, RetrievalService retrievalService,
            CurrentUserService currentUserService) {
        this.textAnalysisService = textAnalysisService;
        this.retrievalService = retrievalService;
        this.currentUserService = currentUserService;
    }

    @GetMapping
    public String form(Model model, HttpSession session) {
        addCommon(model, session);
        return "retrieve";
    }

    @PostMapping
    public String analyze(@RequestParam(required = false) String text,
            @RequestParam(required = false) MultipartFile file,
            @RequestParam(required = false, defaultValue = "3") int neighbours, Model model, HttpSession session)
            throws IOException {

        addCommon(model, session);

        String inputText = (file != null && !file.isEmpty()) ? new String(file.getBytes(), StandardCharsets.UTF_8)
                : (text == null ? "" : text);
        model.addAttribute("inputText", inputText);

        List<String> sentences = textAnalysisService.splitSentences(inputText);
        if (sentences.size() < 2) {
            model.addAttribute("error", "Need at least two sentences (one per line).");
            return "retrieve";
        }

        RetrievalResult result = retrievalService.analyze(sentences, Math.max(1, Math.min(neighbours, 10)));
        if (result.error != null) {
            model.addAttribute("error", result.error);
        } else {
            model.addAttribute("result", result);
        }
        return "retrieve";
    }

    private void addCommon(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
    }
}
