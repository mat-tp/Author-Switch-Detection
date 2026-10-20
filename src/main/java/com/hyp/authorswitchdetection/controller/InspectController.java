package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.InspectResult;
import com.hyp.authorswitchdetection.service.InspectService;
import com.hyp.authorswitchdetection.service.TextAnalysisService;
import com.hyp.authorswitchdetection.service.classifier.ClassifierRegistry;
import features.FeatureExtractor;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Optional;

@Controller
@RequestMapping("/inspect")
public class InspectController {

    // the features are defined by the fitted extractor that was saved with a model
    
    private final TextAnalysisService textAnalysisService;
    private final InspectService inspectService;
    private final CurrentUserService currentUserService;
    private final ClassifierRegistry classifierRegistry;

    public InspectController(TextAnalysisService textAnalysisService, InspectService inspectService,
            CurrentUserService currentUserService, ClassifierRegistry classifierRegistry) {
        this.classifierRegistry = classifierRegistry;
        this.textAnalysisService = textAnalysisService;
        this.inspectService = inspectService;
        this.currentUserService = currentUserService;
    }

    @GetMapping
    public String form(Model model, HttpSession session) {
        addCommon(model, session, "decision_tree");
        return "inspect";
    }

    @PostMapping
    public String inspect(@RequestParam(required = false) String text,
                           @RequestParam(required = false) MultipartFile file,
                           @RequestParam(required = false, defaultValue = "decision_tree") String modelName,
                           Model model, HttpSession session) throws IOException {
        addCommon(model, session, modelName);

        String inputText = resolveInputText(text, file);
        model.addAttribute("inputText", inputText);

        List<String> sentences = textAnalysisService.splitSentences(inputText);
        if (sentences.isEmpty()) {
            model.addAttribute("error", "No text to inspect. Paste some text or upload a .txt file.");
            return "inspect";
        }

        Optional<FeatureExtractor> extractor = inspectService.extractorFor(modelName);

        // every sentence is shown, no limit
        InspectResult result;
        if (extractor.isPresent()) {
            result = inspectService.inspect(sentences, modelName, extractor.get());
        } else {
            // no trained model: the features can still be shown, relative to this text only
            String problem = inspectService.problemWith(modelName);
            result = inspectService.inspect(sentences, modelName, inspectService.extractorFromText(sentences));
            result.note = (problem != null
                    ? "The model '" + modelName + "' can not be used (" + problem + ")."
                    : "The model '" + modelName + "' has not been trained yet.")
                    + " The features below are computed with the default settings and an extractor fitted on this text "
                    + "only, so the z-scores say how a sentence differs from the rest of THIS text, not from the training data.";
        }
        model.addAttribute("result", result);
        return "inspect";
    }

    private String resolveInputText(String text, MultipartFile file) throws IOException {
        if (file != null && !file.isEmpty()) {
            String name = file.getOriginalFilename() == null ? "" : file.getOriginalFilename().toLowerCase();
            if (name.endsWith(".pdf")) {
                // TODO: wire in a PDF text extractor (e.g. Apache PDFBox) here.
                return "";
            }
            return new String(file.getBytes(), StandardCharsets.UTF_8);
        }
        return text == null ? "" : text;
    }

    private void addCommon(Model model, HttpSession session, String selectedModel) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
        model.addAttribute("modelOptions", classifierRegistry.options());
        model.addAttribute("selectedModel", selectedModel);
    }
}
