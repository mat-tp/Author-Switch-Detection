package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import data.DatasetLoader;
import com.hyp.authorswitchdetection.service.*;
import com.hyp.authorswitchdetection.service.classifier.ClassifierRegistry;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

@Controller
public class PredictController {

    private final TextAnalysisService textAnalysisService;
    private final PredictionService predictionService;
    private final TrainingService trainingService;
    private final ClassifierRegistry classifierRegistry;
    private final CurrentUserService currentUserService;

    public PredictController(TextAnalysisService textAnalysisService, PredictionService predictionService,
                              TrainingService trainingService, ClassifierRegistry classifierRegistry,
                              CurrentUserService currentUserService) {
        this.textAnalysisService = textAnalysisService;
        this.predictionService = predictionService;
        this.trainingService = trainingService;
        this.classifierRegistry = classifierRegistry;
        this.currentUserService = currentUserService;
    }

    @GetMapping("/predict")
    public String form(Model model, HttpSession session) {
        addCommon(model, session);
        return "predict";
    }

    @PostMapping("/predict")
    public String predict(@RequestParam(required = false) String text,
                           @RequestParam(required = false) MultipartFile file,
                           @RequestParam(required = false) MultipartFile truthFile,
                           @RequestParam(required = false) String modelName,
                           Model model, HttpSession session) throws IOException {
        addCommon(model, session);

        String inputText = resolveInputText(text, file);
        model.addAttribute("inputText", inputText);
        model.addAttribute("selectedModel", modelName);

        List<String> sentences = textAnalysisService.splitSentences(inputText);
        if (sentences.size() < 2) {
            model.addAttribute("error", "Need at least two sentences (one per line) to detect a switch.");
            return "predict";
        }

        // optional truth-problem-N.json ({"changes": [0, 1, ...]}), one value per sentence pair
        int[] truth = null;
        String truthError = null;
        if (truthFile != null && !truthFile.isEmpty()) {
            try {
                truth = DatasetLoader.extractChanges(new String(truthFile.getBytes(), StandardCharsets.UTF_8));
            } catch (RuntimeException e) {
                truthError = "Could not read the truth file (expected JSON like {\"changes\": [0, 1, 0]}): "
                        + e.getMessage();
            }
        }

        PredictionResult result = predictionService.predict(sentences, modelName, truth);
        if (truthError != null) {
            result.truthError = truthError;
        }
        model.addAttribute("result", result);
        return "predict";
    }

    private void addCommon(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
        boolean trained = trainingService.isAnyModelTrained();
        model.addAttribute("trained", trained);
        model.addAttribute("availableModels", classifierRegistry.allNames());
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
}
