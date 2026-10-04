package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.ModelMetrics;
import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.TrainingService;
import com.hyp.authorswitchdetection.service.classifier.ClassifierRegistry;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class ResultsController {

    private final TrainingService trainingService;
    private final ClassifierRegistry classifierRegistry;
    private final CurrentUserService currentUserService;

    public ResultsController(TrainingService trainingService, ClassifierRegistry classifierRegistry,
                              CurrentUserService currentUserService) {
        this.trainingService = trainingService;
        this.classifierRegistry = classifierRegistry;
        this.currentUserService = currentUserService;
    }

    @GetMapping("/results")
    public String results(@RequestParam(required = false) String modelName, Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);

        model.addAttribute("availableModels", classifierRegistry.allNames());
        model.addAttribute("allMetrics", trainingService.allMetricsByF1Desc());

        String selected = modelName;
        if (selected == null && !classifierRegistry.allNames().isEmpty()) {
            selected = classifierRegistry.allNames().get(0);
        }
        model.addAttribute("selectedModel", selected);

        ModelMetrics metrics = selected == null ? null : trainingService.latestFor(selected);
        model.addAttribute("metrics", metrics);
        return "results";
    }
}
