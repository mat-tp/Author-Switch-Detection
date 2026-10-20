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

import java.util.List;

@Controller
public class HomeController {

    private final TrainingService trainingService;
    private final CurrentUserService currentUserService;
    private final ClassifierRegistry classifierRegistry;

    public HomeController(TrainingService trainingService, CurrentUserService currentUserService,
            ClassifierRegistry classifierRegistry) {
        this.classifierRegistry = classifierRegistry;
        this.trainingService = trainingService;
        this.currentUserService = currentUserService;
    }

    @GetMapping("/")
    public String index(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);

        // what can be used right now is decided by the model files, not by the metrics rows
        List<ClassifierRegistry.Option> options = classifierRegistry.options();
        model.addAttribute("modelOptions", options);
        model.addAttribute("trained", options.stream().anyMatch(o -> o.ready()));

        List<ModelMetrics> runs = trainingService.latestRuns();
        model.addAttribute("metrics", runs.isEmpty() ? null : runs.get(0));
        return "index";
    }
}
