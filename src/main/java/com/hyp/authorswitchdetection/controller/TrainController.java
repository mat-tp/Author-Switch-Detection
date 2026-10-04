package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.TrainingService;
import com.hyp.authorswitchdetection.service.classifier.ClassifierRegistry;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/** This is limited to Admin-only: kick off a (currently simulated) training run and poll its progress. */
@Controller
@RequestMapping("/train")
public class TrainController {

    private final TrainingService trainingService;
    private final ClassifierRegistry classifierRegistry;
    private final CurrentUserService currentUserService;

    public TrainController(TrainingService trainingService, ClassifierRegistry classifierRegistry,
                            CurrentUserService currentUserService) {
        this.trainingService = trainingService;
        this.classifierRegistry = classifierRegistry;
        this.currentUserService = currentUserService;
    }

    @GetMapping
    public String form(Model model, HttpSession session) {
        String guard = requireAdmin(model, session);
        if (guard != null) return guard;

        model.addAttribute("models", classifierRegistry.allNames());
        return "train";
    }

    @PostMapping
    public String start(@RequestParam String modelName, Model model, HttpSession session) {
        String guard = requireAdmin(model, session);
        if (guard != null) return guard;

        trainingService.startTraining(modelName);
        model.addAttribute("models", classifierRegistry.allNames());
        model.addAttribute("started", true);
        return "train";
    }

    @GetMapping("/status")
    @ResponseBody
    public Map<String, Object> status() {
        var s = trainingService.getStatus();
        return Map.of(
                "running", s.running,
                "done", s.done,
                "percent", s.percent,
                "current_task", s.currentTask,
                "error", s.error == null ? "" : s.error,
                "log", s.log
        );
    }

    private String requireAdmin(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
        if (user == null) return "redirect:/auth/login";
        if (!user.isAdmin()) return "redirect:/";
        return null;
    }
}
