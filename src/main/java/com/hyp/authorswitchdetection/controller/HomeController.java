package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.ModelMetrics;
import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.TrainingService;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;

@Controller
public class HomeController {

    private final TrainingService trainingService;
    private final CurrentUserService currentUserService;

    public HomeController(TrainingService trainingService, CurrentUserService currentUserService) {
        this.trainingService = trainingService;
        this.currentUserService = currentUserService;
    }

    @GetMapping("/")
    public String index(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);

        boolean trained = trainingService.isAnyModelTrained();
        model.addAttribute("trained", trained);
        if (trained) {
            List<ModelMetrics> best = trainingService.allMetricsByF1Desc();
            model.addAttribute("metrics", best.isEmpty() ? null : best.get(0));
        }
        return "index";
    }
}
