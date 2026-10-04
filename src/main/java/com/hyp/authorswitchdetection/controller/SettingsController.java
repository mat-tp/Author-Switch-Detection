package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

/*
    Used to also have per-session "default model" / "cross-validation folds"
    hyperparameters here, but Training now always runs on the full dataset with
    its own feature/parameter search (see RealTrainer) — there is nothing left
    for those fields to configure, so this page is read-only info now.
*/
@Controller
@RequestMapping("/settings")
public class SettingsController {

    private final CurrentUserService currentUserService;

    @Value("${app.dataset.root:dataset/mawsa26-pan-zenodo-DATA/}")
    private String datasetRoot;

    @Value("${app.trained-models.dir:data/models}")
    private String trainedModelsDir;

    public SettingsController(CurrentUserService currentUserService) {
        this.currentUserService = currentUserService;
    }

    @GetMapping
    public String form(Model model, HttpSession session) {
        String guard = requireAdmin(model, session);
        if (guard != null) return guard;

        model.addAttribute("datasetRoot", datasetRoot);
        model.addAttribute("trainedModelsDir", trainedModelsDir);
        return "settings";
    }

    private String requireAdmin(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
        if (user == null) return "redirect:/auth/login";
        if (!user.isAdmin()) return "redirect:/";
        return null;
    }
}
