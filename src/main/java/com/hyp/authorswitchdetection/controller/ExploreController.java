package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.DatasetExploreService;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

/**
 * Dataset statistics page. Real numbers depend on a labelled dataset (e.g.
 * the PAN-style mawsa26 corpus) being present on disk at
 * {@code app.dataset.root} (see application.properties) — reads it via
 * {@link DatasetExploreService}, which wraps the ported DatasetLoader.
 * Falls back to a "no dataset configured" state when nothing is found there.
 */
@Controller
public class ExploreController {

    private final CurrentUserService currentUserService;
    private final DatasetExploreService datasetExploreService;

    public ExploreController(CurrentUserService currentUserService, DatasetExploreService datasetExploreService) {
        this.currentUserService = currentUserService;
        this.datasetExploreService = datasetExploreService;
    }

    @GetMapping("/explore")
    public String explore(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
        model.addAttribute("datasetRoot", datasetExploreService.getDatasetRoot());

        boolean configured = datasetExploreService.isConfigured();
        model.addAttribute("datasetConfigured", configured);

        if (configured) {
            model.addAttribute("difficultyStats", datasetExploreService.computeStats());
        }

        return "explore";
    }
}
