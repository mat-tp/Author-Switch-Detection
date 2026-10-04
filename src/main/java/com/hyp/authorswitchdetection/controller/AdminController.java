package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.UserService;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/admin")
public class AdminController {

    private final UserService userService;
    private final CurrentUserService currentUserService;

    public AdminController(UserService userService, CurrentUserService currentUserService) {
        this.userService = userService;
        this.currentUserService = currentUserService;
    }

    // I only care about the current user not all the users
    // todo: using a response entity class to set up the Responses
    @GetMapping("/users")
    public String users(Model model, HttpSession session) {
        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
        if (user == null) return "redirect:/auth/login";
        if (!user.isAdmin()) return "redirect:/";

        model.addAttribute("users", userService.findAll());
        return "admin/users";
    }
}
