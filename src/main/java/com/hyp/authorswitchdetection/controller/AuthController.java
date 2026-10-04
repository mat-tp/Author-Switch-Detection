package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import com.hyp.authorswitchdetection.service.UserService;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;

@Controller
@RequestMapping("/auth")
public class AuthController {

    private final UserService userService;
    private final CurrentUserService currentUserService;

    public AuthController(UserService userService, CurrentUserService currentUserService) {
        this.userService = userService;
        this.currentUserService = currentUserService;
    }

    @GetMapping("/login")
    public String loginForm(Model model) {
        return "auth/login";
    }

    @PostMapping("/login")
    public String login(@RequestParam String username,
                         @RequestParam String password,
                         HttpSession session,
                         Model model) {
        Optional<User> user = userService.authenticate(username, password);
        if (user.isEmpty()) {
            model.addAttribute("error", "Invalid username or password.");
            model.addAttribute("username", username);
            return "auth/login";
        }
        currentUserService.login(session, user.get());
        return "redirect:/";
    }

    @GetMapping("/register")
    public String registerForm() {
        return "auth/register";
    }

    @PostMapping("/register")
    public String register(@RequestParam String username,
                            @RequestParam String password,
                            @RequestParam String password2,
                            Model model) {
        UserService.RegistrationResult result = userService.register(username, password, password2);
        if (result != UserService.RegistrationResult.OK) {
            model.addAttribute("username", username);
            switch (result) {
                case USERNAME_TAKEN -> model.addAttribute("error", "That username is already taken.");
                case PASSWORD_MISMATCH -> model.addAttribute("error", "Passwords do not match.");
                default -> model.addAttribute("error", "Please check your username (3-32 chars) and password (6+ chars).");
            }
            return "auth/register";
        }
        return "redirect:/auth/login?registered=1";
    }

    @GetMapping("/logout")
    public String logout(HttpSession session) {
        currentUserService.logout(session);
        return "redirect:/";
    }
}
