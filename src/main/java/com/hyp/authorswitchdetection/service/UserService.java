package com.hyp.authorswitchdetection.service;

import com.hyp.authorswitchdetection.model.Role;
import com.hyp.authorswitchdetection.model.User;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UserService {

    private final UserStore userStore;

    @Value("${app.admin.username}")
    private String defaultAdminUsername;

    @Value("${app.admin.password}")
    private String defaultAdminPassword;

    public UserService(UserStore userStore) {
        this.userStore = userStore;
    }

    /** Make sure there is always at least one admin account to log in with. */
    @PostConstruct
    public void seedDefaultAdmin() {
        if (!userStore.existsByUsername(defaultAdminUsername)) {
            User admin = new User(defaultAdminUsername, PasswordUtil.hash(defaultAdminPassword), Role.ADMIN);
            userStore.save(admin);
        }
    }

    public enum RegistrationResult { OK, USERNAME_TAKEN, PASSWORD_MISMATCH, INVALID }

    public RegistrationResult register(String username, String password, String password2) {
        if (username == null || username.isBlank() || username.length() < 3 || username.length() > 32) {
            return RegistrationResult.INVALID;
        }
        if (password == null || password.length() < 6) {
            return RegistrationResult.INVALID;
        }
        if (!password.equals(password2)) {
            return RegistrationResult.PASSWORD_MISMATCH;
        }
        if (userStore.existsByUsername(username)) {
            return RegistrationResult.USERNAME_TAKEN;
        }
        userStore.save(new User(username, PasswordUtil.hash(password), Role.USER));
        return RegistrationResult.OK;
    }

    public Optional<User> authenticate(String username, String password) {
        return userStore.findByUsername(username)
                .filter(u -> PasswordUtil.matches(password, u.getPasswordHash()));
    }

    public List<User> findAll() {
        return userStore.findAll();
    }
}
