package com.hyp.authorswitchdetection.service;

import com.hyp.authorswitchdetection.model.User;
import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Service;

import java.util.Optional;

/** Thin wrapper around a plain HttpSession-based login (no Spring Security needed for this app's scope). */
@Service
public class CurrentUserService {

    public static final String SESSION_KEY = "userId";

    private final UserStore userStore;

    public CurrentUserService(UserStore userStore) {

        this.userStore = userStore;
    }

    public void login(HttpSession session, User user) {

        session.setAttribute(SESSION_KEY, user.getId());
    }

    public void logout(HttpSession session) {

        session.invalidate();
    }

    public Optional<User> getCurrentUser(HttpSession session) {
        Object id = session.getAttribute(SESSION_KEY);
        if (id == null) return Optional.empty();
        return userStore.findById((Long) id);
    }
}
