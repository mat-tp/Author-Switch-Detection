package com.hyp.authorswitchdetection.service;

import com.hyp.authorswitchdetection.model.Role;
import com.hyp.authorswitchdetection.model.User;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Users are kept in a plain JSON-lines file on disk Loaded into memory once
 * at startup, and the whole file is rewritten on every save().
 */
@Service
public class UserStore {

    @Value("${app.users.file:data/users.json}")
    private String filePath;

    private final List<User> users = new ArrayList<>();
    private long nextId = 1;

    @PostConstruct
    public void load() {

        File file = new File(filePath);
        if (!file.exists()) {
            return;
        }

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {

                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                User user = parseUser(line);
                users.add(user);
                if (user.getId() != null && user.getId() >= nextId) {
                    nextId = user.getId() + 1;
                }
            }

        } catch (IOException e) {
            System.err.println("Error reading users file: " + filePath);
            System.err.println(e.getMessage());
        }
    }

    public synchronized User save(User user) {

        if (user.getId() == null) {
            user.setId(nextId++);
            users.add(user);
        } else {
            for (int i = 0; i < users.size(); i++) {
                if (users.get(i).getId().equals(user.getId())) {
                    users.set(i, user);
                    break;
                }
            }
        }

        writeToDisk();
        return user;
    }

    public Optional<User> findByUsername(String username) {
        for (User user : users) {
            if (user.getUsername().equalsIgnoreCase(username)) {
                return Optional.of(user);
            }
        }
        return Optional.empty();
    }

    public boolean existsByUsername(String username) {
        return findByUsername(username).isPresent();
    }

    public Optional<User> findById(Long id) {
        for (User user : users) {
            if (user.getId().equals(id)) {
                return Optional.of(user);
            }
        }
        return Optional.empty();
    }

    public List<User> findAll() {
        return new ArrayList<>(users);
    }

    private void writeToDisk() {

        File file = new File(filePath);
        File parent = file.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(
                new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8))) {

            for (User user : users) {
                writer.println(toJsonLine(user));
            }

        } catch (IOException e) {
            System.err.println("Error writing users file: " + filePath);
            System.err.println(e.getMessage());
        }
    }

    private String toJsonLine(User user) {

        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"id\":").append(user.getId()).append(",");
        sb.append("\"username\":\"").append(escape(user.getUsername())).append("\",");
        sb.append("\"passwordHash\":\"").append(escape(user.getPasswordHash())).append("\",");
        sb.append("\"role\":\"").append(user.getRole().name()).append("\",");
        sb.append("\"createdAt\":\"").append(user.getCreatedAt()).append("\"");
        sb.append("}");
        return sb.toString();
    }

    private User parseUser(String line) {

        User user = new User();
        user.setId(JsonLine.extractLong(line, "id"));
        user.setUsername(JsonLine.extractString(line, "username"));
        user.setPasswordHash(JsonLine.extractString(line, "passwordHash"));
        user.setRole(Role.valueOf(JsonLine.extractString(line, "role")));
        user.setCreatedAt(LocalDateTime.parse(JsonLine.extractString(line, "createdAt")));
        return user;
    }

    private String escape(String s) {
        return JsonLine.escape(s);
    }
}
