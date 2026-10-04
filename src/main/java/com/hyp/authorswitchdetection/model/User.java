package com.hyp.authorswitchdetection.model;

import java.time.LocalDateTime;

// Reads/Writes these to a JSON file
public class User {

    private Long id;
    private String username;

    // storing a SHA-256 hash password instead
    private String passwordHash;

    private Role role = Role.USER;
    private LocalDateTime createdAt = LocalDateTime.now();

    public User() {
    }

    public User(String username, String passwordHash, Role role) {
        this.username = username;
        this.passwordHash = passwordHash;
        this.role = role;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }

    public String getPasswordHash() { return passwordHash; }
    public void setPasswordHash(String passwordHash) { this.passwordHash = passwordHash; }

    public Role getRole() { return role; }
    public void setRole(Role role) { this.role = role; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public boolean isAdmin() { return role == Role.ADMIN; }
}
