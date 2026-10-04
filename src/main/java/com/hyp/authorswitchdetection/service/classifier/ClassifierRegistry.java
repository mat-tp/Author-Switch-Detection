package com.hyp.authorswitchdetection.service.classifier;

import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Collects every AuthorSwitchClassifier bean so controllers can list
 * available models without knowing about each implementation. Adding a new
 * model later (e.g. RandomForestClassifier becomes real) requires no change
 * here — Spring just injects it automatically.
 */
@Component
public class ClassifierRegistry {

    private final Map<String, AuthorSwitchClassifier> byName = new LinkedHashMap<>();

    public ClassifierRegistry(List<AuthorSwitchClassifier> classifiers) {
        for (AuthorSwitchClassifier c : classifiers) {
            byName.put(c.getName(), c);
        }
    }

    public List<String> allNames() {
        return List.copyOf(byName.keySet());
    }

    public Optional<AuthorSwitchClassifier> get(String name) {
        return Optional.ofNullable(byName.get(name));
    }
}
