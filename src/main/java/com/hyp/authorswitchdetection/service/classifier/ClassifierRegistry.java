package com.hyp.authorswitchdetection.service.classifier;

import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Collects every AuthorSwitchClassifier bean so controllers can list the available models
 * without knowing about each implementation. A new model only needs to be a Spring component.
 *
 * The pages show the readable name ("Decision Tree"), the key ("decision_tree") is for the
 * forms and the files only.
 */
@Component
public class ClassifierRegistry {

    private final Map<String, AuthorSwitchClassifier> byName = new LinkedHashMap<>();

    /** What a drop-down needs about one model. */
    public record Option(String name, String label, boolean ready, String problem) {
    }

    public ClassifierRegistry(List<AuthorSwitchClassifier> classifiers) {

        // the order of the pages: the three models of the paper first, then the retrieval model
        String[] order = { "decision_tree", "logistic_regression", "neural_network", "retrieval_embedding" };
        for (String name : order) {
            for (AuthorSwitchClassifier c : classifiers) {
                if (c.getName().equals(name)) {
                    byName.put(name, c);
                }
            }
        }
        for (AuthorSwitchClassifier c : classifiers) {
            byName.putIfAbsent(c.getName(), c);
        }
    }

    public List<String> allNames() {
        return List.copyOf(byName.keySet());
    }

    /** The models for a drop-down, with whether each one can be used right now. */
    public List<Option> options() {

        return byName.values().stream()
                .map(c -> new Option(c.getName(), c.getDisplayName(), c.isImplemented(), c.unavailableReason()))
                .toList();
    }

    public String label(String name) {
        return get(name).map(AuthorSwitchClassifier::getDisplayName).orElse(name);
    }

    public Optional<AuthorSwitchClassifier> get(String name) {
        return Optional.ofNullable(byName.get(name));
    }
}
