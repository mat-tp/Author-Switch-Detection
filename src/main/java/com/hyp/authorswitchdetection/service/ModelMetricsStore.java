package com.hyp.authorswitchdetection.service;

import com.hyp.authorswitchdetection.model.ModelMetrics;
import com.hyp.authorswitchdetection.model.ModelMetricsIO;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Training run metrics kept in a plain JSON-lines file on disk (see
 * ModelMetricsIO for the actual read/write logic).
 *
 * This is intentionally stateless: every method re-reads the file instead of
 * relying on an in-memory cache populated once at startup. That is what
 * makes it possible to train a model with the standalone command-line
 * Main.java (via ./scripts/build.sh) — a completely separate JVM process,
 * with no Spring container at all — and see it appear on the Results page
 * immediately, without restarting the web app.
 */
@Service
public class ModelMetricsStore {

    @Value("${app.model-metrics.file:data/model-metrics.json}")
    private String filePath;

    public ModelMetrics save(ModelMetrics m) {
        return ModelMetricsIO.append(filePath, m);
    }

    public int count() {
        return ModelMetricsIO.loadAll(filePath).size();
    }

    public List<ModelMetrics> findAllByOrderByF1Desc() {
        List<ModelMetrics> sorted = new ArrayList<>(ModelMetricsIO.loadAll(filePath));
        sorted.sort(Comparator.comparingDouble(ModelMetrics::getF1).reversed());
        return sorted;
    }

    public Optional<ModelMetrics> findFirstByModelNameOrderByTrainedAtDesc(String modelName) {

        ModelMetrics latest = null;
        for (ModelMetrics m : ModelMetricsIO.loadAll(filePath)) {
            if (!m.getModelName().equals(modelName)) {
                continue;
            }
            if (latest == null || m.getTrainedAt().isAfter(latest.getTrainedAt())) {
                latest = m;
            }
        }
        return Optional.ofNullable(latest);
    }
}
