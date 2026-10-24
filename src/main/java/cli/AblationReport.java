package cli;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import features.FeatureGroup;

/**
 * Writes the human-readable results of Ablation: a Markdown report and the
 * LaTeX tables. (The raw per-run numbers are in ablation_results.csv.)
 */
final class AblationReport {

    // an importance smaller than this (in macro-F1) is called "negligible"
    private static final double NEGLIGIBLE = 0.005;

    private static final Map<String, String> MODEL_TITLES = new LinkedHashMap<>();
    static {
        MODEL_TITLES.put(Ablation.TREE, "Decision Tree");
        MODEL_TITLES.put(Ablation.LOGISTIC, "Logistic Regression");
        MODEL_TITLES.put(Ablation.NEURAL, "Neural Network");
    }

    private AblationReport() {
    }

    static void write(List<Ablation.Row> rows, Ablation.Data data, features.FeatureConfig baseConfig, String outDir,
            double minutes) throws IOException {

        Files.writeString(Paths.get(outDir, "ablation_report.md"), markdown(rows, data, baseConfig, minutes),
                StandardCharsets.UTF_8);
        Files.writeString(Paths.get(outDir, "ablation_tables.tex"), latex(rows), StandardCharsets.UTF_8);
    }

    private static Ablation.Row find(List<Ablation.Row> rows, String kind, String model, String experiment) {
        for (Ablation.Row r : rows) {
            if (r.kind.equals(kind) && r.model.equals(model) && r.experiment.equals(experiment)) {
                return r;
            }
        }
        return null;
    }

    private static Set<String> models(List<Ablation.Row> rows) {
        Set<String> models = new LinkedHashSet<>();
        for (Ablation.Row r : rows) {
            models.add(r.model);
        }
        return models;
    }

    private static boolean ok(Ablation.Row r) {
        return r != null && r.agg.error == null;
    }

    private static String title(String model) {
        return MODEL_TITLES.getOrDefault(model, model);
    }

    private static String num(double v) {
        return Double.isNaN(v) ? "n/a" : String.format(Locale.ROOT, "%.4f", v);
    }

    private static String signed(double v) {
        return Double.isNaN(v) ? "n/a" : String.format(Locale.ROOT, "%+.4f", v);
    }

    /** "0.7123" or "0.7123 ± 0.0081" when several seeds were run. */
    private static String withStd(double mean, double std, int seeds) {
        return seeds > 1 ? String.format(Locale.ROOT, "%.4f ± %.4f", mean, std) : num(mean);
    }

    private static double diffF1(Ablation.Row base, Ablation.Row other, String difficulty) {
        if (!ok(base) || !ok(other)) {
            return Double.NaN;
        }
        Double b = base.agg.macroByDifficulty.get(difficulty);
        Double o = other.agg.macroByDifficulty.get(difficulty);
        return b == null || o == null ? Double.NaN : b - o;
    }

    private static String verdict(double importance, double noise) {
        if (Double.isNaN(importance)) {
            return "failed";
        }
        if (Math.abs(importance) <= Math.max(NEGLIGIBLE, 2 * noise)) {
            return Math.abs(importance) <= NEGLIGIBLE ? "negligible" : "within seed noise";
        }
        return importance > 0 ? "useful" : "removal helps";
    }

    /** One feature group's importance for one model, in table order. */
    private static final class Importance {
        String group;
        double importance;
        double easy, medium, hard;
        double onlyF1;
        double withoutF1;
        String verdict;
    }

    private static List<Importance> importances(List<Ablation.Row> rows, String model) {

        List<Importance> list = new ArrayList<>();
        Ablation.Row base = find(rows, "feature", model, "ALL");
        if (!ok(base)) {
            return list;
        }

        for (FeatureGroup g : FeatureGroup.values()) {
            Ablation.Row without = find(rows, "feature", model, "ALL - " + g.name());
            Ablation.Row only = find(rows, "feature", model, "ONLY " + g.name());

            Importance i = new Importance();
            i.group = g.name();
            i.withoutF1 = ok(without) ? without.agg.score() : Double.NaN;
            i.importance = ok(without) ? base.agg.score() - without.agg.score() : Double.NaN;
            i.easy = diffF1(base, without, "easy");
            i.medium = diffF1(base, without, "medium");
            i.hard = diffF1(base, without, "hard");
            i.onlyF1 = ok(only) ? only.agg.score() : Double.NaN;
            double noise = ok(without) ? Math.max(base.agg.scoreStd(), without.agg.scoreStd()) : 0.0;
            i.verdict = verdict(i.importance, noise);
            list.add(i);
        }

        // biggest contribution first (failed runs last)
        list.sort(Comparator.comparingDouble((Importance i) -> Double.isNaN(i.importance) ? -1e9 : i.importance)
                .reversed());
        return list;
    }

    private static String markdown(List<Ablation.Row> rows, Ablation.Data data, features.FeatureConfig baseConfig,
            double minutes) {

        StringBuilder sb = new StringBuilder();
        int seeds = rows.isEmpty() ? 1 : rows.get(0).agg.seeds;
        String evalName = data.evalIsTest ? "held-out test half" : "validation split (no test split available)";

        sb.append("# Ablation study report\n\n");
        sb.append("- Training pairs: ").append(data.trainY.length).append("\n");
        sb.append("- Evaluation pairs (").append(evalName).append("): ").append(data.evalY.length).append("\n");
        sb.append("- Pair features in the full model: ").append(data.train[0].length).append("\n");
        sb.append("- Baseline feature settings: `").append(baseConfig.signature()).append("`\n");
        sb.append("- Seeds: ").append(Ablation.seeds()).append(seeds > 1
                ? " (each seed = a different random " + (int) (Ablation.SEED_FRACTION * 100)
                        + "% training subsample; numbers are mean ± std)"
                : " (single run, no variance estimate)")
                .append("\n");
        sb.append(String.format(Locale.ROOT, "- Total run time: %.1f min\n\n", minutes));

        sb.append("**Metric:** macro-F1 (the mean of the F1 of the *change* class and of the *no change* class, ")
                .append("pooled over all boundaries - the measure PAN 2026 ranks by) on the ").append(evalName)
                .append(". The F1 of the change class alone is in `ablation_results.csv`. ")
                .append("Thresholds (all three models) are chosen on the validation data by macro-F1, ")
                .append("never on the evaluation data. Parameters and pipeline settings are ranked by **validation** macro-F1.\n\n");

        appendDifficultyBaselines(sb, rows, data);

        sb.append("## 1. Feature-group ablation\n\n");
        sb.append("`Importance(G) = MacroF1_all - MacroF1_without_G`. Positive = the group carries useful information. ")
                .append("Near 0 = it adds nothing the other groups do not already provide. ")
                .append("Negative = the model does better without it. ")
                .append("\"Only G\" trains on that group alone. ")
                .append("The per-difficulty columns are the same difference on easy / medium / hard pairs, each level with the ")
                .append("threshold chosen on that level's own validation data (PAN scores every level as its own dataset).\n\n");

        for (String model : models(rows)) {

            Ablation.Row base = find(rows, "feature", model, "ALL");
            if (base == null) {
                continue;
            }

            sb.append("### ").append(title(model)).append("\n\n");

            if (!ok(base)) {
                sb.append("Baseline run failed: ").append(base.agg.error).append("\n\n");
                continue;
            }

            sb.append("Baseline (all ").append(base.agg.nFeatures).append(" features): macro-F1 = ")
                    .append(withStd(base.agg.score(), base.agg.scoreStd(), seeds)).append("\n\n");

            sb.append("| Rank | Feature group | Macro-F1 without G | Importance | Δ easy | Δ medium | Δ hard | Macro-F1 only G | Verdict |\n");
            sb.append("|---:|---|---:|---:|---:|---:|---:|---:|---|\n");

            int rank = 1;
            for (Importance i : importances(rows, model)) {
                sb.append("| ").append(rank++).append(" | ").append(i.group).append(" | ").append(num(i.withoutF1))
                        .append(" | **").append(signed(i.importance)).append("** | ").append(signed(i.easy))
                        .append(" | ").append(signed(i.medium)).append(" | ").append(signed(i.hard)).append(" | ")
                        .append(num(i.onlyF1)).append(" | ").append(i.verdict).append(" |\n");
            }
            sb.append("\n");
        }

        appendCrossModelSummary(sb, rows);
        appendGroupsByDifficulty(sb, rows);
        appendParameterSection(sb, rows, seeds);
        appendPipelineSection(sb, rows, seeds);
        appendPipelineByDifficulty(sb, rows);
        appendCaveats(sb);

        return sb.toString();
    }

    private static final String[] LEVELS = { "easy", "medium", "hard" };

    /** Baselines and the effect of the threshold, per difficulty level. */
    private static void appendDifficultyBaselines(StringBuilder sb, List<Ablation.Row> rows, Ablation.Data data) {

        if (data.evalYByDifficulty == null || data.evalYByDifficulty.isEmpty()) {
            return;
        }

        sb.append("## 0. The difficulty levels\n\n");
        sb.append("PAN 2026 scores **easy, medium and hard as three separate datasets** (the best systems use a separate ")
                .append("recipe for each), so every table below reports the levels separately. The pairs of one level can be ")
                .append("very different from another (medium has very few changes), so a threshold that is best for all pairs ")
                .append("pooled is not best for any one level. Two numbers are therefore kept for every level: *pooled thr.* = the ")
                .append("one threshold chosen on the pooled validation data (how the first version reported), and *own thr.* = ")
                .append("the threshold chosen on that level's validation data only. The *own thr.* number is the one that corresponds ")
                .append("to a submission for that dataset and is used in all per-level columns of this report. ")
                .append("Run `-Ddifficulty=easy|medium|hard` to train and evaluate on one level only.\n\n");

        sb.append("**Baselines on the evaluation pairs of each level** (macro-F1; the official test baselines are in `docs/RESEARCH.md`):\n\n");
        sb.append("| Level | Pairs | Share of changes | Always \"no change\" | Always \"change\" |\n|---|---:|---:|---:|---:|\n");
        for (String level : LEVELS) {
            int[] truth = data.evalYByDifficulty.get(level);
            if (truth == null || truth.length == 0) {
                continue;
            }
            int[] zeros = new int[truth.length];
            int[] ones = new int[truth.length];
            java.util.Arrays.fill(ones, 1);
            int changes = 0;
            for (int t : truth) {
                changes += t;
            }
            sb.append("| ").append(level).append(" | ").append(truth.length).append(" | ")
                    .append(String.format(Locale.ROOT, "%.1f %%", 100.0 * changes / truth.length)).append(" | ")
                    .append(num(evaluation.Metrics.calcMacroF1(zeros, truth))).append(" | ")
                    .append(num(evaluation.Metrics.calcMacroF1(ones, truth))).append(" |\n");
        }
        sb.append("\n");

        StringBuilder table = new StringBuilder();
        for (String model : allModels(rows)) {
            Ablation.Row base = find(rows, "feature", model, "ALL");
            if (!ok(base)) {
                // a pipeline-only run: the baseline setting of the pipeline rows
                base = null;
                for (Ablation.Row r : rows) {
                    if (r.model.equals(model) && r.isBaseline && ok(r)) {
                        base = r;
                        break;
                    }
                }
            }
            if (base == null) {
                continue;
            }
            for (String level : LEVELS) {
                if (!base.agg.macroByDifficulty.containsKey(level)) {
                    continue;
                }
                table.append("| ").append(title(model)).append(" | ").append(level).append(" | ")
                        .append(num(base.agg.macroByDifficultyGlobal.getOrDefault(level, Double.NaN))).append(" | ")
                        .append(num(base.agg.macroByDifficulty.get(level))).append(" | ")
                        .append(num(base.agg.thresholdByDifficulty.getOrDefault(level, Double.NaN))).append(" |\n");
            }
        }
        if (table.length() > 0) {
            sb.append("**The baseline setting of each model, per level** (evaluation macro-F1):\n\n");
            sb.append("| Model | Level | Pooled thr. | Own thr. | Own threshold |\n|---|---|---:|---:|---:|\n");
            sb.append(table).append("\n");
        }
    }

    /** Every model that has any row. */
    private static List<String> allModels(List<Ablation.Row> rows) {
        List<String> models = new ArrayList<>();
        for (Ablation.Row r : rows) {
            if (!models.contains(r.model)) {
                models.add(r.model);
            }
        }
        return models;
    }

    /** For every feature group: the mean importance over the models, on each level. */
    private static void appendGroupsByDifficulty(StringBuilder sb, List<Ablation.Row> rows) {

        List<String> models = new ArrayList<>(models(rows));
        boolean any = false;
        for (String m : models) {
            any |= ok(find(rows, "feature", m, "ALL"));
        }
        if (!any) {
            return;
        }

        sb.append("### Importance by difficulty level (mean over the models)\n\n");
        sb.append("Mean of `MacroF1_all - MacroF1_without_G` over the models, per level. A group that helps on **hard** is a ")
                .append("style signal (topic is constant there); a group that helps only on **easy** may be a topic signal.\n\n");
        sb.append("| Feature group | easy | medium | hard |\n|---|---:|---:|---:|\n");

        for (FeatureGroup g : FeatureGroup.values()) {
            sb.append("| ").append(g.name()).append(" |");
            for (String level : LEVELS) {
                double sum = 0.0;
                int n = 0;
                for (String m : models) {
                    double d = diffF1(find(rows, "feature", m, "ALL"), find(rows, "feature", m, "ALL - " + g.name()), level);
                    if (!Double.isNaN(d)) {
                        sum += d;
                        n++;
                    }
                }
                sb.append(" ").append(n == 0 ? "n/a" : signed(sum / n)).append(" |");
            }
            sb.append("\n");
        }
        sb.append("\n");
    }

    /** The pipeline settings again, split into the three levels (own threshold, evaluation macro-F1). */
    private static void appendPipelineByDifficulty(StringBuilder sb, List<Ablation.Row> rows) {

        List<String> families = pipelineFamilies(rows);
        if (families.isEmpty()) {
            return;
        }
        List<String> models = pipelineModels(rows);

        sb.append("## 3b. Feature-pipeline ablation, by difficulty level\n\n");
        sb.append("The same runs as section 3, but the evaluation macro-F1 of each level separately (threshold chosen on ")
                .append("that level's validation data). Cell = easy / medium / hard.\n\n");

        for (String family : families) {

            sb.append("### ").append(FAMILY_TITLES.getOrDefault(family, family)).append("\n\n");
            sb.append("| Setting |");
            for (String m : models) {
                sb.append(" ").append(title(m)).append(" |");
            }
            sb.append("\n|---|");
            for (int i = 0; i < models.size(); i++) {
                sb.append("---:|");
            }
            sb.append("\n");

            for (String setting : pipelineSettings(rows, family)) {
                Ablation.Row any = null;
                for (String m : models) {
                    Ablation.Row r = pipelineRow(rows, family, setting, m);
                    if (r != null) {
                        any = r;
                        break;
                    }
                }
                sb.append("| ").append(setting).append(any != null && any.isBaseline ? " ◀" : "").append(" |");
                for (String m : models) {
                    Ablation.Row r = pipelineRow(rows, family, setting, m);
                    if (!ok(r)) {
                        sb.append(r == null ? " not run |" : " failed |");
                        continue;
                    }
                    sb.append(" ");
                    for (int i = 0; i < LEVELS.length; i++) {
                        Double v = r.agg.macroByDifficulty.get(LEVELS[i]);
                        sb.append(i > 0 ? " / " : "").append(v == null ? "n/a" : num(v));
                    }
                    sb.append(" |");
                }
                sb.append("\n");
            }
            sb.append("\n");
        }
    }

    private static void appendCrossModelSummary(StringBuilder sb, List<Ablation.Row> rows) {

        List<String> models = new ArrayList<>(models(rows));
        boolean any = false;
        for (String m : models) {
            if (ok(find(rows, "feature", m, "ALL"))) {
                any = true;
            }
        }
        if (!any || models.size() < 2) {
            return;
        }

        sb.append("### Importance across the models\n\n");
        sb.append("A group that matters for ALL models is much stronger evidence than one that matters for one.\n\n");
        sb.append("| Feature group |");
        for (String m : models) {
            sb.append(" ").append(title(m)).append(" |");
        }
        sb.append("\n|---|");
        for (int i = 0; i < models.size(); i++) {
            sb.append("---:|");
        }
        sb.append("\n");

        for (FeatureGroup g : FeatureGroup.values()) {
            sb.append("| ").append(g.name()).append(" |");
            for (String m : models) {
                Ablation.Row base = find(rows, "feature", m, "ALL");
                Ablation.Row without = find(rows, "feature", m, "ALL - " + g.name());
                sb.append(" ").append(ok(base) && ok(without) ? signed(base.agg.score() - without.agg.score()) : "n/a")
                        .append(" |");
            }
            sb.append("\n");
        }
        sb.append("\n");
    }

    private static void appendParameterSection(StringBuilder sb, List<Ablation.Row> rows, int seeds) {

        boolean anyParam = false;
        for (Ablation.Row r : rows) {
            if (r.kind.equals("param") || r.kind.equals("grid")) {
                anyParam = true;
            }
        }
        if (!anyParam) {
            return;
        }

        sb.append("## 2. Parameter ablation\n\n");
        sb.append("One hyperparameter is changed at a time; everything else stays at the baseline of Main.java. ")
                .append("`◀` = the baseline value, `★` = best validation macro-F1 of that parameter. ")
                .append("The train − validation gap shows over-fitting.\n\n");

        StringBuilder bestSummary = new StringBuilder();

        for (String model : models(rows)) {

            Set<String> params = new LinkedHashSet<>();
            for (Ablation.Row r : rows) {
                if (r.kind.equals("param") && r.model.equals(model)) {
                    params.add(r.param);
                }
            }
            if (params.isEmpty()) {
                continue;
            }

            sb.append("### ").append(title(model)).append("\n\n");
            bestSummary.append("**").append(title(model)).append("**\n\n");
            bestSummary.append("| Parameter | Best value (by validation macro-F1) | Validation macro-F1 | Test macro-F1 | Baseline value | Baseline test macro-F1 |\n");
            bestSummary.append("|---|---|---:|---:|---|---:|\n");

            for (String param : params) {

                List<Ablation.Row> group = new ArrayList<>();
                for (Ablation.Row r : rows) {
                    if (r.kind.equals("param") && r.model.equals(model) && r.param.equals(param)) {
                        group.add(r);
                    }
                }

                Ablation.Row best = null;
                Ablation.Row baseline = null;
                for (Ablation.Row r : group) {
                    if (r.isBaseline) {
                        baseline = r;
                    }
                    if (ok(r) && !Double.isNaN(r.agg.valMacro) && (best == null || r.agg.valMacro > best.agg.valMacro)) {
                        best = r;
                    }
                }

                sb.append("**").append(param).append("**\n\n");
                sb.append("| Value | Train macro-F1 | Val macro-F1 | Test macro-F1 | Train−Val gap | Δ test vs baseline | |\n");
                sb.append("|---|---:|---:|---:|---:|---:|---|\n");

                for (Ablation.Row r : group) {
                    if (!ok(r)) {
                        sb.append("| ").append(r.value).append(" | failed | | | | | ").append(r.agg.error)
                                .append(" |\n");
                        continue;
                    }
                    double delta = ok(baseline) ? r.agg.score() - baseline.agg.score() : Double.NaN;
                    sb.append("| ").append(r.value).append(" | ").append(num(r.agg.trainMacro)).append(" | ")
                            .append(withStd(r.agg.valMacro, r.agg.valMacroStd, seeds)).append(" | ")
                            .append(withStd(r.agg.score(), r.agg.scoreStd(), seeds)).append(" | ")
                            .append(signed(r.agg.trainMacro - r.agg.valMacro)).append(" | ").append(signed(delta))
                            .append(" | ").append(r.isBaseline ? "◀ " : "").append(r == best ? "★" : "")
                            .append(" |\n");
                }
                sb.append("\n");

                if (best != null) {
                    bestSummary.append("| ").append(param).append(" | ").append(best.value).append(" | ")
                            .append(num(best.agg.valMacro)).append(" | ").append(num(best.agg.score())).append(" | ")
                            .append(baseline == null ? "n/a" : baseline.value).append(" | ")
                            .append(ok(baseline) ? num(baseline.agg.score()) : "n/a").append(" |\n");
                }
            }
            bestSummary.append("\n");

            if (model.equals(Ablation.TREE)) {
                appendTreeGrid(sb, rows);
            }
        }

        sb.append("### Best value of each parameter\n\n");
        sb.append("Each parameter is tuned on its own (the others stay at baseline), so combining all the best values ")
                .append("is a *candidate* that still has to be trained and checked, not a guaranteed improvement.\n\n");
        sb.append(bestSummary);
    }

    private static void appendTreeGrid(StringBuilder sb, List<Ablation.Row> rows) {

        List<String> depths = new ArrayList<>();
        List<String> leaves = new ArrayList<>();
        for (Ablation.Row r : rows) {
            if (r.kind.equals("grid") && r.model.equals(Ablation.TREE)) {
                String[] parts = r.value.split("x");
                if (!depths.contains(parts[0])) {
                    depths.add(parts[0]);
                }
                if (!leaves.contains(parts[1])) {
                    leaves.add(parts[1]);
                }
            }
        }
        if (depths.isEmpty()) {
            return;
        }

        for (String metric : new String[] { "Validation macro-F1", "Test macro-F1" }) {
            sb.append("**Decision tree: max_depth (rows) × min_samples_leaf (columns), ").append(metric)
                    .append("**\n\n| depth \\ leaf |");
            for (String leaf : leaves) {
                sb.append(" ").append(leaf).append(" |");
            }
            sb.append("\n|---|");
            for (int i = 0; i < leaves.size(); i++) {
                sb.append("---:|");
            }
            sb.append("\n");
            for (String depth : depths) {
                sb.append("| ").append(depth).append(" |");
                for (String leaf : leaves) {
                    Ablation.Row r = find(rows, "grid", Ablation.TREE, "depth=" + depth + ", leaf=" + leaf);
                    double v = !ok(r) ? Double.NaN : (metric.startsWith("Val") ? r.agg.valMacro : r.agg.score());
                    sb.append(" ").append(num(v)).append(" |");
                }
                sb.append("\n");
            }
            sb.append("\n");
        }
    }


    private static final Map<String, String> FAMILY_TITLES = new LinkedHashMap<>();
    static {
        FAMILY_TITLES.put("casing", "Casing of the text (preserve / lower-case / hybrid)");
        FAMILY_TITLES.put("char_n", "Character n-gram order n");
        FAMILY_TITLES.put("char_buckets", "Hash buckets of the character profile");
        FAMILY_TITLES.put("char_typed", "Typed character n-grams (affix + punctuation)");
        FAMILY_TITLES.put("pos_n", "POS n-gram order");
        FAMILY_TITLES.put("pos_buckets", "Hash buckets of the POS profile");
        FAMILY_TITLES.put("pos_tagger", "POS tagger");
        FAMILY_TITLES.put("function_words", "Function-word profile");
        FAMILY_TITLES.put("pair_representation", "Pair representation (how two sentences become one vector)");
        FAMILY_TITLES.put("z_clip", "Cut of the z-scores of the scalar features");
        FAMILY_TITLES.put("block_differences", "Compressed (4 measures per profile) vs full (every bucket difference)");
        FAMILY_TITLES.put("char_grid", "Character n-gram order x buckets (grid)");
    }

    private static List<String> pipelineFamilies(List<Ablation.Row> rows) {
        List<String> families = new ArrayList<>();
        for (Ablation.Row r : rows) {
            if (r.kind.equals("pipeline") && !families.contains(r.group)) {
                families.add(r.group);
            }
        }
        return families;
    }

    private static List<String> pipelineSettings(List<Ablation.Row> rows, String family) {
        List<String> settings = new ArrayList<>();
        for (Ablation.Row r : rows) {
            if (r.kind.equals("pipeline") && r.group.equals(family) && !settings.contains(r.value)) {
                settings.add(r.value);
            }
        }
        return settings;
    }

    private static Ablation.Row pipelineRow(List<Ablation.Row> rows, String family, String setting, String model) {
        for (Ablation.Row r : rows) {
            if (r.kind.equals("pipeline") && r.group.equals(family) && r.value.equals(setting)
                    && r.model.equals(model)) {
                return r;
            }
        }
        return null;
    }

    private static List<String> pipelineModels(List<Ablation.Row> rows) {
        List<String> models = new ArrayList<>();
        for (Ablation.Row r : rows) {
            if (r.kind.equals("pipeline") && !models.contains(r.model)) {
                models.add(r.model);
            }
        }
        return models;
    }

    private static void appendPipelineSection(StringBuilder sb, List<Ablation.Row> rows, int seeds) {

        List<String> families = pipelineFamilies(rows);
        if (families.isEmpty()) {
            return;
        }
        List<String> models = pipelineModels(rows);

        sb.append("## 3. Feature-pipeline ablation\n\n");
        sb.append("The features are computed again for every setting (casing, character n-gram order and buckets, ")
                .append("POS n-grams, pair representation, ...) and the models are trained with the baseline ")
                .append("hyperparameters. Every cell is `validation / test` macro-F1; `◀` = the baseline setting, ")
                .append("`★` = best validation macro-F1 of that model within the family.\n\n");

        for (String family : families) {

            sb.append("### ").append(FAMILY_TITLES.getOrDefault(family, family)).append("\n\n");

            sb.append("| Setting | Pair features |");
            for (String m : models) {
                sb.append(" ").append(title(m)).append(" |");
            }
            sb.append("\n|---|---:|");
            for (int i = 0; i < models.size(); i++) {
                sb.append("---:|");
            }
            sb.append("\n");

            List<String> settings = pipelineSettings(rows, family);

            // best validation score per model within the family
            Map<String, Double> bestValidation = new LinkedHashMap<>();
            for (String m : models) {
                double best = Double.NEGATIVE_INFINITY;
                for (String setting : settings) {
                    Ablation.Row r = pipelineRow(rows, family, setting, m);
                    if (ok(r) && !Double.isNaN(r.agg.valMacro)) {
                        best = Math.max(best, r.agg.valMacro);
                    }
                }
                bestValidation.put(m, best);
            }

            for (String setting : settings) {

                Ablation.Row any = null;
                for (String m : models) {
                    Ablation.Row r = pipelineRow(rows, family, setting, m);
                    if (r != null) {
                        any = r;
                        break;
                    }
                }

                sb.append("| ").append(setting).append(any != null && any.isBaseline ? " ◀" : "").append(" | ")
                        .append(any != null && any.agg.nFeatures > 0 ? String.valueOf(any.agg.nFeatures) : "n/a")
                        .append(" |");

                for (String m : models) {
                    Ablation.Row r = pipelineRow(rows, family, setting, m);
                    if (!ok(r)) {
                        sb.append(r == null ? " not run |" : " failed |");
                        continue;
                    }
                    boolean best = !Double.isNaN(r.agg.valMacro) && r.agg.valMacro >= bestValidation.get(m) - 1e-12;
                    sb.append(" ").append(num(r.agg.valMacro)).append(" / ")
                            .append(withStd(r.agg.score(), r.agg.scoreStd(), seeds)).append(best ? " ★" : "")
                            .append(" |");
                }
                sb.append("\n");
            }
            sb.append("\n");
        }
    }

    private static void appendCaveats(StringBuilder sb) {

        sb.append("## 4. How to read this (limitations to state in the paper)\n\n");
        sb.append("- **Correlated groups.** If two groups carry the same information (e.g. sentence length and ")
                .append("word length), removing one costs little because the other compensates, so a small ")
                .append("importance does not prove a group is useless. The \"only G\" column helps tell the two ")
                .append("cases apart.\n");
        sb.append("- **One parameter at a time** ignores interactions between parameters (the decision-tree grid is the ")
                .append("only place interactions are looked at).\n");
        sb.append("- **Single run vs seeds.** Without several seeds there is no variance estimate; differences of a few ")
                .append("thousandths of F1 should not be interpreted. Run with `-Dseeds=42,123,456` for the final numbers.\n");
        sb.append("- **POS tags** come from the lightweight rule-based tagger (features.HeuristicPosTagger) unless ")
                .append("`pos_tagger: corenlp` is in the pipeline table, so the POS result describes that tagger.\n");
        sb.append("- **Pipeline settings are compared one at a time** around the baseline. The best value of each family ")
                .append("is a candidate, not a proven combination (the optional `char_grid` family looks at n x buckets together).\n");
        sb.append("- Parameters and pipeline settings are chosen on validation macro-F1; the test column is reported once, for the chosen values, ")
                .append("and should not be used to pick settings.\n");
    }

    private static String tex(String s) {
        return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
                .replace("#", "\\#").replace("±", "$\\pm$").replace("−", "-");
    }

    private static String latex(List<Ablation.Row> rows) {

        StringBuilder sb = new StringBuilder();
        sb.append("% Generated by Ablation.java. Needs \\usepackage{booktabs}.\n\n");

        for (String model : models(rows)) {

            Ablation.Row base = find(rows, "feature", model, "ALL");
            if (!ok(base)) {
                continue;
            }

            sb.append("\\begin{table}[ht]\n\\centering\n\\small\n");
            sb.append("\\caption{Feature-group ablation, ").append(tex(title(model))).append(" (baseline macro-F1 = ")
                    .append(num(base.agg.score())).append("). Importance $= F_{all} - F_{-G}$ (macro-F1).}\n");
            sb.append("\\label{tab:ablation-features-").append(model).append("}\n");
            sb.append("\\begin{tabular}{lrrrrrr}\n\\toprule\n");
            sb.append("Feature group & $F_{-G}$ & Importance & $\\Delta$ easy & $\\Delta$ medium & $\\Delta$ hard & $F_{G\\text{ only}}$ \\\\\n\\midrule\n");

            for (Importance i : importances(rows, model)) {
                sb.append(tex(i.group)).append(" & ").append(num(i.withoutF1)).append(" & ")
                        .append(signed(i.importance)).append(" & ").append(signed(i.easy)).append(" & ")
                        .append(signed(i.medium)).append(" & ").append(signed(i.hard)).append(" & ")
                        .append(num(i.onlyF1)).append(" \\\\\n");
            }
            sb.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n");
        }

        // one table per (model, parameter)
        for (String model : models(rows)) {
            Set<String> params = new LinkedHashSet<>();
            for (Ablation.Row r : rows) {
                if (r.kind.equals("param") && r.model.equals(model)) {
                    params.add(r.param);
                }
            }
            for (String param : params) {
                sb.append("\\begin{table}[ht]\n\\centering\n\\small\n");
                sb.append("\\caption{Parameter ablation, ").append(tex(title(model))).append(": ")
                        .append(tex(param)).append(".}\n");
                sb.append("\\label{tab:ablation-").append(model).append("-").append(param.replace("_", "-"))
                        .append("}\n");
                sb.append("\\begin{tabular}{lrrr}\n\\toprule\nValue & Train & Val & Test \\\\\n\\midrule\n");
                for (Ablation.Row r : rows) {
                    if (r.kind.equals("param") && r.model.equals(model) && r.param.equals(param) && ok(r)) {
                        sb.append(tex(r.value)).append(r.isBaseline ? " (baseline)" : "").append(" & ")
                                .append(num(r.agg.trainMacro)).append(" & ").append(num(r.agg.valMacro)).append(" & ")
                                .append(num(r.agg.score())).append(" \\\\\n");
                    }
                }
                sb.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n");
            }
        }

        // one table per pipeline family (test macro-F1 of every model)
        List<String> pipelineModels = pipelineModels(rows);
        for (String family : pipelineFamilies(rows)) {
            sb.append("\\begin{table}[ht]\n\\centering\n\\small\n");
            sb.append("\\caption{Feature pipeline ablation: ").append(tex(FAMILY_TITLES.getOrDefault(family, family)))
                    .append(" (test macro-F1).}\n");
            sb.append("\\label{tab:pipeline-").append(family.replace("_", "-")).append("}\n");
            sb.append("\\begin{tabular}{lr");
            for (int i = 0; i < pipelineModels.size(); i++) {
                sb.append("r");
            }
            sb.append("}\n\\toprule\nSetting & Features");
            for (String m : pipelineModels) {
                sb.append(" & ").append(tex(title(m)));
            }
            sb.append(" \\\\\n\\midrule\n");
            for (String setting : pipelineSettings(rows, family)) {
                Ablation.Row first = null;
                for (String m : pipelineModels) {
                    Ablation.Row r = pipelineRow(rows, family, setting, m);
                    if (r != null && first == null) {
                        first = r;
                    }
                }
                sb.append(tex(setting)).append(first != null && first.isBaseline ? " (baseline)" : "").append(" & ")
                        .append(first != null ? String.valueOf(first.agg.nFeatures) : "");
                for (String m : pipelineModels) {
                    Ablation.Row r = pipelineRow(rows, family, setting, m);
                    sb.append(" & ").append(ok(r) ? num(r.agg.score()) : "n/a");
                }
                sb.append(" \\\\\n");
            }
            sb.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n");
        }

        return sb.toString();
    }
}
