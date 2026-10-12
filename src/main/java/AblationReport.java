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

    // an importance smaller than this (in F1) is called "negligible"
    private static final double NEGLIGIBLE = 0.005;

    private static final Map<String, String> MODEL_TITLES = new LinkedHashMap<>();
    static {
        MODEL_TITLES.put(Ablation.TREE, "Decision Tree");
        MODEL_TITLES.put(Ablation.LOGISTIC, "Logistic Regression");
        MODEL_TITLES.put(Ablation.NEURAL, "Neural Network");
    }

    private AblationReport() {
    }

    static void write(List<Ablation.Row> rows, Ablation.Data data, String outDir, double minutes) throws IOException {

        Files.writeString(Paths.get(outDir, "ablation_report.md"), markdown(rows, data, minutes),
                StandardCharsets.UTF_8);
        Files.writeString(Paths.get(outDir, "ablation_tables.tex"), latex(rows), StandardCharsets.UTF_8);
    }

    // ---------------------------------------------------------------------
    // lookups
    // ---------------------------------------------------------------------

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
        Double b = base.agg.f1ByDifficulty.get(difficulty);
        Double o = other.agg.f1ByDifficulty.get(difficulty);
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
            i.withoutF1 = ok(without) ? without.agg.evalF1() : Double.NaN;
            i.importance = ok(without) ? base.agg.evalF1() - without.agg.evalF1() : Double.NaN;
            i.easy = diffF1(base, without, "easy");
            i.medium = diffF1(base, without, "medium");
            i.hard = diffF1(base, without, "hard");
            i.onlyF1 = ok(only) ? only.agg.evalF1() : Double.NaN;
            double noise = ok(without) ? Math.max(base.agg.evalStd[3], without.agg.evalStd[3]) : 0.0;
            i.verdict = verdict(i.importance, noise);
            list.add(i);
        }

        // biggest contribution first (failed runs last)
        list.sort(Comparator.comparingDouble((Importance i) -> Double.isNaN(i.importance) ? -1e9 : i.importance)
                .reversed());
        return list;
    }

    // ---------------------------------------------------------------------
    // Markdown
    // ---------------------------------------------------------------------

    private static String markdown(List<Ablation.Row> rows, Ablation.Data data, double minutes) {

        StringBuilder sb = new StringBuilder();
        int seeds = rows.isEmpty() ? 1 : rows.get(0).agg.seeds;
        String evalName = data.evalIsTest ? "held-out test half" : "validation split (no test split available)";

        sb.append("# Ablation study report\n\n");
        sb.append("- Training pairs: ").append(data.trainY.length).append("\n");
        sb.append("- Evaluation pairs (").append(evalName).append("): ").append(data.evalY.length).append("\n");
        sb.append("- Pair features in the full model: ").append(data.train[0].length).append("\n");
        sb.append("- Seeds: ").append(Ablation.seeds()).append(seeds > 1
                ? " (each seed = a different random " + (int) (Ablation.SEED_FRACTION * 100)
                        + "% training subsample; numbers are mean ± std)"
                : " (single run, no variance estimate)")
                .append("\n");
        sb.append(String.format(Locale.ROOT, "- Total run time: %.1f min\n\n", minutes));

        sb.append("**Metric:** F1 of the *change* class on the ").append(evalName).append(". ")
                .append("Thresholds (logistic regression, neural network) are chosen on the validation data, ")
                .append("never on the evaluation data. Parameters are ranked by **validation** F1.\n\n");

        sb.append("## 1. Feature-group ablation\n\n");
        sb.append("`Importance(G) = F1_all - F1_without_G`. Positive = the group carries useful information. ")
                .append("Near 0 = it adds nothing the other groups do not already provide. ")
                .append("Negative = the model does better without it. ")
                .append("\"Only G\" trains on that group alone. ")
                .append("The per-difficulty columns are the same difference on easy / medium / hard pairs.\n\n");

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

            sb.append("Baseline (all ").append(base.agg.nFeatures).append(" features): F1 = ")
                    .append(withStd(base.agg.evalF1(), base.agg.evalStd[3], seeds)).append("\n\n");

            sb.append("| Rank | Feature group | F1 without G | Importance | Δ easy | Δ medium | Δ hard | F1 only G | Verdict |\n");
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
        appendParameterSection(sb, rows, seeds);
        appendCaveats(sb);

        return sb.toString();
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
                sb.append(" ").append(ok(base) && ok(without) ? signed(base.agg.evalF1() - without.agg.evalF1()) : "n/a")
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
                .append("`◀` = the baseline value, `★` = best validation F1 of that parameter. ")
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
            bestSummary.append("| Parameter | Best value (by validation F1) | Validation F1 | Test F1 | Baseline value | Baseline test F1 |\n");
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
                    if (ok(r) && !Double.isNaN(r.agg.valF1) && (best == null || r.agg.valF1 > best.agg.valF1)) {
                        best = r;
                    }
                }

                sb.append("**").append(param).append("**\n\n");
                sb.append("| Value | Train F1 | Val F1 | Test F1 | Train−Val gap | Δ test vs baseline | |\n");
                sb.append("|---|---:|---:|---:|---:|---:|---|\n");

                for (Ablation.Row r : group) {
                    if (!ok(r)) {
                        sb.append("| ").append(r.value).append(" | failed | | | | | ").append(r.agg.error)
                                .append(" |\n");
                        continue;
                    }
                    double delta = ok(baseline) ? r.agg.evalF1() - baseline.agg.evalF1() : Double.NaN;
                    sb.append("| ").append(r.value).append(" | ").append(num(r.agg.trainF1)).append(" | ")
                            .append(withStd(r.agg.valF1, r.agg.valF1Std, seeds)).append(" | ")
                            .append(withStd(r.agg.evalF1(), r.agg.evalStd[3], seeds)).append(" | ")
                            .append(signed(r.agg.trainF1 - r.agg.valF1)).append(" | ").append(signed(delta))
                            .append(" | ").append(r.isBaseline ? "◀ " : "").append(r == best ? "★" : "")
                            .append(" |\n");
                }
                sb.append("\n");

                if (best != null) {
                    bestSummary.append("| ").append(param).append(" | ").append(best.value).append(" | ")
                            .append(num(best.agg.valF1)).append(" | ").append(num(best.agg.evalF1())).append(" | ")
                            .append(baseline == null ? "n/a" : baseline.value).append(" | ")
                            .append(ok(baseline) ? num(baseline.agg.evalF1()) : "n/a").append(" |\n");
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

        for (String metric : new String[] { "Validation F1", "Test F1" }) {
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
                    double v = !ok(r) ? Double.NaN : (metric.startsWith("Val") ? r.agg.valF1 : r.agg.evalF1());
                    sb.append(" ").append(num(v)).append(" |");
                }
                sb.append("\n");
            }
            sb.append("\n");
        }
    }

    private static void appendCaveats(StringBuilder sb) {

        sb.append("## 3. How to read this (limitations to state in the dissertation)\n\n");
        sb.append("- **Correlated groups.** If two groups carry the same information (e.g. sentence length and ")
                .append("word length), removing one costs little because the other compensates, so a small ")
                .append("importance does not prove a group is useless. The \"only G\" column helps tell the two ")
                .append("cases apart.\n");
        sb.append("- **One parameter at a time** ignores interactions between parameters (the decision-tree grid is the ")
                .append("only place interactions are looked at).\n");
        sb.append("- **Single run vs seeds.** Without several seeds there is no variance estimate; differences of a few ")
                .append("thousandths of F1 should not be interpreted. Run with `-Dseeds=42,123,456` for the final numbers.\n");
        sb.append("- **POS tags** come from the lightweight rule-based tagger (features.PosTagger), so the POS result ")
                .append("describes that tagger, not a statistical one.\n");
        sb.append("- Parameters are chosen on validation F1; the test column is reported once, for the chosen values, ")
                .append("and should not be used to pick settings.\n");
    }

    // ---------------------------------------------------------------------
    // LaTeX
    // ---------------------------------------------------------------------

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
            sb.append("\\caption{Feature-group ablation, ").append(tex(title(model))).append(" (baseline F1 = ")
                    .append(num(base.agg.evalF1())).append("). Importance $= F1_{all} - F1_{-G}$.}\n");
            sb.append("\\label{tab:ablation-features-").append(model).append("}\n");
            sb.append("\\begin{tabular}{lrrrrrr}\n\\toprule\n");
            sb.append("Feature group & $F1_{-G}$ & Importance & $\\Delta$ easy & $\\Delta$ medium & $\\Delta$ hard & $F1_{G\\text{ only}}$ \\\\\n\\midrule\n");

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
                sb.append("\\begin{tabular}{lrrr}\n\\toprule\nValue & Train F1 & Val F1 & Test F1 \\\\\n\\midrule\n");
                for (Ablation.Row r : rows) {
                    if (r.kind.equals("param") && r.model.equals(model) && r.param.equals(param) && ok(r)) {
                        sb.append(tex(r.value)).append(r.isBaseline ? " (baseline)" : "").append(" & ")
                                .append(num(r.agg.trainF1)).append(" & ").append(num(r.agg.valF1)).append(" & ")
                                .append(num(r.agg.evalF1())).append(" \\\\\n");
                    }
                }
                sb.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n");
            }
        }

        return sb.toString();
    }
}
