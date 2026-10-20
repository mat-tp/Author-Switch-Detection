import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import com.hyp.authorswitchdetection.model.ModelMetrics;
import com.hyp.authorswitchdetection.model.ModelMetricsIO;

import data.DatasetLoader;
import data.RetrievalData;
import features.FeatureConfig;
import features.FeatureExtractor;
import models.EmbeddingNetwork;
import models.LogisticRegression;
import models.Metrics;
import models.ModelStore;
import models.RetrievalEvaluator;
import models.RetrievalModel;
import models.RetrievalPipeline;
import models.RetrievalPipeline.Documents;
import models.RetrievalPipeline.Result;
import models.RetrievalPipeline.Settings;

/**
 * The retrieval model: the sentences are NOT paired. Every sentence is turned into its feature vector
 * (the same features as the other three models, see FeatureExtractor.embeddingInput) and a small network
 * learns an embedding in which sentences of the same segment (same author) are close and sentences of
 * the neighbouring segment (another author) are far apart (triplet loss, see models.EmbeddingNetwork).
 *
 * It is then used for
 * - retrieval: "which sentences of this document are written like this one"
 * - switch detection: a large distance between the sentences left and right of a position = a change
 *
 * java -Ddataset=dataset/pseudo-mawsa/ -cp out RetrievalMain          train, evaluate, save
 * java -Ddataset=dataset/pseudo-mawsa/ -cp out RetrievalMain ablate   sweep the settings, write a csv
 *
 * Options (-D), besides the feature options of FeatureConfig:
 * ret.epochs 30 | ret.triplets 20000 (per epoch) | ret.batch 64 | ret.lr 0.001 | ret.margin 0.5
 * ret.dim 32 | ret.hidden 128 (comma separated, "" = none) | ret.l2 1e-4 | ret.candidates 1 (negatives per
 * triplet, the hardest is used) | ret.crossdoc 0 | ret.patience 5 | ret.scale 4 (volume of the profiles) |
 * ret.windows 1,2,3 (chosen on validation) | ret.maxsentences (limit of training sentences)
 * dataset | models.dir | metrics.file | full.validation | ret.csv (ablation output)
 */
public class RetrievalMain {

    static final String DATASET = System.getProperty("dataset", "dataset/mawsa26-pan-zenodo-DATA/");
    static final String MODELS_DIR = System.getProperty("models.dir", "data/models/");
    static final String METRICS_FILE = System.getProperty("metrics.file", "data/model-metrics.json");
    static final boolean USE_FULL_VALIDATION = Boolean.getBoolean("full.validation");
    static final String[] DIFFICULTIES = data.Corpus.DIFFICULTIES;

    public static void main(String[] args) throws IOException {

        String mode = args.length > 0 ? args[0] : "train";
        Documents documents = RetrievalPipeline.loadDocuments(DATASET, USE_FULL_VALIDATION);
        if (documents.train.isEmpty()) {
            System.err.println("No training data was loaded from " + DATASET);
            System.exit(1);
        }

        if (mode.equals("ablate")) {
            ablate(documents);
            return;
        }

        Result r = RetrievalPipeline.run(new Settings(), documents, true);
        print(r);
        save(r);
    }

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------

    static void print(Result r) {

        String split = r.onTest ? "test" : "validation";
        System.out.println();
        System.out.println("==================== Retrieval model (" + split + " documents) ====================");
        System.out.println("Retrieval inside a document (relevant = same segment, so a lower bound):");
        System.out.println("  learned embedding : " + r.retrieval);
        System.out.println("  raw feature cosine: " + r.rawBaseline);
        System.out.println("  random vectors    : " + r.randomBaseline);
        System.out.println();
        System.out.println(String.format(Locale.ROOT,
                "Switch detection (window %d, threshold %.3f chosen on validation: macro-F1 %.4f, AUC %.4f):",
                r.window, r.threshold, r.validationMacroF1, r.validationAuc));
        System.out.println(String.format(Locale.ROOT,
                "  learned embedding : %s macro-F1 %.4f | AUC %.4f", split, r.metrics[5], r.testAuc));
        System.out.println(String.format(Locale.ROOT,
                "  raw feature cosine: %s macro-F1 %.4f | AUC %.4f", split, r.rawMacroF1, r.rawAuc));
    }

    static void save(Result r) {

        String path = MODELS_DIR + "retrieval.ser";
        try {
            ModelStore.save(path, ModelStore.RETRIEVAL, r.model, r.threshold, r.extractor, r.description);
            System.out.println("Model saved successfully to " + path);
        } catch (IOException e) {
            System.err.println("Error " + e.getMessage());
            e.printStackTrace();
        }

        ModelMetrics m = new ModelMetrics();
        m.setModelName("retrieval_embedding");
        m.setVersionName("retrieval_embedding_cli_" + (r.onTest ? "test" : "validation"));
        m.setFull(true);
        m.setNProblems(r.trainSentences);
        m.setPrecision(r.metrics[1]);
        m.setRecall(r.metrics[2]);
        m.setF1(r.metrics[3]);
        m.setBalancedAccuracy(r.metrics[4]);
        m.setMacroF1(r.metrics[5]);
        m.setDecisionThreshold(r.threshold);
        m.setEvaluatedOnTest(r.onTest);
        m.setDifficultyBreakdownCsv(r.breakdown);
        m.setTrainedBy("cli");
        m.setFeatureConfig(r.extractor.getConfig().signature());
        m.setChosenConfig(String.format(Locale.ROOT,
                "%s | window=%d | retrieval P@1 %.3f MAP %.3f (raw features: P@1 %.3f MAP %.3f) | switch AUC %.3f (raw %.3f)",
                r.description, r.window, r.retrieval.precisionAt1, r.retrieval.map, r.rawBaseline.precisionAt1,
                r.rawBaseline.map, r.testAuc, r.rawAuc)
                + (r.onTest ? " — evaluated on the held-out test split"
                        : " — no test split was available; evaluated on the validation split"));
        ModelMetricsIO.append(METRICS_FILE, m);
        System.out.println("Metrics row appended to " + METRICS_FILE);
    }

    // ------------------------------------------------------------------
    // Ablation of the retrieval model
    // ------------------------------------------------------------------

    /** One line of the ablation: a name and what changes in the settings. */
    private static class Variant {
        final String family, name;
        final java.util.function.Consumer<Settings> change;

        Variant(String family, String name, java.util.function.Consumer<Settings> change) {
            this.family = family;
            this.name = name;
            this.change = change;
        }
    }

    /**
     * Trains one model per variant (everything else as in the baseline settings) and writes one csv row
     * each. The choice of a setting is to be made on the validation columns, the test columns are the
     * report. The retrieval measures are on the same split as the switch columns.
     */
    static void ablate(Documents documents) throws IOException {

        List<Variant> variants = new ArrayList<>();
        variants.add(new Variant("baseline", "baseline", s -> {
        }));

        for (int dim : new int[] { 8, 16, 64 }) {
            variants.add(new Variant("embedding_size", "dim=" + dim, s -> s.dim = dim));
        }
        for (double margin : new double[] { 0.2, 1.0 }) {
            variants.add(new Variant("margin", "margin=" + margin, s -> s.margin = margin));
        }
        variants.add(new Variant("hidden", "no hidden layer (linear projection)", s -> s.hidden = new int[0]));
        variants.add(new Variant("hidden", "hidden=256,128", s -> s.hidden = new int[] { 256, 128 }));
        variants.add(new Variant("mining", "hardest of 4 negatives", s -> s.candidates = 4));
        variants.add(new Variant("mining", "hardest of 8 negatives", s -> s.candidates = 8));
        variants.add(new Variant("negatives", "+ negatives from other documents (50%)", s -> s.crossDocument = 0.5));
        for (double scale : new double[] { 0.0, 2.0, 8.0 }) {
            variants.add(new Variant("profile_scale", "scale=" + scale
                    + (scale == 0.0 ? " (profiles off, scalar features only)" : ""), s -> s.profileScale = scale));
        }
        for (int[] w : new int[][] { { 1 }, { 3 }, { 5 } }) {
            variants.add(new Variant("window", "window=" + w[0], s -> s.windows = w));
        }
        for (int n = 1; n <= 5; n++) {
            final int order = n;
            variants.add(new Variant("char_ngram", "char n=" + n, s -> s.features.charNGramOrders = new int[] { order }));
        }
        variants.add(new Variant("char_ngram", "char n=1,2,3,4,5",
                s -> s.features.charNGramOrders = new int[] { 1, 2, 3, 4, 5 }));
        for (int b : new int[] { 64, 128, 512, 1024 }) {
            variants.add(new Variant("char_buckets", "buckets=" + b, s -> s.features.charBuckets = b));
        }
        for (FeatureConfig.Casing c : FeatureConfig.Casing.values()) {
            variants.add(new Variant("casing", "casing=" + c.name().toLowerCase(), s -> s.features.casing = c));
        }
        variants.add(new Variant("features", "no POS profile", s -> s.features.posNGramOrders = new int[0]));
        variants.add(new Variant("features", "no function-word profile", s -> s.features.functionWordProfile = false));
        variants.add(new Variant("features", "no character profile", s -> s.features.charNGramOrders = new int[0]));

        String csv = System.getProperty("ret.csv", "data/retrieval-ablation.csv");
        if (new java.io.File(csv).getAbsoluteFile().getParentFile() != null) {
            new java.io.File(csv).getAbsoluteFile().getParentFile().mkdirs();
        }

        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(Paths.get(csv), StandardCharsets.UTF_8))) {

            out.println("family,variant,split,val_macro_f1,val_auc,window,macro_f1,auc,raw_macro_f1,raw_auc,"
                    + "p_at_1,recall_at_5,mrr,map,triplet_acc,raw_p_at_1,raw_map,random_map,train_sentences,description");

            for (Variant v : variants) {

                Settings s = new Settings();
                v.change.accept(s);
                try {
                    s.features.validate();
                } catch (IllegalArgumentException bad) {
                    System.out.println("skipped " + v.name + ": " + bad.getMessage());
                    continue;
                }

                System.out.println();
                System.out.println("---- " + v.family + ": " + v.name);
                Result r = RetrievalPipeline.run(s, documents, false);

                out.println(String.format(Locale.ROOT,
                        "%s,\"%s\",%s,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,\"%s\"",
                        v.family, v.name, r.onTest ? "test" : "validation", r.validationMacroF1, r.validationAuc,
                        r.window, r.metrics[5], r.testAuc, r.rawMacroF1, r.rawAuc, r.retrieval.precisionAt1,
                        r.retrieval.recallAt5, r.retrieval.mrr, r.retrieval.map, r.retrieval.tripletAccuracy,
                        r.rawBaseline.precisionAt1, r.rawBaseline.map, r.randomBaseline.map, r.trainSentences,
                        r.description.replace("\"", "'")));
                out.flush();

                System.out.println(String.format(Locale.ROOT,
                        "   switch macro-F1 %.4f (validation %.4f) AUC %.4f | retrieval MAP %.4f P@1 %.4f",
                        r.metrics[5], r.validationMacroF1, r.testAuc, r.retrieval.map, r.retrieval.precisionAt1));
            }
        }
        System.out.println();
        System.out.println("Retrieval ablation written to " + csv);
    }
}
