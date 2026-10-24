package features;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.util.List;
import java.util.Properties;

/**
 * Stanford CoreNLP implementation of {@link PosTagger}.
 *
 * <p>Training a custom POS tagger from scratch requires significant data and compute time.
 * This class leverages the Stanford CoreNLP pipeline for robust, pre-trained POS tagging.
 * It serves as an alternative to the built-in {@code HeuristicPosTagger}.</p>
 *
 * <p><strong>Usage:</strong> Enabled via system property {@code -Dpos.tagger=corenlp}.
 * Instantiated dynamically by name (via {@code PosTagger.create}), allowing command-line
 * scripts to compile and run without requiring CoreNLP JARs unless explicitly selected.</p>
 *
 * <p><strong>Key Design Details:</strong>
 * <ul>
 *   <li>Uses whitespace tokenization to preserve input token boundaries 1:1.</li>
 *   <li>Maps Penn Treebank tags to coarse {@code PosTagset} categories.</li>
 *   <li>Returns a per-sentence tag array to prevent state accumulation across calls.</li>
 * </ul>
 * </p>
 */
public class POSTagger implements PosTagger {

    private final StanfordCoreNLP pipeline;

    /**
     * Initializes the Stanford CoreNLP pipeline configured strictly for POS tagging.
     */
    public POSTagger() {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos");

        // Tokens are pre-split by TextUtils.tokenizeWithPunctuation; enforce whitespace splitting
        props.setProperty("tokenize.whitespace", "true");

        // Treat input as a single complete sentence to prevent secondary sentence splitting
        props.setProperty("ssplit.isOneSentence", "true");

        this.pipeline = new StanfordCoreNLP(props);
    }

    /**
     * Tags a sequence of input tokens with coarse POS tag IDs.
     *
     * @param tokens List of token strings to tag.
     * @return Array of integer tag representations corresponding 1:1 with input tokens.
     */
    @Override
    public int[] tag(List<String> tokens) {

        int[] tags = new int[tokens.size()];
        java.util.Arrays.fill(tags, PosTagset.OTHER);

        if (tokens.isEmpty()) {
            return tags;
        }

        // Annotate token stream via Stanford CoreNLP pipeline
        Annotation document = new Annotation(String.join(" ", tokens));
        pipeline.annotate(document);

        int next = 0;
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {

                if (next >= tags.length) {
                    return tags;
                }
                tags[next++] = PosTagset.fromPenn(token.get(CoreAnnotations.PartOfSpeechAnnotation.class));
            }
        }

        return tags;
    }

    /**
     * Entry point for quick manual testing.
     * <p>Execution: {@code java -cp <classpath> features.POSTagger "The cat sat on the mat ."}</p>
     *
     * @param args Optional command line text arguments to tag.
     */
    public static void main(String[] args) {

        System.out.println("The cat sat on the mat .");
        String text = args.length > 0 ? String.join(" ", args) : "The cat sat on the mat .";
        List<String> tokens = TextUtils.tokenizeWithPunctuation(text);
        int[] tags = new POSTagger().tag(tokens);

        for (int i = 0; i < tokens.size(); i++) {
            System.out.println(tokens.get(i) + "/" + PosTagset.NAMES[tags[i]]);
        }
    }
}