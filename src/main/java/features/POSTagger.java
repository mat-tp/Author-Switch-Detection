package features;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.util.List;
import java.util.Properties;

/**
 * To properly train the POS Tagger can take a long time and needs alot of data
 * not just the current one.
 * For this reason, we will use the Stanford CoreNLP library to perform POS
 * tagging. The Stanford CoreNLP
 * library is a Java-based natural language processing library that provides
 * various NLP tools, including POS tagging.
 *
 * This is the CoreNLP backed implementation of PosTagger (the built in one is
 * HeuristicPosTagger). It is used with -Dpos.tagger=corenlp, and it is only
 * created by name (see PosTagger.create), so the command line scripts compile and run
 * without the CoreNLP jars as long as the heuristic tagger is used.
 *
 * Changes to the first version of this file:
 * - the missing import of CoreAnnotations is added (the file did not compile)
 * - it returns the COARSE tags of PosTagset (no more int[36] count array that was shared by
 * all the sentences and kept adding up between calls)
 * - the tokens are given to CoreNLP as they are (whitespace tokenisation), so the tags line up
 * one to one with the tokens the features count
 *
 * NOT TESTED in the environment this change was written in (no access to Maven Central).
 * Check it once with: java -cp ... features.POSTagger "The cat sat on the mat ."
 */
public class POSTagger implements PosTagger {

    private final StanfordCoreNLP pipeline;

    public POSTagger() {
        // Set up the Stanford CoreNLP pipeline with the POS tagger
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos");
        // the tokens are already split by TextUtils.tokenizeWithPunctuation
        props.setProperty("tokenize.whitespace", "true");
        // one input = one sentence, never split it again
        props.setProperty("ssplit.isOneSentence", "true");
        this.pipeline = new StanfordCoreNLP(props);
    }

    @Override
    public int[] tag(List<String> tokens) {

        int[] tags = new int[tokens.size()];
        java.util.Arrays.fill(tags, PosTagset.OTHER);

        if (tokens.isEmpty()) {
            return tags;
        }

        // Create an Annotation object with the input text, and annotate it using the pipeline
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

    /** Quick check: java features.POSTagger "The cat sat on the mat ." */
    public static void main(String[] args) {

        String text = args.length > 0 ? String.join(" ", args) : "The cat sat on the mat .";
        List<String> tokens = TextUtils.tokenizeWithPunctuation(text);
        int[] tags = new POSTagger().tag(tokens);

        for (int i = 0; i < tokens.size(); i++) {
            System.out.println(tokens.get(i) + "/" + PosTagset.NAMES[tags[i]]);
        }
    }
}
