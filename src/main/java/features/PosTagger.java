package features;

import java.util.List;

/**
 * Something that gives every token of a sentence a coarse part-of-speech tag
 * (see PosTagset).
 *
 * Two implementations:
 * - HeuristicPosTagger: built in, no libraries, deterministic and fast. A closed-class
 * lexicon plus suffix and context rules. Less accurate than a trained tagger, but a
 * sentence-level style feature only needs the grammatical HABIT of the writer,
 * not a perfect tree (see docs/RESEARCH.md for the measured agreement).
 * - POSTagger: Stanford CoreNLP (more accurate, slower, needs the CoreNLP jars).
 *
 * The tagger is NOT saved in a model (it is rebuilt from FeatureConfig.posTagger),
 * so implementations may keep big objects.
 */
public interface PosTagger {

    /**
     * @param tokens the tokens of ONE sentence (see TextUtils.tokenizeWithPunctuation)
     * @return one coarse tag (PosTagset index) per token, same length as tokens
     */
    int[] tag(List<String> tokens);

    /**
     * The tagger that is chosen in the configuration. "corenlp" is looked up by name,
     * so the project still compiles (and the command line scripts still work) when the
     * CoreNLP jars are not on the class path.
     */
    static PosTagger create(String name) {

        if (name == null || name.equals("heuristic")) {
            return new HeuristicPosTagger();
        }

        if (name.equals("corenlp")) {
            try {
                return (PosTagger) Class.forName("features.POSTagger").getDeclaredConstructor().newInstance();

            } catch (ReflectiveOperationException | LinkageError e) {
                throw new IllegalStateException("pos.tagger=corenlp needs the Stanford CoreNLP jars on the class path "
                        + "(run through Maven, or add them with -cp). Cause: " + e, e);
            }
        }

        throw new IllegalArgumentException("Unknown POS tagger: " + name);
    }
}
