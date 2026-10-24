package evaluation;

import analysis.DocumentSegmentation;

/**
 * Measures how close the estimated author count of {@link DocumentSegmentation} comes to the count given in the
 * truth files. It only reports; it never feeds back into the choice of a threshold.
 */
public final class AuthorCountEvaluation {

    private int documents;
    private int exactMatches;
    private double absoluteErrorSum;
    private double signedErrorSum;

    /**
     * Adds one document to the evaluation.
     *
     * @param estimatedAuthors the estimate of the document
     * @param trueAuthors      the count from the truth file; documents with a count below one are ignored
     */
    public void add(int estimatedAuthors, int trueAuthors) {
        if (trueAuthors < 1) {
            return;
        }
        documents++;
        int error = estimatedAuthors - trueAuthors;
        absoluteErrorSum += Math.abs(error);
        signedErrorSum += error;
        if (error == 0) {
            exactMatches++;
        }
    }

    /** @return the number of documents that entered the evaluation */
    public int documents() {
        return documents;
    }

    /** @return the share of documents whose estimate equals the truth, 0 without documents */
    public double exactMatchRate() {
        return documents == 0 ? 0.0 : (double) exactMatches / documents;
    }

    /** @return the mean absolute difference between estimate and truth, 0 without documents */
    public double meanAbsoluteError() {
        return documents == 0 ? 0.0 : absoluteErrorSum / documents;
    }

    /** @return the mean of (estimate - truth); positive means the estimate is too high on average */
    public double meanSignedError() {
        return documents == 0 ? 0.0 : signedErrorSum / documents;
    }

    /** @return a one-line summary for the command line output */
    public String summary() {
        return String.format(java.util.Locale.ROOT,
                "author count over %d documents: exact %.3f, mean absolute error %.2f, mean signed error %+.2f",
                documents, exactMatchRate(), meanAbsoluteError(), meanSignedError());
    }
}
