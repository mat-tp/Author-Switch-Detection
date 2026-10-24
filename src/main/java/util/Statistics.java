package util;

import java.util.Arrays;

/**
 * Descriptive statistics shared by feature scaling, the difficulty detector and the ablation report.
 *
 * <p>Every method is a pure function. Where a population and a sample estimate both exist, the name says
 * which one is meant, so that no caller has to guess the divisor.</p>
 */
public final class Statistics {

    private Statistics() {
        // utility class, not meant to be instantiated
    }

    /**
     * Arithmetic mean.
     *
     * @param values the observations
     * @return the mean, or 0 for an empty array
     */
    public static double mean(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    /**
     * Population standard deviation (divisor n), used when the observations are the whole population,
     * for example the whole training set of a scaler.
     *
     * @param values the observations
     * @return the standard deviation, or 0 for an empty array
     */
    public static double populationStandardDeviation(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }
        return Math.sqrt(sumOfSquaredDeviations(values, mean(values)) / values.length);
    }

    /**
     * Sample standard deviation (divisor n - 1), used when the observations are a sample, for example
     * the repeated runs of one experiment.
     *
     * @param values the observations
     * @return the standard deviation, or 0 when there are fewer than two observations
     */
    public static double sampleStandardDeviation(double[] values) {
        if (values.length < 2) {
            return 0.0;
        }
        return Math.sqrt(sumOfSquaredDeviations(values, mean(values)) / (values.length - 1));
    }

    /**
     * Nearest-rank percentile.
     *
     * @param values   the observations (the array is not modified)
     * @param fraction the position between 0 (minimum) and 1 (maximum)
     * @return the percentile, or 0 for an empty array
     */
    public static double percentile(double[] values, double fraction) {
        if (values.length == 0) {
            return 0.0;
        }
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int rank = (int) Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1) + 0.5));
        return sorted[rank];
    }

    /**
     * Mean of a block of columns.
     *
     * @param rows        the rows of the matrix
     * @param firstColumn index of the first column of the block
     * @param columnCount number of columns in the block
     * @return one mean per column of the block
     */
    public static double[] columnMeans(Iterable<double[]> rows, int firstColumn, int columnCount) {
        double[] means = new double[columnCount];
        int rowCount = 0;
        for (double[] row : rows) {
            for (int column = 0; column < columnCount; column++) {
                means[column] += row[firstColumn + column];
            }
            rowCount++;
        }
        for (int column = 0; column < columnCount; column++) {
            means[column] /= rowCount;
        }
        return means;
    }

    /**
     * Population standard deviation of a block of columns.
     *
     * @param rows        the rows of the matrix
     * @param firstColumn index of the first column of the block
     * @param means       the column means, as returned by {@link #columnMeans}
     * @return one standard deviation per column of the block
     */
    public static double[] columnStandardDeviations(Iterable<double[]> rows, int firstColumn, double[] means) {
        double[] deviations = new double[means.length];
        int rowCount = 0;
        for (double[] row : rows) {
            for (int column = 0; column < means.length; column++) {
                double difference = row[firstColumn + column] - means[column];
                deviations[column] += difference * difference;
            }
            rowCount++;
        }
        for (int column = 0; column < means.length; column++) {
            deviations[column] = Math.sqrt(deviations[column] / rowCount);
        }
        return deviations;
    }

    /**
     * Replaces every value below a limit, so that a constant column is divided by a harmless number
     * instead of by zero when it is standardised.
     *
     * @param values      the values, changed in place
     * @param limit       values below this limit are replaced
     * @param replacement the value that is used instead
     */
    public static void replaceBelow(double[] values, double limit, double replacement) {
        for (int index = 0; index < values.length; index++) {
            if (values[index] < limit) {
                values[index] = replacement;
            }
        }
    }

    private static double sumOfSquaredDeviations(double[] values, double mean) {
        double sum = 0.0;
        for (double value : values) {
            sum += (value - mean) * (value - mean);
        }
        return sum;
    }
}
