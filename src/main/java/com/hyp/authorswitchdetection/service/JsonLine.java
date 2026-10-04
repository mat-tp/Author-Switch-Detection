package com.hyp.authorswitchdetection.service;

/**
 * Reading and writing single JSON object lines by hand — the same
 * indexOf-based approach DatasetLoader already uses to pull the "changes"
 * array out of a truth-problem-*.json file. Good enough for the small,
 * flat objects UserStore / ModelMetricsStore deal with, and means the app
 * doesn't need a JSON library just to save a handful of users and training
 * runs to disk.
 */
public final class JsonLine {

    private JsonLine() {
        // utility class, not meant to be instantiated
    }

    // "key":"value" -> value (or null if the key isn't present, or the value is null)
    public static String extractString(String json, String key) {

        String marker = "\"" + key + "\":";
        int start = json.indexOf(marker);
        if (start == -1) {
            return null;
        }
        start += marker.length();

        if (json.startsWith("null", start)) {
            return null;
        }

        // skip the opening quote
        start = json.indexOf("\"", start) + 1;
        int end = start;
        while (end < json.length() && !(json.charAt(end) == '"' && json.charAt(end - 1) != '\\')) {
            end++;
        }

        return unescape(json.substring(start, end));
    }

    // "key":123 -> 123 (or null if the key isn't present, or the value is null)
    public static Long extractLong(String json, String key) {

        String raw = extractNumber(json, key);
        return raw == null ? null : Long.parseLong(raw);
    }

    public static Integer extractInteger(String json, String key) {

        String raw = extractNumber(json, key);
        return raw == null ? null : Integer.parseInt(raw);
    }

    public static Double extractDouble(String json, String key) {

        String raw = extractNumber(json, key);
        return raw == null ? null : Double.parseDouble(raw);
    }

    public static boolean extractBoolean(String json, String key, boolean defaultValue) {

        String marker = "\"" + key + "\":";
        int start = json.indexOf(marker);
        if (start == -1) {
            return defaultValue;
        }
        start += marker.length();

        return json.startsWith("true", start);
    }

    private static String extractNumber(String json, String key) {

        String marker = "\"" + key + "\":";
        int start = json.indexOf(marker);
        if (start == -1) {
            return null;
        }
        start += marker.length();

        if (json.startsWith("null", start)) {
            return null;
        }

        int end = start;
        while (end < json.length() && "-.0123456789".indexOf(json.charAt(end)) >= 0) {
            end++;
        }

        return json.substring(start, end);
    }

    public static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String unescape(String s) {
        return s.replace("\\\"", "\"").replace("\\\\", "\\");
    }
}
