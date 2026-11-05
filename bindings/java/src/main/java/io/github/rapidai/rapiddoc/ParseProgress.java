package io.github.rapidai.rapiddoc;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;

/** Human-readable default view; raw internal events remain available to listeners. */
final class ParseProgress implements ProgressListener {
    private final String source;
    private final long started = System.nanoTime();
    private final Map<String, Long> updates = new HashMap<>();
    private final Map<String, long[]> counters = new HashMap<>();
    private final boolean terminal;
    private final PrintStream output;
    private final PrintStream status;
    private boolean line;
    private int lineWidth;
    private String activeStage;

    static boolean useProgressBar() {
        String mode = System.getProperty("rapid_doc.progress.mode", "auto");
        if (mode.equals("bar")) return true;
        if (mode.equals("plain")) return false;
        if (!mode.equals("auto"))
            throw new IllegalArgumentException(
                    "rapid_doc.progress.mode must be auto, bar or plain");
        if (System.console() != null) return true;
        // IDEA's Run console supports carriage-return replacement, even though
        // java.io.Console is unavailable. No ANSI cursor-control sequences needed.
        for (String argument : ManagementFactory.getRuntimeMXBean().getInputArguments())
            if (argument.startsWith("-javaagent:") && argument.contains("idea_rt.jar")) return true;
        return false;
    }

    ParseProgress(String source) {
        this(source, System.out, System.err, useProgressBar());
    }

    ParseProgress(String source, PrintStream output, boolean terminal) {
        this(source, output, output, terminal);
    }

    ParseProgress(String source, PrintStream status, PrintStream output, boolean terminal) {
        this.source = source;
        this.output = output;
        this.status = status;
        this.terminal = terminal;
        status.println("Parsing: " + source);
    }

    public void onProgress(ProgressEvent event) {
        String stage = event.getStage();
        if (stage.equals("batch")) {
            clear();
            counters.clear();
            updates.clear();
            activeStage = null;
            return;
        }
        if (stage.startsWith("download_")) {
            clear();
            return;
        }
        if (stage.equals("document_opened")) {
            clear();
            status.println(source + ": " + event.getTotal() + " pages");
            return;
        }
        if (stage.equals("finalize")) {
            clear();
            status.println(source + ": Post-processing...");
            return;
        }
        if (stage.equals("model_resolved")) {
            // Model loading may take time; keep the current stage visible.
            return;
        }
        String label = label(stage);
        if (label == null || event.getCompleted() < 0 || event.getTotal() <= 0) return;
        long[] previous = counters.get(stage);
        // Table/seal counts are emitted again after pages without these regions.
        // Preserve native callbacks, but do not print the same progress repeatedly.
        if (previous != null
                && previous[0] == event.getCompleted()
                && previous[1] == event.getTotal()) return;
        long now = System.nanoTime();
        Long last = updates.get(stage);
        if (event.getCompleted() != 0
                && event.getCompleted() != event.getTotal()
                && last != null
                && now - last < 1_000_000_000L) return;
        updates.put(stage, now);
        counters.put(stage, new long[] {event.getCompleted(), event.getTotal()});
        if (!stage.equals(activeStage)) {
            clear();
            activeStage = stage;
        }
        int filled =
                (int) Math.min(20, Math.max(0, event.getCompleted() * 20.0 / event.getTotal()));
        StringBuilder bar = new StringBuilder(20);
        for (int index = 0; index < 20; index++) bar.append(index < filled ? '=' : ' ');
        String time =
                event.getElapsedSeconds() >= 0
                        ? String.format(
                                Locale.ROOT, " [processing %.2fs]", event.getElapsedSeconds())
                        : "";
        String message =
                String.format(
                        Locale.ROOT,
                        "%s: %s: %3.0f%% |%s| %d/%d%s",
                        source,
                        label,
                        event.getCompleted() * 100.0 / event.getTotal(),
                        bar,
                        event.getCompleted(),
                        event.getTotal(),
                        time);
        if (terminal) {
            output.print("\r" + message);
            for (int index = message.length(); index < lineWidth; index++) output.print(' ');
            output.flush();
            lineWidth = message.length();
            line = true;
            if (event.getCompleted() == event.getTotal()) clear();
        } else {
            output.println(message);
        }
    }

    private static String label(String stage) {
        switch (stage) {
            case "orientation":
                return "Orientation Predict";
            case "layout":
                return "Layout Predict";
            case "formula":
                return "Formula Predict";
            case "ocr_detection":
                return "PDF/OCR-det Predict";
            case "ocr_recognition":
                return "OCR-rec Predict";
            case "table":
                return "Table Predict";
            case "seal":
                return "Seal Predict";
            case "assemble":
                return "Processing pages";
            default:
                return null;
        }
    }

    void clear() {
        if (line) {
            output.println();
            lineWidth = 0;
            line = false;
        }
    }

    void completed() {
        clear();
        status.println(
                String.format(
                        Locale.ROOT,
                        "%s: completed in %.2fs",
                        source,
                        (System.nanoTime() - started) / 1_000_000_000.0));
    }

    void failed(RuntimeException error) {
        clear();
        status.println(source + ": failed: " + error.getMessage());
    }
}
