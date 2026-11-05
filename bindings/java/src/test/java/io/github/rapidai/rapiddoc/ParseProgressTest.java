package io.github.rapidai.rapiddoc;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ParseProgressTest {
    private String render(boolean bar) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        ParseProgress view =
                new ParseProgress("test.pdf", new PrintStream(bytes, true, "UTF-8"), bar);
        view.onProgress(new ProgressEvent("document_opened", -1, -1, 14, -1, ""));
        view.onProgress(new ProgressEvent("render", 0, -1, -1, -1, ""));
        view.onProgress(new ProgressEvent("layout", 0, -1, -1, -1, ""));
        view.onProgress(new ProgressEvent("layout", -1, 0, 14, 0, ""));
        view.onProgress(new ProgressEvent("layout", -1, 14, 14, 11, ""));
        view.onProgress(new ProgressEvent("table", -1, 0, 0, 0, ""));
        view.onProgress(new ProgressEvent("finalize", -1, -1, -1, -1, ""));
        view.completed();
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    @Test
    void plainOutputHasNoControlSequencesOrInternalCounters() throws Exception {
        String output = render(false).replace("\r\n", "\n");
        assertTrue(output.contains("14 pages"));
        assertTrue(output.contains("100% |====================| 14/14 [processing 11.00s]"));
        assertTrue(output.contains("Post-processing"));
        assertTrue(output.contains("completed in"));
        assertFalse(output.contains("-1") || output.contains("0/0"));
        assertFalse(output.contains("\r") || output.contains("\033"));
        assertFalse(output.contains("ParseProgress onProgress"));
    }

    @Test
    void barUpdatesReplaceOneLineWithoutAnsiOrLoggerPrefixes() throws Exception {
        String output = render(true).replace("\r\n", "\n");
        assertTrue(output.contains("\rtest.pdf: Layout Predict:   0%"));
        assertTrue(output.contains("\rtest.pdf: Layout Predict: 100%"));
        // Start and finish share one physical line until the stage completes.
        String stage =
                output.substring(
                        output.indexOf("\rtest.pdf: Layout"),
                        output.indexOf("[processing 11.00s]"));
        assertFalse(stage.contains("\n"));
        assertFalse(output.contains("\033"));
    }

    @Test
    void outputModeCanBeForcedForRedirectedIdeRuns() {
        String previous = System.getProperty("rapid_doc.progress.mode");
        try {
            System.setProperty("rapid_doc.progress.mode", "bar");
            assertTrue(ParseProgress.useProgressBar());
            System.setProperty("rapid_doc.progress.mode", "plain");
            assertFalse(ParseProgress.useProgressBar());
        } finally {
            if (previous == null) System.clearProperty("rapid_doc.progress.mode");
            else System.setProperty("rapid_doc.progress.mode", previous);
        }
    }

    @Test
    void completedTableIsPrintedOnceDespiteInterleavedPageEventsAndResetsForNextBatch()
            throws Exception {
        ByteArrayOutputStream status = new ByteArrayOutputStream();
        ByteArrayOutputStream bars = new ByteArrayOutputStream();
        ParseProgress view =
                new ParseProgress(
                        "tables.pdf",
                        new PrintStream(status, true, "UTF-8"),
                        new PrintStream(bars, true, "UTF-8"),
                        true);
        view.onProgress(new ProgressEvent("table", -1, 1, 1, .16, ""));
        for (int page = 1; page <= 10; page++) {
            view.onProgress(new ProgressEvent("assemble", -1, page, 14, page * .01, ""));
            view.onProgress(new ProgressEvent("table", -1, 1, 1, .16 + page * .001, ""));
        }
        String first = new String(bars.toByteArray(), StandardCharsets.UTF_8);
        assertEquals(1, first.split("Table Predict: 100%", -1).length - 1);
        view.onProgress(new ProgressEvent("batch", -1, -1, -1, -1, ""));
        view.onProgress(new ProgressEvent("table", -1, 1, 1, .2, ""));
        view.completed();
        String all = new String(bars.toByteArray(), StandardCharsets.UTF_8);
        assertEquals(2, all.split("Table Predict: 100%", -1).length - 1);
        assertFalse(new String(status.toByteArray(), StandardCharsets.UTF_8).contains("Predict"));
        assertTrue(
                new String(status.toByteArray(), StandardCharsets.UTF_8).contains("completed in"));
        assertFalse(all.contains("completed in"));
    }
}
