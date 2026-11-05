package io.github.rapidai.rapiddoc;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class InputOverloadsTest {
    private final Path inputs =
            Paths.get(System.getProperty("rapiddoc.test.root"), "tests/migration/inputs")
                    .toAbsolutePath();

    private RapidDoc engine() {
        return RapidDoc.builder()
                .layoutModel("docvortex_rule")
                .formulaEnabled(false)
                .tableEnabled(false)
                .allowDownload(false)
                .progressLogging(false)
                .build();
    }

    @Test
    void pathBytesStreamsAndListsPreserveOutputOptionsAndOrder() throws Exception {
        ParseOptions options = ParseOptions.builder().imageDirectory("figures").build();
        Path docx = inputs.resolve("synthetic.docx"), pdf = inputs.resolve("pdf_native.pdf");
        List<ProgressEvent> events = new ArrayList<>();
        try (RapidDoc engine = engine()) {
            DocumentResult expected = engine.parse(docx, options);
            byte[] bytes = Files.readAllBytes(docx);
            DocumentResult[] variants = {
                engine.parse(docx.toString(), options),
                engine.parse(docx.toString(), options, events::add),
                engine.parse(bytes, options),
                engine.parse(new ByteArrayInputStream(bytes), options),
                engine.parse(new ByteArrayInputStream(bytes), options, events::add)
            };
            for (DocumentResult result : variants) {
                assertEquals(expected.getMarkdownValue(), result.getMarkdownValue());
                assertEquals(expected.getMiddleJson(), result.getMiddleJson());
                assertEquals(expected.getContentListJson(), result.getContentListJson());
                assertEquals(expected.getImages().keySet(), result.getImages().keySet());
                for (String image : expected.getImages().keySet())
                    assertArrayEquals(
                            expected.getImages().get(image), result.getImages().get(image));
            }
            assertEquals(
                    engine.parse(bytes).getMarkdown(),
                    engine.parse(new ByteArrayInputStream(bytes)).getMarkdown());
            List<Path> paths = Arrays.asList(pdf, docx);
            List<DocumentResult> results = engine.parse(paths, options, events::add);
            assertEquals(engine.parse(pdf, options).getMarkdown(), results.get(0).getMarkdown());
            assertEquals(expected.getMarkdown(), results.get(1).getMarkdown());
            assertTrue(events.stream().anyMatch(e -> e.getStage().equals("document_opened")));
            assertEquals(
                    results.get(0).getMarkdown(),
                    engine.parseAll(paths, options, null).get(0).getMarkdown());
            assertEquals(
                    engine.parse(paths).get(1).getMarkdown(),
                    engine.parseAll(paths).get(1).getMarkdown());
            assertEquals(
                    engine.parse(paths, options).get(1).getMarkdown(),
                    engine.parseAll(paths, options).get(1).getMarkdown());
        }
    }

    @Test
    void streamOwnershipPositionAndReadFailuresArePreserved() throws Exception {
        byte[] data = Files.readAllBytes(inputs.resolve("synthetic.docx"));
        byte[] prefixed = new byte[data.length + 3];
        System.arraycopy(data, 0, prefixed, 3, data.length);
        final boolean[] closed = {false};
        ByteArrayInputStream stream =
                new ByteArrayInputStream(prefixed) {
                    public void close() {
                        closed[0] = true;
                    }
                };
        assertEquals(3, stream.skip(3));
        IOException readFailure = new IOException("input failed");
        InputStream broken =
                new InputStream() {
                    public int read() throws IOException {
                        throw readFailure;
                    }

                    public void close() {
                        closed[0] = true;
                    }
                };
        try (RapidDoc engine = engine()) {
            assertEquals(engine.parse(data).getMarkdown(), engine.parse(stream).getMarkdown());
            assertFalse(closed[0]);
            assertSame(readFailure, assertThrows(IOException.class, () -> engine.parse(broken)));
            assertFalse(closed[0]);
            RuntimeException cancelled = new RuntimeException("cancelled");
            try (InputStream pdf = Files.newInputStream(inputs.resolve("pdf_native.pdf"))) {
                assertSame(
                        cancelled,
                        assertThrows(
                                RuntimeException.class,
                                () ->
                                        engine.parse(
                                                pdf,
                                                ParseOptions.DEFAULT,
                                                e -> {
                                                    throw cancelled;
                                                })));
            }
        }
    }
}
