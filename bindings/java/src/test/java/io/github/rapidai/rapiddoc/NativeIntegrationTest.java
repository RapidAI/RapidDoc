package io.github.rapidai.rapiddoc;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NativeIntegrationTest {
  @Test
  void httpStringsRedirectsAndListsUseTheNativeInputLoader() throws Exception {
    byte[] data = Files.readAllBytes(inputs.resolve("synthetic.docx"));
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext(
        "/",
        exchange -> {
          try {
            String path = exchange.getRequestURI().getPath();
            if (path.equals("/redirect")) {
              exchange.getResponseHeaders().add("Location", "/data");
              exchange.sendResponseHeaders(302, -1);
            } else if (path.equals("/missing")) {
              exchange.sendResponseHeaders(404, -1);
            } else {
              exchange.sendResponseHeaders(200, data.length);
              exchange.getResponseBody().write(data);
            }
          } finally {
            exchange.close();
          }
        });
    server.start();
    try (RapidDoc engine = engine()) {
      String base = "http://127.0.0.1:" + server.getAddress().getPort();
      DocumentResult local = engine.parse(data);
      DocumentResult remote = engine.parse(base + "/redirect?filename=report.docx");
      assertEquals(local.getMarkdown(), remote.getMarkdown());
      assertEquals(local.getMiddleJson(), remote.getMiddleJson());
      remote.writeTo(temp);
      assertTrue(Files.isRegularFile(temp.resolve("report/office/report.md")));
      assertEquals(
          2, engine.parse(Arrays.asList(base + "/data", inputs.resolve("synthetic.docx"))).size());
      assertTrue(
          assertThrows(RapidDocException.class, () -> engine.parse(base + "/missing"))
              .getMessage()
              .contains("404"));
    } finally {
      server.stop(0);
    }
  }

  @Test
  void sameEngineAllowsOverlappingPdfAndOfficeCalls() throws Exception {
    try (RapidDoc engine = engine()) {
      String expected = engine.parse(inputs.resolve("pdf_native.pdf")).getMarkdown();
      ExecutorService pool = Executors.newFixedThreadPool(4);
      CyclicBarrier both = new CyclicBarrier(2);
      ProgressListener listener =
          event -> {
            if (event.getStage().equals("document_opened")) {
              try {
                both.await(20, TimeUnit.SECONDS);
              } catch (Exception error) {
                throw new RuntimeException(error);
              }
            }
          };
      try {
        Future<DocumentResult> first =
            pool.submit(
                () ->
                    engine.parse(inputs.resolve("pdf_native.pdf"), ParseOptions.DEFAULT, listener));
        Future<DocumentResult> second =
            pool.submit(
                () ->
                    engine.parse(inputs.resolve("pdf_native.pdf"), ParseOptions.DEFAULT, listener));
        Future<DocumentResult> office =
            pool.submit(() -> engine.parse(inputs.resolve("synthetic.docx")));
        assertFalse(office.get(30, TimeUnit.SECONDS).getMarkdown().isEmpty());
        assertEquals(expected, first.get(30, TimeUnit.SECONDS).getMarkdown());
        assertEquals(expected, second.get(30, TimeUnit.SECONDS).getMarkdown());
      } finally {
        pool.shutdownNow();
      }
    }
  }

  final Path root =
      Paths.get(System.getProperty("rapiddoc.test.root")).toAbsolutePath().normalize();
  final Path inputs = root.resolve("tests/migration/inputs");
  @TempDir Path temp;

  RapidDoc engine() {
    return RapidDoc.builder()
        .layoutModel("docvortex_rule")
        .formulaEnabled(false)
        .tableEnabled(false)
        .allowDownload(false)
        .build();
  }

  @Test
  void sevenFormatsMatchPythonReference() throws Exception {
    JsonNode before =
        RapidDoc.JSON.readTree(root.resolve("tests/fixtures/java_sdk_expected.json").toFile());
    try (RapidDoc engine = engine()) {
      for (String name :
          Arrays.asList(
              "synthetic.doc",
              "synthetic.docx",
              "synthetic.xls",
              "synthetic.xlsx",
              "synthetic.ppt",
              "synthetic.pptx",
              "pdf_native.pdf")) {
        DocumentResult result =
            engine.parse(
                inputs.resolve(name), ParseOptions.builder().imageDirectory("figures").build());
        JsonNode expected = before.get(name);
        assertNotNull(expected, name);
        assertEquals(expected.get("markdown"), result.getMarkdownValue(), name);
        assertEquals(expected.get("middle_json"), result.getMiddleJson(), name);
        assertEquals(expected.get("content_list_json"), result.getContentListJson(), name);
        assertEquals(expected.path("images").size(), result.getImages().size(), name);
        for (Map.Entry<String, byte[]> image : result.getImages().entrySet()) {
          StringBuilder hash = new StringBuilder();
          for (byte b : MessageDigest.getInstance("SHA-256").digest(image.getValue()))
            hash.append(String.format("%02x", b & 255));
          assertEquals(
              expected.path("images").path(image.getKey()).asText(),
              hash.toString(),
              name + " " + image.getKey());
        }
        assertTrue(result.getPageCount() > 0, name);
      }
    }
  }

  @Test
  void byteInputUnicodePathWritingAndClose() throws Exception {
    Path input = temp.resolve("中文😀.docx");
    Files.copy(inputs.resolve("synthetic.docx"), input);
    RapidDoc engine = engine();
    DocumentResult fromPath = engine.parse(input);
    DocumentResult fromBytes = engine.parse(Files.readAllBytes(input));
    assertEquals(fromPath.getMarkdown(), fromBytes.getMarkdown());
    fromPath.writeTo(temp.resolve("output"));
    assertTrue(Files.exists(temp.resolve("output/中文😀/office/中文😀.md")));
    fromBytes.writeTo(temp.resolve("output"));
    assertTrue(Files.exists(temp.resolve("output/document/office/document.md")));
    assertTrue(engine.getLoadedModels().isEmpty());
    engine.close();
    engine.close();
    assertThrows(IllegalStateException.class, () -> engine.parse(input));
    assertThrows(
        RapidDocException.class, () -> NativeBindings.parsePath(-1, input.toString(), "{}", null));
  }

  @Test
  void callbackExceptionsAndReentryRemainSafe() {
    try (RapidDoc engine = engine()) {
      RuntimeException cancel = new RuntimeException("cancel");
      assertSame(
          cancel,
          assertThrows(
              RuntimeException.class,
              () ->
                  engine.parse(
                      inputs.resolve("pdf_native.pdf"),
                      ParseOptions.DEFAULT,
                      event -> {
                        throw cancel;
                      })));
      List<String> events = new ArrayList<>();
      DocumentResult result =
          engine.parse(
              inputs.resolve("pdf_native.pdf"),
              ParseOptions.DEFAULT,
              event -> {
                events.add(event.getStage());
                assertThrows(
                    RapidDocException.class, () -> engine.parse(inputs.resolve("synthetic.docx")));
              });
      assertFalse(events.isEmpty());
      assertFalse(result.getMarkdown().isEmpty());
    }
  }

  @Test
  void concurrentCloseDoesNotFreeInFlightEngine() throws Exception {
    RapidDoc engine = engine();
    ExecutorService executor = Executors.newSingleThreadExecutor();
    CountDownLatch entered = new CountDownLatch(1), resume = new CountDownLatch(1);
    try {
      Future<DocumentResult> pending =
          executor.submit(
              () ->
                  engine.parse(
                      inputs.resolve("pdf_native.pdf"),
                      ParseOptions.DEFAULT,
                      event -> {
                        entered.countDown();
                        try {
                          if (!resume.await(15, TimeUnit.SECONDS))
                            throw new IllegalStateException("timeout");
                        } catch (InterruptedException e) {
                          throw new RuntimeException(e);
                        }
                      }));
      assertTrue(entered.await(15, TimeUnit.SECONDS));
      engine.close();
      resume.countDown();
      assertFalse(pending.get(30, TimeUnit.SECONDS).getMarkdown().isEmpty());
      assertThrows(
          IllegalStateException.class, () -> engine.parse(inputs.resolve("synthetic.docx")));
    } finally {
      resume.countDown();
      engine.close();
      executor.shutdownNow();
    }
  }

  @Test
  void invalidConfigurationReportsNativeCode() {
    RapidDocException error =
        assertThrows(
            RapidDocException.class,
            () -> RapidDoc.builder().option("unknown_config", true).build());
    assertEquals("invalid_input", error.getCode());
  }

  @Test
  void drawingAndImageWriterUseNativeOutput() throws Exception {
    try (RapidDoc engine = engine()) {
      DocumentResult result =
          engine.parse(
              inputs.resolve("pdf_native.pdf"),
              ParseOptions.builder()
                  .imageDirectory("figures")
                  .drawLayout(true)
                  .drawSpans(true)
                  .drawLineOrder(true)
                  .build());
      assertEquals(3, result.getDrawings().size());
      Path output = temp.resolve("drawings");
      result.writeTo(output);
      output = result.getOutputDirectory(output);
      assertTrue(Files.exists(output.resolve("pdf_native.md")));
      assertTrue(Files.exists(output.resolve("pdf_native_middle.json")));
      for (Map.Entry<String, byte[]> asset : result.getImages().entrySet())
        assertArrayEquals(
            asset.getValue(),
            Files.readAllBytes(output.resolve("figures").resolve(asset.getKey())));
      for (Map.Entry<String, byte[]> drawing : result.getDrawings().entrySet()) {
        assertTrue(
            new String(drawing.getValue(), 0, 4, java.nio.charset.StandardCharsets.US_ASCII)
                .startsWith("%PDF"));
        assertArrayEquals(
            drawing.getValue(),
            Files.readAllBytes(output.resolve("pdf_native_" + drawing.getKey() + ".pdf")));
      }
    }
  }

  @Test
  void scannedPdfRunsCpuOcrAndReusesSessions() {
    Path models = root.resolve("rapid_doc/resources");
    org.junit.jupiter.api.Assumptions.assumeTrue(
        Files.isRegularFile(models.resolve("PP-OCRv6_det_small.onnx"))
            && Files.isRegularFile(models.resolve("PP-OCRv6_rec_small.onnx")),
        "Local OCR model fixture is not installed");
    Map<String, Object> ocr = new HashMap<>();
    ocr.put("seal_enable", false);
    ocr.put("Det.model_path", models.resolve("PP-OCRv6_det_small.onnx").toString());
    ocr.put("Rec.model_path", models.resolve("PP-OCRv6_rec_small.onnx").toString());
    try (RapidDoc engine =
        RapidDoc.builder()
            .layoutModel("docvortex_rule")
            .formulaEnabled(false)
            .tableEnabled(false)
            .ocrConfig(ocr)
            .allowDownload(false)
            .build()) {
      engine.warmup();
      assertEquals(Arrays.asList("detector", "recognizer"), engine.getLoadedModels());
      DocumentResult result = engine.parse(inputs.resolve("pdf_scan.pdf"));
      assertFalse(result.getMarkdown().trim().isEmpty());
      assertEquals(Arrays.asList("detector", "recognizer"), engine.getLoadedModels());
    }
  }

  @Test
  void cpuOcrSessionsCanBeCreatedOnDifferentThreads() throws Exception {
    Path models = root.resolve("rapid_doc/resources");
    org.junit.jupiter.api.Assumptions.assumeTrue(
        Files.isRegularFile(models.resolve("PP-OCRv6_det_small.onnx"))
            && Files.isRegularFile(models.resolve("PP-OCRv6_rec_small.onnx")));
    ExecutorService pool = Executors.newFixedThreadPool(2);
    try {
      Callable<String> parse =
          () -> {
            Map<String, Object> ocr = new HashMap<>();
            ocr.put("seal_enable", false);
            ocr.put("Det.model_path", models.resolve("PP-OCRv6_det_small.onnx").toString());
            ocr.put("Rec.model_path", models.resolve("PP-OCRv6_rec_small.onnx").toString());
            ocr.put("EngineConfig.onnxruntime.intra_op_num_threads", 1);
            try (RapidDoc engine =
                RapidDoc.builder()
                    .layoutModel("docvortex_rule")
                    .formulaEnabled(false)
                    .tableEnabled(false)
                    .ocrConfig(ocr)
                    .allowDownload(false)
                    .build()) {
              engine.warmup();
              return engine.parse(inputs.resolve("pdf_scan.pdf")).getMarkdown();
            }
          };
      Future<String> first = pool.submit(parse);
      Future<String> second = pool.submit(parse);
      String expected = first.get(120, TimeUnit.SECONDS);
      assertFalse(expected.trim().isEmpty());
      assertEquals(expected, second.get(120, TimeUnit.SECONDS));
      // Exercise new sessions after the first engines have closed.
      assertEquals(expected, pool.submit(parse).get(120, TimeUnit.SECONDS));
    } finally {
      pool.shutdownNow();
    }
  }

  @Test
  void downloadStartsBeforeNetworkAndCanBeCancelledWithoutLeavingPartial() throws Exception {
    Path directory = temp.resolve("fresh-models");
    RuntimeException cancellation = new RuntimeException("stop before HTTP request");
    List<String> events = new ArrayList<>();
    try (RapidDoc engine =
        RapidDoc.builder()
            .layoutModel("docvortex_rule")
            .formulaEnabled(false)
            .tableEnabled(false)
            .modelDirectory(directory)
            .build()) {
      assertSame(
          cancellation,
          assertThrows(
              RuntimeException.class,
              () ->
                  engine.warmup(
                      ParseOptions.DEFAULT,
                      event -> {
                        events.add(event.getStage());
                        if (event.getStage().equals("download_started")) {
                          assertTrue(
                              event
                                  .getDetail()
                                  .contains(directory.toString().replace("\\", "\\\\")));
                          throw cancellation;
                        }
                      })));
      assertTrue(events.contains("download_started"));
      try (java.util.stream.Stream<Path> paths = Files.list(directory)) {
        assertEquals(0, paths.count());
      }
    }
  }
}
