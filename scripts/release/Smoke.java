package io.github.rapidai.rapiddoc;

import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

public class Smoke {
  public static void main(String[] args) throws Exception {
    Map<String, Object> reports = new LinkedHashMap<>();
    Map<String, Object> table = new HashMap<>();
    table.put("use_word_box", false);
    try (RapidDoc engine =
        RapidDoc.builder()
            .layoutModel("docvortex_rule")
            .formulaEnabled(false)
            .tableEnabled(false)
            .option("table_config", table)
            .allowDownload(false)
            .progressLogging(false)
            .build()) {
      for (String name :
          Arrays.asList(
              "pdf_native.pdf",
              "synthetic.doc",
              "synthetic.docx",
              "synthetic.xls",
              "synthetic.xlsx",
              "synthetic.ppt",
              "synthetic.pptx")) {
        DocumentResult result =
            engine.parse(
                Paths.get(System.getenv("RAPIDDOC_TEST_ROOT"), "tests/migration/inputs", name),
                ParseOptions.builder().imageDirectory("figures").build());
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("markdown", result.getMarkdownValue());
        fields.put("middle_json", result.getMiddleJson());
        fields.put("content_list_json", result.getContentListJson());
        Map<String, String> images = new TreeMap<>();
        for (Map.Entry<String, byte[]> image : result.getImages().entrySet()) {
          StringBuilder digest = new StringBuilder();
          for (byte b : MessageDigest.getInstance("SHA-256").digest(image.getValue()))
            digest.append(String.format("%02x", b & 255));
          images.put(image.getKey(), digest.toString());
        }
        fields.put("images", images);
        reports.put(name, fields);
      }
    }
    RapidDoc.JSON.writeValue(Paths.get(System.getenv("RAPIDDOC_SMOKE_REPORT")).toFile(), reports);
  }
}
