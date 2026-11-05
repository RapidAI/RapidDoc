package io.github.rapidai.rapiddoc;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

/** Native output. Asset byte arrays transfer once from JNI; getters do not clone them. */
public final class DocumentResult {
  private final JsonNode value;
  private final Map<String, byte[]> images, drawings;
  private final String imageDirectory;
  private final String sourceName, parseMethod;

  DocumentResult(
      NativeBindings.Result result, String imageDirectory, String source, String method) {
    this.imageDirectory = imageDirectory;
    try {
      value = RapidDoc.JSON.readTree(result.json);
    } catch (IOException e) {
      throw new IllegalStateException("Invalid native result", e);
    }
    images = Collections.unmodifiableMap(result.images);
    drawings = Collections.unmodifiableMap(result.drawings);
    source = value.path("metadata").path("source_name").asText(source);
    int dot = source.lastIndexOf('.');
    sourceName = dot > 0 ? source.substring(0, dot) : source;
    parseMethod =
        "office".equals(value.path("metadata").path("format").asText()) ? "office" : method;
  }

  /** For content-list modes use getMarkdownValue instead. */
  public String getMarkdown() {
    JsonNode v = value.get("markdown");
    if (!v.isTextual())
      throw new IllegalStateException(
          "This mode returns structured Markdown; use getMarkdownValue()");
    return v.asText();
  }

  public JsonNode getMarkdownValue() {
    return value.get("markdown");
  }

  public JsonNode getMiddleJson() {
    return value.get("middle_json");
  }

  public JsonNode getContentListJson() {
    return value.get("content_list_json");
  }

  public JsonNode getMetadata() {
    return value.get("metadata");
  }

  public JsonNode getDiagnostics() {
    return value.get("diagnostics");
  }

  public int getSourcePageCount() {
    return value.path("source_page_count").asInt();
  }

  public int getPageCount() {
    return getMiddleJson().path("pdf_info").size();
  }

  public Map<String, byte[]> getImages() {
    return images;
  }

  public Map<String, byte[]> getDrawings() {
    return drawings;
  }

  public JsonNode toJson() {
    return value;
  }

  public String getSourceName() {
    return sourceName;
  }

  public String getParseMethod() {
    return parseMethod;
  }

  /** Resolve root/source-name/method, matching the Python output layout. */
  public Path getOutputDirectory(Path directory) {
    return directory.toAbsolutePath().normalize().resolve(sourceName).resolve(parseMethod);
  }

  /** Writes under root/source-name/method, with original filename prefixes. */
  public void writeTo(Path directory) throws IOException {
    Path root = getOutputDirectory(directory);
    Files.createDirectories(root);
    JsonNode md = getMarkdownValue();
    write(
        root,
        sourceName + (md.isTextual() ? ".md" : "_markdown.json"),
        (md.isTextual() ? md.asText() : md.toPrettyString()).getBytes(StandardCharsets.UTF_8));
    write(
        root,
        sourceName + "_middle.json",
        getMiddleJson().toPrettyString().getBytes(StandardCharsets.UTF_8));
    write(
        root,
        sourceName + "_content_list.json",
        getContentListJson().toPrettyString().getBytes(StandardCharsets.UTF_8));
    for (Map.Entry<String, byte[]> e : images.entrySet())
      write(root, imageDirectory + "/" + e.getKey(), e.getValue());
    for (Map.Entry<String, byte[]> e : drawings.entrySet())
      write(root, sourceName + "_" + e.getKey() + ".pdf", e.getValue());
  }

  private static void write(Path root, String name, byte[] data) throws IOException {
    Path target = root.resolve(name).normalize();
    if (!target.startsWith(root) || target.equals(root))
      throw new IOException("Asset path escapes output directory: " + name);
    Files.createDirectories(target.getParent());
    Files.write(target, data);
  }
}
