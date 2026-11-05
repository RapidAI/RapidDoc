package io.github.rapidai.rapiddoc;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/** A reusable native engine. Close it to release model sessions promptly. */
public final class RapidDoc implements AutoCloseable {
  static final ObjectMapper JSON = new ObjectMapper();
  private static final NativeCleaner CLEANER = new NativeCleaner();
  private final State state;
  private final NativeCleaner.Registration cleanable;
  private final Path outputDirectory;
  private final boolean progressLogging;
  private final String parseMethod;

  private static final class State implements Runnable {
    final AtomicLong handle;

    State(long handle) {
      this.handle = new AtomicLong(handle);
    }

    long get() {
      long v = handle.get();
      if (v == 0) throw new IllegalStateException("RapidDoc is closed");
      return v;
    }

    public void run() {
      long v = handle.getAndSet(0);
      if (v != 0) NativeBindings.close(v);
    }
  }

  private RapidDoc(Builder builder) {
    Path bundle = NativeLoader.load();
    ObjectNode resources = builder.resources.deepCopy();
    if (!resources.hasNonNull("ort_library")
        && System.getenv("RAPIDDOC_ONNXRUNTIME_LIBRARY") == null) {
      Path ort = NativeLoader.ortDirectory();
      if (ort != null) resources.withArray("bundle_dirs").add(ort.toString());
    }
    resources.withArray("bundle_dirs").add(bundle.toString());
    state = new State(NativeBindings.create(builder.config.toString(), resources.toString()));
    cleanable = CLEANER.register(this, state);
    outputDirectory = builder.outputDirectory;
    progressLogging = builder.progressLogging;
    parseMethod = builder.config.path("parse_method").asText("auto");
    if (builder.preload) {
      try {
        warmup();
      } catch (RuntimeException | Error e) {
        close();
        throw e;
      }
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  public DocumentResult parse(String path) {
    return parse(path, ParseOptions.DEFAULT, null);
  }

  public DocumentResult parse(String path, ParseOptions options) {
    return parse(path, options, null);
  }

  public DocumentResult parse(String path, ParseOptions options, ProgressListener listener) {
    Objects.requireNonNull(path);
    if (!path.regionMatches(true, 0, "http://", 0, 7)
        && !path.regionMatches(true, 0, "https://", 0, 8))
      return parse(Paths.get(path), options, listener);
    Objects.requireNonNull(options);
    ParseProgress view =
        listener == null && progressLogging ? new ParseProgress("URL document") : null;
    try {
      DocumentResult result =
          parseSource(path, "document", options, view == null ? listener : view);
      if (view != null) view.completed();
      return result;
    } catch (RuntimeException error) {
      if (view != null) view.failed(error);
      throw error;
    }
  }

  public DocumentResult parse(Path path) {
    return parse(path, ParseOptions.DEFAULT, null);
  }

  public DocumentResult parse(Path path, ParseOptions options) {
    return parse(path, options, null);
  }

  public DocumentResult parse(Path path, ParseOptions options, ProgressListener listener) {
    Objects.requireNonNull(path);
    Objects.requireNonNull(options);
    ParseProgress view =
        listener == null && progressLogging
            ? new ParseProgress(path.getFileName().toString())
            : null;
    try {
      DocumentResult result = parsePath(path, options, view == null ? listener : view);
      if (view != null) view.completed();
      return result;
    } catch (RuntimeException error) {
      if (view != null) view.failed(error);
      throw error;
    }
  }

  private DocumentResult parsePath(Path path, ParseOptions options, ProgressListener listener) {
    return parseSource(
        path.toAbsolutePath().toString(), path.getFileName().toString(), options, listener);
  }

  private DocumentResult parseSource(
      String source, String name, ParseOptions options, ProgressListener listener) {
    DocumentResult result =
        new DocumentResult(
            NativeBindings.parsePath(
                state.get(), source, options.json(), DownloadProgress.wrap(listener)),
            options.imageDirectory(),
            name,
            options.parseMethod(parseMethod));
    if (outputDirectory != null) {
      try {
        result.writeTo(outputDirectory);
      } catch (IOException e) {
        throw new RapidDocException("output_io", e.toString());
      }
    }
    return result;
  }

  public DocumentResult parse(byte[] data) {
    return parse(data, ParseOptions.DEFAULT, null);
  }

  public DocumentResult parse(byte[] data, ParseOptions options) {
    return parse(data, options, null);
  }

  /** Reads from the current position without closing the caller-owned stream. */
  public DocumentResult parse(InputStream input) throws IOException {
    return parse(input, ParseOptions.DEFAULT, null);
  }

  public DocumentResult parse(InputStream input, ParseOptions options) throws IOException {
    return parse(input, options, null);
  }

  /** Materializes one document for the native byte-input API; does not seek or close. */
  public DocumentResult parse(InputStream input, ParseOptions options, ProgressListener listener)
      throws IOException {
    Objects.requireNonNull(input);
    Objects.requireNonNull(options);
    state.get();
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    byte[] buffer = new byte[65536];
    int size;
    while ((size = input.read(buffer)) != -1) {
      if (size == 0) {
        int value = input.read();
        if (value == -1) break;
        bytes.write(value);
      } else {
        bytes.write(buffer, 0, size);
      }
    }
    return parse(bytes.toByteArray(), options, listener);
  }

  public DocumentResult parse(byte[] data, ParseOptions options, ProgressListener listener) {
    Objects.requireNonNull(data);
    Objects.requireNonNull(options);
    ParseProgress view = listener == null && progressLogging ? new ParseProgress("bytes") : null;
    try {
      DocumentResult result =
          new DocumentResult(
              NativeBindings.parseBytes(
                  state.get(),
                  Objects.requireNonNull(data),
                  Objects.requireNonNull(options).json(),
                  DownloadProgress.wrap(view == null ? listener : view)),
              options.imageDirectory(),
              "document",
              options.parseMethod(parseMethod));
      if (view != null) view.completed();
      return result;
    } catch (RuntimeException error) {
      if (view != null) view.failed(error);
      throw error;
    }
  }

  public List<DocumentResult> parse(List<?> paths) {
    return parse(paths, ParseOptions.DEFAULT, null);
  }

  public List<DocumentResult> parse(List<?> paths, ParseOptions options) {
    return parse(paths, options, null);
  }

  /** Reuses sessions; paths are processed sequentially and results retain input order. */
  public List<DocumentResult> parse(
      List<?> paths, ParseOptions options, ProgressListener listener) {
    Objects.requireNonNull(paths);
    Objects.requireNonNull(options);
    for (Object path : paths) Objects.requireNonNull(path);
    List<DocumentResult> results = new ArrayList<>();
    for (Object path : paths)
      if (path instanceof Path) results.add(parse((Path) path, options, listener));
      else if (path instanceof String) results.add(parse((String) path, options, listener));
      else
        throw new IllegalArgumentException("Input list must contain paths or HTTP/HTTPS strings");
    return results;
  }

  /** Compatibility alias for parse(List). */
  public List<DocumentResult> parseAll(List<?> paths) {
    return parse(paths);
  }

  public List<DocumentResult> parseAll(List<?> paths, ParseOptions options) {
    return parse(paths, options);
  }

  /** Compatibility alias; exceptions stop processing, as in the original parseAll. */
  public List<DocumentResult> parseAll(
      List<?> paths, ParseOptions options, ProgressListener listener) {
    return parse(paths, options, listener);
  }

  public void warmup() {
    warmup(ParseOptions.DEFAULT, null);
  }

  public void warmup(ParseOptions options, ProgressListener listener) {
    NativeBindings.warmup(
        state.get(), Objects.requireNonNull(options).json(), DownloadProgress.wrap(listener));
  }

  public List<String> getLoadedModels() {
    try {
      JsonNode roles = JSON.readTree(NativeBindings.loadedModels(state.get()));
      List<String> values = new ArrayList<>();
      roles.forEach(v -> values.add(v.asText()));
      return Collections.unmodifiableList(values);
    } catch (IOException e) {
      throw new IllegalStateException("Invalid native model list", e);
    }
  }

  @Override
  public void close() {
    cleanable.clean();
  }

  public static final class Builder {
    private final ObjectNode config = JSON.createObjectNode(), resources = JSON.createObjectNode();
    private Path outputDirectory;
    private boolean preload;
    private boolean progressLogging = true;

    /** Default parse progress; a custom listener replaces this view. Downloads log separately. */
    public Builder progressLogging(boolean enabled) {
      progressLogging = enabled;
      return this;
    }

    private Builder value(String key, Object value) {
      config.set(key, JSON.valueToTree(value));
      return this;
    }

    public Builder parseMethod(String v) {
      return value("parse_method", Objects.requireNonNull(v));
    }

    public Builder device(String v) {
      return value("device", Objects.requireNonNull(v));
    }

    public Builder language(String v) {
      return value("lang", Objects.requireNonNull(v));
    }

    public Builder formulaEnabled(boolean v) {
      return value("formula_enable", v);
    }

    public Builder tableEnabled(boolean v) {
      return value("table_enable", v);
    }

    public Builder orientationEnabled(boolean v) {
      return value("doc_orientation_classify", v);
    }

    public Builder pageBatchSize(int v) {
      if (v <= 0) throw new IllegalArgumentException("pageBatchSize must be positive");
      return value("pdf_pages_batch", v);
    }

    public Builder markdownMode(String v) {
      return value("make_md_mode", Objects.requireNonNull(v));
    }

    public Builder cropSampling(String v) {
      return value("crop_sampling", Objects.requireNonNull(v));
    }

    private Builder model(String group, String model) {
      config.with(group).put("model_type", Objects.requireNonNull(model));
      return this;
    }

    public Builder layoutModel(String v) {
      return model("layout_config", v);
    }

    public Builder formulaModel(String v) {
      return model("formula_config", v);
    }

    public Builder tableModel(String v) {
      return model("table_config", v);
    }

    public Builder layoutConfig(Map<String, ?> v) {
      return value("layout_config", Objects.requireNonNull(v));
    }

    public Builder ocrConfig(Map<String, ?> v) {
      return value("ocr_config", Objects.requireNonNull(v));
    }

    public Builder formulaConfig(Map<String, ?> v) {
      return value("formula_config", Objects.requireNonNull(v));
    }

    public Builder tableConfig(Map<String, ?> v) {
      return value("table_config", Objects.requireNonNull(v));
    }

    public Builder imageConfig(Map<String, ?> v) {
      return value("image_config", Objects.requireNonNull(v));
    }

    /** Additional shared Rust SDK configuration, without Java-specific defaults. */
    public Builder option(String key, Object v) {
      return value(Objects.requireNonNull(key), v);
    }

    public Builder runtimeDirectory(Path v) {
      resources.withArray("bundle_dirs").add(v.toAbsolutePath().toString());
      return this;
    }

    public Builder modelDirectory(Path v) {
      resources.put("model_directory", v.toAbsolutePath().toString());
      return this;
    }

    public Builder allowDownload(boolean v) {
      resources.put("allow_download", v);
      return this;
    }

    public Builder resource(String key, Path v) {
      resources.put(
          Objects.requireNonNull(key), Objects.requireNonNull(v).toAbsolutePath().toString());
      return this;
    }

    public Builder outputDirectory(Path v) {
      outputDirectory = Objects.requireNonNull(v);
      return this;
    }

    public Builder preloadModels(boolean v) {
      preload = v;
      return this;
    }

    public RapidDoc build() {
      return new RapidDoc(this);
    }
  }
}
