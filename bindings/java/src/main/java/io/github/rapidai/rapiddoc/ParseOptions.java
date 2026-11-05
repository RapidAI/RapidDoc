package io.github.rapidai.rapiddoc;

import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.Objects;

/** Per-call overrides. endPage is EXCLUSIVE, matching the Rust SDK. */
public final class ParseOptions {
  public static final ParseOptions DEFAULT = new Builder().build();
  private final ObjectNode values;

  private ParseOptions(ObjectNode values) {
    this.values = values.deepCopy();
  }

  String json() {
    return values.toString();
  }

  String imageDirectory() {
    return values.path("image_dir_name").asText("images");
  }

  String parseMethod(String fallback) {
    return values.path("parse_method").asText(fallback);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private final ObjectNode values = RapidDoc.JSON.createObjectNode();

    public Builder startPage(int page) {
      if (page < 0) throw new IllegalArgumentException("startPage must be nonnegative");
      values.put("start_page", page);
      return this;
    }

    public Builder endPage(Integer page) {
      if (page != null && page < 0)
        throw new IllegalArgumentException("endPage must be nonnegative");
      values.put("end_page", page);
      return this;
    }

    public Builder parseMethod(String value) {
      values.put("parse_method", Objects.requireNonNull(value));
      return this;
    }

    public Builder formulaEnabled(boolean value) {
      values.put("formula_enable", value);
      return this;
    }

    public Builder tableEnabled(boolean value) {
      values.put("table_enable", value);
      return this;
    }

    public Builder language(String value) {
      values.put("lang", Objects.requireNonNull(value));
      return this;
    }

    public Builder imageDirectory(String value) {
      values.put("image_dir_name", Objects.requireNonNull(value));
      return this;
    }

    public Builder orientationEnabled(boolean value) {
      values.put("doc_orientation_classify", value);
      return this;
    }

    public Builder drawLayout(boolean value) {
      values.put("draw_layout", value);
      return this;
    }

    public Builder drawSpans(boolean value) {
      values.put("draw_span", value);
      return this;
    }

    public Builder drawLineOrder(boolean value) {
      values.put("draw_line_sort", value);
      return this;
    }

    /** Empty byte input needs a format hint; normal inputs are detected natively. */
    public Builder emptyFormat(String value) {
      values.put("empty_format", Objects.requireNonNull(value));
      return this;
    }

    public Builder option(String name, Object value) {
      values.set(Objects.requireNonNull(name), RapidDoc.JSON.valueToTree(value));
      return this;
    }

    public ParseOptions build() {
      return new ParseOptions(values);
    }
  }
}
