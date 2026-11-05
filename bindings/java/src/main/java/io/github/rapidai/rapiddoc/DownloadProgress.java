package io.github.rapidai.rapiddoc;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.Locale;
import java.util.logging.Logger;

/** Default model download logging, independent of optional parse progress views. */
final class DownloadProgress implements ProgressListener {
  private static final Logger LOG = Logger.getLogger("io.github.rapidai.rapiddoc.download");
  private final ProgressListener delegate;
  private double last = -5;

  private DownloadProgress(ProgressListener delegate) {
    this.delegate = delegate;
  }

  static ProgressListener wrap(ProgressListener delegate) {
    return new DownloadProgress(delegate);
  }

  public void onProgress(ProgressEvent event) {
    if (event.getStage().startsWith("source_download_")) {
      if (delegate instanceof ParseProgress) ((ParseProgress) delegate).clear();
      if (event.getStage().equals("source_download_started"))
        System.out.println("Downloading document: " + event.getDetail());
      else if (event.getStage().equals("source_download_completed"))
        System.out.println(
            "Downloaded document: " + event.getDetail() + " (" + event.getCompleted() + " bytes)");
    }
    if (event.getStage().startsWith("download_")) {
      if (delegate instanceof ParseProgress) ((ParseProgress) delegate).clear();
      try {
        JsonNode data = RapidDoc.JSON.readTree(event.getDetail());
        String model = data.path("model_id").asText();
        switch (event.getStage()) {
          case "download_started":
            last = -5;
            LOG.info("Initiating download: " + data.path("url").asText());
            LOG.info("Model destination: " + data.path("path").asText());
            break;
          case "download_progress":
            if (event.getElapsedSeconds() - last >= 5) {
              String size =
                  data.path("total").isNumber()
                      ? String.format(
                          Locale.ROOT, " / %.2f MiB", data.path("total").asDouble() / 1048576)
                      : " (total size unknown)";
              LOG.info(
                  String.format(
                      Locale.ROOT,
                      "Downloading %s: %.2f MiB%s [%.1fs]",
                      model,
                      event.getCompleted() / 1048576.0,
                      size,
                      event.getElapsedSeconds()));
              last = event.getElapsedSeconds();
            }
            break;
          case "download_verifying":
            LOG.info("Verifying SHA-256: " + model);
            break;
          case "download_completed":
            LOG.info("Successfully saved to: " + data.path("path").asText());
            break;
          case "download_failed":
            LOG.warning("Download failed: " + model + ": " + data.path("error").asText());
            break;
          default:
            break;
        }
      } catch (java.io.IOException invalid) {
        LOG.warning("Invalid native download event: " + invalid.getMessage());
      }
    }
    if (delegate != null) delegate.onProgress(event);
  }
}
