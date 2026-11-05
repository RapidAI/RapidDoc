"use strict";
const labels = {
  orientation: "Orientation Predict",
  layout: "Layout Predict",
  formula: "Formula Predict",
  ocr_detection: "PDF/OCR-det Predict",
  ocr_recognition: "OCR-rec Predict",
  table: "Table Predict",
  seal: "Seal Predict",
  assemble: "Processing pages",
};
class ProgressView {
  constructor(
    source,
    enabled,
    mode = "auto",
    status = process.stdout,
    bars = process.stderr,
  ) {
    this.source = source;
    this.enabled = enabled;
    this.status = status;
    this.bars = bars;
    this.terminal = mode === "bar" || (mode === "auto" && !!bars.isTTY);
    this.started = performance.now();
    this.counts = new Map();
    this.width = 0;
    this.line = false;
    this.lastDownload = -5;
    if (enabled) status.write(`Parsing: ${source}\n`);
  }
  clear() {
    if (this.line) {
      this.bars.write("\n");
      this.line = false;
      this.width = 0;
    }
  }
  event(event) {
    if (event.stage.startsWith("source_download_")) {
      this.clear();
      if (event.stage === "source_download_started") this.status.write(`Downloading document: ${event.detail}\n`);
      if (event.stage === "source_download_completed") this.status.write(`Downloaded document: ${event.detail} (${event.completed} bytes)\n`);
      return;
    }
    if (event.stage.startsWith("download_")) {
      this.download(event);
      return;
    }
    if (!this.enabled) return;
    if (event.stage === "batch") {
      this.clear();
      this.counts.clear();
      this.active = null;
      return;
    }
    if (event.stage === "document_opened") {
      this.clear();
      this.status.write(`${this.source}: ${event.total} pages\n`);
      return;
    }
    if (event.stage === "finalize") {
      this.clear();
      this.status.write(`${this.source}: Post-processing...\n`);
      return;
    }
    const label = labels[event.stage],
      done = event.completed,
      total = event.total;
    if (!label || done == null || total == null || done < 0 || total <= 0)
      return;
    const previous = this.counts.get(event.stage),
      now = performance.now();
    if (previous?.done === done && previous?.total === total) return;
    if (previous && done !== 0 && done !== total && now - previous.at < 1000)
      return;
    this.counts.set(event.stage, { done, total, at: now });
    if (this.active !== event.stage) {
      this.clear();
      this.active = event.stage;
    }
    const cells = Math.min(20, Math.max(0, Math.floor((done * 20) / total)));
    const time =
      event.elapsed_seconds != null && event.elapsed_seconds >= 0
        ? ` [processing ${event.elapsed_seconds.toFixed(2)}s]`
        : "";
    const message = `${this.source}: ${label}: ${Math.round(
      (done * 100) / total,
    )
      .toString()
      .padStart(
        3,
      )}% |${"=".repeat(cells)}${" ".repeat(20 - cells)}| ${done}/${total}${time}`;
    if (this.terminal) {
      this.bars.write(
        `\r${message}${" ".repeat(Math.max(0, this.width - message.length))}`,
      );
      this.width = message.length;
      this.line = true;
      if (done === total) this.clear();
    } else this.bars.write(`${message}\n`);
  }
  download(event) {
    this.clear();
    const data = JSON.parse(event.detail);
    let message;
    switch (event.stage) {
      case "download_started":
        this.lastDownload = -5;
        message = `Initiating download: ${data.url}\nModel destination: ${data.path}`;
        break;
      case "download_progress":
        if (data.elapsed_seconds - this.lastDownload < 5) return;
        this.lastDownload = data.elapsed_seconds;
        message = `Downloading ${data.model_id}: ${(data.completed / 1048576).toFixed(2)} MiB${data.total == null ? " (total size unknown)" : ` / ${(data.total / 1048576).toFixed(2)} MiB`} [${data.elapsed_seconds.toFixed(1)}s]`;
        break;
      case "download_verifying":
        message = `Verifying SHA-256: ${data.model_id}`;
        break;
      case "download_completed":
        message = `Successfully saved to: ${data.path}`;
        break;
      case "download_failed":
        message = `Download failed: ${data.model_id}: ${data.error}`;
        break;
      default:
        return;
    }
    this.status.write(`${message}\n`);
  }
  finish(error) {
    this.clear();
    if (this.enabled)
      this.status.write(
        `${this.source}: ${error ? `failed: ${error.message || error}` : `completed in ${((performance.now() - this.started) / 1000).toFixed(2)}s`}\n`,
      );
  }
}
module.exports = { ProgressView };
