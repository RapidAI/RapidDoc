"use strict";
const { Worker } = require("node:worker_threads");
const { AsyncLocalStorage } = require("node:async_hooks");
const callbackContext = new AsyncLocalStorage();
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");
const { fileURLToPath } = require("node:url");
const { ProgressView } = require("./progress");
const { resolveAddon } = require("./platform");

class RapidDocError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "RapidDocError";
    this.code = code;
  }
}
function nativeError(error) {
  return new RapidDocError(error.code, error.message);
}
function cancelled() {
  return new RapidDocError("cancelled", "Operation cancelled");
}

class DocumentResult {
  constructor(
    result,
    imageDirectory = "images",
    source = "document",
    method = "auto",
  ) {
    this.rawJSON = result.json;
    const data = JSON.parse(result.json);
    this.markdown = data.markdown;
    this.middleJSON = data.middle_json;
    this.contentListJSON = data.content_list_json;
    this.metadata = data.metadata;
    this.diagnostics = data.diagnostics;
    this.sourcePageCount = data.source_page_count;
    const buffers = (entries) =>
      Object.fromEntries(
        Object.entries(entries).map(([k, v]) => [
          k,
          Buffer.from(v.buffer, v.byteOffset, v.byteLength),
        ]),
      );
    this.images = buffers(result.images);
    this.drawings = buffers(result.drawings);
    this._imageDirectory = imageDirectory;
    this.sourceName = path.parse(this.metadata?.source_name || source).name || "document";
    this.parseMethod = this.metadata?.format === "office" ? "office" : method;
  }
  outputDirectory(directory) {
    if (
      !this.sourceName ||
      /[\\/:]/.test(this.sourceName) ||
      [".", ".."].includes(this.sourceName) ||
      !["auto", "txt", "ocr", "office"].includes(this.parseMethod)
    ) {
      throw new RapidDocError("invalid_input", "Invalid output name or method");
    }
    return path.resolve(directory, this.sourceName, this.parseMethod);
  }
  async writeTo(directory) {
    const root = this.outputDirectory(directory);
    async function write(name, data) {
      if (
        path.isAbsolute(name) ||
        /^[a-z]:/i.test(name) ||
        name.split(/[\\/]/).includes("..")
      ) {
        throw new RapidDocError("invalid_input", "Unsafe output asset path");
      }
      const target = path.resolve(root, name);
      const relative = path.relative(root, target);
      if (
        !relative ||
        relative.startsWith(`..${path.sep}`) ||
        path.isAbsolute(relative)
      ) {
        throw new RapidDocError(
          "invalid_input",
          "Output asset escapes directory",
        );
      }
      await fsp.mkdir(path.dirname(target), { recursive: true });
      await fsp.writeFile(target, data);
    }
    await write(
      this.sourceName +
        (typeof this.markdown === "string" ? ".md" : "_markdown.json"),
      typeof this.markdown === "string"
        ? this.markdown
        : JSON.stringify(this.markdown),
    );
    await write(
      this.sourceName + "_middle.json",
      JSON.stringify(this.middleJSON, null, 2),
    );
    await write(
      this.sourceName + "_content_list.json",
      JSON.stringify(this.contentListJSON, null, 2),
    );
    for (const [name, data] of Object.entries(this.images))
      await write(`${this._imageDirectory}/${name}`, data);
    for (const [name, data] of Object.entries(this.drawings))
      await write(`${this.sourceName}_${name}.pdf`, data);
  }
}

class RapidDoc {
  constructor(options = {}) {
    let libraryPath;
    try {
      libraryPath = resolveAddon(options);
    } catch (error) {
      throw new RapidDocError("runtime_unavailable", error.message);
    }
    if (!fs.existsSync(libraryPath))
      throw new RapidDocError(
        "runtime_unavailable",
        `Native addon not found: ${libraryPath}`,
      );
    this._logging = options.progressLogging !== false;
    this._parseMethod = options.config?.parse_method || "auto";
    this._mode =
      options.progressMode || process.env.RAPIDDOC_PROGRESS_MODE || "auto";
    if (!["auto", "bar", "plain"].includes(this._mode))
      throw new TypeError("progressMode must be auto, bar or plain");
    this._options = options;
    this._owner = this;
    this._peers = [];
    this._id = 0;
    this._closed = false;
    this._pending = null;
    this._ready = new Promise((resolve, reject) => {
      this._readyResolve = resolve;
      this._readyReject = reject;
    });
    this._ready.catch(() => {}); // Startup failure is surfaced by the next public call.
    const resources = { ...options.resources };
    resources.bundle_dirs = [
      ...(resources.bundle_dirs || []),
      ...(options.runtimeDirectory
        ? [path.resolve(options.runtimeDirectory)]
        : []),
      path.dirname(libraryPath),
    ];
    this._options = { ...options, libraryPath, config: structuredClone(options.config || {}), resources: structuredClone(resources) };
    this._worker = new Worker(path.join(__dirname, "worker.js"), {
      workerData: { libraryPath, config: options.config || {}, resources },
    });
    this._worker.on("message", (message) => {
      this._message(message).catch((error) => this._fail(error));
    });
    this._worker.on("error", (error) => this._fail(error));
    this._worker.on("exit", (code) => {
      if (!this._closed || this._pending)
        this._fail(
          new RapidDocError(
            "runtime_unavailable",
            `Native worker exited (${code})`,
          ),
        );
    });
  }
  _fail(error) {
    this._failure = error;
    this._readyReject(error);
    if (this._pending?.reject) this._pending.reject(error);
    if (this._closeReject) this._closeReject(error);
    this._worker.unref();
  }
  async _message(message) {
    if (message.type === "ready") {
      this._readyResolve();
      if (!this._pending) this._worker.unref();
      return;
    }
    if (message.type === "startupError") {
      this._fail(nativeError(message.error));
      return;
    }
    if (message.type === "closed") {
      if (message.error) this._closeReject(nativeError(message.error));
      else if (this._closeResolve) this._closeResolve();
      return;
    }
    const operation = this._pending;
    if (!operation || operation.id !== message.id) return;
    if (message.type === "progress") {
      try {
        operation.view.event(message.event);
        if (operation.progress && !Atomics.load(operation.sync, 1))
          await callbackContext.run(this._owner, () => operation.progress(message.event));
      } catch (error) {
        operation.callbackFailed = true;
        operation.error = error;
        Atomics.store(operation.sync, 1, 1);
      } finally {
        Atomics.store(operation.sync, 0, 1);
        Atomics.notify(operation.sync, 0);
      }
    } else if (message.type === "result") {
      if (operation.callbackFailed) operation.reject(operation.error);
      else if (Atomics.load(operation.sync, 1)) operation.reject(cancelled());
      else if (message.error) operation.reject(nativeError(message.error));
      else operation.resolve(message.result);
    }
  }
  async _invoke(op, payload, options, progress, signal, source = "") {
    if (this._closed)
      throw new RapidDocError("invalid_input", "RapidDoc is closed");
    if (this._failure) throw this._failure;
    if (callbackContext.getStore() === this._owner)
      throw new RapidDocError("invalid_input", "Engine busy in callback; reentry is not supported");
    if (this._pending) {
      let peer = this._peers.find((engine) => !engine._pending && !engine._closed && !engine._failure);
      if (!peer) {
        peer = new RapidDoc(this._options);
        peer._owner = this._owner;
        this._peers.push(peer);
      }
      return peer._invoke(op, payload, options, progress, signal, source);
    }
    if (
      progress !== undefined &&
      progress !== null &&
      typeof progress !== "function"
    )
      throw new TypeError("progress must be a function");
    if (signal?.aborted) throw cancelled();
    const operation = {
      id: ++this._id,
      sync: new Int32Array(new SharedArrayBuffer(8)),
      progress,
      view: new ProgressView(
        source,
        op === "parse" && this._logging && !progress,
        this._mode,
      ),
    };
    operation.finished = new Promise((resolve) => { operation.finish = resolve; });
    this._pending = operation;
    const abort = () => {
      Atomics.store(operation.sync, 1, 1);
      Atomics.store(operation.sync, 0, 1);
      Atomics.notify(operation.sync, 0);
    };
    signal?.addEventListener("abort", abort, { once: true });
    try {
      await this._ready;
      if (this._closed)
        throw new RapidDocError("invalid_input", "RapidDoc is closed");
      this._worker.ref();
      const result = await new Promise((resolve, reject) => {
        operation.resolve = resolve;
        operation.reject = reject;
        this._worker.postMessage({
          id: operation.id,
          op,
          ...payload,
          options,
          sync: operation.sync.buffer,
        });
      });
      operation.view.finish();
      return result;
    } catch (error) {
      operation.view.finish(error);
      throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
      if (this._pending === operation) this._pending = null;
      operation.finish();
      if (!this._closed) this._worker.unref();
    }
  }
  async parse(input, options = {}, progress, signal) {
    if (Array.isArray(input))
      return this.parseAll(input, options, progress, signal);
    if (input instanceof URL) input = input.protocol === "file:" ? fileURLToPath(input) : input.href;
    let payload, source;
    if (typeof input === "string") {
      const remote = /^https?:\/\//i.test(input);
      payload = { path: remote ? input : path.resolve(input) };
      source = remote ? "URL document" : path.basename(input);
    } else if (input instanceof Uint8Array) {
      payload = { data: Buffer.from(input) };
      source = "bytes";
    } else
      throw new TypeError(
        "input must be a local path, HTTP/HTTPS URL, file URL, Buffer, Uint8Array or array",
      );
    const result = await this._invoke(
      "parse",
      payload,
      options,
      progress,
      signal,
      source,
    );
    return new DocumentResult(
      result,
      options.image_dir_name || "images",
      typeof input === "string" ? path.basename(input) : "document",
      options.parse_method || this._parseMethod,
    );
  }
  async parseReader(reader, options = {}, progress, signal) {
    const chunks = [];
    for await (const chunk of reader) {
      if (signal?.aborted) throw cancelled();
      if (!(chunk instanceof Uint8Array))
        throw new TypeError("reader must yield binary chunks");
      chunks.push(Buffer.from(chunk));
    }
    return this.parse(Buffer.concat(chunks), options, progress, signal);
  }
  async parseAll(inputs, options = {}, progress, signal) {
    const results = [];
    for (const input of inputs)
      results.push(await this.parse(input, options, progress, signal));
    return results;
  }
  warmup(options = {}, progress, signal) {
    return this._invoke("warmup", {}, options, progress, signal);
  }
  async loadedModels() {
    if (callbackContext.getStore() === this._owner)
      throw new RapidDocError("invalid_input", "Engine busy in callback; reentry is not supported");
    const engines = [this, ...this._peers];
    const roles = await Promise.all(engines.map(async (engine) => {
      if (engine._pending) await engine._pending.finished;
      return engine._invoke("models", {}, {}, null);
    }));
    return [...new Set(roles.flat())].sort();
  }
  close() {
    if (this._closePromise) return this._closePromise;
    this._closed = true;
    if (this._pending) {
      Atomics.store(this._pending.sync, 1, 1);
      Atomics.store(this._pending.sync, 0, 1);
      Atomics.notify(this._pending.sync, 0);
    }
    const peersClosed = Promise.all(this._peers.map((engine) => engine.close()));
    peersClosed.catch(() => {});
    this._closePromise = (async () => {
      try {
        await this._ready;
        if (this._failure) throw this._failure;
        this._worker.ref();
        await new Promise((resolve, reject) => {
          this._closeResolve = resolve;
          this._closeReject = reject;
          this._worker.postMessage({ op: "close" });
        });
      } finally {
        await this._worker.terminate();
        await peersClosed;
      }
    })();
    return this._closePromise;
  }
}

module.exports = { RapidDoc, RapidDocError, DocumentResult };
