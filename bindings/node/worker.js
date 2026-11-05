"use strict";
const { parentPort, workerData } = require("node:worker_threads");
let engine;
function failure(error) {
  try {
    return JSON.parse(error.message);
  } catch {
    return {
      code: "runtime_unavailable",
      message: error.message || String(error),
    };
  }
}
try {
  const { NativeEngine } = require(workerData.libraryPath);
  engine = new NativeEngine(
    JSON.stringify(workerData.config),
    JSON.stringify(workerData.resources),
  );
  parentPort.postMessage({ type: "ready" });
} catch (error) {
  parentPort.postMessage({ type: "startupError", error: failure(error) });
}
parentPort.on("message", (message) => {
  if (message.op === "close") {
    try {
      if (engine) engine.close();
      parentPort.postMessage({ type: "closed" });
    } catch (error) {
      parentPort.postMessage({ type: "closed", error: failure(error) });
    }
    return;
  }
  const sync = new Int32Array(message.sync);
  function progress(text) {
    if (Atomics.load(sync, 1)) return false;
    Atomics.store(sync, 0, 0);
    parentPort.postMessage({
      type: "progress",
      id: message.id,
      event: JSON.parse(text),
    });
    // Only this worker waits. The main event loop remains available for callbacks,
    // AbortSignal and other requests. Close/abort wakes the worker immediately.
    while (!Atomics.load(sync, 0) && !Atomics.load(sync, 1))
      Atomics.wait(sync, 0, 0);
    return !Atomics.load(sync, 1);
  }
  try {
    if (!engine) throw new Error("Native engine was not initialized");
    if (Atomics.load(sync, 1)) throw new Error("Operation cancelled");
    let result;
    if (message.op === "parse") {
      result = engine.parse(
        message.path,
        message.data ? Buffer.from(message.data) : undefined,
        JSON.stringify(message.options),
        progress,
      );
    } else if (message.op === "warmup") {
      result = engine.warmup(JSON.stringify(message.options), progress);
    } else if (message.op === "models") result = engine.loadedModels();
    else throw new Error("Unknown worker operation");
    parentPort.postMessage({ type: "result", id: message.id, result });
  } catch (error) {
    parentPort.postMessage({
      type: "result",
      id: message.id,
      error: failure(error),
    });
  }
});
