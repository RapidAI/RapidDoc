"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { resolveAddon } = require("../platform");

test("explicit addon path takes precedence over automatic platform packages", () => {
  assert.equal(
    resolveAddon({ libraryPath: "custom/rapid_doc_node.node" }),
    path.resolve("custom/rapid_doc_node.node"),
  );
});

test("Windows x64 resolves an installed scoped platform package", () => {
  const fs = require("node:fs");
  const os = require("node:os");
  const vm = require("node:vm");
  const { createRequire } = require("node:module");
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "rapid-doc-scoped-"));
  const previous = process.env.RAPIDDOC_NODE_LIBRARY;
  delete process.env.RAPIDDOC_NODE_LIBRARY;
  try {
    const packageRoot = path.join(root, "node_modules", "@hzkitty", "rapid-doc-win32-x64");
    fs.mkdirSync(path.join(packageRoot, "runtime"), { recursive: true });
    fs.writeFileSync(path.join(packageRoot, "package.json"), JSON.stringify({
      name: "@hzkitty/rapid-doc-win32-x64", version: "0.1.0",
    }));
    const addon = path.join(packageRoot, "runtime", "rapid_doc_node.node");
    fs.writeFileSync(addon, "resolution fixture");
    const sandbox = {
      module: { exports: {} }, process, __dirname: root,
      require: createRequire(path.join(root, "platform.js")),
    };
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../platform.js"), "utf8"), sandbox);
    assert.equal(sandbox.module.exports.resolveAddon({ config: { device: "cpu" } }, "win32", "x64"), addon);
  } finally {
    if (previous !== undefined) process.env.RAPIDDOC_NODE_LIBRARY = previous;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("missing CPU and CUDA packages identify the exact installable package", () => {
  // No native runtime is checked into the binding's source directory.
  const previous = process.env.RAPIDDOC_NODE_LIBRARY;
  delete process.env.RAPIDDOC_NODE_LIBRARY;
  try {
    assert.throws(
      () => resolveAddon({ config: { device: "cpu" } }, "linux", "x64"),
      /rapid-doc-linux-x64-gnu@/,
    );
    assert.throws(
      () => resolveAddon({}, "win32", "x64"),
      /@hzkitty\/rapid-doc-win32-x64@/,
    );
    assert.throws(
      () => resolveAddon({ config: { device: "cuda:0" } }, "win32", "x64"),
      /@hzkitty\/rapid-doc-win32-x64-cuda@/,
    );
    assert.throws(
      () => resolveAddon({}, "win32", "arm64"),
      /rapid-doc-win32-arm64@/,
    );
    assert.throws(
      () => resolveAddon({}, "linux", "arm64"),
      /rapid-doc-linux-arm64-gnu@/,
    );
    assert.throws(
      () => resolveAddon({}, "darwin", "arm64"),
      /rapid-doc-darwin-arm64@/,
    );
    assert.throws(
      () => resolveAddon({}, "darwin", "x64"),
      /rapid-doc-darwin-x64@/,
    );
  } finally {
    if (previous !== undefined) process.env.RAPIDDOC_NODE_LIBRARY = previous;
  }
});
