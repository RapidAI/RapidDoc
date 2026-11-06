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
      () => resolveAddon({ config: { device: "cuda:0" } }, "win32", "x64"),
      /rapid-doc-win32-x64-cuda@/,
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
