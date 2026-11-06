"use strict";
const fs = require("node:fs");
const path = require("node:path");

function resolveAddon(
  options,
  platform = process.platform,
  arch = process.arch,
) {
  const explicit = options.libraryPath || process.env.RAPIDDOC_NODE_LIBRARY;
  if (explicit) return path.resolve(explicit);
  const bundled = path.join(__dirname, "runtime", "rapid_doc_node.node");
  if (fs.existsSync(bundled)) return bundled; // Existing standalone SDK archives.
  const names = {
    "win32-x64": "rapid-doc-win32-x64",
    "linux-x64": "rapid-doc-linux-x64-gnu",
    "darwin-x64": "rapid-doc-darwin-x64",
    "darwin-arm64": "rapid-doc-darwin-arm64",
  };
  const base = names[`${platform}-${arch}`];
  if (!base)
    throw new Error(`Unsupported RapidDoc platform: ${platform}-${arch}`);
  const cuda = String(
    options.config?.device || process.env.MINERU_DEVICE_MODE || "cpu",
  ).startsWith("cuda");
  const name = base + (cuda ? "-cuda" : "");
  try {
    return require.resolve(`${name}/runtime/rapid_doc_node.node`);
  } catch (error) {
    if (error.code !== "MODULE_NOT_FOUND") throw error;
    throw new Error(
      `RapidDoc native package missing; install ${name}@${require("./package.json").version}`,
    );
  }
}
module.exports = { resolveAddon };
