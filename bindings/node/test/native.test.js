"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { Readable } = require("node:stream");
const { createHash } = require("node:crypto");
const http = require("node:http");
const { RapidDoc } = require("..");
const root = process.env.RAPIDDOC_TEST_ROOT;
const native = { skip: !root, timeout: 120000 };
const input = (name) => path.join(root, "tests/migration/inputs", name);

test("HTTP string and URL object use native downloading and preserve output names", native, async () => {
  const bytes = fs.readFileSync(input("synthetic.docx"));
  const server = http.createServer((req, res) => {
    if (req.url.startsWith("/redirect")) { res.writeHead(302, { Location: "/data" }); res.end(); }
    else if (req.url === "/missing") { res.writeHead(404); res.end(); }
    else { res.writeHead(200, { "Content-Length": bytes.length }); res.end(bytes); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const e = engine();
  try {
    const local = await e.parse(bytes);
    for (const source of [`${base}/redirect?filename=report.docx`, new URL(`${base}/data?filename=report.docx`)]) {
      const result = await e.parse(source);
      assert.equal(result.markdown, local.markdown);
      assert.deepEqual(result.middleJSON, local.middleJSON);
      assert.equal(result.sourceName, "report");
    }
    await assert.rejects(e.parse(`${base}/missing`), /404/);
    await assert.rejects(e.parse(`${base}/data`, {}, (event) => {
      if (event.stage === "source_download_started") throw new Error("cancel URL");
    }), /cancel URL/);
  } finally { await e.close(); await new Promise((resolve) => server.close(resolve)); }
});

test("same engine runs overlapping PDF and Office requests and closes every worker", native, async () => {
  const e = engine();
  try {
    const expected = (await e.parse(input("pdf_native.pdf"))).markdown;
    let entered = 0;
    let release;
    const both = new Promise((resolve) => { release = resolve; });
    const progress = async (event) => {
      if (event.stage === "document_opened") {
        if (++entered === 2) release();
        await both;
      }
    };
    const results = await Promise.all([
      e.parse(input("pdf_native.pdf"), {}, progress),
      e.parse(input("pdf_native.pdf"), {}, progress),
      e.parse(input("synthetic.docx")),
      e.parse(input("synthetic.xlsx")),
    ]);
    assert.equal(entered, 2);
    assert.equal(results[0].markdown, expected);
    assert.equal(results[1].markdown, expected);
    assert.ok(results.slice(2).every((result) => result.markdown.length));
    assert.deepEqual(await e.loadedModels(), []);
  } finally {
    await e.close();
  }
  await assert.rejects(e.parse(input("synthetic.docx")), /closed/);
});
function engine(extra = {}) {
  return new RapidDoc({
    libraryPath: process.env.RAPIDDOC_NODE_LIBRARY,
    runtimeDirectory: process.env.RAPIDDOC_NODE_RUNTIME,
    config: {
      layout_config: { model_type: "docvortex_rule" },
      formula_enable: false,
      table_enable: false,
      crop_sampling: "opencv4",
    },
    resources: { allow_download: false },
    progressLogging: false,
    ...extra,
  });
}
test(
  "seven formats match shared reference, owned assets and drawings survive close",
  native,
  async () => {
    const e = engine();
    const references = JSON.parse(
      fs.readFileSync(path.join(root, "tests/fixtures/java_sdk_expected.json")),
    );
    let retained;
    try {
      for (const [name, ref] of Object.entries(references)) {
        const result = await e.parse(input(name), {
          image_dir_name: "figures",
        });
        assert.deepEqual(result.markdown, ref.markdown, name);
        assert.deepEqual(result.middleJSON, ref.middle_json, name);
        assert.deepEqual(result.contentListJSON, ref.content_list_json, name);
        assert.deepEqual(
          Object.keys(result.images).sort(),
          Object.keys(ref.images).sort(),
        );
        for (const [asset, bytes] of Object.entries(result.images)) {
          assert.ok(Buffer.isBuffer(bytes));
          assert.equal(
            createHash("sha256").update(bytes).digest("hex"),
            ref.images[asset],
          );
        }
        retained = result;
      }
      const result = await e.parse(input("pdf_native.pdf"), {
        draw_layout: true,
        draw_span: true,
      });
      assert.ok(Object.keys(result.drawings).length >= 2);
      const output = path.join(root, "build/node-acceptance/output");
      await result.writeTo(output);
      assert.equal(
        fs.readFileSync(
          path.join(result.outputDirectory(output), "pdf_native.md"),
          "utf8",
        ),
        result.markdown,
      );
      for (const name of [
        "pdf_native_middle.json",
        "pdf_native_content_list.json",
        "pdf_native_layout.pdf",
        "pdf_native_span.pdf",
      ]) {
        assert.ok(
          fs.existsSync(path.join(result.outputDirectory(output), name)),
        );
      }
      assert.ok(
        Object.values(result.drawings).every(
          (b) => b.subarray(0, 4).toString() === "%PDF",
        ),
      );
      await e.close();
      assert.ok(retained.markdown.length);
      await assert.rejects(e.parse(input("synthetic.docx")), /closed/);
    } finally {
      await e.close();
    }
  },
);
test("repeated worker recreation releases each engine safely", native, async () => {
  for (let i = 0; i < 8; i++) {
    const e = engine();
    try {
      assert.ok((await e.parse(input("synthetic.docx"))).markdown.length);
    } finally {
      await e.close();
    }
  }
});

test(
  "bytes, binary stream, file URL, list, error recovery and worker heartbeat",
  native,
  async () => {
    const e = engine();
    let ticks = 0;
    const timer = setInterval(() => ticks++, 1);
    try {
      const filename = input("synthetic.docx"),
        bytes = fs.readFileSync(filename);
      const a = await e.parse(bytes);
      const b = await e.parseReader(
        Readable.from([bytes.subarray(0, 15), bytes.subarray(15)]),
      );
      const c = await e.parse(require("node:url").pathToFileURL(filename));
      assert.equal(a.rawJSON, b.rawJSON);
      assert.equal(a.rawJSON, c.rawJSON);
      const output = path.join(root, "build/node-acceptance/named-output");
      await a.writeTo(output);
      await c.writeTo(output);
      assert.ok(
        fs.existsSync(path.join(output, "document/office/document.md")),
      );
      assert.ok(
        fs.existsSync(path.join(output, "synthetic/office/synthetic.md")),
      );
      assert.equal(
        (await e.parse([filename, input("synthetic.xls")])).length,
        2,
      );
      await assert.rejects(e.parse(Buffer.from("invalid document")));
      assert.equal((await e.parse(filename)).markdown, a.markdown);
      assert.ok(ticks > 0, "main Node event loop remained responsive");
      assert.deepEqual(await e.loadedModels(), []);
    } finally {
      clearInterval(timer);
      await e.close();
    }
  },
);
test(
  "async callback errors, reentry, cancellation and close from callback do not deadlock",
  native,
  async () => {
    const e = engine();
    try {
      const failure = new Error("original callback error");
      await assert.rejects(
        e.parse(input("pdf_native.pdf"), {}, async () => {
          throw failure;
        }),
        (x) => x === failure,
      );
      await assert.rejects(
        e.parse(input("pdf_native.pdf"), {}, () => {
          throw 0;
        }),
        (x) => x === 0,
      );
      let called = false;
      await e.parse(input("pdf_native.pdf"), {}, async () => {
        if (called) return;
        called = true;
        await assert.rejects(e.loadedModels(), /busy/);
      });
      assert.ok(called);
      const controller = new AbortController();
      await assert.rejects(
        e.parse(
          input("pdf_native.pdf"),
          {},
          () => controller.abort(),
          controller.signal,
        ),
        (x) => x.code === "cancelled",
      );
      assert.ok((await e.parse(input("synthetic.doc"))).markdown.length);
      await assert.rejects(
        e.parse(input("pdf_native.pdf"), {}, async () => {
          await e.close();
        }),
        (x) => x.code === "cancelled",
      );
    } finally {
      await e.close();
    }
  },
);
test(
  "separate engines can parse concurrently and startup errors propagate",
  native,
  async () => {
    const a = engine(),
      b = engine();
    try {
      const values = await Promise.all([
        a.parse(input("synthetic.doc")),
        b.parse(input("synthetic.pptx")),
      ]);
      assert.ok(values.every((r) => r.markdown.length));
    } finally {
      await Promise.all([a.close(), b.close()]);
    }
    const broken = engine({ config: { nonexistent: true } });
    await assert.rejects(broken.parse(input("synthetic.doc")), /nonexistent/);
    await assert.rejects(broken.close());
  },
);
test(
  "CPU OCR sessions match the existing Java/Go reference",
  native,
  async () => {
    const e = engine({
      config: {
        layout_config: { model_type: "docvortex_rule" },
        formula_enable: false,
        table_enable: false,
        crop_sampling: "opencv4",
        ocr_config: {
          seal_enable: false,
          "Det.model_path": path.join(
            root,
            "rapid_doc/resources/PP-OCRv6_det_small.onnx",
          ),
          "Rec.model_path": path.join(
            root,
            "rapid_doc/resources/PP-OCRv6_rec_small.onnx",
          ),
          "EngineConfig.onnxruntime.intra_op_num_threads": 1,
        },
      },
    });
    try {
      await e.warmup();
      assert.ok((await e.loadedModels()).length >= 2);
      const reference = JSON.parse(
        fs.readFileSync(
          path.join(
            root,
            "build/runtime-compatibility/microsoft21-sxs-model-output.json",
          ),
        ),
      );
      for (const name of ["pdf_scan.pdf", "pdf_mixed.pdf"]) {
        const result = await e.parse(input(name));
        assert.deepEqual(result.markdown, reference[name].markdown);
        assert.deepEqual(result.middleJSON, reference[name].middle_json);
        assert.deepEqual(
          result.contentListJSON,
          reference[name].content_list_json,
        );
      }
    } finally {
      await e.close();
    }
  },
);
