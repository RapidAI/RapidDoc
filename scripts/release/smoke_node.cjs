const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { RapidDoc } = require("rapid-doc");
(async () => {
  const results = {};
  for (let round = 0; round < 3; round++) {
    const engine = new RapidDoc({
      config: {
        layout_config: { model_type: "docvortex_rule" },
        formula_enable: false,
        table_enable: false,
        table_config: { use_word_box: false },
      },
      progressLogging: false,
    });
    try {
      for (const name of [
        "pdf_native.pdf",
        "synthetic.doc",
        "synthetic.docx",
        "synthetic.xls",
        "synthetic.xlsx",
        "synthetic.ppt",
        "synthetic.pptx",
      ]) {
        const r = await engine.parse(
          path.join(
            process.env.RAPIDDOC_TEST_ROOT,
            "tests/migration/inputs",
            name,
          ),
          { image_dir_name: "figures" },
        );
        if (!r.markdown.length) throw new Error(`Empty result: ${name}`);
        results[name] = {
          markdown: r.markdown,
          middle_json: r.middleJSON,
          content_list_json: r.contentListJSON,
          images: Object.fromEntries(
            Object.entries(r.images).map(([n, b]) => [
              n,
              crypto.createHash("sha256").update(b).digest("hex"),
            ]),
          ),
        };
      }
    } finally {
      await engine.close();
    }
  }
  fs.writeFileSync(process.env.RAPIDDOC_SMOKE_REPORT, JSON.stringify(results));
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
