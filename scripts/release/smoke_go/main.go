package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	r "github.com/RapidAI/RapidDoc/bindings/go"
	"os"
	"path/filepath"
)

func main() {
	off := false
	e, err := r.New(r.Config{LibraryPath: os.Getenv("RAPIDDOC_NATIVE_LIBRARY"), ProgressLogging: &off,
		Options: r.Options{"layout_config": map[string]any{"model_type": "docvortex_rule"}, "formula_enable": false,
			"table_enable": false, "table_config": map[string]any{"use_word_box": false}},
		Resources: r.Resources{AllowDownload: &off}})
	if err != nil {
		panic(err)
	}
	defer e.Close()
	reports := map[string]any{}
	for _, name := range []string{"pdf_native.pdf", "synthetic.doc", "synthetic.docx", "synthetic.xls", "synthetic.xlsx", "synthetic.ppt", "synthetic.pptx"} {
		result, err := e.Parse(filepath.Join(os.Getenv("RAPIDDOC_TEST_ROOT"), "tests/migration/inputs", name), r.Options{"image_dir_name": "figures"}, nil)
		if err != nil {
			panic(err)
		}
		hashes := map[string]string{}
		for name, data := range result.Images {
			hashes[name] = fmt.Sprintf("%x", sha256.Sum256(data))
		}
		reports[name] = map[string]any{"markdown": result.MarkdownValue, "middle_json": result.MiddleJSON,
			"content_list_json": result.ContentListJSON, "images": hashes}
	}
	data, err := json.Marshal(reports)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(os.Getenv("RAPIDDOC_SMOKE_REPORT"), data, 0644); err != nil {
		panic(err)
	}
}
