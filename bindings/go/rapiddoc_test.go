package rapiddoc

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"
)

func testRoot(t *testing.T) string {
	t.Helper()
	root := os.Getenv("RAPIDDOC_TEST_ROOT")
	if root == "" {
		t.Skip("set RAPIDDOC_TEST_ROOT and RAPIDDOC_NATIVE_LIBRARY for native acceptance")
	}
	return root
}
func testConfig(t *testing.T) Config {
	testRoot(t)
	disabled := false
	return Config{Options: Options{"layout_config": map[string]any{"model_type": "docvortex_rule"}, "formula_enable": false, "table_enable": false, "crop_sampling": "opencv4"}, Resources: Resources{AllowDownload: &disabled}, ProgressLogging: &disabled, LibraryPath: os.Getenv("RAPIDDOC_NATIVE_LIBRARY")}
}
func mustEngine(t *testing.T) *Engine {
	t.Helper()
	e, err := New(testConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() })
	return e
}

func TestHTTPInputsUseSharedDownloader(t *testing.T) {
	e := mustEngine(t)
	data, err := os.ReadFile(filepath.Join(testRoot(t), "tests/migration/inputs/synthetic.docx"))
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/redirect" {
			http.Redirect(w, r, "/data", http.StatusFound)
			return
		}
		if r.URL.Path == "/missing" {
			http.NotFound(w, r)
			return
		}
		w.Write(data)
	}))
	defer server.Close()
	local, err := e.ParseBytes(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	remote, err := e.Parse(server.URL+"/redirect?filename=report.docx", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if remote.Markdown != local.Markdown || remote.SourceName != "report" {
		t.Fatal("remote output/name differs")
	}
	if _, err = e.Parse(server.URL+"/missing", nil, nil); err == nil {
		t.Fatal("HTTP 404 accepted")
	}
	cancel := errors.New("cancel URL")
	_, err = e.Parse(server.URL+"/data", nil, func(event Event) error {
		if event.Stage == "source_download_started" {
			return cancel
		}
		return nil
	})
	if err != cancel {
		t.Fatal("download cancellation not preserved", err)
	}
}
func equalJSON(t *testing.T, expected, actual []byte) {
	t.Helper()
	var a, b any
	if err := json.Unmarshal(expected, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(actual, &b); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("JSON differs from reference")
	}
}

func TestSevenFormatsReferenceAndOwnedOutput(t *testing.T) {
	root := testRoot(t)
	e := mustEngine(t)
	data, err := os.ReadFile(filepath.Join(root, "tests/fixtures/java_sdk_expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	var expected map[string]struct {
		Markdown json.RawMessage   `json:"markdown"`
		Middle   json.RawMessage   `json:"middle_json"`
		Content  json.RawMessage   `json:"content_list_json"`
		Images   map[string]string `json:"images"`
	}
	if err = json.Unmarshal(data, &expected); err != nil {
		t.Fatal(err)
	}
	var retained *Result
	for _, name := range []string{"synthetic.doc", "synthetic.docx", "synthetic.xls", "synthetic.xlsx", "synthetic.ppt", "synthetic.pptx", "pdf_native.pdf"} {
		t.Run(name, func(t *testing.T) {
			r, err := e.Parse(filepath.Join(root, "tests/migration/inputs", name), Options{"image_dir_name": "figures"}, nil)
			if err != nil {
				t.Fatal(err)
			}
			ref := expected[name]
			equalJSON(t, ref.Markdown, r.MarkdownValue)
			equalJSON(t, ref.Middle, r.MiddleJSON)
			equalJSON(t, ref.Content, r.ContentListJSON)
			if len(ref.Images) != len(r.Images) {
				t.Fatal("image count")
			}
			for n, b := range r.Images {
				hash := sha256.Sum256(b)
				if hex.EncodeToString(hash[:]) != ref.Images[n] {
					t.Fatal("asset differs", n)
				}
			}
			retained = r
			if name == "synthetic.docx" {
				raw, _ := os.ReadFile(filepath.Join(root, "tests/migration/inputs", name))
				stream, err := e.ParseReader(bytes.NewReader(raw), Options{"image_dir_name": "figures"}, nil)
				if err != nil {
					t.Fatal(err)
				}
				equalJSON(t, r.MiddleJSON, stream.MiddleJSON)
			}
		})
	}
	if err = e.Close(); err != nil {
		t.Fatal(err)
	}
	if err = e.Close(); err != nil {
		t.Fatal(err)
	}
	if err = retained.WriteTo(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	if _, err = e.Parse("does-not-matter", nil, nil); err == nil {
		t.Fatal("closed engine accepted")
	}
}

func TestCallbacksCancellationReentryAndPanic(t *testing.T) {
	root := testRoot(t)
	e := mustEngine(t)
	input := filepath.Join(root, "tests/migration/inputs/pdf_native.pdf")
	cancel := errors.New("cancel from Go")
	_, err := e.Parse(input, nil, func(Event) error { return cancel })
	if err != cancel {
		t.Fatalf("callback error not preserved: %v", err)
	}
	called := false
	_, err = e.Parse(input, nil, func(Event) error {
		if !called {
			called = true
			_, err := e.Parse(input, nil, nil)
			var failure *NativeError
			if !errors.As(err, &failure) || failure.Code != "invalid_input" {
				return fmt.Errorf("reentry must fail without blocking: %v", err)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	func() {
		defer func() {
			if recover() != "callback panic" {
				t.Error("callback panic not restored")
			}
		}()
		e.Parse(input, nil, func(Event) error { panic("callback panic") })
	}()
	if _, err = e.Parse(input, nil, nil); err != nil {
		t.Fatal("engine not usable after cancellation", err)
	}
}

func TestCPUOCRAndSessions(t *testing.T) {
	root := testRoot(t)
	c := testConfig(t)
	c.Options["ocr_config"] = map[string]any{"seal_enable": false, "Det.model_path": filepath.Join(root, "rapid_doc/resources/PP-OCRv6_det_small.onnx"), "Rec.model_path": filepath.Join(root, "rapid_doc/resources/PP-OCRv6_rec_small.onnx"), "EngineConfig.onnxruntime.intra_op_num_threads": 1}
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if _, err = e.Warmup(nil, nil); err != nil {
		t.Fatal(err)
	}
	roles, err := e.LoadedModels()
	if err != nil || len(roles) != 2 {
		t.Fatal(roles, err)
	}
	data, err := os.ReadFile(filepath.Join(root, "build/runtime-compatibility/microsoft21-sxs-model-output.json"))
	if err != nil {
		t.Fatal(err)
	}
	var reference map[string]struct {
		Markdown json.RawMessage `json:"markdown"`
		Middle   json.RawMessage `json:"middle_json"`
		Content  json.RawMessage `json:"content_list_json"`
	}
	json.Unmarshal(data, &reference)
	for _, name := range []string{"pdf_scan.pdf", "pdf_mixed.pdf"} {
		r, err := e.Parse(filepath.Join(root, "tests/migration/inputs", name), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		ref := reference[name]
		equalJSON(t, ref.Markdown, r.MarkdownValue)
		equalJSON(t, ref.Middle, r.MiddleJSON)
		equalJSON(t, ref.Content, r.ContentListJSON)
	}
}

func TestConcurrentEnginesAndInvalidConfig(t *testing.T) {
	c := testConfig(t)
	bad := c
	bad.Options = Options{"unknown_config": true}
	if e, err := New(bad); err == nil {
		e.Close()
		t.Fatal("unknown config accepted")
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			e, err := New(c)
			if err != nil {
				t.Error(err)
				return
			}
			defer e.Close()
			r, err := e.Parse(filepath.Join(testRoot(t), "tests/migration/inputs/synthetic.docx"), nil, nil)
			if err != nil || r == nil || r.Markdown == "" {
				t.Error("concurrent parse", err)
			}
		}()
	}
	wg.Wait()
}

func TestSameEngineOverlappingPDFAndOffice(t *testing.T) {
	e := mustEngine(t)
	root := testRoot(t)
	input := filepath.Join(root, "tests/migration/inputs/pdf_native.pdf")
	want, err := e.Parse(input, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	entered, resume := make(chan struct{}, 2), make(chan struct{})
	type outcome struct {
		result *Result
		err    error
	}
	done := make(chan outcome, 3)
	for i := 0; i < 2; i++ {
		go func() {
			r, err := e.Parse(input, nil, func(event Event) error {
				if event.Stage == "document_opened" {
					entered <- struct{}{}
					select {
					case <-resume:
					case <-time.After(20 * time.Second):
						return errors.New("overlap timeout")
					}
				}
				return nil
			})
			done <- outcome{r, err}
		}()
	}
	go func() {
		r, err := e.Parse(filepath.Join(root, "tests/migration/inputs/synthetic.docx"), nil, nil)
		done <- outcome{r, err}
	}()
	for i := 0; i < 2; i++ {
		select {
		case <-entered:
		case <-time.After(20 * time.Second):
			close(resume)
			t.Fatal("same-engine PDF calls did not overlap")
		}
	}
	close(resume)
	for i := 0; i < 3; i++ {
		select {
		case got := <-done:
			if got.err != nil {
				t.Fatal(got.err)
			}
			if got.result.ParseMethod != "office" && got.result.Markdown != want.Markdown {
				t.Fatal("PDF output changed")
			}
		case <-time.After(30 * time.Second):
			t.Fatal("concurrent parse timeout")
		}
	}
}

func TestWriterRejectsTraversal(t *testing.T) {
	r := &Result{Images: map[string][]byte{"../../escape": {1}}, imageDirectory: "images"}
	if err := r.WriteTo(t.TempDir()); err == nil {
		t.Fatal("asset traversal accepted")
	}
}

func TestWriterUsesSourceAndMethod(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"论文", "报告"} {
		r := &Result{SourceName: name, ParseMethod: "auto", Markdown: "![figure](figures/a.png)",
			MiddleJSON: []byte(`{}`), ContentListJSON: []byte(`[]`), imageDirectory: "figures",
			Images: map[string][]byte{"a.png": {1, 2}}, Drawings: map[string][]byte{"layout": []byte("%PDF")}}
		if err := r.WriteTo(root); err != nil {
			t.Fatal(err)
		}
		for _, relative := range []string{name + ".md", name + "_middle.json", name + "_content_list.json", name + "_layout.pdf", "figures/a.png"} {
			if _, err := os.Stat(filepath.Join(root, name, "auto", relative)); err != nil {
				t.Fatal(err)
			}
		}
	}
	r := &Result{SourceName: "document", ParseMethod: "office", MarkdownValue: []byte(`[]`), MiddleJSON: []byte(`{}`), ContentListJSON: []byte(`[]`)}
	if err := r.WriteTo(root); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "document", "office", "document_markdown.json")); err != nil {
		t.Fatal(err)
	}
}
