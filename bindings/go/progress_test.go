package rapiddoc

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func integer(value int64) *int64     { return &value }
func seconds(value float64) *float64 { return &value }
func stageEvent(stage string, done, total int64, elapsed float64) Event {
	return Event{Stage: stage, Completed: integer(done), Total: integer(total), ElapsedSeconds: seconds(elapsed)}
}
func captureView(terminal, enabled bool) (*consoleProgress, *bytes.Buffer, *bytes.Buffer) {
	status, bars := &bytes.Buffer{}, &bytes.Buffer{}
	return &consoleProgress{source: "test.pdf", status: status, bars: bars, enabled: enabled, terminal: terminal, started: time.Now(), counts: make(map[string]shownProgress), lastDownload: -5}, status, bars
}

func TestDefaultProgressMatchesJavaFormatAndDeduplicates(t *testing.T) {
	for _, terminal := range []bool{true, false} {
		view, status, bars := captureView(terminal, true)
		view.update(Event{Stage: "document_opened", Total: integer(14)})
		view.update(Event{Stage: "render"})
		view.update(stageEvent("layout", 0, 14, 0))
		view.update(stageEvent("layout", 14, 14, 8.58))
		view.update(stageEvent("formula", 5, 5, 3.20))
		view.update(stageEvent("ocr_detection", 14, 14, .25))
		view.update(stageEvent("ocr_recognition", 2, 2, .13))
		view.update(stageEvent("table", 0, 1, 0))
		view.update(stageEvent("assemble", 0, 14, 0))
		for index := 0; index < 10; index++ {
			view.update(stageEvent("table", 1, 1, .09+float64(index)*.001))
		}
		view.update(stageEvent("assemble", 14, 14, .35))
		view.update(Event{Stage: "finalize"})
		view.finish(nil)
		text := bars.String()
		for _, want := range []string{
			"test.pdf: Layout Predict: 100% |====================| 14/14 [processing 8.58s]",
			"Formula Predict:", "PDF/OCR-det Predict:", "OCR-rec Predict:", "Table Predict:", "Processing pages:",
		} {
			if !strings.Contains(text, want) {
				t.Fatalf("missing %q in %s", want, text)
			}
		}
		if strings.Count(text, "Table Predict: 100%") != 1 {
			t.Fatal("duplicate table completion")
		}
		if terminal && !strings.Contains(text, "\rtest.pdf:") {
			t.Fatal("interactive output must replace current line")
		}
		if !terminal && strings.Contains(text, "\r") {
			t.Fatal("redirected output has carriage return")
		}
		if strings.Contains(text, "\x1b") || strings.Contains(text, "-1/-1") || strings.Contains(text, "render:") {
			t.Fatal("internal/control output leaked")
		}
		if !strings.Contains(status.String(), "14 pages") || !strings.Contains(status.String(), "Post-processing...") || !strings.Contains(status.String(), "completed in") {
			t.Fatal(status.String())
		}
		view.update(Event{Stage: "batch"})
		view.update(stageEvent("table", 1, 1, .2))
		if strings.Count(bars.String(), "Table Predict: 100%") != 2 {
			t.Fatal("next batch was suppressed")
		}
	}
}

func TestDownloadsRemainVisibleWithCustomProgressAndErrorsPreserved(t *testing.T) {
	view, status, bars := captureView(true, false)
	cancellation := errors.New("callback cancel")
	data := `{"model_id":"model","url":"https://example.invalid/model","path":"models/model.onnx","completed":1048576,"total":2097152,"elapsed_seconds":1}`
	callback := view.callback(func(Event) error { return cancellation })
	if err := callback(Event{Stage: "download_started", Detail: &data}); err != cancellation {
		t.Fatal(err)
	}
	for _, stage := range []string{"download_progress", "download_verifying", "download_completed"} {
		view.update(Event{Stage: stage, Detail: &data})
	}
	if !strings.Contains(status.String(), "Initiating download:") || !strings.Contains(status.String(), "1.00 MiB / 2.00 MiB") || !strings.Contains(status.String(), "Successfully saved to:") {
		t.Fatal(status.String())
	}
	view.update(stageEvent("layout", 1, 1, 1))
	view.finish(nil)
	if bars.Len() != 0 || strings.Contains(status.String(), "completed in") {
		t.Fatal("custom progress duplicated the default parse view")
	}
}

func TestUnknownDownloadSizeAndFailedParse(t *testing.T) {
	view, status, _ := captureView(false, true)
	detail, _ := json.Marshal(map[string]any{"model_id": "model", "completed": 1024, "total": nil, "elapsed_seconds": 0})
	data := string(detail)
	view.update(Event{Stage: "download_progress", Detail: &data})
	view.finish(errors.New("test failure"))
	if !strings.Contains(status.String(), "total size unknown") || !strings.Contains(status.String(), "failed: test failure") || strings.Contains(status.String(), "completed in") {
		t.Fatal(status.String())
	}
}
