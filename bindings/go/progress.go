package rapiddoc

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// The stage names/format match the Java view; processing time is native active
// time, while completion reports wall time. No parsing is performed here.
var stageLabels = map[string]string{
	"orientation": "Orientation Predict", "layout": "Layout Predict",
	"formula": "Formula Predict", "ocr_detection": "PDF/OCR-det Predict",
	"ocr_recognition": "OCR-rec Predict", "table": "Table Predict",
	"seal": "Seal Predict", "assemble": "Processing pages",
}

type shownProgress struct {
	done, total int64
	at          time.Time
}
type consoleProgress struct {
	source                  string
	status, bars            io.Writer
	enabled, terminal, line bool
	started                 time.Time
	width                   int
	active                  string
	counts                  map[string]shownProgress
	lastDownload            float64
}

func newConsoleProgress(source, mode string, enabled bool) *consoleProgress {
	terminal := mode == "bar" || (mode == "auto" && terminalOutput())
	view := &consoleProgress{source: source, status: os.Stdout, bars: os.Stderr,
		enabled: enabled, terminal: terminal, started: time.Now(), counts: make(map[string]shownProgress), lastDownload: -5}
	if enabled {
		fmt.Fprintf(view.status, "Parsing: %s\n", source)
	}
	return view
}
func (p *consoleProgress) clear() {
	if p.line {
		fmt.Fprintln(p.bars)
		p.line = false
		p.width = 0
	}
}
func (p *consoleProgress) update(event Event) error {
	if strings.HasPrefix(event.Stage, "source_download_") {
		p.clear()
		name := "document"
		if event.Detail != nil {
			name = *event.Detail
		}
		if event.Stage == "source_download_started" {
			fmt.Fprintf(p.status, "Downloading document: %s\n", name)
		}
		if event.Stage == "source_download_completed" {
			fmt.Fprintf(p.status, "Downloaded document: %s\n", name)
		}
		return nil
	}
	if strings.HasPrefix(event.Stage, "download_") {
		p.download(event)
		return nil
	}
	if !p.enabled {
		return nil
	}
	switch event.Stage {
	case "batch":
		p.clear()
		p.counts = make(map[string]shownProgress)
		p.active = ""
		return nil
	case "document_opened":
		p.clear()
		if event.Total != nil {
			fmt.Fprintf(p.status, "%s: %d pages\n", p.source, *event.Total)
		}
		return nil
	case "finalize":
		p.clear()
		fmt.Fprintf(p.status, "%s: Post-processing...\n", p.source)
		return nil
	}
	label, ok := stageLabels[event.Stage]
	if !ok || event.Completed == nil || event.Total == nil || *event.Completed < 0 || *event.Total <= 0 {
		return nil
	}
	done, total := *event.Completed, *event.Total
	previous, seen := p.counts[event.Stage]
	if seen && previous.done == done && previous.total == total {
		return nil
	}
	now := time.Now()
	if seen && done != 0 && done != total && now.Sub(previous.at) < time.Second {
		return nil
	}
	p.counts[event.Stage] = shownProgress{done, total, now}
	if p.active != event.Stage {
		p.clear()
		p.active = event.Stage
	}
	filled := int(float64(done) * 20 / float64(total))
	if filled < 0 {
		filled = 0
	}
	if filled > 20 {
		filled = 20
	}
	elapsed := ""
	if event.ElapsedSeconds != nil && *event.ElapsedSeconds >= 0 {
		elapsed = fmt.Sprintf(" [processing %.2fs]", *event.ElapsedSeconds)
	}
	message := fmt.Sprintf("%s: %s: %3.0f%% |%s%s| %d/%d%s", p.source, label, float64(done)*100/float64(total), strings.Repeat("=", filled), strings.Repeat(" ", 20-filled), done, total, elapsed)
	if p.terminal {
		padding := p.width - len(message)
		if padding < 0 {
			padding = 0
		}
		fmt.Fprintf(p.bars, "\r%s%s", message, strings.Repeat(" ", padding))
		p.width = len(message)
		p.line = true
		if done == total {
			p.clear()
		}
	} else {
		fmt.Fprintln(p.bars, message)
	}
	return nil
}
func (p *consoleProgress) download(event Event) {
	p.clear()
	if event.Detail == nil {
		return
	}
	var data struct {
		Model     string  `json:"model_id"`
		URL       string  `json:"url"`
		Path      string  `json:"path"`
		Completed uint64  `json:"completed"`
		Total     *uint64 `json:"total"`
		Elapsed   float64 `json:"elapsed_seconds"`
		Error     string  `json:"error"`
	}
	if err := json.Unmarshal([]byte(*event.Detail), &data); err != nil {
		fmt.Fprintf(p.status, "Invalid native download event: %v\n", err)
		return
	}
	switch event.Stage {
	case "download_started":
		p.lastDownload = -5
		fmt.Fprintf(p.status, "Initiating download: %s\nModel destination: %s\n", data.URL, data.Path)
	case "download_progress":
		if data.Elapsed-p.lastDownload >= 5 {
			size := " (total size unknown)"
			if data.Total != nil {
				size = fmt.Sprintf(" / %.2f MiB", float64(*data.Total)/1048576)
			}
			fmt.Fprintf(p.status, "Downloading %s: %.2f MiB%s [%.1fs]\n", data.Model, float64(data.Completed)/1048576, size, data.Elapsed)
			p.lastDownload = data.Elapsed
		}
	case "download_verifying":
		fmt.Fprintf(p.status, "Verifying SHA-256: %s\n", data.Model)
	case "download_completed":
		fmt.Fprintf(p.status, "Successfully saved to: %s\n", data.Path)
	case "download_failed":
		fmt.Fprintf(p.status, "Download failed: %s: %s\n", data.Model, data.Error)
	}
}
func (p *consoleProgress) callback(custom Progress) Progress {
	return func(event Event) error {
		if err := p.update(event); err != nil {
			return err
		}
		if custom != nil {
			return custom(event)
		}
		return nil
	}
}
func (p *consoleProgress) finish(err error) {
	p.clear()
	if !p.enabled {
		return
	}
	if err != nil {
		fmt.Fprintf(p.status, "%s: failed: %v\n", p.source, err)
	} else {
		fmt.Fprintf(p.status, "%s: completed in %.2fs\n", p.source, time.Since(p.started).Seconds())
	}
}
