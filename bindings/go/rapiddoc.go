// Package rapiddoc calls the shared native Rust document SDK through C ABI v1.
package rapiddoc

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"
)

type Options map[string]any
type Resources struct {
	BundleDirs      []string `json:"bundle_dirs,omitempty"`
	OrtLibrary      string   `json:"ort_library,omitempty"`
	PDFiumLibrary   string   `json:"pdfium_library,omitempty"`
	GeosLibrary     string   `json:"geos_library,omitempty"`
	OperatorLibrary string   `json:"operator_library,omitempty"`
	ModelDirectory  string   `json:"model_directory,omitempty"`
	AllowDownload   *bool    `json:"allow_download,omitempty"`
}
type Config struct {
	LibraryPath     string
	Options         Options
	Resources       Resources
	ProgressLogging *bool  // nil enables the default view; custom callbacks replace it.
	ProgressMode    string // auto, bar or plain; empty reads RAPIDDOC_PROGRESS_MODE, then auto.
}
type Event struct {
	Stage          string   `json:"stage"`
	PageIndex      *int64   `json:"page_index"`
	Completed      *int64   `json:"completed"`
	Total          *int64   `json:"total"`
	ElapsedSeconds *float64 `json:"elapsed_seconds"`
	Detail         *string  `json:"detail"`
}
type Progress func(Event) error
type NativeError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *NativeError) Error() string { return e.Code + ": " + e.Message }

type Result struct {
	SourcePageCount int
	Diagnostics     json.RawMessage
	RawJSON         json.RawMessage
	Markdown        string
	MarkdownValue   json.RawMessage
	MiddleJSON      json.RawMessage
	ContentListJSON json.RawMessage
	Metadata        json.RawMessage
	Images          map[string][]byte
	Drawings        map[string][]byte
	imageDirectory  string
	SourceName      string // Original filename stem; document for byte/reader input.
	ParseMethod     string // Requested PDF method, or office for Office formats.
}

// Engine shares native model sessions across concurrent requests. Close is idempotent.
// Synchronous parse reentry from a progress callback is rejected.
type Engine struct {
	api             *nativeAPI
	handle          atomic.Uint64
	progressLogging bool
	progressMode    string
	parseMethod     string
}
type nativeAPI struct {
	invoke func(uint64, uint32, []byte, []byte, uintptr) unsafe.Pointer
	free   func(unsafe.Pointer)
}
type rawBytes struct {
	data unsafe.Pointer
	len  uintptr
}
type rawAsset struct {
	name rawBytes
	data rawBytes
	kind uint32
}
type rawResponse struct {
	status int32
	json   rawBytes
	assets unsafe.Pointer
	count  uintptr
	owner  unsafe.Pointer
}
type callbackState struct {
	progress   Progress
	err        error
	panicValue any
}

var libraryCache = struct {
	sync.Mutex
	items map[string]*nativeAPI
}{items: make(map[string]*nativeAPI)}
var callbacks sync.Map
var callbackID atomic.Uint64

func copyBytes(raw rawBytes) []byte {
	if raw.len == 0 {
		return nil
	}
	return append([]byte(nil), unsafe.Slice((*byte)(raw.data), int(raw.len))...)
}
func notify(data unsafe.Pointer, length, token uintptr) (cancel uintptr) {
	value, ok := callbacks.Load(token)
	if !ok {
		return 1
	}
	state := value.(*callbackState)
	defer func() {
		if p := recover(); p != nil {
			state.panicValue = p
			cancel = 1
		}
	}()
	var event Event
	if err := json.Unmarshal(copyBytes(rawBytes{data, length}), &event); err != nil {
		state.err = err
		return 1
	}
	if err := state.progress(event); err != nil {
		state.err = err
		return 1
	}
	return 0
}

func New(config Config) (*Engine, error) {
	mode := config.ProgressMode
	if mode == "" {
		mode = os.Getenv("RAPIDDOC_PROGRESS_MODE")
	}
	if mode == "" {
		mode = "auto"
	}
	if mode != "auto" && mode != "bar" && mode != "plain" {
		return nil, errors.New("progress mode must be auto, bar or plain")
	}
	path := config.LibraryPath
	if path == "" {
		path = os.Getenv("RAPIDDOC_NATIVE_LIBRARY")
	}
	if path == "" {
		return nil, errors.New("set Config.LibraryPath or RAPIDDOC_NATIVE_LIBRARY to the native SDK library")
	}
	if strings.IndexByte(path, 0) >= 0 {
		return nil, errors.New("native library path contains NUL")
	}
	path, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if runtime.GOARCH != "amd64" {
		return nil, fmt.Errorf("native SDK currently packaged for amd64, not %s", runtime.GOARCH)
	}
	api, err := openLibrary(path)
	if err != nil {
		return nil, err
	}
	config.Resources.BundleDirs = append(config.Resources.BundleDirs, filepath.Dir(path))
	input, err := json.Marshal(config.Options)
	if err != nil {
		return nil, err
	}
	if string(input) == "null" {
		input = []byte("{}")
	}
	resources, err := json.Marshal(config.Resources)
	if err != nil {
		return nil, err
	}
	engine := &Engine{api: api, progressMode: mode, progressLogging: config.ProgressLogging == nil || *config.ProgressLogging}
	engine.parseMethod, _ = config.Options["parse_method"].(string)
	if engine.parseMethod == "" {
		engine.parseMethod = "auto"
	}
	data, _, _, err := engine.call(0, input, resources, nil)
	if err != nil {
		return nil, err
	}
	var result struct {
		Handle uint64 `json:"handle"`
	}
	if err = json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	engine.handle.Store(result.Handle)
	runtime.SetFinalizer(engine, func(e *Engine) { _ = e.Close() })
	return engine, nil
}

func (e *Engine) call(op uint32, input, options []byte, progress Progress) ([]byte, map[string][]byte, map[string][]byte, error) {
	handle := e.handle.Load()
	if op != 0 && op != 4 && handle == 0 {
		return nil, nil, nil, errors.New("RapidDoc is closed")
	}
	var token uintptr
	state := &callbackState{progress: progress}
	if progress != nil {
		token = uintptr(callbackID.Add(1))
		callbacks.Store(token, state)
		defer callbacks.Delete(token)
	}
	pointer := e.api.invoke(handle, op, input, options, token)
	runtime.KeepAlive(e)
	if pointer == nil {
		return nil, nil, nil, errors.New("native call returned no response")
	}
	defer e.api.free(pointer)
	if state.panicValue != nil {
		panic(state.panicValue)
	}
	if state.err != nil {
		return nil, nil, nil, state.err
	}
	response := (*rawResponse)(pointer)
	data := copyBytes(response.json)
	if response.status != 0 {
		var failure NativeError
		if err := json.Unmarshal(data, &failure); err != nil {
			return nil, nil, nil, err
		}
		return nil, nil, nil, &failure
	}
	images, drawings := map[string][]byte{}, map[string][]byte{}
	for _, asset := range unsafe.Slice((*rawAsset)(response.assets), int(response.count)) {
		name := string(copyBytes(asset.name))
		value := copyBytes(asset.data)
		if asset.kind == 0 {
			images[name] = value
		} else {
			drawings[name] = value
		}
	}
	return data, images, drawings, nil
}

func (e *Engine) Close() error {
	handle := e.handle.Swap(0)
	if handle == 0 {
		return nil
	}
	// Do not wait for an active callback; native calls retain their own lease.
	pointer := e.api.invoke(handle, 4, nil, nil, 0)
	runtime.KeepAlive(e)
	if pointer == nil {
		return errors.New("native close returned no response")
	}
	defer e.api.free(pointer)
	response := (*rawResponse)(pointer)
	if response.status != 0 {
		var err NativeError
		_ = json.Unmarshal(copyBytes(response.json), &err)
		return &err
	}
	return nil
}
func (e *Engine) parse(op uint32, input []byte, options Options, progress Progress) (result *Result, err error) {
	source := "bytes"
	if op == 1 {
		source = filepath.Base(string(input))
		if strings.HasPrefix(strings.ToLower(string(input)), "http://") || strings.HasPrefix(strings.ToLower(string(input)), "https://") {
			source = "URL document"
		}
	}
	view := newConsoleProgress(source, e.progressMode, e.progressLogging && progress == nil)
	defer func() {
		if p := recover(); p != nil {
			view.clear()
			panic(p)
		}
		view.finish(err)
	}()
	opts, err := json.Marshal(options)
	if err != nil {
		return nil, err
	}
	if string(opts) == "null" {
		opts = []byte("{}")
	}
	data, images, drawings, err := e.call(op, input, opts, view.callback(progress))
	if err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	if err = json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	result = &Result{
		RawJSON:         data,
		Diagnostics:     fields["diagnostics"],
		MarkdownValue:   fields["markdown"],
		MiddleJSON:      fields["middle_json"],
		ContentListJSON: fields["content_list_json"],
		Metadata:        fields["metadata"],
		Images:          images,
		Drawings:        drawings,
		imageDirectory:  "images",
	}
	_ = json.Unmarshal(result.MarkdownValue, &result.Markdown)
	_ = json.Unmarshal(fields["source_page_count"], &result.SourcePageCount)
	if name, ok := options["image_dir_name"].(string); ok {
		result.imageDirectory = name
	}
	result.SourceName = "document"
	if op == 1 {
		result.SourceName = source
		if dot := strings.LastIndex(source, "."); dot > 0 {
			result.SourceName = source[:dot]
		}
	}
	result.ParseMethod = e.parseMethod
	if method, ok := options["parse_method"].(string); ok {
		result.ParseMethod = method
	}
	var metadata struct {
		Format     string `json:"format"`
		SourceName string `json:"source_name"`
	}
	_ = json.Unmarshal(result.Metadata, &metadata)
	if metadata.SourceName != "" {
		result.SourceName = strings.TrimSuffix(metadata.SourceName, filepath.Ext(metadata.SourceName))
	}
	if metadata.Format == "office" {
		result.ParseMethod = "office"
	}
	return result, nil
}
func (e *Engine) Parse(path string, options Options, progress Progress) (*Result, error) {
	return e.parse(1, []byte(path), options, progress)
}
func (e *Engine) ParseBytes(data []byte, options Options, progress Progress) (*Result, error) {
	return e.parse(2, data, options, progress)
}

// ParseReader reads one document from the current position and never closes it.
func (e *Engine) ParseReader(reader io.Reader, options Options, progress Progress) (*Result, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	return e.ParseBytes(data, options, progress)
}
func (e *Engine) ParseAll(paths []string, options Options, progress Progress) ([]*Result, error) {
	results := make([]*Result, 0, len(paths))
	for _, path := range paths {
		r, err := e.Parse(path, options, progress)
		if err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, nil
}
func (e *Engine) Warmup(options Options, progress Progress) ([]string, error) {
	opts, err := json.Marshal(options)
	if err != nil {
		return nil, err
	}
	if string(opts) == "null" {
		opts = []byte("{}")
	}
	view := newConsoleProgress("warmup", e.progressMode, false)
	data, _, _, err := e.call(3, nil, opts, view.callback(progress))
	if err != nil {
		return nil, err
	}
	var roles []string
	err = json.Unmarshal(data, &roles)
	return roles, err
}
func (e *Engine) LoadedModels() ([]string, error) {
	data, _, _, err := e.call(5, nil, nil, nil)
	if err != nil {
		return nil, err
	}
	var roles []string
	err = json.Unmarshal(data, &roles)
	return roles, err
}

// OutputDirectory resolves root/source-name/method, matching Python output.
func (r *Result) OutputDirectory(directory string) string {
	name, method := r.SourceName, r.ParseMethod
	if name == "" {
		name = "document"
	}
	if method == "" {
		method = "auto"
	}
	return filepath.Join(directory, name, method)
}

// WriteTo persists results under root/source-name/method with filename prefixes.
func (r *Result) WriteTo(directory string) error {
	if strings.ContainsAny(r.SourceName, `/\:`) || r.SourceName == "." || r.SourceName == ".." || strings.ContainsAny(r.ParseMethod, `/\:`) || r.ParseMethod == "." || r.ParseMethod == ".." {
		return errors.New("invalid output name or method")
	}
	root, err := filepath.Abs(r.OutputDirectory(directory))
	if err != nil {
		return err
	}
	write := func(name string, data []byte) error {
		if filepath.IsAbs(name) || filepath.VolumeName(name) != "" {
			return errors.New("absolute asset path")
		}
		target := filepath.Join(root, name)
		relative, err := filepath.Rel(root, target)
		if err != nil || relative == ".." || len(relative) > 3 && relative[:3] == ".."+string(filepath.Separator) {
			return errors.New("asset path escapes output directory")
		}
		if err = os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return err
		}
		return os.WriteFile(target, data, 0644)
	}
	markdown := []byte(r.Markdown)
	if r.Markdown == "" && len(r.MarkdownValue) > 0 && r.MarkdownValue[0] != '"' {
		markdown = r.MarkdownValue
	}
	prefix := r.SourceName
	if prefix == "" {
		prefix = "document"
	}
	markdownName := prefix + ".md"
	if len(r.MarkdownValue) > 0 && r.MarkdownValue[0] != '"' {
		markdownName = prefix + "_markdown.json"
	}
	for name, data := range map[string][]byte{markdownName: markdown, prefix + "_middle.json": r.MiddleJSON, prefix + "_content_list.json": r.ContentListJSON} {
		if err = write(name, data); err != nil {
			return err
		}
	}
	for name, data := range r.Images {
		if err = write(filepath.Join(r.imageDirectory, name), data); err != nil {
			return err
		}
	}
	for name, data := range r.Drawings {
		if err = write(prefix+"_"+name+".pdf", data); err != nil {
			return err
		}
	}
	return nil
}
