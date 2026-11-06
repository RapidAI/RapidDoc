//go:build (windows || linux || darwin) && cgo

package rapiddoc

/*
#cgo linux LDFLAGS: -ldl
#include <stdlib.h>
#include "bridge.h"
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

func terminalOutput() bool { return C.rd_go_terminal() != 0 }

//export goProgress
func goProgress(data unsafe.Pointer, length C.size_t, token C.uintptr_t) C.int32_t {
	return C.int32_t(notify(data, uintptr(length), uintptr(token)))
}
func openLibrary(path string) (*nativeAPI, error) {
	libraryCache.Lock()
	defer libraryCache.Unlock()
	if api, ok := libraryCache.items[path]; ok {
		return api, nil
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	name := C.CString(path)
	defer C.free(unsafe.Pointer(name))
	library := C.rd_go_open(name)
	if library == nil {
		return nil, errors.New(C.GoString(C.rd_go_error()))
	}
	api := &nativeAPI{invoke: func(handle uint64, op uint32, input, opts []byte, token uintptr) unsafe.Pointer {
		var in, out unsafe.Pointer
		if len(input) > 0 {
			in = unsafe.Pointer(&input[0])
		}
		if len(opts) > 0 {
			out = unsafe.Pointer(&opts[0])
		}
		response := C.rd_go_call(library, C.uint64_t(handle), C.uint32_t(op), in, C.size_t(len(input)), out, C.size_t(len(opts)), C.uintptr_t(token))
		runtime.KeepAlive(input)
		runtime.KeepAlive(opts)
		return response
	}, free: func(pointer unsafe.Pointer) { C.rd_go_free(library, pointer) }}
	libraryCache.items[path] = api
	return api, nil
}
