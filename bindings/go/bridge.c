//go:build (windows || linux) && cgo

#include "bridge.h"
#include "rapid_doc.h"
#include <stdlib.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

typedef rd_response *(*invoke)(uint64_t, uint32_t, const uint8_t *, size_t,
                              const uint8_t *, size_t, rd_progress, void *);
typedef void (*release)(rd_response *);
typedef struct {void *library; invoke call; release free;} api;
static _Thread_local char failure[512];
extern int32_t goProgress(void *, size_t, uintptr_t);
static int32_t notify(const uint8_t *data, size_t len, void *context) {
    return goProgress((void *)data, len, (uintptr_t)context);
}
static void *symbol(void *library, const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)library, name);
#else
    return dlsym(library, name);
#endif
}
static void unload(void *library) {
#ifdef _WIN32
    FreeLibrary((HMODULE)library);
#else
    dlclose(library);
#endif
}
void *rd_go_open(const char *path) {
    void *library;
#ifdef _WIN32
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    wchar_t *wide = count ? malloc(sizeof(wchar_t) * count) : NULL;
    if (!wide) {snprintf(failure, sizeof(failure), "Invalid UTF-8 path or allocation failed"); return NULL;}
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, count);
    library = LoadLibraryExW(wide, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    free(wide);
    if (!library) {snprintf(failure, sizeof(failure), "LoadLibraryExW failed, Windows error %lu", GetLastError()); return NULL;}
#else
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) {snprintf(failure, sizeof(failure), "%s", dlerror()); return NULL;}
#endif
    uint32_t (*version)(void) = (uint32_t (*)(void))symbol(library, "rd_abi_version");
    invoke call = (invoke)symbol(library, "rd_invoke");
    release free_response = (release)symbol(library, "rd_response_free");
    if (!version || !call || !free_response || version() != 1) {
        snprintf(failure, sizeof(failure), "Native library does not implement C ABI v1");
        unload(library); return NULL;
    }
    api *value = malloc(sizeof(api));
    if (!value) {snprintf(failure, sizeof(failure), "Cannot allocate native loader"); unload(library); return NULL;}
    *value = (api){library, call, free_response};
    return value;
}
const char *rd_go_error(void) {return failure;}
void *rd_go_call(void *value, uint64_t engine, uint32_t op, const void *input,
                size_t len, const void *options, size_t options_len, uintptr_t token) {
    return ((api *)value)->call(engine, op, input, len, options, options_len,
                               token ? notify : NULL, (void *)token);
}
void rd_go_free(void *value, void *response) {((api *)value)->free(response);}
int rd_go_terminal(void) {
#ifdef _WIN32
    return _isatty(_fileno(stderr));
#else
    return isatty(fileno(stderr));
#endif
}
_Static_assert(sizeof(void *) == 8, "Only 64-bit native packages are supported");
_Static_assert(sizeof(rd_bytes) == 16, "Unexpected C ABI bytes layout");
_Static_assert(sizeof(rd_asset) == 40, "Unexpected C ABI asset layout");
_Static_assert(sizeof(rd_response) == 48, "Unexpected C ABI response layout");
