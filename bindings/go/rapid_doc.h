#ifndef RAPID_DOC_H
#define RAPID_DOC_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { const uint8_t *data; size_t len; } rd_bytes;
typedef struct { rd_bytes name, data; uint32_t kind; } rd_asset;
typedef struct {
    int32_t status; /* 0 success, 1 error; error JSON has code/message */
    rd_bytes json;
    const rd_asset *assets;
    size_t asset_count;
    void *owner; /* private; do not modify */
} rd_response;
/* Synchronous borrowed UTF-8 JSON event. Nonzero cancels. Never retain pointers. */
typedef int32_t (*rd_progress)(const uint8_t *, size_t, void *);
uint32_t rd_abi_version(void);
/* operations: 0=create(config JSON, resources JSON), 1=path(UTF-8, options JSON),
 * 2=bytes(document, options JSON), 3=warmup(empty, options JSON), 4=close, 5=models.
 * JSON defaults to {} when empty. Create returns {"handle":uint64}.
 * Assets: kind 0=image, 1=drawing (name without .pdf).
 * All response pointers remain valid until exactly one rd_response_free.
 * Same engine supports concurrent calls; synchronous parse reentry from a callback is rejected.
 * Close is idempotent; an already-running call retains its own engine lease.
 * Caller must provide valid readable buffers and valid function pointers. */
rd_response *rd_invoke(uint64_t, uint32_t, const uint8_t *, size_t,
                       const uint8_t *, size_t, rd_progress, void *);
void rd_response_free(rd_response *);
#ifdef __cplusplus
}
#endif
#endif
