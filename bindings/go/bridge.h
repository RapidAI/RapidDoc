#include <stdint.h>
#include <stddef.h>
void *rd_go_open(const char *path);
const char *rd_go_error(void);
void *rd_go_call(void *api, uint64_t engine, uint32_t op, const void *input,
                size_t len, const void *options, size_t options_len, uintptr_t token);
void rd_go_free(void *api, void *response);
int rd_go_terminal(void);
