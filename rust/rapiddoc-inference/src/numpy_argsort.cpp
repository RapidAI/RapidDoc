// Native adapter to the unmodified NumPy 2.4.6 x86-simd-sort backend.
// Source revision and BSD-3-Clause license: ../vendor/x86-simd-sort/.
#include "x86simdsort-static-incl.h"
#ifdef _WIN32
#define RD_EXPORT __declspec(dllexport)
#else
#define RD_EXPORT
#endif
extern "C" RD_EXPORT int RAPIDDOC_SORT_NAME(double *values, size_t *indices, size_t count) noexcept {
    try {
        x86simdsortStatic::argsort(values, indices, count, true);
        return 0;
    } catch (...) {
        return 1;
    }
}
