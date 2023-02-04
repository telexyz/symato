#ifndef CUTILS_H
#define CUTILS_H

#include <inttypes.h>

#define force_inline inline __attribute__((always_inline))
#define no_inline __attribute__((noinline))
#define __unused __attribute__((unused))
#define xglue(x, y) x ## y
#define glue(x, y) xglue(x, y)
#ifndef offsetof
#define offsetof(type, field) ((size_t) &((type *)0)->field)
#endif
#define countof(x) (sizeof(x) / sizeof(x[0]))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

typedef int BOOL;

#ifndef FALSE
enum {
    FALSE = 0,
    TRUE = 1,
};
#endif

typedef struct {
    uint16_t u16;
} bfloat16_t;

#if defined(__x86_64__)
static inline int64_t get_cycles(void)
{
    uint32_t low,high;
    int64_t val;
    asm volatile("rdtsc" : "=a" (low), "=d" (high));
    val = high;
    val <<= 32;
    val |= low;
    return val;
}
#else
static inline int64_t get_cycles(void)
{
    int64_t val;
    asm volatile ("rdtsc" : "=A" (val));
    return val;
}
#endif

static inline int max_int(int a, int b)
{
    if (a > b)
        return a;
    else
        return b;
}

static inline int min_int(int a, int b)
{
    if (a < b)
        return a;
    else
        return b;
}

static inline size_t max_size_t(size_t a, size_t b)
{
    if (a > b)
        return a;
    else
        return b;
}

static inline size_t min_size_t(size_t a, size_t b)
{
    if (a < b)
        return a;
    else
        return b;
}

static inline ssize_t max_ssize_t(ssize_t a, ssize_t b)
{
    if (a > b)
        return a;
    else
        return b;
}

static inline ssize_t min_ssize_t(ssize_t a, ssize_t b)
{
    if (a < b)
        return a;
    else
        return b;
}

static inline int clamp_int(int val, int min_val, int max_val)
{
    if (val < min_val)
        return min_val;
    else if (val > max_val)
        return max_val;
    else
        return val;
}

static inline float clamp_float(float val, float min_val, float max_val)
{
    if (val < min_val)
        return min_val;
    else if (val > max_val)
        return max_val;
    else
        return val;
}

/* WARNING: undefined if a = 0 */
static inline int clz32(unsigned int a)
{
    return __builtin_clz(a);
}

/* WARNING: undefined if a = 0 */
static inline int clz64(uint64_t a)
{
    return __builtin_clzll(a);
}

static inline int floor_log2(uint64_t a)
{
    return 63 - clz64(a);
}

static inline int ceil_log2(uint64_t a)
{
    if (a <= 1)
        return 0;
    else
        return 64 - clz64(a - 1);
}

static inline float squaref(float x)
{
    return x * x;
}

#define DUP8(a) a, a, a, a, a, a, a, a

#endif /* CUTILS_H */

