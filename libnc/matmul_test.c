/*
 * LibNC matrix multplication benchmark
 * 
 * Copyright (c) 2018-2021 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>
#include <getopt.h>
#include <stdarg.h>
#include <ctype.h>

#include "cutils.h"
#include "libnc.h"

#define K_INCR 128

static no_inline void mat_mul_ref(float *tab_c, size_t c_stride,
                                  const float *tab_a, size_t a_stride,
                                  const float *tab_b, size_t b_stride,
                                  BOOL a_trans, BOOL b_trans, int m, int n, int k)
{
    double sum, a, b;
    int i, j, l, l0, l1;

    for(i = 0; i < m; i++) {
        for(j = 0; j < n; j++) {
            for(l0 = 0; l0 < k; l0 += K_INCR) {
                l1 = min_int(k, l0 + K_INCR);
                sum = 0;
                for(l = l0; l < l1; l++) {
                    if (a_trans)
                        a = tab_a[i * a_stride + l];
                    else
                        a = tab_a[i + l * a_stride];
                    if (b_trans)
                        b = tab_b[j + b_stride * l];
                    else
                        b = tab_b[l + b_stride * j];
                    sum = fmaf(a, b, sum);
                }
                tab_c[i + j * c_stride] += sum;
            }
        }
    }
}

/* new matrix of 'm' rows and 'n' columns */
static NCTensor *init_mat(NCDevice *d, NCTypeEnum type,
                          int m, int n, NCRNDState *rnd_state)
{
    NCTensor *tab_a;
    tab_a = nc_new_tensor_2d(d, type, n, m);
    nc_tensor_set_rnd_unif(tab_a, 0, 1, rnd_state);
    return tab_a;
}

static int64_t get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

typedef struct {
    double flop_per_cycle;
    double Gflops;
} MatMulResult;

static void mat_profile(NCContext *s, NCDevice *d,
                        MatMulResult *res,
                        BOOL a_trans, BOOL b_trans, int m, int n, int k,
                        int mat_count, NCTypeEnum type,
                        BOOL check_result)
{
    NCTensor *tab_a, *tab_b, *tab_c;
    int it, nb_its;
    NCRNDState *rnd_state;
    int64_t ti, ti1, flops;

    assert(mat_count == 1);
    if (mat_count > 1) {
        assert(!a_trans);
    }
    
    rnd_state = nc_rnd_init(d, 123);
    if (a_trans)
        tab_a = init_mat(d, type, m, k, rnd_state);
    else
        tab_a = init_mat(d, type, k * mat_count, m, rnd_state);

    if (b_trans)
        tab_b = init_mat(d, type, k, n, rnd_state);
    else
        tab_b = init_mat(d, type, n, k, rnd_state);
    
    tab_c = nc_new_tensor_2d(d, type, m, n);

#if 0
    tab_a_f16 = NULL;
    if (type == MAT_MUL_TYPE_F16) {
        tab_a_f16 = nc_malloc(sizeof(nc_float16_t) *
                              a_stride * (k * mat_count));
        convert_f32_to_f16(tab_a_f16, tab_a, a_stride * (k * mat_count));
    }
#endif

    /* check the result */
    if (check_result) {
        int i, j;
        float a, b, e, err;
        double err2, sum2, threshold;
        float *tab_a1, *tab_b1, *tab_c1, *tab_c_ref1;
        size_t a_stride, b_stride, c_stride;
        NCTensor *tab_a_cpu, *tab_b_cpu, *tab_c_cpu, *tab_c_ref;
        
        tab_c_ref = init_mat(d, type, n, m, rnd_state);
        nc_tensor_copy(tab_c, tab_c_ref);

        tab_c = nc_matmul_add(nc_dup_tensor(tab_a), nc_dup_tensor(tab_b),
                              tab_c, a_trans, b_trans);

        switch(type) {
        case NC_TYPE_F32:
            threshold = 2e-5;
            break;
        case NC_TYPE_BF16:
            threshold = 2e-2;
            break;
        default:
            abort();
        }

        tab_a_cpu = nc_convert(nc_tensor_to_cpu_device(nc_dup_tensor(tab_a)),
                               NC_TYPE_F32);
        tab_b_cpu = nc_convert(nc_tensor_to_cpu_device(nc_dup_tensor(tab_b)),
                               NC_TYPE_F32);
        tab_c_cpu = nc_convert(nc_tensor_to_cpu_device(nc_dup_tensor(tab_c)),
                               NC_TYPE_F32);
        tab_c_ref = nc_convert(nc_tensor_to_cpu_device(tab_c_ref),
                               NC_TYPE_F32);
        
        tab_a1 = nc_tensor_get_ptr(tab_a_cpu, &a_stride);
        tab_b1 = nc_tensor_get_ptr(tab_b_cpu, &b_stride);
        tab_c1 = nc_tensor_get_ptr(tab_c_cpu, &c_stride);
        tab_c_ref1 = nc_tensor_get_ptr(tab_c_ref, &c_stride);
        
        mat_mul_ref(tab_c_ref1, c_stride,
                    tab_a1, a_stride,
                    tab_b1, b_stride,
                    a_trans, b_trans, m, n, k);
        err2 = 0;
        sum2 = 0;
        err = 0;
        for(i = 0; i < m; i++) {
            for(j = 0; j < n; j++) {
                a = tab_c1[i + j * c_stride];
                b = tab_c_ref1[i + j * c_stride];
                e = a - b;
                sum2 += b * b;
                err2 += e * e;
                e = fabsf(e);
                if (e > err)
                    err = e;
            }
        }
        err2 = err2 / sum2;
        err = err / sqrt(sum2 / (m * n));
        printf("RMS=%0.2e max=%0.2e\n", sqrt(err2), err);
        if (err > threshold) {
            printf("ERROR\n");
            exit(1);
        }
        nc_free_tensor(tab_a_cpu);
        nc_free_tensor(tab_b_cpu);
        nc_free_tensor(tab_c_cpu);
        nc_free_tensor(tab_c_ref);
    }

    nb_its = 1;
    for(;;) {
        nc_tensor_set_zero(tab_c);
        ti1 = get_time_ns();
        ti = get_cycles();
        for(it = 0; it < nb_its; it++) {
            tab_c = nc_matmul_add(nc_dup_tensor(tab_a), nc_dup_tensor(tab_b),
                                  tab_c, a_trans, b_trans);
        }
        ti = get_cycles() - ti;
        ti1 = get_time_ns() - ti1;
        if (ti1 >= 1000000000)
            break;
        nb_its *= 2;
    }

    flops = (int64_t)m * n * k * 2 * mat_count * nb_its;
    res->flop_per_cycle = (double)flops / (double)ti;
    res->Gflops = (double)flops / (double)ti1;
    nc_free_tensor(tab_a);
    nc_free_tensor(tab_b);
    nc_free_tensor(tab_c);
    nc_rnd_end(rnd_state);
}

typedef struct {
    BOOL a_trans;
    BOOL b_trans;
    int m, n, k;
    int mat_count;
    NCTypeEnum type;    
} MatMulParams;

static const MatMulParams matmul_params[] = {
    { 0,  0, 16388,   320,  2560, 1, NC_TYPE_F32 },
    { 1,  0,  2560,   320, 16388, 1, NC_TYPE_F32 },
    { 0,  1, 16388,  2560,   320, 1, NC_TYPE_F32 },
    { 0,  0,  2048,    16,   512, 1, NC_TYPE_F32 },
    { 1,  0,   512,    16,  2048, 1, NC_TYPE_F32 },
    { 0,  1,  2048,   512,   320, 1, NC_TYPE_F32 },
    { 0,  0,  2048,   320,   512, 1, NC_TYPE_F32 },
    { 1,  0,   512,   320,  2048, 1, NC_TYPE_F32 },

    
    { 0,  0, 16388,   320,  1760, 1, NC_TYPE_F32 },
    { 1,  0,  1760,   320, 16388, 1, NC_TYPE_F32 },
    { 0,  1, 16388,  1760,   320, 1, NC_TYPE_F32 },
    { 0,  0,  1408,    16,   352, 1, NC_TYPE_F32 },
    { 1,  0,   352,    16,  1408, 1, NC_TYPE_F32 },
    { 0,  1,  1408,   352,   320, 1, NC_TYPE_F32 },
    { 1,  0,   352,   320,  1408, 1, NC_TYPE_F32 },
    { 0,  0,  1408,   320,   352, 1, NC_TYPE_F32 },
};

typedef struct {
    MatMulParams p;
    MatMulResult r;
} MatMulEntry;

int load_results(MatMulEntry **pr_tab, const char *filename)
{
    FILE *f;
    MatMulEntry *r_tab, *r;
    int r_count, r_size, type;
    char line[1024], *p;
    
    f = fopen(filename, "r");
    if (!f) {
        *pr_tab = NULL;
        return 0;
    }
    r_tab = NULL;
    r_count = 0;
    r_size = 0;
    for(;;) {
        if (fgets(line, sizeof(line), f) == NULL)
            break;
        p = line;
        while (isspace(*p))
            p++;
        if (*p == '#' || *p == '\0')
            continue;
        if ((r_count + 1) > r_size) {
            r_size = max_int(r_count + 1, max_int(4, r_size * 3 / 2));
            r_tab = realloc(r_tab, sizeof(r_tab[0]) * r_size);
        }
        r = &r_tab[r_count++];
        sscanf(p, "%d %d %d %d %d %d %d %lf %lf",
               &type,
               &r->p.a_trans,
               &r->p.b_trans,
               &r->p.m,
               &r->p.n,
               &r->p.k,
               &r->p.mat_count,
               &r->r.flop_per_cycle,
               &r->r.Gflops);
        r->p.type = type;
    }
    fclose(f);
    *pr_tab = r_tab;
    return r_count;
}

const MatMulEntry *find_entry(const MatMulEntry *r_tab, int r_count,
                              const MatMulParams *p)
{
    const MatMulParams *r;
    int i;
    
    for(i = 0; i < r_count; i++) {
        r = &r_tab[i].p;
        if (r->a_trans == p->a_trans &&
            r->b_trans == p->b_trans &&
            r->m == p->m &&
            r->n == p->n &&
            r->k == p->k) {
            return &r_tab[i];
        }
    }
    return NULL;
}

/* we print at least 3 significant digits with at most 5 chars, except
   if larger than 9999T. The value is rounded to zero. */
char *get_si_prefix(char *buf, int buf_size, uint64_t val)
{
    static const char suffixes[4] = "kMGT";
    uint64_t base;
    int i;

    if (val <= 999) {
        snprintf(buf, buf_size, "%" PRId64, val);
    } else {
        base = 1000;
        for(i=0;i<4;i++) {
            /* Note: we round to 0 */
            if (val < base * 10) {
                snprintf(buf, buf_size, "%0.2f%c", 
                         floor((val * 100.0) / base) / 100.0,
                         suffixes[i]);
                break;
            } else if (val < base * 100) {
                snprintf(buf, buf_size, "%0.1f%c", 
                         floor((val * 10.0) / base) / 10.0,
                         suffixes[i]);
                break;
            } else if (val < base * 1000 || (i == 3)) {
                snprintf(buf, buf_size,
                         "%" PRId64 "%c", 
                         val / base,
                         suffixes[i]);
                break;
            }
            base = base * 1000;
        }
    }
    return buf;
}

static void dump_header(BOOL check_result)
{
    printf("%4s %2s %2s %5s %5s %5s %5s %5s %5s\n",
           "TYPE", "TA", "TB", "M", "N", "K", "COUNT", "Flops", "REF");
}

void bench(BOOL check_result, const char *save_filename,
           const char *ref_filename,
           const MatMulParams *params_tab, int params_count,
           const char *device_name, int nb_threads)
{
    const MatMulParams *mp;
    int i;
    FILE *f;
    MatMulEntry *r_tab;
    const MatMulEntry *ref;
    int r_count, color;
    MatMulResult res;
    NCContext *s;
    NCDevice *d;
    char buf[64];
    
    if (ref_filename) {
        r_count = load_results(&r_tab, ref_filename);
    } else {
        r_count = 0;
        r_tab = NULL;
    }
    
    if (save_filename) {
        f = fopen(save_filename, "w");
        if (!f) {
            perror(save_filename);
            exit(1);
        }
    } else {
        f = NULL;
    }

    s = nc_context_init(nb_threads);
    d = nc_new_device(s, device_name);
    if (!d) {
        fprintf(stderr, "Device %s is not available\n", device_name);
        exit(1);
    }

    dump_header(check_result);
    for(i = 0; i < params_count; i++) {
        mp = &params_tab[i];
        mat_profile(s, d, &res, mp->a_trans, mp->b_trans, mp->m, mp->n, mp->k,
                    mp->mat_count, mp->type, check_result);
        if (f) {
            fprintf(f, "%4d %2d %2d %5d %5d %5d %5d %7.2f %7.2f\n",
                    mp->type,
                    mp->a_trans, mp->b_trans, mp->m, mp->n, mp->k,
                    mp->mat_count,
                    res.flop_per_cycle,
                    res.Gflops);
        }
        printf("%4s %2d %2d %5d %5d %5d %5d",
               nc_type_name_table[mp->type],
               mp->a_trans, mp->b_trans, mp->m, mp->n, mp->k, mp->mat_count);
        ref = find_entry(r_tab, r_count, mp);
        if (ref && res.Gflops < ref->r.Gflops * 0.95) {
            color = 31;
        } else if (ref && res.Gflops > ref->r.Gflops * 1.05) {
            color = 37;
        } else {
            color = 0;
        }
        if (color != 0) {
            printf("\x1b[%d;1m", color);
        }
        printf(" %5s", get_si_prefix(buf, sizeof(buf),
                                     (int64_t)(res.Gflops * 1e9)));
        if (color != 0) {
            printf("\x1b[0m");
        }

        if (ref) {
            printf(" %5s", get_si_prefix(buf, sizeof(buf),
                                         (int64_t)(ref->r.Gflops * 1e9))); 
        }
        printf("\n");
    }
    if (f) {
        fclose(f);
    }

    free(r_tab);
    nc_context_end(s);
}


int main(int argc, char **argv)
{
    int nb_threads, c;
    BOOL check_result;
    const char *ref_filename, *device_name;
    
    nb_threads = 1;
    check_result = FALSE;
    ref_filename = "matmul_test.ref";
    device_name = "cpu";
    for(;;) {
        c = getopt(argc, argv, "T:hcr:d:");
        if (c == -1)
            break;
        switch(c) {
        case 'h':
            printf("usage: matmul_test [options] [a_trans b_trans m n k count type]\n"
                   "\n"
                   "Options:\n"
                   "-h                 help\n"
                   "-d device          select the compute device (cpu or cuda)\n"
                   "-T n_threads       set the number of threads\n"
                   "-c                 check the result in full test\n"
                   "-r ref_filename\n"
                   );
            exit(1);
        case 'T':
            nb_threads = atoi(optarg);
            break;
        case 'c':
            check_result = TRUE;
            break;
        case 'r':
            ref_filename = optarg;
            break;
        case 'd':
            device_name = optarg;
            break;
        default:
            exit(1);
        }
    }
    if ((argc - optind) < 5) {
        bench(check_result, "matmul_test.txt", ref_filename,
              matmul_params, countof(matmul_params),
              device_name, nb_threads);
    } else {
        MatMulParams mp_s, *mp = &mp_s;
        mp->a_trans = atoi(argv[optind]);
        mp->b_trans = atoi(argv[optind + 1]);
        mp->m = atoi(argv[optind + 2]);
        mp->n = atoi(argv[optind + 3]);
        mp->k = atoi(argv[optind + 4]);
        mp->mat_count = 1;
        mp->type = NC_TYPE_F32;
        if ((optind + 5) < argc)
            mp->mat_count = atoi(argv[optind + 5]);
        if ((optind + 6) < argc) {
            const char *type_str = argv[optind + 6];
            if (!strcmp(type_str, "f32")) {
                mp->type = NC_TYPE_F32;
            } else if (!strcmp(type_str, "bf16")) {
                mp->type = NC_TYPE_BF16;
            } else {
                fprintf(stderr, "unsupported type: %s\n", type_str);
                exit(1);
            }
        }
        bench(check_result, NULL, ref_filename, mp, 1, device_name, nb_threads);
    }
    return 0;
}
