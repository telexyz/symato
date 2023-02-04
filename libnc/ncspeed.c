/*
 * LibNC speed test
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
#include <sys/time.h>

#include "cutils.h"
#include "libnc.h"

typedef enum {
    OP_memcpy,
    OP_add,
    OP_mul,
    OP_sum,
    OP_reduce_sum_sqr,
    OP_sigmoid,
    OP_reduce_sum_col,
    OP_soft_max,
    OP_layer_norm,
    OP_convert_bf16,
    OP_rnd_unif,
    OP_rnd_dropout,
    OP_masked_fill,
    
    OP_count,
} NCTestOp;

#define MAX_ARGS 4

#define M_F32 (1 << NC_TYPE_F32)
#define M_BF16 (1 << NC_TYPE_BF16)

typedef struct {
    const char *name;
    int n_args;
    int ops_per_element;
    float mem_per_element;
    int type_mask;
} NCTestOPdef;

static NCTestOPdef test_op_def[] = {
    { "memcpy",          1, 1, 2, M_F32 | M_BF16 },
    { "add",             2, 1, 3, M_F32 | M_BF16 },
    { "mul",             2, 1, 3, M_F32 | M_BF16 },
    { "sum",             1, 1, 1, M_F32 | M_BF16 },
    { "reduce_sum_sqr",  1, 2, 1, M_F32 | M_BF16 },
    { "sigmoid",         1, 1, 2, M_F32 },
    { "reduce_sum_col",  2, 1, 1, M_F32 },
    { "soft_max",        1, 1, 2, M_F32 | M_BF16 },
    { "layer_norm",      1, 1, 2, M_F32 | M_BF16 },
    { "convert_bf16",    1, 2, (2 + 2 * 2) / 4.0, M_F32 },
    { "rnd_unif",        1, 1, 1, M_F32 | M_BF16 },
    { "rnd_dropout",     1, 1, 1, M_F32 | M_BF16 },
    { "masked_fill",     2, 1, 1, M_F32 | M_BF16 },
};

static int find_op(const char *name)
{
    int op;
    for(op = 0; op < countof(test_op_def); op++) {
        if (!strcmp(test_op_def[op].name, name))
            return op;
    }
    return -1;
}

static int64_t get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static NCTensor *exec_op(NCTestOp op, NCTensor **args, NCRNDState *rnd_state)
{
    NCTensor *res;
    switch(op) {
    case OP_memcpy:
        res = nc_new_tensor_from_tensor_nz(args[0]);
        nc_tensor_copy(res, args[0]);
        break;
    case OP_add:
        res = nc_add(nc_dup_tensor(args[0]),
                     nc_dup_tensor(args[1]));
        break;
    case OP_mul:
        res = nc_mul(nc_dup_tensor(args[0]),
                     nc_dup_tensor(args[1]));
        break;
    case OP_sum:
        res = nc_sum(nc_dup_tensor(args[0]));
        break;
    case OP_reduce_sum_sqr:
        res = nc_reduce_sum_sqr(nc_dup_tensor(args[0]));
        //        res = nc_sum(nc_mul(nc_dup_tensor(args[0]), nc_dup_tensor(args[0])));
        break;
    case OP_sigmoid:
        res = nc_sigmoid(nc_dup_tensor(args[0]));
        break;
    case OP_reduce_sum_col:
        res = nc_reduce_sum(nc_dup_tensor(args[0]),
                            nc_dup_tensor(args[1]), 1);
        break;
    case OP_soft_max:
        res = nc_soft_max(nc_dup_tensor(args[0]));
        break;
    case OP_layer_norm:
        res = nc_layer_norm(nc_dup_tensor(args[0]), 1e-5);
        break;
    case OP_convert_bf16:
        res = nc_convert(nc_convert(nc_dup_tensor(args[0]), NC_TYPE_BF16), NC_TYPE_F32);
        break;
    case OP_rnd_unif:
        res = nc_new_tensor_from_tensor_nz(args[0]);
        nc_tensor_set_rnd_unif(res, 0, 1.0, rnd_state);
        break;
    case OP_rnd_dropout:
        res = nc_new_tensor_from_tensor_nz(args[0]);
        nc_tensor_set_dropout(res, 0.1, rnd_state);
        break;
    case OP_masked_fill:
        res = nc_masked_fill(nc_dup_tensor(args[0]),
                             nc_dup_tensor(args[1]),
                             1.0, FALSE);
        break;
    default:
        abort();
    }
    return res;
}

/* compare the results between device and CPU */
static void check_op(NCContext *s, NCDevice *device,
                     NCTestOp op, NCTypeEnum type,
                     int n_args, NCTensor **args)
{
    NCTensor *cpu_args[MAX_ARGS], *cpu_res, *res;
    double err2, sum2, err_max, err_rel;
    float *a_ptr, *b_ptr;
    const size_t *dims;
    size_t n, i;
    int n_dims;
    NCRNDState *cpu_rnd_state = NULL, *dev_rnd_state = NULL;
    
    for(i = 0; i < n_args; i++) {
        cpu_args[i] = nc_tensor_to_cpu_device(nc_dup_tensor(args[i]));
    }
    if (op == OP_rnd_unif || op == OP_rnd_dropout) {
        cpu_rnd_state = nc_rnd_init(nc_new_cpu_device(s), 1234);
        dev_rnd_state = nc_rnd_init(device, 1234);
    }
    cpu_res = exec_op(op, cpu_args, cpu_rnd_state);
    res = exec_op(op, args, dev_rnd_state);
    if (op == OP_rnd_unif || op == OP_rnd_dropout) {
        nc_rnd_end(cpu_rnd_state);
        nc_rnd_end(dev_rnd_state);
    }
    
    if (type != NC_TYPE_F32) {
        cpu_res = nc_convert(cpu_res, NC_TYPE_F32);
        res = nc_convert(res, NC_TYPE_F32);
    }

    res = nc_tensor_to_cpu_device(res);

    for(i = 0; i < n_args; i++) {
        nc_free_tensor(cpu_args[i]);
    }
    
    /* sum of squares */
    dims = nc_tensor_get_dims(cpu_res, &n_dims);
    n = 1;
    for(i = 0; i < n_dims; i++)
        n *= dims[i];

    a_ptr = nc_tensor_get_ptr(res, NULL);
    b_ptr = nc_tensor_get_ptr(cpu_res, NULL);

    sum2 = 0;
    err2 = 0;
    err_max = 0;
    for(i = 0; i < n; i++) {
        float a, b, d;
        a = a_ptr[i];
        b = b_ptr[i];
        //        printf("%d: r=%e ref=%e\n", (int)i, a, b);
        d = fabsf(a - b);
        err2 += d * d;
        if (d > err_max)
            err_max = d;
        sum2 += b * b;
    }

    err_rel = err2;
    if (sum2 != 0.0)
        err_rel /= sum2;
    
    printf(" %7.1e %7.1e", sqrt(err_rel), err_max);
    fflush(stdout);
    
    nc_free_tensor(res);
    nc_free_tensor(cpu_res);
}


static BOOL header_disp;

static void nc_test_op_type(NCContext *s, NCDevice *d, NCTestOp op,
                            NCTypeEnum arg_type,
                            int n, int w)
{
    NCTensor *args[MAX_ARGS], *res;
    int i, n_args, it, nb_its;
    int64_t ti, ti1;
    NCRNDState *rnd_state;
    double gop_per_sec;
    
    if (!header_disp) {
        header_disp = TRUE;
        printf("N=%d W=%d\n", n, w);
        printf("%20s %4s %7s %7s %10s %10s\n",
               "OP", "TYPE", "RMS", "MAX", "Gop/s", "GB/s");
    }
    
    rnd_state = nc_rnd_init(d, 1234);
    
    printf("%20s %4s", test_op_def[op].name, nc_type_name_table[arg_type]);
    fflush(stdout);
    
    n_args = test_op_def[op].n_args;
    
    switch(op) {
    case OP_reduce_sum_col:
        args[0] = nc_new_tensor_1d(d, NC_TYPE_F32, w);
        args[1] = nc_new_tensor_2d(d, NC_TYPE_F32, w, (n + w - 1) / w);
        break;
    case OP_soft_max:
    case OP_layer_norm:
        args[0] = nc_new_tensor_2d(d, NC_TYPE_F32, w, (n + w - 1) / w);
        break;
    case OP_masked_fill:
        {
            NCTensor *x1;
            int32_t *tab;
            RNDState rnd_state2;
            rnd_init(&rnd_state2, 1235);
            args[0] = nc_new_tensor_1d(d, NC_TYPE_F32, n);
            x1 = nc_new_tensor_1d(nc_new_cpu_device(s), NC_TYPE_I32, n);
            tab = nc_tensor_get_ptr(x1, NULL);
            for(i = 0; i < n; i++) {
                tab[i] = rnd_unif_u32(&rnd_state2) & 1;
            }
            args[1] = nc_tensor_to_device(x1, d);
        }
        break;
    default:
        for(i = 0; i < n_args; i++) {
            args[i] = nc_new_tensor_1d(d, NC_TYPE_F32, n);
        }
        break;
    }
    
    for(i = 0; i < n_args; i++) {
        if (nc_tensor_get_item_type(args[i]) == NC_TYPE_F32) {
            nc_tensor_set_rnd_unif(args[i], 0, 1, rnd_state);
            if (arg_type != NC_TYPE_F32) {
                args[i] = nc_convert(args[i], arg_type);
            }
        }
    }
    
    check_op(s, d, op, arg_type, n_args, args);
    
    nb_its = 1;
    for(;;) {
        ti1 = get_time_ns();
        ti = 0;
        for(it = 0; it < nb_its; it++) {
            ti -= get_time_ns();
            res = exec_op(op, args, rnd_state);
            nc_synchronize(d);
            ti += get_time_ns();
            nc_free_tensor(res);
        }
        ti1 = get_time_ns() - ti1;
        if (ti1 >= 100000000)
            break;
        nb_its *= 2;
    }
    
    for(i = 0; i < n_args; i++) {
        nc_free_tensor(args[i]);
    }
    nc_rnd_end(rnd_state);
    
    gop_per_sec = (double)nb_its * (double)n / (double)ti;
    
    printf(" %10.1f %10.1f\n",
           gop_per_sec * test_op_def[op].ops_per_element,
           gop_per_sec * test_op_def[op].mem_per_element *
           nc_type_size_table[arg_type]);
}

static void nc_test_op(NCContext *s, NCDevice *d, NCTestOp op,
                       int n, int w)
{
    int type;
    for(type = 0; type < NC_TYPE_COUNT; type++) {
        if (test_op_def[op].type_mask & (1 << type)) {
            nc_test_op_type(s, d, op, type, n, w);
        }
    }
}

void help(void)
{
    printf("usage: nc2speed [options] [operation]...\n"
           "\n"
           "Options:\n"
           "-h                 help\n"
           "-d device          select the compute device (cpu or cuda)\n"
           );
    exit(1);
}

int main(int argc, char **argv)
{
    int c, i, op, n, w;
    const char *device_name;
    NCContext *s;
    NCDevice *dev;
    
    device_name = "cpu";
    n = 1000000;
    w = 1024;
    for(;;) {
        c = getopt(argc, argv, "hd:vn:w:");
        if (c == -1)
            break;
        switch(c) {
        case 'h':
            help();
        case 'd':
            device_name = optarg;
            break;
        case 'n':
            n = (int)strtod(optarg, NULL);
            break;
        case 'w':
            w = (int)strtod(optarg, NULL);
            break;
        default:
            exit(1);
        }
    }

    s = nc_context_init(1);
    dev = nc_new_device(s, device_name);
    if (!dev) {
        fprintf(stderr, "Device %s is not available\n", device_name);
        exit(1);
    }
    
    if (optind >= argc) {
        for(op = 0; op < OP_count; op++) {
            nc_test_op(s, dev, op, n, w);
        }
    } else {
        for(i = optind; i < argc; i++) {
            op = find_op(argv[i]);
            if (op < 0) {
                fprintf(stderr, "unknown operation: %s\n", argv[i]);
                exit(1);
            }
            nc_test_op(s, dev, op, n, w);
        }
    }
    nc_context_end(s);
    return 0;
}
