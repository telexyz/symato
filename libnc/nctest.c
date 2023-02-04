/*
 * LibNC test
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

static void backward_save_grad(void *opaque, NCTensor *g,
                               NCTensor *get_col_index)
{
    NCParam *p = opaque;
    char buf[64];
    
    if (0) {
        snprintf(buf, sizeof(buf), "dy/d%s", p->name);
        nc_dump_tensor(buf, g, 0);
    }
    p->saved_grad = g;
}

static __unused void nc_test_backward_scalar(NCContext *s, NCDevice *d,
                                             BOOL test_hessian_product)
{
    NCParamList param_list;
    NCTensor *y, *x1, *x2;
    NCParam *param_x1, *param_x2;
    int i, flags;
    
    nc_param_list_init(&param_list);
    x1 = nc_new_f32(d, 0.0);
    param_x1 = nc_new_param(&param_list, &x1, "x1");
    x2 = nc_new_f32(d, 0.0);
    param_x2 = nc_new_param(&param_list, &x2, "x2");

    for(i = 0; i < 2; i++) {
        nc_tensor_set_f32(x1, (float)i + 3);
        nc_tensor_set_f32(x2, (float)i + 1);
        //        nc_dump_tensor("x1", x1, 0);
        
        y = nc_mul(nc_dup_tensor(x1), nc_dup_tensor(x1));
        y = nc_add(y, nc_mul(nc_dup_tensor(x1), nc_dup_tensor(x2)));
        //        nc_dump_tensor("y", y, 0);

        //        nc_dump_graph(y);

        if (test_hessian_product)
            flags = NC_BW_KEEP_GRAD_GRAPH;
        else
            flags = 0;
        nc_backward(y, nc_new_f32(d, 1.0), backward_save_grad, flags);
        
        /* y = x1^2 + x1 * x2

           Gradient:
           dy/dx1 = 2*x1 + x2
           dy/dx2 = x1

           Hessian:
           dy/dx1^2 = 2
           dy/dx2^2 = 0
           dy/dx1dx2 = 1
           H = [ 2 , 1 ]
               [ 1 , 0 ]
               
           Hessian vector product:
           v = [ 1, 1 ]
           Hv = [ 3, 1 ]
        */
            
        /* check the gradient */
        {
            float g1, expected_g1;
            g1 = nc_get_scalar_f32(param_x1->saved_grad);
            expected_g1 = 2 * nc_get_scalar_f32(x1) +
                nc_get_scalar_f32(x2);
            if (g1 != expected_g1) {
                printf("ERROR: g1=%g expected=%g\n", g1, expected_g1);
                exit(1);
            }
            
            g1 = nc_get_scalar_f32(param_x2->saved_grad);
            expected_g1 = nc_get_scalar_f32(x1);
            if (g1 != expected_g1) {
                printf("ERROR: g2=%g expected=%g\n", g1, expected_g1);
                exit(1);
            }
        }

        if (test_hessian_product) {
            NCTensor *g, *tab[2], *v;
            
            /* build the gradient vector: g = [ dy/dx1 dy/dx2 ] */
            tab[0] = nc_reshape_1d(param_x1->saved_grad, 1);
            tab[1] = nc_reshape_1d(param_x2->saved_grad, 1);
            g = nc_vconcat(tab, 2);
            
            //            nc_dump_graph(g);
            
            /* compute the hessian vector product */
            v = nc_new_tensor_1d(d, NC_TYPE_F32, 2);
            nc_tensor_set_f32(v, 1.0);
            
            nc_backward(g, v, backward_save_grad, 0);

            nc_free_tensor(g);

            /* check the result */
            {
                float hv1, expected_hv1;
                
                hv1 = nc_get_scalar_f32(param_x1->saved_grad);
                expected_hv1 = 3;
                if (hv1 != expected_hv1) {
                    printf("ERROR: hv1=%g expected=%f\n", hv1, expected_hv1);
                    exit(1);
                }
                hv1 = nc_get_scalar_f32(param_x2->saved_grad);
                expected_hv1 = 1;
                if (hv1 != expected_hv1) {
                    printf("ERROR: hv2=%g expected=%f\n", hv1, expected_hv1);
                    exit(1);
                }
            }
        }
        nc_free_tensor(param_x1->saved_grad);
        nc_free_tensor(param_x2->saved_grad);
        param_x1->saved_grad = NULL;
        param_x2->saved_grad = NULL;
        
        nc_free_tensor(y);
    }
    nc_param_list_end(&param_list);
}

typedef NCTensor *GradientTestFunc(void *opaque);

/* 'eps' : epsilon used to compute the gradient numerically.
   'error_max': maximum absolute error. 
   'rel_error_max' : maximum relative error. */
static void test_approx_gradient(NCContext *s, GradientTestFunc *func,
                                 void *opaque,
                                 NCParamList *param_list,
                                 float eps, float error_max,
                                 float rel_error_max,
                                 BOOL verbose, BOOL use_bf16)
{
    struct list_head *el, *el1;
    NCParam *p, *p1;
    NCTensor *x, *loss, *grad0;
    NCTensorData *d, dbuf;
    size_t n_params, param_idx, tab_pos[NC_N_DIMS_MAX], n;
    int i;
    BOOL header_output = FALSE;
    BOOL is_error;
    
    list_for_each(el, &param_list->param_list) {
        p = list_entry(el, NCParam, link);
        x = *p->pval;
        assert(nc_tensor_get_item_type(x) == NC_TYPE_F32);
        d = nc_tensor_get_data(&dbuf, x);
        
        n_params = 1;
        for(i = 0; i < d->n_dims; i++)
            n_params *= d->dims[i];

        for(param_idx = 0; param_idx < n_params; param_idx++) {
            float y0, y1, x0, g_approx, g, error, rel_error;
            
            /* build an index */
            n = param_idx;
            for(i = 0; i < d->n_dims; i++) {
                tab_pos[i] = n % d->dims[i];
                n /= d->dims[i];
            }

            x0 = nc_get1_f32(x, d->n_dims, tab_pos);

            /* approx gradient = (f(x0+eps)-f(x0-eps))/(2*eps) */
            nc_set1_f32(x, d->n_dims, tab_pos, x0 + eps);
            loss = func(opaque);
            y1 = nc_get_scalar_f32(loss);
            nc_free_tensor(loss);
            
            nc_set1_f32(x, d->n_dims, tab_pos, x0 - eps);
            loss = func(opaque);
            y0 = nc_get_scalar_f32(loss);
            nc_free_tensor(loss);
            
            g_approx = (y1 - y0) / (2 * eps);

            /* compute the gradient using back propagation */
            nc_set1_f32(x, d->n_dims, tab_pos, x0);

            if (use_bf16) {
                /* if BF16 test, compute the gradient using BF16 precision */
                list_for_each(el1, &param_list->param_list) {
                    p1 = list_entry(el1, NCParam, link);
                    /* save the original parameter in 'low_part' */
                    p1->low_part = nc_dup_tensor(*p1->pval);
                    *p1->pval = nc_convert(*p1->pval, NC_TYPE_BF16);
                }
            }
            
            loss = func(opaque);
            grad0 = nc_new_f32(nc_get_tensor_device(loss), 1.0);
            if (use_bf16) {
                grad0 = nc_convert(grad0, NC_TYPE_BF16);
            }
            nc_backward(loss, grad0, backward_save_grad, 0);
            nc_free_tensor(loss);
            
            g = nc_get1_f32(p->saved_grad, d->n_dims, tab_pos);

            if (use_bf16) {
                /* restore the original parameters */
                list_for_each(el1, &param_list->param_list) {
                    p1 = list_entry(el1, NCParam, link);
                    nc_free_tensor(*p1->pval);
                    *p1->pval = p1->low_part;
                    p1->low_part = NULL;
                }
            }

            /* free the computed gradients */
            list_for_each(el1, &param_list->param_list) {
                p1 = list_entry(el1, NCParam, link);
                nc_free_tensor(p1->saved_grad);
                p1->saved_grad = NULL;
            }
            error = fabs(g_approx - g);
            if (g != 0) {
                rel_error = fabsf((g_approx - g) / g);
            } else {
                rel_error = 0;
            }
            is_error = (error > error_max || rel_error > rel_error_max);
            if (verbose || is_error) {
                if (!header_output) {
                    printf("%10s %6s %10s %10s %10s %10s %10s\n",
                           "NAME", "INDEX", "VAL", "G_APPROX", "G_BACKPROP", "ERROR", "REL_ERROR");
                    header_output = TRUE;
                }
                if (is_error)
                    printf("ERROR:\n");
                printf("%10s %6d %+10.2e %+10.2e %+10.2e %10.2e %10.2e\n",
                       p->name, (int)param_idx, x0, g_approx, g, error,
                       rel_error);
                if (is_error)
                    exit(1);
            }
        }
    }
}


#define MAX_ARGS 4

typedef struct {
    NCContext *s;
    NCTensor *args[MAX_ARGS];
    int op_index;
} GradientTestState;

typedef enum {
    G_INIT_U1,
    G_INIT_NZ_U1,
    G_INIT_NZ_P1,
} GradientInitEnum;

typedef struct {
    NCTypeEnum item_type;
    GradientInitEnum init;
    int n_dims;
    size_t dims[3];
    int range;
} GradientArgDef;

typedef struct {
    int op_index;
    const char *name;
    int n_args;
    GradientArgDef args[MAX_ARGS];
    float eps;
    float error_max;
    float rel_error_max;
    BOOL use_bf16;
} GradientTestDef;

#define DEF_ARG { NC_TYPE_F32, G_INIT_U1, 1, { 10 } }

static const GradientTestDef gradient_test[] = {
    { 0, "mul", 2, { DEF_ARG, DEF_ARG }, 1e-3, 1e-4, INFINITY },
    { 0, "mul", 2, { DEF_ARG, DEF_ARG }, 5e-3, 5e-3, INFINITY, TRUE },
    { 0, "mul_dup0", 2, {
            DEF_ARG,
            { NC_TYPE_F32, G_INIT_U1, 0, { } },
        }, 1e-3, 1e-4, INFINITY },
    { 0, "mul_dup1", 2, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 3 } },
            { NC_TYPE_F32, G_INIT_U1, 1, { 4 } },
        }, 1e-3, 1e-4, INFINITY },
    { 0, "mul_dup1", 2, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 3 } },
            { NC_TYPE_F32, G_INIT_U1, 1, { 4 } },
        }, 5e-3, 5e-3, INFINITY, TRUE },
    { 10, "sub", 2, { DEF_ARG, DEF_ARG }, 1e-3, 1e-4, INFINITY },
    { 11, "div", 2, {
            DEF_ARG,
            { NC_TYPE_F32, G_INIT_NZ_U1, 1, { 10 } }
        }, 1e-3, INFINITY, 1e-3 },
    { 12, "log", 1, {
            { NC_TYPE_F32, G_INIT_NZ_P1, 1, { 10 } }
        }, 1e-3, INFINITY, 1e-2 },
    { 2, "sigmoid", 1, { DEF_ARG }, 1e-3, 1e-2, INFINITY },
    { 2, "sigmoid", 1, { DEF_ARG }, 1e-3, 1e-2, INFINITY, TRUE },
    { 4, "tanh", 1, { DEF_ARG }, 1e-3, 3e-2, INFINITY },
    { 4, "tanh", 1, { DEF_ARG }, 1e-3, 3e-2, INFINITY, TRUE },
    { 5, "relu", 1, { DEF_ARG }, 1e-3, 1e-2, INFINITY },
    { 5, "relu", 1, { DEF_ARG }, 1e-3, 1e-2, INFINITY, TRUE },
    { 8, "lstm_clamped", 4, { DEF_ARG, DEF_ARG, DEF_ARG, DEF_ARG },
      1e-3, 1e-4, INFINITY },
    { 8, "lstm_clamped", 4, { DEF_ARG, DEF_ARG, DEF_ARG, DEF_ARG },
      1e-3, 1e-2, INFINITY, TRUE },
    { 9, "lerp", 3, { DEF_ARG, DEF_ARG, DEF_ARG },
      1e-3, 1e-3, INFINITY },
    { 30, "gelu", 1, { DEF_ARG }, 1e-2, 2e-2, INFINITY },
    { 30, "gelu", 1, { DEF_ARG }, 1e-2, 2e-2, INFINITY, TRUE },
    { 13, "max", 2, { DEF_ARG, DEF_ARG }, 1e-3, 1e-3, INFINITY },
    { 14, "min", 2, { DEF_ARG, DEF_ARG }, 1e-3, 1e-3, INFINITY },
    
    { 3, "soft_max", 1, { DEF_ARG }, 1e-3, 1e-2, INFINITY },
    { 3, "soft_max", 1, { DEF_ARG }, 1e-3, 1e-2, INFINITY, TRUE },
    
    { 1, "concat", 1, { DEF_ARG }, 1e-3, 1e-3, INFINITY },
    { 17, "get_col", 2, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 2, 3 } },
            { NC_TYPE_I32, G_INIT_U1, 1, { 10 }, 3 },
        }, 1e-3, 1e-3, INFINITY },

    { 19, "get_element", 2, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 3, 4 } },
            { NC_TYPE_I32, G_INIT_U1, 1, { 4 }, 3 },
        }, 1e-3, 1e-3, INFINITY },
    { 24, "slt_mat_set", 1, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 3 } },
        }, 1e-3, 1e-3, INFINITY },
    { 25, "rel_shift", 1, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 5, 3 } },
        }, 1e-3, 1e-3, INFINITY },
    { 26, "reduce_sum", 1, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 3 } }
        }, 1e-3, 1e-3, INFINITY },
    { 27, "slice0", 1, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 3 } }
        }, 1e-3, 1e-3, INFINITY },
    { 28, "slice1", 1, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 3, 4 } }
        }, 1e-3, 1e-3, INFINITY },
    { 29, "slice_add", 2, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 3 } },
            { NC_TYPE_F32, G_INIT_U1, 2, { 2, 3 } }
        }, 1e-3, 1e-3, INFINITY },

    { 6, "layer_norm", 1, { DEF_ARG }, 1e-3, 1.1e-4, INFINITY },
    { 6, "layer_norm", 1, { DEF_ARG }, 1e-3, 1.2e-3, INFINITY, TRUE },
    { 7, "rms_norm", 1, { DEF_ARG }, 1e-3, 1e-4, INFINITY },
    { 7, "rms_norm", 1, { DEF_ARG }, 1e-3, 4e-4, INFINITY, TRUE },

    { 21, "indexed_log(soft_max)", 2, {
            DEF_ARG,
            { NC_TYPE_I32, G_INIT_U1, 0, { }, 10 },
        }, 1e-3, 1e-1, INFINITY },
    
    { 18, "add_col", 3, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 2, 5 } },
            { NC_TYPE_I32, G_INIT_U1, 1, { 5 }, 3 },
            { NC_TYPE_F32, G_INIT_U1, 2, { 2, 3 } },
        }, 1e-3, 1e-3, INFINITY },
    { 20, "add_element", 3, {
            { NC_TYPE_F32, G_INIT_U1, 1, { 4 } },
            { NC_TYPE_I32, G_INIT_U1, 1, { 4 }, 3 },
            { NC_TYPE_F32, G_INIT_U1, 2, { 3, 4 } },
        }, 1e-3, 1e-3, INFINITY },
    
    { 15, "transpose", 1, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 3 } }
        }, 1e-3, 1e-3, INFINITY },
    { 16, "matmul", 2, {
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 4 } },
            { NC_TYPE_F32, G_INIT_U1, 2, { 4, 1 } },
        }, 1e-3, 1e-3, INFINITY },
};

static NCTensor *my_test_func(void *opaque)
{
    GradientTestState *gs = opaque;
    NCTensor **args = gs->args;
    NCTensor *x1 = args[0];
    NCTensor *loss, *t0;
    
    switch(gs->op_index) {
    case 0:
        t0 = nc_mul(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]));
        loss = nc_sum(t0);
        break;
    case 1:
        {
            NCTensor *tab[2];
            tab[0] = nc_dup_tensor(x1);
            tab[1] = nc_dup_tensor(x1);
            t0 = nc_concat(tab, 2, 0);
            loss = nc_sum(t0);
        }
        break;
    case 2:
        t0 = nc_sigmoid(nc_dup_tensor(x1));
        loss = nc_sum(t0);
        break;
    case 3:
        t0 = nc_soft_max(nc_dup_tensor(x1));
        /* select the first element */
        t0 = nc_resize(t0, 1);
        loss = nc_reshape(t0, 0, NULL);
        break;
    case 4:
        t0 = nc_tanh(nc_dup_tensor(x1));
        loss = nc_sum(t0);
        break;
    case 5:
        t0 = nc_relu(nc_dup_tensor(x1));
        loss = nc_sum(t0);
        break;
    case 6:
        t0 = nc_layer_norm(nc_dup_tensor(x1), 1e-5);
        /* select the first element */
        t0 = nc_resize(t0, 1);
        loss = nc_reshape(t0, 0, NULL);
        break;
    case 7:
        t0 = nc_rms_norm(nc_dup_tensor(x1), 1e-5);
        /* select the first element */
        t0 = nc_resize(t0, 1);
        loss = nc_reshape(t0, 0, NULL);
        break;
    case 8:
        t0 = nc_lstm_clamped(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]),
                             nc_dup_tensor(args[2]), nc_dup_tensor(args[3]));
        loss = nc_sum(t0);
        break;
    case 9:
        t0 = nc_lerp(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]),
                     nc_dup_tensor(args[2]));
        loss = nc_sum(t0);
        break;
    case 10:
        t0 = nc_sub(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]));
        loss = nc_sum(t0);
        break;
    case 11:
        t0 = nc_div(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]));
        loss = nc_sum(t0);
        break;
    case 12:
        t0 = nc_log(nc_dup_tensor(args[0]));
        loss = nc_sum(t0);
        break;
    case 13:
        t0 = nc_max(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]));
        loss = nc_sum(t0);
        break;
    case 14:
        t0 = nc_min(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]));
        loss = nc_sum(t0);
        break;
    case 15:
        t0 = nc_transpose(nc_dup_tensor(args[0]));
        loss = nc_sum(t0);
        break;
    case 16:
        t0 = nc_matmul(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]));
        loss = nc_sum(t0);
        break;
    case 17:
        t0 = nc_get_col(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]));
        loss = nc_sum(t0);
        break;
    case 18:
        t0 = nc_add_col(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]),
                        nc_dup_tensor(args[2]));
        t0 = nc_rms_norm(t0, 1e-5);
        loss = nc_sum(t0);
        break;
    case 19:
        t0 = nc_get_element(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]));
        loss = nc_sum(t0);
        break;
    case 20:
        t0 = nc_add_element(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]),
                            nc_dup_tensor(args[2]));
        t0 = nc_rms_norm(t0, 1e-5);
        loss = nc_sum(t0);
        break;
    case 21:
        {
            NCTensor *y;
            y = nc_soft_max(nc_dup_tensor(args[0]));
            loss = nc_indexed_log(y, nc_dup_tensor(args[1]));
        }
        break;
    case 24:
        t0 = nc_slt_mat_set(nc_dup_tensor(args[0]), 1, 1.0);
        loss = nc_sum(t0);
        break;
    case 25:
        t0 = nc_rel_shift(nc_dup_tensor(args[0]), -3, 1);
        loss = nc_sum(t0);
        break;
    case 26:
        t0 = nc_reduce_sum(NULL, nc_dup_tensor(args[0]), 1);
        loss = nc_sum(t0);
        break;
    case 27:
        t0 = nc_slice(nc_dup_tensor(args[0]), 0, 1, 3);
        loss = nc_sum(t0);
        break;
    case 28:
        t0 = nc_slice(nc_dup_tensor(args[0]), 1, 1, 3);
        loss = nc_sum(t0);
        break;
    case 29:
        t0 = nc_slice_add(nc_dup_tensor(args[0]), nc_dup_tensor(args[1]),
                          0, 1);
        loss = nc_sum(t0);
        break;
    case 30:
        t0 = nc_gelu(nc_dup_tensor(x1));
        loss = nc_sum(t0);
        break;
    default:
        abort();
    }
    return loss;
}

static __unused void nc_test_approx_gradient_op(NCContext *s, NCDevice *dev,
                                                BOOL verbose,
                                                const GradientTestDef *gd)
{
    NCParamList param_list;
    NCRNDState *rnd_state;
    GradientTestState gs_s, *gs = &gs_s;
    int j;
    
    printf("Testing %s (%s)\n", gd->name, gd->use_bf16 ? "bf16" : "f32");

    gs->s = s;

    rnd_state = nc_rnd_init(dev, 1234);

    nc_param_list_init(&param_list);
    for(j = 0; j < gd->n_args; j++) {
        NCTensorData *d, dbuf;
        float min_val, val;
        int k;

        gs->args[j] = nc_new_tensor(dev, gd->args[j].item_type,
                                    gd->args[j].n_dims,
                                    gd->args[j].dims);
        if (gd->args[j].item_type == NC_TYPE_F32) {
            nc_new_param(&param_list, &gs->args[j], "x%d", j);
        }

        d = nc_tensor_get_data(&dbuf, gs->args[j]);
        switch(gd->args[j].init) {
        case G_INIT_U1:
            if (gd->args[j].item_type == NC_TYPE_I32) {
                RNDState rnd_state2;
                rnd_init(&rnd_state2, 1235);

                if (d->n_dims == 0) {
                    nc_set1_i32(gs->args[j], 0, NULL,
                                rnd_unif_u32(&rnd_state2) % gd->args[j].range);
                } else if (d->n_dims == 1) {
                    for(k = 0; k < d->dims[0]; k++) {
                        nc_set1_i32_1d(gs->args[j], k, rnd_unif_u32(&rnd_state2) % gd->args[j].range);
                    }
                } else {
                    abort();
                }
            } else {
                nc_tensor_set_rnd_unif(gs->args[j], 0.0, 1.0, rnd_state);
            }
            break;
        case G_INIT_NZ_U1:
            nc_tensor_set_rnd_unif(gs->args[j], 0.0, 1.0, rnd_state);
            min_val = 1e-1;
            for(k = 0; k < d->dims[0]; k++) {
                val = nc_get1_f32_1d(gs->args[j], k);
                if (fabsf(val) < min_val) {
                    if (val < 0)
                        val = -min_val;
                    else
                        val = min_val;
                    nc_set1_f32_1d(gs->args[j], k, val);
                }
            }
            break;
        case G_INIT_NZ_P1:
            nc_tensor_set_rnd_unif(gs->args[j], 0.0, 1.0, rnd_state);
            min_val = 1e-1;
            for(k = 0; k < d->dims[0]; k++) {
                val = nc_get1_f32_1d(gs->args[j], k);
                val = fabsf(val);
                if (val < min_val)
                    val = min_val;
                nc_set1_f32_1d(gs->args[j], k, val);
            }
            break;
        default:
            abort();
        }
    }

    gs->op_index = gd->op_index;

    test_approx_gradient(s, my_test_func, gs, &param_list,
                         gd->eps, gd->error_max, gd->rel_error_max,
                         verbose, gd->use_bf16);

    for(j = 0; j < gd->n_args; j++) {
        if (gd->args[j].item_type != NC_TYPE_F32) {
            nc_free_tensor(gs->args[j]);
        }
    }

    nc_param_list_end(&param_list);
    nc_rnd_end(rnd_state);
}

static __unused void nc_test_approx_gradient(NCContext *s, NCDevice *dev,
                                             BOOL verbose)
{
    int i;
    
    for(i = 0; i < countof(gradient_test); i++) {
        nc_test_approx_gradient_op(s, dev, verbose, &gradient_test[i]);
    }
}

static void nc_basic_test(NCContext *s, NCDevice *dev)
{
    NCTensor *t0, *t1;
    float res;
    int i, n;
    
    t0 = nc_new_f32(dev, 2.0);
    t1 = nc_new_f32(dev, 3.0);
    t0 = nc_mul(t0, t1);
    res = nc_get_scalar_f32(t0);
    //    printf("res=%f\n", res);
    assert(res == 6.0f);
    nc_free_tensor(t0);

    n = 10000;
    t0 = nc_new_tensor_1d(dev, NC_TYPE_F32, n);
    for(i = 0; i < n; i++)
        nc_set1_f32_1d(t0, i, (float)(i + 1));
    t0 = nc_sum(t0);
    res = nc_get_scalar_f32(t0);
    //    printf("res=%f\n", res);
    assert(res == (float)(n * (n + 1) / 2));
    nc_free_tensor(t0);
}

void help(void)
{
    printf("usage: nc2test [options]\n"
           "\n"
           "Options:\n"
           "-h                 help\n"
           "-d device          select the compute device (cpu or cuda)\n"
           );
    exit(1);
}

int main(int argc, char **argv)
{
    int c;
    const char *device_name;
    NCContext *s;
    NCDevice *dev;
    BOOL verbose;
    
    device_name = "cpu";
    verbose = FALSE;
    for(;;) {
        c = getopt(argc, argv, "hd:v");
        if (c == -1)
            break;
        switch(c) {
        case 'h':
            help();
        case 'd':
            device_name = optarg;
            break;
        case 'v':
            verbose = TRUE;
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
    
    nc_basic_test(s, dev);
    if (!strcmp(device_name, "cpu")) {
        nc_test_backward_scalar(s, dev, TRUE);
    }
    nc_test_approx_gradient(s, dev, verbose);
    
    nc_context_end(s);
    return 0;
}
