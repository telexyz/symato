/*
 * LibNC
 * 
 * Copyright (c) 2018-2019 Fabrice Bellard
 *
 */
#ifndef LIBNC_H
#define LIBNC_H

#include <inttypes.h>
#include "cutils.h"
#include "list.h"

/* profiling */

typedef enum {
    PROF_EVAL,
    PROF_GRAD,
    PROF_SGD,
    PROF_UPDATE,
    PROF_WRITE_SYM,
    PROF_PROBE,
    PROF_TOTAL,
    PROF_COUNT,
} ProfEnum;

#ifdef PROFILE

extern int64_t prof_cycles[PROF_COUNT];
extern int64_t prof_samples[PROF_COUNT];
extern int64_t prof_ops[PROF_COUNT];

static inline void prof_start(int idx)
{
    prof_cycles[idx] -= get_cycles();
}

static inline void prof_end(int idx)
{
    prof_cycles[idx] += get_cycles();
    prof_samples[idx]++;
}

static inline void prof_end_ops(int idx, int n_ops)
{
    prof_cycles[idx] += get_cycles();
    prof_ops[idx] += n_ops;
    prof_samples[idx]++;
}

#else

static inline void prof_start(int idx)
{
}

static inline void prof_end(int idx)
{
}

static inline void prof_end_ops(int idx, int n_ops)
{
}

#endif

void nc_prof_dump(void);

/* Automatic Differentiation Engine */

typedef struct NCContext NCContext;
typedef struct NCDevice NCDevice;
typedef struct NCTensor NCTensor;
typedef struct NCTensorBuffer NCTensorBuffer;
typedef struct NCNode NCNode;
typedef struct NCRNDState NCRNDState;
typedef struct NCSGDOptState NCSGDOptState;

typedef enum {
    NC_TYPE_F32,
    NC_TYPE_BF16,
    NC_TYPE_F16,
    NC_TYPE_I8,
    NC_TYPE_I16,
    NC_TYPE_I32,
    NC_TYPE_COUNT,
} NCTypeEnum;

extern size_t nc_type_size_table[NC_TYPE_COUNT];
extern const char *nc_type_name_table[NC_TYPE_COUNT];

#define NC_N_DIMS_MAX 4 /* maximum number of axis for tensors */

typedef struct NCTensorData {
    NCTypeEnum item_type;
    size_t item_size;
    void *data;
    size_t stride; /* in elements */
    size_t n_strides; /* prod(j = 1 ... n_dims, dims[j]); */
    int n_dims;
    const size_t *dims; /* n_dims length */
    const size_t *strides; /* n_dims length, strides in bytes */
} NCTensorData;

void *nc_malloc(size_t size);
void *nc_mallocz(size_t size);
void nc_free(void *ptr);

NCContext *nc_context_init(int nb_threads);
void nc_context_end(NCContext *m);

NCDevice *nc_new_cpu_device(NCContext *m);
NCDevice *nc_new_cuda_device(NCContext *m, int device_index);
NCDevice *nc_new_device(NCContext *m, const char *device_name);
void nc_synchronize(NCDevice *d);

NCTensorBuffer *nc_new_tensor_buffer(NCDevice *d, size_t size);
NCTensorBuffer *nc_dup_tensor_buffer(const NCTensorBuffer *b);
void nc_free_tensor_buffer(NCTensorBuffer *b);

NCTensor *nc_new_tensor(NCDevice *d, NCTypeEnum type,
                        int n_dims, const size_t *dims);
NCTensor *nc_new_tensor_from_tensor(const NCTensor *x);
NCTensor *nc_new_tensor_from_tensor_nz(const NCTensor *x);
NCTensor *nc_new_scalar(NCDevice *d, NCTypeEnum type);
NCTensor *nc_new_tensor_1d(NCDevice *d, NCTypeEnum type, size_t len);
NCTensor *nc_new_tensor_2d(NCDevice *d, NCTypeEnum type, size_t n0, size_t n1);
NCTensor *nc_new_tensor_3d(NCDevice *d, NCTypeEnum type,
                           size_t n0, size_t n1, size_t n2);
NCTensor *nc_new_tensor_4d(NCDevice *d, NCTypeEnum type,
                           size_t n0, size_t n1, size_t n2, size_t n3);
NCTensor *__attribute__((format(printf, 2, 3))) nc_tensor_set_name(NCTensor *x, const char *fmt, ...);
NCTensor *nc_dup_tensor(const NCTensor *x);
void nc_free_tensor(NCTensor *x);
void nc_dump_tensor(const char *name, NCTensor *x, size_t n);
uint32_t nc_tensor_get_hash(NCTensor *x);
void nc_dump_tensor_hash(const char *name, const NCTensor *x);
NCNode *nc_get_node(NCTensor *x);
/* create an alias to tensor 'x1'. Gradient is not propagated thru it */
NCTensor *nc_slice_alias(const NCTensor *x1, int axis, size_t start, size_t end);

NCTypeEnum nc_tensor_get_item_type(const NCTensor *x);
NCTensorData *nc_tensor_get_data(NCTensorData *sd, const NCTensor *x);
/* Return a pointer to the tensor data. If *pstride is non NULL,
   return the stride (in elements) of the first dimension. */
void *nc_tensor_get_ptr(NCTensor *x, size_t *pstride);
const size_t *nc_tensor_get_dims(const NCTensor *x, int *pn_dims);
void nc_tensor_set_zero(NCTensor *y);
void nc_tensor_set_f32(NCTensor *y, float val);
NCRNDState *nc_rnd_init(NCDevice *d, uint32_t seed);
void nc_rnd_end(NCRNDState *s);
void nc_tensor_set_rnd_unif(NCTensor *y, float avg, float range,
                            NCRNDState *rnd_state);
void nc_tensor_set_dropout(NCTensor *y, float prob, NCRNDState *rnd_state);

void nc_set1_i32(NCTensor *y, int n_dims, const size_t *tab_indexes,
                 int32_t val);
void nc_set1_i32_1d(NCTensor *y, size_t i0, int32_t val);
void nc_set1_i32_2d(NCTensor *y, size_t i0, size_t i1, int32_t val);
void nc_set1_f32(NCTensor *y, int n_dims, const size_t *tab_indexes,
                 float val);
void nc_set1_f32_1d(NCTensor *y, size_t i0, float val);

int32_t nc_get1_i32(const NCTensor *x, int n_dims, const size_t *tab_indexes);
float nc_get1_f32(const NCTensor *x, int n_dims, const size_t *tab_indexes);
float nc_get1_f32_1d(const NCTensor *x, size_t i0);
float nc_get_scalar_f32(const NCTensor *x);

void nc_tensor_copy(NCTensor *dst, NCTensor *src);
void nc_tensor_convert(NCTensor *dst, NCTensor *src);

void nc_dump_dims(const char *str, NCTensor *x);
size_t nc_get_heap_size(NCContext *m);
NCContext *nc_get_tensor_context(const NCTensor *x);
NCTensor *nc_tensor_to_device(NCTensor *x, NCDevice *d);
NCTensor *nc_tensor_to_cpu_device(NCTensor *x);
NCDevice *nc_get_tensor_device(const NCTensor *x);
                                 
/* element wise operations */
NCTensor *nc_convert(NCTensor *x, NCTypeEnum new_type);
NCTensor *nc_add(NCTensor *x1, NCTensor *x2);
NCTensor *nc_neg(NCTensor *x);
NCTensor *nc_sub(NCTensor *x1, NCTensor *x2);
NCTensor *nc_mul(NCTensor *x1, NCTensor *x2);
NCTensor *nc_div(NCTensor *x1, NCTensor *x2);
NCTensor *nc_recip(NCTensor *x);
NCTensor *nc_min(NCTensor *x1, NCTensor *x2);
NCTensor *nc_max(NCTensor *x1, NCTensor *x2);
/* select x1[i] if z[i] = 0 and x2[i] otherwise */
NCTensor *nc_select(NCTensor *z, NCTensor *x1, NCTensor *x2);
/* set y[i] = x1[i] if mask[i] = 0 and y[i] = c if mask[i] != 0. If
   mask_inv is TRUE, 'mask' is inverted */
NCTensor *nc_masked_fill(NCTensor *x, NCTensor *mask, float c, BOOL mask_inv);
NCTensor *nc_sigmoid(NCTensor *x);
NCTensor *nc_tanh(NCTensor *x);
NCTensor *nc_relu(NCTensor *x);
NCTensor *nc_gelu(NCTensor *x);
NCTensor *nc_log(NCTensor *x);
/* return cp * fg + min(1 - fg, ig) * in */
NCTensor *nc_lstm_clamped(NCTensor *cp, NCTensor *in,
                          NCTensor *fg, NCTensor *ig);
/* return a * (1 - t) + b * t */
NCTensor *nc_lerp(NCTensor *a, NCTensor *b, NCTensor *t);

/* other operations */
NCTensor *nc_new_vec_f32(NCDevice *d, size_t n, float val);
NCTensor *nc_new_f32(NCDevice *d, float val);
NCTensor *nc_reshape(NCTensor *x, int n_dims, const size_t *dims);
NCTensor *nc_reshape_1d(NCTensor *x, size_t n0);
NCTensor *nc_reshape_2d(NCTensor *x, size_t n0, size_t n1);
NCTensor *nc_reshape_3d(NCTensor *x, size_t n0, size_t n1, size_t n2);
NCTensor *nc_reshape_4d(NCTensor *x, size_t n0, size_t n1, size_t n2,
                        size_t n3);
/* duplicate the tensor by adding n_dims dimensions */
NCTensor *nc_repeat(NCTensor *x, int n_dims, const size_t *dims);
NCTensor *nc_repeat_1d(NCTensor *x, size_t n);
/* return y0 + sum over the dimensions > n_dims of 'x'. y0 = NULL
   is supported */
NCTensor *nc_reduce_sum(NCTensor *y0, NCTensor *x, int n_dims);
/* sum all the elements of a tensor */
NCTensor *nc_sum(NCTensor *x);
/* sum of squares */
NCTensor *nc_reduce_sum_sqr(NCTensor *x);
NCTensor *nc_slice(NCTensor *x, int axis, size_t start, size_t end);
NCTensor *nc_slice_add(NCTensor *y0, NCTensor *x, int axis, size_t start);
/* concatenation along axis 'axis' */
NCTensor *nc_concat(NCTensor **inputs, int n_inputs, int axis);
/* shortcut for axis = 0 */
NCTensor *nc_vconcat(NCTensor **inputs, int n_inputs);
/* shortcut for axis = 1 */
NCTensor *nc_hconcat(NCTensor **inputs, int n_inputs);
/* split along axis 'axis'. If tab_size = NULL, split equally. */
void nc_split(NCTensor **tab_y, NCTensor *x, int n_outputs,
              const size_t *tab_size, int axis);
/* shortcut for axis = 0 */
void nc_vsplit(NCTensor **tab_y, NCTensor *x, int n_outputs,
               const size_t *tab_size);
/* shortcut for axis = 1 */
void nc_hsplit(NCTensor **tab_y, NCTensor *x, int n_outputs,
               const size_t *tab_size);

typedef enum {
    NC_PAD_ZERO,
    NC_PAD_DUP, /* duplicate element */
    /* trim types, dual to padding */
    NC_TRIM_NORMAL = NC_PAD_ZERO,
    NC_TRIM_SUM, /* add trimmed elements to the edge */
} NCPadEnum;

/* pad (len > 0) or trim (len < 0) the axis 0 of 'x' */
NCTensor *nc_pad(NCTensor *x, ssize_t left_len, NCPadEnum left_op,
                 ssize_t right_len, NCPadEnum right_op);
/* shortcut to nc_pad() */
NCTensor *nc_resize(NCTensor *x, size_t n);

/* if x is not contiguous then create a new contiguous tensor and copy
   x to it. Otherwise, return 'x'. */
NCTensor *nc_make_contiguous(NCTensor *x);
/* Return a new tensor sharing the same buffer as 'x' with the permuted
   dimensions. axis[i] is the corresponding axis in 'x' */
NCTensor *nc_permute_alias(NCTensor *x, int n_dims, const int *axis);
/* same as nc_permute_alias but calls nc_make_contiguous after. */
NCTensor *nc_permute(NCTensor *x, int n_dims, const int *axis);
/* special case of nc_permute() */
NCTensor *nc_transpose(NCTensor *x);
NCTensor *nc_matmul(NCTensor *w, NCTensor *x);
/* return w*x + y0. w and x can be optionally transposed. y0 can be NULL */
NCTensor *nc_matmul_add(NCTensor *w, NCTensor *x, NCTensor *y0,
                        BOOL w_trans, BOOL x_trans);
NCTensor *nc_matmul_stride(NCTensor *w, NCTensor *x);
/* return a matrix where each column is the column x[i] of matrix 'w' */
NCTensor *nc_get_col(NCTensor *w, NCTensor *x);
/* add the vectors 'z' at column number 'x' in matrix 'w'. */
NCTensor *nc_add_col(NCTensor *z, NCTensor *x, NCTensor *w);
/* select the x-th element in each column of 'w' */
NCTensor *nc_get_element(NCTensor *w, NCTensor *x);
/* add z to the x-th element in each column of 'w' */
NCTensor *nc_add_element(NCTensor *z, NCTensor *x, NCTensor *w);
NCTensor *nc_soft_max(NCTensor *x);
/* Equivalent to y = log(get_element(x, eout)). It is expected to be
   used as nc_index_log(nc_soft_max(x), eout) so that the gradient
   computation is optimized. */
NCTensor *nc_indexed_log(NCTensor *x, NCTensor *eout);
NCTensor *nc_layer_norm(NCTensor *x, float eps);
NCTensor *nc_rms_norm(NCTensor *x, float eps);
NCTensor *nc_slt_mat_set(NCTensor *x, size_t pos, float c);
/* shift the column 'i' by 'pos + i * mult' elements and pad with with zeros */
NCTensor *nc_rel_shift(NCTensor *x, ssize_t pos, ssize_t mult);

/* auto differentiation */

/* get_col_index is non NULL in the sparse gradient case */
typedef void NCParamUpdateFunc(void *opaque, NCTensor *grad,
                               NCTensor *get_col_index);

/* add a 'parameter' graph node to 'x' and return 'x'. */
NCTensor *nc_set_param(NCTensor *x, void *opaque);
/* return a new tensor with its graph removed */
NCTensor *nc_stop_grad(NCTensor *x);

/* manipulation of graph nodes */
NCNode *nc_dup_node(const NCNode *n);
void nc_free_node(NCNode *n);
void nc_combine_nodes(NCContext *m, NCNode **tab_op1, int count,
                      int axis, int elem_size, const size_t *tab_elem_size);
NCNode *nc_concat_node(NCContext *m, NCNode **inputs, int count,
                       int axis, const size_t *tab_size);
void nc_concat_optimization(NCContext *m, NCNode **concat_nodes, int count);
void nc_node_set_parent(NCNode *n, int arg_index, const NCNode *n1);
void nc_node_set_arg(NCNode *n, int arg_index, const NCTensor *x);

#define NC_BW_KEEP_GRAD_GRAPH (1 << 0)
/* optimize the nc_get_col() gradient */
#define NC_BW_SPARSE_GRAD     (1 << 1)

void nc_backward(const NCTensor *x, NCTensor *grad,
                 NCParamUpdateFunc *param_update_func, int flags);
void nc_dump_graph(NCTensor *x);

/* utilities for function parameters */

typedef struct {
    struct list_head link;
    NCTensor **pval; /* pointer to the tensor location */
    char *name; /* parameter name */
    NCTensor *low_part; /* if BF16 parameter, additional 16 bit precision */
    NCTensor *saved_grad; /* debug */
    /* SGD opt data */
    struct SGDOptVarState *sgd_opt;
} NCParam;

typedef struct {
    struct list_head param_list;
    BOOL add_graph;
} NCParamList;

void nc_param_list_init(NCParamList *pl);
void nc_param_list_set_graph(NCParamList *pl, BOOL add_graph);
NCParam *nc_new_param_str(NCParamList *pl, NCTensor **pval, const char *str);
__attribute__((format(printf, 3, 4))) NCParam *nc_new_param(NCParamList *pl, NCTensor **pval, const char *fmt, ...);
void nc_param_list_end(NCParamList *pl);

NCParam *nc_find_param(NCParamList *pl, const char *name);
size_t nc_get_param_count(NCParamList *pl);
void nc_save_coefs(NCParamList *pl, const char *filename);
void nc_load_coefs(NCParamList *pl, const char *filename);
void nc_save_state(NCParamList *pl, const char *filename);
void nc_load_state(NCParamList *pl, const char *filename);

/* SGD optimizer */

typedef enum {
    SGD_OPT_BASIC,
    SGD_OPT_ADAM,
    SGD_OPT_TEST,
} SGDOptAlgoEnum;

typedef struct {
    SGDOptAlgoEnum algo;
    union {
        struct {
            float beta1;
            float beta2;
            float eps;
            float gradient_clip; /* if != 0, per parameter gradient clipping */
        } adam;
    } u;
    float lr;
} SGDOptParams;

NCSGDOptState *nc_sgd_opt_init(NCContext *m, const SGDOptParams *p);
void nc_sgd_opt_end(NCSGDOptState *s);
void sgd_opt_update_var(void *opaque, NCTensor *yg, NCTensor *get_col_index);

/* set the SGD optimizer 's' to all parameters of the model */
void nc_sgd_opt_set_all(NCParamList *param_list, NCSGDOptState *s);

/* set the SGD optimizer 's' to the variable 'x'. Remove it if s = NULL */
void nc_sgd_opt_set(NCParam *x, NCSGDOptState *s);
void nc_sgd_opt_update(NCSGDOptState *s);
/* force the learning rate */
void nc_sgd_opt_set_lr(NCSGDOptState *s, float lr);
float nc_sgd_opt_get_lr(NCSGDOptState *s);

/* for SGD_OPT_TEST */
NCTensor *nc_sgd_opt_get_grad(NCParam *p);

/* misc utilities (to be removed) */

typedef struct {
    uint32_t seed;
    /* used by Gaussian generator */
    int idx;
    float y1;
} RNDState;

typedef struct {
    uint16_t u16;
} nc_float16_t;

void rnd_init(RNDState *s, uint32_t seed);
uint32_t rnd_unif_u32(RNDState *s);
float rnd_unif(RNDState *s);
void rnd_unif_vec(float *tab, size_t n, float mu, float range,
                  RNDState *s);
void rnd_unif_mat(float *tab, size_t stride, size_t h, size_t w,
                  float mu, float sigma, RNDState *s);

float vec_sum_f32(const float *tab, size_t n);

typedef struct  {
    float val;
    uint32_t idx;
} NCTopKEntry;

/* Return the k largest values among prob[0...n_symb-1] such that k is
   the largest value such that k <= topk and sum(i=0 .. k - 2,
   prob[tab[i]]) < topp.

   It is assumed that prob[i] >= 0. The function returns (k, tab,
   sum). 'sum' is the sum of the k returned values. 'tab' must be
   freed with nc_free(). */
int nc_topk(NCTopKEntry **ptab, double *psum,
            const float *prob, size_t n, int topk, float topp);

#endif /* LIBNC_H */
