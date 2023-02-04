/*
 * Dump LibNC parameters
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

#include "cutils.h"
#include "list.h"

#define NC_COEF_MAGIC 0x23f4aefa

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t n_dims;
    uint32_t name_len;
} NCCoefHeader;

struct list_head coefs_list;

#define N_DIMS_MAX 16

typedef struct {
    struct list_head link;
    char *name;
    float *data;
    size_t size;
    int n_dims;
    int dims[N_DIMS_MAX];
    void *opaque;
    float mu, sigma, min, max;
} NCVar;

void *mallocz(size_t size)
{
    void *ptr;
    ptr = malloc(size);
    if (!ptr)
        return NULL;
    memset(ptr, 0, size);
    return ptr;
}

void nc_load_coefs(const char *filename)
{
    FILE *f;
    NCCoefHeader h;
    NCVar *v;
    int i;
    size_t size;
    
    f = fopen(filename, "rb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    init_list_head(&coefs_list);
    
    for(;;) {
        if (fread(&h, 1, sizeof(h), f) != sizeof(h))
            break;
        if (h.magic != NC_COEF_MAGIC) {
            fprintf(stderr, "Invalid magic 0x%08x at offset 0x%" PRIx64 "\n", h.magic, (int64_t)ftell(f) - sizeof(h));
            exit(1);
        }
        if (h.n_dims > N_DIMS_MAX) {
            fprintf(stderr, "Invalid number of dimensions");
            exit(1);
        }
        if (h.type != 0) {
            fprintf(stderr, "Unsupported type: %d\n", h.type);
            exit(1);
        }
        v = malloc(sizeof(*v));
        v->n_dims = h.n_dims;
        fread(v->dims, 1, sizeof(v->dims[0]) * v->n_dims, f);
        v->name = malloc(h.name_len + 1);
        fread(v->name, 1, h.name_len, f);
        v->name[h.name_len] = '\0';
        size = 1;
        for(i = 0; i < v->n_dims; i++)
            size *= v->dims[i];
        v->size = size;
        v->data = malloc(sizeof(float) * v->size);
        fread(v->data, 1, sizeof(float) * v->size, f);
        list_add_tail(&v->link, &coefs_list);
    }
}

void nc_compute_coef_stats(void)
{
    struct list_head *el;
    NCVar *v;
    size_t i;
    double sum, sum2, a, min, max;
    
    list_for_each(el, &coefs_list) {
        v = list_entry(el, NCVar, link);
        sum = sum2 = 0;
        min = INFINITY;
        max = -INFINITY;
        for(i = 0; i < v->size; i++) {
            a = v->data[i];
            sum += a;
            sum2 += a * a;
            if (a > max)
                max = a;
            if (a < min)
                min = a;
        }
        v->mu = sum / v->size;
        if (v->size == 1)
            v->sigma = 0;
        else
            v->sigma = sqrt(sum2 / v->size - v->mu * v->mu);
        v->min = min;
        v->max = max;
    }
}

void nc_dump_coef_stats(void)
{
    struct list_head *el;
    NCVar *v;
    int i;
    char buf[64], *q;
    
    printf("%-40s %10s %10s %10s %10s\n",
           "NAME[DIMS]", "mu", "sigma", "min", "max");
    list_for_each(el, &coefs_list) {
        v = list_entry(el, NCVar, link);

        q = buf;
        q += snprintf(q, buf + sizeof(buf) - q, "%s[", v->name);
        for(i = 0; i < v->n_dims; i++) {
            q += snprintf(q, buf + sizeof(buf) - q,
                          "%s%u", i != 0 ? "," : "",
                          (int)v->dims[v->n_dims - 1 - i]);
        }
        q += snprintf(q, buf + sizeof(buf) - q, "]");
        printf("%-40s %10.3g %10.3g %10.3g %10.3g", buf,
               v->mu, v->sigma, v->min, v->max);
        printf("\n");
    }
}

static int to_pixel(float a, float mu, float sigma)
{
    float mult;
    mult = 128.0f / (3 * sigma);
    a = a * mult + 128;
    if (a < 0)
        a = 0;
    else if (a > 255)
        a = 255;
    return lrintf(a);
}

typedef struct {
    int x1;
    int y1;
} VarPos;

/* dump the coefficients as a PGM image. */
static void nc_dump_img(const char *filename)
{
    struct list_head *el;
    NCVar *v;
    FILE *f;
    int width, height, x, y, y1, x1, pad, w, max_width, line_height, h;
    uint8_t *img_data;
    VarPos *vp;
    
    /* layout */
    pad = 2;
    max_width = 1280;
    width = 0;
    height = 0;
    y1 = 0;
    x1 = 0;
    line_height = 0;
    list_for_each(el, &coefs_list) {
        v = list_entry(el, NCVar, link);
        if (v->n_dims > 2) {
            continue;
        } else if (v->n_dims == 2) {
            w = v->dims[1];
        } else {
            w = 1;
        }
        vp = malloc(sizeof(*vp));
        v->opaque = vp;
        h = v->dims[0];

        /* create a new line */
        if (x1 > 0 && (x1 + w) > max_width) {
            y1 += line_height + pad;
            x1 = 0;
            line_height = 0;
        }

        vp->x1 = x1;
        vp->y1 = y1;
        //        printf("%s: x1=%d y1=%d w=%d h=%d\n", v->name, vp->x1, vp->y1, w, h);
        line_height = max_int(line_height, h);
        width = max_int(width, x1 + w);
        height = max_int(height, y1 + line_height);
        x1 += w + pad;
    }
    //    printf("size=%d %d\n", width, height);
    img_data = mallocz(width * height);
    
    list_for_each(el, &coefs_list) {
        v = list_entry(el, NCVar, link);
        vp = v->opaque;
        if (v->n_dims > 2) {
            continue;
        } else if (v->n_dims == 2) {
            w = v->dims[1];
        } else {
            w = 1;
        }
        x1 = vp->x1;
        y1 = vp->y1;
        for(y = 0; y < v->dims[0]; y++) {
            for(x = 0; x < w; x++) {
                img_data[(y1 + y) * width + (x1 + x)] =
                    to_pixel(v->data[x * v->dims[0] + y], v->mu, v->sigma);
            }
        }
        free(vp);
        v->opaque = NULL;
    }

    f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }
    fprintf(f,"P5\n%d %d\n%d\n", width, height, 255);
    fwrite(img_data, 1, width * height, f);
    free(img_data);
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: dump_coefs coef_file [pgm_file]\n");
        exit(1);
    }

    nc_load_coefs(argv[1]);
    nc_compute_coef_stats();
    if (argc >= 3) {
        nc_dump_img(argv[2]);
    } else {
        nc_dump_coef_stats();
    }
    return 0;
}
