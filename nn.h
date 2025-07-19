#ifndef NN_H_

#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define MAT_AT(m, i, j) m.es[(i) *(m).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m, 0);
#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

// Should be Dynamic.
typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es; // Beginning element of pointer.
} Mat;

Mat mat_alloc (size_t rows, size_t cols);
Mat mat_row(Mat m, size_t row);

void mat_rand (Mat m, float low, float high);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a);
void mat_print(Mat m, const char *name, size_t padding);
void mat_fill(Mat m, float x);
void mat_sig(Mat m);
void mat_copy(Mat dest, Mat a);

float rand_float(void);
float sigmoidf(float x);


typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activation is count + 1.
} NN;

NN nn_alloc(size_t *layers, size_t arch_count);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);


#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float(void)
{ return (float) rand() / (float) RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_dot(Mat dest, Mat a, Mat b)
{

    NN_ASSERT(a.cols == b.rows);

    NN_ASSERT(dest.rows == a.rows);
    NN_ASSERT(dest.cols == b.cols);

    size_t n = a.cols;

    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.cols; ++j) {
            for (size_t k = 0; k < n; ++k) {
                MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}
void mat_copy(Mat dest, Mat a)
{
    NN_ASSERT(dest.rows == a.rows);
    NN_ASSERT(dest.cols == a.cols);

    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.cols; ++j) {
            MAT_AT(dest, i, j) = MAT_AT(a, i, j);
        }
    }
}

void mat_sum(Mat dest, Mat a)
{
    NN_ASSERT(dest.rows == a.rows);
    NN_ASSERT(dest.cols == a.cols);

    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.rows; ++j) {
            MAT_AT(dest, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int) padding, "" , name);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%*s    %f ", (int) padding, "", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}


void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = (rand_float() * (high - low)) + low;
        }
    }
}

float sigmoidf(float x)
{
    return 1 / (1 + exp(-x));
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}


NN nn_alloc(size_t *layers, size_t arch_count) 
{
    NN_ASSERT(arch_count > 0);

    NN nn;

    nn.count = arch_count - 1; // Inner Layers - Input layer.

    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);


    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);


    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, layers[0]);

    for (size_t i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = mat_alloc(nn.as[i - 1].cols, layers[i]);
        nn.bs[i-1] = mat_alloc(1, layers[i]);
        nn.as[i]   = mat_alloc(1, layers[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; ++i) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

#endif // NN_IMPLEMENTATION
