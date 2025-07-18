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
#define MAT_PRINT(m) mat_print(m, #m);

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
void mat_print(Mat m, const char *name);
void mat_fill(Mat m, float x);
void mat_sig(Mat m);
void mat_copy(Mat dest, Mat a);

float rand_float(void);
float sigmoidf(float x);


#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
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

void mat_print(Mat m, const char *name)
{
    printf("%s = ", name);
    printf("[\n");
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("    %f, ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
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
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

float sigmoidf(float x)
{
    return 1 / (1 * exp(-x));
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

#endif // NN_IMPLEMENTATION
