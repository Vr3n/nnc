#define NN_IMPLEMENTATION
#include "nn.h"


typedef struct {
    Mat x, a1, a2;

    Mat w1, b1
    Mat w2, b2;

} Xor;



void forwardXor(Xor m)
{
    // Layer 1
    mat_dot(m.a1, m.x,m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    // Layer 2
    mat_dot(m.a2,m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);
}

float cost (Xor m, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == m.a2.cols);

    size_t n = ti.rows;
    float results = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m.x, x);
        forwardXor(m);

        size_t mc = to.cols;

        for (size_t j = 0; j < mc; ++j) {
            float y_diff = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
            results += y_diff * y_diff;
        }

    }
    return results / n;
}

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};


int main (void)
{
    srand(42);

    size_t stride = 3;
    size_t n = sizeof(td) / sizeof(td[0]) / stride;
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = &td[2] // same as td + 2
    };

    MAT_PRINT(ti);
    MAT_PRINT(to);

    Xor xr;

    xr.x = mat_alloc(1, 2);

    xr.w1 = mat_alloc(2, 2);
    xr.b1 = mat_alloc(1, 2);
    xr.a1 = mat_alloc(1, 2);

    xr.w2 = mat_alloc(2, 1);
    xr.b2 = mat_alloc(1, 1);
    xr.a2 = mat_alloc(1, 1);

    mat_rand(xr.w1, 0, 1);
    mat_rand(xr.b1, 0, 1);
    mat_rand(xr.w2, 0, 1);
    mat_rand(xr.b2, 0, 1);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {

            MAT_AT(xr.x, 0, 0) = i;
            MAT_AT(xr.x, 0, 1) = j;
            forwardXor(xr);
            float y = *xr.a2.es;

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    return 0;
}
