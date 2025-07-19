#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>


typedef struct {
    Mat x, a1, a2;

    Mat w1, b1;
    Mat w2, b2;

} Xor;

Xor xor_alloc()
{
    Xor xr;

    xr.x = mat_alloc(1, 2);

    xr.w1 = mat_alloc(2, 2);
    xr.b1 = mat_alloc(1, 2);
    xr.a1 = mat_alloc(1, 2);

    xr.w2 = mat_alloc(2, 1);
    xr.b2 = mat_alloc(1, 1);
    xr.a2 = mat_alloc(1, 1);
    return xr;
}


void forward_xor(Xor m) {
    // Layer 1
    mat_dot(m.a1, m.x, m.w1);
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
        forward_xor(m);

        size_t mc = to.cols;

        for (size_t j = 0; j < mc; ++j) {
            float y_diff = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
            results += y_diff * y_diff;
        }
    }
    return results / n;
}

void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to)
{
    float saved;

    float c = cost(m, ti, to);

    for (size_t i = 0; i < m.w1.rows; ++i) {
        for (size_t j = 0; j < m.w1.cols; ++j) {
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c) / eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b1.rows; ++i) {
        for (size_t j = 0; j < m.b1.cols; ++j) {
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c) / eps;
            MAT_AT(m.b1, i, j) = saved;

        }
    }

    for (size_t i = 0; i < m.w2.rows; ++i) {
        for (size_t j = 0; j < m.w2.cols; ++j) {
            saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c) / eps;
            MAT_AT(m.w2, i, j) = saved;

        }
    }

    for (size_t i = 0; i < m.b2.rows; ++i) {
        for (size_t j = 0; j < m.b2.cols; ++j) {
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c) / eps;
            MAT_AT(m.b2, i, j) = saved;

        }
    }

}


float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

void xor_learn (Xor m, Xor g, float rate)
{

    for (size_t i = 0; i < m.w1.rows; ++i) {
        for (size_t j = 0; j < m.w1.cols; ++j) {
            MAT_AT(m.w1, i, j) -= rate * MAT_AT(g.w1, i, j);
        }
    }

    for (size_t i = 0; i < m.b1.rows; ++i) {
        for (size_t j = 0; j < m.b1.cols; ++j) {
            MAT_AT(m.b1, i, j) -= rate * MAT_AT(g.b1, i, j);
        }
    }

    for (size_t i = 0; i < m.w2.rows; ++i) {
        for (size_t j = 0; j < m.w2.cols; ++j) {
            MAT_AT(m.w2, i, j) -= rate * MAT_AT(g.w2, i, j);
        }
    }

    for (size_t i = 0; i < m.b2.rows; ++i) {
        for (size_t j = 0; j < m.b2.cols; ++j) {
            MAT_AT(m.b2, i, j) -= rate * MAT_AT(g.b2, i, j);
        }
    }

}


int main (void)
{
    size_t arch[] = {28*28, 16, 16, 10};

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN_PRINT(nn);

    return 0;

    size_t stride = 3;
    size_t n = (sizeof(td) / sizeof(td[0])) / stride;
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

    Xor xr = xor_alloc();
    Xor gr = xor_alloc();

    mat_rand(xr.w1, 0.0f, 1.0f);
    mat_rand(xr.b1, 0.0f, 1.0f);
    mat_rand(xr.w2, 0.0f, 1.0f);
    mat_rand(xr.b2, 0.0f, 1.0f);

    float eps = 1e-2;
    float rate = 1e-2;

    for (size_t i = 0; i < 9000;++i) {
        finite_diff(xr, gr, eps, ti, to);
        xor_learn(xr, gr, rate);
        printf("%zu: cost = %f\n", i, cost(xr, ti, to));
    }

    printf("---------------------------------\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            MAT_AT(xr.x, 0, 0) = i;
            MAT_AT(xr.x, 0, 1) = j;

            forward_xor(xr);

            float y = *xr.a2.es;

            printf("%zu ^ %zu: %f\n", i, j, y);
        }
    }


    return 0;
}
