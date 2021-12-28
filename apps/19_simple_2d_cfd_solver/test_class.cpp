#include "test_class.h"
#include <vector>
#include <cstring>

using std::vector;

void Solver::kernel1D_set_bounds(int N, int b, float *dst, float *x) {
    for (int i = 1; i < N; i++) {
        dst[i * N] = b == 1 ? -x[1 + i * N] : x[1 + i * N];
        dst[N - 1 + i * N] = b == 1 ? -x[N - 2 + i * N] : x[N - 2 + i * N];
        dst[i] = b == 2 ? -x[i + N] : x[i + N];
        dst[i + (N - 1) * N] = b == 2 ? -x[i + (N - 2) * N] : x[i + (N - 2) * N];
    }
    dst[0] = 0.5f * (x[1] + x[N]);
    dst[(N - 1) * N] = 0.5f * (x[1 + (N - 1) * N] + x[(N - 2) * N]);
    dst[N - 1] = 0.5f * (x[N - 2] + x[N - 1 + N]);
    dst[N - 1 + (N - 1) * N] = 0.5f * (x[N - 2 + (N - 1) * N] + x[N - 1 + (N - 2) * N]);
}


void Solver::perform(float *out_density) {
    // VELOCITY_STEP
    int copy_size = size*size*sizeof(float);
    //// add_sources
    kernel1D_AddSources(size * size, dt, vx.data(), vx0.data());
    kernel1D_AddSources(size * size, dt, vy.data(), vy0.data());

    //// diffuse
    for (int k = 0; k < STEPS_NUM; k++) {
        if (k % 2 == 0) {
            kernel2D_diffuse((size - 1), size - 1, 1, extra_vec.data(), vx0.data(), vx.data(), visc);
            memcpy(transfer_vec.data(), extra_vec.data(), copy_size);
            kernel1D_set_bounds((int) size, 1, extra_vec.data(), transfer_vec.data());
        } else {
            kernel2D_diffuse(size - 1, size - 1, 1, vx0.data(), extra_vec.data(), vx.data(), visc);
            memcpy(transfer_vec.data(), vx0.data(), copy_size);
            kernel1D_set_bounds((int) size, 1, vx0.data(), transfer_vec.data());
        }

    }

    for (int k = 0; k < STEPS_NUM; k++) {
        if (k % 2 == 0) {
            kernel2D_diffuse(size - 1, size - 1, 2, extra_vec.data(), vy0.data(), vy.data(), visc);
            memcpy(transfer_vec.data(), extra_vec.data(), copy_size);
            kernel1D_set_bounds((int) size, 2, extra_vec.data(), transfer_vec.data());
        } else {
            kernel2D_diffuse(size - 1, size - 1, 2, vy0.data(), extra_vec.data(), vy.data(), visc);
            memcpy(transfer_vec.data(), vy0.data(), copy_size);
            kernel1D_set_bounds((int) size, 2, vy0.data(), transfer_vec.data());
        }

    }

    //// project
    kernel2D_Project_1(size - 1, size - 1, vx0.data(), vy0.data(), vx.data(), vy.data());
    memcpy(transfer_vec.data(), vy.data(), copy_size);
    kernel1D_set_bounds(size, 0, vy.data(), transfer_vec.data());
    memcpy(transfer_vec.data(), vx.data(), copy_size);
    kernel1D_set_bounds(size, 0, vx.data(), transfer_vec.data());
    for (int k = 0; k < STEPS_NUM; k++) {
        if (k % 2 == 0) {
            kernel2D_Project_2(size - 1, size - 1, extra_vec.data(), vx.data(), vy.data());
            memcpy(transfer_vec.data(), extra_vec.data(), copy_size);
            kernel1D_set_bounds(size, 0, extra_vec.data(), transfer_vec.data());
        } else {
            kernel2D_Project_2(size - 1, size - 1, vx.data(), extra_vec.data(), vy.data());
            memcpy(transfer_vec.data(), vx.data(), copy_size);
            kernel1D_set_bounds(size, 0, vx.data(), transfer_vec.data());
        }

    }

    kernel2D_Project_3(size - 1, size - 1, vx0.data(), vy0.data(), vx.data(), vy.data());
    memcpy(transfer_vec.data(), vx0.data(), copy_size);
    kernel1D_set_bounds(size, 1, vx0.data(), transfer_vec.data());
    memcpy(transfer_vec.data(), vy0.data(), copy_size);
    kernel1D_set_bounds(size, 2, vy0.data(), transfer_vec.data());

    //// advect
    kernel2D_Advect(size - 1, size - 1, dt, 1, vx.data(), vx0.data(), vx0.data(), vy0.data());
    memcpy(transfer_vec.data(), vx.data(), copy_size);
    kernel1D_set_bounds(size, 1, vx.data(), transfer_vec.data());

    kernel2D_Advect(size - 1, size - 1, dt, 1, vy.data(), vy0.data(), vx0.data(), vy0.data());
    memcpy(transfer_vec.data(), vy.data(), copy_size);
    kernel1D_set_bounds(size, 2, vy.data(), transfer_vec.data());


    //// project
    kernel2D_Project_1(size - 1, size - 1, vx.data(), vy.data(), vx0.data(), vy0.data());
    memcpy(transfer_vec.data(), vy0.data(), copy_size);
    kernel1D_set_bounds(size, 0, vy0.data(), transfer_vec.data());
    memcpy(transfer_vec.data(), vx0.data(), copy_size);
    kernel1D_set_bounds(size, 0, vx0.data(), transfer_vec.data());
    for (int k = 0; k < STEPS_NUM; k++) {
        if (k % 2 == 0) {
            kernel2D_Project_2(size - 1, size - 1, extra_vec.data(), vx0.data(), vy0.data());
            memcpy(transfer_vec.data(), extra_vec.data(), copy_size);
            kernel1D_set_bounds(size, 0, extra_vec.data(), transfer_vec.data());
        } else {
            kernel2D_Project_2(size - 1, size - 1, vx0.data(), extra_vec.data(), vy0.data());
            memcpy(transfer_vec.data(), vx0.data(), copy_size);
            kernel1D_set_bounds(size, 0, vx0.data(), transfer_vec.data());
        }

    }

    kernel2D_Project_3(size - 1, size - 1, vx.data(), vy.data(), vx0.data(), vy0.data());
    memcpy(transfer_vec.data(), vx.data(), copy_size);
    kernel1D_set_bounds(size, 1, vx.data(), transfer_vec.data());
    memcpy(transfer_vec.data(), vy.data(), copy_size);
    kernel1D_set_bounds(size, 2, vy.data(), transfer_vec.data());

    //DENCITY_STEP

    //// add_sources
    kernel1D_AddSources(size * size, dt, density.data(), density_prev.data());

    //// diffuse
    for (int k = 0; k < STEPS_NUM; k++) {
        if (k % 2 == 0) {
            kernel2D_diffuse(size - 1, size - 1, 0, extra_vec.data(), density_prev.data(), density.data(), diff);
            memcpy(transfer_vec.data(), extra_vec.data(), copy_size);
            kernel1D_set_bounds(size, 0, extra_vec.data(), transfer_vec.data());
        } else {
            kernel2D_diffuse(size - 1, size - 1, 0, density_prev.data(), extra_vec.data(), density.data(), diff);
            memcpy(transfer_vec.data(), density_prev.data(), copy_size);
            kernel1D_set_bounds(size, 0, density_prev.data(), transfer_vec.data());
        }
    }

    //// advect
    kernel2D_Advect(size - 1, size - 1, dt, 0, density.data(),
                    density_prev.data(), vx.data(), vy.data());
    memcpy(transfer_vec.data(), density.data(), copy_size);
    kernel1D_set_bounds(size, 0, density.data(), transfer_vec.data());

    memcpy(out_density, density.data(), density.size() * sizeof(float));
}

void Solver::kernel1D_AddSources(int N, float dt_, float *v, const float *v0) {
    for (int i = 0; i < N; i++) {
        v[i] += dt_ * v0[i];
    }
}

void Solver::kernel2D_diffuse(int h, int w, int b, float *src, float *x, const float *x0, float diffuse) {
    for (int i = 1; i < h; i++) {
        for (int j = 1; j < w; j++) {
            float a = dt * diffuse * (float) (size - 2) * (float) (size - 2);
            src[i + j * size] = (x0[i + j * size] + a * (x[i - 1 + j * size] + x[i + 1 + j * size] +
                                                         x[i + (j - 1) * size] + x[i + (j + 1) * size])) /
                                (1 + 4 * a);
        }
    }

//    for (int k = 1; k <= h; k++) {
//            int i = (k - 1) / (size - 2) + 1;
//            int j = (k + 1) % (size - 2) + 2;
//            while (j > (size - 2)) {
//                j -= (size - 2);
//            }
//            float a = dt * diffuse * (float) (size - 2) * (float) (size - 2);
//            src[i + j * size] = (x0[i + j * size] + a * (x[i - 1 + j * size] + x[i + 1 + j * size] +
//                                                         x[i + (j - 1) * size] + x[i + (j + 1) * size])) /
//                                (1 + 4 * a);
//    }
}


void
Solver::kernel2D_Advect(int h, int w, float dt_, int b, float *d, const float *d0, const float *u, const float *v) {

    for (int i = 1; i < h; i++) {
        for (int j = 1; j < w; j++) {
            int N = size - 2;
            int i0, j0, i1, j1;
            float x, y, s0, t0, s1, t1, dt0;
            dt0 = dt_ * (float) (N);
            x = (float) i - dt0 * u[i + j * size];
            y = (float) j - dt0 * v[i + j * size];
            if (x < 0.5f) x = 0.5f;
            if (x > (float) N + 0.5f) x = (float) ((float) N + 0.5f);
            i0 = (int) x;
            i1 = i0 + 1;
            if (y < 0.5f) y = 0.5f;
            if (y > (float) N + 0.5f) y = (float) ((float) N + 0.5f);
            j0 = (int) y;
            j1 = j0 + 1;
            s1 = x - (float) i0;
            s0 = 1 - s1;
            t1 = y - (float) j0;
            t0 = 1 - t1;
            d[i + j * size] = s0 * (t0 * d0[i0 + j0 * size] + t1 * d0[i0 + j1 * size]) +
                              s1 * (t0 * d0[i1 + j0 * size] + t1 * d0[i1 + j1 * size]);
        }
    }
}

void Solver::kernel2D_Project_1(int h, int w, float *u, float *v, float *p, float *div) {
    for (int i = 1; i < h; i++) {
        for (int j = 1; j < w; j++) {
            float f = 1.0f / (float) (size - 2);

            div[i + j * size] = -0.5f * f * (u[i + 1 + j * size] - u[i - 1 + j * size] +
                                             v[i + (j + 1) * size] - v[i + (j - 1) * size]);
            p[i + j * size] = 0;
        }
    }

}

void Solver::kernel2D_Project_2(int h, int w, float *src, float *p, float *div) {

    for (int i = 1; i < h; i++) {
        for (int j = 1; j < w; j++) {
            src[i + j * size] = (div[i + j * size] + p[i - 1 + j * size] + p[i + 1 + j * size] +
                                 p[i + (j - 1) * size] + p[i + (j + 1) * size]) / 4;
        }
    }
}

void Solver::kernel2D_Project_3(int h, int w, float *u, float *v, float *p, float *div) {

    for (int i = 1; i < h; i++) {
        for (int j = 1; j < w; j++) {
            float z = 1.0f / (float) (size-2);
            u[i + j * size] -= 0.5f * (p[i + 1 + j * size] - p[i - 1 + j * size]) / z;
            v[i + j * size] -= 0.5f * (p[i + (j + 1) * size] - p[i + (j - 1) * size]) / z;
        }
    }
}

void Solver::setParameters(int size, const vector<float> &density, const vector<float> &vx, const vector<float> &vy,
                           float dt, float visc,
                           float diff) {
    this->size = size;
    this->density = density;
    this->density_prev = this->density;
    this->vx = vx;
    this->extra_vec.resize(size * size);
    this->transfer_vec.resize(size * size);
    this->vx0 = this->vx;
    this->vy = vy;
    this->vy0 = this->vy;
    this->dt = dt;
    this->visc = visc;
    this->diff = diff;
}

Solver::Solver() = default;
