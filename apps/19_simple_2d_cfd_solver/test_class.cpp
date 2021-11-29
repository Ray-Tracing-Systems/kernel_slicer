#include "test_class.h"
#include <vector>
#include <cstring>

using std::vector;

void Solver::kernel1D_set_bounds(int N, int b, float *x) {
    for (int i = 1; i < N; i++) {
        x[i * N] = b == 1 ? -x[1 + i * N] : x[1 + i * N];
        x[N - 1 + i * N] = b == 1 ? -x[N - 2 + i * N] : x[N - 2 + i * N];
        x[i] = b == 2 ? -x[i + N] : x[i + N];
        x[i + (N - 1) * N] = b == 2 ? -x[i + (N - 2) * N] : x[i + (N - 2) * N];
    }
    x[0] = 0.5 * (x[1] + x[N]);
    x[(N - 1) * N] = 0.5 * (x[1 + (N - 1) * N] + x[(N - 2) * N]);
    x[N - 1] = 0.5 * (x[N - 2] + x[N - 1 + N]);
    x[N - 1 + (N - 1) * N] = 0.5 * (x[N - 2 + (N - 1) * N] + x[N - 1 + (N - 2) * N]);
}

void Solver::perform(float *out_density) {
    //velocity_step
    kernel1D_AddSources(size * size, dt, vx.data(), vx0.data());
    kernel1D_AddSources(size * size, dt, vy.data(), vy0.data());

    //swap(vx0, vx);
    //swap(vy0, vy);

    for (int k = 0; k < STEPS_NUM; k++) {
        kernel2D_Diffuse(size - 2, size - 2, 1, vx0.data(), vx.data(), visc);
        kernel1D_set_bounds((int) size, 1, vx0.data());

    }
    for (int k = 0; k < STEPS_NUM; k++) {
        kernel2D_Diffuse(size - 2, size - 2, 2, vy0.data(), vy.data(), visc);
        kernel1D_set_bounds((int) size, 2, vy0.data());
    }

    kernel2D_Project_1(size - 2, size - 2, vx0.data(), vy0.data(), vx.data(), vy.data());
    for (int k = 0; k < STEPS_NUM; k++) {
        kernel2D_Project_2(size - 2, size - 2, vx0.data(), vy0.data(), vx.data(), vy.data());
        kernel1D_set_bounds(size, 0, vx.data());
    }

    kernel2D_Project_3(size - 2, size - 2, vx0.data(), vy0.data(), vx.data(), vy.data());

    //swap(vx0, vx);
    //swap(vy0, vy);

    kernel2D_Advect(size - 2, size - 2, dt, 1, vx.data(), vx0.data(), vx0.data(), vy0.data());
    kernel1D_set_bounds(size, 1, vx.data());

    kernel2D_Advect(size - 2, size - 2, dt, 2, vy.data(), vy0.data(), vx0.data(), vy0.data());
    kernel1D_set_bounds(size, 2, vy.data());


    kernel2D_Project_1(size - 2, size - 2, vx.data(), vy.data(), vx0.data(), vy0.data());
    kernel1D_set_bounds(size, 0, vy0.data());
    kernel1D_set_bounds(size, 0, vx0.data());
    for (int k = 0; k < STEPS_NUM; k++) {
        kernel2D_Project_2(size - 2, size - 2, vx.data(), vy.data(), vx0.data(), vy0.data());
        kernel1D_set_bounds(size, 0, vx0.data());
    }

    kernel2D_Project_3(size - 2, size - 2, vx.data(), vy.data(), vx0.data(), vy0.data());
    kernel1D_set_bounds(size, 1, vx.data());
    kernel1D_set_bounds(size, 2, vy.data());

    //density step
    //
    kernel1D_AddSources(size * size, dt, density.data(), density_prev.data());
    //swap(density_prev, density);
    
    for (int k = 0; k < STEPS_NUM; k++) {
        kernel2D_Diffuse(size - 2, size - 2, 0, density_prev.data(), density.data(), diff);
        kernel1D_set_bounds(size, 0, density_prev.data());
    }

    //swap(density_prev, density);
    kernel2D_Advect(size - 2, size - 2, dt, 0, density.data(), density_prev.data(), vx.data(), vy.data());

    kernel1D_set_bounds(size, 0, density.data());

    memcpy(out_density, density.data(), density.size() * sizeof(float));
}

void Solver::kernel1D_AddSources(int N, float dt_, float *v, const float *v0) {
    for (int i = 0; i < N; i++) {
        v[i] += dt_ * v0[i];
    }
}

void Solver::kernel2D_Diffuse(int h, int w, int b, float *x, const float *x0, float diffuse) {

    for (int i = 1; i <= h; i++) {
        for (int j = 1; j <= w; j++) {
            float a = dt * diffuse * (float) (size - 2) * (float) (size - 2);
            x[i + j * size] = (x0[i + j * size] + a * (x[i - 1 + j * size] + x[i + 1 + j * size] +
                                                       x[i + (j - 1) * size] + x[i + (j + 1) * size])) /
                              (1 + 4 * a);
        }
    }
}


void
Solver::kernel2D_Advect(int h, int w, float dt_, int b, float *d, const float *d0, const float *u, const float *v) {

    for (int i = 1; i <= h; i++) {
        for (int j = 1; j <= w; j++) {
            int N = h;
            int i0, j0, i1, j1;
            float x, y, s0, t0, s1, t1, dt0;
            dt0 = dt_ * (float) (N);
            x = i - dt0 * u[i + j * size];
            y = j - dt0 * v[i + j * size];
            if (x < 0.5f) x = 0.5f;
            if (x > N + 0.5f) x = (float) (N + 0.5f);
            i0 = (int) x;
            i1 = i0 + 1;
            if (y < 0.5f) y = 0.5f;
            if (y > N + 0.5f) y = (float) (N + 0.5f);
            j0 = (int) y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;
            d[i + j * size] = s0 * (t0 * d0[i0 + j0 * size] + t1 * d0[i0 + j1 * size]) +
                              s1 * (t0 * d0[i1 + j0 * size] + t1 * d0[i1 + j1 * size]);
        }
    }
}

void Solver::kernel2D_Project_1(int h, int w, float *u, float *v, float *p, float *div) {
    for (int i = 1; i <= h; i++) {
        for (int j = 1; j <= w; j++) {
            float h = 1.0f / (float) (size - 2);

            div[i + j * size] = -0.5f * h * (u[i + 1 + j * size] - u[i - 1 + j * size] +
                                            v[i + (j + 1) * size] - v[i + (j - 1) * size]);
            p[i + j * size] = 0;
        }
    }

}

void Solver::kernel2D_Project_2(int h, int w, float *u, float *v, float *p, float *div) {

    for (int i = 1; i <= h; i++) {
        for (int j = 1; j <= w; j++) {
            p[i + j * size] = (div[i + j * size] + p[i - 1 + j * size] + p[i + 1 + j * size] +
                               p[i + (j - 1) * size] + p[i + (j + 1) * size]) / 4;
        }
    }
}

void Solver::kernel2D_Project_3(int h, int w, float *u, float *v, float *p, float *div) {

    for (int i = 1; i <= h; i++) {
        for (int j = 1; j <= w; j++) {
            float z = 1.0f / (float) h;
            u[i + j * size] -= 0.5 * (p[i + 1 + j * size] - p[i - 1 + j * size]) / z;
            v[i + j * size] -= 0.5 * (p[i + (j + 1) * size] - p[i + (j - 1) * size]) / z;
        }
    }
}

void Solver::setParameters(int size, const vector<float> &density, const vector<float> &vx, const vector<float> &vy, float dt, float visc,
                           float diff) {
    this->size = size;
    this->density = density;
    this->density_prev = this->density;
    this->vx = vx;
    this->vx0 = this->vx;
    this->vy = vy;
    this->vy0 = this->vy;
    this->dt = dt;
    this->visc = visc;
    this->diff = diff;
}

Solver::Solver() = default;

