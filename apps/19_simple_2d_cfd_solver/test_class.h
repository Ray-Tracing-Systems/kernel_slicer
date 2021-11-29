#ifndef TEST_TEST_CLASS_H
#define TEST_TEST_CLASS_H

#include <vector>

using std::vector;

class Solver {

public:
    int size = 0;
    float dt = 0.1;
    float visc = 1;
    float diff = 0.5;

    static constexpr int STEPS_NUM = 20;

    vector<float> density;
    vector<float> density_prev;

    vector<float> vx;
    vector<float> vy;

    vector<float> vx0;
    vector<float> vy0;

    Solver();

    void setParameters(int size, const vector<float> &density, const vector<float> &vx, const vector<float> &vy, float dt, float visc, float diff);

    void perform(float *out_density);

    void kernel1D_set_bounds(int N, int b, float *x);

    void kernel1D_AddSources(int N, float dt_, float *v, const float *v0);

    void kernel2D_Diffuse(int h, int w, int b, float *x, const float *x0, float diffuse);

    void kernel2D_Advect(int h, int w, float dt_, int b, float *d, const float *d0, const float *u, const float *v);

    void kernel2D_Project_1(int h, int w, float *u, float *v, float *p, float *div);

    void kernel2D_Project_2(int h, int w, float *u, float *v, float *p, float *div);

    void kernel2D_Project_3(int h, int w, float *u, float *v, float *p, float *div);

};


#endif //TEST_TEST_CLASS_H
