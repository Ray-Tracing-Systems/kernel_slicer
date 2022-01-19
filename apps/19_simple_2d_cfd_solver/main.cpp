#include "stb_image_write.h"
#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "vk_utils.h"
#include "vk_pipeline.h"
#include "vk_copy.h"
#include "vk_buffers.h"
#include "test_class_gpu.h"
#include "test_class.h"
#include "test_class_generated.h"

std::vector<float> solve_cpu(int N, const std::vector<float>& density, const std::vector<float>& vx, const std::vector<float>& vy) {
    Solver s = Solver();
    s.setParameters(N, density, vx, vy, 0.033, 0, 0);

    std::vector<float> out(N * N);
    for (int i = 0; i < 50; ++ i) {
        std::cout << i <<  std::endl;
        s.perform(out.data());
        save_image(N, "images/" + std::to_string(i) + ".jpeg", out);
    }
    return out;
}

std::vector<float> solve_gpu(int N, const std::vector<float>& density, const std::vector<float>& vx, const std::vector<float>& vy);

int main(int argc, const char** argv) {
//    srand(42);
    const int N = 100;
    std::vector<float> density(N * N);
    std::vector<float> vx(N*N);
    std::vector<float> vy(N*N);

    for (int i = 0; i < N * N; ++i)
        density[i] = randfrom(0, 1);

    for (int i = 0; i < N * N; ++i)
        vx[i] = randfrom(-5, 5);

    for (int i = 0; i < N * N; ++i)
        vy[i] = randfrom(-5, 5);

    auto cpu_res = solve_cpu(N, density, vx, vy);
    save_image(N,"zout_cpu.bmp", cpu_res);

    auto gpu_res = solve_gpu(N, density, vx, vy);
    save_image(N, "zout_gpu.bmp", gpu_res);

    return 0;
}