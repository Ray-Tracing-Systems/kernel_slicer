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

std::vector<float> solve_cpu(int N) {
    std::vector<float> out(N * N);
    std::vector<float> density(N * N);
    std::vector<float> vx(N*N);
    std::vector<float> vy(N*N);

    for (int i = 0; i < N * N; ++i) 
      density[i] = randfrom(0, 1);

    for (int i = 0; i < N * N; ++i)
      vx[i] = randfrom(-5, 5);
    
    for (int i = 0; i < N * N; ++i) 
      vy[i] = randfrom(-5, 5);

    Solver s = Solver();
    s.setParameters(N, density, vx, vy, 0.033, 0.001, 0.00001);
    
    for (int i = 0; i < 50; ++ i) {
        s.perform(out.data());
        save_image("images/" + std::to_string(i) + ".jpeg", out);
    }
    return out;
}

std::vector<float> solve_gpu(int N);

int main() {
    //auto x = solve_gpu();
    auto cpu_res = solve_cpu(50);
    save_image("z_out_cpu.jpg", cpu_res);
    return 0;
}