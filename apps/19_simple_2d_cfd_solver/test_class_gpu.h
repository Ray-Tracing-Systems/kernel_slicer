//
// Created by timofeq on 28.11.2021.
//

#ifndef TEST_TEST_CLASS_GPU_H
#define TEST_TEST_CLASS_GPU_H

#include <vector>

std::vector<float> solve_gpu();
double randfrom(double min, double max);
void save_image(int N, const std::string &image_name, std::vector<float> density);

#endif //TEST_TEST_CLASS_GPU_H
