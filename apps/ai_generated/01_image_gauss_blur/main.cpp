#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "blur.h"
#include "Image2d.h"

int main()
{
  const int width = 1024;
  const int height = 768;
  const int kernelRadius = 5;
  const float sigma = 2.0f;

  // Загрузка входного изображения
  std::vector<float> inputImage(width * height * 4);
  std::vector<float> outputImage(width * height * 4);
    
  // Здесь должна быть загрузка изображения в inputImage
    
  // Создание экземпляра класса
  bool onGPU = false; // или false для CPU реализации
  std::shared_ptr<GaussianBlur> pImpl = nullptr;
    
  //if (onGPU)
  //{
  //  auto ctx = vk_utils::globalContextGet(false, 0);
  //  pImpl = CreateGaussianBlur_Generated(ctx, width * height);
  //}
  //else
  //{
    pImpl = std::make_shared<GaussianBlur>(width, height);
  //}
  
  // Выполнение размытия
  pImpl->CommitDeviceData();
  pImpl->Run(width, height, inputImage.data(), outputImage.data(), kernelRadius, sigma);
  
  // Сохранение результата
  if (onGPU)
  {
    LiteImage::SaveBMP("blurred_gpu.bmp", outputImage.data(), width, height);
  }
  else
  {
    LiteImage::SaveBMP("blurred_cpu.bmp", outputImage.data(), width, height);
  }
  
  return 0;
}