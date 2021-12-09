#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "test_class.h"
#include "Bitmap.h"

std::shared_ptr<TestClass> CreateTestClass_Generated();

int main(int argc, const char** argv)
{
  //test_class_cpu();
  //test_class_gpu();
  
  bool onGPU = true;
  std::shared_ptr<TestClass> pImpl = nullptr;
  if(onGPU)
    pImpl = CreateTestClass_Generated();
  else
    pImpl = std::make_shared<TestClass>();

  std::vector<uint> pixelData(WIN_WIDTH*WIN_HEIGHT);  
  pImpl->MainFuncBlock(WIN_WIDTH, WIN_HEIGHT, pixelData.data(), 1);
  
  if(onGPU)
    SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  pImpl = nullptr;
  return 0;
}
