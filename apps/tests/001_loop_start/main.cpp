#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>

#include "test_class.h"

#include "vk_context.h"
std::shared_ptr<Numbers> CreateNumbers_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int32_t array_summ_cpu(const std::vector<int32_t>& array);
int32_t array_summ_gpu(const std::vector<int32_t>& array);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<int32_t> array(1024);
  for(size_t i=0;i<array.size();i++)
  {
    if(i%3 == 0)
      array[i] = i;
    else
      array[i] = -i;
  }

  std::shared_ptr<Numbers> pImpl = nullptr;
  bool onGPU = true;
  
  if(onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers);
    pImpl = CreateNumbers_Generated(ctx, array.size());
  }
  else
    pImpl = std::make_shared<Numbers>();

  pImpl->CommitDeviceData();
  pImpl->CalcArraySumm(array.data(), unsigned(array.size()));

  if(onGPU)
    std::cout << "[gpu]: array summ = " << pImpl->m_summ << std::endl;
  else
    std::cout << "[cpu]: array summ = " << pImpl->m_summ << std::endl;


  return 0;
}
