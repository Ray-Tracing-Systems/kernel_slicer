#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "vk_context.h"
std::shared_ptr<Numbers> CreateNumbers_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 

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
  
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  std::shared_ptr<Numbers> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateNumbers_Generated(ctx, array.size());
  }
  else
    pImpl = std::make_shared<Numbers>();

  pImpl->CalcArraySumm(array.data(), unsigned(array.size()));

  JSONLog::write("array summ", pImpl->m_summ);
  if(onGPU)
    JSONLog::saveToFile("zout_gpu.json");
  else
    JSONLog::saveToFile("zout_cpu.json");
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("CalcArraySumm", timings);
  std::cout << "CalcArraySumm(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "CalcArraySumm(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "CalcArraySumm(ovrh) = " << timings[3]              << " ms " << std::endl;
  return 0;
}
