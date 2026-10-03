#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<RandomAdd> CreateRandomAdd_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures RandomAdd_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::shared_ptr<RandomAdd> pImpl = nullptr;
  ArgParser args(argc, argv);

  const uint32_t ITER_NUM  = args.getOptionValue<int>("--iters", 256*256); // number of InterlockedAdd3f calls
  const uint32_t CELLS_NUM = args.getOptionValue<int>("--cells", 256);     // number of float4 pixels in buffer
  const uint32_t SAME_NUM  = args.getOptionValue<int>("--same",  1);       // consecutive iterations which add to the same pixel
  const uint32_t RUNS_NUM  = args.getOptionValue<int>("--runs",  1);       // all runs accumulate into the same buffer
  std::vector<LiteMath::float4> outputArray(CELLS_NUM, LiteMath::float4(0.0f));

  bool onGPU = args.hasOption("--gpu");
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = RandomAdd_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateRandomAdd_Generated(ctx, ITER_NUM);
  }
  else
  #endif
    pImpl = std::make_shared<RandomAdd>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  for(uint32_t run = 0; run < RUNS_NUM; run++)
    pImpl->AddOnes(ITER_NUM, (float*)outputArray.data(), CELLS_NUM, SAME_NUM);

  // reference: integer histogram of the same pseudo-random indices
  //
  std::vector<uint32_t> reference(outputArray.size(), 0);
  for(uint32_t i=0; i < ITER_NUM; i++)
    reference[PseudoRandomIndex(i / SAME_NUM, CELLS_NUM)] += RUNS_NUM;

  const LiteMath::float4 val(1.0f, 2.0f, 3.0f, 0.0f); // 4-th channel is not used and must stay zero

  for(int i=0;i<std::min<int>(10, CELLS_NUM);i++)
    std::cout << i << "\t(" << outputArray[i].x << ", " << outputArray[i].y << ", " << outputArray[i].z << ", " << outputArray[i].w << ")\t" << reference[i] << std::endl;

  bool   passed = true;
  double summ[4] = {0.0, 0.0, 0.0, 0.0};
  for(size_t i=0;i<outputArray.size();i++)
  {
    const LiteMath::float4 expected = val*float(reference[i]);
    for(int k=0;k<4;k++)
    {
      summ[k] += outputArray[i][k];
      if(outputArray[i][k] != expected[k])
      {
        std::cout << "FAILED at " << i << "[" << k << "]: " << outputArray[i][k] << " != " << expected[k] << std::endl;
        passed = false;
      }
    }
    if(!passed)
      break;
  }
  for(int k=0;k<4;k++)
  {
    const double expected = double(ITER_NUM)*double(RUNS_NUM)*val[k];
    std::cout << "summ[" << k << "] = " << summ[k] << " (expected " << expected << ")" << std::endl;
    if(summ[k] != expected)
      passed = false;
  }
  if(onGPU)
  {
    float timings[4] = {0,0,0,0};
    pImpl->GetExecutionTime("kernel1D_AddOnes", timings);
    std::cout << "kernel1D_AddOnes(exec) = " << timings[0] << " ms (avg), " << timings[1] << " ms (min), " << timings[2] << " ms (max)" << std::endl;
  }
  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

  std::vector<float> outputFlat(outputArray.size()*4);
  for(size_t i=0;i<outputArray.size();i++)
    for(int k=0;k<4;k++)
      outputFlat[i*4+k] = outputArray[i][k];
  if(outputFlat.size() <= 4*1024)
    JSONLog::write("array", outputFlat);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return passed ? 0 : 1;
}
