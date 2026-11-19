#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <string>
#include <numeric>

#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"
#include "test_class.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<BlockReduce> CreateBlockReduce_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif
#ifdef USE_ISPC
std::shared_ptr<BlockReduce> CreateBlockReduce_ISPC();
#endif

// Usage: testapp [--gpu | --ispc] [--runs N]
// Fixed sizes (block borders and the maximum size) are checked first, then N random sizes.
//
int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  const bool onGPU  = args.hasOption("--gpu");
  const bool onISPC = args.hasOption("--ispc");
  const int  runs   = args.getOptionValue<int>("--runs", 4);

  std::shared_ptr<BlockReduce> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl    = CreateBlockReduce_Generated(ctx, MAX_SIZE);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateBlockReduce_ISPC();
  else
  #endif
    pImpl = std::make_shared<BlockReduce>();
  pImpl->CommitDeviceData();

  std::vector<uint> sizes = {1, 255, 256, 257, 65535, 65536, 65537, 1000003, MAX_SIZE};
  std::mt19937 gen(12345);
  std::uniform_int_distribution<uint> sizeDist(1, MAX_SIZE);
  for(int i = 0; i < runs; i++)
    sizes.push_back(sizeDist(gen));

  std::uniform_int_distribution<int> valueDist(-100, 100);
  std::vector<int> data(MAX_SIZE);
  for(auto& x : data)
    x = valueDist(gen);

  int failed = 0;
  for(size_t i = 0; i < sizes.size(); i++)
  {
    const uint size = sizes[i];
    int sum = 0;
    pImpl->Reduce(data.data(), &sum, size);
    const int ref = std::accumulate(data.begin(), data.begin() + size, 0);
    if(sum != ref)
      failed++;
    std::cout << "size = " << size << ", sum = " << sum << ", ref = " << ref << (sum == ref ? "" : "  FAILED!") << std::endl;
    JSONLog::write("sum_" + std::to_string(i), sum);
  }

  JSONLog::write("check", failed == 0 ? "PASSED!" : "FAILED!");
  JSONLog::saveToFile(onGPU ? "zout_gpu.json" : (onISPC ? "zout_ispc.json" : "zout_cpu.json"));
  std::cout << (failed == 0 ? "PASSED" : "FAILED") << std::endl;

  pImpl = nullptr;           // Vulkan objects of the class first, then the context
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
