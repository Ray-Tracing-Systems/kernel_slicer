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
std::shared_ptr<BlockScan> CreateBlockScan_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif
#ifdef USE_ISPC
std::shared_ptr<BlockScan> CreateBlockScan_ISPC();
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

  std::shared_ptr<BlockScan> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl    = CreateBlockScan_Generated(ctx, MAX_SIZE);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateBlockScan_ISPC();
  else
  #endif
    pImpl = std::make_shared<BlockScan>();
  pImpl->CommitDeviceData();

  std::vector<uint> sizes = {1, 255, 256, 257, 65535, 65536, 65537, 1000003, MAX_SIZE};
  std::mt19937 gen(12345);
  std::uniform_int_distribution<uint> sizeDist(1, MAX_SIZE);
  for(int i = 0; i < runs; i++)
    sizes.push_back(sizeDist(gen));

  std::uniform_int_distribution<int> valueDist(0, 10);    // the total sum fits into 'int'
  std::vector<int> data(MAX_SIZE), result(MAX_SIZE), ref(MAX_SIZE);
  for(auto& x : data)
    x = valueDist(gen);

  int failed = 0;
  for(size_t i = 0; i < sizes.size(); i++)
  {
    const uint size = sizes[i];
    std::fill(result.begin(), result.end(), -1);
    pImpl->ExclusiveScan(data.data(), result.data(), size);
    std::exclusive_scan(data.begin(), data.begin() + size, ref.begin(), 0);

    int64_t firstDiff = -1;
    for(uint j = 0; j < size; j++)
    {
      if(result[j] != ref[j])
      {
        firstDiff = j;
        break;
      }
    }
    if(firstDiff >= 0)
      failed++;

    std::cout << "size = " << size << ", last = " << result[size-1];
    if(firstDiff >= 0)
      std::cout << "  FAILED at " << firstDiff << ": " << result[firstDiff] << " != " << ref[firstDiff];
    std::cout << std::endl;
    JSONLog::write("last_" + std::to_string(i), result[size-1]);
    JSONLog::write("diff_" + std::to_string(i), firstDiff);
  }

  JSONLog::write("check", failed == 0 ? "PASSED!" : "FAILED!");
  JSONLog::saveToFile(onGPU ? "zout_gpu.json" : (onISPC ? "zout_ispc.json" : "zout_cpu.json"));
  std::cout << (failed == 0 ? "PASSED" : "FAILED") << std::endl;
  return 0;
}
