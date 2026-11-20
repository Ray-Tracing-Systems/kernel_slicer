#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <string>

#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"
#include "test_class.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<Histogram> CreateHistogram_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif
#ifdef USE_ISPC
std::shared_ptr<Histogram> CreateHistogram_ISPC();
#endif

// Usage: testapp [--gpu | --ispc] [--runs N]
// Fixed sizes (block borders and the maximum size) are checked first, then N random sizes.
// Half of the values fall into one bin, so atomics of different threads often hit the same address.
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
  const uint MAX_SIZE = 1u << 24;

  std::shared_ptr<Histogram> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl    = CreateHistogram_Generated(ctx, MAX_SIZE);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateHistogram_ISPC();
  else
  #endif
    pImpl = std::make_shared<Histogram>();
  pImpl->CommitDeviceData();

  std::vector<uint> sizes = {1, 255, 4095, 4096, 4097, 65537, 1000003, MAX_SIZE};
  std::mt19937 gen(12345);
  std::uniform_int_distribution<uint> sizeDist(1, MAX_SIZE);
  for(int i = 0; i < runs; i++)
    sizes.push_back(sizeDist(gen));

  std::uniform_int_distribution<uint> valueDist(0, 0xFFFFFFFFu);
  std::vector<uint> data(MAX_SIZE);
  for(auto& x : data)
    x = (valueDist(gen) % 2 == 0) ? 7u : valueDist(gen);

  int failed = 0;
  std::vector<uint> hist(NUM_BINS), ref(NUM_BINS);
  for(size_t i = 0; i < sizes.size(); i++)
  {
    const uint size = sizes[i];
    std::fill(hist.begin(), hist.end(), 0xFFFFFFFFu);
    pImpl->Compute(data.data(), hist.data(), size);

    std::fill(ref.begin(), ref.end(), 0u);
    for(uint j = 0; j < size; j++)
      ref[data[j] % NUM_BINS]++;

    uint64_t checksum = 0;
    int diffBin = -1;
    for(uint b = 0; b < NUM_BINS; b++)
    {
      checksum += uint64_t(hist[b])*(b + 1);
      if(hist[b] != ref[b] && diffBin < 0)
        diffBin = int(b);
    }
    if(diffBin >= 0)
      failed++;
    std::cout << "size = " << size << ", bin[7] = " << hist[7];
    if(diffBin >= 0)
      std::cout << "  FAILED at bin " << diffBin << ": " << hist[diffBin] << " != " << ref[diffBin];
    std::cout << std::endl;
    JSONLog::write("checksum_" + std::to_string(i), checksum);
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
