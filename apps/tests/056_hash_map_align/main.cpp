#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<HashMapAlign> CreateHashMapAlign_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures HashMapAlign_Generated_ListRequiredDeviceFeatures();
#endif

static inline uint32_t PseudoRandom(uint32_t i) // LCG step from iteration number with high bits mixed into low ones
{
  const uint32_t x = i * 1103515245u + 12345u;
  return (x ^ (x >> 16));
}

static inline uint32_t UniqueKey(uint32_t i) { return i*7919u + 12345u; } // 7919 is odd, so different 'i' give different keys

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::shared_ptr<HashMapAlign> pImpl = nullptr;
  ArgParser args(argc, argv);

  const uint32_t KEYS_NUM = args.getOptionValue<int>("--keys", 10000);   // unique keys
  const uint32_t DUPS_NUM = args.getOptionValue<int>("--dups", 100000);  // repeating keys for atomics

  std::vector<uint32_t> keys(KEYS_NUM), dupKeys(DUPS_NUM);
  for(uint32_t i=0; i < KEYS_NUM; i++)
    keys[i] = UniqueKey(i);
  for(uint32_t i=0; i < DUPS_NUM; i++)
    dupKeys[i] = keys[PseudoRandom(i) % std::max(KEYS_NUM/2, 1u)];

  // queries: even ones are present keys, odd ones are absent keys
  //
  const uint32_t QUERY_NUM = 2*KEYS_NUM;
  std::vector<uint32_t> queries(QUERY_NUM);
  for(uint32_t i=0; i < QUERY_NUM; i++)
    queries[i] = (i % 2 == 0) ? keys[PseudoRandom(i) % KEYS_NUM] : UniqueKey(KEYS_NUM + i);

  bool onGPU = args.hasOption("--gpu");
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = HashMapAlign_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateHashMapAlign_Generated(ctx, std::max(DUPS_NUM, QUERY_NUM));
  }
  else
  #endif
    pImpl = std::make_shared<HashMapAlign>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->Reserve(KEYS_NUM);
  pImpl->CommitDeviceData();
  pImpl->Build(keys.data(), KEYS_NUM, dupKeys.data(), DUPS_NUM);

  std::vector<float4> out(QUERY_NUM*4, float4(0.0f));
  pImpl->Lookup(queries.data(), QUERY_NUM, out.data());

  // reference
  //
  std::map<uint32_t, uint32_t> refCount;
  for(auto k : dupKeys)
    refCount[k]++;
  std::map<uint32_t, uint32_t> refPresent;
  for(auto k : keys)
    refPresent[k] = 1;

  bool passed = true;
  for(uint32_t i=0; i < QUERY_NUM && passed; i++)
  {
    const uint32_t key = queries[i];
    float4 exp[4] = {float4(-1.0f), float4(-1.0f), float4(-1.0f), float4(-1.0f)};
    if(refPresent.find(key) != refPresent.end())
    {
      exp[0] = float4(ValueF2(key).x, ValueF2(key).y, ValueF3(key).y, ValueF3(key).z);
      exp[1] = ValueF4(key);
      exp[2] = ValuePart(key).vel;
    }
    auto pCount = refCount.find(key);
    if(pCount != refCount.end())
    {
      exp[3].x = float(pCount->second);
      exp[3].y = float(key % 1000);
      exp[3].z = -2.0f*float(pCount->second);
      exp[3].w = 0.5f*float(pCount->second);
      exp[2].x += float(pCount->second);
    }
    if(i == 0)
      exp[3].w = float(refCount.size());

    for(int j=0;j<4;j++)
    {
      for(int k=0;k<4;k++)
      {
        if(out[i*4+j][k] != exp[j][k])
        {
          std::cout << "FAILED at query " << i << ", key = " << key << ", out[" << j << "][" << k << "] = " << out[i*4+j][k] << " != " << exp[j][k] << std::endl;
          passed = false;
        }
      }
    }
  }

  std::cout << "unique keys = " << KEYS_NUM << ", repeating keys = " << DUPS_NUM << " (" << refCount.size() << " unique)" << std::endl;
  if(onGPU)
  {
    for(const char* kernelName : {"kernel1D_Fill", "kernel1D_Accum", "kernel1D_Lookup"})
    {
      float timings[4] = {0,0,0,0};
      pImpl->GetExecutionTime(kernelName, timings);
      std::cout << kernelName << "(exec) = " << timings[0] << " ms (avg), " << timings[1] << " ms (min), " << timings[2] << " ms (max)" << std::endl;
    }
  }
  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return passed ? 0 : 1;
}
