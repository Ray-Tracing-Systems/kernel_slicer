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
std::shared_ptr<HashMapTest> CreateHashMapTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures HashMapTest_Generated_ListRequiredDeviceFeatures();
#endif

static inline uint32_t PseudoRandom(uint32_t i) // LCG step from iteration number with high bits mixed into low ones
{
  const uint32_t x = i * 1103515245u + 12345u;
  return (x ^ (x >> 16));
}

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::shared_ptr<HashMapTest> pImpl = nullptr;
  ArgParser args(argc, argv);

  const uint32_t KEYS_NUM  = args.getOptionValue<int>("--keys",  256*256); // number of (repeating) keys for histogram
  const uint32_t SAME_NUM  = args.getOptionValue<int>("--same",  4);       // consecutive keys which are equal
  const uint32_t ITEMS_NUM = args.getOptionValue<int>("--items", 10000);   // number of unique items
  const uint32_t KEY_RANGE = 1u << 24;                                     // keys are sparse in [-KEY_RANGE/2, KEY_RANGE/2)
  const float    RES_SCALE = args.getOptionValue<float>("--reserve_scale", 1.0f); // < 1 to check overflow of GPU hash table

  // input data
  //
  std::vector<int> keys(KEYS_NUM);
  for(uint32_t i=0; i < KEYS_NUM; i++)
    keys[i] = int(PseudoRandom(i / SAME_NUM) % KEY_RANGE) - int(KEY_RANGE/2);

  std::vector<int>   itemKeys(ITEMS_NUM);
  std::vector<float> itemVals(ITEMS_NUM);
  for(uint32_t i=0; i < ITEMS_NUM; i++)
  {
    itemKeys[i] = int(i)*7919 - 1000000;  // unique
    itemVals[i] = float(i)*0.5f + 1.0f;
  }

  // queries: half of them hit histogram keys, quarter hit item keys, the rest are (most likely) absent
  //
  const uint32_t QUERY_NUM = 2*ITEMS_NUM;
  std::vector<int> queries(QUERY_NUM);
  for(uint32_t i=0; i < QUERY_NUM; i++)
  {
    if(i % 2 == 0)
      queries[i] = keys[PseudoRandom(i) % KEYS_NUM];
    else if(i % 4 == 1)
      queries[i] = itemKeys[PseudoRandom(i) % ITEMS_NUM];
    else
      queries[i] = int(PseudoRandom(i + 7) % KEY_RANGE) - int(KEY_RANGE/2) + 1;
  }

  // reference
  //
  std::map<int, uint32_t> refHist;
  for(auto k : keys)
    refHist[k]++;
  std::map<int, float> refItems;
  std::map<int, uint32_t> refIds;
  for(uint32_t i=0; i < ITEMS_NUM; i++)
  {
    refItems[itemKeys[i]] = itemVals[i];
    refIds  [itemKeys[i]] = i;
  }

  bool onGPU = args.hasOption("--gpu");
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = HashMapTest_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateHashMapTest_Generated(ctx, std::max(KEYS_NUM, QUERY_NUM));
  }
  else
  #endif
    pImpl = std::make_shared<HashMapTest>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->Reserve(uint32_t(float(KEYS_NUM/SAME_NUM + 1)*RES_SCALE), uint32_t(float(ITEMS_NUM)*RES_SCALE)); // max possible number of unique keys
  pImpl->CommitDeviceData();
  pImpl->Build(keys.data(), KEYS_NUM, itemKeys.data(), itemVals.data(), ITEMS_NUM);

  std::vector<uint32_t> outCount (QUERY_NUM, 0xFFFFFFFE);
  std::vector<uint32_t> outCount2(QUERY_NUM, 0xFFFFFFFE);
  std::vector<float>    outVal   (QUERY_NUM, 0.0f);
  std::vector<uint32_t> outId    (QUERY_NUM, 0xFFFFFFFE);
  pImpl->Lookup(queries.data(), QUERY_NUM, outCount.data(), outCount2.data(), outVal.data(), outId.data());

  // check
  //
  bool passed = true;
  uint32_t hitsHist = 0, hitsItems = 0;
  for(uint32_t i=0; i < QUERY_NUM; i++)
  {
    const auto pHist    = refHist.find(queries[i]);
    const auto pItem    = refItems.find(queries[i]);
    const uint32_t expC = (pHist != refHist.end())  ? pHist->second : 0u;
    const float    expV = (pItem != refItems.end()) ? pItem->second : -1.0f;
    const uint32_t expI = (pItem != refItems.end()) ? refIds[queries[i]] : 0xFFFFFFFF;
    hitsHist  += (expC != 0)     ? 1 : 0;
    hitsItems += (expV != -1.0f) ? 1 : 0;

    if(i < 10)
      std::cout << i << "\tkey = " << queries[i] << "\tcount = " << outCount[i] << " (" << expC << ")\tval = " << outVal[i] << " (" << expV << ")" << std::endl;

    if(outCount[i] != expC || outCount2[i] != expC || outVal[i] != expV || outId[i] != expI)
    {
      std::cout << "FAILED at " << i << ", key = " << queries[i] << ": count = " << outCount[i] << ", count2 = " << outCount2[i] << " (" << expC 
                << "), val = " << outVal[i] << " (" << expV << "), id = " << outId[i] << " (" << expI << ")" << std::endl;
      passed = false;
      break;
    }
  }

  std::cout << "unique keys = " << refHist.size() << ", items = " << refItems.size() << std::endl;
  std::cout << "queries = " << QUERY_NUM << ", hist hits = " << hitsHist << ", item hits = " << hitsItems << std::endl;

  if(onGPU)
  {
    for(const char* kernelName : {"kernel1D_CountKeys", "kernel1D_InsertItems", "kernel1D_Lookup"})
    {
      float timings[4] = {0,0,0,0};
      pImpl->GetExecutionTime(kernelName, timings);
      std::cout << kernelName << "(exec) = " << timings[0] << " ms (avg), " << timings[1] << " ms (min), " << timings[2] << " ms (max)" << std::endl;
    }
  }
  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

  if(QUERY_NUM <= 4*1024)
  {
    JSONLog::write("count", outCount);
    JSONLog::write("val",   outVal);
  }
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return passed ? 0 : 1;
}
