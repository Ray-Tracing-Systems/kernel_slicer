#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>
#include <random>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<MultiNet> CreateMultiNet_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures MultiNet_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif
  ArgParser args(argc, argv);
  const uint32_t SIZE     = args.getOptionValue<int>("--size", 100000);
  const uint32_t NUM_NETS = 5;

  std::mt19937 gen(777);
  std::uniform_real_distribution<float> distIn(-1.0f, 1.0f);
  std::uniform_int_distribution<uint32_t> distNet(0, NUM_NETS-1), distIdx(0, NN_OUTPUT_DIM-1);
  std::vector<uint32_t>        netId(SIZE);
  std::vector<float>           input(SIZE*NN_INPUT_DIM);
  std::vector<LiteMath::uint4> indices(SIZE);
  for(auto& x : input) x = distIn(gen);
  for(uint32_t i = 0; i < SIZE; i++)
  {
    netId[i]   = distNet(gen);
    indices[i] = LiteMath::uint4(distIdx(gen), distIdx(gen), distIdx(gen), distIdx(gen));
  }
  std::vector<float4> out(SIZE);

  std::shared_ptr<MultiNet> pImpl = nullptr;

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = MultiNet_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateMultiNet_Generated(ctx, SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<MultiNet>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  // random weights, uniform in [-2/sqrt(inDim), 2/sqrt(inDim)], biases in [-1/sqrt(inDim), 1/sqrt(inDim)]
  const uint32_t dims[NN_LAYERS+1] = {NN_INPUT_DIM, NN_HIDDEN_DIM, NN_HIDDEN_DIM, NN_OUTPUT_DIM};
  for(uint32_t net = 0; net < NUM_NETS; net++)
  {
    std::vector<float> matrices[NN_LAYERS], biases[NN_LAYERS];
    const float* pMatrices[NN_LAYERS];
    const float* pBiases[NN_LAYERS];
    for(uint32_t layer = 0; layer < NN_LAYERS; layer++)
    {
      const float scale = 1.0f/std::sqrt(float(dims[layer]));
      std::uniform_real_distribution<float> dist(-scale, scale);
      matrices[layer].resize(dims[layer]*dims[layer+1]);
      biases[layer].resize(dims[layer+1]);
      for(auto& w : matrices[layer]) w = 2.0f*dist(gen);
      for(auto& b : biases[layer])   b = dist(gen);
      pMatrices[layer] = matrices[layer].data();
      pBiases[layer]   = biases[layer].data();
    }
    pImpl->AddNetwork(pMatrices, pBiases);
  }

  pImpl->CommitDeviceData();
  pImpl->Run(netId.data(), input.data(), indices.data(), out.data(), SIZE);

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Run", timings);
  std::cout << "Run(exec) = " << timings[0] << " ms" << std::endl;
  std::cout << "out[0] = " << out[0].x << " " << out[0].y << " " << out[0].z << " " << out[0].w << std::endl;

  std::vector<float> outFlat(SIZE*4);
  for(uint32_t i = 0; i < SIZE; i++)
    for(int j = 0; j < 4; j++)
      outFlat[i*4+j] = out[i][j];
  JSONLog::write("out", outFlat);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
