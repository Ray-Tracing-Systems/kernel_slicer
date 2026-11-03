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
std::shared_ptr<SimpleMLP> CreateSimpleMLP_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures SimpleMLP_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif
  const uint32_t SIZE = 10000;

  std::mt19937 gen(12345);
  std::vector<float> input(SIZE*NN_INPUT_DIM);
  std::uniform_real_distribution<float> distIn(-1.0f, 1.0f);
  for(auto& x : input) x = distIn(gen);
  std::vector<float> out(SIZE*NN_OUTPUT_DIM);

  std::shared_ptr<SimpleMLP> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = SimpleMLP_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateSimpleMLP_Generated(ctx, SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<SimpleMLP>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  // random weights, uniform in [-2/sqrt(inDim), 2/sqrt(inDim)], biases in [-1/sqrt(inDim), 1/sqrt(inDim)]
  const uint32_t dims[4] = {NN_INPUT_DIM, NN_HIDDEN_DIM, NN_HIDDEN_DIM, NN_OUTPUT_DIM};
  for(int layer = 0; layer < 3; layer++)
  {
    const float scale = 1.0f/std::sqrt(float(dims[layer]));
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> matrix(dims[layer]*dims[layer+1]), bias(dims[layer+1]);
    for(auto& w : matrix) w = 2.0f*dist(gen);
    for(auto& b : bias)   b = dist(gen);
    pImpl->SetLayer(layer, matrix.data(), bias.data(), dims[layer], dims[layer+1]);
  }

  pImpl->CommitDeviceData();
  pImpl->Run(input.data(), out.data(), SIZE);

  std::cout << "out[0..3] = " << out[0] << " " << out[1] << " " << out[2] << " " << out[3] << std::endl;

  JSONLog::write("out", out);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
