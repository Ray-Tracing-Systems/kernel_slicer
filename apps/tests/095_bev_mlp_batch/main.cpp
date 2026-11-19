#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <algorithm>

#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"
#include "test_class.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<BatchMLP> CreateBatchMLP_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures BatchMLP_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_ISPC
std::shared_ptr<BatchMLP> CreateBatchMLP_ISPC();
#endif

// reference: the same network for one vector with 'LiteNN::LinearForward' (results of layers are rounded to fp16)
//
static void Reference(const BatchMLP& a_mlp, const float* a_in, float* a_out)
{
  float buf0[NN_HIDDEN_DIM], buf1[NN_HIDDEN_DIM];
  LiteNN::LinearForward_ReLU(a_mlp.m_weights.data(), a_mlp.m_matOffs[0], a_mlp.m_biasOffs[0], a_in, NN_INPUT_DIM,  buf0, NN_HIDDEN_DIM);
  LiteNN::LinearForward_ReLU(a_mlp.m_weights.data(), a_mlp.m_matOffs[1], a_mlp.m_biasOffs[1], buf0, NN_HIDDEN_DIM, buf1, NN_HIDDEN_DIM);
  LiteNN::LinearForward     (a_mlp.m_weights.data(), a_mlp.m_matOffs[2], a_mlp.m_biasOffs[2], buf1, NN_HIDDEN_DIM, a_out, NN_OUTPUT_DIM);
}

// Usage: testapp [--gpu | --ispc] [--bench N]
// Several sizes (block borders) are checked against the reference, then N vectors are timed.
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
  const uint bench  = args.getOptionValue<int>("--bench", 1 << 20);
  const uint MAX_SIZE = std::max(bench, 100003u);

  std::shared_ptr<BatchMLP> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto features = BatchMLP_Generated_ListRequiredDeviceFeatures(); // cooperative matrices, fp16
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl         = CreateBatchMLP_Generated(ctx, MAX_SIZE);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateBatchMLP_ISPC();
  else
  #endif
    pImpl = std::make_shared<BatchMLP>();

  // random weights, uniform in [-2/sqrt(inDim), 2/sqrt(inDim)], biases in [-1/sqrt(inDim), 1/sqrt(inDim)]
  std::mt19937 gen(12345);
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

  std::uniform_real_distribution<float> valueDist(-1.0f, 1.0f);
  std::vector<float> input(MAX_SIZE*NN_INPUT_DIM), output(MAX_SIZE*NN_OUTPUT_DIM);
  for(auto& x : input)
    x = valueDist(gen);

  int failed = 0;
  for(uint size : {1u, 63u, 64u, 65u, 100003u})
  {
    std::fill(output.begin(), output.end(), 0.0f);
    pImpl->Run(input.data(), output.data(), size);

    float maxErr = 0.0f;
    for(uint i = 0; i < size; i++)
    {
      float ref[NN_OUTPUT_DIM];
      Reference(*pImpl, input.data() + i*NN_INPUT_DIM, ref);
      for(uint j = 0; j < NN_OUTPUT_DIM; j++)
        maxErr = std::max(maxErr, std::abs(output[i*NN_OUTPUT_DIM + j] - ref[j])/(std::abs(ref[j]) + 1.0f));
    }
    const bool ok = (maxErr < 1e-2f);
    failed += ok ? 0 : 1;
    std::cout << "size = " << size << ", max error = " << maxErr << (ok ? "" : "  FAILED!") << std::endl;
    JSONLog::write("check_" + std::to_string(size), ok ? "PASSED!" : "FAILED!");
  }
  JSONLog::write("out", std::vector<float>(output.begin(), output.begin() + 4*NN_OUTPUT_DIM)); // compared with tolerance, see ktest.json
  JSONLog::write("check", failed == 0 ? "PASSED!" : "FAILED!");
  JSONLog::saveToFile(onGPU ? "zout_gpu.json" : (onISPC ? "zout_ispc.json" : "zout_cpu.json"));

  if(bench > 0 && (onGPU || onISPC))                      // time only, results are checked above
  {
    float minTime = 1e30f;
    for(int run = 0; run < 5; run++)
    {
      pImpl->Run(input.data(), output.data(), bench);
      float timings[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      pImpl->GetExecutionTime("Run", timings);
      minTime = std::min(minTime, timings[0]);
    }
    const double flops = 2.0*double(NN_INPUT_DIM*NN_HIDDEN_DIM + NN_HIDDEN_DIM*NN_HIDDEN_DIM + NN_HIDDEN_DIM*NN_OUTPUT_DIM)*double(bench);
    std::cout << "Run " << bench << " vectors: " << minTime << " ms, " << flops/(double(minTime)*1e6) << " GFLOPS" << std::endl;
  }

  std::cout << (failed == 0 ? "PASSED" : "FAILED") << std::endl;

  pImpl = nullptr;           // Vulkan objects of the class first, then the context
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
