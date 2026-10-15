#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<CommitTest> CreateCommitTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures CommitTest_Generated_ListRequiredDeviceFeatures();
#endif

static std::vector<float> MakeData(size_t a_size, float a_scale)
{
  std::vector<float> data(a_size);
  for(size_t i=0;i<data.size();i++)
    data[i] = a_scale*float(i % 17) + 1.0f;
  return data;
}

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const unsigned int OUT_SIZE = 2048;

  std::shared_ptr<CommitTest> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = CommitTest_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateCommitTest_Generated(ctx, OUT_SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<CommitTest>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  // each iteration changes size and content of member vectors and commits them again
  //
  const size_t sizes [3] = {256, 1024, 128};
  const float  scales[3] = {1.0f, 0.5f, 2.0f};
  for(int iter = 0; iter < 3; iter++)
  {
    std::vector<float> out1(OUT_SIZE), out2(OUT_SIZE);
    pImpl->SetData(MakeData(sizes[iter], scales[iter]));
    pImpl->CommitDeviceData();
    pImpl->Process(out1.data(), OUT_SIZE);
    pImpl->ProcessRTVBlock(out2.data(), OUT_SIZE, 1);
    pImpl->Process(out1.data(), OUT_SIZE);           // second call without commit

    std::cout << "iter = " << iter << ", summ = " << pImpl->m_summ << ", out1[1] = " << out1[1] << ", out2[1] = " << out2[1] << std::endl;
    JSONLog::write(("summ_" + std::to_string(iter)).c_str(), pImpl->m_summ);
    JSONLog::write(("out1_" + std::to_string(iter)).c_str(), out1);
    JSONLog::write(("out2_" + std::to_string(iter)).c_str(), out2);
  }

  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
