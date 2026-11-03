#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>


#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "vk_context.h"
//#include "test_class_generated.h"
std::shared_ptr<SimpleTest> CreateSimpleTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures SimpleTest_Generated_ListRequiredDeviceFeatures();

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const std::vector<float> input = {0.0f, 1.0f, -2.5f, 0.125f, 65504.0f, 6.103515625e-5f, 0.1f, 1.0f/3.0f, -1000.75f, 3.14159265f, 0.3f, -0.3f, 1.0e-6f, 65519.0f};
  const uint32_t size = uint32_t(input.size());
  std::vector<uint32_t> bits(size, 0);
  std::vector<float>    output(size, 0.0f);

  std::shared_ptr<SimpleTest> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto rdf = SimpleTest_Generated_ListRequiredDeviceFeatures();
    auto ctx = vk_utils::globalContextInit(rdf.extensionNames, enableValidationLayers, a_preferredDeviceId, &rdf.features2);
    pImpl    = CreateSimpleTest_Generated(ctx, size);
  }
  else
    pImpl = std::make_shared<SimpleTest>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->Convert(size, input.data(), bits.data(), output.data());

  std::vector<float> bitsAsFloat(size); // exact for 16 bit values
  for(uint32_t i=0;i<size;i++) {
    std::cout << input[i] << "\t" << bits[i] << "\t" << output[i] << std::endl;
    bitsAsFloat[i] = float(bits[i]);
  }

  JSONLog::write("bits", bitsAsFloat);
  JSONLog::write("output", output);
  JSONLog::saveToFile("zout_"+backendName+".json");
  
  pImpl = nullptr;
  vk_utils::globalContextDestroy();
  return 0;
}
