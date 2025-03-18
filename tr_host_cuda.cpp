#include "kslicer.h"
#include "template_rendering.h"

void kslicer::CudaCodeGen::GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  kslicer::ApplyJsonToTemplate("templates_cuda/gen_class.cu", fullSuffix + ".cu", jsonHost);
}