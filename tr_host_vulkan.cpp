#include "kslicer.h"
#include "template_rendering.h"

void kslicer::VulkanCodeGen::GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  kslicer::ApplyJsonToTemplate("templates/vk_class.h",        fullSuffix + ".h", jsonHost);
  kslicer::ApplyJsonToTemplate("templates/vk_class.cpp",      fullSuffix + ".cpp", jsonHost);
  kslicer::ApplyJsonToTemplate("templates/vk_class_ds.cpp",   fullSuffix + "_ds.cpp", jsonHost);
  kslicer::ApplyJsonToTemplate("templates/vk_class_init.cpp", fullSuffix + "_init.cpp", jsonHost);
  if(a_settings.genSeparateGPUAPI)
    kslicer::ApplyJsonToTemplate("templates/vk_class_api.h",  fullSuffix + "_api.h", jsonHost);
}

void kslicer::VulkanCodeGen::GenerateHostDevFeatures(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  kslicer::ApplyJsonToTemplate("templates/vk_class_init.cpp", fullSuffix + "_init.cpp", jsonHost);
}