#include "kslicer.h"
#include "template_rendering.h"

void kslicer::WGPUCodeGen::GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  kslicer::ApplyJsonToTemplate("templates_wk/wk_class.h",   fullSuffix + ".h",   jsonHost);
  kslicer::ApplyJsonToTemplate("templates_wk/wk_class.cpp", fullSuffix + ".cpp", jsonHost);
}

void kslicer::WGPUCodeGen::GenerateHostDevFeatures(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  //kslicer::ApplyJsonToTemplate("templates_wk/wk_class.cpp", fullSuffix + ".cpp", jsonHost);
}