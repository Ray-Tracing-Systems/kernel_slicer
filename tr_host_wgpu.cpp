#include "kslicer.h"
#include "template_rendering.h"
#include "class_gen.h"

void kslicer::WGPUCodeGen::GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  kslicer::ApplyJsonToTemplate("templates_wk/wk_class.h",   fullSuffix + ".h",   jsonHost);
  kslicer::ApplyJsonToTemplate("templates_wk/wk_class.cpp", fullSuffix + ".cpp", jsonHost);
}

void kslicer::WGPUCodeGen::GenerateHostDevFeatures(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  //kslicer::ApplyJsonToTemplate("templates_wk/wk_class.cpp", fullSuffix + ".cpp", jsonHost);
}

std::string kslicer::GetControlFuncDeclWGPU(const clang::FunctionDecl* fDecl, clang::CompilerInstance& compiler)
{
  std::string text = fDecl->getNameInfo().getName().getAsString() + "Cmd(WGPUCommandEncoder a_commandEncoder";
  if(fDecl->getNumParams()!= 0)
    text += ", ";
  for(unsigned i=0;i<fDecl->getNumParams();i++)
  {
    auto pParam = fDecl->getParamDecl(i);
    //const clang::QualType typeOfParam =	pParam->getType();
    //std::string typeStr = typeOfParam.getAsString();
    text += kslicer::GetRangeSourceCode(pParam->getSourceRange(), compiler);
    if(i!=fDecl->getNumParams()-1)
      text += ", ";
  }

  return text + ")";
}