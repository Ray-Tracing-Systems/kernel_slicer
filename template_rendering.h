#ifndef KSLICER_TEMPLATE_RENDERING_H
#define KSLICER_TEMPLATE_RENDERING_H

#include <iostream>
#include <string>
#include <inja.hpp>

#include "kslicer.h"

namespace kslicer
{
  void PrintVulkanBasicsFile(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo);

  nlohmann::json PrepareJsonForAllCPP(const MainClassInfo& a_classInfo, const std::vector<MainFuncInfo>& a_methodsToGenerate, const std::string& a_genIncude, const uint32_t threadsOrder[3]);

  void ApplyJsonToTemplate(const std::string& a_declTemplateFilePath, const std::string& a_suffix, const nlohmann::json& a_data); 


  void PrintGeneratedCLFile(const std::string& a_inFileName, const std::string& a_outFileName, const MainClassInfo& a_classInfo, 
                            const std::unordered_map<std::string, bool>& usedFiles, 
                            const std::unordered_map<std::string, clang::SourceRange>& usedFunctions,
                            const clang::CompilerInstance& compiler,
                            const uint32_t    threadsOrder[3]);
};

#endif