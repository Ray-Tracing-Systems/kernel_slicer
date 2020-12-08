#ifndef KSLICER_TEMPLATE_RENDERING_H
#define KSLICER_TEMPLATE_RENDERING_H

#include <iostream>
#include <string>

#include "kslicer.h"

namespace kslicer
{
  void PrintVulkanBasicsFile(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo);
 
  std::string PrintGeneratedClassDecl(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo, 
                                      const std::vector<MainFuncInfo>& a_methodsToGenerate);

  void PrintGeneratedClassImpl(const std::string& a_declTemplateFilePath, const std::string& a_includeName, const MainClassInfo& a_classInfo,
                               const std::vector<MainFuncInfo>& a_methodsToGenerate); 

  void PrintGeneratedCLFile(const std::string& a_inFileName, const std::string& a_outFileName, const MainClassInfo& a_classInfo, 
                            const std::unordered_map<std::string, bool>& usedFiles, 
                            const std::unordered_map<std::string, clang::SourceRange>& usedFunctions,
                            const clang::CompilerInstance& compiler);
};

#endif