#ifndef KSLICER_TEMPLATE_RENDERING_H
#define KSLICER_TEMPLATE_RENDERING_H

#include <iostream>
#include <string>

#include "kslicer.h"

namespace kslicer
{
  void PrintVulkanBasicsFile(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo);
  std::string PrintGeneratedClassDecl(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo); 
  void PrintGeneratedClassImpl(const std::string& a_declTemplateFilePath, const std::string& a_includeName, const MainClassInfo& a_classInfo,
                               const std::string& a_mainFuncCodeGen); 
};

#endif