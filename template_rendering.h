#ifndef KSLICER_TEMPLATE_RENDERING_H
#define KSLICER_TEMPLATE_RENDERING_H

#include <iostream>
#include "kslicer.h"

namespace kslicer
{
  void PrintGeneratedClassDecl(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo, std::ostream& a_out); 
};

#endif