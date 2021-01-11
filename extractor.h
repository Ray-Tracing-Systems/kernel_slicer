#ifndef KSLICER_EXTRACTOR_H
#define KSLICER_EXTRACTOR_H

#include "kslicer.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iostream>

namespace kslicer
{
  using namespace llvm;
  using namespace clang;

  void ExtractUsedCode(MainClassInfo& a_codeInfo, std::unordered_map<std::string, clang::SourceRange>& a_usedFunctions, const clang::CompilerInstance& a_compiler);

};


#endif