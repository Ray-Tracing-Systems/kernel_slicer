#ifndef KSLICER_EXTRACTOR_H
#define KSLICER_EXTRACTOR_H

#include "kslicer.h"
#include "clang/Tooling/Tooling.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iostream>

namespace kslicer
{
  using namespace llvm;
  using namespace clang;

  std::vector<kslicer::FuncData>    ExtractUsedFunctions(MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler);
  std::vector<kslicer::DeclInClass> ExtractUsedTC(const  std::vector<kslicer::DeclInClass>& a_listedNames, const clang::CXXRecordDecl* classAstNode, const clang::CompilerInstance& a_compiler);
  std::vector<kslicer::DeclInClass> ExtractTCFromClass(const std::string& a_className, const clang::CXXRecordDecl* classAstNode,  
                                                       const clang::CompilerInstance& compiler, clang::tooling::ClangTool& Tool);

};


#endif