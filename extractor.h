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
  std::vector<kslicer::FuncData>    ExtractUsedFromVFH(MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler, std::unordered_map<uint64_t, kslicer::FuncData>& usedFunctions);

  std::vector<kslicer::DeclInClass> ExtractUsedTC(const  std::vector<kslicer::DeclInClass>& a_listedNames, const clang::CXXRecordDecl* classAstNode, const clang::CompilerInstance& a_compiler);
  std::vector<kslicer::DeclInClass> ExtractTCFromClass(const std::string& a_className, const clang::CXXRecordDecl* classAstNode, 
                                                       const clang::CompilerInstance& compiler, clang::tooling::ClangTool& Tool);

  std::unordered_map<std::string, kslicer::DataMemberInfo> ExtractUsedMemberData(kslicer::KernelInfo* pKernel, const kslicer::FuncData& a_funcData, const std::vector<kslicer::FuncData>& a_otherMambers,
                                                                                 std::unordered_map<std::string, kslicer::UsedContainerInfo>& a_auxContainers, 
                                                                                 MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler);

  std::vector< std::unordered_map<std::string, std::string> > ArgMatchTraversal(kslicer::KernelInfo* pKernel, const kslicer::FuncData& a_funcData, const std::vector<kslicer::FuncData>& a_otherMambers,
                                                                                MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler);

  std::vector<std::string> ExtractDefines(const clang::CompilerInstance& a_compiler);      
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////  NodesMarker  //////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  class NodesMarker : public RecursiveASTVisitor<NodesMarker> // mark all subsequent nodes to be rewritten, put their ash codes in 'rewrittenNodes'
  {
  public:
    NodesMarker(std::unordered_set<uint64_t>& a_rewrittenNodes) : m_rewrittenNodes(a_rewrittenNodes){}
    bool VisitStmt(Stmt* expr);

  private:
    std::unordered_set<uint64_t>& m_rewrittenNodes;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////  NewKernelExtractor  ///////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct KernelNodes
  {
    const clang::Stmt* loopStart  = nullptr;
    const clang::Expr* loopStride = nullptr;
    const clang::Expr* loopSize   = nullptr; 
    const clang::Stmt* loopBody   = nullptr;
  };

  KernelNodes ExtractKernelForLoops(const clang::Stmt* kernelBody, int a_loopsNumber, const clang::CompilerInstance& a_compiler);

}

#endif