#ifndef KSLICER_CLASS_GEN
#define KSLICER_CLASS_GEN

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"

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
  
  /**\brief put all args together with comma or ',' to gave unique key for any concrete argument sequence.
      \return unique strig key which you can pass in std::unordered_map for example 
  */
  std::string MakeKernellCallSignature(const std::string& a_mainFuncName, const std::vector<ArgReferenceOnCall>& a_args, const std::unordered_map<std::string, UsedContainerInfo>& a_usedContainers);

  class MainFunctionRewriter : public RecursiveASTVisitor<MainFunctionRewriter>
  {
  public:
    
    MainFunctionRewriter(Rewriter &R, const clang::CompilerInstance& a_compiler, MainFuncInfo& a_mainFunc, 
                         const std::vector<InOutVarInfo>& a_args, MainClassInfo* a_pCodeInfo) : 
                         m_rewriter(R), m_compiler(a_compiler), m_sm(R.getSourceMgr()), 
                         m_dsTagId(0), m_mainFuncName(a_mainFunc.Name), m_mainFuncLocals(a_mainFunc.Locals),
                         m_pCodeInfo(a_pCodeInfo), m_allClassMembers(a_pCodeInfo->allDataMembers), m_mainFunc(a_mainFunc), allDescriptorSetsInfo(a_pCodeInfo->allDescriptorSetsInfo),
                         m_kernels(a_pCodeInfo->kernels) 
    { 
      for(const auto& arg : a_args) 
        m_argsOfMainFunc[arg.name] = arg;
    }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
    bool VisitCallExpr(CallExpr* f);
    bool VisitIfStmt(IfStmt* ifExpr);
  
    std::string                                              mainFuncCmdName;
    std::unordered_map<std::string, uint32_t>                dsIdBySignature;
    std::vector< KernelCallInfo >&                           allDescriptorSetsInfo;
    std::unordered_map<std::string, DataMemberInfo>&         m_allClassMembers;
    const std::unordered_map<std::string, DataLocalVarInfo>& m_mainFuncLocals;

  private:

    std::vector<ArgReferenceOnCall> ExtractArgumentsOfAKernelCall(CallExpr* f);
    std::string MakeKernelCallCmdString(CXXMemberCallExpr* f);
    std::string MakeServiceKernelCallCmdString(CallExpr* call);

    Rewriter&                      m_rewriter;
    const clang::SourceManager&    m_sm;
    const clang::CompilerInstance& m_compiler;
    uint32_t m_dsTagId;
    
    std::string m_mainFuncName;
    std::unordered_map<std::string, InOutVarInfo>      m_argsOfMainFunc;
    const std::unordered_map<std::string, KernelInfo>& m_kernels;
    MainFuncInfo& m_mainFunc;

    std::unordered_set<uint64_t> m_alreadyProcessedCalls;
    MainClassInfo* m_pCodeInfo = nullptr;
  };

  std::vector<InOutVarInfo> ListPointerParamsOfMainFunc(const CXXMethodDecl* a_node);
  
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

  void ObtainKernelsDecl(std::unordered_map<std::string, KernelInfo>& a_kernelsData, const clang::CompilerInstance& compiler, const std::string& a_mainClassName, const MainClassInfo& a_codeInfo);
  std::string GetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, 
                                      const std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair>& threadIds,
                                      const std::string a_names[3]);
}


#endif
