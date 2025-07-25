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

  class MainFunctionRewriterVulkan : public RecursiveASTVisitor<MainFunctionRewriterVulkan>
  {
  public:
    
    MainFunctionRewriterVulkan(Rewriter &R, const clang::CompilerInstance& a_compiler, MainFuncInfo& a_mainFunc, const std::vector<InOutVarInfo>& a_args, MainClassInfo* a_pCodeInfo) : 
                               m_rewriter(R), m_compiler(a_compiler), m_sm(R.getSourceMgr()), 
                               m_mainFuncName(a_mainFunc.Name), m_mainFuncLocals(a_mainFunc.Locals),
                               m_pCodeInfo(a_pCodeInfo), m_allClassMembers(a_pCodeInfo->allDataMembers), allDescriptorSetsInfo(a_pCodeInfo->allDescriptorSetsInfo),
                               m_kernels(a_pCodeInfo->kernels), m_mainFunc(a_mainFunc)
    { 
      for(const auto& arg : a_args) 
        m_argsOfMainFunc[arg.name] = arg;

      m_pRewrittenNodes = std::make_shared< std::unordered_set<uint64_t> > ();
    }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
    bool VisitCallExpr(CallExpr* f);
    bool VisitIfStmt(IfStmt* ifExpr);
    bool VisitMemberExpr(MemberExpr* expr);
  
    std::string                                              mainFuncCmdName;
    std::unordered_map<std::string, uint32_t>                dsIdBySignature;

  private:

    std::vector<ArgReferenceOnCall> ExtractArgumentsOfAKernelCall(CallExpr* f, const std::unordered_set<std::string>& a_excludeList);
    std::string MakeKernelCallCmdString(CXXMemberCallExpr* f);
    std::string MakeServiceKernelCallCmdString(CallExpr* call, const std::string& a_name);

    Rewriter&                      m_rewriter;
    const clang::CompilerInstance& m_compiler;
    const clang::SourceManager&    m_sm;

  public:

    std::string                                              m_mainFuncName;
    const std::unordered_map<std::string, DataLocalVarInfo>& m_mainFuncLocals;
    MainClassInfo*                                           m_pCodeInfo = nullptr;
    std::unordered_map<std::string, DataMemberInfo>&         m_allClassMembers;
    std::vector< KernelCallInfo >&                           allDescriptorSetsInfo;

  private:
    
    std::unordered_map<std::string, KernelInfo>&       m_kernels;
    std::unordered_map<std::string, InOutVarInfo>      m_argsOfMainFunc;
    MainFuncInfo&                                      m_mainFunc;
    std::shared_ptr< std::unordered_set<uint64_t> >    m_pRewrittenNodes;
    std::unordered_map<uint64_t, std::string>          m_workAround;

    void ReplaceTextOrWorkAround(clang::SourceRange a_range, const std::string& a_text);
    bool WasNotRewrittenYet(const clang::Stmt* expr) const;
    void MarkRewritten(const clang::Stmt* expr);
    std::string RecursiveRewrite(const clang::Stmt* expr);
  };

  class MainFunctionRewriterWGPU : public RecursiveASTVisitor<MainFunctionRewriterWGPU>
  {
  public:
    
    MainFunctionRewriterWGPU(Rewriter &R, const clang::CompilerInstance& a_compiler, MainFuncInfo& a_mainFunc, const std::vector<InOutVarInfo>& a_args, MainClassInfo* a_pCodeInfo) : 
                             m_rewriter(R), m_compiler(a_compiler), m_sm(R.getSourceMgr()), 
                             m_mainFuncName(a_mainFunc.Name), m_mainFuncLocals(a_mainFunc.Locals),
                             m_pCodeInfo(a_pCodeInfo), m_allClassMembers(a_pCodeInfo->allDataMembers), allDescriptorSetsInfo(a_pCodeInfo->allDescriptorSetsInfo),
                             m_kernels(a_pCodeInfo->kernels), m_mainFunc(a_mainFunc)
    { 
      for(const auto& arg : a_args) 
        m_argsOfMainFunc[arg.name] = arg;

      m_pRewrittenNodes = std::make_shared< std::unordered_set<uint64_t> > ();
    }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
    bool VisitCallExpr(CallExpr* f);
    bool VisitMemberExpr(MemberExpr* expr);
  
    std::string                                              mainFuncCmdName;
    std::unordered_map<std::string, uint32_t>                dsIdBySignature;

  private:

    std::vector<ArgReferenceOnCall> ExtractArgumentsOfAKernelCall(CallExpr* f, const std::unordered_set<std::string>& a_excludeList);
    std::string MakeKernelCallCmdString(CXXMemberCallExpr* f);
    std::string MakeServiceKernelCallCmdString(CallExpr* call, const std::string& a_name);

    Rewriter&                      m_rewriter;
    const clang::CompilerInstance& m_compiler;
    const clang::SourceManager&    m_sm;

  public:

    std::string                                              m_mainFuncName;
    const std::unordered_map<std::string, DataLocalVarInfo>& m_mainFuncLocals;
    MainClassInfo*                                           m_pCodeInfo = nullptr;
    std::unordered_map<std::string, DataMemberInfo>&         m_allClassMembers;
    std::vector< KernelCallInfo >&                           allDescriptorSetsInfo;

  private:
    
    std::unordered_map<std::string, KernelInfo>&       m_kernels;
    std::unordered_map<std::string, InOutVarInfo>      m_argsOfMainFunc;
    MainFuncInfo&                                      m_mainFunc;
    std::shared_ptr< std::unordered_set<uint64_t> >    m_pRewrittenNodes;
    std::unordered_map<uint64_t, std::string>          m_workAround;

    void ReplaceTextOrWorkAround(clang::SourceRange a_range, const std::string& a_text);
    bool WasNotRewrittenYet(const clang::Stmt* expr) const;
    void MarkRewritten(const clang::Stmt* expr);
    std::string RecursiveRewrite(const clang::Stmt* expr);
  };

  class MainFunctionRewriterCUDA : public RecursiveASTVisitor<MainFunctionRewriterCUDA>
  {
  public:
    
    MainFunctionRewriterCUDA(Rewriter &R, const clang::CompilerInstance& a_compiler, MainFuncInfo& a_mainFunc, const std::vector<InOutVarInfo>& a_args, MainClassInfo* a_pCodeInfo) : 
                             m_rewriter(R), m_compiler(a_compiler), m_sm(R.getSourceMgr()), m_pCodeInfo(a_pCodeInfo) {}

    MainClassInfo*   m_pCodeInfo = nullptr;

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* f);
    bool VisitCallExpr(clang::CallExpr* f);  

  private:

    Rewriter&                      m_rewriter;
    const clang::CompilerInstance& m_compiler;
    const clang::SourceManager&    m_sm;
  };
  
  std::string GetControlFuncDeclWGPU(const clang::FunctionDecl* fDecl, clang::CompilerInstance& compiler);
  std::string GetControlFuncDeclVulkan(const clang::FunctionDecl* fDecl, clang::CompilerInstance& compiler);
  std::string GetControlFuncDeclCUDA(const clang::FunctionDecl* fDecl, clang::CompilerInstance& compiler, bool a_gpuSuffix = false);

  std::vector<InOutVarInfo> ListParamsOfMainFunc(const CXXMethodDecl* a_node, const clang::CompilerInstance& compiler);
  void ObtainKernelsDecl(std::unordered_map<std::string, KernelInfo>& a_kernelsData, const clang::CompilerInstance& compiler, const std::string& a_mainClassName, const MainClassInfo& a_codeInfo);
}

#endif
