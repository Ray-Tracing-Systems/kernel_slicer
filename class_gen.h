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
  std::string MakeKernellCallSignature(const std::vector<ArgReferenceOnCall>& a_args, const std::string& a_mainFuncName);

  class MainFuncASTVisitor : public RecursiveASTVisitor<MainFuncASTVisitor>
  {
  public:
    
    MainFuncASTVisitor(Rewriter &R, const clang::CompilerInstance& a_compiler, MainFuncInfo& a_mainFunc, 
                       const std::vector<InOutVarInfo>& a_args, MainClassInfo* a_pCodeInfo) : 
                       m_rewriter(R), m_compiler(a_compiler), m_sm(R.getSourceMgr()), 
                       m_kernellCallTagId(0), m_mainFuncName(a_mainFunc.Name), m_mainFuncLocals(a_mainFunc.Locals),
                       m_pCodeInfo(a_pCodeInfo), m_allClassMembers(a_pCodeInfo->allDataMembers), m_mainFunc(a_mainFunc) 
    { 
      for(const auto& k : a_pCodeInfo->kernels)
        m_kernels[k.name] = k;   

      for(const auto& arg : a_args) 
        m_argsOfMainFunc[arg.name] = arg;
    }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
    bool VisitIfStmt(IfStmt* ifExpr);
  
    std::string                                              mainFuncCmdName;
    std::unordered_map<std::string, uint32_t>                dsIdBySignature;
    std::vector< KernelCallInfo >                            m_kernCallTypes;
    std::unordered_map<std::string, DataMemberInfo>&         m_allClassMembers;
    const std::unordered_map<std::string, DataLocalVarInfo>& m_mainFuncLocals;

  private:

    std::vector<ArgReferenceOnCall> ExtractArgumentsOfAKernelCall(CXXMemberCallExpr* f);
    std::string MakeKernelCallCmdString(CXXMemberCallExpr* f);

    Rewriter&                      m_rewriter;
    const clang::SourceManager&    m_sm;
    const clang::CompilerInstance& m_compiler;
    uint32_t m_kernellCallTagId;
    
    std::string m_mainFuncName;
    std::unordered_map<std::string, InOutVarInfo> m_argsOfMainFunc;
    std::unordered_map<std::string, KernelInfo>   m_kernels;
    MainFuncInfo& m_mainFunc;

    std::unordered_set<uint64_t> m_alreadyProcessedCalls;
    MainClassInfo* m_pCodeInfo = nullptr;
  };

  std::vector<InOutVarInfo> ListPointerParamsOfMainFunc(const CXXMethodDecl* a_node);

  class KernelReplacerASTVisitor : public RecursiveASTVisitor<KernelReplacerASTVisitor> // replace all expressions with class variables to kgen_data buffer access
  {
  public:
    
    KernelReplacerASTVisitor(Rewriter &R, const clang::CompilerInstance& a_compiler, const std::string& a_mainClassName, 
                             const std::vector<kslicer::DataMemberInfo>& a_variables, 
                             const std::vector<kslicer::KernelInfo::Arg>& a_args,
                             const std::string& a_fakeOffsetExpr,
                             const bool a_needToModifyReturn) : 
                             m_rewriter(R), m_compiler(a_compiler), m_mainClassName(a_mainClassName), m_args(a_args), m_fakeOffsetExp(a_fakeOffsetExpr), m_needModifyExitCond(a_needToModifyReturn) 
    { 
      m_variables.reserve(a_variables.size());
      for(const auto& var : a_variables) 
        m_variables[var.name] = var;
    }

    bool VisitMemberExpr(MemberExpr* expr);
    bool VisitUnaryOperator(UnaryOperator* expr);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
    bool VisitReturnStmt(ReturnStmt* ret);
  
  private:

    bool CheckIfExprHasArgumentThatNeedFakeOffset(const std::string& exprStr);

    Rewriter&   m_rewriter;
    const clang::CompilerInstance& m_compiler;
    std::string m_mainClassName;
    std::unordered_map<std::string, kslicer::DataMemberInfo> m_variables;
    const std::vector<kslicer::KernelInfo::Arg>&             m_args;
    const std::string&                                       m_fakeOffsetExp;
    bool                                                     m_needModifyExitCond;
  };

  void ObtainKernelsDecl(std::vector<KernelInfo>& a_kernelsData, const clang::CompilerInstance& compiler, const std::string& a_mainClassName, const MainClassInfo& a_codeInfo);

}

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);

#endif
