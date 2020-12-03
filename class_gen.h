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
    
    MainFuncASTVisitor(Rewriter &R, const std::string& a_mainFuncName, 
                       const std::unordered_map<std::string, InOutVarInfo>& a_args, 
                       const std::unordered_map<std::string, DataMemberInfo>& a_members,
                       const std::unordered_map<std::string, DataLocalVarInfo>& a_locals) : 
                       m_rewriter(R), m_sm(R.getSourceMgr()), m_kernellCallTagId(0), m_mainFuncName(a_mainFuncName), 
                       m_argsOfMainFunc(a_args), m_allClassMembers(a_members), m_mainFuncLocals(a_locals) { }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
  
    std::string                               mainFuncCmdName;
    std::unordered_map<std::string, uint32_t> dsIdBySignature;
    std::vector< KernelCallInfo >             m_kernCallTypes;
    const std::unordered_map<std::string, DataMemberInfo>&   m_allClassMembers;
    const std::unordered_map<std::string, DataLocalVarInfo>& m_mainFuncLocals;

  private:

    std::vector<ArgReferenceOnCall> ExtractArgumentsOfAKernelCall(CXXMemberCallExpr* f);

    Rewriter& m_rewriter;
    clang::SourceManager& m_sm;
    uint32_t m_kernellCallTagId;
    
    std::string m_mainFuncName;
    std::unordered_map<std::string, InOutVarInfo> m_argsOfMainFunc;
  };

  std::unordered_map<std::string, InOutVarInfo> ListPointerParamsOfMainFunc(const CXXMethodDecl* a_node);

  class KernelReplacerASTVisitor : public RecursiveASTVisitor<KernelReplacerASTVisitor> // replace all expressions with class variables to kgen_data buffer access
  {
  public:
    
    KernelReplacerASTVisitor(Rewriter &R, const std::string& a_mainClassName, const std::vector<kslicer::DataMemberInfo>& a_variables) : 
                             m_rewriter(R), m_mainClassName(a_mainClassName) 
    { 
      m_variables.reserve(a_variables.size());
      for(const auto& var : a_variables) 
        m_variables[var.name] = var;
    }

    bool VisitMemberExpr(MemberExpr* expr);
  
  private:
    Rewriter&   m_rewriter;
    std::string m_mainClassName;
    std::unordered_map<std::string, kslicer::DataMemberInfo> m_variables;
  };

  void ObtainKernelsDecl(std::vector<KernelInfo>& a_kernelsData, clang::SourceManager& sm, const std::string& a_mainClassName);

}

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);

#endif
