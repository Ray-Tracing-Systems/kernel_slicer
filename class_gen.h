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

  class MainFuncASTVisitor : public RecursiveASTVisitor<MainFuncASTVisitor>
  {
  public:
    
    MainFuncASTVisitor(Rewriter &R) : m_rewriter(R) { }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
  
    std::string mainFuncCmdName;

  private:
    Rewriter& m_rewriter;
  };

  std::string ProcessMainFunc(const CXXMethodDecl* a_node, clang::CompilerInstance& compiler, const std::string& a_mainClassName, 
                              std::string& a_outFuncDecl);

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

}


#endif
