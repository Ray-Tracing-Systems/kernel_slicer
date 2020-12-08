#ifndef KSLICER_INITIAL_PASS_H
#define KSLICER_INITIAL_PASS_H

#include "kslicer.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"

#include <unordered_set>

namespace kslicer
{
  using namespace clang;
  
  // RecursiveASTVisitor is the big-kahuna visitor that traverses everything in the AST.
  //
  class InitialPassRecursiveASTVisitor : public RecursiveASTVisitor<InitialPassRecursiveASTVisitor>
  {
  public:
    
    std::string MAIN_CLASS_NAME;
    std::string MAIN_FILE_INCLUDE;
  
    InitialPassRecursiveASTVisitor(std::vector<std::string>& a_mainFunctionNames, std::string main_class, const ASTContext& a_astContext, clang::SourceManager& a_sm) : 
                                   MAIN_CLASS_NAME(main_class), m_astContext(a_astContext), m_sourceManager(a_sm)  
    {
      m_mainFuncts.reserve(a_mainFunctionNames.size());
      for(const auto& name : a_mainFunctionNames)
        m_mainFuncts.insert(name);
    }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitFieldDecl    (FieldDecl* var);
  
    std::unordered_map<std::string, KernelInfo>           functions;
    std::unordered_map<std::string, DataMemberInfo>       dataMembers;
    std::unordered_map<std::string, const CXXMethodDecl*> m_mainFuncNodes;
  
  private:
    void ProcessKernelDef(const CXXMethodDecl *f);
  
    const ASTContext&     m_astContext;
    clang::SourceManager& m_sourceManager;

    std::unordered_set<std::string> m_mainFuncts;
  };
  
  class InitialPassASTConsumer : public ASTConsumer
  {
   public:
  
    InitialPassASTConsumer (std::vector<std::string>& a_mainFunctionNames, std::string main_class, const ASTContext& a_astContext, clang::SourceManager& a_sm) : rv(a_mainFunctionNames, main_class, a_astContext, a_sm) { }
    bool HandleTopLevelDecl(DeclGroupRef d) override;
    InitialPassRecursiveASTVisitor rv;
  };

};

#endif
