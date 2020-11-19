#ifndef KSLICER_INITIAL_PASS_H
#define KSLICER_INITIAL_PASS_H

#include "kslicer.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"

namespace kslicer
{
  using namespace clang;
  
  // RecursiveASTVisitor is the big-kahuna visitor that traverses everything in the AST.
  //
  class InitialPassRecursiveASTVisitor : public RecursiveASTVisitor<InitialPassRecursiveASTVisitor>
  {
  public:
    
    std::string MAIN_NAME;
    std::string MAIN_CLASS_NAME;
    std::string MAIN_FILE_INCLUDE;
  
    InitialPassRecursiveASTVisitor(std::string main_name, std::string main_class, const ASTContext& a_astContext, clang::SourceManager& a_sm) : 
                                   MAIN_NAME(main_name), MAIN_CLASS_NAME(main_class), m_mainFuncNode(nullptr), m_astContext(a_astContext), m_sourceManager(a_sm)  { }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitFieldDecl    (FieldDecl* var);
  
    std::unordered_map<std::string, KernelInfo>     functions;
    std::unordered_map<std::string, DataMemberInfo> dataMembers;
    const CXXMethodDecl* m_mainFuncNode;
  
  private:
    void ProcessKernelDef(const CXXMethodDecl *f);
    void ProcessMainFunc(const CXXMethodDecl *f);
  
    const ASTContext&     m_astContext;
    clang::SourceManager& m_sourceManager;
  };
  
  class InitialPassASTConsumer : public ASTConsumer
  {
   public:
  
    InitialPassASTConsumer(std::string main_name, std::string main_class, const ASTContext& a_astContext, clang::SourceManager& a_sm) : rv(main_name, main_class, a_astContext, a_sm) { }
    bool HandleTopLevelDecl(DeclGroupRef d) override;
    InitialPassRecursiveASTVisitor rv;
  };

};

#endif
