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
  
    InitialPassRecursiveASTVisitor(std::vector<std::string>& a_mainFunctionNames, std::string main_class, CompilerInstance& a_compiler, const MainClassInfo& a_codeInfo) : 
                                   MAIN_CLASS_NAME(main_class), m_compiler(a_compiler), m_astContext(a_compiler.getASTContext()), m_sourceManager(a_compiler.getSourceManager()), m_codeInfo(a_codeInfo)  
    {
      m_mainFuncts.reserve(a_mainFunctionNames.size());
      for(const auto& name : a_mainFunctionNames)
        m_mainFuncts.insert(name);
    }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitFieldDecl    (FieldDecl* var);

    bool VisitCXXRecordDecl(CXXRecordDecl* record);
    bool VisitTypeDecl     (TypeDecl* record);
    bool VisitVarDecl      (VarDecl* pTargetVar);
  
    std::unordered_map<std::string, KernelInfo>           functions;
    std::unordered_map<std::string, DataMemberInfo>       dataMembers;
    std::unordered_map<std::string, const clang::CXXMethodDecl*> m_mainFuncNodes;
    std::unordered_map<std::string, const clang::CXXMethodDecl*> m_setters;
  
    std::unordered_map<std::string, KernelInfo>           otherFunctions;
    std::vector<const clang::CXXConstructorDecl* >        ctors;

    const CXXRecordDecl* m_mainClassASTNode = nullptr;
    std::vector<const clang::CXXRecordDecl*> m_classList;
    std::vector<kslicer::DeclInClass> GetExtractedDecls();

  private:
    void ProcessKernelDef(const CXXMethodDecl *f,  std::unordered_map<std::string, KernelInfo>& a_funcList, const std::string& a_className);
    bool NeedToProcessDeclInFile(std::string a_fileName);

    CompilerInstance&     m_compiler;
    const ASTContext&     m_astContext;
    clang::SourceManager& m_sourceManager;

    std::unordered_set<std::string> m_mainFuncts;
    const MainClassInfo&            m_codeInfo;

    uint32_t m_currId = 0;
    std::unordered_map<std::string, kslicer::DeclInClass> m_transferredDecl;
  };
  
  class InitialPassASTConsumer : public ASTConsumer
  {
   public:
  
    InitialPassASTConsumer (std::vector<std::string>& a_mainFunctionNames, std::string main_class, CompilerInstance& a_compiler, const MainClassInfo& a_codeInfo) : 
                            rv(a_mainFunctionNames, main_class, a_compiler, a_codeInfo) { }
    bool HandleTopLevelDecl(DeclGroupRef d) override;
    InitialPassRecursiveASTVisitor rv;
  };
}

#endif
