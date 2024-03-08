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
  struct ClassInfo
  {
    ClassInfo(){}
    ClassInfo(const std::string& a_name) : name(a_name) {}
    std::string                                                  name;
    const clang::CXXRecordDecl*                                  astNode = nullptr;
    std::unordered_map<std::string, KernelInfo>                  functions;
    std::unordered_map<std::string, KernelInfo>                  otherFunctions;
    std::unordered_map<std::string, DataMemberInfo>              dataMembers;
    std::unordered_map<std::string, const clang::CXXMethodDecl*> m_mainFuncNodes;
    std::unordered_map<std::string, const clang::CXXMethodDecl*> m_setters;
    std::unordered_map<std::string, const clang::CXXMethodDecl*> allMemberFunctions;
    std::vector<const clang::CXXConstructorDecl* >               ctors;
  };

  std::string PerformClassComposition(ClassInfo& mainClassInfo, const ClassInfo& apiClassInfo, const ClassInfo& implClassInfo);


  using namespace clang;

  // RecursiveASTVisitor is the big-kahuna visitor that traverses everything in the AST.
  //
  class InitialPassRecursiveASTVisitor : public RecursiveASTVisitor<InitialPassRecursiveASTVisitor>
  {
  public:

    std::string MAIN_CLASS_NAME;
    std::string MAIN_FILE_INCLUDE;

    InitialPassRecursiveASTVisitor(std::vector<std::string>& a_mainFunctionNames,
                                   std::string main_class,
                                   std::vector<std::string> compos_classes,
                                   CompilerInstance& a_compiler, MainClassInfo& a_codeInfo) :
                                   MAIN_CLASS_NAME(main_class), m_compiler(a_compiler), m_astContext(a_compiler.getASTContext()), m_sourceManager(a_compiler.getSourceManager()), m_codeInfo(a_codeInfo)
    {
      m_mainFuncts.reserve(a_mainFunctionNames.size());
      for(const auto& name : a_mainFunctionNames)
        m_mainFuncts.insert(name);
      for(const auto& name : compos_classes)
        m_composedClassInfo[name] = ClassInfo(name);
      mci.name = MAIN_CLASS_NAME;
    }

    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitFieldDecl    (FieldDecl* var);

    bool VisitCXXRecordDecl(CXXRecordDecl* record);
    bool VisitTypeDecl     (TypeDecl* record);
    bool VisitVarDecl      (VarDecl* pTargetVar);

    ClassInfo mci; // main class info
    std::unordered_map<std::string, ClassInfo>  m_composedClassInfo;

    std::vector<const clang::CXXRecordDecl*> m_classList;
    std::vector<kslicer::DeclInClass> GetExtractedDecls();

    const std::unordered_map<std::string, kslicer::DeclInClass>& GetOtherTypeDecls() const { return m_storedDecl;}

  private:
    void ProcessKernelDef(const CXXMethodDecl *f,  std::unordered_map<std::string, KernelInfo>& a_funcList, const std::string& a_className);
    bool IsMainClassName(const std::string& a_typeName);

    CompilerInstance&     m_compiler;
    const ASTContext&     m_astContext;
    clang::SourceManager& m_sourceManager;

    std::unordered_set<std::string> m_mainFuncts;
    MainClassInfo&                  m_codeInfo;

    uint32_t m_currId = 0;
    std::unordered_map<std::string, kslicer::DeclInClass> m_transferredDecl;
    std::unordered_map<std::string, kslicer::DeclInClass> m_storedDecl;
  };

  class InitialPassASTConsumer : public ASTConsumer
  {
   public:

    InitialPassASTConsumer (std::vector<std::string>& a_mainFunctionNames,
                            std::string main_class,
                            std::vector<std::string> compos_classes,
                            CompilerInstance& a_compiler, MainClassInfo& a_codeInfo) :
                            rv(a_mainFunctionNames, main_class, compos_classes, a_compiler, a_codeInfo) { }
    bool HandleTopLevelDecl(DeclGroupRef d) override;
    InitialPassRecursiveASTVisitor rv;
  };

  std::string ClearTypeName(const std::string& a_typeName);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class HeaderLister : public clang::PPCallbacks
{
public:

  HeaderLister(kslicer::MainClassInfo* a_pInfo) : m_pGlobInfo(a_pInfo) {}

  /*void InclusionDirective(clang::SourceLocation HashLoc,
                          const clang::Token &IncludeTok,
                          llvm::StringRef FileName, bool IsAngled,
                          clang::CharSourceRange FilenameRange,
                          const clang::FileEntry *File,
                          llvm::StringRef SearchPath,
                          llvm::StringRef RelativePath,
                          const clang::Module *Imported,
                          clang::SrcMgr::CharacteristicKind FileType) override */

  void InclusionDirective (clang::SourceLocation HashLoc,
                      const clang::Token &IncludeTok,
                      clang::StringRef FileName,
                      bool IsAngled,
                      clang::CharSourceRange FilenameRange,
                      clang::OptionalFileEntryRef File,
                      clang::StringRef SearchPath,
                      clang::StringRef RelativePath,
                      const clang::Module *Imported,
                      clang::SrcMgr::CharacteristicKind FileType)
  {
    if(!IsAngled && File != nullptr)
    {
      assert(File != nullptr);
      std::string filename = std::string(RelativePath.begin(), RelativePath.end());
      m_pGlobInfo->allIncludeFiles[filename] = false;
    }
  }

private:

  kslicer::MainClassInfo* m_pGlobInfo;

};

#endif
