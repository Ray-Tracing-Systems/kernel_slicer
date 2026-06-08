#ifndef SHADERS_ISPC_H_
#define SHADERS_ISPC_H_
#include "kslicer.h"
#include "shaders_clspv.h"

namespace kslicer {

  class ISPCRewriter : public FunctionRewriter2 ///!< BASE CLASS FOR ALL NEW BACKENDS
  {
  public:
    ISPCRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo) : FunctionRewriter2(R,a_compiler,a_codeInfo) { Init();}
    ~ISPCRewriter(){ }

    bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)   override;
    bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl) override;

    bool VisitVarDecl_Impl(clang::VarDecl* decl)                  override;
    bool VisitDeclStmt_Impl(clang::DeclStmt* decl)                override;
    bool VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr)  override;

    bool VisitMemberExpr_Impl(clang::MemberExpr* expr)             override;
    bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  override; 
    bool VisitFieldDecl_Impl(clang::FieldDecl* decl)               override;
    bool VisitUnaryOperator_Impl(clang::UnaryOperator* op)         override;
    bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)     override;
    bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
    bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) override; 
    bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;

    bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)            override;
    bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;
    bool VisitCallExpr_Impl(clang::CallExpr* f)                                        override;

    bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) override;
    bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr)                 override;
    bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)                       override;

    // Also important functions to use(!)
    //
    bool        NeedsVectorTypeRewrite(const std::string& a_str) override;
    std::string RewriteStdVectorTypeStr(const std::string& a_str) const override;
    std::string RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const override;
    
    std::string RecursiveRewrite(const clang::Stmt* expr) override;
    std::string RewriteFuncDecl(clang::FunctionDecl* fDecl) override;

    std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override;
  private:
    void Init();
    std::unordered_map<std::string, std::string> m_typesReplacement;
    std::unordered_map<std::string, std::string> m_funReplacements;
  };

  struct ISPCCodeGen : public IHostCodeGen
  {
    std::string Name() const override { return "ISPC"; }
    void GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) override;
  };

  struct ISPCCompiler : ClspvCompiler
  {
    ISPCCompiler(bool a_useCPP, const std::string& a_prefix);
    std::string UBOAccess(const std::string& a_name) const override { return std::string("ubo[0].") + a_name; };
    std::string Name() const override { return "ISPC"; }
    void        GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;
    bool        IsISPC() const override { return true; }
    std::string BuildCommand(const std::string& a_inputFile) const override;
    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string ReplaceCallFromStdNamespace(const std::string& a_call, const std::string& a_typeName) const override;
    bool        BuffersAsPointersInShaders() const override { return false; }
    bool        SupportAtomicGlobal(const KernelInfo::ReductionAccess& acc) const override { return true; }

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;
  };

}

#endif