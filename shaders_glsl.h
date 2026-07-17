#ifndef SHADERS_GLSL_H_
#define SHADERS_GLSL_H_
#include "kslicer.h"

namespace kslicer {

  /**
  \brief process local functions
  */
  class GLSLFunctionRewriter : public FunctionRewriter //
  {
  public:
  
    GLSLFunctionRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit);
    ~GLSLFunctionRewriter(){}
  
    bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl) override;
    bool VisitCallExpr_Impl(clang::CallExpr* f)             override;
    bool VisitVarDecl_Impl(clang::VarDecl* decl)            override;
    bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast) override;
    bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
    bool VisitMemberExpr_Impl(clang::MemberExpr* expr)         override;
    bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr)   override;
    bool VisitDeclStmt_Impl(clang::DeclStmt* decl)             override;
    bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)  override;
    bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;
  
    bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f) override;
    bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;
  
    std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override;
    IRecursiveRewriteOverride* m_pKernelRewriter = nullptr;
  
    std::string RewriteStdVectorTypeStr(const std::string& a_str) const override;
    std::string RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const override;
    std::string RewriteImageType(const std::string& a_containerType, const std::string& a_containerDataType, kslicer::TEX_ACCESS a_accessType, std::string& outImageFormat) const override;
  
    std::unordered_map<std::string, std::string> m_vecReplacements;
    std::unordered_map<std::string, std::string> m_funReplacements;
    std::vector<std::pair<std::string, std::string> > m_vecReplacements2;

  
    std::string RewriteFuncDecl(clang::FunctionDecl* fDecl) override;
    std::string RecursiveRewrite(const clang::Stmt* expr) override;
    void        ApplyDefferedWorkArounds();
    
    struct BadRewqriteResult
    {
      std::string text;
      bool        isSingle;
      bool        isRewritten;
    };

    void        Get2DIndicesOfFloat4x4(const clang::CXXOperatorCallExpr* expr, const clang::Expr* out[3]);
  
    bool        NeedsVectorTypeRewrite(const std::string& a_str) override;
    std::string CompleteFunctionCallRewrite(clang::CallExpr* call);  
  };


  struct GLSLCompiler : IShaderCompiler
  {
    GLSLCompiler(const std::string& a_prefix);
    std::string UBOAccess(const std::string& a_name) const override { return std::string("ubo.") + a_name; };
    std::string ProcessBufferType(const std::string& a_typeName) const override;
    
    bool        IsSingleShader()                     const override { return false;}
    bool        MemberFunctionsAreSupported()        const override { return true; }
    std::string ShaderFolder()                       const override { return std::string("shaders") + ToLowerCase(m_suffix); }
    std::string ShaderSingleFile()                   const override { return ""; }

    void GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;

    std::string LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const override;
    void        GetThreadSizeNames(std::string a_strs[3])               const override;
    std::string GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    std::string GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;

    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string Name() const override { return "GLSL"; }

    std::string RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const override;
    std::string RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds) override; 
    
    std::string IndirectBufferDataType() const override { return "uvec4 "; }

  private:
    const std::string& m_suffix;
    void ProcessVectorTypesString(std::string& a_str);
  };

}


#endif