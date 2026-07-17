#ifndef SHADERS_CUDA_H_
#define SHADERS_CUDA_H_

#include "kslicer.h"


namespace kslicer {

  class CudaRewriter : public FunctionRewriter2 ///!< BASE CLASS FOR ALL NEW BACKENDS
  {
  public:
    CudaRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo) : FunctionRewriter2(R,a_compiler,a_codeInfo) { Init();}
    ~CudaRewriter(){ }

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
    
    //
    //
    std::string RecursiveRewrite(const clang::Stmt* expr) override;
    std::string RewriteFuncDecl(clang::FunctionDecl* fDecl) override;

    std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override;
  private:
    void Init();
    std::unordered_map<std::string, std::string> m_typesReplacement;
    std::unordered_map<std::string, std::string> m_funReplacements;
  };


  struct CudaCodeGen : public IHostCodeGen
  {
    CudaCodeGen(const std::string& a_actualCUDAImpl) : m_actualCUDAImpl(a_actualCUDAImpl) {}
    std::string Name() const override { return m_actualCUDAImpl; }
    void GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) override;
    bool IsCUDA() const override { return true; }
    std::string m_actualCUDAImpl;
  };

  struct CudaCompiler : IShaderCompiler
  {
    CudaCompiler(const std::string& a_prefix);
    std::string UBOAccess(const std::string& a_name) const override 
    {
      if(a_name.find(".size()") != std::string::npos) // kernelJson["IndirectSizeX"]  = a_classInfo.pShaderCC->UBOAccess(exprContent);
        return a_name;
      else
        return std::string("ubo.") + a_name; 
    } //  { return a_name; }
    std::string ReplaceSizeCapacityExpr(const std::string& a_str) const override { return a_str; }
    std::string ProcessBufferType(const std::string& a_typeName) const override;

    bool        IsSingleShader()                     const override { return true; }
    bool        MemberFunctionsAreSupported()        const override { return true; }
    std::string ShaderFolder()                       const override { return ""; }
    std::string ShaderSingleFile()                   const override { return ""; }
    
    bool        IsGLSL() const override { return false; }
    bool        IsISPC() const override { return false; }
    bool        IsCUDA() const override { return true;  }

    void GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;

    std::string LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const override;
    void        GetThreadSizeNames(std::string a_strs[3])               const override;
    std::string GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    std::string GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    bool        SupportAtomicGlobal(const KernelInfo::ReductionAccess& acc) const override { return true; }

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;

    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string Name() const override { return "CUDA"; }

    std::string RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const override;
    std::string RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds) override; 

    std::string IndirectBufferDataType() const override { return "uint4 "; }

  private:
    const std::string& m_suffix;
    std::unordered_map<std::string, std::string> m_typesReplacement;
  };

}

#endif

