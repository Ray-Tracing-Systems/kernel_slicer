#ifndef SHADERS_CLSPV_H_
#define SHADERS_CLSPV_H_
#include "kslicer.h"


namespace kslicer {


  struct ClspvCompiler : IShaderCompiler
  {
    ClspvCompiler(bool a_useCPP, const std::string& a_prefix);
    std::string UBOAccess(const std::string& a_name) const override { return std::string("ubo->") + a_name; };
    bool        IsSingleShader()   const override { return true; }
    std::string ShaderFolder()     const override { return "clspv_shaders_aux"; }
    std::string ShaderSingleFile() const override { return "z_generated.cl"; }
    bool        BuffersAsPointersInShaders() const override { return true; }

    void        GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;

    bool        UseSeparateUBOForArguments() const override { return m_useCpp; }
    bool        UseSpecConstForWgSize()      const override { return m_useCpp; }

    std::string LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3])                               const override;
    std::string ReplaceCallFromStdNamespace(const std::string& a_call, const std::string& a_typeName) const override;
    void        GetThreadSizeNames(std::string a_strs[3])                                             const override;
    std::string GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    std::string GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;

    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string Name() const override { return "OpenCL"; }

    std::string RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const override;

  protected:
    virtual std::string BuildCommand(const std::string& a_inputFile = "") const;
    bool m_useCpp;
    const std::string& m_suffix;
  };


    
}

#endif