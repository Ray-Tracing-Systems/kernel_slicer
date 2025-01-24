#include "kslicer.h"
#include "template_rendering.h"

#ifdef _WIN32
  #include <sys/types.h>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::SlangRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)        
{
  auto hash = kslicer::GetHashOfSourceRange(fDecl->getBody()->getSourceRange());
  if(m_codeInfo->m_functionsDone.find(hash) == m_codeInfo->m_functionsDone.end()) // it is important to put functions in 'm_functionsDone'
  {
    kslicer::RewrittenFunction done;
    done.funDecl = kslicer::GetRangeSourceCode(fDecl->getSourceRange(),            m_compiler); 
    auto posBrace = done.funDecl.find("{");
    if(posBrace != std::string::npos)
      done.funDecl = done.funDecl.substr(0,posBrace); // discard func body source code
    done.funBody = kslicer::GetRangeSourceCode(fDecl->getBody()->getSourceRange(), m_compiler);
    m_codeInfo->m_functionsDone[hash] = done;
  }  
  return true; 
}

bool kslicer::SlangRewriter::VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)      { return true; }
bool kslicer::SlangRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)             { return true; }
bool kslicer::SlangRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  { return true; } 
bool kslicer::SlangRewriter::VisitFieldDecl_Impl(clang::FieldDecl* decl)               { return true; }
bool kslicer::SlangRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) { return true; } 
bool kslicer::SlangRewriter::VisitCallExpr_Impl(clang::CallExpr* f)                    { return true; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions

bool kslicer::SlangRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* op)
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitVarDecl_Impl(clang::VarDecl* decl) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)           
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true;   
}

bool kslicer::SlangRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl)             
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel

bool kslicer::SlangRewriter::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  if(m_kernelMode)
  {
    // ...
  }

  return true;
}

bool  kslicer::SlangRewriter::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) 
{
  if(m_kernelMode)
  {
    // ...
  }

  return true;
}

std::string kslicer::SlangRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
    
  SlangRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
    return m_rewriter.getRewrittenText(range);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


kslicer::SlangCompiler::SlangCompiler(const std::string& a_prefix) : m_suffix(a_prefix)
{

}


std::string kslicer::SlangCompiler::LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const
{
  // uint3 a_localTID  : SV_GroupThreadID
  // uint3 a_globalTID : SV_DispatchThreadID
  if(a_kernelDim == 1)
    return "a_localTID.x";
  else if(a_kernelDim == 2)
  {
    std::stringstream strOut;
    strOut << "a_localTID.x + " << a_wgSize[0] << "*a_localTID.y";
    return strOut.str();
  }
  else if(a_kernelDim == 3)
  {
    std::stringstream strOut;
    strOut << "a_localTID.x + " << a_wgSize[0] << "*a_localTID.y + " << a_wgSize[0]*a_wgSize[1] << "*a_localTID.z";
    return strOut.str();
  }
  else
  {
    std::cout << "  [SlangCompiler::LocalIdExpr]: Error, bad kernelDim = " << a_kernelDim << std::endl;
    return "a_localTID.x";
  }
}

void kslicer::SlangCompiler::GetThreadSizeNames(std::string a_strs[3]) const
{
  a_strs[0] = "iNumElementsX";
  a_strs[1] = "iNumElementsY";
  a_strs[2] = "iNumElementsZ";
}

std::string kslicer::SlangCompiler::RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const
{
  return std::string("{ uint offset = atomicAdd(") + UBOAccess(memberNameB) + ", 1); " + memberNameA + "[offset] = " + newElemValue + ";}";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<kslicer::FunctionRewriter> kslicer::SlangCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                                                                    MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit)
{
  return std::make_shared<SlangRewriter>(R, a_compiler, a_codeInfo);
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::SlangCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                                                                  MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, const std::string& fakeOffs)
{
  auto pFunc = std::make_shared<SlangRewriter>(R, a_compiler, a_codeInfo);
  pFunc->m_kernelMode = true;
  return std::make_shared<KernelRewriter2>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs, pFunc);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void kslicer::SlangCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  const auto& mainClassFileName = a_codeInfo->mainClassFileName;
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;

  #ifdef _WIN32
  const std::string scriptName = "build_slang.bat";
  #else
  const std::string scriptName = "build_slang.sh";
  #endif

  std::filesystem::path folderPath = mainClassFileName.parent_path();
  std::filesystem::path shaderPath = folderPath / this->ShaderFolder();
  std::filesystem::path incUBOPath = folderPath / "include";
  std::filesystem::create_directory(shaderPath);
  std::filesystem::create_directory(incUBOPath);

  // generate header for all used functions in GLSL code
  //
  std::string headerCommon = "common" + ToLowerCase(m_suffix) + "_slang.h";
  std::filesystem::path templatesFolder("templates_slang");
  kslicer::ApplyJsonToTemplate(templatesFolder / "common_generated_slang.h", shaderPath / headerCommon, a_kernelsJson);

  const std::filesystem::path templatePath       = templatesFolder / (a_codeInfo->megakernelRTV ? "generated_mega.slang" : "generated.slang");
  const std::filesystem::path templatePathUpdInd = templatesFolder / "update_indirect.slang";
  const std::filesystem::path templatePathRedFin = templatesFolder / "reduction_finish.slang";
  
  nlohmann::json copy, kernels, intersections;
  for (auto& el : a_kernelsJson.items())
  {
    //std::cout << el.key() << std::endl;
    if(std::string(el.key()) == "Kernels")
      kernels = a_kernelsJson[el.key()];
    else
      copy[el.key()] = a_kernelsJson[el.key()];
  }

  std::ofstream buildSH(shaderPath / scriptName);
  #if not __WIN32__
  buildSH << "#!/bin/sh" << std::endl;
  #endif
  for(auto& kernel : kernels.items())
  {
    nlohmann::json currKerneJson = copy;
    currKerneJson["Kernel"] = kernel.value();

    std::string kernelName     = std::string(kernel.value()["Name"]);
    bool useRayTracingPipeline = kernel.value()["UseRayGen"];
    const bool vulkan11        = kernel.value()["UseSubGroups"];
    const bool vulkan12        = useRayTracingPipeline;

    std::string outFileName           = kernelName + ".slang";
    std::filesystem::path outFilePath = shaderPath / outFileName;
    kslicer::ApplyJsonToTemplate(templatePath.c_str(), outFilePath, currKerneJson);
    
    buildSH << "slangc " << outFileName.c_str() << " -o " << kernelName.c_str() << ".comp.spv" << " -DGLSL -I.. ";
    for(auto folder : ignoreFolders)
      buildSH << "-I" << folder.c_str() << " ";
    //if(vulkan12)
    //  buildSH << "--target-env vulkan1.2 ";
    //else if(vulkan11)
    //  buildSH << "--target-env vulkan1.1 ";
    //if(useRayTracingPipeline)
    //  buildSH << "-S rgen ";
    buildSH << std::endl;
  
    //if(kernel.value()["IsIndirect"])
    //{
    //  outFileName = kernelName + "_UpdateIndirect.slang";
    //  outFilePath = shaderPath / outFileName;
    //  kslicer::ApplyJsonToTemplate(templatePathUpdInd.c_str(), outFilePath, currKerneJson);
    //  buildSH << "glslangValidator -V ";
    //  if(vulkan11)
    //    buildSH << "--target-env vulkan1.1 ";
    //  buildSH << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
    //  for(auto folder : ignoreFolders)
    //   buildSH << "-I" << folder.c_str() << " ";
    //  buildSH << std::endl;
    //}

    //if(kernel.value()["FinishRed"])
    //{
    //  outFileName = kernelName + "_Reduction.slang";
    //  outFilePath = shaderPath / outFileName;
    //  kslicer::ApplyJsonToTemplate(templatePathRedFin.c_str(), outFilePath, currKerneJson);
    //  buildSH << "glslangValidator -V ";
    //  if(vulkan11)
    //    buildSH << "--target-env vulkan1.1 ";
    //  buildSH << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
    //  for(auto folder : ignoreFolders)
    //   buildSH << "-I" << folder.c_str() << " ";
    //  buildSH << std::endl;
    //}
  }

}


std::string kslicer::SlangCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter)
{
  std::string typeInCL = a_decl.type;
  std::string result = "";
  switch(a_decl.kind)
  {
    case kslicer::DECL_IN_CLASS::DECL_STRUCT:
    result = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    ReplaceFirst(typeInCL, "_Bool", "bool");
    result = typeInCL + " " + a_decl.name + " = " + kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    result = "typedef " + typeInCL + " " + a_decl.name + ";";
    break;
    default:
    break;
  };
  return result;
}

