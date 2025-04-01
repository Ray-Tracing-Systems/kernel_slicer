#include "kslicer.h"
#include "template_rendering.h"
#include <iostream>

#ifndef _WIN32
  #include <sys/types.h>
#endif

std::string kslicer::FunctionRewriter::RewriteStdVectorTypeStr(const std::string& a_str) const
{
  const bool isConst = (a_str.find("const ") != std::string::npos);
  std::string typeStr = a_str;
  ReplaceFirst(typeStr, "struct LiteMath::", "");
  ReplaceFirst(typeStr, "LiteMath::", "");
  ReplaceFirst(typeStr, "struct glm::", "");
  ReplaceFirst(typeStr, "glm::", "");
  ReplaceFirst(typeStr, "const ",    "");
  ReplaceFirst(typeStr, m_codeInfo->mainClassName + "::", "");
  ReplaceFirst(typeStr, "struct float4x4", "float4x4");       // small inconvinience in math library
  if(isConst)
    typeStr = std::string("const ") + typeStr;
  return typeStr;
}


bool kslicer::IsVectorContructorNeedsReplacement(const std::string& a_typeName)
{
  static std::unordered_set<std::string> m_ctorReplacement;
  static bool first = true;
  if(first)
  {
    m_ctorReplacement.insert("float2");
    m_ctorReplacement.insert("float3");
    m_ctorReplacement.insert("float4");
    m_ctorReplacement.insert("float2x2");
    m_ctorReplacement.insert("float3x3");
    m_ctorReplacement.insert("float4x4");
    m_ctorReplacement.insert("int2");
    m_ctorReplacement.insert("int3");
    m_ctorReplacement.insert("int4");
    m_ctorReplacement.insert("uint2");
    m_ctorReplacement.insert("uint3");
    m_ctorReplacement.insert("uint4");
    m_ctorReplacement.insert("complex");
    first = false;
  }

  return m_ctorReplacement.find(a_typeName) != m_ctorReplacement.end();
}

kslicer::ClspvCompiler::ClspvCompiler(bool a_useCPP, const std::string& a_prefix) : m_useCpp(a_useCPP), m_suffix(a_prefix)
{

}

std::string kslicer::ClspvCompiler::BuildCommand(const std::string& a_inputFile) const
{
  if(m_useCpp)
    return std::string("../clspv ") + ShaderSingleFile() + " -o " + ShaderSingleFile() + ".spv -pod-ubo -cl-std=CLC++ -inline-entry-points";
  else
    return std::string("../clspv ") + ShaderSingleFile() + " -o " + ShaderSingleFile() + ".spv -pod-pushconstant";
}


void kslicer::ClspvCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  const auto& mainClassFileName = a_codeInfo->mainClassFileName;
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;

  std::filesystem::path folderPath = mainClassFileName.parent_path();
  std::filesystem::path incUBOPath = folderPath / "include";
  std::filesystem::create_directory(incUBOPath);

  const std::string templatePath = "templates/generated.cl";
  const std::filesystem::path outFileName  = mainClassFileName.parent_path() / "z_generated.cl";
  kslicer::ApplyJsonToTemplate(templatePath, outFileName, a_kernelsJson);

  std::ofstream buildSH(mainClassFileName.parent_path() / "z_build.sh");
  #if not __WIN32__
  buildSH << "#!/bin/sh" << std::endl;
  #endif
  std::string build = this->BuildCommand();
  buildSH << build.c_str() << " ";
  for(auto folder : ignoreFolders) {
    if(folder.string().find("TINYSTL") != std::string::npos)
      continue;
    buildSH << "-I" << folder.c_str() << " ";
  }
  buildSH << std::endl;
  buildSH.close();
}

std::string kslicer::ClspvCompiler::LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const
{
  if(a_kernelDim == 1)
    return "get_local_id(0)";
  else if(a_kernelDim == 2)
  {
    std::stringstream strOut;
    strOut << "get_local_id(0) + " << a_wgSize[0] << "*get_local_id(1)";
    return strOut.str();
  }
  else if(a_kernelDim == 3)
  {
    std::stringstream strOut;
    strOut << "get_local_id(0) + " << a_wgSize[0] << "*get_local_id(1) + " << a_wgSize[0]*a_wgSize[1] << "*get_local_id(2)";
    return strOut.str();
  }
  else
  {
    std::cout << "  [ClspvCompiler::LocalIdExpr]: Error, bad kernelDim = " << a_kernelDim << std::endl;
    return "get_local_id(0)";
  }
}

void kslicer::ClspvCompiler::GetThreadSizeNames(std::string a_strs[3]) const
{
  a_strs[0] = "kgen_iNumElementsX";
  a_strs[1] = "kgen_iNumElementsY";
  a_strs[2] = "kgen_iNumElementsZ";
}

std::string kslicer::ClspvCompiler::GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const
{
  return "unknownCLSubgroupOperation";
}

std::string kslicer::ClspvCompiler::GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const
{
  std::string res = "";
  switch(a_access.type)
  {
    case KernelInfo::REDUCTION_TYPE::ADD_ONE:
    case KernelInfo::REDUCTION_TYPE::ADD:
    res = "atomic_add";
    break;

    case KernelInfo::REDUCTION_TYPE::SUB:
    case KernelInfo::REDUCTION_TYPE::SUB_ONE:
    res = "atomic_sub";
    break;

    case KernelInfo::REDUCTION_TYPE::FUNC:
    {
      if(a_access.funcName == "min" || a_access.funcName == "std::min") res = "atomic_min";
      if(a_access.funcName == "max" || a_access.funcName == "std::max") res = "atomic_max";
    }
    break;

    default:
    break;
  };

  auto lastSymb  = a_access.dataType[a_access.dataType.size()-1];
  auto firstSimb = a_access.dataType[0]; // 'u' or 'i'
  if(isdigit(lastSymb))
  {
    res.push_back(lastSymb);  
    if(firstSimb == 'u')
      res.push_back(firstSimb);  
  }

  return res;
}


std::string kslicer::ClspvCompiler::ReplaceCallFromStdNamespace(const std::string& a_call, const std::string& a_typeName) const
{
  std::string call = a_call;
  if(a_typeName == "float" || a_typeName == "const float" || a_typeName == "float2" || a_typeName == "const float2" ||
     a_typeName == "float3" || a_typeName == "const float3" || a_typeName == "float4" || a_typeName == "const float4")
  {
    ReplaceFirst(call, "std::min", "fmin");
    ReplaceFirst(call, "std::max", "fmax");
    ReplaceFirst(call, "std::abs", "fabs");

    if(call == "min")
      ReplaceFirst(call, "min", "fmin");
    if(call == "max")
      ReplaceFirst(call, "max", "fmax");
    if(call == "abs")
      ReplaceFirst(call, "abs", "fabs");
  }
  ReplaceFirst(call, "std::", "");
  return call;
}

std::string kslicer::ClspvCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter)
{
  std::string typeInCL = a_decl.type;
  ReplaceFirst(typeInCL, "const", "__constant static");
  std::string result = "";
  switch(a_decl.kind)
  {
    case kslicer::DECL_IN_CLASS::DECL_STRUCT:
    result = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    result = typeInCL + " " + a_decl.name + " = " + kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    result = "typedef " + typeInCL + " " + a_decl.name + ";";
    break;
    default:
    break;
  };
  ReplaceFirst(result, "LiteMath::", "");
  ReplaceFirst(result, "std::", "");
  return result;
}

std::string kslicer::ClspvCompiler::RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const
{
  return std::string("{ uint offset = atomic_inc(&") + UBOAccess(memberNameB) + "); " + memberNameA + "[offset] = " + newElemValue + ";}";
}

std::shared_ptr<kslicer::FunctionRewriter> kslicer::ClspvCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit)
{
  return std::make_shared<kslicer::FunctionRewriter>(R, a_compiler, a_codeInfo);
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::ClspvCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                                  kslicer::KernelInfo& a_kernel, const std::string& fakeOffs)
{
  return std::make_shared<kslicer::KernelRewriter>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::KernelRewriter::KernelRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, const std::string& a_fakeOffsetExpr) :
                                        m_rewriter(R), m_compiler(a_compiler), m_codeInfo(a_codeInfo), m_mainClassName(a_codeInfo->mainClassName),
                                        m_args(a_kernel.args), m_fakeOffsetExp(a_fakeOffsetExpr), m_kernelIsBoolTyped(a_kernel.isBoolTyped),
                                        m_currKernel(a_kernel)
{
  m_pRewrittenNodes = std::make_shared< std::unordered_set<uint64_t> >();
  const auto& a_variables = a_codeInfo->dataMembers;
  m_variables.reserve(a_variables.size());
  for(const auto& var : a_variables)
    m_variables[var.name] = var;

  auto tidArgs = a_codeInfo->GetKernelTIDArgs(a_kernel);
  for(const auto& arg : tidArgs)
    m_threadIdArgs.push_back(arg.name);
  m_explicitIdISPC = a_kernel.explicitIdISPC;
  if(tidArgs.size() > 0)
    m_threadIdExplicitIndexISPC = tidArgs[tidArgs.size()-1].name;
}


bool kslicer::KernelRewriter::WasNotRewrittenYet(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return true;
  if(clang::isa<clang::NullStmt>(expr))
    return true;
  auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
  return (m_pRewrittenNodes->find(exprHash) == m_pRewrittenNodes->end());
}

void kslicer::KernelRewriter::MarkRewritten(const clang::Stmt* expr)
{
  kslicer::MarkRewrittenRecursive(expr, *m_pRewrittenNodes);
}

void kslicer::DisplayVisitedNodes(const std::unordered_set<uint64_t>& a_nodes)
{
  std::vector<uint64_t> allNodes;
  allNodes.reserve(a_nodes.size());
  for(auto p : a_nodes)
    allNodes.push_back(p);

  std::sort(allNodes.begin(), allNodes.end());
  for(size_t i=0; i<allNodes.size(); i++)
    std::cout << i << "\t" << allNodes[i] << std::endl;
}


