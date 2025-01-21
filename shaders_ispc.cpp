#include "kslicer.h"
#include "template_rendering.h"

#ifdef _WIN32
  #include <sys/types.h>
#endif

bool kslicer::FunctionRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)
{
  auto hash = kslicer::GetHashOfSourceRange(fDecl->getBody()->getSourceRange());
  if(m_codeInfo->m_functionsDone.find(hash) == m_codeInfo->m_functionsDone.end())
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


kslicer::ISPCCompiler::ISPCCompiler(bool a_useCPP, const std::string& a_prefix) : ClspvCompiler(a_useCPP, a_prefix)
{

}

std::string kslicer::ISPCCompiler::BuildCommand(const std::string& a_inputFile) const
{
  return std::string("ispc ") + a_inputFile + " --target=\"avx2-i32x8\" -O2 ";
}

void kslicer::ISPCCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  const auto& mainIncluideName  = a_codeInfo->mainClassFileInclude;
  const auto& mainClassFileName = a_codeInfo->mainClassFileName;
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;

  {
    const auto nameRel  = mainIncluideName.substr(mainIncluideName.find_last_of("/")+1);
    const auto dotPos   = nameRel.find_last_of(".");
    const auto ispcName = nameRel.substr(0, dotPos) + "_kernels.h";
    a_kernelsJson["MainISPCFile"] = ispcName;
  }

  std::filesystem::path folderPath = mainClassFileName.parent_path();
  std::filesystem::path incUBOPath = folderPath / "include";
  std::filesystem::create_directory(incUBOPath);

  const std::string templatePath = "templates_ispc/generated.ispc";
  std::filesystem::path outFileName = mainClassFileName;
  outFileName.replace_extension("");
  outFileName.concat("_kernels.ispc");
  std::filesystem::path outCppName = mainClassFileName;
  outCppName.replace_extension("");
  outCppName.concat("_ispc.cpp");

  kslicer::ApplyJsonToTemplate(templatePath, outFileName, a_kernelsJson);
  kslicer::ApplyJsonToTemplate("templates_ispc/ispc_class.cpp", outCppName, a_kernelsJson);

  std::ofstream buildSH(mainClassFileName.parent_path() / "z_build_ispc.sh");
  #if not __WIN32__
  buildSH << "#!/bin/sh" << std::endl;
  #endif
  std::filesystem::path kernelTarget = mainClassFileName;
  kernelTarget.replace_extension("");
  kernelTarget.concat("_kernels.o");
  std::filesystem::path kernelHeader = mainClassFileName;
  kernelHeader.replace_extension("");
  kernelHeader.concat("_kernels.h");
  std::string build = this->BuildCommand(outFileName.u8string()) + " -o " + kernelTarget.u8string() + " -h " + kernelHeader.u8string();
  buildSH << build.c_str() << " ";
  for(auto folder : ignoreFolders)
    buildSH << "-I" << folder.c_str() << " ";
  buildSH << std::endl;
  buildSH.close();
}

std::unordered_map<std::string, std::string> ListISPCVectorReplacements()
{
  std::unordered_map<std::string, std::string> m_vecReplacements;
  m_vecReplacements["float2"] = "float2";
  m_vecReplacements["float3"] = "float3";
  m_vecReplacements["float4"] = "float4";
  m_vecReplacements["int2"]   = "int2";
  m_vecReplacements["int3"]   = "int3";
  m_vecReplacements["int4"]   = "int4";
  m_vecReplacements["uint2"]  = "uint2";
  m_vecReplacements["uint3"]  = "uint3";
  m_vecReplacements["uint4"]  = "uint4";
  //m_vecReplacements["float4x4"] = "mat4";
  m_vecReplacements["_Bool"] = "bool";
  return m_vecReplacements;
}

const std::string ConvertVecTypesToISPC(const std::string& a_typeName,
                                        const std::string& a_argName)

{
  static const auto vecTypes = ListISPCVectorReplacements();
  std::string nameToSearch = a_typeName;
  ReplaceFirst(nameToSearch, "const ", "");
  while(nameToSearch[nameToSearch.size()-1] == ' ')
    nameToSearch = nameToSearch.substr(0, nameToSearch.size()-1);
  if(vecTypes.find(nameToSearch) != vecTypes.end() || nameToSearch.find("struct") != std::string::npos)
  {
    if (nameToSearch.find("struct") != std::string::npos)
      ReplaceFirst(nameToSearch, "struct ", "");

    if(a_typeName.find("const ") != std::string::npos)
      return "(const ispc::" + nameToSearch + "*)" + a_argName;
    else
      return "(ispc::" + nameToSearch + "*)" + a_argName;
  }
  return a_argName;
}

std::string kslicer::ISPCCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter)
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

std::string kslicer::ISPCCompiler::ReplaceCallFromStdNamespace(const std::string& a_call, const std::string& a_typeName) const
{
  std::string call = a_call;
  ReplaceFirst(call, "std::", "");
  return call;
}
