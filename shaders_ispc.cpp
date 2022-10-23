#include "kslicer.h"
#include "template_rendering.h"

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif

kslicer::ISPCCompiler::ISPCCompiler(bool a_useCPP)
{

}

std::string GetFolderPath(const std::string& a_filePath);

std::string kslicer::ISPCCompiler::BuildCommand(const std::string& a_inputFile) const
{
  return std::string("ispc ") + a_inputFile + " --target=\"avx2-i32x8\" -O2 ";
}

void kslicer::ISPCCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo) 
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

  std::string folderPath = GetFolderPath(mainClassFileName);
  std::string incUBOPath = folderPath + "/include";
  #ifdef WIN32
  mkdir(incUBOPath.c_str());
  #else
  mkdir(incUBOPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif

  const std::string templatePath = "templates_ispc/generated.ispc";
  const auto dotPos              = mainClassFileName.find_last_of(".");
  const std::string outFileName  = mainClassFileName.substr(0, dotPos) + "_kernels.ispc";
  const std::string outCppName   = mainClassFileName.substr(0, dotPos) + "_ispc.cpp";

  kslicer::ApplyJsonToTemplate(templatePath, outFileName, a_kernelsJson);  
  kslicer::ApplyJsonToTemplate("templates_ispc/ispc_class.cpp", outCppName, a_kernelsJson);
  
  std::ofstream buildSH(GetFolderPath(mainClassFileName) + "/z_build_ispc.sh");
  buildSH << "#!/bin/sh" << std::endl;
  std::string build = this->BuildCommand(outFileName) + " -o " + mainClassFileName.substr(0, dotPos) + "_kernels.o -h " + mainClassFileName.substr(0, dotPos) + "_kernels.h";
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

std::string kslicer::ISPCCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler)
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
