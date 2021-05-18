#include "kslicer.h"
#include "template_rendering.h"
#include <iostream>

kslicer::ClspvCompiler::ClspvCompiler(bool a_useCPP) : m_useCpp(a_useCPP)
{
  m_ctorReplacement["float2"] = "make_float2";
  m_ctorReplacement["float3"] = "make_float3";
  m_ctorReplacement["float4"] = "make_float4";
  m_ctorReplacement["int2"]   = "make_int2";
  m_ctorReplacement["int3"]   = "make_int3";
  m_ctorReplacement["int4"]   = "make_int4";
  m_ctorReplacement["uint2"]  = "make_uint2";
  m_ctorReplacement["uint3"]  = "make_uint3";
  m_ctorReplacement["uint4"]  = "make_uint4";
}


std::string kslicer::ClspvCompiler::BuildCommand() const 
{
  if(m_useCpp) 
    return std::string("../clspv ") + ShaderSingleFile() + " -o " + ShaderSingleFile() + ".spv -pod-ubo -cl-std=CLC++ -inline-entry-points";  
  else
    return std::string("../clspv ") + ShaderSingleFile() + " -o " + ShaderSingleFile() + ".spv -pod-pushconstant";
} 

std::string GetFolderPath(const std::string& a_filePath);

void kslicer::ClspvCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const std::string& mainClassFileName, const std::vector<std::string>& includeToShadersFolders) 
{
  const std::string templatePath = "templates/generated.cl";
  const std::string outFileName  = GetFolderPath(mainClassFileName) + "/z_generated.cl";
  kslicer::ApplyJsonToTemplate(templatePath, outFileName, a_kernelsJson);  

  std::ofstream buildSH(GetFolderPath(mainClassFileName) + "/z_build.sh");
  buildSH << "#!/bin/sh" << std::endl;
  std::string build = this->BuildCommand();
  buildSH << build.c_str() << " ";
  for(auto folder : includeToShadersFolders)
    buildSH << "-I" << folder.c_str() << " ";
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

bool kslicer::ClspvCompiler::IsVectorTypeNeedsContructorReplacement(const std::string& a_typeName) const 
{
  return (m_ctorReplacement.find(a_typeName) != m_ctorReplacement.end());
}

std::string kslicer::ClspvCompiler::VectorTypeContructorReplace(const std::string& a_typeName, const std::string& a_call) const  
{ 
  std::string call = a_call;
  auto p = m_ctorReplacement.find(a_typeName);
  ReplaceFirst(call, p->first + "(", p->second + "(");
  return call; 
}
