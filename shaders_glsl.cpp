#include "kslicer.h"
#include "template_rendering.h"
#include <iostream>

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif


std::string GetFolderPath(const std::string& a_filePath);


void kslicer::GLSLCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const std::string& mainClassFileName, const std::vector<std::string>& includeToShadersFolders)
{
  const std::string templatePath = "templates_glsl/generated.glsl";
  
  nlohmann::json copy, kernels;
  for (auto& el : a_kernelsJson.items())
  {
    //std::cout << el.key() << std::endl;
    if(std::string(el.key()) == "Kernels")
      kernels = a_kernelsJson[el.key()];
    else
      copy[el.key()] = a_kernelsJson[el.key()];
  }
    
  std::string folderPath = GetFolderPath(mainClassFileName);
  std::string shaderPath = folderPath + "/" + this->ShaderFolder();
  #ifdef WIN32
  mkdir(shaderPath.c_str());
  #else
  mkdir(shaderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif
    
  std::ofstream buildSH(shaderPath + "/build.sh");
  buildSH << "#!/bin/sh" << std::endl;
  for(auto& kernel : kernels.items())
  {
    nlohmann::json currKerneJson = copy;
    currKerneJson["Kernels"] = std::vector<std::string>();
    currKerneJson["Kernels"].push_back(kernel.value());
    
    std::string outFileName = std::string(kernel.value()["Name"]) + ".glsl";
    std::string outFilePath = shaderPath + "/" + outFileName;
    kslicer::ApplyJsonToTemplate(templatePath.c_str(), outFilePath, currKerneJson);
    buildSH << "glslangValidator -V " << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
    for(auto folder : includeToShadersFolders)
     buildSH << "-I" << folder.c_str() << " ";
    buildSH << std::endl;

    //glslangValidator -e myEntryPoint // is needed for auxilary kernels!
  }
    
  //nlohmann::json emptyJson;
  //std::string outFileServ = shaderPath + "/" + "serv_kernels.cpp";
  //kslicer::ApplyJsonToTemplate("templates/ser_circle.cxx", outFileServ, emptyJson);
  //buildSH << "../../circle -shader -c -emit-spirv " << outFileServ.c_str() << " -o " << outFileServ.c_str() << ".spv" << " -DUSE_CIRCLE_CC -I.. " << std::endl;
  
  buildSH.close();
}

std::string kslicer::GLSLCompiler::LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const
{
  if(a_kernelDim == 1)
    return "gl_LocalInvocationID.x";
  else if(a_kernelDim == 2)
  {
    std::stringstream strOut;
    strOut << "gl_LocalInvocationID.x + " << a_wgSize[0] << "*gl_LocalInvocationID.y";
    return strOut.str();
  }
  else if(a_kernelDim == 3)
  {
    std::stringstream strOut;
    strOut << "gl_LocalInvocationID.x + " << a_wgSize[0] << "*gl_LocalInvocationID.y + " << a_wgSize[0]*a_wgSize[1] << "*gl_LocalInvocationID.z";
    return strOut.str();
  }
  else
  {
    std::cout << "  [GLSLCompiler::LocalIdExpr]: Error, bad kernelDim = " << a_kernelDim << std::endl;
    return "gl_LocalInvocationID.x";
  }
}

std::string kslicer::GLSLCompiler::ProcessBufferType(const std::string& a_typeName) const 
{ 
  std::string type = a_typeName;
  ReplaceFirst(type, "*", "");
  ReplaceFirst(type, "const", "");
  return type; 
};