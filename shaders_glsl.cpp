#include "kslicer.h"
#include <iostream>

std::string kslicer::GLSLCompiler::BuildCommand() const 
{ 
  return std::string("../circle -shader -c -emit-spirv ") + ShaderSingleFile() + " -o " + ShaderSingleFile() + ".spv -DUSE_CIRCLE_CC"; 
}
   
void kslicer::GLSLCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const std::string& mainClassFileName, const std::vector<std::string>& includeToShadersFolders)
{
  // "templates_glsl/generated.glsl"
  // // clspv unfortunately force use to use this hacky way to set desired destcripror set (see -distinct-kernel-descriptor-sets option of clspv).
  // // we create for each kernel with indirect dispatch seperate file with first dummy kernel (which will be bound to zero-th descriptor set)
  // // and our XXX_IndirectUpdate kerner which in that way will be bound to the first descriptor set.  
  // //
  // if(inputCodeInfo.m_indirectBufferSize != 0) 
  // {
  //   nlohmann::json copy, kernels;
  //   for (auto& el : json.items())
  //   {
  //     //std::cout << el.key() << std::endl;
  //     if(std::string(el.key()) == "Kernels")
  //       kernels = json[el.key()];
  //     else
  //       copy[el.key()] = json[el.key()];
  //   }
  // 
  //   std::string folderPath = GetFolderPath(inputCodeInfo.mainClassFileName);
  //   std::string shaderPath = folderPath + "/" + inputCodeInfo.pShaderCC->ShaderFolder();
  //   #ifdef WIN32
  //   mkdir(shaderPath.c_str());
  //   #else
  //   mkdir(shaderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  //   #endif
  // 
  //   //for(auto& kernel : kernels.items())
  //   //{
  //   //  if(kernel.value()["IsIndirect"])
  //   //  {
  //   //    nlohmann::json currKerneJson = copy;
  //   //    currKerneJson["Kernels"] = std::vector<std::string>();
  //   //    currKerneJson["Kernels"].push_back(kernel.value());
  //   //
  //   //    std::string outFileName = std::string(kernel.value()["Name"]) + "_UpdateIndirect" + ".cl";
  //   //    std::string outFilePath = shaderPath + "/" + outFileName;
  //   // 
  //   //    std::ofstream file("debug.json");
  //   //    file << currKerneJson.dump(2);
  //   //
  //   //    kslicer::ApplyJsonToTemplate("templates/indirect.cl", outFilePath, currKerneJson);
  //   //    buildSH << "../clspv " << outFilePath.c_str() << " -o " << outFilePath.c_str() << ".spv -pod-pushconstant -distinct-kernel-descriptor-sets -I." << std::endl;
  //   //  }
  //   //}
  // 
  // }
  
  /*
    nlohmann::json copy, kernels;
    for (auto& el : json.items())
    {
      //std::cout << el.key() << std::endl;
      if(std::string(el.key()) == "Kernels")
        kernels = json[el.key()];
      else
        copy[el.key()] = json[el.key()];
    }
    
    std::string folderPath = GetFolderPath(inputCodeInfo.mainClassFileName);
    std::string shaderPath = folderPath + "/" + inputCodeInfo.pShaderCC->ShaderFolder();
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
      
      std::string outFileName = std::string(kernel.value()["Name"]) + ".cpp";
      std::string outFilePath = shaderPath + "/" + outFileName;
      kslicer::ApplyJsonToTemplate("templates/gen_circle.cxx", outFilePath, currKerneJson);
      buildSH << "../../circle -shader -c -emit-spirv " << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DUSE_CIRCLE_CC -I.. " << std::endl;
    }
    
    nlohmann::json emptyJson;
    std::string outFileServ = shaderPath + "/" + "serv_kernels.cpp";
    kslicer::ApplyJsonToTemplate("templates/ser_circle.cxx", outFileServ, emptyJson);
    buildSH << "../../circle -shader -c -emit-spirv " << outFileServ.c_str() << " -o " << outFileServ.c_str() << ".spv" << " -DUSE_CIRCLE_CC -I.. " << std::endl;

    buildSH.close();
    */
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