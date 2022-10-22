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
