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

void kslicer::ISPCCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo) 
{
  const auto& mainClassFileName = a_codeInfo->mainClassFileName;
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;
  
  std::string folderPath = GetFolderPath(mainClassFileName);
  std::string incUBOPath = folderPath + "/include";
  #ifdef WIN32
  mkdir(incUBOPath.c_str());
  #else
  mkdir(incUBOPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif

  const std::string templatePath = "templates_ispc/generated.ispc";
  const std::string outFileName  = mainClassFileName.substr(0, mainClassFileName.find_last_of(".")) + "_kernels.ispc";
  kslicer::ApplyJsonToTemplate(templatePath, outFileName, a_kernelsJson);  

  //std::ofstream buildSH(GetFolderPath(mainClassFileName) + "/z_build.sh");
  //buildSH << "#!/bin/sh" << std::endl;
  //std::string build = this->BuildCommand();
  //buildSH << build.c_str() << " ";
  //for(auto folder : ignoreFolders)
  //  buildSH << "-I" << folder.c_str() << " ";
  //buildSH << std::endl;
  //buildSH.close();
}
