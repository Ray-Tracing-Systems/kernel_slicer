#include "template_rendering.h"
#include <inja.hpp>

// Just for convenience
using namespace inja;
using json = nlohmann::json;

void kslicer::PrintGeneratedClassDecl(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo)
{
  std::string rawname;
  {
    size_t lastindex = a_classInfo.mainClassFileName.find_last_of("."); 
    assert(lastindex != std::string::npos);
    rawname = a_classInfo.mainClassFileName.substr(0, lastindex); 
  }

  std::string folderPath;
  {
    size_t lastindex = a_classInfo.mainClassFileName.find_last_of("/"); 
    assert(lastindex != std::string::npos);   
    folderPath = a_classInfo.mainClassFileName.substr(0, lastindex); 
  }

  std::string mainInclude = a_classInfo.mainClassFileInclude;
  
  if(mainInclude.find(folderPath) != std::string::npos)  // cut off folder path
    mainInclude = mainInclude.substr(folderPath.size() + 1);

  std::stringstream strOut;
  strOut << "#include \"" << mainInclude.c_str() << "\"" << std::endl;

  json data;
  data["Includes"]      = strOut.str();
  data["MainClassName"] = a_classInfo.mainClassName;
  data["MainFuncName"]  = a_classInfo.mainFuncName;
  
  inja::Environment env;
  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, data);
  
  std::ofstream fout(rawname + "_generated.h");
  fout << result.c_str() << std::endl;
  fout.close();
} 
