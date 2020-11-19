#include "template_rendering.h"
#include <inja.hpp>

// Just for convenience
using namespace inja;
using json = nlohmann::json;

void kslicer::PrintGeneratedClassDecl(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo, std::ostream& a_out)
{
  json data;
  data["Includes"] = "";
  data["MainClassName"] = a_classInfo.mainClassName;
  data["MainFuncName"]  = a_classInfo.mainFuncName;
  
  inja::Environment env;
  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, data);
  
  a_out << result.c_str() << std::endl;
} 
