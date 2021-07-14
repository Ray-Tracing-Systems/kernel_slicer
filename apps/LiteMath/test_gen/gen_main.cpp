#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include <inja.hpp>

//#ifdef WIN32
//  #include <direct.h>     // for windows mkdir
//#else
//  #include <sys/stat.h>   // for linux mkdir
//  #include <sys/types.h>
//#endif

void ApplyJsonToTemplate(const std::string& a_declTemplateFilePath, const std::string& a_outFilePath, const nlohmann::json& a_data)
{
  inja::Environment env;
  env.set_trim_blocks(true);
  env.set_lstrip_blocks(true);

  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, a_data);
  
  std::ofstream fout(a_outFilePath);
  fout << result.c_str() << std::endl;
  fout.close();
}

int main(int argc, const char** argv)
{
  nlohmann::json test;
  
  test["Number"] = 101;
  test["Name"]   = "base_arith";
  test["Type"]   = "float4";
  test["TypeS"]  = "float";
  test["VecLen"]  = 4;
  test["ValuesA"] = "1.0f, 2.0f, -3.0f, 4.0f";
  test["ValuesB"] = "5.0f, -5.0f, 6.0f, 7.0f";
  test["IsFloat"] = (test["TypeS"] == "float") || (test["TypeS"] == "double");  

  nlohmann::json data;
  data["Test"] = test;

  //#ifdef WIN32
  //mkdir("generated");
  //#else
  //mkdir("generated", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  //#endif

  ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_float4.cpp", data);

  return 0;
}
