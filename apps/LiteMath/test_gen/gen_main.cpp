#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include <vector>
#include <string>  

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
  nlohmann::json data;
  data["AllTests"] = std::vector<std::string>();
  int currNumber   = 100;
  
  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"] = currNumber;
    test["Type"]   = "float4";
    test["TypeS"]  = "float";
    test["VecLen"]  = 4;
    test["ValuesA"] = std::vector<int>({1, 2, -3, 4});
    test["ValuesB"] = std::vector<int>({5, -5, 6, 4});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xFFFFFFFF, 0xF0F00000, 0x00000000});
    test["IsFloat"]  = true;
    test["IsSigned"] = true;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_float4.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "uint4";
    test["TypeS"]   = "uint";
    test["VecLen"]  = 4;
    test["ValuesA"] = std::vector<uint32_t>({1, 2, uint32_t(-3), 4});
    test["ValuesB"] = std::vector<uint32_t>({5, uint32_t(-5), 6, 4});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xFFFFFFFF, 0xF0F00000, 0x00000000});
    test["IsFloat"]  = false;
    test["IsSigned"] = false;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_uint4.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "int4";
    test["TypeS"]   = "int";
    test["VecLen"]  = 4;
    test["ValuesA"] = std::vector<int>({1, 2, -3, 4});
    test["ValuesB"] = std::vector<int>({5, -5, 6, 4});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xFFFFFFFF, 0xF0F00000, 0x00000000});
    test["IsFloat"]  = false;
    test["IsSigned"] = true;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_int4.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "uint3";
    test["TypeS"]   = "uint";
    test["VecLen"]  = 3;
    test["ValuesA"] = std::vector<uint32_t>({1, 2, uint32_t(-3)});
    test["ValuesB"] = std::vector<uint32_t>({5, uint32_t(-5), 6});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xFFFFFFFF, 0xF0F00000});
    test["IsFloat"]  = false;
    test["IsSigned"] = false;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_uint3.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "int3";
    test["TypeS"]   = "int";
    test["VecLen"]  = 3;
    test["ValuesA"] = std::vector<int>({1, 2, -3});
    test["ValuesB"] = std::vector<int>({5, -5, 6});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xFFFFFFFF, 0xF0F00000});
    test["IsFloat"]  = false;
    test["IsSigned"] = true;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_int3.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  } 

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"] = currNumber;
    test["Type"]   = "float3";
    test["TypeS"]  = "float";
    test["VecLen"]  = 3;
    test["ValuesA"] = std::vector<int>({-1, 2, -3});
    test["ValuesB"] = std::vector<int>({3, -4, 4});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xFFFFFFFF, 0xF0F00000});
    test["IsFloat"]  = true;
    test["IsSigned"] = true;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_float3.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"] = currNumber;
    test["Type"]   = "float2";
    test["TypeS"]  = "float";
    test["VecLen"]  = 2;
    test["ValuesA"] = std::vector<int>({-1, 2});
    test["ValuesB"] = std::vector<int>({3, -4});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xFFFFFFFF});
    test["IsFloat"]  = true;
    test["IsSigned"] = true;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_float2.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }

  ApplyJsonToTemplate("templates/tests.h",        "../tests/tests.h", data);
  ApplyJsonToTemplate("templates/tests_main.cpp", "../tests/tests_main.cpp", data);

  return 0;
}
