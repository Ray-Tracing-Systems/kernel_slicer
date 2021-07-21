#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include <vector>
#include <string>  

#include <inja.hpp>

void ApplyJsonToTemplate(const std::string& a_declTemplateFilePath, const std::string& a_outFilePath, const nlohmann::json& a_data)
{
  inja::Environment env;
  env.set_trim_blocks(false);
  env.set_lstrip_blocks(true);

  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, a_data);
  
  std::ofstream fout(a_outFilePath);
  fout << result.c_str() << std::endl;
  fout.close();
}

std::string ByConstRefOrValue(const std::string& a_name)
{
  //return a_name;
  return std::string("const ") + a_name;
  //return std::string("const ") + a_name + "&";
}

int main(int argc, const char** argv)
{
  nlohmann::json data;
  data["AllTests"] = std::vector<std::string>();
  int currNumber   = 100;
  
  std::vector<std::string> XYZW4 = std::vector<std::string>({"x","y","z","w"});
  std::vector<std::string> XYZW3 = std::vector<std::string>({"x","y","z"});
  std::vector<std::string> XYZW2 = std::vector<std::string>({"x","y"});

  std::string openExpr  = "{";
  std::string closeExpr = "}";

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "uint4";
    test["TypeS"]   = "uint";
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 4;
    test["XYZW"]    = XYZW4;
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
    test["TypeC"]  = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 4;
    test["XYZW"]    = XYZW4;
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
    test["Number"] = currNumber;
    test["Type"]   = "float4";
    test["TypeS"]  = "float";
    test["TypeC"]  = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 4;
    test["XYZW"]    = XYZW4;
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
    test["Type"]    = "uint3";
    test["TypeS"]   = "uint";
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 3;
    test["XYZW"]    = XYZW3;
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
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 3;
    test["XYZW"]    = XYZW3;
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
    test["Number"]  = currNumber;
    test["Type"]    = "float3";
    test["TypeS"]   = "float";
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 3;
    test["XYZW"]    = XYZW3;
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
    test["Number"]  = currNumber;
    test["Type"]    = "uint2";
    test["TypeS"]   = "uint";
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 2;
    test["XYZW"]    = XYZW2;
    test["ValuesA"] = std::vector<uint32_t>({1, 2});
    test["ValuesB"] = std::vector<uint32_t>({5, uint32_t(-5)});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xF0F00000});
    test["IsFloat"]  = false;
    test["IsSigned"] = false;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_uint2.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "int2";
    test["TypeS"]   = "int";
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 2;
    test["XYZW"]    = XYZW2;
    test["ValuesA"] = std::vector<int>({1, 2});
    test["ValuesB"] = std::vector<int>({5, -5});
    test["ValuesC"] = std::vector<uint32_t>({0xFFFFFFFF, 0xF0F00000});
    test["IsFloat"]  = false;
    test["IsSigned"] = true;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_int2.cpp", dataLocal);
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
    test["TypeC"]  = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 2;
    test["XYZW"]    = XYZW2;
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
   
  /*
  { 
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "ushort4";
    test["TypeS"]   = "ushort";
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 4;
    test["XYZW"]    = XYZW4;
    test["ValuesA"] = std::vector<unsigned short>({1, 2, (unsigned short)(-3), 4});
    test["ValuesB"] = std::vector<unsigned short>({5, (unsigned short)(-5), 6, 4});
    test["ValuesC"] = std::vector<unsigned short>({0xFFFF, 0xFFFF, 0xF0F0, 0x0000});
    test["IsFloat"]  = false;
    test["IsSigned"] = false;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_ushort4.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }
  
  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "ushort2";
    test["TypeS"]   = "ushort";
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 2;
    test["XYZW"]    = XYZW2;
    test["ValuesA"] = std::vector<unsigned short>({1, 2});
    test["ValuesB"] = std::vector<unsigned short>({5, (unsigned short)(-5) });
    test["ValuesC"] = std::vector<unsigned short>({0x0000FFFF, 0x0000FFFF});
    test["IsFloat"]  = false;
    test["IsSigned"] = false;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_ushort2.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }

  {
    nlohmann::json dataLocal;
    dataLocal["Tests"] = std::vector<std::string>();

    nlohmann::json test;
    test["Number"]  = currNumber;
    test["Type"]    = "uchar4";
    test["TypeS"]   = "uchar";
    test["TypeC"]   = ByConstRefOrValue(test["Type"]);
    test["VecLen"]  = 4;
    test["XYZW"]    = XYZW4;
    test["ValuesA"] = std::vector<unsigned char>({1, 2, (unsigned char)(-3), 4});
    test["ValuesB"] = std::vector<unsigned char>({5, (unsigned char)(-5), 6, 4});
    test["ValuesC"] = std::vector<unsigned char>({0x000000FF, 0x000000FF, 0x000000F0, 0});
    test["IsFloat"]  = false;
    test["IsSigned"] = false;  
  
    dataLocal["Tests"].push_back(test);
    ApplyJsonToTemplate("templates/tests_arith.cpp", "../tests/tests_uchar4.cpp", dataLocal);
    currNumber += 10;

    data["AllTests"].push_back(dataLocal);
  }*/

  data["OPN"]      = openExpr;
  data["CLS"]      = closeExpr;

  ApplyJsonToTemplate("templates/tests.h",        "../tests/tests.h", data);
  ApplyJsonToTemplate("templates/tests_main.cpp", "../tests/tests_main.cpp", data);
  ApplyJsonToTemplate("templates/lite_math.h",    "../LiteMath.h", data);

  return 0;
}
