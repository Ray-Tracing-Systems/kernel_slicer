#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
#include "tests/tests.h"

using TestFuncType = bool (*)();

struct TestRun
{
  TestFuncType pTest;
  const char*  pTestName;
};

int main(int argc, const char** argv)
{
 
  TestRun tests[] = { {test000_scalar_functions_f, "test000_scalar_functions_f"},
                      {test001_dot_cross_f4,       "test001_dot_cross_f4"},
                      {test002_dot_cross_f3,       "test002_dot_cross_f3"},
                      {test003_length_f4,          "test003_length_f4"},
                      {test004_colpack,            "test004_colpack"},
                      {test005_matrix_elements,    "test005_matrix_elements"},
                      {test006_any_all,            "test006_any_all"},
                      {test007_reflect,            "test007_reflect"},
                      {test008_normalize,          "test008_normalize"},
                      {test009_refract,            "test009_refract"},
                      {test010_faceforward,        "test010_faceforward"},
                      
                      {test100_basev_float4,     "test100_basev_float4"},
                      {test101_basek_float4,   "test101_basek_float4"},
                      {test102_unaryv_float4,  "test102_unaryv_float4"},
                      {test102_unaryk_float4,  "test102_unaryk_float4"}, 
                      {test103_cmpv_float4,    "test103_cmpv_float4"}, 
                      {test104_shuffle_float4, "test104_shuffle_float4"},
                      {test105_extract_splat_float4, "test105_extract_splat_float4"},
                      {test107_funcv_float4,         "test107_funcv_float4"},
                      {test108_funcfv_float4,        "test108_funcfv_float4"},
                      {test109_cast_convert_float4,  "test109_cast_convert_float4"},
                      {test110_other_functions_float4, "test110_other_functions_float4"},

                      };
  
  const auto arraySize = sizeof(tests)/sizeof(TestRun);
  
  for(int i=0;i<int(arraySize);i++)
  {
    const bool res = tests[i].pTest();
    std::cout << "test\t" << std::setfill('0') << std::setw(3) << i+1 << "\t" << tests[i].pTestName << "\t";
    if(res)
      std::cout << "PASSED!";
    else 
      std::cout << "FAILED!\tFAILED!";
    std::cout << std::endl;
    std::cout.flush();
  }
  
  return 0;
}
