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
 
  TestRun tests[] = { {test000_scalar_funcs,    "test000_scalar_funcs"},
                      {test001_dot_cross_f4,    "test001_dot_cross_f4"},
                      {test002_dot_cross_f3,    "test002_dot_cross_f3"},
                      {test003_length_float4,   "test003_length_float4"},
                      {test004_colpack_f4x4,    "test004_colpack_f4x4"},
                      {test005_matrix_elems,    "test005_matrix_elems"},
                      {test006_any_all,         "test006_any_all"},
                      {test007_reflect,         "test007_reflect"},
                      {test008_normalize,       "test008_normalize"},
                      {test009_refract,         "test009_refract"},
                      {test010_faceforward,     "test010_faceforward"},
                      
                      {test100_basev_float4,         "test100_basev_float4"},
                      {test101_basek_float4,         "test101_basek_float4"},
                      {test102_unaryv_float4,        "test102_unaryv_float4"},
                      {test102_unaryk_float4,        "test102_unaryk_float4"}, 
                      {test103_cmpv_float4,          "test103_cmpv_float4"}, 
                      {test104_shuffle_float4,       "test104_shuffle_float4"},
                      {test105_exsplat_float4,       "test105_exsplat_float4"},
                      {test107_funcv_float4,         "test107_funcv_float4"},
                      {test108_funcfv_float4,        "test108_funcfv_float4"},
                      {test109_cstcnv_float4,        "test109_cstcnv_float4"},
                      {test110_other_float4,        "test110_other_float4"},

                      {test110_basev_float3,         "test110_basev_float3"},
                      {test111_basek_float3,         "test111_basek_float3"},
                      {test112_unaryv_float3,        "test112_unaryv_float3"},
                      {test112_unaryk_float3,        "test112_unaryk_float3"}, 
                      {test113_cmpv_float3,          "test113_cmpv_float3"}, 
                      {test114_shuffle_float3,       "test114_shuffle_float3"},
                      {test115_exsplat_float3,       "test115_exsplat_float3"},
                      {test117_funcv_float3,         "test117_funcv_float3"},
                      {test118_funcfv_float3,        "test118_funcfv_float3"},
                      {test119_cstcnv_float3,        "test119_cstcnv_float3"},
                      {test120_other_float3,        "test120_other_float3"},

                      };
  
  const auto arraySize = sizeof(tests)/sizeof(TestRun);
  
  for(int i=0;i<int(arraySize);i++)
  {
    const bool res = tests[i].pTest();
    std::cout << "test " << std::setfill('0') << std::setw(3) << i << " " << tests[i].pTestName << "\t";
    if(res)
      std::cout << "PASSED!";
    else 
      std::cout << "FAILED!";
    std::cout << std::endl;
    std::cout.flush();
  }
  
  return 0;
}
