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
 
  TestRun tests[] = { {test000_scalar_funcs,  "test000_scalar_funcs"},
                      {test001_dot_cross_f4,  "test001_dot_cross_f4"},
                      {test002_dot_cross_f3,  "test002_dot_cross_f3"},
                      {test003_length_float4, "test003_length_float4"},
                      {test004_colpack_f4x4,  "test004_colpack_f4x4"},
                      {test005_matrix_elems,  "test005_matrix_elems"},
                      {test006_any_all,       "test006_any_all"},
                      {test007_reflect,       "test007_reflect"},
                      {test008_normalize,     "test008_normalize"},
                      {test009_refract,       "test009_refract"},
                      {test010_faceforward,   "test010_faceforward"},
                      {test011_mattranspose,  "test011_mattranspose"},


                      {test100_basev_uint4,         "test100_basev_uint4"},
                      {test101_basek_uint4,         "test101_basek_uint4"},
                      {test102_unaryv_uint4,        "test102_unaryv_uint4"},
                      {test102_unaryk_uint4,        "test102_unaryk_uint4"}, 
                      {test103_cmpv_uint4,          "test103_cmpv_uint4"}, 
                      {test104_shuffle_uint4,       "test104_shuffle_uint4"},
                      {test105_exsplat_uint4,       "test105_exsplat_uint4"},
                      {test107_funcv_uint4,         "test107_funcv_uint4"},

                      {test108_logicv_uint4,        "test108_logicv_uint4"},
                      {test109_cstcnv_uint4,        "test109_cstcnv_uint4"},

                      {test110_other_uint4,        "test110_other_uint4"},



                      {test110_basev_int4,         "test110_basev_int4"},
                      {test111_basek_int4,         "test111_basek_int4"},
                      {test112_unaryv_int4,        "test112_unaryv_int4"},
                      {test112_unaryk_int4,        "test112_unaryk_int4"}, 
                      {test113_cmpv_int4,          "test113_cmpv_int4"}, 
                      {test114_shuffle_int4,       "test114_shuffle_int4"},
                      {test115_exsplat_int4,       "test115_exsplat_int4"},
                      {test117_funcv_int4,         "test117_funcv_int4"},

                      {test118_logicv_int4,        "test118_logicv_int4"},
                      {test119_cstcnv_int4,        "test119_cstcnv_int4"},

                      {test120_other_int4,        "test120_other_int4"},



                      {test120_basev_float4,         "test120_basev_float4"},
                      {test121_basek_float4,         "test121_basek_float4"},
                      {test122_unaryv_float4,        "test122_unaryv_float4"},
                      {test122_unaryk_float4,        "test122_unaryk_float4"}, 
                      {test123_cmpv_float4,          "test123_cmpv_float4"}, 
                      {test124_shuffle_float4,       "test124_shuffle_float4"},
                      {test125_exsplat_float4,       "test125_exsplat_float4"},
                      {test127_funcv_float4,         "test127_funcv_float4"},

                      {test128_funcfv_float4,        "test128_funcfv_float4"},
                      {test129_cstcnv_float4,        "test129_cstcnv_float4"},

                      {test130_other_float4,        "test130_other_float4"},



                      {test130_basev_uint3,         "test130_basev_uint3"},
                      {test131_basek_uint3,         "test131_basek_uint3"},
                      {test132_unaryv_uint3,        "test132_unaryv_uint3"},
                      {test132_unaryk_uint3,        "test132_unaryk_uint3"}, 
                      {test133_cmpv_uint3,          "test133_cmpv_uint3"}, 
                      {test134_shuffle_uint3,       "test134_shuffle_uint3"},
                      {test135_exsplat_uint3,       "test135_exsplat_uint3"},
                      {test137_funcv_uint3,         "test137_funcv_uint3"},

                      {test138_logicv_uint3,        "test138_logicv_uint3"},
                      {test139_cstcnv_uint3,        "test139_cstcnv_uint3"},

                      {test140_other_uint3,        "test140_other_uint3"},



                      {test140_basev_int3,         "test140_basev_int3"},
                      {test141_basek_int3,         "test141_basek_int3"},
                      {test142_unaryv_int3,        "test142_unaryv_int3"},
                      {test142_unaryk_int3,        "test142_unaryk_int3"}, 
                      {test143_cmpv_int3,          "test143_cmpv_int3"}, 
                      {test144_shuffle_int3,       "test144_shuffle_int3"},
                      {test145_exsplat_int3,       "test145_exsplat_int3"},
                      {test147_funcv_int3,         "test147_funcv_int3"},

                      {test148_logicv_int3,        "test148_logicv_int3"},
                      {test149_cstcnv_int3,        "test149_cstcnv_int3"},

                      {test150_other_int3,        "test150_other_int3"},



                      {test150_basev_float3,         "test150_basev_float3"},
                      {test151_basek_float3,         "test151_basek_float3"},
                      {test152_unaryv_float3,        "test152_unaryv_float3"},
                      {test152_unaryk_float3,        "test152_unaryk_float3"}, 
                      {test153_cmpv_float3,          "test153_cmpv_float3"}, 
                      {test154_shuffle_float3,       "test154_shuffle_float3"},
                      {test155_exsplat_float3,       "test155_exsplat_float3"},
                      {test157_funcv_float3,         "test157_funcv_float3"},

                      {test158_funcfv_float3,        "test158_funcfv_float3"},
                      {test159_cstcnv_float3,        "test159_cstcnv_float3"},

                      {test160_other_float3,        "test160_other_float3"},



                      {test160_basev_uint2,         "test160_basev_uint2"},
                      {test161_basek_uint2,         "test161_basek_uint2"},
                      {test162_unaryv_uint2,        "test162_unaryv_uint2"},
                      {test162_unaryk_uint2,        "test162_unaryk_uint2"}, 
                      {test163_cmpv_uint2,          "test163_cmpv_uint2"}, 
                      {test164_shuffle_uint2,       "test164_shuffle_uint2"},
                      {test165_exsplat_uint2,       "test165_exsplat_uint2"},
                      {test167_funcv_uint2,         "test167_funcv_uint2"},

                      {test168_logicv_uint2,        "test168_logicv_uint2"},
                      {test169_cstcnv_uint2,        "test169_cstcnv_uint2"},

                      {test170_other_uint2,        "test170_other_uint2"},



                      {test170_basev_int2,         "test170_basev_int2"},
                      {test171_basek_int2,         "test171_basek_int2"},
                      {test172_unaryv_int2,        "test172_unaryv_int2"},
                      {test172_unaryk_int2,        "test172_unaryk_int2"}, 
                      {test173_cmpv_int2,          "test173_cmpv_int2"}, 
                      {test174_shuffle_int2,       "test174_shuffle_int2"},
                      {test175_exsplat_int2,       "test175_exsplat_int2"},
                      {test177_funcv_int2,         "test177_funcv_int2"},

                      {test178_logicv_int2,        "test178_logicv_int2"},
                      {test179_cstcnv_int2,        "test179_cstcnv_int2"},

                      {test180_other_int2,        "test180_other_int2"},



                      {test180_basev_float2,         "test180_basev_float2"},
                      {test181_basek_float2,         "test181_basek_float2"},
                      {test182_unaryv_float2,        "test182_unaryv_float2"},
                      {test182_unaryk_float2,        "test182_unaryk_float2"}, 
                      {test183_cmpv_float2,          "test183_cmpv_float2"}, 
                      {test184_shuffle_float2,       "test184_shuffle_float2"},
                      {test185_exsplat_float2,       "test185_exsplat_float2"},
                      {test187_funcv_float2,         "test187_funcv_float2"},

                      {test188_funcfv_float2,        "test188_funcfv_float2"},
                      {test189_cstcnv_float2,        "test189_cstcnv_float2"},

                      {test190_other_float2,        "test190_other_float2"},


                      };
  
  const auto arraySize = sizeof(tests)/sizeof(TestRun);
  
  for(int i=0;i<int(arraySize);i++)
  {
    const bool res = tests[i].pTest();
    std::cout << "test " << std::setfill('0') << std::setw(3) << i << " " << tests[i].pTestName << "\t";
    if(res)
      std::cout << "PASSED!";
    else 
      std::cout << "FAILED!" << "\t(!!!)";
    std::cout << std::endl;
    std::cout.flush();
  }
  
  return 0;
}
