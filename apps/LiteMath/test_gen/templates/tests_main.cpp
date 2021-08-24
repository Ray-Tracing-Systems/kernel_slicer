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
                      {% for Tests in AllTests %}
                      {% for Test  in Tests.Tests %}
                      {test{{Test.Number}}_basev_{{Test.Type}},         "test{{Test.Number}}_basev_{{Test.Type}}"},
                      {test{{Test.Number+1}}_basek_{{Test.Type}},         "test{{Test.Number+1}}_basek_{{Test.Type}}"},
                      {test{{Test.Number+2}}_unaryv_{{Test.Type}},        "test{{Test.Number+2}}_unaryv_{{Test.Type}}"},
                      {test{{Test.Number+2}}_unaryk_{{Test.Type}},        "test{{Test.Number+2}}_unaryk_{{Test.Type}}"}, 
                      {test{{Test.Number+3}}_cmpv_{{Test.Type}},          "test{{Test.Number+3}}_cmpv_{{Test.Type}}"}, 
                      {test{{Test.Number+4}}_shuffle_{{Test.Type}},       "test{{Test.Number+4}}_shuffle_{{Test.Type}}"},
                      {test{{Test.Number+5}}_exsplat_{{Test.Type}},       "test{{Test.Number+5}}_exsplat_{{Test.Type}}"},
                      {test{{Test.Number+7}}_funcv_{{Test.Type}},         "test{{Test.Number+7}}_funcv_{{Test.Type}}"},
                      {% if Test.IsFloat %}
                      {test{{Test.Number+8}}_funcfv_{{Test.Type}},        "test{{Test.Number+8}}_funcfv_{{Test.Type}}"},
                      {test{{Test.Number+9}}_cstcnv_{{Test.Type}},        "test{{Test.Number+9}}_cstcnv_{{Test.Type}}"},
                      {% else %}
                      {test{{Test.Number+8}}_logicv_{{Test.Type}},        "test{{Test.Number+8}}_logicv_{{Test.Type}}"},
                      {test{{Test.Number+9}}_cstcnv_{{Test.Type}},        "test{{Test.Number+9}}_cstcnv_{{Test.Type}}"},
                      {% endif %}
                      {test{{Test.Number+10}}_other_{{Test.Type}},        "test{{Test.Number+10}}_other_{{Test.Type}}"},
                      {% endfor %}
                      {% endfor %}
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