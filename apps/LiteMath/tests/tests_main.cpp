#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
#include "tests/tests.h"

bool f4_test001_arith()
{
  const float4 Cx1(1.0f, 2.0f, 3.0f, 4.0f);
  const float4 Cx2(5.0f, 5.0f, 6.0f, 7.0f);

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  float result1[4];
  float result2[4];
  float result3[4];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  // check 
  //
  bool passed = true;

  float expr1[4], expr2[4], expr3[4];

  for(int i=0;i<4;i++)
  {
    expr1[i] = Cx1[i] - Cx2[i];
    expr2[i] = (Cx1[i] + Cx2[i])*Cx1[i];
    expr3[i] = (Cx2[i] - Cx1[i])/Cx1[i];

    if(fabs(result1[i] - expr1[i]) > 1e-6f || fabs(result2[i] - expr2[i]) > 1e-6f || fabs(result3[i] - expr3[i]) > 1e-5f) // 1e-2f
      passed = false;
  }

  if(!passed)
  {
    std::cout << "exp1_res:" << result1[0] << " " << result1[1] << " " << result1[2] << " " << result1[3] << std::endl;
    std::cout << "exp1_ref:" << expr1  [0] << " " << expr1  [1] << " " << expr1  [2] << " " << expr1[3] << std::endl;
    std::cout << std::endl;

    std::cout << "exp2_res:" << result2[0] << " " << result2[1] << " " << result2[2] << " " << result2[3] << std::endl;
    std::cout << "exp2_ref:" << expr2  [0] << " " << expr2  [1] << " " << expr2  [2] << " " << expr2[3] << std::endl;
    std::cout << std::endl;

    std::cout << "exp3_res:" << result3[0] << " " << result3[1] << " " << result3[2] << " " << result3[3] << std::endl;
    std::cout << "exp3_ref:" << expr3  [0] << " " << expr3  [1] << " " << expr3  [2] << " " << expr3[3] << std::endl;
    std::cout << std::endl;
  }
  
  return passed;
}

using TestFuncType = bool (*)();

struct TestRun
{
  TestFuncType pTest;
  const char*  pTestName;
};

int main(int argc, const char** argv)
{
 
  test000_scalar_functions_f();
  
  test101_base_arith_float4();

  //TestRun tests[] = { f4_test001_arith, "f4_test001_arith",
  //                    };
  //
  //const auto arraySize = sizeof(tests)/sizeof(TestRun);
  //
  //for(int i=0;i<int(arraySize);i++)
  //{
  //  const bool res = tests[i].pTest();
  //  std::cout << "test\t" << std::setfill('0') << std::setw(3) << i+1 << "\t" << tests[i].pTestName << "\t";
  //  if(res)
  //    std::cout << "PASSED!";
  //  else 
  //    std::cout << "FAILED!\tFAILED!";
  //  std::cout << std::endl;
  //  std::cout.flush();
  //}
  
  return 0;
}
