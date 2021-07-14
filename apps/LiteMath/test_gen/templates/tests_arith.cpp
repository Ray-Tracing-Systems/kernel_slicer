#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
using namespace LiteMath;

template<typename T>
static void PrintRR(const char* name1, const char* name2, T res[], T ref[], int size)
{
  std::cout << name1 << ": ";
  for(int i=0;i<4;i++)
    std::cout << res[i] << " ";
  std::cout << std::endl;
  std::cout << name2 << ": "; 
   for(int i=0;i<4;i++)
    std::cout << ref[i] << " ";
  std::cout << std::endl;
  std::cout << std::endl;
}

bool test{{Test.Number}}_{{Test.Name}}_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({{Test.ValuesA}});
  const {{Test.Type}} Cx2({{Test.ValuesB}});

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  {{Test.TypeS}} result1[{{Test.VecLen}}];
  {{Test.TypeS}} result2[{{Test.VecLen}}];
  {{Test.TypeS}} result3[{{Test.VecLen}}];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  // check 
  //
  bool passed = true;

  float expr1[{{Test.VecLen}}], expr2[{{Test.VecLen}}], expr3[{{Test.VecLen}}];

  for(int i=0;i<{{Test.VecLen}};i++)
  {
    expr1[i] = Cx1[i] - Cx2[i];
    expr2[i] = (Cx1[i] + Cx2[i])*Cx1[i];
    expr3[i] = (Cx2[i] - Cx1[i])/Cx1[i];
    
    {% if Test.IsFloat %}
    if(fabs(result1[i] - expr1[i]) > 1e-6f || fabs(result2[i] - expr2[i]) > 1e-6f || fabs(result3[i] - expr3[i]) > 1e-6f) 
      passed = false;
    {% else %}
    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i]) 
      passed = false;
    {% endif %}
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, {{Test.VecLen}});
    PrintRR("exp2_res", "exp2_res", result2, expr2, {{Test.VecLen}}); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, {{Test.VecLen}});
  }
  
  return passed;
}