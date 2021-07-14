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

## for Test in Tests
bool test{{Test.Number}}_basev_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.Type}} Cx2({% for Val in Test.ValuesB %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  {{Test.TypeS}} result1[{{Test.VecLen}}];
  {{Test.TypeS}} result2[{{Test.VecLen}}];
  {{Test.TypeS}} result3[{{Test.VecLen}}];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  float expr1[{{Test.VecLen}}], expr2[{{Test.VecLen}}], expr3[{{Test.VecLen}}];
  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    expr1[i] = Cx1[i] - Cx2[i];
    expr2[i] = (Cx1[i] + Cx2[i])*Cx1[i];
    expr3[i] = (Cx2[i] - Cx1[i])/Cx1[i];
    
    {% if Test.IsFloat %}
    if(fabs(result1[i] - expr1[i]) > 1e-6f || fabs(result2[i] - expr2[i]) > 1e-6f || fabs(result3[i] - expr3[i]) > 1e-6f) 
    {% else %}
    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i]) 
    {% endif %}
      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, {{Test.VecLen}});
    PrintRR("exp2_res", "exp2_res", result2, expr2, {{Test.VecLen}}); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, {{Test.VecLen}});
  }
  
  return passed;
}

bool test{{Test.Number+1}}_basek_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.TypeS}} Cx2 = {% for Val in Test.ValuesB %}{% if loop.index == 0 %}{{Test.TypeS}}({{Val}}){% endif %}{% endfor %};

  const {{Test.Type}} Cx3 = Cx2*(Cx2 + Cx1) - {{Test.TypeS}}(2);
  const {{Test.Type}} Cx4 = {{Test.TypeS}}(1) + (Cx1 + Cx2)*Cx2;
  
  const {{Test.Type}} Cx5 = {{Test.TypeS}}(3) - Cx2/(Cx2 - Cx1);
  const {{Test.Type}} Cx6 = (Cx2 + Cx1)/Cx2 + {{Test.TypeS}}(5)/Cx1;

  CVEX_ALIGNED(16) {{Test.TypeS}} result1[{{Test.VecLen}}];
  CVEX_ALIGNED(16) {{Test.TypeS}} result2[{{Test.VecLen}}];
  CVEX_ALIGNED(16) {{Test.TypeS}} result3[{{Test.VecLen}}];
  CVEX_ALIGNED(16) {{Test.TypeS}} result4[{{Test.VecLen}}];

  store(result1, Cx3);
  store(result2, Cx4);
  store(result3, Cx5);
  store(result4, Cx6);
  
  bool passed = true;
  for(int i=0;i<4;i++)
  {
    const {{Test.TypeS}} expr1 = Cx2*(Cx2 - Cx1[i]) - {{Test.TypeS}}(2);
    const {{Test.TypeS}} expr2 = {{Test.TypeS}}(1) + (Cx1[i] + Cx2)*Cx2;
    const {{Test.TypeS}} expr3 = {{Test.TypeS}}(3) - Cx2/(Cx2 - Cx1[i]);
    const {{Test.TypeS}} expr4 = (Cx2 + Cx1[i])/Cx2 + {{Test.TypeS}}(5)/Cx1[i];
    
    {% if Test.IsFloat %}
    if(fabs(result1[i] - expr1) > 1e-6f || fabs(result2[i] - expr2) > 1e-6f || fabs(result3[i] - expr3) > 1e-6f || fabs(result4[i] - expr4) > 1e-6f )
    {% else %}
    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i] || result4[i] != expr4[i]) 
    {% endif %}
      passed = false;
  }

  return passed;
}

## endfor