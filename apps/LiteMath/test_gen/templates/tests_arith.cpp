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
  
  {{Test.TypeS}} expr1[{{Test.VecLen}}], expr2[{{Test.VecLen}}], expr3[{{Test.VecLen}}];
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

  CVEX_ALIGNED(16) {{Test.TypeS}} result1[4]; 
  CVEX_ALIGNED(16) {{Test.TypeS}} result2[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result3[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result4[4];

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

bool test{{Test.Number+2}}_unaryv_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.Type}} Cx2({% for Val in Test.ValuesB %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  {{Test.TypeS}} result1[{{Test.VecLen}}];
  {{Test.TypeS}} result2[{{Test.VecLen}}];
  {{Test.TypeS}} result3[{{Test.VecLen}}];
  {{Test.TypeS}} result4[{{Test.VecLen}}];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  {{Test.TypeS}} expr1[{{Test.VecLen}}], expr2[{{Test.VecLen}}], expr3[{{Test.VecLen}}], expr4[{{Test.VecLen}}];
  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    expr1[i] = Cx1[i] + Cx2[i];
    expr2[i] = Cx1[i] - Cx2[i];
    expr3[i] = Cx1[i] * Cx2[i];
    expr4[i] = Cx1[i] / Cx2[i];
    
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
    PrintRR("exp4_res", "exp4_res", result4, expr4, {{Test.VecLen}});
  }
  
  return passed;
}

bool test{{Test.Number+2}}_unaryk_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.TypeS}} Cx2 = {% for Val in Test.ValuesB %}{% if loop.index == 0 %}{{Test.TypeS}}({{Val}}){% endif %}{% endfor %};

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  {{Test.TypeS}} result1[{{Test.VecLen}}];
  {{Test.TypeS}} result2[{{Test.VecLen}}];
  {{Test.TypeS}} result3[{{Test.VecLen}}];
  {{Test.TypeS}} result4[{{Test.VecLen}}];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  {{Test.TypeS}} expr1[{{Test.VecLen}}], expr2[{{Test.VecLen}}], expr3[{{Test.VecLen}}], expr4[{{Test.VecLen}}];
  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    expr1[i] = Cx1[i] + Cx2;
    expr2[i] = Cx1[i] - Cx2;
    expr3[i] = Cx1[i] * Cx2;
    expr4[i] = Cx1[i] / Cx2;
    
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
    PrintRR("exp4_res", "exp4_res", result4, expr4, {{Test.VecLen}});
  }
  
  return passed;
}

bool test{{Test.Number+3}}_cmpv_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.Type}} Cx2({% for Val in Test.ValuesB %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});

  auto Cx3 = (Cx1 < Cx2 );
  auto Cx4 = (Cx1 > Cx2 );
  auto Cx5 = (Cx1 <= Cx2);
  auto Cx6 = (Cx1 <= Cx2);
  auto Cx7 = (Cx1 == Cx2);
  auto Cx8 = (Cx1 != Cx2);

  uint32_t result1[{{Test.VecLen}}];
  uint32_t result2[{{Test.VecLen}}];
  uint32_t result3[{{Test.VecLen}}];
  uint32_t result4[{{Test.VecLen}}];
  uint32_t result5[{{Test.VecLen}}];
  uint32_t result6[{{Test.VecLen}}];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  store_u(result5, Cx7);
  store_u(result6, Cx8);
  
  uint32_t expr1[{{Test.VecLen}}], expr2[{{Test.VecLen}}], expr3[{{Test.VecLen}}], expr4[{{Test.VecLen}}], expr5[{{Test.VecLen}}], expr6[{{Test.VecLen}}];
  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    expr1[i] = Cx1[i] <  Cx2[i] ? 0xFFFFFFFF : 0;
    expr2[i] = Cx1[i] >  Cx2[i] ? 0xFFFFFFFF : 0;
    expr3[i] = Cx1[i] <= Cx2[i] ? 0xFFFFFFFF : 0;
    expr4[i] = Cx1[i] >= Cx2[i] ? 0xFFFFFFFF : 0;
    expr5[i] = Cx1[i] == Cx2[i] ? 0xFFFFFFFF : 0;
    expr6[i] = Cx1[i] != Cx2[i] ? 0xFFFFFFFF : 0;
    
    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i] || result4[i] != expr4[i] || result5[i] != expr5[i] || result6[i] != expr6[i]) 
      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, {{Test.VecLen}});
    PrintRR("exp2_res", "exp2_res", result2, expr2, {{Test.VecLen}}); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, {{Test.VecLen}});
    PrintRR("exp4_res", "exp4_res", result4, expr4, {{Test.VecLen}});
    PrintRR("exp5_res", "exp5_res", result5, expr5, {{Test.VecLen}});
    PrintRR("exp6_res", "exp6_res", result6, expr6, {{Test.VecLen}});
  }
  
  return passed;
}

bool test{{Test.Number+4}}_blendv_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.Type}} Cx2({% for Val in Test.ValuesB %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});

  const auto mask1 = (Cx1 < Cx2);
  const auto mask2 = (Cx1 >= Cx2);
  
  const auto Cx3 = blend(Cx1, Cx2, mask1);
  const auto Cx4 = blend(Cx1, Cx2, mask2);
  
  {{Test.TypeS}} result1[{{Test.VecLen}}];
  {{Test.TypeS}} result2[{{Test.VecLen}}];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  
  {{Test.TypeS}} expr1[{{Test.VecLen}}], expr2[{{Test.VecLen}}], expr3[{{Test.VecLen}}], expr4[{{Test.VecLen}}], expr5[{{Test.VecLen}}], expr6[{{Test.VecLen}}];
  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    expr1[i] = Cx1[i] <  Cx2[i] ? Cx1[i] : Cx2[i];
    expr2[i] = Cx1[i] >= Cx2[i] ? Cx1[i] : Cx2[i];
  
    if(result1[i] != expr1[i] || result2[i] != expr2[i]) 
      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, {{Test.VecLen}});
    PrintRR("exp2_res", "exp2_res", result2, expr2, {{Test.VecLen}}); 
  }
  
  return passed;
}

{% if Test.IsFloat %}

//bool test{{Test.Number+3}}_funcv_{{Test.Type}}()
//{
//  
//}

{% else %}

{% endif %}


{% if 0 %}
bool test{{Test.Number+10}}_shuffle4_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});

  const {{Test.Type}} Cr1 = shuffle_zyxw(Cx1);
  const {{Test.Type}} Cr2 = shuffle_yzxw(Cx1);
  const {{Test.Type}} Cr3 = shuffle_zxyw(Cx1);
  const {{Test.Type}} Cr4 = shuffle_xyxy(Cx1);
  const {{Test.Type}} Cr5 = shuffle_zwzw(Cx1);

  CVEX_ALIGNED(16) {{Test.TypeS}} result1[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result2[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result3[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result4[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result5[4];
  store(result1, Cr1);
  store(result2, Cr2);
  store(result3, Cr3);
  store(result4, Cr4);
  store(result5, Cr5);

  const bool b1 = (result1[0] == {{Test.TypeS}}(3) ) && (result1[1] == {{Test.TypeS}}(2) ) && (result1[2] == {{Test.TypeS}}(1) ) && (result1[3] == {{Test.TypeS}}(4) );
  const bool b2 = (result2[0] == {{Test.TypeS}}(2) ) && (result2[1] == {{Test.TypeS}}(3) ) && (result2[2] == {{Test.TypeS}}(1) ) && (result2[3] == {{Test.TypeS}}(4) );
  const bool b3 = (result3[0] == {{Test.TypeS}}(3) ) && (result3[1] == {{Test.TypeS}}(1) ) && (result3[2] == {{Test.TypeS}}(2) ) && (result3[3] == {{Test.TypeS}}(4) );
  const bool b4 = (result4[0] == {{Test.TypeS}}(1) ) && (result4[1] == {{Test.TypeS}}(2) ) && (result4[2] == {{Test.TypeS}}(1) ) && (result4[3] == {{Test.TypeS}}(2) );
  const bool b5 = (result5[0] == {{Test.TypeS}}(3) ) && (result5[1] == {{Test.TypeS}}(4) ) && (result5[2] == {{Test.TypeS}}(3) ) && (result5[3] == {{Test.TypeS}}(4) );
 
  const bool passed = (b1 && b2 && b3 && b4 && b5);

  if(!passed)
  {
    std::cout << result1[0] << " " << result1[1] << " " << result1[2] << " " << result1[3] << std::endl;
    std::cout << result2[0] << " " << result2[1] << " " << result2[2] << " " << result2[3] << std::endl;
    std::cout << result3[0] << " " << result3[1] << " " << result3[2] << " " << result3[3] << std::endl;
    std::cout << result4[0] << " " << result4[1] << " " << result4[2] << " " << result4[3] << std::endl;
    std::cout << result5[0] << " " << result5[1] << " " << result5[2] << " " << result5[3] << std::endl;
  }

  return passed;
}
{% endif %}

## endfor