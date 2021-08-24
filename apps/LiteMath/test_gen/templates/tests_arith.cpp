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
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    const {{Test.TypeS}} expr1 = Cx2*(Cx2 + Cx1[i]) - {{Test.TypeS}}(2);
    const {{Test.TypeS}} expr2 = {{Test.TypeS}}(1) + (Cx1[i] + Cx2)*Cx2;
    const {{Test.TypeS}} expr3 = {{Test.TypeS}}(3) - Cx2/(Cx2 - Cx1[i]);
    const {{Test.TypeS}} expr4 = (Cx2 + Cx1[i])/Cx2 + {{Test.TypeS}}(5)/Cx1[i];
    
    {% if Test.IsFloat %}
    if(fabs(result1[i] - expr1) > 1e-6f || fabs(result2[i] - expr2) > 1e-6f || fabs(result3[i] - expr3) > 1e-6f || fabs(result4[i] - expr4) > 1e-6f )
    {% else %}
    if(result1[i] != expr1 || result2[i] != expr2 || result3[i] != expr3 || result4[i] != expr4) 
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
  auto Cx6 = (Cx1 >= Cx2);
  auto Cx7 = (Cx1 == Cx2);
  auto Cx8 = (Cx1 != Cx2);

  const auto Cx9  = blend(Cx1, Cx2, Cx3);
  const auto Cx10 = blend(Cx1, Cx2, Cx6);

  uint32_t result1[{{Test.VecLen}}];
  uint32_t result2[{{Test.VecLen}}];
  uint32_t result3[{{Test.VecLen}}];
  uint32_t result4[{{Test.VecLen}}];
  uint32_t result5[{{Test.VecLen}}];
  uint32_t result6[{{Test.VecLen}}];
  {{Test.TypeS}} result7[{{Test.VecLen}}];
  {{Test.TypeS}} result8[{{Test.VecLen}}];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  store_u(result5, Cx7);
  store_u(result6, Cx8);
  store_u(result7, Cx9);
  store_u(result8, Cx10);
  
  uint32_t expr1[{{Test.VecLen}}], expr2[{{Test.VecLen}}], expr3[{{Test.VecLen}}], expr4[{{Test.VecLen}}], expr5[{{Test.VecLen}}], expr6[{{Test.VecLen}}];
  {{Test.TypeS}} expr7[{{Test.VecLen}}],  expr8[{{Test.VecLen}}];
  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    expr1[i] = Cx1[i] <  Cx2[i] ? 0xFFFFFFFF : 0;
    expr2[i] = Cx1[i] >  Cx2[i] ? 0xFFFFFFFF : 0;
    expr3[i] = Cx1[i] <= Cx2[i] ? 0xFFFFFFFF : 0;
    expr4[i] = Cx1[i] >= Cx2[i] ? 0xFFFFFFFF : 0;
    expr5[i] = Cx1[i] == Cx2[i] ? 0xFFFFFFFF : 0;
    expr6[i] = Cx1[i] != Cx2[i] ? 0xFFFFFFFF : 0;
    expr7[i] = Cx1[i] <  Cx2[i] ? Cx1[i] : Cx2[i];
    expr8[i] = Cx1[i] >= Cx2[i] ? Cx1[i] : Cx2[i];
    
    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i] || result4[i] != expr4[i] || 
       result5[i] != expr5[i] || result6[i] != expr6[i] || result7[i] != expr7[i] || result8[i] != expr8[i]) 
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
    PrintRR("exp7_res", "exp7_res", result7, expr7, {{Test.VecLen}});
    PrintRR("exp8_res", "exp8_res", result8, expr8, {{Test.VecLen}});
  }
  
  return passed;
}

bool test{{Test.Number+4}}_shuffle_{{Test.Type}}()
{ 
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});

  {% if Test.VecLen == 4 %}
  const {{Test.Type}} Cr1 = shuffle_zyxw(Cx1);
  const {{Test.Type}} Cr2 = shuffle_zxyw(Cx1);
  const {{Test.Type}} Cr3 = shuffle_yzxw(Cx1);
  const {{Test.Type}} Cr6 = shuffle_yxzw(Cx1);
  const {{Test.Type}} Cr7 = shuffle_xzyw(Cx1);
  
  const {{Test.Type}} Cr4 = shuffle_xyxy(Cx1);
  const {{Test.Type}} Cr5 = shuffle_zwzw(Cx1);

  CVEX_ALIGNED(16) {{Test.TypeS}} result1[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result2[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result3[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result4[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result5[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result6[4];
  CVEX_ALIGNED(16) {{Test.TypeS}} result7[4];

  store(result1, Cr1);
  store(result2, Cr2);
  store(result3, Cr3);
  store(result4, Cr4);
  store(result5, Cr5);
  store(result6, Cr6);
  store(result7, Cr7);

  const bool b1 = (result1[0] == Cx1[2]) && (result1[1] == Cx1[1]) && (result1[2] == Cx1[0]) && (result1[3] == Cx1[3]);
  const bool b2 = (result2[0] == Cx1[2]) && (result2[1] == Cx1[0]) && (result2[2] == Cx1[1]) && (result2[3] == Cx1[3]);
  const bool b3 = (result3[0] == Cx1[1]) && (result3[1] == Cx1[2]) && (result3[2] == Cx1[0]) && (result3[3] == Cx1[3]);
  const bool b6 = (result6[0] == Cx1[1]) && (result6[1] == Cx1[0]) && (result6[2] == Cx1[2]) && (result6[3] == Cx1[3]);
  const bool b7 = (result7[0] == Cx1[0]) && (result7[1] == Cx1[2]) && (result7[2] == Cx1[1]) && (result7[3] == Cx1[3]);
  
  const bool b4 = (result4[0] == Cx1[0]) && (result4[1] == Cx1[1]) && (result4[2] == Cx1[0]) && (result4[3] == Cx1[1]);
  const bool b5 = (result5[0] == Cx1[2]) && (result5[1] == Cx1[3]) && (result5[2] == Cx1[2]) && (result5[3] == Cx1[3]);
 
  const bool passed = (b1 && b2 && b3 && b4 && b5 && b6 && b7);

  if(!passed)
  {
    std::cout << result1[0] << " " << result1[1] << " " << result1[2] << " " << result1[3] << std::endl;
    std::cout << result2[0] << " " << result2[1] << " " << result2[2] << " " << result2[3] << std::endl;
    std::cout << result3[0] << " " << result3[1] << " " << result3[2] << " " << result3[3] << std::endl;
    std::cout << result4[0] << " " << result4[1] << " " << result4[2] << " " << result4[3] << std::endl;
    std::cout << result5[0] << " " << result5[1] << " " << result5[2] << " " << result5[3] << std::endl;
    std::cout << result6[0] << " " << result6[1] << " " << result6[2] << " " << result6[3] << std::endl;
    std::cout << result7[0] << " " << result7[1] << " " << result7[2] << " " << result7[3] << std::endl;
  }

  return passed;

  {% else if Test.VecLen == 3 %}
  const {{Test.Type}} Cr1 = shuffle_zyx(Cx1);
  const {{Test.Type}} Cr2 = shuffle_zxy(Cx1);
  const {{Test.Type}} Cr3 = shuffle_yzx(Cx1);
  const {{Test.Type}} Cr4 = shuffle_yxz(Cx1);
  const {{Test.Type}} Cr5 = shuffle_xzy(Cx1);

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

  const bool b1 = (result1[0] == Cx1[2]) && (result1[1] == Cx1[1]) && (result1[2] == Cx1[0]);
  const bool b2 = (result2[0] == Cx1[2]) && (result2[1] == Cx1[0]) && (result2[2] == Cx1[1]);
  const bool b3 = (result3[0] == Cx1[1]) && (result3[1] == Cx1[2]) && (result3[2] == Cx1[0]);
  const bool b4 = (result4[0] == Cx1[1]) && (result4[1] == Cx1[0]) && (result4[2] == Cx1[2]);
  const bool b5 = (result5[0] == Cx1[0]) && (result5[1] == Cx1[2]) && (result5[2] == Cx1[1]);
  
  const bool passed = (b1 && b2 && b3 && b4 && b5);
  if(!passed)
  {
    std::cout << result1[0] << " " << result1[1] << " " << result1[2] << std::endl;
    std::cout << result2[0] << " " << result2[1] << " " << result2[2] << std::endl;
    std::cout << result3[0] << " " << result3[1] << " " << result3[2] << std::endl;
    std::cout << result4[0] << " " << result4[1] << " " << result4[2] << std::endl;
    std::cout << result5[0] << " " << result5[1] << " " << result5[2] << std::endl;
  }
  return passed;
  {% else %}
  return true;
  {% endif %}
}

bool test{{Test.Number+5}}_exsplat_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  {% for i in range(Test.VecLen) %}
  const {{Test.Type}} Cr{{loop.index}} = splat_{{loop.index}}(Cx1);{% endfor %}
  {% for i in range(Test.VecLen) %}
  const {{Test.TypeS}} s{{loop.index}} = extract_{{loop.index}}(Cx1);{% endfor %}
  {% for i in range(Test.VecLen) %}
  {{Test.TypeS}} result{{loop.index}}[{{Test.VecLen}}];{% endfor %}
  {% for i in range(Test.VecLen) %}
  store_u(result{{loop.index}}, Cr{{loop.index}});{% endfor %}
  
  bool passed = true;
  for (int i = 0; i<{{Test.VecLen}}; i++)
  {
    {% for i in range(Test.VecLen) %}
    if((result{{loop.index}}[i] != Cx1[{{loop.index}}]))
      passed = false;{% endfor %}
  }
  {% for i in range(Test.VecLen) %}
  if(s{{loop.index}} != Cx1[{{loop.index}}])
    passed = false;{% endfor %}
  return passed;
}

bool test{{Test.Number+7}}_funcv_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.Type}} Cx2({% for Val in Test.ValuesB %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});

  {% if Test.IsSigned %}  
  auto Cx3 = sign(Cx1);
  auto Cx4 = abs(Cx1);
  {% endif %}
  auto Cx5 = clamp(Cx1, {{Test.TypeS}}(2), {{Test.TypeS}}(3) );
  auto Cx6 = min(Cx1, Cx2);
  auto Cx7 = max(Cx1, Cx2);

  {{Test.TypeS}} Cm = hmin(Cx1);
  {{Test.TypeS}} CM = hmax(Cx1);
  {{Test.TypeS}} horMinRef = Cx1[0];
  {{Test.TypeS}} horMaxRef = Cx1[0];
  
  {% if Test.VecLen == 4 %}
  {{Test.TypeS}} Cm3 = hmin3(Cx1);
  {{Test.TypeS}} CM3 = hmax3(Cx1);
  {{Test.TypeS}} horMinRef3 = Cx1[0];
  {{Test.TypeS}} horMaxRef3 = Cx1[0];
  {% endif %}
  
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    horMinRef = std::min(horMinRef, Cx1[i]);
    horMaxRef = std::max(horMaxRef, Cx1[i]);
    {% if Test.VecLen == 4 %}
    if(i<3)
    {
      horMinRef3 = std::min(horMinRef3, Cx1[i]);
      horMaxRef3 = std::max(horMaxRef3, Cx1[i]);
    }
    {% endif %}
  }

  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    {% if Test.IsSigned %}  
    if(Cx3[i] != sign(Cx1[i]))
      passed = false;
    if(Cx4[i] != abs(Cx1[i]))
      passed = false;
    {% endif %}
    if(Cx5[i] != clamp(Cx1[i], {{Test.TypeS}}(2), {{Test.TypeS}}(3) ))
      passed = false;
    if(Cx6[i] != min(Cx1[i], Cx2[i]))
      passed = false;
    if(Cx7[i] != max(Cx1[i], Cx2[i]))
      passed = false;
  }

  if(horMinRef != Cm)
    passed = false;
  if(horMaxRef != CM)
    passed = false;
  {% if Test.VecLen == 4 %}
  if(horMinRef3 != Cm3)
    passed = false;
  if(horMaxRef3 != CM3)
    passed = false;
  {% endif %}

  return passed;
}


{% if Test.IsFloat %}

bool test{{Test.Number+8}}_funcfv_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.Type}} Cx2({% for Val in Test.ValuesB %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  
  auto Cx3 = mod(Cx1, Cx2);
  auto Cx4 = fract(Cx1);
  auto Cx5 = ceil(Cx1);
  auto Cx6 = floor(Cx1);
  auto Cx7 = sign(Cx1);
  auto Cx8 = abs(Cx1);

  auto Cx9  = clamp(Cx1, -2.0f, 2.0f);
  auto Cx10 = min(Cx1, Cx2);
  auto Cx11 = max(Cx1, Cx2);

  auto Cx12 = mix (Cx1, Cx2, 0.5f);
  auto Cx13 = lerp(Cx1, Cx2, 0.5f);
  
  auto Cx14 = sqrt(Cx8); 
  auto Cx15 = inversesqrt(Cx8);

  auto Cx18 = rcp(Cx1);

  {{Test.TypeS}} ref[19][{{Test.VecLen}}];
  {{Test.TypeS}} res[19][{{Test.VecLen}}];
  memset(ref, 0, 19*sizeof({{Test.TypeS}})*{{Test.VecLen}});
  memset(res, 0, 19*sizeof({{Test.TypeS}})*{{Test.VecLen}});

  store_u(res[3],  Cx3);
  store_u(res[4],  Cx4);
  store_u(res[5],  Cx5);
  store_u(res[6],  Cx6);
  store_u(res[7],  Cx7);
  store_u(res[8],  Cx8);
  store_u(res[9],  Cx9);
  store_u(res[10], Cx10);
  store_u(res[11], Cx11);
  store_u(res[12], Cx12);
  store_u(res[13], Cx13);
  store_u(res[14], Cx14);
  store_u(res[15], Cx15);
  store_u(res[18], Cx18);

  for(int i=0;i<{{Test.VecLen}};i++)
  {
    ref[3][i] = mod(Cx1[i], Cx2[i]);
    ref[4][i] = fract(Cx1[i]);
    ref[5][i] = ceil(Cx1[i]);
    ref[6][i] = floor(Cx1[i]);
    ref[7][i] = sign(Cx1[i]);
    ref[8][i] = abs(Cx1[i]);

    ref[9][i]  = clamp(Cx1[i], {{Test.TypeS}}(-2), {{Test.TypeS}}(2) );
    ref[10][i] = min(Cx1[i], Cx2[i]);
    ref[11][i] = max(Cx1[i], Cx2[i]);

    ref[12][i] = mix (Cx1[i], Cx2[i], 0.5f);
    ref[13][i] = lerp(Cx1[i], Cx2[i], 0.5f);

    ref[14][i] = sqrt(Cx8[i]);
    ref[15][i] = inversesqrt(Cx8[i]);
    ref[18][i] = rcp(Cx1[i]);
  }
  
  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    for(int j=3;j<=18;j++)
    {
      if(abs( res[j][i]-ref[j][i]) > 1e-6f)
      {
        if(j == 15 || j == 18)
        {
          if(abs(res[j][i]-ref[j][i]) > 5e-4f)
          {
            passed = false;
            break;
          }
        }
        else
        {
          passed = false;
          break;
        }
      }
    }
  }

  return passed;
}

bool test{{Test.Number+9}}_cstcnv_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  
  const int{{Test.VecLen}}  Cr1 = to_int32(Cx1);
  const uint{{Test.VecLen}} Cr2 = to_uint32(Cx1);
  const int{{Test.VecLen}}  Cr3 = as_int32(Cx1);
  const uint{{Test.VecLen}} Cr4 = as_uint32(Cx1);

  int          result1[{{Test.VecLen}}];
  unsigned int result2[{{Test.VecLen}}];
  int          result3[{{Test.VecLen}}];
  unsigned int result4[{{Test.VecLen}}];

  store_u(result1, Cr1);
  store_u(result2, Cr2);
  store_u(result3, Cr3);
  store_u(result4, Cr4);

  int          ref1[{{Test.VecLen}}];
  unsigned int ref2[{{Test.VecLen}}];
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    ref1[i] = int(Cx1[i]);
    ref2[i] = (unsigned int)(Cx1[i]); 
  }

  int          ref3[{{Test.VecLen}}];
  unsigned int ref4[{{Test.VecLen}}];

  memcpy(ref3, &Cr3, sizeof(int)*{{Test.VecLen}});
  memcpy(ref4, &Cr4, sizeof(uint)*{{Test.VecLen}});
  
  bool passed = true;
  for (int i=0; i<{{Test.VecLen}}; i++)
  {
    if (result1[i] != ref1[i] || result2[i] != ref2[i] || result3[i] != ref3[i] || result4[i] != ref4[i])
    {
      passed = false;
      break;
    }
  }
  return passed;
}


{% else %}

bool test{{Test.Number+8}}_logicv_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.Type}} Cx2({% for Val in Test.ValuesB %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  const {{Test.Type}} Cx3({% for Val in Test.ValuesC %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});

  const auto Cr0 = (Cx1 & (~Cx3)) | Cx2;
  const auto Cr1 = (Cx2 & Cx3)    | Cx1;
  const auto Cr2 = (Cx1 << 8); 
  const auto Cr3 = (Cx3 >> 9); 
  const auto Cr4 = (Cx1 << 8) | (Cx2 >> 17); 
  const auto Cr5 = (Cx3 << 9) | (Cx3 >> 4); 

  {{Test.TypeS}} ref[6][{{Test.VecLen}}];
  {{Test.TypeS}} res[6][{{Test.VecLen}}];
  store_u(res[0],  Cr0);
  store_u(res[1],  Cr1);
  store_u(res[2],  Cr2);
  store_u(res[3],  Cr3);
  store_u(res[4],  Cr4);
  store_u(res[5],  Cr5);
  
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    ref[0][i] = (Cx1[i] & (~Cx3[i])) | Cx2[i];
    ref[1][i] = (Cx2[i] & Cx3[i])    | Cx1[i];
    ref[2][i] = (Cx1[i] << 8); 
    ref[3][i] = (Cx3[i] >> 9); 
    ref[4][i] = (Cx1[i] << 8) | (Cx2[i] >> 17); 
    ref[5][i] = (Cx3[i] << 9) | (Cx3[i] >> 4); 
  }
  
  bool passed = true;
  for(int i=0;i<{{Test.VecLen}};i++)
  {
    for(int j=0;j<=5;j++)
      if(res[j][i] != ref[j][i])
        passed = false;
  }
  return passed;
}

bool test{{Test.Number+9}}_cstcnv_{{Test.Type}}()
{
  const {{Test.Type}} Cx1({% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %});
  
  const float{{Test.VecLen}} Cr1 = to_float32(Cx1);
  const float{{Test.VecLen}} Cr2 = as_float32(Cx1);

  float result1[4];
  float result2[4];
  store_u(result1, Cr1);
  store_u(result2, Cr2);

  float ref1[{{Test.VecLen}}];
  for(int i=0;i<{{Test.VecLen}};i++)
    ref1[i] = float(Cx1[i]);
  float ref2[{{Test.VecLen}}];
  memcpy(ref2, &Cx1, sizeof(float)*{{Test.VecLen}});
  
  bool passed = true;
  for (int i=0; i<{{Test.VecLen}}; i++)
  {
    if (result1[i] != ref1[i] || memcmp(result2, ref2, sizeof({{Test.TypeS}})*{{Test.VecLen}}) != 0)
    {
      passed = false;
      break;
    }
  }
  return passed;
}

{% endif %}

bool test{{Test.Number+10}}_other_{{Test.Type}}() // dummy test
{
  const {{Test.TypeS}} CxData[{{Test.VecLen}}] = { {% for Val in Test.ValuesA %} {{Test.TypeS}}({{Val}}){% if loop.index1 != Test.VecLen %}, {% endif %} {% endfor %}};
  const {{Test.Type}}  Cx1(CxData);
  const {{Test.Type}}  Cx2({{Test.Type}}(1));
 
  const {{Test.Type}}  Cx3 = Cx1 + Cx2;
  {{Test.TypeS}} result1[{{Test.VecLen}}];
  {{Test.TypeS}} result2[{{Test.VecLen}}];
  {{Test.TypeS}} result3[{{Test.VecLen}}];
  store_u(result1, Cx1);
  store_u(result2, Cx2);
  store_u(result3, Cx3);

  bool passed = true;
  for (int i=0; i<{{Test.VecLen}}; i++)
  {
    {% if Test.IsFloat %}
    if (fabs(result1[i] + {{Test.TypeS}}(1) - result3[i]) > 1e-10f || fabs(result2[i] - {{Test.TypeS}}(1) > 1e-10f) )
    {% else %}
    if (result1[i] + {{Test.TypeS}}(1) != result3[i] || result2[i] != {{Test.TypeS}}(1))
    {% endif %}
    {
      passed = false;
      break;
    }
  }

  {% if Test.VecLen == 4 %}
  const {{Test.TypeS}}  dat3 = dot3(Cx1, Cx2);
  const {{Test.TypeS}}  dat4 = dot4(Cx1, Cx2);
  const {{Test.Type}}   crs4 = cross3(Cx1, Cx2);
  {% endif %}
  const {{Test.TypeS}}  dat5 = dot  (Cx1, Cx2);
  {% if Test.VecLen >= 3 %}
  const {{Test.Type}}   crs3 = cross(Cx1, Cx2);
  const {{Test.TypeS}} crs_ref[3] = { Cx1[1]*Cx2[2] - Cx1[2]*Cx2[1], 
                                      Cx1[2]*Cx2[0] - Cx1[0]*Cx2[2], 
                                      Cx1[0]*Cx2[1] - Cx1[1]*Cx2[0] };
  {% endif %}
  {% if Test.IsFloat %}
  {% if Test.VecLen == 4 %}
  passed = passed && (std::abs(dat3 - (Cx1.x*Cx2.x + Cx1.y*Cx2.y + Cx1.z*Cx2.z)) < 1e-6f);
  passed = passed && (std::abs(dat4 - (Cx1.x*Cx2.x + Cx1.y*Cx2.y + Cx1.z*Cx2.z + Cx1.w*Cx2.w)) < 1e-6f);
  passed = passed && (std::abs(dot(crs4-crs3, crs4-crs3)) < 1e-6f);
  {% endif %}
  {
    {{Test.TypeS}} sum = {{Test.TypeS}}(0);
    for(int i=0;i<{{Test.VecLen}};i++)
      sum += Cx1[i]*Cx2[i];
    passed = passed && (std::abs(sum - dat5) < 1e-6f);
    {% if Test.VecLen >= 3 %}
    for(int i=0;i<3;i++)
      passed = passed && (std::abs(crs3[i] - crs_ref[i]) < 1e-6f);
    {% endif %}
  }
  {% else %}
  {% if Test.VecLen == 4 %}
  passed = passed && (dat3 == Cx1.x*Cx2.x + Cx1.y*Cx2.y + Cx1.z*Cx2.z);
  passed = passed && (dat4 == Cx1.x*Cx2.x + Cx1.y*Cx2.y + Cx1.z*Cx2.z + Cx1.w*Cx2.w);
  passed = passed && (dot(crs4-crs3, crs4-crs3) == 0);
  {% endif %}
  {
    {{Test.TypeS}} sum = {{Test.TypeS}}(0);
    for(int i=0;i<{{Test.VecLen}};i++)
      sum += Cx1[i]*Cx2[i];
    passed = passed && (sum == dat5);
    {% if Test.VecLen >= 3 %}
    for(int i=0;i<3;i++)
      passed = passed && (crs3[i] == crs_ref[i]);
    {% endif %}
  }
  {% endif %}

  return passed;
}

## endfor