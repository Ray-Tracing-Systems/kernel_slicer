#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
using namespace LiteMath;

bool test000_scalar_funcs()
{
  volatile float x = 3.75f;
  volatile float z = -1.25f;

  float y01 = mod(x,0.5); // mod returns the value of x modulo y. This is computed as x - y * floor(x/y). 
  float y02 = mod(z,0.5); // mod returns the value of x modulo y. This is computed as x - y * floor(x/y). 
  
  float y03 = fract(x);   // return only the fraction part of a number; This is calculated as x - floor(x). 
  float y04 = fract(z);   // return only the fraction part of a number; This is calculated as x - floor(x).
  
  float y05 = ceil(x);          // nearest integer that is greater than or equal to x
  float y06 = ceil(z);          // nearest integer that is greater than or equal to x

  float y07 = floor(x);         // nearest integer less than or equal to x
  float y08 = floor(z);         // nearest integer less than or equal to x

  float y09 = sign(x);          // sign returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x is greater than 0.0. 
  float y10 = sign(z);          // sign returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x is greater than 0.0. 
  float y11 = sign(0.0f); 

  float y12 = abs(x);           // return the absolute value of x
  float y13 = abs(z);           // return the absolute value of x

  float y14 = clamp(x,0.0,1.0); // constrain x to lie between 0.0 and 1.0
  float y15 = clamp(z,0.0,1.0); // constrain x to lie between 0.0 and 1.0
  
  float y16 = min(x,z);    // return the lesser of x and z
  float y17 = max(x,z);    // return the greater of x and z 
  
  float y18 = mix (2.0f, 4.0f, 0.5f); // linear interpolate, same as lerp
  float y19 = lerp(2.0f, 4.0f, 0.5f); // linear interpolate
  
  float y20 = smoothstep(2.0f, 4.0f, 3.25f);
  float y21 = smoothstep(2.0f, 4.0f, 3.5f);
  float y22 = smoothstep(2.0f, 4.0f, 3.75f);

  float y23 = sqrt(x);

  float y26 = inversesqrt(x);   // 1.0f / sqrt(x)
  float y27 = rcp(x);           // fast reciprocal

  // check 
  //
  bool passed = true;
  passed = passed && (y01 == y02)  && (y01 == 0.25f);
  passed = passed && (y03 == y04)  && (y03 == 0.75f);
  passed = passed && (y05 == 4.0f) && (y06 == -1.0f);
  passed = passed && (y07 == 3.0f) && (y08 == -2.0f);
  passed = passed && (y09 == 1.0f) && (y10 == -1.0f) && (y11 == 0.0f);
  passed = passed && (y12 == 3.75f) && (y13 == 1.25f);
  passed = passed && (y14 == 1.0f)  && (y15 == 0.0f);
  passed = passed && (y16 == -1.25f) && (y17 == 3.75f);
  passed = passed && (y18 == 3.0f) && (y19 == 3.0f);
  passed = passed && (y20 > 0.0f) && (y20 < 1.0f) && (y21 > 0.0f) && (y21 < 1.0f) && (y22 > 0.0f) && (y22 < 1.0f);
  passed = passed && abs(y23*y23 - x) < 1e-6f && abs((1.0f/y26)*(1.0f/y26) - x) < 1e-5f;
  passed = passed && fabs(1.0f/y27 - x) < 1e-4f; 

  return passed;
}

bool test001_dot_cross_f4()
{
  const float4 Cx1 = { 1.0f, 2.0f, 3.0f, 4.0f };
  const float4 Cx2 = { 5.0f, 6.0f, 7.0f, 8.0f };

  const float  dot1 = dot3f(Cx1, Cx2);
  const float4 dot2 = dot3v(Cx1, Cx2);
  const float  dot3 = dot4f(Cx1, Cx2);
  const float4 dot4 = dot4v(Cx1, Cx2);
  const float  dot5 = dot  (Cx1, Cx2);
  const float4 crs3 = cross(Cx1, Cx2);

  CVEX_ALIGNED(16) float result1[4];
  CVEX_ALIGNED(16) float result2[4];
  CVEX_ALIGNED(16) float result3[4];
  store(result1, dot2);
  store(result2, dot4);
  store(result3, crs3);

  const float ref_dp3 = 1.0f*5.0f + 2.0f*6.0f + 3.0f*7.0f;
  const float ref_dp4 = 1.0f*5.0f + 2.0f*6.0f + 3.0f*7.0f + 4.0f*8.0f;

  const float crs_ref[3] = { Cx1[1]*Cx2[2] - Cx1[2]*Cx2[1], 
                             Cx1[2]*Cx2[0] - Cx1[0]*Cx2[2], 
                             Cx1[0]*Cx2[1] - Cx1[1]*Cx2[0] };

  const bool b1 = fabs(dot1 - ref_dp3) < 1e-6f;
  const bool b2 = fabs(result1[0] - ref_dp3) < 1e-6f && 
                  fabs(result1[1] - ref_dp3) < 1e-6f && 
                  fabs(result1[2] - ref_dp3) < 1e-6f &&
                  fabs(result1[3] - ref_dp3) < 1e-6f;

  const bool b3 = fabs(dot3 - ref_dp4) < 1e-6f;
  const bool b4 = fabs(result2[0] - ref_dp4) < 1e-6f &&
                  fabs(result2[1] - ref_dp4) < 1e-6f &&
                  fabs(result2[2] - ref_dp4) < 1e-6f &&
                  fabs(result2[3] - ref_dp4) < 1e-6f;

  const bool b5 = fabs(result3[0] - crs_ref[0]) < 1e-6f && 
                  fabs(result3[1] - crs_ref[1]) < 1e-6f &&
                  fabs(result3[2] - crs_ref[2]) < 1e-6f;

  const bool b6 = fabs(dot5 - dot3) < 1e-10f;

  return b1 && b2 && b3 && b4 && b5 && b6;
}

bool test002_dot_cross_f3()
{
  const float3 Cx1(1.0f, 2.0f, 3.0f);
  const float3 Cx2(5.0f, 6.0f, 7.0f);

  const float   dot1 = dot(Cx1, Cx2);
  const float3  crs3 = cross(Cx1, Cx2);

  CVEX_ALIGNED(16) float result3[4];
  store(result3, crs3);

  const float ref_dp3 = 1.0f*5.0f + 2.0f*6.0f + 3.0f*7.0f;
  const float crs_ref[3] = { Cx1[1]*Cx2[2] - Cx1[2]*Cx2[1], 
                             Cx1[2]*Cx2[0] - Cx1[0]*Cx2[2], 
                             Cx1[0]*Cx2[1] - Cx1[1]*Cx2[0] };

  const bool b1 = fabs(dot1 - ref_dp3) < 1e-6f;
  const bool b5 = fabs(result3[0] - crs_ref[0]) < 1e-6f && 
                  fabs(result3[1] - crs_ref[1]) < 1e-6f &&
                  fabs(result3[2] - crs_ref[2]) < 1e-6f;

  return b1 && b5;
}

bool test003_length_float4()
{
  const float4 Cx1 = { 1.0f, 2.0f, 3.0f, 4.0f };

  const float   dot1 = length3f(Cx1);
  const float4 dot2  = length3v(Cx1);
  const float   dot3 = length4f(Cx1);
  const float4 dot4  = length4v(Cx1);

  CVEX_ALIGNED(16) float result1[4];
  CVEX_ALIGNED(16) float result2[4];
  store(result1, dot2);
  store(result2, dot4);

  const float ref_dp3 = sqrtf(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f);
  const float ref_dp4 = sqrtf(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f + 4.0f*4.0f);

  const bool b1 = fabs(dot1 - ref_dp3) < 1e-6f;
  const bool b2 = fabs(result1[0] - ref_dp3) < 1e-6f && 
                  fabs(result1[1] - ref_dp3) < 1e-6f && 
                  fabs(result1[2] - ref_dp3) < 1e-6f &&
                  fabs(result1[3] - ref_dp3) < 1e-6f;

  const bool b3 = fabs(dot3 - ref_dp4) < 1e-6f;
  const bool b4 = fabs(result2[0] - ref_dp4) < 1e-6f &&
                  fabs(result2[1] - ref_dp4) < 1e-6f &&
                  fabs(result2[2] - ref_dp4) < 1e-6f &&
                  fabs(result2[3] - ref_dp4) < 1e-6f;
                  
  return (b1 && b2 && b3 && b4);
}

bool test004_colpack_f4x4()
{
  const float4 Cx1 = { 0.25f, 0.5f, 0.0, 1.0f };

  const unsigned int packed_rgba = color_pack_rgba(Cx1);
  const unsigned int packed_bgra = color_pack_bgra(Cx1);

  const bool passed = ((packed_bgra == 0xFF408000) || (packed_bgra == 0xFF3f7f00)) && 
                      ((packed_rgba == 0xFF008040) || (packed_rgba == 0xFF007f3f));

  if(!passed)
  {
    std::cout << std::hex << "bgra_res: " << packed_bgra << std::endl;
    std::cout << std::hex << "bgra_ref: " << 0xFF408000  << std::endl;

    std::cout << std::hex << "rgba_res: " << packed_rgba << std::endl;
    std::cout << std::hex << "rgba_ref: " << 0xFF008040  << std::endl;
  }

  return passed;
}

bool test005_matrix_elems()
{
  float4x4 m;
  m(1,2) = 3.0f;
  m(3,1) = 4.0f; 
  return (m[1][2] == 3.0f) && (m[3][1] == 4.0f) && (m[0][0] == 1.0f) && (m[1][1] == 1.0f) && (m[0][3] == 0.0f);
}

bool test006_any_all()
{
  const float4 Cx1 = { 1.0f, 2.0f, 7.0f, 2.0f };
  const float4 Cx2 = { 5.0f, 6.0f, 7.0f, 4.0f };
  const float4 Cx3 = { 5.0f, 6.0f, 8.0f, -4.0f };

  const auto cmp1 = (Cx1 > Cx2);
  const auto cmp2 = (Cx1 <= Cx3);

  const bool b1 = any_of(cmp1);
  const bool b2 = all_of(cmp1);

  const bool b3 = any_of(cmp2);
  const bool b4 = all_of(cmp2);

  return (!b1 && !b2 && b3 && !b4);
}

bool test007_reflect()
{
  const float4 dir4(1.0f, -1.0f, 0.0f, 0.0f);
  const float3 dir3(1.0f, -1.0f, 0.0f);
  const float2 dir2(1.0f, -1.0f);

  const float4 n4(0.0f, 1.0f, 0.0f, 0.0f);
  const float3 n3(0.0f, 1.0f, 0.0f);
  const float2 n2(0.0f, 1.0f);

  auto  r4 = reflect(dir4, n4);
  auto  r3 = reflect(dir3, n3);
  auto  r2 = reflect(dir2, n2);

  const float cos41 = dot3f(dir4, n4);
  const float cos42 = dot3f(r4, n4);

  const float cos31 = dot(dir4, n4);
  const float cos32 = dot(r4, n4);

  const float cos21 = dot(dir4, n4);
  const float cos22 = dot(r4, n4);

  return abs(cos41+cos42) < 1e-6f && abs(cos31+cos32) < 1e-6f && abs(cos21+cos22) < 1e-6f;
}

bool test008_normalize()
{
  const float4 dir4 = { 1.0f, 2.0f, 3.0f, 4.0f };
  const float3 dir3 = { 1.0f, 2.0f, 3.0f};
  const float2 dir2 = { 1.0f, 2.0f };
  
  const float4 n5  = normalize(dir4);
  const float4 n4  = normalize3(dir4);
  const float3 n3  = normalize(dir3);
  const float2 n2  = normalize(dir2);

  bool ok5 = length3(n5-n4) > 0.15f;
  bool ok4 = abs(length3f(n4) - 1.0f) < 1e-6f && abs( dot3f(n4, dir4/length3f(dir4)) - 1.0f) < 1e-6f;
  bool ok3 = abs(length(n3)   - 1.0f) < 1e-6f && abs( dot  (n3, dir3/length(dir3))   - 1.0f) < 1e-6f;
  bool ok2 = abs(length(n2)   - 1.0f) < 1e-6f && abs( dot  (n2, dir2/length(dir2))   - 1.0f) < 1e-6f;

  return ok5 && ok4 && ok3 && ok2;
}

bool test009_refract()
{
  float4 dir4 = { 10.0f, -1.0f, 0.0f, 0.0f };
  float3 dir3 = { 10.0f, -1.0f, 0.0f};
  float2 dir2 = { 10.0f, -1.0f };

  dir4 = normalize(dir4);
  dir3 = normalize(dir3);
  dir2 = normalize(dir2);

  const float4 n4  = { 0.0f, 1.0f, 0.0f, 0.0f };
  const float3 n3  = { 0.0f, 1.0f, 0.0f };
  const float2 n2  = { 0.0f, 1.0f };
  
  auto  r41 = refract(dir4, n4, 1.0f);
  auto  r31 = refract(dir3, n3, 1.0f);
  auto  r21 = refract(dir2, n2, 1.0f);
  
  auto  r42 = refract(dir4, n4, 10.0f);
  auto  r32 = refract(dir3, n3, 10.0f);
  auto  r22 = refract(dir2, n2, 10.0f);

  bool ok4 = length3f(r41-dir4) < 1e-6f && length3f(r42) == 0.0f;
  bool ok3 = length  (r31-dir3) < 1e-6f && length  (r32) == 0.0f;
  bool ok2 = length  (r21-dir2) < 1e-6f && length  (r22) == 0.0f;

  return ok4 && ok3 && ok2;
}

bool test010_faceforward()
{
  const float4 dir4 = { 1.0f, -1.0f, 0.0f, 0.0f };
  const float3 dir3(1.0f, -1.0f, 0.0f);
  const float2 dir2 = { 1.0f, -1.0f };

  const float4 n4  = { 0.0f, -1.0f, 0.0f, 0.0f };
  const float3 n3  = { 0.0f, -1.0f, 0.0f };
  const float2 n2  = { 0.0f, -1.0f };

  auto  r41 = faceforward(n4, dir4, n4);
  auto  r31 = faceforward(n3, dir3, n3);
  auto  r21 = faceforward(n2, dir2, n2);
  
  bool ok4 = abs(dot3f(r41, n4) + 1.0f) < 1e-6f;
  bool ok3 = abs(dot  (r31, n3) + 1.0f) < 1e-6f;
  bool ok2 = abs(dot  (r21, n2) + 1.0f) < 1e-6f;

  return ok4 && ok3 && ok2;
}

bool test011_mattranspose()
{
  float4x4 m(1.0f,2.0f,3.0f,4.0f,
             5.0f,6.0f,7.0f,8.0f,
             9.0f,10.0f,11.0f,12.0f,
             13.0f,14.0f,15.0f,16.0f);

  float4x4 m2 = transpose(m);
  
  double error = 0.0;
  for(int i=0;i<4;i++)
  {
    for(int j=0;j<4;j++)
      error += fabs( m(i,j) - m2(j,i));
  }
  
  return error < 1e-20f;
}
