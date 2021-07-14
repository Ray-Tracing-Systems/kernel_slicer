#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"

bool test000_scalar_functions_f()
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
  float y24 = mad(x, y01, y02); // fused multyply-add
  float y25 = fma(x, y01, y02); // fused multyply-add
  float y26 = inversesqrt(x);   // 1.0f / sqrt(x)
  
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
  passed = passed && (y24 == y25) && (y24 == x*y01 + y02);

  return passed;
}

bool test001_dot_cross_f4()
{
  const float4 Cx1 = { 1.0f, 2.0f, 3.0f, 4.0f };
  const float4 Cx2 = { 5.0f, 6.0f, 7.0f, 8.0f };

  const float   dot1 = dot3f(Cx1, Cx2);
  const float4 dot2  = dot3v(Cx1, Cx2);
  const float   dot3 = dot4f(Cx1, Cx2);
  const float4 dot4  = dot4v(Cx1, Cx2);
  const float4 crs3  = cross(Cx1, Cx2);

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

  return b1 && b2 && b3 && b4 && b5;
}

bool test002_dot_cross_f3()
{
  const float3 Cx1 = { 1.0f, 2.0f, 3.0f };
  const float3 Cx2 = { 5.0f, 6.0f, 7.0f };

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

bool test003_length_f4()
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
