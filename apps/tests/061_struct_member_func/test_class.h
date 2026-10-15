#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "LiteMath.h"

using LiteMath::float3;
using LiteMath::float4;
using LiteMath::dot;
using LiteMath::normalize;
using LiteMath::length;

// user data structure with member functions which are used in kernels
struct SphereLight
{
  SphereLight() : posAndRadius(0.0f, 0.0f, 0.0f, 1.0f), color(1.0f) {}
  SphereLight(float4 a_posAndRadius, float4 a_color) : posAndRadius(a_posAndRadius), color(a_color) {}

  float4 posAndRadius;
  float4 color;

  float Attenuation(float a_dist) const;                                    // defined out of structure, in test_class.cpp
  static float Luminance(float4 a_color) { return 0.2126f*a_color.x + 0.7152f*a_color.y + 0.0722f*a_color.z; }

  float3 Pos()    const { return float3(posAndRadius.x, posAndRadius.y, posAndRadius.z); }
  float  Radius() const { return posAndRadius.w; }

  float Eval(float3 a_point, float3 a_normal) const
  {
    const float3 toLight = Pos() - a_point;
    const float  dist2   = std::max(dot(toLight, toLight), 1e-6f);
    const float  cosTheta = std::max(dot(normalize(toLight), a_normal), 0.0f);
    return color.w*cosTheta*Radius()*Radius()/dist2;
  }

  float Intensity() { return color.w; }                   // non-const, but does not modify fields: must not be '[mutating]'
  void  ScaleTwice(float a_mult) { Scale(a_mult); Scale(a_mult); } // modifies fields via other member function

  void Scale(float a_mult)          // non-const member function, modifies fields
  {
    color.w        *= a_mult;
    posAndRadius.w  = std::sqrt(posAndRadius.w*posAndRadius.w*a_mult);
  }
};

class StructMembers
{
public:
  StructMembers(){}

  void SetLights(const std::vector<SphereLight>& a_lights) { m_lights = a_lights; }

  virtual void Run(float* a_out [[size("a_size")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_Shade(float* a_out, uint32_t a_size);

  std::vector<SphereLight> m_lights;
};
