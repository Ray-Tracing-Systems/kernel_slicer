#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#define LAYOUT_STD140
#include "LiteMath.h"
using namespace LiteMath;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SphHarm
{
  static constexpr uint32_t COEFS_COUNT = 9;
  static constexpr float    PI          = 3.14159265358979323846f;
  static constexpr float    SQRT_PI     = 1.772453851f;

public:
  SphHarm(){}

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  virtual void ProcessPixels(const uint32_t* a_data __attribute__((size("a_width", "a_height"))), uint32_t a_width, uint32_t a_height);
  void kernel2D_IntegrateSphHarm(const uint32_t* a_data, uint32_t a_width, uint32_t a_height);
  
  void GetCoefficients(LiteMath::float3 out_coeff[COEFS_COUNT]) const { memcpy(out_coeff, coefs, sizeof(LiteMath::float3)*COEFS_COUNT); }

  uint32_t m_width, m_height;
  LiteMath::float3 coefs[COEFS_COUNT];
};

#endif