#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include "include/OpenCLMath.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SphHarm
{
  static constexpr uint32_t COEFS_COUNT = 9;
  static constexpr float    PI          = 3.14159265358979323846f;
  static constexpr float    SQRT_PI     = 1.772453851f;

public:
  void ProcessPixels(uint32_t* a_data, uint32_t a_width, uint32_t a_height);

  void GetCoefficients(LiteMath::float3 out_coeff[COEFS_COUNT]) const { memcpy(out_coeff, coefs, sizeof(LiteMath::float3)*COEFS_COUNT); }
  
  void kernel2D_IntegrateSphHarm(uint32_t* a_data, uint32_t a_width, uint32_t a_height);
  void kernel1D_FinalizeCoeff(uint32_t a_size, uint32_t a_width, uint32_t a_height);

  uint32_t m_width, m_height;
  LiteMath::float3 coefs[COEFS_COUNT];
};

#endif