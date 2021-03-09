#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>
#include "include/OpenCLMathCPU.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SphHarm
{
public:
  static constexpr uint32_t COEFS_COUNT = 9;

  void ProcessPixels(uint32_t* a_data, uint32_t a_width, uint32_t a_height);

  void GetCoefficients(LiteMath::float3 out_coeff[COEFS_COUNT]) const { memcpy(out_coeff, coefs, sizeof(LiteMath::float3)*COEFS_COUNT); }
  
  void kernel2D_IntegrateSphHarm(uint32_t* a_data, uint32_t a_width, uint32_t a_height);

  uint32_t m_width, m_height;
  LiteMath::float3 coefs[COEFS_COUNT];
};

#endif