#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include "include/OpenCLMathCPU.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SphHarm
{
public:
  static const uint32_t COEFS_COUNT = 9;

  void ProcessPixels(uint32_t* a_data, uint32_t a_width, uint32_t a_height);

  std::array<LiteMath::float3, COEFS_COUNT> GetCoefficients() const { return coefs; }
  
  void kernel2D_IntegrateSphHarm(uint32_t* a_data, uint32_t a_width, uint32_t a_height);

  uint32_t               m_width, m_height;
  std::array<LiteMath::float3, COEFS_COUNT> coefs;
};

#endif