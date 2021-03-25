#ifndef TestClass_UBO_H
#define TestClass_UBO_H

#include "OpenCLMath.h"

struct TestClass_UBO_Data
{
  float4x4 m_worldViewProjInv;
  float3 camPos;
  unsigned int dummy0;
  unsigned int m_indicesReordered_capacity;
  unsigned int m_indicesReordered_size;
  unsigned int m_intervals_capacity;
  unsigned int m_intervals_size;
  unsigned int m_nodes_capacity;
  unsigned int m_nodes_size;
  unsigned int m_randomGens_capacity;
  unsigned int m_randomGens_size;
  unsigned int m_vNorm4f_capacity;
  unsigned int m_vNorm4f_size;
  unsigned int m_vPos4f_capacity;
  unsigned int m_vPos4f_size;
  unsigned int spheresMaterials_capacity;
  unsigned int spheresMaterials_size;
  unsigned int dummy_last;
};

#endif

