#ifndef TestClass_UBO_H
#define TestClass_UBO_H

struct TestClass_UBO_Data
{
  float4x4 m_worldViewProjInv;
  unsigned int m_randomGens_capacity;
  unsigned int m_randomGens_size;
  unsigned int spheresMaterials_capacity;
  unsigned int spheresMaterials_size;
  unsigned int spheresPosRadius_capacity;
  unsigned int spheresPosRadius_size;
  unsigned int dummy_last;
};

#endif

