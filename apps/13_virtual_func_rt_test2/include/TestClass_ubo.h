#ifndef TestClass_UBO_H
#define TestClass_UBO_H

#ifndef GLSL
#include "OpenCLMath.h"
#else
#define float4x4 mat4
#define float3   vec3
#define float4   vec4
#define uint32_t uint
#endif

struct TestClass_UBO_Data
{
  float4x4 m_worldViewProjInv;
  float3 camPos;
  float4 m_lightSphere;
  float3 testColor;
  uint32_t m_emissiveMaterialId;
  uint m_indicesReordered_capacity;
  uint m_indicesReordered_size;
  uint m_intervals_capacity;
  uint m_intervals_size;
  uint m_materialData_capacity;
  uint m_materialData_size;
  uint m_materialIds_capacity;
  uint m_materialIds_size;
  uint m_materialOffsets_capacity;
  uint m_materialOffsets_size;
  uint m_nodes_capacity;
  uint m_nodes_size;
  uint m_randomGens_capacity;
  uint m_randomGens_size;
  uint m_vNorm4f_capacity;
  uint m_vNorm4f_size;
  uint m_vPos4f_capacity;
  uint m_vPos4f_size;
  uint dummy_last;
};

#endif

