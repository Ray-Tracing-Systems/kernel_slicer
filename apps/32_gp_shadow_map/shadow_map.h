#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
#include "Image2d.h"
#include "raster.h"

using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::int2;
using LiteMath::clamp;
using LiteMath::normalize;
using LiteMath::dot;
using LiteImage::Image2D;

struct ShadowVaryings  // vertex shader output, pixel shader input
{
  float4 normal;    ///< world space
  float4 color;
  float4 lightPos;  ///< clip space position for the light camera (orthographic, w = 1)
};

/**
\brief Small scene (plane, cube, sphere, torus) with a shadow map: two graphics passes with different pipelines
       (depth only pass from the light and main pass), then an ordinary kernel for tone mapping.
*/
class ShadowRender
{
public:
  ShadowRender() {}
  virtual ~ShadowRender() {}

  void SetViewport(int a_width, int a_height); ///< host only: geometry, cameras, images and pipeline state

  virtual void Render(int a_width, int a_height, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

protected:

  float4 kernelVS_Shadow(uint32_t a_vertId) const;                        ///< depth only pass, no pixel shader
  float4 kernelVS_Main(uint32_t a_vertId, ShadowVaryings* a_out) const;
  float4 kernelPS_Main(const ShadowVaryings& a_in) const;
  void   kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  float  ShadowFactor(float4 a_lightPos, float a_bias) const;             ///< 3x3 PCF, 1 is lit

  std::vector<float4>   m_positions;
  std::vector<float4>   m_normals;
  std::vector<float4>   m_colors;
  std::vector<uint32_t> m_indices;

  float4x4              m_viewProj;
  float4x4              m_lightViewProj;
  float4                m_lightDir;       ///< direction to the light
  int                   m_shadowSize = 2048;

  Image2D<float>        m_shadowMap;
  Image2D<float4>       m_color;
  Image2D<float>        m_depth;
  LiteRS::Pipeline      m_shadowPass;
  LiteRS::Pipeline      m_mainPass;
  float                 m_timeRender = 0.0f;
};
