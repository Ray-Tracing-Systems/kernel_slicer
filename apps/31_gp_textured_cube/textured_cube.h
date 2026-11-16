#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
#include "Image2d.h"
#include "raster.h"

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::int2;
using LiteMath::clamp;
using LiteMath::normalize;
using LiteMath::dot;
using LiteImage::Image2D;
using LiteImage::Sampler;

struct TexVaryings  // vertex shader output, pixel shader input
{
  float4 normal;    ///< world space
  float2 uv;
};

/**
\brief Rotating cube with a texture: indexed draw, texture sampling in pixel shader, RGBA8 color target, back face culling.
*/
class TexturedCube
{
public:
  TexturedCube() {}
  virtual ~TexturedCube() {}

  void SetViewport(int a_width, int a_height); ///< host only: texture, geometry, camera and pipeline state

  virtual void Render(int a_width, int a_height, float a_angle, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

protected:

  float4 kernelVS_Cube(uint32_t a_vertId, float a_angle, TexVaryings* a_out) const;
  float4 kernelPS_Cube(const TexVaryings& a_in) const;
  virtual void   kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  std::vector<float4>   m_positions;
  std::vector<float4>   m_normals;
  std::vector<float2>   m_texCoords;
  std::vector<uint32_t> m_indices;

  Image2D<uint32_t>     m_texture;
  Sampler               m_sampler;
  float4x4              m_viewProj;
  float4                m_lightDir;

  Image2D<uint32_t>     m_color;     ///< RGBA8
  Image2D<float>        m_depth;
  LiteRS::Pipeline      m_pipeline;
  float                 m_timeRender = 0.0f;
};
