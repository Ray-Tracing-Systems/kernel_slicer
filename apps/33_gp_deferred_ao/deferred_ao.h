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
using LiteMath::cross;
using LiteMath::dot;
using LiteImage::Image2D;

struct GBufferVaryings  // vertex shader output, pixel shader input
{
  float4 posView;       ///< view space position
  float4 normalView;    ///< view space normal
  float4 color;
};

struct GBufferOut       // pixel shader output: color targets 0, 1, 2
{
  float4 posView;
  float4 normalView;
  float4 albedo;        ///< w = 1 for geometry, 0 for background
};

/**
\brief Deferred shading with screen space ambient occlusion: G-buffer is drawn by graphics pipeline into 3 color
       targets, then ordinary kernels compute AO, blur it and shade the image.
*/
class DeferredRender
{
public:
  DeferredRender() {}
  virtual ~DeferredRender() {}

  void SetViewport(int a_width, int a_height); ///< host only: geometry, camera, AO kernel, images and pipeline state

  virtual void Render(int a_width, int a_height, uint32_t* a_out [[size("a_width*a_height")]], uint32_t* a_outAO [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

  static constexpr int AO_SAMPLES = 16;

protected:

  float4     kernelVS_GBuffer(uint32_t a_vertId, GBufferVaryings* a_out) const;
  GBufferOut kernelPS_GBuffer(const GBufferVaryings& a_in) const;
  virtual void       kernel2D_SSAO(int a_width, int a_height);
  virtual void       kernel2D_BlurAO(int a_width, int a_height);
  virtual void       kernel2D_Shade(int a_width, int a_height, uint32_t* a_out, uint32_t* a_outAO);

  std::vector<float4>   m_positions;
  std::vector<float4>   m_normals;
  std::vector<float4>   m_colors;
  std::vector<uint32_t> m_indices;
  std::vector<float4>   m_aoKernel;    ///< AO_SAMPLES points in the unit hemisphere (z > 0)

  float4x4              m_view;
  float4x4              m_proj;
  float4                m_lightDirView; ///< direction to the light in view space
  float                 m_aoRadius = 1.0f;

  Image2D<float4>       m_gPos;
  Image2D<float4>       m_gNormal;
  Image2D<float4>       m_gAlbedo;
  Image2D<float>        m_gDepth;
  Image2D<float>        m_ao;
  Image2D<float>        m_aoBlurred;
  LiteRS::Pipeline      m_gbufferPass;
  float                 m_timeRender = 0.0f;
};
