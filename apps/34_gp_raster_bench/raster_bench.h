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

struct BenchVaryings  // vertex shader output, pixel shader input
{
  float4 normal;  ///< world space
  float4 color;
};

/**
\brief Benchmark of the graphics pipeline with a controlled load, three scenes with the same shading:
       SMALL  -- many spheres with small triangles (1-4 pixels),
       MEDIUM -- spheres with triangles of about 100 pixels,
       LARGE  -- 16 big planes covering the image drawn from back to front: every pixel is shaded 16 times.
*/
class RasterBench
{
public:
  RasterBench() {}
  virtual ~RasterBench() {}

  void SetViewport(int a_width, int a_height, LiteImage::Layout a_layout = LiteImage::Layout::TILED4x4, int a_binSize = 64); ///< host only: geometry, camera, images

  virtual void RenderSmall (int a_width, int a_height, uint32_t* a_out [[size("a_width*a_height")]]);
  virtual void RenderMedium(int a_width, int a_height, uint32_t* a_out [[size("a_width*a_height")]]);
  virtual void RenderLarge (int a_width, int a_height, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

  LiteRS::Pipeline& GetPipeline() { return m_pipeline; } ///< host only: CPU settings and statistics of the rasterizer

protected:

  float4 kernelVS_Small (uint32_t a_vertId, BenchVaryings* a_out) const;
  float4 kernelVS_Medium(uint32_t a_vertId, BenchVaryings* a_out) const;
  float4 kernelVS_Large (uint32_t a_vertId, BenchVaryings* a_out) const;
  float4 kernelPS_Shade (const BenchVaryings& a_in) const;
  virtual void kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  float4 Transform(uint32_t a_vertId, BenchVaryings* a_out) const;      ///< common part of vertex shaders

  std::vector<float4>   m_positions;  ///< vertices of all scenes
  std::vector<float4>   m_normals;
  std::vector<float4>   m_colors;
  std::vector<uint32_t> m_indicesSmall;
  std::vector<uint32_t> m_indicesMedium;
  std::vector<uint32_t> m_indicesLarge;

  float4x4              m_viewProj;
  float4                m_lightDir;

  Image2D<float4>       m_color;
  Image2D<float>        m_depth;
  LiteRS::Pipeline      m_pipeline;
  float                 m_timeRender[3] = {0.0f, 0.0f, 0.0f};
};
