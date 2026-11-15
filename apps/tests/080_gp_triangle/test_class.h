#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
#include "Image2d.h"
#include "raster.h"

using LiteMath::float4;
using LiteMath::int2;
using LiteMath::clamp;
using LiteImage::Image2D;

struct ColorVaryings  // vertex shader output, pixel shader input
{
  float4 color;
};

/**
\brief One triangle with per-vertex colors drawn by graphics pipeline (LiteRS), then an ordinary kernel reads the image.
*/
class TriangleRender
{
public:
  TriangleRender() {}
  virtual ~TriangleRender() {}

  void SetViewport(int a_width, int a_height); ///< host only: images, pipeline state and geometry

  virtual void Render(int a_width, int a_height, float a_angle, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

protected:

  float4 kernelVS_Color(uint32_t a_vertId, float a_angle, ColorVaryings* a_out) const; ///< returns clip space position
  float4 kernelPS_Color(const ColorVaryings& a_in) const;                             ///< returns color
  void   kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);              ///< ordinary kernel

  std::vector<float4> m_positions;
  std::vector<float4> m_colors;
  Image2D<float4>     m_color;
  Image2D<float>      m_depth;
  LiteRS::Pipeline    m_pipeline;
  float               m_timeRender = 0.0f;
};
