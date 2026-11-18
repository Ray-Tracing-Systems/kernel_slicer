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
\brief Non indexed instanced draws (LiteRS::DrawInstancedTriangles): a grid of quads, the vertex shader places and
       colors every quad by its instance id. A depth only instanced pass before it cuts holes in the grid.
*/
class InstancedQuads
{
public:
  InstancedQuads() {}
  virtual ~InstancedQuads() {}

  void SetViewport(int a_width, int a_height); ///< host only: images, pipeline state and geometry

  virtual void Render(int a_width, int a_height, float a_angle, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

  static constexpr uint32_t GRID_SIZE = 8; ///< GRID_SIZE x GRID_SIZE quads
  static constexpr uint32_t HOLES     = 2; ///< HOLES x HOLES holes

protected:

  float4 kernelVS_Quad(uint32_t a_vertId, uint32_t a_instId, float a_angle, ColorVaryings* a_out) const; ///< returns clip space position
  float4 kernelVS_Hole(uint32_t a_vertId, uint32_t a_instId) const;                                    ///< depth only pass
  float4 kernelPS_Quad(const ColorVaryings& a_in) const;
  virtual void kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  std::vector<float4> m_quad;      ///< 6 vertices (two triangles) of a unit quad [-1,1]^2
  Image2D<float4>     m_color;
  Image2D<float>      m_depth;
  LiteRS::Pipeline    m_pipeline;
  float               m_timeRender = 0.0f;
};
