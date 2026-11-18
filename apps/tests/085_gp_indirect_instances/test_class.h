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
\brief Indirect draws: instance counts are computed by kernels on GPU, so the draws are vkCmdDrawIndexedIndirect.
         * kernel1D_Cull selects visible cells of a GRID x GRID grid with push_back to m_visible, then all visible cells are
           drawn by one DrawIndexedInstancedTriangles with 'uint32_t(m_visible.size())' instances (index count
           'uint32_t(m_quadIndices.size())' is also read on GPU);
         * kernel1D_SetMarkers writes the POD member m_markerNum, it is the instance count of the second draw.
       Every frame selects a different number of cells; cells are pixel aligned, so CPU and GPU images are equal.
*/
class IndirectCells
{
public:
  IndirectCells() {}
  virtual ~IndirectCells() {}

  static constexpr uint32_t GRID       = 256; ///< GRID x GRID cells
  static constexpr int      CELL_SIZE  = 2;   ///< cell is CELL_SIZE x CELL_SIZE pixels
  static constexpr int      IMAGE_SIZE = 512; ///< GRID*CELL_SIZE

  void SetViewport(int a_width, int a_height); ///< host only: images, pipeline state and geometry

  virtual void Render(int a_width, int a_height, uint32_t a_frame, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

protected:

  virtual void kernel1D_Cull(uint32_t a_cellNum, uint32_t a_frame);         ///< visible cells --> m_visible
  virtual void kernel1D_SetMarkers(uint32_t a_threads, uint32_t a_frame);   ///< one thread: m_markerNum on GPU
  float4 kernelVS_Cell  (uint32_t a_vertId, uint32_t a_instId, ColorVaryings* a_out) const;
  float4 kernelVS_Marker(uint32_t a_vertId, uint32_t a_instId, ColorVaryings* a_out) const;
  float4 kernelPS_Color(const ColorVaryings& a_in) const;
  virtual void kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  std::vector<float4>   m_quad;        ///< 4 vertices of the unit square [0,1]^2
  std::vector<uint32_t> m_quadIndices; ///< 2 triangles
  std::vector<uint32_t> m_visible;     ///< ids of visible cells, written by kernel1D_Cull
  uint32_t              m_markerNum = 0;

  Image2D<float4>       m_color;
  Image2D<float>        m_depth;
  LiteRS::Pipeline      m_pipeline;
  float                 m_timeRender = 0.0f;
};
