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
\brief Ordinary (not indirect) draws with counts from local variables and arguments of the control function.

A count which reads data of the class directly in the arguments of a draw command ('m_n', 'uint32_t(m_vec.size())') is
read on GPU, then the draw is indirect (tests 085, 086). If the value is first read into a local variable of the control
function and the variable is passed to the command, the draw is an ORDINARY draw (vkCmdDraw / vkCmdDrawIndexed): the count
is computed on host when commands are recorded, so changes of class data made by kernels on GPU are not seen by it.
Arguments of the control function and constants also give ordinary draws.

The last draw mixes both kinds: its index count is a local variable (host), its instance count is a member written by a
kernel (GPU), so the draw is indirect and the host count is passed to the shader which writes the indirect command.
*/
class LocalCountDraws
{
public:
  LocalCountDraws() {}
  virtual ~LocalCountDraws() {}

  static constexpr uint32_t ROW_SIZE = 8; ///< quads in a row

  void SetViewport(int a_width, int a_height); ///< host only: images, pipeline state and geometry

  virtual void Render(int a_width, int a_height, uint32_t a_instNum, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

protected:

  virtual void kernel1D_SetGPUCount(uint32_t a_threads, uint32_t a_value);  ///< one thread: m_gpuInstNum on GPU
  float4 kernelVS_Quad(uint32_t a_vertId, uint32_t a_instId, uint32_t a_row, ColorVaryings* a_out) const;
  float4 kernelPS_Color(const ColorVaryings& a_in) const;
  virtual void kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  std::vector<float4>   m_quadVerts;   ///< 6 vertices: two triangles of a unit square [0,1]^2
  std::vector<uint32_t> m_quadIndices; ///< 0, 1, ..., 5
  uint32_t              m_instNum    = 0; ///< set on host
  uint32_t              m_gpuInstNum = 0; ///< written by kernel1D_SetGPUCount

  Image2D<float4>       m_color;
  Image2D<float>        m_depth;
  LiteRS::Pipeline      m_pipeline;
  float                 m_timeRender = 0.0f;
};
