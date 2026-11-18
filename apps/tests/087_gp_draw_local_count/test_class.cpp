#include "test_class.h"

#include <chrono>
#include <cmath>
#include <string>

void LocalCountDraws::SetViewport(int a_width, int a_height)
{
  m_quadVerts   = { float4(0.0f, 0.0f, 0.5f, 1.0f), float4(1.0f, 0.0f, 0.5f, 1.0f), float4(1.0f, 1.0f, 0.5f, 1.0f),
                    float4(0.0f, 0.0f, 0.5f, 1.0f), float4(1.0f, 1.0f, 0.5f, 1.0f), float4(0.0f, 1.0f, 0.5f, 1.0f) };
  m_quadIndices = { 0, 1, 2, 3, 4, 5 };
  m_instNum     = 5;

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::NONE;
  m_pipeline.clearColor = float4(0.1f, 0.1f, 0.15f, 1.0f);
}

void LocalCountDraws::Render(int a_width, int a_height, uint32_t a_instNum, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  // Counts are read into local variables of the control function first: the draws below which use only them are
  // ORDINARY draws, the counts are computed on host. Write 'uint32_t(m_quadIndices.size())' or 'm_instNum' directly
  // in the arguments of a draw command to read them on GPU (indirect draw).
  //
  const uint32_t indexNum = uint32_t(m_quadIndices.size());
  const uint32_t vertNum  = uint32_t(m_quadVerts.size());
  const uint32_t instNum  = m_instNum;

  kernel1D_SetGPUCount(1, a_instNum + 2);

  LiteRS::Clear(m_pipeline);
  // ordinary draw: both counts are local variables
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &LocalCountDraws::kernelVS_Quad, &LocalCountDraws::kernelPS_Color,
                                        m_quadIndices.data(), indexNum, instNum, 0u);
  // ordinary draw: a local variable and an argument of the control function
  LiteRS::DrawInstancedTriangles(m_pipeline, this, &LocalCountDraws::kernelVS_Quad, &LocalCountDraws::kernelPS_Color,
                                 vertNum, a_instNum, 1u);
  // ordinary draw: constants
  LiteRS::DrawInstancedTriangles(m_pipeline, this, &LocalCountDraws::kernelVS_Quad, &LocalCountDraws::kernelPS_Color,
                                 6, ROW_SIZE, 2u);
  // indirect draw: the index count is a local variable (computed on host), the instance count is a member written on GPU
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &LocalCountDraws::kernelVS_Quad, &LocalCountDraws::kernelPS_Color,
                                        m_quadIndices.data(), indexNum, m_gpuInstNum, 3u);
  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void LocalCountDraws::kernel1D_SetGPUCount(uint32_t a_threads, uint32_t a_value)
{
  for(uint32_t i = 0; i < a_threads; i++)
    m_gpuInstNum = a_value;
}

float4 LocalCountDraws::kernelVS_Quad(uint32_t a_vertId, uint32_t a_instId, uint32_t a_row, ColorVaryings* a_out) const
{
  const float4 p    = m_quadVerts[a_vertId];
  const float  cell = 2.0f/float(ROW_SIZE);   // 64 pixels for 512x512: vertices are on pixel borders
  const float  x    = -1.0f + cell*(float(a_instId) + 0.125f + 0.75f*p.x);
  const float  y    = -1.0f + cell*(float(a_row)    + 0.125f + 0.75f*p.y);
  a_out->color = float4(0.25f + 0.25f*float(a_row), float(a_instId)/float(ROW_SIZE), 1.0f - 0.25f*float(a_row), 1.0f);
  return float4(x, y, p.z, 1.0f);
}

float4 LocalCountDraws::kernelPS_Color(const ColorVaryings& a_in) const
{
  return a_in.color;
}

static inline uint32_t PackRGB8(float4 a_color) // R in the lowest byte
{
  const uint32_t r = uint32_t(clamp(a_color.x, 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = uint32_t(clamp(a_color.y, 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = uint32_t(clamp(a_color.z, 0.0f, 1.0f)*255.0f + 0.5f);
  return r | (g << 8) | (b << 16);
}

void LocalCountDraws::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[int2(x, y)]);
  }
}

void LocalCountDraws::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
