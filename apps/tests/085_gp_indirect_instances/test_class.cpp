#include "test_class.h"

#include <chrono>
#include <cmath>
#include <string>

static inline uint32_t HashCell(uint32_t a_x) // integer hash, the same on CPU and GPU
{
  a_x ^= a_x >> 16; a_x *= 0x7feb352dU;
  a_x ^= a_x >> 15; a_x *= 0x846ca68bU;
  a_x ^= a_x >> 16;
  return a_x;
}

void IndirectCells::SetViewport(int a_width, int a_height)
{
  m_quad        = { float4(0.0f, 0.0f, 0.5f, 1.0f), float4(1.0f, 0.0f, 0.5f, 1.0f), float4(1.0f, 1.0f, 0.5f, 1.0f), float4(0.0f, 1.0f, 0.5f, 1.0f) };
  m_quadIndices = { 0, 1, 2, 0, 2, 3 };
  m_visible.reserve(GRID*GRID); // capacity of the GPU buffer

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::NONE;
  m_pipeline.clearColor = float4(0.1f, 0.1f, 0.15f, 1.0f);
}

void IndirectCells::Render(int a_width, int a_height, uint32_t a_frame, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  kernel1D_Cull(GRID*GRID, a_frame);
  kernel1D_SetMarkers(1, a_frame);

  LiteRS::Clear(m_pipeline);
  // both counts read data of the class (sizes of member vectors), so they are computed on GPU: indirect draw
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &IndirectCells::kernelVS_Cell, &IndirectCells::kernelPS_Color,
                                        m_quadIndices.data(), uint32_t(m_quadIndices.size()), uint32_t(m_visible.size()));
  // instance count is a POD member written by kernel1D_SetMarkers on GPU: indirect draw
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &IndirectCells::kernelVS_Marker, &IndirectCells::kernelPS_Color,
                                        m_quadIndices.data(), uint32_t(m_quadIndices.size()), m_markerNum);
  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void IndirectCells::kernel1D_Cull(uint32_t a_cellNum, uint32_t a_frame)
{
  m_visible.resize(0);
  for(uint32_t cellId = 0; cellId < a_cellNum; cellId++)
  {
    if(HashCell(cellId*4 + a_frame) % 100 < 20 + 25*a_frame) // a different random subset of cells in every frame
      m_visible.push_back(cellId);
  }
}

void IndirectCells::kernel1D_SetMarkers(uint32_t a_threads, uint32_t a_frame)
{
  for(uint32_t i = 0; i < a_threads; i++)
    m_markerNum = 1 + a_frame;
}

float4 IndirectCells::kernelVS_Cell(uint32_t a_vertId, uint32_t a_instId, ColorVaryings* a_out) const
{
  const uint32_t cellId = m_visible[a_instId];
  const float4   p      = m_quad[a_vertId];
  const float    x      = (float(cellId % GRID) + p.x)/float(GRID);
  const float    y      = (float(cellId / GRID) + p.y)/float(GRID);
  a_out->color = float4(float((cellId*37) % 256)/255.0f, float((cellId*101) % 256)/255.0f, float((cellId*13) % 256)/255.0f, 1.0f);
  return float4(2.0f*x - 1.0f, 2.0f*y - 1.0f, p.z, 1.0f);
}

float4 IndirectCells::kernelVS_Marker(uint32_t a_vertId, uint32_t a_instId, ColorVaryings* a_out) const
{
  const float4 p    = m_quad[a_vertId];
  const float  size = 0.125f;                       // 64 pixels: vertices are on pixel borders
  const float  x0   = -0.875f + 0.375f*float(a_instId);
  a_out->color = float4(1.0f, 1.0f, 1.0f, 1.0f);
  return float4(x0 + size*p.x, x0 + size*p.y, 0.25f, 1.0f); // in front of the cells
}

float4 IndirectCells::kernelPS_Color(const ColorVaryings& a_in) const
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

void IndirectCells::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[int2(x, y)]);
  }
}

void IndirectCells::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
