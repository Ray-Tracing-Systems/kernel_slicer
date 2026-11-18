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

void IndirectTriangles::SetViewport(int a_width, int a_height)
{
  m_tris.reserve(GRID*GRID); // capacity of the GPU buffer
  m_holeVerts = { float4(-0.6f, -0.6f, 0.25f, 1.0f), float4(-0.1f, -0.6f, 0.25f, 1.0f), float4(-0.35f, -0.1f, 0.25f, 1.0f),
                  float4( 0.2f,  0.1f, 0.25f, 1.0f), float4( 0.7f,  0.2f, 0.25f, 1.0f), float4( 0.4f,  0.7f, 0.25f, 1.0f) };
  m_holeNum   = uint32_t(m_holeVerts.size()/3);

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::NONE;
  m_pipeline.clearColor = float4(0.1f, 0.1f, 0.15f, 1.0f);
}

void IndirectTriangles::Render(int a_width, int a_height, uint32_t a_frame, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  kernel1D_MakeTriangles(GRID*GRID, a_frame);

  LiteRS::Clear(m_pipeline);
  // vertex counts are expressions of data of the class (a POD member, the size of a member vector): indirect draws
  LiteRS::DrawTriangles(m_pipeline, this, &IndirectTriangles::kernelVS_Hole, 3*m_holeNum);
  LiteRS::DrawTriangles(m_pipeline, this, &IndirectTriangles::kernelVS_Tri, &IndirectTriangles::kernelPS_Color, 3*uint32_t(m_tris.size()));
  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void IndirectTriangles::kernel1D_MakeTriangles(uint32_t a_cellNum, uint32_t a_frame)
{
  m_tris.resize(0);
  for(uint32_t cellId = 0; cellId < a_cellNum; cellId++)
  {
    const uint32_t hash = HashCell(cellId*2 + a_frame);
    if(hash % 100 < 30 + 40*a_frame)
    {
      const float cellSize = 2.0f/float(GRID);
      const float x0       = -1.0f + cellSize*float(cellId % GRID);
      const float y0       = -1.0f + cellSize*float(cellId / GRID);
      const float shift    = 0.25f*cellSize*float((hash >> 8) % 4)/3.0f; // triangles stay inside their cells, they do not overlap
      GenTriangle tri;
      tri.p0    = float4(x0 + 0.1f*cellSize,         y0 + 0.1f*cellSize,  0.5f, 1.0f);
      tri.p1    = float4(x0 + 0.9f*cellSize,         y0 + 0.1f*cellSize + shift, 0.5f, 1.0f);
      tri.p2    = float4(x0 + 0.3f*cellSize + shift, y0 + 0.9f*cellSize,  0.5f, 1.0f);
      tri.color = float4(float(hash % 256)/255.0f, float((hash >> 16) % 256)/255.0f, 0.5f + 0.5f*float(a_frame), 1.0f);
      m_tris.push_back(tri);
    }
  }
}

float4 IndirectTriangles::kernelVS_Tri(uint32_t a_vertId, ColorVaryings* a_out) const
{
  const GenTriangle tri    = m_tris[a_vertId/3];
  const uint32_t    corner = a_vertId % 3;
  a_out->color = tri.color;
  return (corner == 0) ? tri.p0 : ((corner == 1) ? tri.p1 : tri.p2);
}

float4 IndirectTriangles::kernelVS_Hole(uint32_t a_vertId) const
{
  return m_holeVerts[a_vertId];
}

float4 IndirectTriangles::kernelPS_Color(const ColorVaryings& a_in) const
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

void IndirectTriangles::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[int2(x, y)]);
  }
}

void IndirectTriangles::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
