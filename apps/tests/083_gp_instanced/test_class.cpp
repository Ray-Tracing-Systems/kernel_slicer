#include "test_class.h"

#include <chrono>
#include <cmath>
#include <string>

void InstancedQuads::SetViewport(int a_width, int a_height)
{
  m_quad = { float4(-1.0f, -1.0f, 0.0f, 1.0f), float4( 1.0f, -1.0f, 0.0f, 1.0f), float4( 1.0f, 1.0f, 0.0f, 1.0f),
             float4(-1.0f, -1.0f, 0.0f, 1.0f), float4( 1.0f,  1.0f, 0.0f, 1.0f), float4(-1.0f, 1.0f, 0.0f, 1.0f) };

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::NONE;
  m_pipeline.depthTest  = true;
  m_pipeline.clearColor = float4(0.1f, 0.1f, 0.15f, 1.0f);
}

void InstancedQuads::Render(int a_width, int a_height, float a_angle, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  LiteRS::Clear(m_pipeline);
  LiteRS::DrawInstancedTriangles(m_pipeline, this, &InstancedQuads::kernelVS_Hole, uint32_t(m_quad.size()), HOLES*HOLES);
  LiteRS::DrawInstancedTriangles(m_pipeline, this, &InstancedQuads::kernelVS_Quad, &InstancedQuads::kernelPS_Quad, uint32_t(m_quad.size()), GRID_SIZE*GRID_SIZE, a_angle);
  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

float4 InstancedQuads::kernelVS_Quad(uint32_t a_vertId, uint32_t a_instId, float a_angle, ColorVaryings* a_out) const
{
  const uint32_t x     = a_instId % GRID_SIZE;
  const uint32_t y     = a_instId / GRID_SIZE;
  const float    cellW = 2.0f/float(GRID_SIZE);
  const float    angle = a_angle*float(a_instId % 5 + 1);
  const float    c     = std::cos(angle);
  const float    s     = std::sin(angle);
  const float4   p     = m_quad[a_vertId]*(0.35f*cellW);
  const float    cx    = -1.0f + cellW*(float(x) + 0.5f);
  const float    cy    = -1.0f + cellW*(float(y) + 0.5f);
  a_out->color = float4(float(x)/float(GRID_SIZE - 1), float(y)/float(GRID_SIZE - 1), (a_vertId < 3) ? 1.0f : 0.3f, 1.0f);
  return float4(cx + c*p.x - s*p.y, cy + s*p.x + c*p.y, 0.5f, 1.0f);
}

float4 InstancedQuads::kernelVS_Hole(uint32_t a_vertId, uint32_t a_instId) const
{
  const float4 p  = m_quad[a_vertId]*0.2f;
  const float  cx = (a_instId % HOLES == 0) ? -0.5f : 0.5f;
  const float  cy = (a_instId / HOLES == 0) ? -0.5f : 0.5f;
  return float4(cx + p.x, cy + p.y, 0.25f, 1.0f);
}

float4 InstancedQuads::kernelPS_Quad(const ColorVaryings& a_in) const
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

void InstancedQuads::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[int2(x, y)]);
  }
}

void InstancedQuads::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
