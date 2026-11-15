#include "test_class.h"

#include <chrono>
#include <cmath>
#include <string>

void TriangleRender::SetViewport(int a_width, int a_height)
{
  m_positions = { float4( 0.0f, -0.75f, 0.5f, 1.0f), float4(0.75f, 0.6f, 0.5f, 1.0f), float4(-0.75f, 0.6f, 0.5f, 1.0f) };
  m_colors    = { float4( 1.0f,  0.0f,  0.0f, 1.0f), float4(0.0f,  1.0f, 0.0f, 1.0f), float4( 0.0f,  0.0f, 1.0f, 1.0f) };

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::NONE;
  m_pipeline.depthTest  = true;
  m_pipeline.clearColor = float4(0.1f, 0.1f, 0.15f, 1.0f);
}

void TriangleRender::Render(int a_width, int a_height, float a_angle, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  LiteRS::Clear(m_pipeline);
  LiteRS::DrawTriangles(m_pipeline, this, &TriangleRender::kernelVS_Color, &TriangleRender::kernelPS_Color, 3, a_angle);
  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

float4 TriangleRender::kernelVS_Color(uint32_t a_vertId, float a_angle, ColorVaryings* a_out) const
{
  const float4 p = m_positions[a_vertId];
  const float  c = std::cos(a_angle);
  const float  s = std::sin(a_angle);
  a_out->color = m_colors[a_vertId];
  return float4(c*p.x - s*p.y, s*p.x + c*p.y, p.z, 1.0f);
}

float4 TriangleRender::kernelPS_Color(const ColorVaryings& a_in) const
{
  return a_in.color;
}

static inline uint32_t PackRGB8(float4 a_color) // R in the lowest byte, gamma 2.2
{
  const uint32_t r = uint32_t(clamp(std::pow(a_color.x, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = uint32_t(clamp(std::pow(a_color.y, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = uint32_t(clamp(std::pow(a_color.z, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  return r | (g << 8) | (b << 16);
}

void TriangleRender::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[int2(x, y)]);
  }
}

void TriangleRender::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
