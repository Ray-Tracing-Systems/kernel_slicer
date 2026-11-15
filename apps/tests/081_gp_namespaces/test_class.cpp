#include "test_class.h"

#include <chrono>
#include <cmath>
#include <string>

void NamespaceRender::SetViewport(int a_width, int a_height)
{
  m_quadPos     = { LiteMath::float4(-0.8f, -0.8f, 0.0f, 1.0f), LiteMath::float4(0.8f, -0.8f, 0.0f, 1.0f),
                    LiteMath::float4( 0.8f,  0.8f, 0.0f, 1.0f), LiteMath::float4(-0.8f, 0.8f, 0.0f, 1.0f) };
  m_quadColors  = { float4(1.0f, 0.2f, 0.2f, 1.0f), float4(0.2f, 1.0f, 0.2f, 1.0f), float4(0.2f, 0.2f, 1.0f, 1.0f), float4(1.0f, 1.0f, 0.2f, 1.0f) };
  m_quadIndices = { 0, 1, 2, 0, 2, 3 };
  m_triPos      = { float4(0.0f, -0.5f, 0.3f, 1.0f), float4(0.5f, 0.4f, 0.3f, 1.0f), float4(-0.5f, 0.4f, 0.3f, 1.0f) };

  const LiteMath::float4x4 view = LiteMath::lookAt(float3(0.0f, 0.0f, 3.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));
  const float4x4           proj = OpenglToVulkanProjectionMatrixFix()*perspectiveMatrix(45.0f, float(a_width)/float(a_height), 0.1f, 100.0f);
  m_viewProj = proj*view;

  m_color.resize(a_width, a_height);
  m_depth.resize(a_width, a_height);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::NONE;
  m_pipeline.depthTest  = true;
  m_pipeline.clearColor = float4(0.1f, 0.1f, 0.15f, 1.0f);
}

void NamespaceRender::Render(int a_width, int a_height, float a_angle, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  Clear(m_pipeline); // argument dependent lookup: 'm_pipeline' is 'LiteRS::Pipeline'
  DrawIndexedTriangles(m_pipeline, this, &NamespaceRender::kernelVS_Quad, &NamespaceRender::kernelPS_Quad,
                       m_quadIndices.data(), uint32_t(m_quadIndices.size()), a_angle);                           // using-declaration
  LiteRS::DrawTriangles(m_pipeline, this, &NamespaceRender::kernelVS_Tri, &NamespaceRender::kernelPS_Tri, 3, a_angle); // explicit namespace
  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

LiteMath::float4 NamespaceRender::kernelVS_Quad(uint32_t a_vertId, float a_angle, QuadVaryings* a_out) const
{
  const LiteMath::float4x4 model = LiteMath::rotate4x4Y(0.5f*a_angle)*LiteMath::rotate4x4X(0.25f*LiteMath::M_PI*a_angle);
  const LiteMath::float4   pos   = m_quadPos[a_vertId];
  a_out->color = LiteMath::clamp(m_quadColors[a_vertId], LiteMath::float4(0.0f), LiteMath::float4(1.0f));
  a_out->uv    = LiteMath::float2(pos.x, pos.y)*0.5f + LiteMath::float2(0.5f, 0.5f);
  return m_viewProj*(model*pos);
}

float4 NamespaceRender::kernelPS_Quad(const QuadVaryings& a_in) const
{
  const float checker = (int(a_in.uv.x*8.0f) + int(a_in.uv.y*8.0f)) % 2 == 0 ? 1.0f : 0.6f;
  const float shade   = lm::clamp(lm::dot(lm::normalize(float3(a_in.uv.x, a_in.uv.y, 1.0f)), float3(0.0f, 0.0f, 1.0f)), 0.0f, 1.0f);
  return lerp(a_in.color, float4(1.0f, 1.0f, 1.0f, 1.0f), 0.2f)*(checker*shade);
}

float4 NamespaceRender::kernelVS_Tri(uint32_t a_vertId, float a_angle, TriVaryings* a_out) const
{
  const float4x4 model = rotate4x4Z(a_angle);
  const float4   pos   = m_triPos[a_vertId];
  a_out->color = float4(a_vertId == 0 ? 1.0f : 0.0f, a_vertId == 1 ? 1.0f : 0.0f, a_vertId == 2 ? 1.0f : 0.0f, 1.0f);
  return m_viewProj*(model*pos);
}

LiteMath::float4 NamespaceRender::kernelPS_Tri(const TriVaryings& a_in) const
{
  return clamp(a_in.color*1.5f, 0.0f, 1.0f);
}

static inline uint32_t PackRGB8(LiteMath::float4 a_color) // R in the lowest byte, gamma 2.2
{
  const uint32_t r = uint32_t(LiteMath::clamp(std::pow(a_color.x, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = uint32_t(clamp(std::pow(a_color.y, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = uint32_t(lm::clamp(std::pow(a_color.z, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  return r | (g << 8) | (b << 16);
}

void NamespaceRender::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[LiteMath::int2(x, y)]);
  }
}

void NamespaceRender::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
