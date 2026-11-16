#include "shadow_map.h"
#include "simple_meshes.h"

#include <chrono>
#include <cmath>
#include <string>

using LiteRS::Clear;
using LiteRS::DrawIndexedTriangles;

void ShadowRender::SetViewport(int a_width, int a_height)
{
  const SimpleMeshes::Mesh scene = SimpleMeshes::CreateTestScene();
  m_positions = scene.positions;
  m_normals   = scene.normals;
  m_colors    = scene.colors;
  m_indices   = scene.indices;

  const float4x4 view = LiteMath::lookAt(float3(0.0f, 4.5f, 8.0f), float3(0.0f, 0.5f, 0.0f), float3(0.0f, 1.0f, 0.0f));
  const float4x4 proj = LiteMath::OpenglToVulkanProjectionMatrixFix()*LiteMath::perspectiveMatrix(45.0f, float(a_width)/float(a_height), 0.1f, 100.0f);
  m_viewProj = proj*view;

  const float3   lightDir  = normalize(float3(-0.5f, 1.0f, 0.4f));
  const float4x4 lightView = LiteMath::lookAt(lightDir*10.0f, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));
  const float4x4 lightProj = LiteMath::OpenglToVulkanProjectionMatrixFix()*LiteMath::ortoMatrix(-7.0f, 7.0f, -7.0f, 7.0f, 1.0f, 20.0f);
  m_lightViewProj = lightProj*lightView;
  m_lightDir      = float4(lightDir.x, lightDir.y, lightDir.z, 0.0f);

  m_shadowMap.resize(m_shadowSize, m_shadowSize, LiteImage::Layout::TILED4x4);
  m_shadowPass.SetDepthTarget(m_shadowMap);
  m_shadowPass.SetViewport(0, 0, m_shadowSize, m_shadowSize);
  m_shadowPass.cullMode = LiteRS::CullMode::FRONT; // back faces of closed objects in the shadow map remove shadow acne on lit faces

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_mainPass.SetColorTarget(m_color);
  m_mainPass.SetDepthTarget(m_depth);
  m_mainPass.SetViewport(0, 0, a_width, a_height);
  m_mainPass.cullMode   = LiteRS::CullMode::BACK;
  m_mainPass.clearColor = float4(0.35f, 0.45f, 0.6f, 1.0f);
}

void ShadowRender::Render(int a_width, int a_height, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  Clear(m_shadowPass);
  DrawIndexedTriangles(m_shadowPass, this, 
                       &ShadowRender::kernelVS_Shadow, 
                       m_indices.data(), uint32_t(m_indices.size()));

  Clear(m_mainPass);
  DrawIndexedTriangles(m_mainPass, this, 
                       &ShadowRender::kernelVS_Main, &ShadowRender::kernelPS_Main, 
                       m_indices.data(), uint32_t(m_indices.size()));

  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

float4 ShadowRender::kernelVS_Shadow(uint32_t a_vertId) const
{
  return m_lightViewProj*m_positions[a_vertId];
}

float4 ShadowRender::kernelVS_Main(uint32_t a_vertId, ShadowVaryings* a_out) const
{
  const float4 pos = m_positions[a_vertId];
  a_out->normal    = m_normals[a_vertId];
  a_out->color     = m_colors[a_vertId];
  a_out->lightPos  = m_lightViewProj*pos;
  return m_viewProj*pos;
}

float ShadowRender::ShadowFactor(float4 a_lightPos, float a_bias) const
{
  const float sx    = (a_lightPos.x*0.5f + 0.5f)*float(m_shadowSize);   // NDC y = -1 is the top row, as for the main camera
  const float sy    = (a_lightPos.y*0.5f + 0.5f)*float(m_shadowSize);
  const float depth = a_lightPos.z - a_bias;
  float lit = 0.0f;
  for(int dy = -1; dy <= 1; dy++)
  {
    for(int dx = -1; dx <= 1; dx++)
    {
      const int px = clamp(int(sx) + dx, 0, m_shadowSize - 1);
      const int py = clamp(int(sy) + dy, 0, m_shadowSize - 1);
      lit += (depth <= m_shadowMap[int2(px, py)]) ? 1.0f : 0.0f;
    }
  }
  return lit*(1.0f/9.0f);
}

float4 ShadowRender::kernelPS_Main(const ShadowVaryings& a_in) const
{
  const float4 n       = normalize(a_in.normal);
  const float  cosL    = dot(n, m_lightDir);
  const float  diffuse = std::max(cosL, 0.0f);
  const float  bias    = 0.002f;
  const float  shadow  = (diffuse > 0.0f) ? ShadowFactor(a_in.lightPos, bias) : 0.0f;
  const float4 color   = a_in.color*(0.15f + 0.85f*diffuse*shadow);
  return float4(color.x, color.y, color.z, 1.0f);
}

static inline uint32_t PackRGB8(float4 a_color) // R in the lowest byte, gamma 2.2
{
  const uint32_t r = uint32_t(clamp(std::pow(a_color.x, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = uint32_t(clamp(std::pow(a_color.y, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = uint32_t(clamp(std::pow(a_color.z, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  return r | (g << 8) | (b << 16);
}

void ShadowRender::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[int2(x, y)]);
  }
}

void ShadowRender::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
