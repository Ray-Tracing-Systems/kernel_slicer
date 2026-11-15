#include "deferred_ao.h"
#include "simple_meshes.h"

#include <chrono>
#include <cmath>
#include <string>

void DeferredRender::SetViewport(int a_width, int a_height)
{
  const SimpleMeshes::Mesh scene = SimpleMeshes::CreateTestScene();
  m_positions = scene.positions;
  m_normals   = scene.normals;
  m_colors    = scene.colors;
  m_indices   = scene.indices;

  m_view = LiteMath::lookAt(float3(0.0f, 4.5f, 8.0f), float3(0.0f, 0.5f, 0.0f), float3(0.0f, 1.0f, 0.0f));
  m_proj = LiteMath::OpenglToVulkanProjectionMatrixFix()*LiteMath::perspectiveMatrix(45.0f, float(a_width)/float(a_height), 0.1f, 100.0f);
  const float3 lightDir = normalize(float3(-0.5f, 1.0f, 0.4f));
  m_lightDirView        = normalize(m_view*float4(lightDir.x, lightDir.y, lightDir.z, 0.0f));

  uint32_t state = 7; // AO samples in the hemisphere, denser near the center; xorshift keeps the sequence the same on all platforms
  auto u01 = [&state]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return float(state & 0xFFFFFF)/16777216.0f; };
  m_aoKernel.resize(AO_SAMPLES);
  for(int i = 0; i < AO_SAMPLES; i++)
  {
    float4 dir;
    do { dir = float4(2.0f*u01() - 1.0f, 2.0f*u01() - 1.0f, u01(), 0.0f); } while(dot(dir, dir) > 1.0f || dot(dir, dir) < 1e-4f);
    const float scale = 0.1f + 0.9f*float(i*i)/float(AO_SAMPLES*AO_SAMPLES);
    m_aoKernel[i] = dir*scale;
  }

  const LiteImage::Layout layout = LiteImage::Layout::TILED4x4;
  m_gPos.resize     (a_width, a_height, layout);
  m_gNormal.resize  (a_width, a_height, layout);
  m_gAlbedo.resize  (a_width, a_height, layout);
  m_gDepth.resize   (a_width, a_height, layout);
  m_ao.resize       (a_width, a_height, layout);
  m_aoBlurred.resize(a_width, a_height, layout);

  m_gbufferPass.SetColorTarget(m_gPos,    0);
  m_gbufferPass.SetColorTarget(m_gNormal, 1);
  m_gbufferPass.SetColorTarget(m_gAlbedo, 2);
  m_gbufferPass.SetDepthTarget(m_gDepth);
  m_gbufferPass.SetViewport(0, 0, a_width, a_height);
  m_gbufferPass.cullMode   = LiteRS::CullMode::BACK;
  m_gbufferPass.clearColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

void DeferredRender::Render(int a_width, int a_height, uint32_t* a_out, uint32_t* a_outAO)
{
  auto start = std::chrono::high_resolution_clock::now();

  LiteRS::Clear(m_gbufferPass);
  LiteRS::DrawIndexedTriangles(m_gbufferPass, this, &DeferredRender::kernelVS_GBuffer, &DeferredRender::kernelPS_GBuffer, m_indices.data(), uint32_t(m_indices.size()));

  kernel2D_SSAO  (a_width, a_height);
  kernel2D_BlurAO(a_width, a_height);
  kernel2D_Shade (a_width, a_height, a_out, a_outAO);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

float4 DeferredRender::kernelVS_GBuffer(uint32_t a_vertId, GBufferVaryings* a_out) const
{
  a_out->posView    = m_view*m_positions[a_vertId];
  a_out->normalView = m_view*m_normals[a_vertId];
  a_out->color      = m_colors[a_vertId];
  return m_proj*a_out->posView;
}

GBufferOut DeferredRender::kernelPS_GBuffer(const GBufferVaryings& a_in) const
{
  GBufferOut res;
  res.posView    = a_in.posView;
  res.normalView = normalize(a_in.normalView);
  res.albedo     = float4(a_in.color.x, a_in.color.y, a_in.color.z, 1.0f);
  return res;
}

static inline float HashToAngle(int x, int y) // per pixel rotation of AO samples
{
  uint32_t h = uint32_t(x)*1973u + uint32_t(y)*9277u + 26699u;
  h = (h ^ (h >> 13))*0x5bd1e995u;
  h = h ^ (h >> 15);
  return float(h & 0xFFFF)*(6.2831853f/65536.0f);
}

void DeferredRender::kernel2D_SSAO(int a_width, int a_height)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      float ao = 1.0f;
      if(m_gAlbedo[int2(x, y)].w > 0.0f)
      {
        const float4 p     = m_gPos[int2(x, y)];
        const float4 n     = m_gNormal[int2(x, y)];
        const float  angle = HashToAngle(x, y);
        const float4 rvec  = float4(std::cos(angle), std::sin(angle), 0.0f, 0.0f);
        const float4 t     = normalize(rvec - n*dot(rvec, n));
        const float4 b     = cross(n, t);

        float occlusion = 0.0f;
        for(int i = 0; i < AO_SAMPLES; i++)
        {
          const float4 k = m_aoKernel[i];
          const float4 s = p + (t*k.x + b*k.y + n*k.z)*m_aoRadius;
          const float4 c = m_proj*float4(s.x, s.y, s.z, 1.0f);
          const int    sx = int((c.x/c.w*0.5f + 0.5f)*float(a_width));
          const int    sy = int((c.y/c.w*0.5f + 0.5f)*float(a_height));
          if(sx < 0 || sy < 0 || sx >= a_width || sy >= a_height || m_gAlbedo[int2(sx, sy)].w == 0.0f)
            continue;
          const float sceneZ = m_gPos[int2(sx, sy)].z;                       // view space z is negative in front of the camera
          const float range  = clamp(m_aoRadius/std::abs(p.z - sceneZ), 0.0f, 1.0f);
          occlusion += (sceneZ >= s.z + 0.02f) ? range : 0.0f;
        }
        ao = clamp(1.0f - 2.0f*occlusion/float(AO_SAMPLES), 0.0f, 1.0f);
      }
      m_ao[int2(x, y)] = ao;
    }
  }
}

void DeferredRender::kernel2D_BlurAO(int a_width, int a_height) // 4x4 box filter, removes the noise of sample rotations
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      float summ = 0.0f;
      for(int dy = -2; dy < 2; dy++)
        for(int dx = -2; dx < 2; dx++)
          summ += m_ao[int2(clamp(x + dx, 0, a_width - 1), clamp(y + dy, 0, a_height - 1))];
      m_aoBlurred[int2(x, y)] = summ*(1.0f/16.0f);
    }
  }
}

static inline uint32_t PackRGB8(float4 a_color) // R in the lowest byte, gamma 2.2
{
  const uint32_t r = uint32_t(clamp(std::pow(a_color.x, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = uint32_t(clamp(std::pow(a_color.y, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = uint32_t(clamp(std::pow(a_color.z, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  return r | (g << 8) | (b << 16);
}

void DeferredRender::kernel2D_Shade(int a_width, int a_height, uint32_t* a_out, uint32_t* a_outAO)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      const float4 albedo = m_gAlbedo[int2(x, y)];
      const float  ao     = m_aoBlurred[int2(x, y)];
      float4 color = float4(0.35f, 0.45f, 0.6f, 1.0f);
      if(albedo.w > 0.0f)
      {
        const float diffuse = std::max(dot(m_gNormal[int2(x, y)], m_lightDirView), 0.0f);
        color = albedo*(0.5f*ao + 0.5f*diffuse*ao);
      }
      a_out  [y*a_width + x] = PackRGB8(color);
      a_outAO[y*a_width + x] = PackRGB8(float4(ao, ao, ao, 1.0f));
    }
  }
}

void DeferredRender::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
