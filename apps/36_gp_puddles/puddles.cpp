#include "puddles.h"

#include <chrono>
#include <cmath>
#include <string>

// render targets of a pipeline are set in the file processed by kslicer, the scene is loaded in 'puddles_scene.cpp'
//
void PuddleRender::SetViewport(int a_width, int a_height)
{
  m_viewProj = LiteMath::OpenglToVulkanProjectionMatrixFix()*LiteMath::perspectiveMatrix(m_camFov, float(a_width)/float(a_height), m_camNear, m_camFar)*m_camView;

  const LiteImage::Layout layout = LiteImage::Layout::TILED4x4;
  m_gPos.resize   (a_width, a_height, layout);
  m_gNormal.resize(a_width, a_height, layout);
  m_gAlbedo.resize(a_width, a_height, layout);
  m_gDepth.resize (a_width, a_height, layout);

  m_gbufferPass.SetColorTarget(m_gPos,    0);
  m_gbufferPass.SetColorTarget(m_gNormal, 1);
  m_gbufferPass.SetColorTarget(m_gAlbedo, 2);
  m_gbufferPass.SetDepthTarget(m_gDepth);
  m_gbufferPass.SetViewport(0, 0, a_width, a_height);
  m_gbufferPass.cullMode   = LiteRS::CullMode::NONE; // the plane and some meshes are not closed
  m_gbufferPass.clearColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// control functions ////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PuddleRender::RenderGBuffer(int a_width, int a_height)
{
  auto start = std::chrono::high_resolution_clock::now();

  LiteRS::Clear(m_gbufferPass);
  for(uint32_t mesh = 0; mesh < m_meshNum; mesh++) // all instances of a mesh are drawn by one command
  {
    const uint32_t firstIndex = m_meshFirstIndex[mesh];
    const uint32_t indexNum   = m_meshIndexNum[mesh];
    const uint32_t instNum    = m_meshInstNum[mesh];
    LiteRS::DrawIndexedInstancedTriangles(m_gbufferPass, this, &PuddleRender::kernelVS_GBuffer, &PuddleRender::kernelPS_GBuffer,
                                          m_indices.data() + firstIndex, indexNum, instNum, m_meshVertOffset[mesh], m_meshFirstInst[mesh]);
  }

  m_timeGBuffer = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void PuddleRender::ShadeAndReflect(int a_width, int a_height, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();
  kernel2D_Shade(a_width, a_height, a_out);
  m_timeShade = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// shaders and kernels //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float4 PuddleRender::kernelVS_GBuffer(uint32_t a_vertId, uint32_t a_instId, uint32_t a_vertOffset, uint32_t a_firstInst, GBufferVaryings* a_out) const
{
  const uint32_t inst = a_firstInst + a_instId;
  const float4   p    = m_positions[a_vertOffset + a_vertId];
  const float4   n    = m_normals  [a_vertOffset + a_vertId];
  a_out->pos    = m_instMatrix[inst]*float4(p.x, p.y, p.z, 1.0f);
  a_out->normal = m_instNormalMatrix[inst]*float4(n.x, n.y, n.z, 0.0f);
  a_out->color  = m_instColor[inst];
  return m_viewProj*a_out->pos;
}

GBufferOut PuddleRender::kernelPS_GBuffer(const GBufferVaryings& a_in) const
{
  const float4 n = normalize(float4(a_in.normal.x, a_in.normal.y, a_in.normal.z, 0.0f));
  GBufferOut res;
  res.pos    = a_in.pos;
  res.normal = float4(n.x, n.y, n.z, a_in.color.w);
  res.albedo = float4(a_in.color.x, a_in.color.y, a_in.color.z, 1.0f);
  return res;
}

static inline float Hash2D(int x, int y)
{
  uint32_t h = uint32_t(x)*374761393u + uint32_t(y)*668265263u;
  h = (h ^ (h >> 13))*1274126177u;
  h = h ^ (h >> 16);
  return float(h & 0xFFFFFFu)*(1.0f/16777216.0f);
}

static inline float ValueNoise(float x, float y)
{
  const float fx = std::floor(x);
  const float fy = std::floor(y);
  const int   ix = int(fx);
  const int   iy = int(fy);
  const float tx = x - fx;
  const float ty = y - fy;
  const float sx = tx*tx*(3.0f - 2.0f*tx);
  const float sy = ty*ty*(3.0f - 2.0f*ty);
  const float a  = Hash2D(ix,     iy);
  const float b  = Hash2D(ix + 1, iy);
  const float c  = Hash2D(ix,     iy + 1);
  const float d  = Hash2D(ix + 1, iy + 1);
  return (a + (b - a)*sx) + ((c + (d - c)*sx) - (a + (b - a)*sx))*sy;
}

static inline float SmoothStep(float a_edge0, float a_edge1, float a_x)
{
  const float t = clamp((a_x - a_edge0)/(a_edge1 - a_edge0), 0.0f, 1.0f);
  return t*t*(3.0f - 2.0f*t);
}

static inline uint32_t PackRGB8(float4 a_color) // R in the lowest byte, gamma 2.2
{
  const uint32_t r = uint32_t(clamp(std::pow(a_color.x, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = uint32_t(clamp(std::pow(a_color.y, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = uint32_t(clamp(std::pow(a_color.z, 1.0f/2.2f), 0.0f, 1.0f)*255.0f + 0.5f);
  return r | (g << 8) | (b << 16);
}

float PuddleRender::PuddleMask(float a_x, float a_z) const
{
  const float x = a_x*m_puddleScale;
  const float z = a_z*m_puddleScale;
  const float n = 0.6f*ValueNoise(x, z) + 0.3f*ValueNoise(2.0f*x + 17.0f, 2.0f*z + 5.0f) + 0.1f*ValueNoise(4.0f*x + 3.0f, 4.0f*z + 11.0f);
  return SmoothStep(0.45f, 0.50f, n);
}

float4 PuddleRender::SkyColor(float4 a_dir) const
{
  const float t = clamp(a_dir.y, 0.0f, 1.0f);
  return m_skyColor*(1.0f - 0.4f*t) + float4(0.25f, 0.35f, 0.6f, 0.0f)*t;
}

float PuddleRender::SunVisibility(float4 a_pos, float4 a_normal) const
{
  const float4 origin = a_pos + a_normal*1e-3f;
  const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(float4(origin.x, origin.y, origin.z, 1e-3f), float4(m_sunDir.x, m_sunDir.y, m_sunDir.z, m_sceneSize));
  return inShadow ? 0.0f : 1.0f;
}

float4 PuddleRender::HitColor(CRT_Hit a_hit, float4 a_rayDir) const
{
  const uint32_t inst = a_hit.instId;
  const uint32_t mesh = m_instMesh[inst];
  const uint32_t base = m_meshFirstIndex[mesh] + a_hit.primId*3;
  const uint32_t vOff = m_meshVertOffset[mesh];
  const float4   nA   = m_normals[vOff + m_indices[base + 0]];
  const float4   nB   = m_normals[vOff + m_indices[base + 1]];
  const float4   nC   = m_normals[vOff + m_indices[base + 2]];
  const float    u    = a_hit.coords[1];
  const float    v    = a_hit.coords[0];
  const float4   nObj = nA*(1.0f - u - v) + nB*u + nC*v;
  float4 n = normalize(m_instNormalMatrix[inst]*float4(nObj.x, nObj.y, nObj.z, 0.0f));
  if(dot(n, a_rayDir) > 0.0f)
    n = n*(-1.0f);
  const float lambert = std::max(dot(n, m_sunDir), 0.0f);
  return m_instColor[inst]*(m_skyColor*0.35f + m_sunColor*lambert);
}

void PuddleRender::kernel2D_Shade(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      const float4 albedo = m_gAlbedo[int2(x, y)];
      float4 color = m_skyColor;
      if(albedo.w > 0.0f)
      {
        const float4 pos     = m_gPos[int2(x, y)];
        const float4 gn      = m_gNormal[int2(x, y)];
        const float4 n       = float4(gn.x, gn.y, gn.z, 0.0f);
        const float  lambert = std::max(dot(n, m_sunDir), 0.0f);
        const float  sun     = (lambert > 0.0f) ? SunVisibility(pos, n) : 0.0f;
        color = albedo*(m_skyColor*0.35f + m_sunColor*(sun*lambert));

        const float puddle = (gn.w > 0.5f) ? PuddleMask(pos.x, pos.z) : 0.0f;
        if(puddle > 0.0f) // reflection of the scene in water: a ray from the ground in the mirror direction
        {
          const float4 v       = normalize(float4(pos.x - m_camPos.x, pos.y - m_camPos.y, pos.z - m_camPos.z, 0.0f));
          const float4 r       = float4(v.x, -v.y, v.z, 0.0f);
          const CRT_Hit hit    = m_pAccelStruct->RayQuery_NearestHit(float4(pos.x, pos.y + 1e-3f, pos.z, 1e-3f), float4(r.x, r.y, r.z, m_sceneSize));
          const float4 refl    = (hit.primId != 0xFFFFFFFFu) ? HitColor(hit, r) : SkyColor(r);
          const float  c       = 1.0f - std::max(-v.y, 0.0f);
          const float  fresnel = 0.3f + 0.7f*c*c*c*c*c; // more than for real water, to make puddles visible
          const float4 wet     = color*0.3f;
          const float4 water   = wet*(1.0f - fresnel) + refl*fresnel;
          color = color*(1.0f - puddle) + water*puddle;
        }
      }
      a_out[y*a_width + x] = PackRGB8(color);
    }
  }
}

void PuddleRender::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "RenderGBuffer")
    a_out[0] = m_timeGBuffer;
  else if(std::string(a_funcName) == "ShadeAndReflect")
    a_out[0] = m_timeShade;
}
