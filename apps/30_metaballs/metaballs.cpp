#include "metaballs.h"
#include <algorithm>
#include <chrono>
#include <cmath>

using LiteMath::dot;
using LiteMath::cross;
using LiteMath::normalize;
using LiteMath::reflect;
using LiteMath::clamp;
using LiteMath::to_float3;
using LiteMath::to_float4;

// NOTE for the Slang back end: class members and kernel arguments are never written directly
// inside std:: calls; they are copied to locals first (see the kernel-slicer skill).

static inline float  sqr(float x) { return x*x; }
static inline float  luminance(float3 c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

static inline uint32_t PackRGBA8(float3 c)
{
  const uint32_t r = (uint32_t)(clamp(c.x, 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = (uint32_t)(clamp(c.y, 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = (uint32_t)(clamp(c.z, 0.0f, 1.0f)*255.0f + 0.5f);
  return 0xFF000000 | (b << 16) | (g << 8) | r;
}

//////////////////////////////////////////////////////////////////////////////////////////////////// host-only setup

Metaballs::Metaballs()
{
  const float3 camPos = float3(0.0f, 1.3f, 5.2f);
  const float3 target = float3(0.0f, -0.1f, 0.0f);
  const float  fov    = 42.0f*3.14159265f/180.0f;
  const float  scale  = std::tan(0.5f*fov);
  const float3 fwd    = normalize(target - camPos);
  const float3 right  = normalize(cross(fwd, float3(0.0f, 1.0f, 0.0f)));
  const float3 up     = cross(right, fwd);
  m_camPos     = to_float4(camPos, 0.0f);
  m_camForward = to_float4(fwd, 0.0f);
  m_camRight   = to_float4(right*scale, 0.0f);
  m_camUp      = to_float4(up*scale, 0.0f);
  m_lightDir   = to_float4(normalize(float3(-0.5f, 1.0f, 0.6f)), 0.0f);

  // default scene: 7 balls arranged in a loose ring around a central blob
  std::vector<float4> pos, col;
  pos.push_back(float4( 0.00f, 0.10f,  0.00f, 1.30f)); col.push_back(float4(0.95f, 0.95f, 0.95f, 0.70f)); // chrome-like center
  for(int i = 0; i < 6; i++)
  {
    const float a = float(i)*(2.0f*3.14159265f/6.0f);
    const float r = (i % 2 == 0) ? 1.25f : 1.05f;
    const float y = (i % 2 == 0) ? -0.15f : 0.45f;
    pos.push_back(float4(r*std::cos(a), y, r*std::sin(a), (i % 2 == 0) ? 0.95f : 0.80f));
  }
  col.push_back(float4(0.90f, 0.15f, 0.10f, 0.35f));
  col.push_back(float4(0.95f, 0.75f, 0.10f, 0.35f));
  col.push_back(float4(0.15f, 0.80f, 0.20f, 0.35f));
  col.push_back(float4(0.10f, 0.70f, 0.90f, 0.35f));
  col.push_back(float4(0.25f, 0.20f, 0.95f, 0.35f));
  col.push_back(float4(0.85f, 0.20f, 0.85f, 0.35f));
  SetBalls(pos, col);
}

void Metaballs::SetBalls(const std::vector<float4>& a_posRadius, const std::vector<float4>& a_colors)
{
  m_ballsNum      = std::min(int(a_posRadius.size()), MAX_BALLS);
  m_ballPosRadius = a_posRadius;
  m_ballColor     = a_colors;
  m_ballPosRadius.resize(m_ballsNum);
  m_ballColor.resize(m_ballsNum);
}

void Metaballs::SetMaxImageSize(int w, int h)
{
  m_hdr.resize(size_t(w)*size_t(h));
}

//////////////////////////////////////////////////////////////////////////////////////////////////// helpers (GPU code)

// Wyvill-like kernel with compact support: k(d) = (1 - d^2/R^2)^3 for d < R, 0 otherwise
float Metaballs::Field(float3 p) const
{
  float summ = 0.0f;
  for(int i = 0; i < m_ballsNum; i++)
  {
    const float4 b  = m_ballPosRadius[i];
    const float3 d  = p - to_float3(b);
    const float  s  = dot(d, d)/(b.w*b.w);
    if(s < 1.0f)
      summ += (1.0f - s)*(1.0f - s)*(1.0f - s);
  }
  return summ;
}

float3 Metaballs::FieldNormal(float3 p) const
{
  float3 grad = float3(0.0f, 0.0f, 0.0f);
  for(int i = 0; i < m_ballsNum; i++)
  {
    const float4 b  = m_ballPosRadius[i];
    const float3 d  = p - to_float3(b);
    const float  r2 = b.w*b.w;
    const float  s  = dot(d, d)/r2;
    if(s < 1.0f)
      grad += d*(-6.0f*(1.0f - s)*(1.0f - s)/r2);  // d/dp (1-s)^3
  }
  return normalize(grad*(-1.0f));
}

float4 Metaballs::SurfaceColor(float3 p) const
{
  float4 colorSumm = float4(0.0f, 0.0f, 0.0f, 0.0f);
  float  weightSum = 0.0f;
  for(int i = 0; i < m_ballsNum; i++)
  {
    const float4 b = m_ballPosRadius[i];
    const float3 d = p - to_float3(b);
    const float  s = dot(d, d)/(b.w*b.w);
    if(s < 1.0f)
    {
      const float wgt = (1.0f - s)*(1.0f - s)*(1.0f - s);
      colorSumm += m_ballColor[i]*wgt;
      weightSum += wgt;
    }
  }
  return (weightSum > 0.0f) ? colorSumm*(1.0f/weightSum) : float4(0.5f, 0.5f, 0.5f, 0.0f);
}

float2 Metaballs::InfluenceInterval(float3 ro, float3 rd) const
{
  float tMin = 1e30f;
  float tMax = -1e30f;
  for(int i = 0; i < m_ballsNum; i++)
  {
    const float4 b    = m_ballPosRadius[i];
    const float3 oc   = ro - to_float3(b);
    const float  bb   = dot(oc, rd);
    const float  c    = dot(oc, oc) - b.w*b.w;
    const float  disc = bb*bb - c;
    if(disc > 0.0f)
    {
      const float sq = std::sqrt(disc);
      tMin = std::min(tMin, -bb - sq);
      tMax = std::max(tMax, -bb + sq);
    }
  }
  return float2(std::max(tMin, 0.0f), tMax);
}

// uniform marching inside the influence interval + bisection refinement at the first sign change
float Metaballs::TraceBalls(float3 ro, float3 rd, float tLimit) const
{
  const float  threshold = m_threshold;
  const float2 span      = InfluenceInterval(ro, rd);
  const float  tEnd      = std::min(span.y, tLimit);
  if(span.x >= tEnd)
    return -1.0f;

  const float dt    = (tEnd - span.x)/float(MARCH_STEPS);
  float       tPrev = span.x;
  float       fPrev = Field(ro + rd*tPrev) - threshold;
  float       tHit  = -1.0f;
  for(int step = 1; step <= MARCH_STEPS; step++)
  {
    const float t = span.x + dt*float(step);
    const float f = Field(ro + rd*t) - threshold;
    if(fPrev < 0.0f && f >= 0.0f)
    {
      float a = tPrev, bnd = t;
      for(int k = 0; k < REFINE_STEPS; k++)
      {
        const float mid = 0.5f*(a + bnd);
        if(Field(ro + rd*mid) >= threshold) bnd = mid;
        else                                a   = mid;
      }
      tHit = bnd;
      break;
    }
    tPrev = t;
    fPrev = f;
  }
  return tHit;
}

float3 Metaballs::Sky(float3 rd) const
{
  const float3 lightDir = to_float3(m_lightDir);
  const float  up       = clamp(rd.y*0.5f + 0.5f, 0.0f, 1.0f);
  const float3 horizon  = float3(0.85f, 0.80f, 0.75f);
  const float3 zenith   = float3(0.20f, 0.40f, 0.80f);
  const float  sunDot   = std::max(dot(rd, lightDir), 0.0f);
  const float  sun      = std::pow(sunDot, 400.0f)*40.0f + std::pow(sunDot, 16.0f)*0.4f;
  return horizon + (zenith - horizon)*up + float3(1.0f, 0.9f, 0.7f)*sun;
}

// iterative path: primary ray + up to (MAX_BOUNCES-1) mirror reflections, direct light with hard shadows
float3 Metaballs::TracePath(float3 a_ro, float3 a_rd) const
{
  const float3 lightDir   = to_float3(m_lightDir);
  const float  floorY     = m_floorY;
  const float3 lightColor = float3(1.0f, 0.95f, 0.85f)*2.2f;
  const float3 ambient    = float3(0.10f, 0.12f, 0.16f);

  float3 ro         = a_ro;
  float3 rd         = a_rd;
  float3 radiance   = float3(0.0f, 0.0f, 0.0f);
  float3 throughput = float3(1.0f, 1.0f, 1.0f);

  for(int bounce = 0; bounce < MAX_BOUNCES; bounce++)
  {
    // floor plane y = floorY
    float tFloor = 1e30f;
    if(rd.y < -1e-6f)
    {
      const float t = (floorY - ro.y)/rd.y;
      if(t > 1e-4f)
        tFloor = t;
    }

    const float tBall = TraceBalls(ro, rd, tFloor);

    float3 pos, nrm, albedo;
    float  reflectivity, shininess;
    if(tBall > 0.0f)
    {
      pos          = ro + rd*tBall;
      nrm          = FieldNormal(pos);
      const float4 c = SurfaceColor(pos);
      albedo       = to_float3(c);
      reflectivity = c.w;
      shininess    = 90.0f;
    }
    else if(tFloor < 1e29f)
    {
      pos          = ro + rd*tFloor;
      nrm          = float3(0.0f, 1.0f, 0.0f);
      const int cx = int(std::floor(pos.x*1.5f));
      const int cz = int(std::floor(pos.z*1.5f));
      const bool odd = ((cx + cz) & 1) != 0;
      albedo       = odd ? float3(0.85f, 0.85f, 0.82f) : float3(0.12f, 0.12f, 0.14f);
      const float fade = std::exp(-0.005f*tFloor*tFloor);  // distance fog on the floor: larger coefficient = denser fog
      albedo       = albedo*fade + float3(0.6f, 0.6f, 0.6f)*(1.0f - fade);
      reflectivity = 0.25f;
      shininess    = 30.0f;
    }
    else
    {
      radiance += throughput*Sky(rd);
      break;
    }

    // direct lighting with a hard shadow from the metaballs
    const float3 shadowOrg = pos + nrm*2e-3f;
    const bool   inShadow  = TraceBalls(shadowOrg, lightDir, 1e30f) > 0.0f;
    const float  nDotL     = std::max(dot(nrm, lightDir), 0.0f);
    const float3 halfVec   = normalize(lightDir - rd);
    const float  spec      = std::pow(std::max(dot(nrm, halfVec), 0.0f), shininess);
    const float  visible   = inShadow ? 0.0f : 1.0f;
    const float3 direct    = (albedo*nDotL + float3(1.0f, 1.0f, 1.0f)*spec*0.6f)*lightColor*visible + albedo*ambient;

    // Schlick-like fresnel boost of the reflectivity at grazing angles
    const float cosI    = std::max(dot(nrm, rd*(-1.0f)), 0.0f);
    const float fresnel = reflectivity + (1.0f - reflectivity)*std::pow(1.0f - cosI, 5.0f)*0.5f;

    radiance   += throughput*direct*(1.0f - fresnel);
    throughput  = throughput*fresnel;
    ro          = shadowOrg;
    rd          = reflect(rd, nrm);

    if(bounce == MAX_BOUNCES - 1)            // path cut: add the sky seen in the last reflection
      radiance += throughput*Sky(rd);
  }
  return radiance;
}

//////////////////////////////////////////////////////////////////////////////////////////////////// kernels

void Metaballs::kernel2D_Trace(int w, int h, int a_spp, float4* a_hdr)
{
  #pragma omp parallel for schedule(dynamic)
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      const float3 camPos   = to_float3(m_camPos);
      const float3 camRight = to_float3(m_camRight);
      const float3 camUp    = to_float3(m_camUp);
      const float3 camFwd   = to_float3(m_camForward);
      const float  aspect   = float(w)/float(h);
      const int    spp      = a_spp;

      float3 color = float3(0.0f, 0.0f, 0.0f);
      for(int sy = 0; sy < spp; sy++)
      {
        for(int sx = 0; sx < spp; sx++)
        {
          const float u  = ((float(x) + (float(sx) + 0.5f)/float(spp))/float(w))*2.0f - 1.0f;
          const float v  = ((float(y) + (float(sy) + 0.5f)/float(spp))/float(h))*2.0f - 1.0f; // BMP rows go bottom-up
          const float3 rd = normalize(camFwd + camRight*(u*aspect) + camUp*v);
          color += TracePath(camPos, rd);
        }
      }
      a_hdr[y*w + x] = to_float4(color*(1.0f/float(spp*spp)), 1.0f);
    }
  }
}

void Metaballs::kernel1D_AvgLogLum(int a_size, const float4* a_hdr)
{
  m_sumLogLum = 0.0f;
  for(int i = 0; i < a_size; i++)
  {
    const float lum = luminance(to_float3(a_hdr[i]));
    m_sumLogLum += std::log(1e-4f + lum);            // sum-reduction
  }
}

void Metaballs::kernel2D_ToneMap(int w, int h, float a_invPixels, const float4* a_hdr, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      const float  sumLogLum = m_sumLogLum;                 // locals: see the Slang note above
      const float  invPixels = a_invPixels;
      const float  avgLum    = std::exp(sumLogLum*invPixels);
      const float  exposure  = 0.22f/std::max(avgLum, 1e-4f);
      const float3 c         = to_float3(a_hdr[y*w + x])*exposure;
      float3 mapped;
      mapped.x = std::pow(c.x/(1.0f + c.x), 1.0f/2.2f);   // Reinhard + gamma
      mapped.y = std::pow(c.y/(1.0f + c.y), 1.0f/2.2f);
      mapped.z = std::pow(c.z/(1.0f + c.z), 1.0f/2.2f);
      a_out[y*w + x] = PackRGBA8(mapped);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////// control function

void Metaballs::Render(int w, int h, int a_spp, uint32_t* a_out)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_Trace(w, h, a_spp, m_hdr.data());
  kernel1D_AvgLogLum(w*h, m_hdr.data());
  kernel2D_ToneMap(w, h, 1.0f/float(w*h), m_hdr.data(), a_out);
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}
