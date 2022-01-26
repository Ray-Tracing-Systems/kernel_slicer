#include "raymarch.h"

static inline float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar)
{
  const float ymax = zNear * tanf(fovy * 3.14159265358979323846f / 360.0f);
  const float xmax = ymax * aspect;
  const float left = -xmax;
  const float right = +xmax;
  const float bottom = -ymax;
  const float top = +ymax;
  const float temp = 2.0f * zNear;
  const float temp2 = right - left;
  const float temp3 = top - bottom;
  const float temp4 = zFar - zNear;
  float4x4 res;
  res.set_col(0, float4{ temp / temp2, 0.0f, 0.0f, 0.0f });
  res.set_col(1, float4{ 0.0f, temp / temp3, 0.0f, 0.0f });
  res.set_col(2, float4{ (right + left) / temp2,  (top + bottom) / temp3, (-zFar - zNear) / temp4, -1.0 });
  res.set_col(3, float4{ 0.0f, 0.0f, (-temp * zFar) / temp4, 0.0f });
  return res;
}

uint32_t color_convert(float3 color)
{
  uint32_t R = uint32_t(255u * color.x);
  uint32_t G = uint32_t(255u * color.y);
  uint32_t B = uint32_t(255u * color.z);

  uint32_t res_color = 0xFF000000 | R | (G << 8) | (B << 16);

  return res_color;
}

static bool Inside(float3 p, float3 pMin, float3 pMax) {
  return (p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y && p.y <= pMax.y && p.z >= pMin.z && p.z <= pMax.z);
}

static inline float3 Normalize(float3 p, float3 pMin, float3 pMax) {
  return float3((p.x - pMin.x) / (pMax.x - pMin.x), (p.y - pMin.y) / (pMax.y - pMin.y),
                (p.z - pMin.z) / (pMax.z - pMin.z));
}

bool RayBoxIntersection(const float4& ray_pos, const float4& ray_dir, const float3& boxMin,  const float3& boxMax,
                        float &tmin, float &tmax)
{
  float4 raydir = ray_dir;
  raydir.x = 1.0f / raydir.x;
  raydir.y = 1.0f / raydir.y;
  raydir.z = 1.0f / raydir.z;

  float lo = raydir.x*((boxMin).x - (ray_pos).x);
  float hi = raydir.x*((boxMax).x - (ray_pos).x);

  tmin = std::min(lo, hi);
  tmax = std::max(lo, hi);

  float lo1 = raydir.y*((boxMin).y - (ray_pos).y);
  float hi1 = raydir.y*((boxMax).y - (ray_pos).y);

  tmin = std::max(tmin, std::min(lo1, hi1));
  tmax = std::min(tmax, std::max(lo1, hi1));

  float lo2 = raydir.z*((boxMin).z - (ray_pos).z);
  float hi2 = raydir.z*((boxMax).z - (ray_pos).z);

  tmin = std::max(tmin, std::min(lo2, hi2));
  tmax = std::min(tmax, std::max(lo2, hi2));

  return (tmin <= tmax) && (tmax > 0.f);
}


bool RaySphereIntersection(const float4 ray_pos, const float4 ray_dir, float3 center, float radius,
                           float &tmin, float &tmax)
{
  float3 p0 = to_float3(ray_pos);
  float3 p1 = to_float3(ray_pos) + 2 * to_float3(ray_dir);

  float l = (p1 - p0).x;
  float m = (p1 - p0).y;
  float n = (p1 - p0).z;

  float A = l * l + n * n + m * m;
  float B = 2 * (p0.x - center.x) * l + 2 * (p0.y - center.y) * m + 2 * (p0.z - center.z)* n;
  float C = (p0.x - center.x) * (p0.x - center.x) +
            (p0.y - center.y) * (p0.y - center.y) +
            (p0.z - center.z) * (p0.z - center.z) - radius*radius;

  float D = B * B - 4 * A * C;
  if (D < 0)
    return false;

  float res1 = (std::sqrt(D) - B)  / (2 * A);
  float res2 = (-std::sqrt(D) - B) / (2 * A);

  tmin = std::min(res1, res2);
  tmax = std::max(res1, res2);

  return (tmin <= tmax) && (tmax > 0.f);
}

float4 EyeRayDir4(float x, float y, float w, float h, float4x4 a_mProjInv) // g_mViewProjInv
{
  float4 pos = float4( 2.0f * (x + 0.5f) / w - 1.0f, 
                      -2.0f * (y + 0.5f) / h + 1.0f, 
                       0.0f, 
                       1.0f );

  pos = a_mProjInv*pos;
  pos = pos/pos.w;
  pos.y *= (-1.0f);   // TODO: do we need remove this (???)
  const float lenInv = 1.0f/std::sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
  pos *= lenInv;
  pos.w = 1e37f;
  return pos;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float3 RayMarcher::RayFunc(float tmin, float tmax, float *alpha, const float4 *ray_pos, const float4 *ray_dir)
{
  const float dt = 0.005f;
  const float sigma_a = 10;
  const float sigma_s = 10;

  float t  = tmin;
  *alpha  = 0.0f;
  float3 color(0, 0, 0);

  int n = 0;
  while(t < tmax)
  {
    if(std::exp(-*alpha) < .005f)
      break;
    float D = SampleDensity(*ray_pos + t * *ray_dir, true);
    color += float3(D);
    t += dt;
    n++;
    *alpha += dt * (sigma_a + sigma_s) * D;
  }

  return color / n;
}

float RayMarcher::GetDensity(int x, int y, int z)
{
  x = clamp(x, 0, m_gridResolution.x - 1);
  y = clamp(y, 0, m_gridResolution.y - 1);
  z = clamp(z, 0, m_gridResolution.z - 1);
  return m_densityField[z * m_gridResolution.x * m_gridResolution.y + y * m_gridResolution.x + x];
}


float RayMarcher::SampleDensity(float4 pos, bool trilinear)
{
  if(!Inside(to_float3(pos), SCENE_BOX_MIN, SCENE_BOX_MAX))
    return 0.0f;

  float3 vox = Normalize(to_float3(pos), SCENE_BOX_MIN, SCENE_BOX_MAX);
  vox.x = vox.x * m_gridResolution.x + .5f;
  vox.y = vox.y * m_gridResolution.y + .5f;
  vox.z = vox.z * m_gridResolution.z + .5f;

  int vx = (int)(vox.x), vy = (int)(vox.y), vz = (int)(vox.z);

  if(!trilinear)
  {
    return GetDensity(vx, vy, vz);
  }

  float dx = vox.x - vx, dy = vox.y - vy, dz = vox.z - vz;
  float d00 = lerp(GetDensity(vx, vy, vz), GetDensity(vx + 1, vy, vz), dx);
  float d10 = lerp(GetDensity(vx, vy + 1, vz), GetDensity(vx + 1, vy + 1, vz), dx);
  float d01 = lerp(GetDensity(vx, vy, vz + 1), GetDensity(vx + 1, vy, vz + 1), dx);
  float d11 = lerp(GetDensity(vx, vy + 1, vz + 1), GetDensity(vx + 1, vy + 1, vz + 1), dx);
  float d0 = lerp(d00, d10, dy);
  float d1 = lerp(d01, d11, dy);
  return lerp(d0, d1, dz);
}

void RayMarcher::Init(const std::vector<float> &a_densityField, const int3 &a_gridResolution)
{
  m_densityField   = a_densityField;
  m_gridResolution = a_gridResolution;

  InitView();
}

void RayMarcher::InitView()
{
  m_camPos = float4(0.0f, 0.0f, 10.0f, 1.0f);
  float aspect   = 1.0f;
  float fov      = 60.0f;
  auto proj      = perspectiveMatrix(fov, aspect, 0.001f, 100.0f);
  auto worldView = lookAt(to_float3(m_camPos), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));

  m_invProjView = inverse4x4(proj * transpose(inverse4x4(worldView)));
}

void RayMarcher::Execute(uint32_t tidX, uint32_t tidY, uint32_t* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tidX, tidY, &rayPosAndNear, &rayDirAndFar);

  kernel_RayMarch(tidX, tidY, &rayPosAndNear, &rayDirAndFar, out_color);
}

void RayMarcher::kernel_InitEyeRay(uint32_t tidX, uint32_t tidY, float4* rayPosAndNear, float4* rayDirAndFar)
{
  *rayPosAndNear = m_camPos;
  *rayDirAndFar  = EyeRayDir4(float(tidX), float(tidY), float(m_width), float(m_height), m_invProjView);
}

void RayMarcher::kernel_RayMarch(uint32_t tidX, uint32_t tidY, const float4* rayPosAndNear, const float4* rayDirAndFar, uint32_t* out_color)
{
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  float tmin = 1e38f;
  float tmax = 0;

//  float3 center(0, 0, 0);
//  float radius = 1.5f;
//  if(!RaySphereIntersection(rayPos, rayDir, center, radius, tmin, tmax))
  if(!RayBoxIntersection(rayPos, rayDir, SCENE_BOX_MIN, SCENE_BOX_MAX, tmin, tmax))
  {
    out_color[tidY * m_width + tidX] = color_convert(BACKGROUND_COLOR);
    return;
  }

  float alpha = 0.0f;
  float3 color = RayFunc(tmin, tmax, &alpha, &rayPos, &rayDir);//float3(1.0f);
  out_color[tidY * m_width + tidX] = color_convert(color);
}