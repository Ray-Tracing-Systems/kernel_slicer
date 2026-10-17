/////////////////////////////////////////////////////////////////////
// Hand-written prototype of what kslicer ISPC back end should
// generate for 'TestClass' (IPV, ray tracing via Embree packets).
/////////////////////////////////////////////////////////////////////
#include "LiteMathISPC.h"

#ifdef USE_EMBREE4
#include "embree4/rtcore.isph"
#else
#include "embree3/rtcore.isph"
#endif

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////

#define WIN_WIDTH  1024
#define WIN_HEIGHT 1024

static const uniform uint HIT_TRIANGLE_GEOM   = 0;
static const uniform uint HIT_FLAT_LIGHT_GEOM = 1;

struct Lite_Hit
{
  float t;
  int   primId;
  int   instId;
  int   geomId;
};

struct LightGeom
{
  float4 boxMin;
  float4 boxMax;
};

// ISPC aligns short vectors (uint<2>, float<2>, float<3>) to 16 bytes, so in data structures stored in buffers 
// they must be replaced with plain structs to have the same layout as LiteMath types 
//
struct uint2_s 
{ 
  uint x; 
  uint y; 
};

struct RandomGen
{
  uint2_s state;
};

struct CRT_Hit
{
  float t;
  uint  primId;
  uint  instId;
  uint  geomId;
  float coords[4];
};

struct TestClass_ISPC_UBO_Data
{
  float4x4  m_worldViewProjInv;
  LightGeom m_lightGeom;
  float4    m_camPos;            // float3 in C++ class, LiteMath::float3 (12 bytes) and ISPC float<3> (16 bytes) have different layout
  uint      m_emissiveMaterialId;
  uint      m_materialIds_size;
  uint      m_materialIds_capacity;
  uint      m_materials_size;
  uint      m_materials_capacity;
  uint      m_randomGens_size;
  uint      m_randomGens_capacity;
  uint      dummy_last;
};

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

static inline float4 mul4x4x4(const uniform float4x4& m, float4 v) // column-major as in LiteMath; uniform matrix is not copied to varying
{
  return m.m_col[0]*v.x + m.m_col[1]*v.y + m.m_col[2]*v.z + m.m_col[3]*v.w;
}

static inline float3 EyeRayDir(float x, float y, float w, float h, const uniform float4x4& a_mViewProjInv)
{
  float4 pos = make_float4( 2.0f * (x + 0.5f) / w - 1.0f,
                           -2.0f * (y + 0.5f) / h + 1.0f,
                            0.0f,
                            1.0f );

  pos = mul4x4x4(a_mViewProjInv, pos);
  pos = pos/pos.w;

  pos.y *= (-1.0f);

  return normalize(to_float3(pos));
}

static inline uint NextState(varying RandomGen* uniform gen)
{
  const uint x = gen->state.x * 17 + gen->state.y * 13123;
  gen->state.x = (x << 13) ^ x;
  gen->state.y ^= (x << 7);
  return x;
}

static inline float2 rndFloat2_Pseudo(varying RandomGen* uniform gen)
{
  uint x = NextState(gen);

  const uint x1 = (x * (x * x * 15731 + 74323) + 871483);
  const uint y1 = (x * (x * x * 13734 + 37828) + 234234);

  const float scale = (1.0f / 4294967296.0f);

  return make_float2((float)(x1), (float)(y1))*scale;
}

static inline uint RealColorToUint32_f3(float3 real_color)
{
  float r = real_color.x*255.0f;
  float g = real_color.y*255.0f;
  float b = real_color.z*255.0f;
  uint red = (uint)r, green = (uint)g, blue = (uint)b;
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}

/////////////////////////////////////////////////////////////////////
/////////////////// ray queries via Embree //////////////////////////
/////////////////////////////////////////////////////////////////////

// 'm_pAccelStruct->RayQuery_NearestHit(pos, dir)' ==> 'm_pAccelStruct_RayQuery_NearestHit(scene, geomIdByInst, pos, dir)'
// rtcIntersectV traces all active program instances as a packet of programCount rays (rtcIntersect4/8/16)
//
static inline CRT_Hit m_pAccelStruct_RayQuery_NearestHit(RTCScene a_scene, const uniform uint a_geomIdByInst[], float4 posAndNear, float4 dirAndFar)
{
  RTCRayHit rayhit;
  rayhit.ray.org_x = posAndNear.x;
  rayhit.ray.org_y = posAndNear.y;
  rayhit.ray.org_z = posAndNear.z;
  rayhit.ray.tnear = posAndNear.w;
  rayhit.ray.dir_x = dirAndFar.x;
  rayhit.ray.dir_y = dirAndFar.y;
  rayhit.ray.dir_z = dirAndFar.z;
  rayhit.ray.tfar  = dirAndFar.w;
  rayhit.ray.time  = 0.0f;
  rayhit.ray.mask  = -1;
  rayhit.ray.flags = 0;
  rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  #ifdef USE_EMBREE4
  rtcIntersectV(a_scene, &rayhit);
  #else
  uniform RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  rtcIntersectV(a_scene, &context, &rayhit);
  #endif

  CRT_Hit result;
  result.t = rayhit.ray.tfar;
  if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
  {
    result.geomId    = a_geomIdByInst[rayhit.hit.instID[0]];
    result.instId    = rayhit.hit.instID[0];
    result.primId    = rayhit.hit.primID;
    result.coords[1] = rayhit.hit.u;
    result.coords[0] = rayhit.hit.v;
    result.coords[2] = 1.0f - rayhit.hit.v - rayhit.hit.u;
    result.coords[3] = 0.0f;
  }
  else
  {
    result.geomId = (uint)(-1);
    result.instId = (uint)(-1);
    result.primId = (uint)(-1);
  }
  return result;
}

static inline bool m_pAccelStruct_RayQuery_AnyHit(RTCScene a_scene, float4 posAndNear, float4 dirAndFar)
{
  RTCRay ray;
  ray.org_x = posAndNear.x;
  ray.org_y = posAndNear.y;
  ray.org_z = posAndNear.z;
  ray.tnear = posAndNear.w;
  ray.dir_x = dirAndFar.x;
  ray.dir_y = dirAndFar.y;
  ray.dir_z = dirAndFar.z;
  ray.tfar  = dirAndFar.w;
  ray.time  = 0.0f;
  ray.mask  = -1;
  ray.flags = 0;

  #ifdef USE_EMBREE4
  rtcOccludedV(a_scene, &ray);
  #else
  uniform RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  rtcOccludedV(a_scene, &context, &ray);
  #endif

  return (ray.tfar < 0.0f);
}

/////////////////////////////////////////////////////////////////////
/////////////////// member functions ////////////////////////////////
/////////////////////////////////////////////////////////////////////

static inline Lite_Hit TestClass_RayTrace(float4 rayPos, float4 rayDir,
                                          RTCScene a_scene, const uniform uint a_geomIdByInst[], uniform TestClass_ISPC_UBO_Data ubo[1])
{
  CRT_Hit hit = m_pAccelStruct_RayQuery_NearestHit(a_scene, a_geomIdByInst, rayPos, rayDir);

  Lite_Hit res;
  res.primId = hit.primId;
  res.instId = hit.instId;
  res.geomId = hit.geomId;
  res.t      = hit.t;

  // intersect flat light under roof
  {
    const float tLightHit  = (ubo[0].m_lightGeom.boxMax.y - rayPos.y)/max(rayDir.y, 1e-6f);
    const float4 hit_point = rayPos + tLightHit*rayDir;

    bool is_hit = (hit_point.x > ubo[0].m_lightGeom.boxMin.x) && (hit_point.x < ubo[0].m_lightGeom.boxMax.x) &&
                  (hit_point.z > ubo[0].m_lightGeom.boxMin.z) && (hit_point.z < ubo[0].m_lightGeom.boxMax.z) &&
                  (tLightHit < res.t);

    if(is_hit)
    {
      res.primId = 0;
      res.instId = -1;
      res.geomId = HIT_FLAT_LIGHT_GEOM;
      res.t      = tLightHit;
    }
    else
      res.geomId = HIT_TRIANGLE_GEOM;
  }

  return res;
}

static inline float4 TestClass_ShadowRayOrigin(float4 rayPos, float4 rayDir, float hitT, uniform TestClass_ISPC_UBO_Data ubo[1])
{
  const float3 hitPos    = to_float3(rayPos) + hitT*0.99995f*to_float3(rayDir);
  const float3 boxCenter = 0.5f*(to_float3(ubo[0].m_lightGeom.boxMin) + to_float3(ubo[0].m_lightGeom.boxMax));
  return to_float4(hitPos + 1e-6f*normalize(boxCenter - hitPos), 0.0f);
}

static inline float TestClass_CalcShadow(uint tid, float4 surfPos, uniform RandomGen m_randomGens[], RTCScene a_scene,
                                         uniform TestClass_ISPC_UBO_Data ubo[1])
{
  RandomGen gen = m_randomGens[tid];

  float shadow = 0.0f;
  const uniform int numSamples = 64;

  for(uniform int i=0;i<numSamples;i++)
  {
    const float2 uv = rndFloat2_Pseudo(&gen);
    const float3 samplePos  = to_float3(ubo[0].m_lightGeom.boxMin) + make_float3(uv.x, 0.5f, uv.y)*to_float3(ubo[0].m_lightGeom.boxMax - ubo[0].m_lightGeom.boxMin);

    const float3 shadowRayPos = to_float3(surfPos);
    const float  hitDist      = length(samplePos - shadowRayPos);
    const float3 shadowRayDir = normalize(samplePos - shadowRayPos);

    const bool inShadow = m_pAccelStruct_RayQuery_AnyHit(a_scene, to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));

    if(inShadow)
      shadow += 1.0f;
  }

  return 1.0f - shadow*(1.0f/(float)(numSamples));
}

static inline uint TestClass_GetTestColor(Lite_Hit hit, float shadow, uniform const uint m_materialIds[], uniform const float4 m_materials[])
{
  if(hit.geomId == HIT_FLAT_LIGHT_GEOM)
    return RealColorToUint32_f3(make_float3(1,1,1));

  const uint   mtId  = m_materialIds[hit.primId];
  const float4 mdata = m_materials[mtId];
  return RealColorToUint32_f3(to_float3(mdata)*(0.1f + 0.9f*shadow));
}

/////////////////////////////////////////////////////////////////////
/////////////////// kernels /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

task void kernel2D_PackXY_Task(uniform uint tidX, uniform uint tidY, uniform uint out_pakedXY[],
                               uniform TestClass_ISPC_UBO_Data ubo[1],
                               uniform int _start, uniform int _end, uniform int _span)
{
  const uniform int ystart = _start + taskIndex * _span;
  const uniform int yend   = _start + min((uniform int)((taskIndex+1) * _span), _end);
  foreach(y = ystart ... yend, x = 0 ... tidX) {
    out_pakedXY[y*WIN_WIDTH + x] = ((y << 16) & 0xFFFF0000) | (x & 0x0000FFFF);
  }
}

export void kernel2D_PackXY_ISPC(uniform uint tidX, uniform uint tidY, uniform uint out_pakedXY[],
                                 uniform TestClass_ISPC_UBO_Data ubo[1])
{
  const uniform int span   = 4;
  const uniform int length = tidY;
  launch[(length + span - 1)/span] kernel2D_PackXY_Task(tidX, tidY, out_pakedXY, ubo, 0, tidY, span);
  sync;
}

task void kernel1D_CastSingleRay_Task(uniform uint tid, uniform const uint in_pakedXY[], uniform uint out_color[],
                                      uniform const uint m_materialIds[], uniform const float4 m_materials[], uniform RandomGen m_randomGens[],
                                      void* uniform m_pAccelStruct_scene, uniform const uint m_pAccelStruct_geomIdByInst[],
                                      uniform TestClass_ISPC_UBO_Data ubo[1],
                                      uniform int _start, uniform int _end, uniform int _span)
{
  RTCScene a_scene = (RTCScene)m_pAccelStruct_scene;
  const uniform int xstart = _start + taskIndex * _span;
  const uniform int xend   = _start + min((uniform int)((taskIndex+1) * _span), _end);
  foreach(i = xstart ... xend) {
    const uint XY = in_pakedXY[i];
    const uint x  = (XY & 0x0000FFFF);
    const uint y  = (XY & 0xFFFF0000) >> 16;

    const float4 rayPos = ubo[0].m_camPos; // to_float4(m_camPos, 0.0f), w is zero
    const float4 rayDir = to_float4(EyeRayDir(x, y, WIN_WIDTH, WIN_HEIGHT, ubo[0].m_worldViewProjInv), FLT_MAX);

    const Lite_Hit hit = TestClass_RayTrace(rayPos, rayDir, a_scene, m_pAccelStruct_geomIdByInst, ubo);
    if((hit.primId != -1) && (hit.t < rayDir.w))
    {
      const float4 surfPos = TestClass_ShadowRayOrigin(rayPos, rayDir, hit.t, ubo);
      const float  shadow  = TestClass_CalcShadow(i, surfPos, m_randomGens, a_scene, ubo);
      out_color[i] = TestClass_GetTestColor(hit, shadow, m_materialIds, m_materials);
    }
  }
}

export void kernel1D_CastSingleRay_ISPC(uniform uint tid, uniform const uint in_pakedXY[], uniform uint out_color[],
                                        uniform const uint m_materialIds[], uniform const float4 m_materials[], uniform RandomGen m_randomGens[],
                                        void* uniform m_pAccelStruct_scene, uniform const uint m_pAccelStruct_geomIdByInst[],
                                        uniform TestClass_ISPC_UBO_Data ubo[1])
{
  const uniform int span   = 1024;
  const uniform int length = tid;
  launch[(length + span - 1)/span] kernel1D_CastSingleRay_Task(tid, in_pakedXY, out_color, m_materialIds, m_materials, m_randomGens,
                                                               m_pAccelStruct_scene, m_pAccelStruct_geomIdByInst, ubo, 0, tid, span);
  sync;
}
