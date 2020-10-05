#ifndef BASIC_PROJ_LOGIC_H
#define BASIC_PROJ_LOGIC_H

// (1) Dealing with code/math that differs for CPU and GPU
//
#ifdef __OPENCL_VERSION__
  #include "OpenCLMath.h" // include some additional math or code you need to be applied in OpenCL kernels
#else
  #include "LiteMath.h"   // implementation of _same_ functions on the CPU
  using namespace LiteMath;
#endif

// (2) put you ligic or math code that will be same for CPU and GPU
//
#define INV_TWOPI 0.15915494309189533577f

static inline float2 make_float2(float a, float b)
{
  float2 res;
  res.x = a;
  res.y = b;
  return res;
}

static inline float3 make_float3(float a, float b, float c)
{
  float3 res;
  res.x = a;
  res.y = b;
  res.z = c;
  return res;
}

static inline float4 make_float4(float a, float b, float c, float d)
{
  float4 res;
  res.x = a;
  res.y = b;
  res.z = c;
  res.w = d;
  return res;
}

static inline float2 to_float2(float4 f4)
{
  float2 res;
  res.x = f4.x;
  res.y = f4.y;
  return res;
}

static inline float3 to_float3(float4 f4)
{
  float3 res;
  res.x = f4.x;
  res.y = f4.y;
  res.z = f4.z;
  return res;
}

static inline float4 to_float4(float3 v, float w)
{
  float4 res;
  res.x = v.x;
  res.y = v.y;
  res.z = v.z;
  res.w = w;
  return res;
}

static inline float4 mul4x4x4(float4x4 m, float4 v)
{
  float4 res;
  res.x = v.x * m.m_col[0].x + v.y * m.m_col[1].x + v.z * m.m_col[2].x + v.w * m.m_col[3].x;
  res.y = v.x * m.m_col[0].y + v.y * m.m_col[1].y + v.z * m.m_col[2].y + v.w * m.m_col[3].y;
  res.z = v.x * m.m_col[0].z + v.y * m.m_col[1].z + v.z * m.m_col[2].z + v.w * m.m_col[3].z;
  res.w = v.x * m.m_col[0].w + v.y * m.m_col[1].w + v.z * m.m_col[2].w + v.w * m.m_col[3].w;
  return res;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Lite_Hit
{
  float t;
  int   primId; 
  int   instId;
  int   geomId;
};

static inline float3 EyeRayDir(float x, float y, float w, float h, float4x4 a_mViewProjInv) // g_mViewProjInv
{
  float4 pos = make_float4( 2.0f * (x + 0.5f) / w - 1.0f, 
                           -2.0f * (y + 0.5f) / h + 1.0f, 
                            0.0f, 
                            1.0f );

  pos = mul4x4x4(a_mViewProjInv, pos);
  pos /= pos.w;

  pos.y *= (-1.0f);

  return normalize(to_float3(pos));
}

inline float2 RaySphereHit(float3 orig, float3 dir, float4 sphere) // see Ray Tracing Gems Book
{
  const float3 center = to_float3(sphere);
  const float  radius = sphere.w;

  // Hearn and Baker equation 10-72 for when radius^2 << distance between origin and center
	// Also at https://www.cg.tuwien.ac.at/courses/EinfVisComp/Slides/SS16/EVC-11%20Ray-Tracing%20Slides.pdf
	// Assumes ray direction is normalized
	//dir = normalize(dir);
	const float3 deltap   = center - orig;
	const float ddp       = dot(dir, deltap);
	const float deltapdot = dot(deltap, deltap);

	// old way, "standard", though it seems to be worse than the methods above
	//float discriminant = ddp * ddp - deltapdot + radius * radius;
	float3 remedyTerm  = deltap - ddp * dir;
	float discriminant = radius * radius - dot(remedyTerm, remedyTerm);

  float2 result(0,0);
	if (discriminant >= 0.0f)
	{
		const float sqrtVal = sqrt(discriminant);
		// include Press, William H., Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery, 
		// "Numerical Recipes in C," Cambridge University Press, 1992.
		const float q = (ddp >= 0) ? (ddp + sqrtVal) : (ddp - sqrtVal);
		// we don't bother testing for division by zero
		const float t1 = q;
		const float t2 = (deltapdot - radius * radius) / q;
    result.x = fmin(t1,t2);
    result.y = fmax(t1,t2);
  }
  
  return result;
}


inline Lite_Hit RayTraceImpl(float3 rayPos, float3 rayDir) 
{
  // some imple here ... 
  //
  Lite_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = MAXFLOAT;

  const float2 tNearAndFar = RaySphereHit(rayPos, rayDir, make_float4(0,0,-2.0f,1.0f));

  if(tNearAndFar.x < tNearAndFar.y)
  {
    res.t      = tNearAndFar.x;
    res.primId = 0;
  }

  return res;
}

#define WIN_WIDTH  512
#define WIN_HEIGHT 512

uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#endif