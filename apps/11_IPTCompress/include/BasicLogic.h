#ifndef BASIC_PROJ_LOGIC_H
#define BASIC_PROJ_LOGIC_H

#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct Lite_HitT
{
  float t;
  int   primId; 
  int   instId;
  int   geomId;
} Lite_Hit;

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

static inline float2 RaySphereHit(float3 orig, float3 dir, float4 sphere) // see Ray Tracing Gems Book
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

  float2 result = {0,0};
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

static inline uint RealColorToUint32_f3(float3 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  unsigned int red = (unsigned int)r, green = (unsigned int)g, blue = (unsigned int)b;
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}

static inline uint RealColorToUint32(float4 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  float  a = real_color.w*255.0f;

  unsigned int red   = (unsigned int)r;
  unsigned int green = (unsigned int)g;
  unsigned int blue  = (unsigned int)b;
  unsigned int alpha = (unsigned int)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

#define WIN_WIDTH  512
#define WIN_HEIGHT 512

static uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void CoordinateSystem(float3 v1, float3* v2, float3* v3)
{
  float invLen = 1.0f;

  if (fabs(v1.x) > fabs(v1.y))
  {
    invLen = 1.0f / sqrt(v1.x*v1.x + v1.z*v1.z);
    (*v2)  = make_float3((-1.0f) * v1.z * invLen, 0.0f, v1.x * invLen);
  }
  else
  {
    invLen = 1.0f / sqrt(v1.y * v1.y + v1.z * v1.z);
    (*v2)  = make_float3(0.0f, v1.z * invLen, (-1.0f) * v1.y * invLen);
  }

  (*v3) = cross(v1, (*v2));
}


#define GEPSILON      5e-6f
#define DEPSILON      1e-20f

enum THREAD_FLAGS { THREAD_IS_DEAD = 2147483648};

static inline float3 MapSampleToCosineDistribution(float r1, float r2, float3 direction, float3 hit_norm, float power)
{
  if(power >= 1e6f)
    return direction;

  const float sin_phi = sin(M_TWOPI * r1);
  const float cos_phi = cos(M_TWOPI * r1);

  //sincos(2.0f*r1*3.141592654f, &sin_phi, &cos_phi);

  const float cos_theta = pow(1.0f - r2, 1.0f / (power + 1.0f));
  const float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

  float3 deviation;
  deviation.x = sin_theta*cos_phi;
  deviation.y = sin_theta*sin_phi;
  deviation.z = cos_theta;

  float3 ny = direction, nx, nz;
  CoordinateSystem(ny, &nx, &nz);

  {
    float3 temp = ny;
    ny = nz;
    nz = temp;
  }

  float3 res = nx*deviation.x + ny*deviation.y + nz*deviation.z;

  float invSign = dot(direction, hit_norm) > 0.0f ? 1.0f : -1.0f;

  if (invSign*dot(res, hit_norm) < 0.0f) // reflected ray is below surface #CHECK_THIS
  {
    res = (-1.0f)*nx*deviation.x + ny*deviation.y - nz*deviation.z;
    //belowSurface = true;
  }

  return res;
}

static inline float epsilonOfPos(float3 hitPos) { return fmax(fmax(fabs(hitPos.x), fmax(fabs(hitPos.y), fabs(hitPos.z))), 2.0f*GEPSILON)*GEPSILON; }

/**
\brief offset reflected ray position by epsilon;
\param  a_hitPos      - world space position on surface
\param  a_surfaceNorm - surface normal at a_hitPos
\param  a_sampleDir   - ray direction in which we are going to trace reflected ray
\return offseted ray position
*/
static inline float3 OffsRayPos(const float3 a_hitPos, const float3 a_surfaceNorm, const float3 a_sampleDir)
{
  const float signOfNormal2 = dot(a_sampleDir, a_surfaceNorm) < 0.0f ? -1.0f : 1.0f;
  const float offsetEps     = epsilonOfPos(a_hitPos);
  return a_hitPos + signOfNormal2*offsetEps*a_surfaceNorm;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct MaterialT
{
  float4 color;
} SphereMaterial;

static inline bool   IsMtlEmissive(__global const SphereMaterial* a_mtl)       { return (a_mtl->color.w != 0.0f); }
static inline float3 GetMtlDiffuseColor(__global const SphereMaterial* a_mtl)  { return to_float3(a_mtl->color); }
static inline float3 GetMtlEmissiveColor(__global const SphereMaterial* a_mtl) { return to_float3(a_mtl->color)*a_mtl->color.w; }

#endif