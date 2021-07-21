#pragma once 
//select impl with define or include concrete header
#include "OpenCLMath.h"

///////////////////////////////////////////////////////////////////
///// Auxilary functions which are not in the core of library /////
///////////////////////////////////////////////////////////////////

namespace LiteMath
{


static inline float2 reflect(float2 dir, float2 normal) // float3 and float4 versions are defined in the core of the library
{ 
  return normal * dot(dir, normal) * (-2.0f) + dir;     
}

static inline float2 refract(const float2 incidentVec, const float2 normal, float eta)
{
  float N_dot_I = dot(normal, incidentVec);
  float k = 1.f - eta * eta * (1.f - N_dot_I * N_dot_I);
  if (k < 0.f)
    return float2(0.f, 0.f);
  else
    return eta * incidentVec - (eta * N_dot_I + sqrt(k)) * normal;
}

static inline float3 refract(const float3 incidentVec, const float3 normal, float eta)
{
  float N_dot_I = dot(normal, incidentVec);
  float k = 1.f - eta * eta * (1.f - N_dot_I * N_dot_I);
  if (k < 0.f)
    return float3(0.f, 0.f, 0.f);
  else
    return eta * incidentVec - (eta * N_dot_I + sqrt(k)) * normal;
}

static inline float4 refract(const float4 incidentVec, const float4 normal, float eta)
{
  float N_dot_I = dot(normal, incidentVec);
  float k = 1.f - eta * eta * (1.f - N_dot_I * N_dot_I);
  if (k < 0.f)
    return float4(0.f, 0.f, 0.f, 0.f);
  else
    return eta * incidentVec - (eta * N_dot_I + sqrt(k)) * normal;
}

static inline float2 faceforward(float2 N, float2 I, float2 Ng) // A floating-point, surface normal vector that is facing the view direction
{
  return dot(I, Ng) < 0.0f ? N : (-1.0f)*N;
}

static inline float3 faceforward(float3 N, float3 I, float3 Ng)
{
  return dot(I, Ng) < 0.0f ? N : (-1.0f)*N;
}

static inline float4 faceforward(float4 N, float4 I, float4 Ng)
{
  return dot3f(I, Ng) < 0.0f ? N : (-1.0f)*N;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline bool IntersectBox2Box2(float2 box1Min, float2 box1Max, float2 box2Min, float2 box2Max)
{
  return box1Min.x <= box2Max.x && box2Min.x <= box1Max.x &&
         box1Min.y <= box2Max.y && box2Min.y <= box1Max.y;
}

static inline bool IntersectBox2Box2(int2 box1Min, int2 box1Max, int2 box2Min, int2 box2Max)
{
  return box1Min.x <= box2Max.x && box2Min.x <= box1Max.x &&
         box1Min.y <= box2Max.y && box2Min.y <= box1Max.y;
}

inline static float4 color_unpack_bgra(int packedColor)
{
  const int red   = (packedColor & 0x00FF0000) >> 16;
  const int green = (packedColor & 0x0000FF00) >> 8;
  const int blue  = (packedColor & 0x000000FF) >> 0;
  const int alpha = (packedColor & 0xFF000000) >> 24;
  return float4((float)red, (float)green, (float)blue, (float)alpha)*(1.0f / 255.0f);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Look At matrix creation
// return the inverse view matrix
//
static inline float4x4 lookAt(float3 eye, float3 center, float3 up)
{
  float3 x, y, z; // basis; will make a rotation matrix
  z.x = eye.x - center.x;
  z.y = eye.y - center.y;
  z.z = eye.z - center.z;
  z = normalize(z);
  y.x = up.x;
  y.y = up.y;
  y.z = up.z;
  x = cross(y, z); // X vector = Y cross Z
  y = cross(z, x); // Recompute Y = Z cross X
  // cross product gives area of parallelogram, which is < 1.0 for
  // non-perpendicular unit-length vectors; so normalize x, y here
  x = normalize(x);
  y = normalize(y);
  float4x4 M;
  M.set_col(0, float4{ x.x, y.x, z.x, 0.0f });
  M.set_col(1, float4{ x.y, y.y, z.y, 0.0f });
  M.set_col(2, float4{ x.z, y.z, z.z, 0.0f });
  M.set_col(3, float4{ -x.x * eye.x - x.y * eye.y - x.z*eye.z,
                       -y.x * eye.x - y.y * eye.y - y.z*eye.z,
                       -z.x * eye.x - z.y * eye.y - z.z*eye.z,
                       1.0f });
  return M;
}
  
/////////////////////////////////////////
/////////////// Boxes stuff /////////////
/////////////////////////////////////////

struct CVEX_ALIGNED(16) Box4f 
{ 
  inline Box4f()
  {
    boxMin = LiteMath::splat( +std::numeric_limits<float>::infinity() );
    boxMax = LiteMath::splat( -std::numeric_limits<float>::infinity() );   
  }

  inline Box4f(const float4& a_bMin, const float4& a_bMax)
  {
    boxMin = a_bMin;
    boxMax = a_bMax;   
  }

  inline void include(const LiteMath::float4 p) // please note that this function may override Start/Count pair, so use it carefully
  {                                           
    boxMin = LiteMath::min(boxMin, p);
    boxMax = LiteMath::max(boxMax, p);
  }

  inline void include(const Box4f& b) // please note that this function may override Start/Count pair, so use it carefully
  {                                     
    boxMin = LiteMath::min(boxMin, b.boxMin);
    boxMax = LiteMath::max(boxMax, b.boxMax);
  } 

  inline void intersect(const Box4f& a_box) 
  {
    boxMin = LiteMath::max(boxMin, a_box.boxMin);
    boxMax = LiteMath::min(boxMax, a_box.boxMax);
  }

  inline float surfaceArea() const
  {
    const float4 abc = boxMax - boxMin;
    return 2.0f*(abc[0]*abc[1] + abc[0]*abc[2] + abc[1]*abc[2]);
  }

  inline float volume() const 
  {
    const float4 abc = boxMax - boxMin;
    return abc[0]*abc[1]*abc[2];       // #TODO: hmul3
  }

  inline void SetStart(uint i) { boxMin = packUIntW(boxMin, uint(i)); }
  inline void SetCount(uint i) { boxMax = packUIntW(boxMax, uint(i)); }
  inline uint GetStart() const { return extractUIntW(boxMin); }
  inline uint GetCount() const { return extractUIntW(boxMax); }
  inline bool AxisAligned(int axis, float split) const { return (boxMin[axis] == boxMax[axis]) && (boxMin[axis]==split); }

  float4 boxMin; // as_int(boxMin4f.w) may store index of the object inside the box (or start index of the object sequence)
  float4 boxMax; // as_int(boxMax4f.w) may store size (count) of objects inside the box
};

struct CVEX_ALIGNED(16) Ray4f 
{
  inline Ray4f(){}
  inline Ray4f(const float4& pos, const float4& dir) : posAndNear(pos), dirAndFar(dir) { }
  inline Ray4f(const float4& pos, const float4& dir, float tNear, float tFar) : posAndNear(pos), dirAndFar(dir) 
  { 
    posAndNear = packFloatW(posAndNear, tNear);
    dirAndFar  = packFloatW(dirAndFar,  tFar);
  }

  inline Ray4f(const float3& pos, const float3& dir, float tNear, float tFar) : posAndNear(to_float4(pos,tNear)), dirAndFar(to_float4(dir, tFar)) { }

  inline float GetNear() const { return extract_3(posAndNear); }
  inline float GetFar()  const { return extract_3(dirAndFar); }

  float4 posAndNear;
  float4 dirAndFar;
};

/////////////////////////////////////////
/////////////// rays stuff //////////////
/////////////////////////////////////////

/**
\brief Computes near and far intersection of ray and box
\param  rayPos     - input ray origin
\param  rayDirInv  - input inverse ray dir (1.0f/rayDirection)
\param  boxMin     - input box min
\param  boxMax     - input box max
\return (tnear, tfar); if tnear > tfar, no interection is found. 
*/
static inline float2 Ray4fBox4fIntersection(float4 rayPos, float4 rayDirInv, float4 boxMin, float4 boxMax)
{
  const float4 lo   = rayDirInv*(boxMin - rayPos);
  const float4 hi   = rayDirInv*(boxMax - rayPos);

  const float4 vmin = LiteMath::min(lo, hi);
  const float4 vmax = LiteMath::max(lo, hi);
  return float2(hmax3(vmin), hmin3(vmax));
  //return float2(std::max(std::max(vmin[0], vmin[1]), vmin[2]), 
  //              std::min(std::min(vmax[0], vmax[1]), vmax[2]));
}
  
/**
\brief Create eye ray for target x,y and Proj matrix
\param  x - input x coordinate of pixel
\param  y - input y coordinate of pixel
\param  w - input framebuffer image width  
\param  h - input framebuffer image height
\param  a_mProjInv - input inverse projection matrix
\return Eye ray direction; the fourth component will contain +INF as tfar according to Ray4f tnear/tfar storage agreement 
*/
static inline float4 EyeRayDir4f(float x, float y, float w, float h, float4x4 a_mProjInv) // g_mViewProjInv
{
  float4 pos = float4( 2.0f * (x + 0.5f) / w - 1.0f, 
                      -2.0f * (y + 0.5f) / h + 1.0f, 
                       0.0f, 
                       1.0f );

  pos = a_mProjInv*pos;
  pos = pos/pos.w;
  pos.y *= (-1.0f); // TODO: remove this (???)
  pos = normalize(pos);
  pos.w = INF_POSITIVE;
  return pos;
}

/**
\brief  calculate overlapping area (volume) of 2 bounding boxes and return it if form of bounding box
\param  box1 - input first box 
\param  box2 - input second box
\return overlaped volume bounding box. If no intersection found, return zero sized bounding box 
*/
inline Box4f BoxBoxOverlap(const Box4f& box1, const Box4f& box2)
{
  Box4f tempBox1 = box1;
  Box4f tempBox2 = box2;
  float4 res_min;
  float4 res_max;
  
  for(int axis = 0; axis < 3; ++axis){ // #TODO: unroll loop and vectorize code
    // sort boxes by min
    if(tempBox2.boxMin[axis] < tempBox1.boxMin[axis]){
      float tempSwap = tempBox1.boxMin[axis];
      tempBox1.boxMin[axis] = tempBox2.boxMin[axis];
      tempBox2.boxMin[axis] = tempSwap;

      tempSwap = tempBox1.boxMax[axis];
      tempBox1.boxMax[axis] = tempBox2.boxMax[axis];
      tempBox2.boxMax[axis] = tempSwap;
    }
    // check the intersection
    if(tempBox1.boxMax[axis] < tempBox2.boxMin[axis])
      return Box4f(box1.boxMax, box1.boxMax);

    // Intersected box
    res_min[axis] = tempBox2.boxMin[axis];
    res_max[axis] = std::min(tempBox1.boxMax[axis], tempBox2.boxMax[axis]);
  }
  return Box4f(res_min, res_max);
}

/////////////////////////////////////////
//////// frustum traversal stuff ////////
/////////////////////////////////////////

static unsigned int log2uint (unsigned int val)
{
  if (val == 0) return std::numeric_limits<uint>::max();
  if (val == 1) return 0;
  unsigned int res = 0;
  while (val >>= 1) ++res;
  return res;
}

static float4x4 mulColRow(const float4& a_col, const float4& a_row)
{
  float4x4 result;
  for(int i = 0; i < 4; ++i)
  {
    float4 row;
    for(int j = 0; j < 4; ++j)
    {
      row[j] = a_col[i] * a_row[j];
    }
    result.set_row(i, row);
  }
  return result;
}

static inline float3 getPointProjectionToPlane(float3 point, float3 plane_normal, float3 plane_point)
{
  return point - dot(point - plane_point, plane_normal) * plane_normal;
}

static inline float4 getPointProjectionToPlane(float4 point, float4 plane_normal, float4 plane_point)
{
  return point - dot4f(point - plane_point, plane_normal) * plane_normal;
}

static inline float pointToPlaneDist(float3 point, float3 plane_normal, float3 plane_point)
{
  float4 v = float4((point - plane_point).x, (point - plane_point).y, (point - plane_point).z, 1.0f);
  float4 n = float4(plane_normal.x, plane_normal.y, plane_normal.z, 1.0f);
  return dot4f(v, n);
}

static inline float pointToPlaneDist(float4 point, float4 plane_normal, float4 plane_point)
{
  return dot4f(point - plane_point, plane_normal);
}

//plane as coefficients A, B, C, D
static inline float pointToPlaneDist(float3 point, float4 plane)
{
  return dot4f(float4(point.x, point.y, point.z, 1.0f), plane) / sqrtf(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z); // #TODO: use dot3f for denominator!!!
}

//plane as coefficients A, B, C, D
static inline float pointToPlaneDist(float4 point, float4 plane)
{
  return dot4f(point, plane) / sqrtf(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z); // #TODO: use dot3f for denominator!!!
}

static inline float4 rotateAroundAxis(float4 vector, float4 axis, float angle)
{
  axis = normalize(axis);
  float4 res  = vector * cosf(angle) + cross(axis, vector) * sinf (angle) + axis * dot3f(axis, vector) * (1 - cosf(angle));
  res.w = 1.0f;
  return res;
}



}