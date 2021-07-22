#include <cmath>
#include <limits>           // for std::numeric_limits
#include <cstring>          // for memcpy

#ifdef M_PI
#undef M_PI // same if we have such macro some-where else ...
#endif

#ifdef min 
#undef min // if we on windows, need thid due to macro definitions in windows.h; same if we have such macro some-where else.
#endif

#ifdef max
#undef min // if we on windows, need thid due to macro definitions in windows.h; same if we have such macro some-where else.
#endif

#include <algorithm>        // for std::min/std::max 
#include <initializer_list> //

#include "OpenCLMath.h"

namespace LiteMath
{ 
  typedef unsigned int   uint;
  typedef unsigned short ushort;
  typedef unsigned char  uchar;

  ///////////////////////////////////////////////////////////////////
  ///// Auxilary functions which are not in the core of library /////
  ///////////////////////////////////////////////////////////////////
  
  static inline uint color_pack_rgba(const float3 rel_col) { return cvex::color_pack_rgba(rel_col.v); }
  static inline uint color_pack_bgra(const float3 rel_col) { return cvex::color_pack_bgra(rel_col.v); }
  
  inline static float4 color_unpack_bgra(int packedColor)
  {
    const int red   = (packedColor & 0x00FF0000) >> 16;
    const int green = (packedColor & 0x0000FF00) >> 8;
    const int blue  = (packedColor & 0x000000FF) >> 0;
    const int alpha = (packedColor & 0xFF000000) >> 24;
    return float4((float)red, (float)green, (float)blue, (float)alpha)*(1.0f / 255.0f);
  }

  inline static float4 color_unpack_rgba(int packedColor)
  {
    const int blue  = (packedColor & 0x00FF0000) >> 16;
    const int green = (packedColor & 0x0000FF00) >> 8;
    const int red   = (packedColor & 0x000000FF) >> 0;
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
  
  struct Box4f 
  { 
    inline Box4f()
    {
      boxMin = float4( +std::numeric_limits<float>::infinity() );
      boxMax = float4( -std::numeric_limits<float>::infinity() );   
    }
  
    inline Box4f(const float4 a_bMin, const float4 a_bMax)
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
  
    inline void setStart(uint i) { boxMin = packUIntW(boxMin, uint(i)); }
    inline void setCount(uint i) { boxMax = packUIntW(boxMax, uint(i)); }
    inline uint getStart() const { return extractUIntW(boxMin); }
    inline uint getCount() const { return extractUIntW(boxMax); }
    inline bool isAxisAligned(int axis, float split) const { return (boxMin[axis] == boxMax[axis]) && (boxMin[axis]==split); }
  
    float4 boxMin; // as_int(boxMin4f.w) may store index of the object inside the box (or start index of the object sequence)
    float4 boxMax; // as_int(boxMax4f.w) may store size (count) of objects inside the box
  };
  
  struct Ray4f 
  {
    inline Ray4f(){}
    inline Ray4f(const float4& pos, const float4& dir) : posAndNear(pos), dirAndFar(dir) { }
    inline Ray4f(const float4& pos, const float4& dir, float tNear, float tFar) : posAndNear(pos), dirAndFar(dir) 
    { 
      posAndNear = packFloatW(posAndNear, tNear);
      dirAndFar  = packFloatW(dirAndFar,  tFar);
    }
  
    inline Ray4f(const float3& pos, const float3& dir, float tNear, float tFar) : posAndNear(to_float4(pos,tNear)), dirAndFar(to_float4(dir, tFar)) { }
  
    inline float getNear() const { return extract_3(posAndNear); }
    inline float getFar()  const { return extract_3(dirAndFar); }

    inline void  setNear(float tNear) { posAndNear = packFloatW(posAndNear, tNear); } 
    inline void  setFar (float tFar)  { dirAndFar  = packFloatW(dirAndFar,  tFar); } 
  
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
    const float4 vmin = min(lo, hi);
    const float4 vmax = max(lo, hi);
    return float2(hmax3(vmin), hmin3(vmax));
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
    pos.y *= (-1.0f);      // TODO: do we need remove this (???)
    pos = normalize3(pos);
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

};

