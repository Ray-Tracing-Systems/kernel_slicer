#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
using LiteMath::float4x4;
using LiteMath::float4;
using LiteMath::float3;
using LiteMath::float2;
using LiteMath::uint;
using LiteMath::uint2;
using LiteMath::min;
using LiteMath::max;

#include "CrossRT.h" // special include for ray tracing

static inline float2 RayBoxIntersection2(float3 rayOrigin, float3 rayDirInv, float3 boxMin, float3 boxMax)
{
  const float lo  = rayDirInv.x*(boxMin.x - rayOrigin.x);
  const float hi  = rayDirInv.x*(boxMax.x - rayOrigin.x);
  const float lo1 = rayDirInv.y*(boxMin.y - rayOrigin.y);
  const float hi1 = rayDirInv.y*(boxMax.y - rayOrigin.y);
  const float lo2 = rayDirInv.z*(boxMin.z - rayOrigin.z);
  const float hi2 = rayDirInv.z*(boxMax.z - rayOrigin.z);

  const float tmin = std::max(std::min(lo, hi), std::min(lo1, hi1));
  const float tmax = std::min(std::max(lo, hi), std::max(lo1, hi1));

  return float2(std::max(tmin, std::min(lo2, hi2)), 
                std::min(tmax, std::max(lo2, hi2)));
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
		const float sqrtVal = std::sqrt(discriminant);
		// include Press, William H., Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery, 
		// "Numerical Recipes in C," Cambridge University Press, 1992.
		const float q = (ddp >= 0) ? (ddp + sqrtVal) : (ddp - sqrtVal);
		// we don't bother testing for division by zero
		const float t1 = q;
		const float t2 = (deltapdot - radius * radius) / q;
    result.x = std::min(t1,t2);
    result.y = std::max(t1,t2);
  }
  
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct BFRayTrace;

struct AbtractPrimitive                         // This is implementation deal, can be any
{
  static constexpr uint32_t TAG_EMPTY     = 0;  // !!! #REQUIRED by kernel slicer: Empty/Default impl must have zero both tag and offset
  static constexpr uint32_t TAG_TRIANGLES = 1; 
  static constexpr uint32_t TAG_BOXES     = 2; 
  static constexpr uint32_t TAG_SPHERES   = 3; 

  AbtractPrimitive(){}  // Dispatching on GPU hierarchy must not have destructors, especially virtual 
  virtual ~AbtractPrimitive(){}     

  virtual uint32_t GetTag() const { return 0; };
  virtual uint32_t Intersect(float4 rayPosAndNear,float4 rayDirAndFar, CRT_LeafInfo info, CRT_Hit* pHit, BFRayTrace* pData) const { return TAG_EMPTY; }

  uint32_t m_tag    = TAG_EMPTY;
  uint32_t m_primId = 0;
  uint32_t dummy1;
  uint32_t dummy2;
  float4   boxMin;
  float4   boxMax;
};

struct BLASInfo
{
  uint32_t startPrim;
  uint32_t sizePrims;
  uint32_t startAABB;
  uint32_t sizeAABBs;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct BFRayTrace : public ISceneObject
{
  BFRayTrace(){}
  ~BFRayTrace();

  const char* Name() const override { return "BFRayTrace"; }

  void     ClearGeom() override { primitives.clear(); primitives.reserve(1000); startEnd.clear(); allBoxes.reserve(1024); allBoxes.clear(); } 

  uint32_t AddGeom_Triangles3f(const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber,
                               uint32_t a_flags = BUILD_HIGH, size_t vByteStride = sizeof(float)*3) override;
                               
  void UpdateGeom_Triangles3f(uint32_t a_geomId, const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber,
                              uint32_t a_flags = BUILD_HIGH, size_t vByteStride = sizeof(float)*3) override;
  
  
  uint32_t AddGeom_AABB(uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber, void** a_customPrimPtrs, size_t a_customPrimCount) override;
  void     UpdateGeom_AABB(uint32_t a_geomId, uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber, void** a_customPrimPtrs, size_t a_customPrimCount) override;

  void     ClearScene() override { m_instMatricesFwd.clear(); m_instMatricesInv.clear(); m_instStartEnd.clear(); } 
  void     CommitScene(uint32_t options = BUILD_HIGH) override {}
  uint32_t AddInstanceMotion(uint32_t a_geomId, const LiteMath::float4x4* a_matrices, uint32_t a_matrixNumber) override {return 0;}
  uint32_t AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix) override;
  void     UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix) override {}

  CRT_Hit RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override;
  bool    RayQuery_AnyHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override { return false; }  

  CRT_Hit RayQuery_NearestHitMotion(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar, float time) override { return RayQuery_NearestHit(posAndNear, dirAndFar); }
  bool    RayQuery_AnyHitMotion(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar, float time) override { return RayQuery_AnyHit(posAndNear, dirAndFar); }

  std::vector<float4>            trivets;
  std::vector<uint32_t>          indices;
  std::vector<AbtractPrimitive*> primitives;
  std::vector<BLASInfo>          startEnd;
  std::vector<CRT_AABB>          allBoxes;

  std::vector<BLASInfo>          m_instStartEnd;
  std::vector<float4x4>          m_instMatricesFwd; ///< instance matrices
  std::vector<float4x4>          m_instMatricesInv; ///< inverse instance matrices
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct AABBPrim : public AbtractPrimitive
{
  AABBPrim(float4 a_boxMin, float4 a_boxMax, uint32_t a_primId) 
  { 
    boxMin = a_boxMin; 
    boxMax = a_boxMax; 
    m_tag  = GetTag();
    m_primId = a_primId; 
  }

  uint32_t GetTag() const override { return TAG_BOXES; }      

  uint32_t Intersect(float4 rayPosAndNear, float4 rayDirAndFar, CRT_LeafInfo info, CRT_Hit* pHit, BFRayTrace* pData) const override 
  { 
    const float3 rayDirInv = 1.0f/to_float3(rayDirAndFar);

    const float4 myBoxMin = boxMin;
    const float4 myBoxMax = boxMax;
    const float2 tMinMax  = RayBoxIntersection2( to_float3(rayPosAndNear), rayDirInv, to_float3(myBoxMin), to_float3(myBoxMax) );
    
    if(tMinMax.x <= tMinMax.y && tMinMax.y >= rayPosAndNear.w && tMinMax.x <= rayDirAndFar.w)
    {
      pHit->t      = tMinMax.x;
      pHit->primId = m_primId;    
      return TAG_BOXES; 
    }
    else
      return TAG_EMPTY;
  }
};

struct TrianglePrim : public AbtractPrimitive
{
  TrianglePrim(float4 A, float4 B, float4 C, uint32_t a_triId) { boxMin   = min(A, min(B, C)); 
                                                                 boxMax   = max(A, max(B, C)); 
                                                                 m_tag    = GetTag();  
                                                                 m_primId = a_triId; 
                                                                 }

  uint32_t GetTag()   const override { return TAG_TRIANGLES; }   

  uint32_t Intersect(float4 rayPosAndNear, float4 rayDirAndFar, CRT_LeafInfo info, CRT_Hit* pHit, BFRayTrace* pData) const override 
  { 
    const float3 rayPos = to_float3(rayPosAndNear);
    const float3 rayDir = to_float3(rayDirAndFar);

    const uint32_t A = pData->indices[m_primId*3+0];
    const uint32_t B = pData->indices[m_primId*3+1];
    const uint32_t C = pData->indices[m_primId*3+2];
 
    const float3 A_pos = to_float3(pData->trivets[A]);
    const float3 B_pos = to_float3(pData->trivets[B]);
    const float3 C_pos = to_float3(pData->trivets[C]);
  
    const float3 edge1 = B_pos - A_pos;
    const float3 edge2 = C_pos - A_pos;
    const float3 pvec = cross(rayDir, edge2);
    const float3 tvec = rayPos - A_pos;
    const float3 qvec = cross(tvec, edge1);
  
    const float invDet = 1.0f / dot(edge1, pvec);
    const float v = dot(tvec, pvec) * invDet;
    const float u = dot(qvec, rayDir) * invDet;
    const float t = dot(edge2, qvec) * invDet;
  
    if (v >= -1e-6f && u >= -1e-6f && (u + v <= 1.0f + 1e-6f) && t > rayPosAndNear.w && t < rayDirAndFar.w)
    {
      pHit->t      = t;
      pHit->primId = int(m_primId);
      return TAG_TRIANGLES; 
    }
    else
      return TAG_EMPTY;
  }   
};

struct SpherePrim : public AbtractPrimitive
{
  SpherePrim(float4 a_sphData, uint32_t a_primId) 
  {
    boxMin.x = a_sphData.x - a_sphData.w;
    boxMin.y = a_sphData.y - a_sphData.w;
    boxMin.z = a_sphData.z - a_sphData.w; 
    boxMin.w = 0.0f;

    boxMax.x = a_sphData.x + a_sphData.w;
    boxMax.y = a_sphData.y + a_sphData.w;
    boxMax.z = a_sphData.z + a_sphData.w; 
    boxMax.w = 1.0f;
  }
  
  SpherePrim(float4 a_boxMin, float4 a_boxMax, uint32_t a_primId) 
  { 
    boxMin   = a_boxMin; 
    boxMax   = a_boxMax; 
    m_tag    = GetTag();  
    m_primId = a_primId;
  }

  uint32_t GetTag() const override { return TAG_SPHERES; }     

  uint32_t Intersect(float4 rayPosAndNear, float4 rayDirAndFar, CRT_LeafInfo info, CRT_Hit* pHit, BFRayTrace* pData) const override 
  { 
    float4 sphere = (boxMin + boxMax)*0.5f;
    sphere.w      = (boxMax.x - boxMin.x)*0.5f;
    
    const float2 tm0 = RaySphereHit(to_float3(rayPosAndNear), to_float3(rayDirAndFar), sphere);
    const bool hit   = (tm0.x < tm0.y) && (tm0.y > rayPosAndNear.w) && (tm0.x < rayDirAndFar.w);
    if(hit)
    {
      pHit->t      = tm0.x;
      pHit->primId = m_primId;
      return TAG_SPHERES; 
    }
    else
      return TAG_EMPTY;
  } 
};

struct EmptyPrim : public AbtractPrimitive
{
  EmptyPrim() { m_tag = GetTag();  }  

  uint32_t GetTag() const override { return TAG_EMPTY; }    
  uint32_t Intersect(float4 rayPosAndNear, float4 rayDirAndFar, CRT_LeafInfo info, CRT_Hit* pHit, BFRayTrace* pData) const override { return TAG_EMPTY; }  
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class TestClass 
{
public:

  TestClass(int w, int h);
  virtual ~TestClass(){}

  virtual void InitScene(int numBoxes, int numTris);

  virtual void Render(uint tidX, uint tidY, uint* out_color __attribute__((size("tidX", "tidY"))));
  virtual void RenderBlock(uint tidX, uint tidY, uint* out_color, uint32_t a_numPasses = 1);

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  virtual void kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY); // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  virtual void kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                               int* out_hit, uint tidX, uint tidY);
  
  virtual void kernel_TestColor(const int* in_hit, uint* out_color, uint tidX, uint tidY);
  
  static constexpr int WIN_WIDTH  = 512;
  static constexpr int WIN_HEIGHT = 512;

protected:
  
  static uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

  std::vector<float4>   spheres;  // (1)
  std::vector<float4>   boxes;    // (2)
  std::vector<float4>   trivets;  // (3)
  std::vector<uint32_t> indices;

  std::shared_ptr<ISceneObject> m_pRayTraceImpl;

  float m_widthInv;
  float m_heightInv;

  float m_time1;
};


#endif