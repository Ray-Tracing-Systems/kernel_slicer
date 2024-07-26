#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
using LiteMath::float4;
using LiteMath::float3;
using LiteMath::float2;
using LiteMath::uint;
using LiteMath::min;
using LiteMath::max;

#include "CrossRT.h" // special include for ray tracing

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct AbtractPrimitive                           // This is implementation deal, can be any
{
  static constexpr uint32_t TAG_EMPTY     = 0;    // !!! #REQUIRED by kernel slicer: Empty/Default impl must have zero both tag and offset
  static constexpr uint32_t TAG_TRIANGLES = 1; 
  static constexpr uint32_t TAG_BOXES     = 2; 
  static constexpr uint32_t TAG_SPHERES   = 3; 

  AbtractPrimitive(){}  // Dispatching on GPU hierarchy must not have destructors, especially virtual      

  virtual uint32_t GetTag() const { return 0; };

  uint32_t m_tag = TAG_EMPTY;
  uint32_t m_dummy;
  float4   boxMin;
  float4   boxMax;
};

struct AABBPrim : public AbtractPrimitive
{
  AABBPrim(float4 a_boxMin, float4 a_boxMax) { boxMin = a_boxMin; 
                                               boxMax = a_boxMax; 
                                               m_tag  = GetTag();  }
  ~AABBPrim() = delete;  

  uint32_t GetTag()   const override { return TAG_BOXES; }      
};

struct TrianglePrim : public AbtractPrimitive
{
  TrianglePrim(float4 A, float4 B, float4 C) { boxMin = min(A, min(B, C)); 
                                               boxMax = max(A, max(B, C)); 
                                               m_tag = GetTag();  }
  ~TrianglePrim() = delete;  

  uint32_t GetTag()   const override { return TAG_TRIANGLES; }      
};

struct SpherePrim : public AbtractPrimitive
{
  SpherePrim(float4 a_boxMin, float4 a_boxMax) { boxMin = a_boxMin; 
                                                 boxMax = a_boxMax; 
                                                 m_tag = GetTag();  }
  ~SpherePrim() = delete;  

  uint32_t GetTag()   const override { return TAG_SPHERES; }      
};

struct EmptyPrim : public AbtractPrimitive
{
  EmptyPrim() { m_tag = GetTag();  }
  ~EmptyPrim() = delete;  

  uint32_t GetTag()   const override { return TAG_EMPTY; }      
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct BFRayTrace : public ISceneObject
{
  BFRayTrace(){}
  ~BFRayTrace(){}

  const char* Name() const override { return "BFRayTrace"; }

  void     ClearGeom() override { primitives.clear(); }

  uint32_t AddGeom_Triangles3f(const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber,
                               uint32_t a_flags = BUILD_HIGH, size_t vByteStride = sizeof(float)*3) override;
                               
  void UpdateGeom_Triangles3f(uint32_t a_geomId, const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber,
                              uint32_t a_flags = BUILD_HIGH, size_t vByteStride = sizeof(float)*3) override;
  
  
  uint32_t AddGeom_AABB(uint32_t a_typeId, const CRT_AABB8f* boxMinMaxF8, size_t a_boxNumber) override;
  void     UpdateGeom_AABB(uint32_t a_geomId, uint32_t a_typeId, const CRT_AABB8f* boxMinMaxF8, size_t a_boxNumber) override;

  void     ClearScene() override {} 
  void     CommitScene(uint32_t options = BUILD_HIGH) override {}
  uint32_t AddInstanceMotion(uint32_t a_geomId, const LiteMath::float4x4* a_matrices, uint32_t a_matrixNumber) override {return 0;}
  uint32_t AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix) override { return 0;}
  void     UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix) override {}

  CRT_Hit RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar);
  bool    RayQuery_AnyHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) { return false; }  

  CRT_Hit RayQuery_NearestHitMotion(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar, float time) override { return RayQuery_NearestHit(posAndNear, dirAndFar); }
  bool    RayQuery_AnyHitMotion(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar, float time) override { return RayQuery_AnyHit(posAndNear, dirAndFar); }

  std::vector<float4>   spheres;
  std::vector<float4>   boxes;
  std::vector<float4>   trivets;
  std::vector<uint32_t> indices;

  std::vector<AbtractPrimitive> primitives;
};


enum WINDOW_SIZE{WIN_WIDTH = 512, WIN_HEIGHT = 512};
static uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

class TestClass 
{
public:

  TestClass(int w, int h);
  virtual ~TestClass(){}

  virtual void InitScene(int numBoxes, int numTris);

  virtual void BFRT_ReadAndCompute(uint tidX, uint tidY, uint* out_color __attribute__((size("tidX", "tidY"))));
  virtual void BFRT_ReadAndComputeBlock(uint tidX, uint tidY, uint* out_color, uint32_t a_numPasses = 1);

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  virtual void kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY); // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  virtual void kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                               int* out_hit, uint tidX, uint tidY);
  
  virtual void kernel_TestColor(const int* in_hit, uint* out_color, uint tidX, uint tidY);

protected:
  
  std::vector<float4>   spheres;  // (1)
  std::vector<float4>   boxes;    // (2)
  std::vector<float4>   trivets;  // (3)
  std::vector<uint32_t> indices;

  std::shared_ptr<ISceneObject> m_pRayTraceImpl;

  float m_widthInv;
  float m_heightInv;

  float m_time1;
  float m_time2;
};


#endif