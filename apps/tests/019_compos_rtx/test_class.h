#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>

#include "CrossRT.h" // special include for ray tracing
#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL


class TestClass 
{
public:

  TestClass(int w, int h);
  virtual ~TestClass(){ m_pAccelStruct = nullptr; }

  virtual void InitScene();

  virtual void BFRT_ReadAndCompute(uint tidX, uint tidY, uint* out_color __attribute__((size("tidX", "tidY"))));
  virtual void BFRT_ReadAndComputeBlock(uint tidX, uint tidY, uint* out_color, uint32_t a_numPasses = 1);

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  virtual void kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY); // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  virtual void kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                               int* out_hit, uint tidX, uint tidY);
  
  virtual void kernel_TestColor(const int* in_hit, uint* out_color, uint tidX, uint tidY);

protected:

  void InitTris(size_t numTris, std::vector<float4>& verts, std::vector<uint32_t>& indices);

  std::shared_ptr<ISceneObject>  m_pAccelStruct = nullptr;

  float m_widthInv;
  float m_heightInv;

  float m_time1;
  float m_time2;
};

struct BFRayTrace : public ISceneObject
{
  BFRayTrace(){}
  ~BFRayTrace(){}

  void     ClearGeom() override{}

  uint32_t AddGeom_Triangles3f(const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber,
                               uint32_t a_flags = BUILD_HIGH, size_t vByteStride = sizeof(float)*3) override;
                               
  void UpdateGeom_Triangles3f(uint32_t a_geomId, const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber,
                              uint32_t a_flags = BUILD_HIGH, size_t vByteStride = sizeof(float)*3) override {}
  
  void     ClearScene() override {} 
  void     CommitScene(uint32_t options = BUILD_HIGH) override {}
  uint32_t AddInstanceMotion(uint32_t a_geomId, const LiteMath::float4x4* a_matrices, uint32_t a_matrixNumber) override {return 0;}
  uint32_t AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix) override { return 0;}
  void     UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix) override {}

  CRT_Hit RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar);
  bool    RayQuery_AnyHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) { return false; }  

  CRT_Hit RayQuery_NearestHitMotion(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar, float time) override { return RayQuery_NearestHit(posAndNear, dirAndFar); }
  bool    RayQuery_AnyHitMotion(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar, float time) override { return RayQuery_AnyHit(posAndNear, dirAndFar); }
  
  std::vector<float4> trivets;
  float testOffset = 0.0f;
};


#endif