#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>

#include "include/BasicLogic.h"

// Test of inheritance inside a composed class: the main class holds IRayTraceImpl, implemented by
// DerivedRT : MidRT : BaseRT : IRayTraceImpl. Data members and member functions are spread over the
// hierarchy, DerivedRT overrides RayTrace, and BaseRT::RayTraceShadow calls the virtual RayTrace,
// which must resolve to DerivedRT::RayTrace.

struct IRayTraceImpl
{
  IRayTraceImpl(){}
  virtual ~IRayTraceImpl(){}

  virtual void InitBoxesAndTris(int numBoxes, int numTris) = 0;
  virtual int  RayTrace(float4 rayPosAndNear, float4 rayDirAndFar) = 0;
  virtual int  RayTraceShadow(float4 rayPosAndNear, float4 rayDirAndFar) = 0;

  virtual IRayTraceImpl* UnderlyingImpl(int id) { return this; }
};

struct BaseRT : public IRayTraceImpl
{
  BaseRT(){}
  ~BaseRT(){}

  int  RayTrace(float4 rayPosAndNear, float4 rayDirAndFar) override;       // triangles only, overridden in DerivedRT
  int  RayTraceShadow(float4 rayPosAndNear, float4 rayDirAndFar) override; // calls the virtual RayTrace

  int  IntersectTris(float3 rayPos, float3 rayDir, float tNear, float tFar); // helper used by derived classes

  std::vector<float4> trivets;
  float testOffset = 1.0f;
};

struct MidRT : public BaseRT
{
  MidRT(){}
  ~MidRT(){}

  void InitBoxesAndTris(int numBoxes, int numTris) override;                    // host code
  int  IntersectBoxes(float3 rayPos, float3 rayDirInv, float tNear, float tFar); // uses its own data

  std::vector<float4> boxes;
  int   boxStep = 4;
};

struct DerivedRT : public MidRT
{
  DerivedRT(){}
  ~DerivedRT(){}

  int RayTrace(float4 rayPosAndNear, float4 rayDirAndFar) override; // boxes (MidRT) and triangles (BaseRT)

  int hitBias = 1000;
};

class TestClass
{
public:

  TestClass(int w, int h);
  virtual ~TestClass(){}

  virtual void BFRT_ReadAndCompute(uint tidX, uint tidY, uint* out_color __attribute__((size("tidX", "tidY"))));
  virtual void BFRT_ReadAndComputeBlock(uint tidX, uint tidY, uint* out_color, uint32_t a_numPasses = 1);

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  virtual void kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY);
  virtual void kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, int* out_hit, uint tidX, uint tidY);
  virtual void kernel_TestColor(const int* in_hit, uint* out_color, uint tidX, uint tidY);

protected:

  std::shared_ptr<IRayTraceImpl> m_pRayTraceImpl;

  float m_widthInv;
  float m_heightInv;

  float m_time1;
  float m_time2;
};

#endif
