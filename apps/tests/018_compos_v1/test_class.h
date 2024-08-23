#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL


struct IRayTraceImpl 
{
  IRayTraceImpl(){}
  virtual ~IRayTraceImpl(){}

  virtual void InitBoxesAndTris(int numBoxes, int numTris) = 0;
  virtual int  RayTrace(float4 rayPosAndNear, float4 rayDirAndFar) = 0;

  virtual IRayTraceImpl* UnderlyingImpl(int id) { return this; }
};

struct BFRayTrace : public IRayTraceImpl
{
  BFRayTrace(){}
  ~BFRayTrace(){}

  void InitBoxesAndTris(int numBoxes, int numTris) override;
  int  RayTrace(float4 rayPosAndNear, float4 rayDirAndFar) override;
  IRayTraceImpl* UnderlyingImpl(int id) override { return this; }

//protected:
  
  std::vector<float4> boxes;
  std::vector<float4> trivets;
  float testOffset = 1.0f;
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

  virtual void kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY); // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  virtual void kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                               int* out_hit, uint tidX, uint tidY);
  
  virtual void kernel_TestColor(const int* in_hit, uint* out_color, uint tidX, uint tidY);

protected:

  std::shared_ptr<IRayTraceImpl> m_pRayTraceImpl;

  float m_widthInv;
  float m_heightInv;

  float m_time1;
  float m_time2;
};


#endif