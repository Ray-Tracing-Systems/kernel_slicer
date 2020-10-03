#include <iostream>
#include <fstream>
#include "include/LiteMath.h"

using namespace LiteMath;
#define INV_TWOPI  0.15915494309189533577f

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct MisDataT
{
  float matSamplePdf;       ///< previous angle pdf  (pdfW) that were used for sampling material.  
  float cosThetaPrev;       ///< previous angle cos; it allow to compute projected angle pdf (pdfWP = pdfW/cosThetaPrev);
  int   prevMaterialOffset; ///< offset in material buffer to material leaf (elemental brdf) that were sampled on prev bounce; it is needed to disable caustics;
  int   isSpecular;         ///< indicate if bounce was pure specular;

} MisData;

static inline float3 make_float3(float a, float b, float c) { return float3{a,b,c};   }

struct Lite_Hit
{
  float t;
  int   primId; 
  int   instId;
  int   geomId;
};


inline Lite_Hit RayTraceImpl(float3 rayPos, float3 rayDir) 
{
  // some imple here ... 
  //
  Lite_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t = length(rayPos) + length(rayDir);
  return res;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const int tid = 0;

int g_someGlobalVariable = 0;
int g_someGlobalVariableReferencedInFunction = 0;


class TestClass // : public DataClass
{
public:
  
  float3 PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, unsigned int flags);

  void  kernel_InitAccumData(float3* accumColor, float3* accumuThoroughput, float3* currColor);

  void  kernel_RayTrace(const float3* in_ray_pos, const float3* in_ray_dir, 
                        Lite_Hit* out_hit);
  
  void  kernel_TestColor(const Lite_Hit* in_hit, float3* out_color);

private:
};

void TestClass::kernel_InitAccumData(float3* accumColor, float3* accumuThoroughput, float3* currColor)
{
  accumColor[tid]        = make_float3(0,0,0);
  accumuThoroughput[tid] = make_float3(1,1,1);
  currColor[tid]         = make_float3(0,0,0);
  g_someGlobalVariableReferencedInFunction = 55;
}

void  TestClass::kernel_RayTrace(const float3* in_ray_pos, const float3* in_ray_dir, 
                                 Lite_Hit* out_hit)
{
  const float3 ray_pos = in_ray_pos[tid];
  const float3 ray_dir = in_ray_dir[tid];

  out_hit[tid] = RayTraceImpl(ray_pos, ray_dir);
}

void  TestClass::kernel_TestColor(const Lite_Hit* in_hit, float3* out_color)
{
  float x = 2.0f;
  if(in_hit[tid].primId != -1)
    out_color[tid] = make_float3(1,1,1);
  else
    out_color[tid] = make_float3(0,0,0);
} 


float3 TestClass::PathTrace(float3 ray_pos, float3 ray_dir, MisData misPrev, int a_currDepth, unsigned int flags)
{
  //#pragma hycc exclude(depth, a_currDepth, m_maxDepth)

  float3 accumColor, accumuThoroughput, currColor;
  kernel_InitAccumData(&accumColor, &accumuThoroughput, &currColor);

  Lite_Hit hit;
  kernel_RayTrace(&ray_pos, &ray_dir, 
                  &hit);

  kernel_TestColor(&hit, &accumColor);

  return accumColor;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_class_exec()
{
  TestClass test;
  float3 testData = test.PathTrace(float3(0,10,5), float3(0,0,1), MisData(), 0, 0);
  std::cout << testData.x << " " << testData.y << " " << testData.z << std::endl;
  return;
}