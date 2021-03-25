#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL
#include "include/crandom.h"

#include <vector>
#include <iostream>
#include <fstream>

#include "include/bvh.h"

static inline float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar)
{
  const float ymax = zNear * tanf(fovy * 3.14159265358979323846f / 360.0f);
  const float xmax = ymax * aspect;
  const float left = -xmax;
  const float right = +xmax;
  const float bottom = -ymax;
  const float top = +ymax;
  const float temp = 2.0f * zNear;
  const float temp2 = right - left;
  const float temp3 = top - bottom;
  const float temp4 = zFar - zNear;
  float4x4 res;
  res.m_col[0] = float4{ temp / temp2, 0.0f, 0.0f, 0.0f };
  res.m_col[1] = float4{ 0.0f, temp / temp3, 0.0f, 0.0f };
  res.m_col[2] = float4{ (right + left) / temp2,  (top + bottom) / temp3, (-zFar - zNear) / temp4, -1.0 };
  res.m_col[3] = float4{ 0.0f, 0.0f, (-temp * zFar) / temp4, 0.0f };
  return res;
}

enum GEOM_FLAGS{ HAS_TANGENT    = 1,
  UNUSED2        = 2,
  UNUSED4        = 4,
  HAS_NO_NORMALS = 8};

struct SimpleMesh
{
  static const uint64_t POINTS_IN_TRIANGLE = 3;
  SimpleMesh(){}
  SimpleMesh(int a_vertNum, int a_indNum) { Resize(a_vertNum, a_indNum); }

  inline size_t VerticesNum()  const { return vPos4f.size(); }
  inline size_t IndicesNum()   const { return indices.size();  }
  inline size_t TrianglesNum() const { return IndicesNum() / POINTS_IN_TRIANGLE;  }
  inline void   Resize(uint32_t a_vertNum, uint32_t a_indNum)
  {
    vPos4f.resize(a_vertNum);
    vNorm4f.resize(a_vertNum);
    vTang4f.resize(a_vertNum);
    vTexCoord2f.resize(a_vertNum);
    indices.resize(a_indNum);
    matIndices.resize(a_indNum/3);
  };

  inline size_t SizeInBytes() const
  {
    return vPos4f.size()*sizeof(float)*4  +
           vNorm4f.size()*sizeof(float)*4 +
           vTang4f.size()*sizeof(float)*4 +
           vTexCoord2f.size()*sizeof(float)*2 +
           indices.size()*sizeof(int) +
           matIndices.size()*sizeof(int);
  }
  std::vector<LiteMath::float4> vPos4f;      //
  std::vector<LiteMath::float4> vNorm4f;     //
  std::vector<LiteMath::float4> vTang4f;     //
  std::vector<float2>                       vTexCoord2f; //
  std::vector<unsigned int>                 indices;     // size = 3*TrianglesNum() for triangle mesh, 4*TrianglesNum() for quad mesh
  std::vector<unsigned int>                 matIndices;  // size = 1*TrianglesNum()
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TestClass // : public DataClass
{
public:

  TestClass(int a_maxThreads = 1)
  {
    const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, 0.01f, 100.0f);
    m_worldViewProjInv  = inverse4x4(proj);
    InitSpheresScene(10);
    InitRandomGens(a_maxThreads);
  }

  void InitRandomGens(int a_maxThreads);
  int LoadScene(const char* bvhPath, const char* meshPath);

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY);
  void CastSingleRay(uint tid, uint* in_pakedXY, uint* out_color);
  void StupidPathTrace(uint tid, uint a_maxDepth, uint* in_pakedXY, float4* out_color);

  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);

  void kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar);        // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  bool kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                       Lite_Hit* out_hit, const uint* indicesReordered, const float4* meshVerts);
  
  void kernel_GetMaterialColor(uint tid, const Lite_Hit* in_hit, 
                               uint* out_color);

  void kernel_InitAccumData(uint tid, float4* accumColor, float4* accumuThoroughput);
  
  void kernel_NextBounce(uint tid, const Lite_Hit* in_hit, 
                         float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput);

  void kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color);

  void kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, 
                                float4* out_color);

protected:
  float3 camPos = float3(0.0f, 0.85f, 2.75f);
  void InitSpheresScene(int a_numSpheres, int a_seed = 0);

//  BVHTree                      m_bvhTree;
  std::vector<struct BVHNode>  m_nodes;
  std::vector<struct Interval> m_intervals;
  std::vector<uint32_t>        m_indicesReordered;
  std::vector<uint32_t>        m_materialIds;

  std::vector<LiteMath::float4> m_vPos4f;      // copy from m_mesh
  std::vector<LiteMath::float4> m_vNorm4f;     // copy from m_mesh

  float4x4                     m_worldViewProjInv;
  std::vector<SphereMaterial>  spheresMaterials;
  std::vector<RandomGen>       m_randomGens;
};

#endif