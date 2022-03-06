#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/cglobals.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL
#include "include/crandom.h"
#include "include/cmaterial.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>

#include "CrossRT.h"    // special include for ray tracing
#include "texture2d.h"  // special include for textures



class Integrator // : public DataClass
{
public:

  Integrator(int a_maxThreads = 1)
  {
    InitRandomGens(a_maxThreads);
    m_pAccelStruct = std::shared_ptr<ISceneObject>(CreateSceneRT(""), [](ISceneObject *p) { DeleteSceneRT(p); } ); 
    m_light.norm = float4(0,-1,0,0);
  }

  ~Integrator()
  {
    m_pAccelStruct = nullptr;
  }

  void InitRandomGens(int a_maxThreads);
  virtual int LoadScene(const char* bvhPath);

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY   __attribute__((size("tidX", "tidY"))));
  void CastSingleRay  (uint tid, const uint* in_pakedXY __attribute__((size("tid"))), 
                                       uint* out_color  __attribute__((size("tid"))) );
  void NaivePathTrace (uint tid, uint a_maxDepth, const uint* in_pakedXY __attribute__((size("tid"))), 
                                                      float4* out_color  __attribute__((size("tid"))) );
  void PathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY __attribute__((size("tid"))), 
                                                       float4* out_color __attribute__((size("tid"))) );

  virtual void PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passNum);
  virtual void CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passNum);
  virtual void NaivePathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum);
  virtual void PathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum);

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  virtual void UpdateMembersPlainData() {}                               // will be overriden in generated class, optional function
  //virtual void UpdateMembersVectorData() {}                              // will be overriden in generated class, optional function
  //virtual void UpdateMembersTexureData() {}                              // will be overriden in generated class, optional function

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);

  void kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar);        // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
  void kernel_InitEyeRay2(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumuThoroughput, RandomGen* gen, uint* rayFlags);        

  bool kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                       Lite_Hit* out_hit, float2* out_bars);

  void kernel_RayTrace2(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar,
                        float4* out_hit1, float4* out_hit2, uint* rayFlags);

  void kernel_GetRayColor(uint tid, const Lite_Hit* in_hit, const uint* in_pakedXY, uint* out_color);

  void kernel_NextBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2, const float4* in_shadeColor,
                         float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput, RandomGen* a_gen, MisData* a_prevMisData, uint* rayFlags);
  
  void kernel_SampleLightSource(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar, 
                                const float4* in_hitPart1, const float4* in_hitPart2, 
                                const uint* rayFlags, 
                                RandomGen* a_gen, float4* out_shadeColor);

  void kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color);

  void kernel_ContributeToImage(uint tid, const float4* a_accumColor, const RandomGen* gen, const uint* in_pakedXY, 
                                float4* out_color);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  static constexpr uint INTEGRATOR_STUPID_PT = 0;
  static constexpr uint INTEGRATOR_SHADOW_PT = 1;
  static constexpr uint INTEGRATOR_MIS_PT    = 2;

  static constexpr uint RAY_FLAG_IS_DEAD      = 0x80000000;
  static constexpr uint RAY_FLAG_OUT_OF_SCENE = 0x40000000;
  static constexpr uint RAY_FLAG_HIT_LIGHT    = 0x20000000;
  static constexpr uint RAY_FLAG_HAS_NON_SPEC = 0x10000000; // at least one bounce was non specular
  //static constexpr uint RAY_FLAG_DUMMY        = 0x08000000;
  //static constexpr uint RAY_FLAG_DUMMY        = 0x04000000;
  //static constexpr uint RAY_FLAG_DUMMY        = 0x02000000;
  //static constexpr uint RAY_FLAG_DUMMY        = 0x01000000;

  static inline bool isDeadRay     (uint a_flags)  { return (a_flags & RAY_FLAG_IS_DEAD) != 0; }
  static inline bool hasNonSpecular(uint a_flags)  { return (a_flags & RAY_FLAG_HAS_NON_SPEC) != 0; }
  static inline uint extractMatId(uint a_flags)    { return (a_flags & 0x00FFFFFF); }       
  static inline uint packMatId(uint a_flags, uint a_matId) { return (a_flags & 0xFF000000) | (a_matId & 0x00FFFFFF); }       
  static inline uint maxMaterials()             { return 0x00FFFFFF+1; }

  void SetIntegratorType(const uint a_type) { m_intergatorType = a_type; }
  void SetViewport(int a_xStart, int a_yStart, int a_width, int a_height) 
  { 
    m_winStartX = a_xStart; 
    m_winStartY = a_yStart;
    m_winWidth  = a_width;  // todo: remember a_width for first call as pitch and dont change pitch anymore?
    m_winHeight = a_height;
  }

protected:

  int m_winStartX;
  int m_winStartY;
  int m_winWidth;
  int m_winHeight;

  float LightPdfSelectRev(int a_lightId);
  float LightEvalPDF(int a_lightId, float3 ray_pos, float3 ray_dir, const SurfaceHit* pSurfaceHit);

  BsdfSample MaterialSampleAndEval(int a_materialId, float4 rands, float3 v, float3 n, float2 tc);
  BsdfEval   MaterialEval         (int a_materialId, float3 l,     float3 v, float3 n, float2 tc);

  float3 m_camPos = float3(0.0f, 0.85f, 4.5f);
  void InitSceneMaterials(int a_numSpheres, int a_seed = 0);

  std::vector<GLTFMaterial>   m_materials;
  std::vector<uint32_t>        m_matIdOffsets;  ///< offset = m_matIdOffsets[geomId]
  std::vector<uint32_t>        m_matIdByPrimId; ///< matId  = m_matIdByPrimId[offset + primId]
  std::vector<uint32_t>        m_triIndices;    ///< (A,B,C) = m_triIndices[(offset + primId)*3 + 0/1/2]

  std::vector<uint32_t>        m_vertOffset;    ///< vertOffs = m_vertOffset[geomId]
  std::vector<float4>          m_vNorm4f;       ///< vertNorm = m_vNorm4f[vertOffs + vertId]
  std::vector<float2>          m_vTexc2f;       ///< vertTexc = m_vTexc2f[vertOffs + vertId]

  float4x4                     m_projInv;
  float4x4                     m_worldViewInv;
  std::vector<RandomGen>       m_randomGens;
  std::vector<float4x4>        m_normMatrices; ///< per instance normal matrix, local to world

  std::shared_ptr<ISceneObject> m_pAccelStruct = nullptr;

  RectLightSource m_light;
  uint m_intergatorType = INTEGRATOR_STUPID_PT;

  float naivePtTime  = 0.0f;
  float shadowPtTime = 0.0f; 

  //// textures
  //
  std::vector< std::shared_ptr<ITexture2DCombined> > m_textures; ///< all textures, right now represented via combined image/sampler

};

#endif