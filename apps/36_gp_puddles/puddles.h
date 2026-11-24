#pragma once
#include <vector>
#include <cstdint>
#include <memory>

#include "LiteMath.h"
#include "Image2d.h"
#include "raster.h"
#include "CrossRT.h"

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::int2;
using LiteMath::clamp;
using LiteMath::normalize;
using LiteMath::dot;
using LiteMath::cross;
using LiteImage::Image2D;

struct GBufferVaryings  // vertex shader output, pixel shader input
{
  float4 pos;           ///< world space position
  float4 normal;        ///< world space normal
  float4 color;         ///< albedo; w = 1 for surfaces which may have puddles (the ground)
};

struct GBufferOut       // pixel shader output: color targets 0, 1, 2
{
  float4 pos;
  float4 normal;        ///< w = 1 for surfaces which may have puddles
  float4 albedo;        ///< w = 1 for geometry, 0 for background
};

/**
\brief Hybrid rendering: G-buffer of a scene with many instances is drawn by graphics pipeline (one instanced draw for
       every mesh), then a compute kernel shades it and traces rays: shadows from the sun and reflections in puddles
       on the ground.

The scene is loaded from Hydra XML: meshes are stored in common vertex and index buffers, instances of mesh k occupy
[m_meshFirstInst[k], m_meshFirstInst[k] + m_meshInstNum[k]) of instance arrays, the same instances are added to the
acceleration structure in the same order, so 'CRT_Hit::instId' is the index of instance arrays.
*/
class PuddleRender
{
public:
  PuddleRender()
  {
    m_pAccelStruct = std::shared_ptr<ISceneObject>(CreateSceneRT(""));
  }
  virtual ~PuddleRender() { m_pAccelStruct = nullptr; }

  bool LoadScene(const char* a_scenePath);     ///< host only: meshes, instances, materials, camera, acceleration structure
  void SetViewport(int a_width, int a_height); ///< host only: images, camera matrices and pipeline state

  virtual void RenderGBuffer(int a_width, int a_height);                                                   ///< rasterization
  virtual void ShadeAndReflect(int a_width, int a_height, uint32_t* a_out [[size("a_width*a_height")]]); ///< ray tracing

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

  static constexpr uint32_t MAX_MESHES = 16;

protected:

  float4     kernelVS_GBuffer(uint32_t a_vertId, uint32_t a_instId, uint32_t a_vertOffset, uint32_t a_firstInst, GBufferVaryings* a_out) const;
  GBufferOut kernelPS_GBuffer(const GBufferVaryings& a_in) const;
  virtual void kernel2D_Shade(int a_width, int a_height, uint32_t* a_out);

  float4 SkyColor(float4 a_dir) const;
  float  PuddleMask(float a_x, float a_z) const;
  float4 HitColor(CRT_Hit a_hit, float4 a_rayDir) const;   ///< color of a surface seen in a puddle
  float  SunVisibility(float4 a_pos, float4 a_normal) const;

  // geometry of all meshes
  //
  std::vector<float4>   m_positions;
  std::vector<float4>   m_normals;
  std::vector<uint32_t> m_indices;      ///< indices of mesh k start at m_meshFirstIndex[k], they are local to the mesh

  uint32_t              m_meshNum = 0;
  uint32_t              m_meshFirstIndex[MAX_MESHES];
  uint32_t              m_meshIndexNum  [MAX_MESHES];
  uint32_t              m_meshVertOffset[MAX_MESHES];
  uint32_t              m_meshFirstInst [MAX_MESHES];
  uint32_t              m_meshInstNum   [MAX_MESHES];

  // instances, sorted by meshes
  //
  std::vector<float4x4> m_instMatrix;
  std::vector<float4x4> m_instNormalMatrix;
  std::vector<float4>   m_instColor;    ///< albedo; w = 1 for the ground
  std::vector<uint32_t> m_instMesh;

  float4x4              m_viewProj;
  float4                m_camPos;
  float4x4              m_camView;            ///< host only
  float                 m_camFov  = 45.0f;    ///< host only
  float                 m_camNear = 0.01f;    ///< host only
  float                 m_camFar  = 100.0f;   ///< host only
  float4                m_sunDir      = float4(-0.35f, 0.75f, -0.55f, 0.0f);
  float4                m_sunColor    = float4(1.1f, 1.0f, 0.85f, 0.0f);
  float4                m_skyColor    = float4(0.35f, 0.45f, 0.7f, 0.0f);
  float                 m_puddleScale = 0.12f;  ///< frequency of puddle noise on the ground
  float                 m_sceneSize   = 200.0f; ///< maximal length of rays

  Image2D<float4>       m_gPos;
  Image2D<float4>       m_gNormal;
  Image2D<float4>       m_gAlbedo;
  Image2D<float>        m_gDepth;
  LiteRS::Pipeline      m_gbufferPass;

  std::shared_ptr<ISceneObject> m_pAccelStruct = nullptr;

  float                 m_timeGBuffer = 0.0f;
  float                 m_timeShade   = 0.0f;
};
