#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
#include "Image2d.h"
#include "raster.h"

using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::int2;
using LiteMath::clamp;
using LiteMath::normalize;
using LiteMath::dot;
using LiteImage::Image2D;

struct ShadeVaryings  // vertex shader output, pixel shader input
{
  float4 normal;
  float4 color;
};

/**
\brief Indexed instanced draws (LiteRS::DrawIndexedInstancedTriangles) with data of instances in member vectors:
         * cubes and spheres share vertex buffers and the vertex shader, but have different index buffers;
         * the ground is an ordinary indexed draw in the same pass;
         * a depth only instanced pass of small spheres in front of the camera cuts round holes in the image.
*/
class InstancedMeshes
{
public:
  InstancedMeshes() {}
  virtual ~InstancedMeshes() {}

  void SetViewport(int a_width, int a_height); ///< host only: images, pipeline state, geometry and instances

  virtual void Render(int a_width, int a_height, float a_angle, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

protected:

  float4 kernelVS_Mesh(uint32_t a_vertId, uint32_t a_instId, uint32_t a_firstInstance, float a_angle, ShadeVaryings* a_out) const;
  float4 kernelVS_Ground(uint32_t a_vertId, ShadeVaryings* a_out) const;
  float4 kernelVS_Occluder(uint32_t a_vertId, uint32_t a_instId) const; ///< depth only pass
  float4 kernelPS_Shade(const ShadeVaryings& a_in) const;
  virtual void kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  std::vector<float4>   m_positions;      ///< vertices of all meshes
  std::vector<float4>   m_normals;
  std::vector<uint32_t> m_groundIndices;  ///< index buffers refer to the common vertex buffers
  std::vector<uint32_t> m_cubeIndices;
  std::vector<uint32_t> m_sphereIndices;

  std::vector<float4x4> m_instMatrices;   ///< cubes, then spheres
  std::vector<float4>   m_instColors;
  std::vector<float4>   m_occluders;      ///< (x, y, z) -- center in clip space, w -- radius
  uint32_t              m_cubeNum   = 0;
  uint32_t              m_sphereNum = 0;

  float4x4              m_viewProj;
  float4                m_lightDir;

  Image2D<float4>       m_color;
  Image2D<float>        m_depth;
  LiteRS::Pipeline      m_pipeline;
  float                 m_timeRender = 0.0f;
};
