#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
#include "Image2d.h"
#include "raster.h"

// LiteMath and LiteRS names are used in all possible ways: with explicit namespace, after using-declaration,
// after using-directive, via namespace alias and via argument dependent lookup (see test_class.cpp)
//
using namespace LiteMath;           // using-directive: float4, float4x4, dot, ... without 'LiteMath::'
using LiteImage::Image2D;
using LiteRS::Pipeline;             // using-declaration of type: member 'Pipeline m_pipeline'
using LiteRS::DrawIndexedTriangles; // using-declaration of command
namespace lm = LiteMath;            // namespace alias: 'lm::normalize(...)'

struct QuadVaryings  // vertex shader output, pixel shader input
{
  float4 color;
  float2 uv;
};

struct TriVaryings
{
  float4 color;
};

/**
\brief Indexed quad and triangle drawn by graphics pipeline (LiteRS), LiteMath and LiteRS names with and without namespace.
*/
class NamespaceRender
{
public:
  NamespaceRender() {}
  virtual ~NamespaceRender() {}

  void SetViewport(int a_width, int a_height); ///< host only: images, pipeline state and geometry

  virtual void Render(int a_width, int a_height, float a_angle, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

protected:

  LiteMath::float4 kernelVS_Quad(uint32_t a_vertId, float a_angle, QuadVaryings* a_out) const;
  float4           kernelPS_Quad(const QuadVaryings& a_in) const;
  float4           kernelVS_Tri (uint32_t a_vertId, float a_angle, TriVaryings* a_out) const;
  LiteMath::float4 kernelPS_Tri (const TriVaryings& a_in) const;
  void             kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  std::vector<LiteMath::float4> m_quadPos;
  std::vector<float4>           m_quadColors;
  std::vector<uint32_t>         m_quadIndices;
  std::vector<float4>           m_triPos;
  Image2D<float4>               m_color;
  Image2D<float>                m_depth;
  Pipeline                      m_pipeline;
  float4x4                      m_viewProj;
  float                         m_timeRender = 0.0f;
};
