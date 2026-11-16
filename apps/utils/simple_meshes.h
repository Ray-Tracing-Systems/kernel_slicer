#pragma once
// Procedural meshes for graphics pipeline samples (host code only): cube, sphere, torus, plane.
// Triangles are counter-clockwise when seen from outside (front faces for LiteRS::FrontFace::COUNTER_CLOCKWISE
// with LiteMath::OpenglToVulkanProjectionMatrixFix() applied to the projection matrix).

#include <vector>
#include <cstdint>
#include <cmath>
#include "LiteMath.h"

namespace SimpleMeshes
{
  using LiteMath::float2;
  using LiteMath::float3;
  using LiteMath::float4;
  using LiteMath::float4x4;

  struct Mesh
  {
    std::vector<float4>   positions;  ///< (x,y,z,1)
    std::vector<float4>   normals;    ///< (x,y,z,0)
    std::vector<float2>   texCoords;
    std::vector<float4>   colors;     ///< per vertex color (albedo)
    std::vector<uint32_t> indices;    ///< triangle list

    void AddVertex(float3 a_pos, float3 a_normal, float2 a_uv, float4 a_color)
    {
      positions.push_back(float4(a_pos.x, a_pos.y, a_pos.z, 1.0f));
      normals.push_back(float4(a_normal.x, a_normal.y, a_normal.z, 0.0f));
      texCoords.push_back(a_uv);
      colors.push_back(a_color);
    }

    /// two triangles (a,b,c) and (a,c,d) of a quad given counter-clockwise
    void AddQuad(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
    {
      indices.push_back(a); indices.push_back(b); indices.push_back(c);
      indices.push_back(a); indices.push_back(c); indices.push_back(d);
    }
  };

  /// cube [-a_halfSize, a_halfSize]^3 with separate vertices for every face (24 vertices), uv in [0,1] on every face
  static inline Mesh CreateCube(float a_halfSize, float4 a_color)
  {
    const float3 normals[6] = { float3(1,0,0), float3(-1,0,0), float3(0,1,0), float3(0,-1,0), float3(0,0,1), float3(0,0,-1) };
    Mesh mesh;
    for(int f = 0; f < 6; f++)
    {
      const float3 n = normals[f];
      const float3 u = (std::abs(n.y) > 0.5f) ? float3(1,0,0) : LiteMath::cross(float3(0,1,0), n); // tangent
      const float3 v = LiteMath::cross(n, u);                                                     // bitangent, (u,v,n) is right-handed
      const uint32_t first = uint32_t(mesh.positions.size());
      const float2 uv[4] = { float2(0,1), float2(1,1), float2(1,0), float2(0,0) };
      const float  su[4] = { -1, 1, 1, -1 };
      const float  sv[4] = { -1, -1, 1, 1 };
      for(int i = 0; i < 4; i++)
        mesh.AddVertex((n + u*su[i] + v*sv[i])*a_halfSize, n, uv[i], a_color);
      mesh.AddQuad(first + 0, first + 1, first + 2, first + 3);
    }
    return mesh;
  }

  static inline Mesh CreateSphere(float a_radius, int a_slices, int a_stacks, float4 a_color)
  {
    Mesh mesh;
    for(int j = 0; j <= a_stacks; j++)
    {
      const float theta = LiteMath::M_PI*float(j)/float(a_stacks);   // from +Y down to -Y
      for(int i = 0; i <= a_slices; i++)
      {
        const float  phi = 2.0f*LiteMath::M_PI*float(i)/float(a_slices);
        const float3 n   = float3(std::sin(theta)*std::cos(phi), std::cos(theta), -std::sin(theta)*std::sin(phi));
        mesh.AddVertex(n*a_radius, n, float2(float(i)/float(a_slices), float(j)/float(a_stacks)), a_color);
      }
    }
    for(int j = 0; j < a_stacks; j++)
    {
      for(int i = 0; i < a_slices; i++)
      {
        const uint32_t a = uint32_t(j*(a_slices + 1) + i);
        const uint32_t b = a + uint32_t(a_slices + 1);
        mesh.AddQuad(a, b, b + 1, a + 1);
      }
    }
    return mesh;
  }

  /// torus around Y axis
  static inline Mesh CreateTorus(float a_majorRadius, float a_minorRadius, int a_majorSegments, int a_minorSegments, float4 a_color)
  {
    Mesh mesh;
    for(int j = 0; j <= a_majorSegments; j++)
    {
      const float  phi = 2.0f*LiteMath::M_PI*float(j)/float(a_majorSegments);
      const float3 dir = float3(std::cos(phi), 0.0f, -std::sin(phi));
      for(int i = 0; i <= a_minorSegments; i++)
      {
        const float  psi = 2.0f*LiteMath::M_PI*float(i)/float(a_minorSegments);
        const float3 n   = dir*std::cos(psi) + float3(0,1,0)*std::sin(psi);
        mesh.AddVertex(dir*a_majorRadius + n*a_minorRadius, n, float2(float(j)/float(a_majorSegments), float(i)/float(a_minorSegments)), a_color);
      }
    }
    for(int j = 0; j < a_majorSegments; j++)
    {
      for(int i = 0; i < a_minorSegments; i++)
      {
        const uint32_t a = uint32_t(j*(a_minorSegments + 1) + i);
        const uint32_t b = a + uint32_t(a_minorSegments + 1);
        mesh.AddQuad(a, b, b + 1, a + 1);
      }
    }
    return mesh;
  }

  /// plane y = 0, [-a_halfSize, a_halfSize]^2, normal +Y
  static inline Mesh CreatePlane(float a_halfSize, float4 a_color)
  {
    Mesh mesh;
    mesh.AddVertex(float3(-a_halfSize, 0,  a_halfSize), float3(0,1,0), float2(0,0), a_color);
    mesh.AddVertex(float3( a_halfSize, 0,  a_halfSize), float3(0,1,0), float2(1,0), a_color);
    mesh.AddVertex(float3( a_halfSize, 0, -a_halfSize), float3(0,1,0), float2(1,1), a_color);
    mesh.AddVertex(float3(-a_halfSize, 0, -a_halfSize), float3(0,1,0), float2(0,1), a_color);
    mesh.AddQuad(0, 1, 2, 3);
    return mesh;
  }

  /// appends a_src transformed by a_matrix to a_dst (normals are transformed by the inverse transpose)
  static inline void Append(Mesh& a_dst, const Mesh& a_src, const float4x4& a_matrix)
  {
    const float4x4 normalMatrix = LiteMath::transpose(LiteMath::inverse4x4(a_matrix));
    const uint32_t first        = uint32_t(a_dst.positions.size());
    for(size_t i = 0; i < a_src.positions.size(); i++)
    {
      a_dst.positions.push_back(a_matrix*a_src.positions[i]);
      float4 n = normalMatrix*a_src.normals[i];
      n.w = 0.0f;
      a_dst.normals.push_back(LiteMath::normalize(n));
      a_dst.texCoords.push_back(a_src.texCoords[i]);
      a_dst.colors.push_back(a_src.colors[i]);
    }
    for(uint32_t index : a_src.indices)
      a_dst.indices.push_back(first + index);
  }

  /// the scene of samples 32 and 33: ground plane, cube, sphere and torus
  static inline Mesh CreateTestScene()
  {
    Mesh scene;
    Append(scene, CreatePlane(6.0f, float4(0.8f, 0.8f, 0.8f, 1.0f)), float4x4());
    Append(scene, CreateCube(0.75f, float4(0.9f, 0.3f, 0.2f, 1.0f)), LiteMath::translate4x4(float3(-2.0f, 0.75f, 0.5f))*LiteMath::rotate4x4Y(0.6f));
    Append(scene, CreateSphere(0.9f, 48, 24, float4(0.2f, 0.5f, 0.9f, 1.0f)), LiteMath::translate4x4(float3(0.3f, 0.9f, -1.2f)));
    Append(scene, CreateTorus(0.9f, 0.3f, 64, 24, float4(0.3f, 0.8f, 0.3f, 1.0f)), LiteMath::translate4x4(float3(2.1f, 1.0f, 1.0f))*LiteMath::rotate4x4X(1.1f));
    return scene;
  }
};
