#include "raster_bench.h"
#include "simple_meshes.h"

#include <chrono>
#include <cmath>
#include <string>

/// appends a_mesh to vertex arrays of all scenes and its indices to a_indices
static void AppendMesh(const SimpleMeshes::Mesh& a_mesh, std::vector<float4>& a_pos, std::vector<float4>& a_norm, std::vector<float4>& a_col, std::vector<uint32_t>& a_indices)
{
  const uint32_t first = uint32_t(a_pos.size());
  a_pos.insert (a_pos.end(),  a_mesh.positions.begin(), a_mesh.positions.end());
  a_norm.insert(a_norm.end(), a_mesh.normals.begin(),   a_mesh.normals.end());
  a_col.insert (a_col.end(),  a_mesh.colors.begin(),    a_mesh.colors.end());
  for(uint32_t index : a_mesh.indices)
    a_indices.push_back(first + index);
}

static float4 GridColor(int i, int j, int n)
{
  return float4(0.3f + 0.7f*float(i)/float(n), 0.4f, 0.3f + 0.7f*float(j)/float(n), 1.0f);
}

void RasterBench::SetViewport(int a_width, int a_height, LiteImage::Layout a_layout, int a_binSize)
{
  m_positions.clear();
  m_normals.clear();
  m_colors.clear();
  m_indicesSmall.clear();
  m_indicesMedium.clear();
  m_indicesLarge.clear();

  // SMALL: 16x16 spheres of 64x32 segments, about 520k triangles of 1-4 pixels at 1024x1024
  //
  const int smallN = 16;
  for(int j = 0; j < smallN; j++)
  {
    for(int i = 0; i < smallN; i++)
    {
      SimpleMeshes::Mesh sphere;
      SimpleMeshes::Append(sphere, SimpleMeshes::CreateSphere(0.13f, 64, 32, GridColor(i, j, smallN)),
                           LiteMath::translate4x4(float3(-2.0f + 4.0f*(float(i) + 0.5f)/float(smallN), -2.0f + 4.0f*(float(j) + 0.5f)/float(smallN), 0.0f)));
      AppendMesh(sphere, m_positions, m_normals, m_colors, m_indicesSmall);
    }
  }

  // MEDIUM: 6x6 spheres of 24x12 segments, about 20k triangles of 100 pixels
  //
  const int mediumN = 6;
  for(int j = 0; j < mediumN; j++)
  {
    for(int i = 0; i < mediumN; i++)
    {
      SimpleMeshes::Mesh sphere;
      SimpleMeshes::Append(sphere, SimpleMeshes::CreateSphere(0.3f, 24, 12, GridColor(i, j, mediumN)),
                           LiteMath::translate4x4(float3(-2.0f + 4.0f*(float(i) + 0.5f)/float(mediumN), -2.0f + 4.0f*(float(j) + 0.5f)/float(mediumN), 0.0f)));
      AppendMesh(sphere, m_positions, m_normals, m_colors, m_indicesMedium);
    }
  }

  // LARGE: 16 big rotated planes at different depths drawn from back to front, every pixel passes the depth test 16 times
  //
  const int largeN = 16;
  for(int k = 0; k < largeN; k++)
  {
    const float z = -1.5f + 3.0f*float(k)/float(largeN - 1);
    SimpleMeshes::Mesh plane;
    SimpleMeshes::Append(plane, SimpleMeshes::CreatePlane(2.2f, GridColor(k, largeN - 1 - k, largeN)),
                         LiteMath::translate4x4(float3(0.0f, 0.0f, z))*LiteMath::rotate4x4Z(0.2f*float(k))*LiteMath::rotate4x4X(1.5707963f));
    AppendMesh(plane, m_positions, m_normals, m_colors, m_indicesLarge);
  }

  const float4x4 view = LiteMath::lookAt(float3(0.0f, 0.0f, 5.5f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));
  const float4x4 proj = LiteMath::OpenglToVulkanProjectionMatrixFix()*LiteMath::perspectiveMatrix(45.0f, float(a_width)/float(a_height), 0.1f, 100.0f);
  m_viewProj = proj*view;
  const float3 lightDir = normalize(float3(-0.4f, 0.6f, 1.0f));
  m_lightDir = float4(lightDir.x, lightDir.y, lightDir.z, 0.0f);

  m_color.resize(a_width, a_height, a_layout, a_binSize);
  m_depth.resize(a_width, a_height, a_layout, a_binSize);
  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::NONE; // planes are seen from both sides, spheres are drawn with back faces
  m_pipeline.clearColor = float4(0.1f, 0.1f, 0.12f, 1.0f);
}

void RasterBench::RenderSmall(int a_width, int a_height, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();
  LiteRS::Clear(m_pipeline);
  LiteRS::DrawIndexedTriangles(m_pipeline, this, &RasterBench::kernelVS_Small, &RasterBench::kernelPS_Shade, m_indicesSmall.data(), uint32_t(m_indicesSmall.size()));
  kernel2D_ToOutput(a_width, a_height, a_out);
  m_timeRender[0] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void RasterBench::RenderMedium(int a_width, int a_height, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();
  LiteRS::Clear(m_pipeline);
  LiteRS::DrawIndexedTriangles(m_pipeline, this, &RasterBench::kernelVS_Medium, &RasterBench::kernelPS_Shade, m_indicesMedium.data(), uint32_t(m_indicesMedium.size()));
  kernel2D_ToOutput(a_width, a_height, a_out);
  m_timeRender[1] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void RasterBench::RenderLarge(int a_width, int a_height, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();
  LiteRS::Clear(m_pipeline);
  LiteRS::DrawIndexedTriangles(m_pipeline, this, &RasterBench::kernelVS_Large, &RasterBench::kernelPS_Shade, m_indicesLarge.data(), uint32_t(m_indicesLarge.size()));
  kernel2D_ToOutput(a_width, a_height, a_out);
  m_timeRender[2] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

float4 RasterBench::Transform(uint32_t a_vertId, BenchVaryings* a_out) const
{
  a_out->normal = m_normals[a_vertId];
  a_out->color  = m_colors[a_vertId];
  return m_viewProj*m_positions[a_vertId];
}

float4 RasterBench::kernelVS_Small (uint32_t a_vertId, BenchVaryings* a_out) const { return Transform(a_vertId, a_out); }
float4 RasterBench::kernelVS_Medium(uint32_t a_vertId, BenchVaryings* a_out) const { return Transform(a_vertId, a_out); }
float4 RasterBench::kernelVS_Large (uint32_t a_vertId, BenchVaryings* a_out) const { return Transform(a_vertId, a_out); }

float4 RasterBench::kernelPS_Shade(const BenchVaryings& a_in) const
{
  const float4 n       = normalize(a_in.normal);
  const float  diffuse = std::abs(dot(n, m_lightDir));
  const float4 color   = a_in.color*(0.2f + 0.8f*diffuse);
  return float4(color.x, color.y, color.z, 1.0f);
}

static inline uint32_t PackRGB8(float4 a_color) // R in the lowest byte
{
  const uint32_t r = uint32_t(clamp(a_color.x, 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = uint32_t(clamp(a_color.y, 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = uint32_t(clamp(a_color.z, 0.0f, 1.0f)*255.0f + 0.5f);
  return r | (g << 8) | (b << 16);
}

void RasterBench::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[int2(x, y)]);
  }
}

void RasterBench::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "RenderSmall")
    a_out[0] = m_timeRender[0];
  else if(std::string(a_funcName) == "RenderMedium")
    a_out[0] = m_timeRender[1];
  else if(std::string(a_funcName) == "RenderLarge")
    a_out[0] = m_timeRender[2];
}
