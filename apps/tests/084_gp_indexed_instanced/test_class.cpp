#include "test_class.h"
#include "simple_meshes.h"

#include <chrono>
#include <cmath>
#include <string>

/// appends vertices of a_mesh to the common vertex buffers and returns its index buffer
static std::vector<uint32_t> AppendMesh(const SimpleMeshes::Mesh& a_mesh, std::vector<float4>& a_positions, std::vector<float4>& a_normals)
{
  const uint32_t first = uint32_t(a_positions.size());
  for(size_t i = 0; i < a_mesh.positions.size(); i++)
  {
    a_positions.push_back(a_mesh.positions[i]);
    a_normals.push_back(a_mesh.normals[i]);
  }
  std::vector<uint32_t> indices(a_mesh.indices.size());
  for(size_t i = 0; i < a_mesh.indices.size(); i++)
    indices[i] = first + a_mesh.indices[i];
  return indices;
}

void InstancedMeshes::SetViewport(int a_width, int a_height)
{
  m_groundIndices = AppendMesh(SimpleMeshes::CreatePlane(4.0f, float4(1.0f)),          m_positions, m_normals);
  m_cubeIndices   = AppendMesh(SimpleMeshes::CreateCube(0.3f, float4(1.0f)),           m_positions, m_normals);
  m_sphereIndices = AppendMesh(SimpleMeshes::CreateSphere(0.35f, 16, 8, float4(1.0f)), m_positions, m_normals);

  // 5x5 grid: cubes in the cells with even (i + j), spheres in the other ones
  //
  for(int kind = 0; kind < 2; kind++)
  {
    for(int j = 0; j < 5; j++)
    {
      for(int i = 0; i < 5; i++)
      {
        if((i + j) % 2 != kind)
          continue;
        const float3 pos = float3(float(i - 2)*1.2f, 0.4f + 0.1f*float(i), float(j - 2)*1.2f);
        m_instMatrices.push_back(LiteMath::translate4x4(pos)*LiteMath::rotate4x4Y(0.3f*float(i*5 + j)));
        m_instColors.push_back(float4(0.2f + 0.2f*float(i), 0.3f + 0.15f*float(j), kind == 0 ? 0.9f : 0.2f, 1.0f));
      }
    }
    if(kind == 0)
      m_cubeNum = uint32_t(m_instMatrices.size());
  }
  m_sphereNum = uint32_t(m_instMatrices.size()) - m_cubeNum;

  m_occluders = { float4(-0.6f, -0.6f, 0.05f, 0.15f), float4(0.6f, -0.5f, 0.05f, 0.1f), float4(0.0f, 0.7f, 0.05f, 0.2f) };

  const float4x4 view = LiteMath::lookAt(float3(0.0f, 5.0f, 8.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));
  const float4x4 proj = LiteMath::OpenglToVulkanProjectionMatrixFix()*LiteMath::perspectiveMatrix(45.0f, float(a_width)/float(a_height), 0.1f, 100.0f);
  m_viewProj = proj*view;
  m_lightDir = normalize(float4(0.4f, 1.0f, 0.5f, 0.0f));

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::BACK;
  m_pipeline.clearColor = float4(0.1f, 0.1f, 0.15f, 1.0f);
}

void InstancedMeshes::Render(int a_width, int a_height, float a_angle, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  LiteRS::Clear(m_pipeline);
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &InstancedMeshes::kernelVS_Occluder,
                                        m_sphereIndices.data(), uint32_t(m_sphereIndices.size()), uint32_t(m_occluders.size()));
  LiteRS::DrawIndexedTriangles(m_pipeline, this, &InstancedMeshes::kernelVS_Ground, &InstancedMeshes::kernelPS_Shade,
                               m_groundIndices.data(), uint32_t(m_groundIndices.size()));
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &InstancedMeshes::kernelVS_Mesh, &InstancedMeshes::kernelPS_Shade,
                                        m_cubeIndices.data(), uint32_t(m_cubeIndices.size()), m_cubeNum, 0, a_angle);
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &InstancedMeshes::kernelVS_Mesh, &InstancedMeshes::kernelPS_Shade,
                                        m_sphereIndices.data(), uint32_t(m_sphereIndices.size()), m_sphereNum, m_cubeNum, a_angle);
  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

float4 InstancedMeshes::kernelVS_Mesh(uint32_t a_vertId, uint32_t a_instId, uint32_t a_firstInstance, float a_angle, ShadeVaryings* a_out) const
{
  const uint32_t instId = a_firstInstance + a_instId;
  const float4x4 model  = m_instMatrices[instId]*LiteMath::rotate4x4X(a_angle);
  a_out->normal = model*m_normals[a_vertId];
  a_out->color  = m_instColors[instId];
  return m_viewProj*(model*m_positions[a_vertId]);
}

float4 InstancedMeshes::kernelVS_Ground(uint32_t a_vertId, ShadeVaryings* a_out) const
{
  a_out->normal = m_normals[a_vertId];
  a_out->color  = float4(0.7f, 0.7f, 0.7f, 1.0f);
  return m_viewProj*m_positions[a_vertId];
}

float4 InstancedMeshes::kernelVS_Occluder(uint32_t a_vertId, uint32_t a_instId) const
{
  const float4 occluder = m_occluders[a_instId];
  const float4 p        = m_positions[a_vertId] - float4(0.0f, 0.0f, 0.0f, 1.0f); // sphere of radius 0.35 around the origin
  const float  scale    = occluder.w/0.35f;
  return float4(occluder.x + p.x*scale, occluder.y + p.y*scale, occluder.z, 1.0f);
}

float4 InstancedMeshes::kernelPS_Shade(const ShadeVaryings& a_in) const
{
  const float diffuse = std::max(dot(normalize(a_in.normal), m_lightDir), 0.0f);
  const float4 color  = a_in.color*(0.25f + 0.75f*diffuse);
  return float4(color.x, color.y, color.z, 1.0f);
}

static inline uint32_t PackRGB8(float4 a_color) // R in the lowest byte
{
  const uint32_t r = uint32_t(clamp(a_color.x, 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t g = uint32_t(clamp(a_color.y, 0.0f, 1.0f)*255.0f + 0.5f);
  const uint32_t b = uint32_t(clamp(a_color.z, 0.0f, 1.0f)*255.0f + 0.5f);
  return r | (g << 8) | (b << 16);
}

void InstancedMeshes::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = PackRGB8(m_color[int2(x, y)]);
  }
}

void InstancedMeshes::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
