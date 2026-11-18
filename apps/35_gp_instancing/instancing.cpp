#include "instancing.h"
#include "simple_meshes.h"

#include <chrono>
#include <cmath>
#include <string>

static uint32_t Hash(uint32_t a_x) // integer hash, deterministic random numbers for host code
{
  a_x ^= a_x >> 16; a_x *= 0x7feb352dU;
  a_x ^= a_x >> 15; a_x *= 0x846ca68bU;
  a_x ^= a_x >> 16;
  return a_x;
}

static float Random01(uint32_t a_seed) { return float(Hash(a_seed) & 0xFFFFFF)/float(0x1000000); }

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

void InstancedField::SetViewport(int a_width, int a_height)
{
  // meshes, all of them fit into the unit sphere
  //
  m_groundIndices = AppendMesh(SimpleMeshes::CreatePlane(20.0f, float4(1.0f)),             m_positions, m_normals);
  m_cubeIndices   = AppendMesh(SimpleMeshes::CreateCube(0.6f, float4(1.0f)),              m_positions, m_normals);
  m_sphereIndices = AppendMesh(SimpleMeshes::CreateSphere(1.0f, 24, 12, float4(1.0f)),    m_positions, m_normals);
  m_torusIndices  = AppendMesh(SimpleMeshes::CreateTorus(0.7f, 0.3f, 32, 12, float4(1.0f)), m_positions, m_normals);

  // instances on a grid, the mesh of every cell is chosen randomly; instances of the same mesh are stored together
  //
  const int   GRID    = 32;
  const float SPACING = 1.1f;
  const float4 palette[MESH_NUM] = { float4(0.90f, 0.35f, 0.20f, 1.0f), float4(0.25f, 0.55f, 0.95f, 1.0f), float4(0.35f, 0.85f, 0.35f, 1.0f) };
  for(uint32_t mesh = 0; mesh < MESH_NUM; mesh++)
  {
    m_firstInstance[mesh] = uint32_t(m_instPosScale.size());
    for(int j = 0; j < GRID; j++)
    {
      for(int i = 0; i < GRID; i++)
      {
        const uint32_t cell = uint32_t(j*GRID + i);
        if(Hash(cell*7 + 1) % MESH_NUM != mesh)
          continue;
        const float scale = 0.30f + 0.15f*Random01(cell*7 + 2);
        const float x     = (float(i) - 0.5f*float(GRID - 1))*SPACING;
        const float z     = (float(j) - 0.5f*float(GRID - 1))*SPACING;
        const float tint  = 0.7f + 0.3f*Random01(cell*7 + 3);
        m_instPosScale.push_back(float4(x, scale, z, scale));                       // y = scale: the object lies on the ground
        m_instAnim.push_back(float4(2.0f*Random01(cell*7 + 4) - 1.0f,                // rotation speed
                                    6.2831853f*Random01(cell*7 + 5),                  // phase
                                    0.6f*Random01(cell*7 + 6), 0.0f));                // bounce height
        m_instColor.push_back(palette[mesh]*tint);
      }
    }
    m_instanceNum[mesh] = uint32_t(m_instPosScale.size()) - m_firstInstance[mesh];
  }

  const float3   camPos = float3(0.0f, 9.0f, 21.0f);
  const float4x4 view   = LiteMath::lookAt(camPos, float3(0.0f, 0.0f, 2.0f), float3(0.0f, 1.0f, 0.0f));
  const float4x4 proj   = LiteMath::OpenglToVulkanProjectionMatrixFix()*LiteMath::perspectiveMatrix(45.0f, float(a_width)/float(a_height), 0.1f, 100.0f);
  m_viewProj = proj*view;
  m_camPos   = float4(camPos.x, camPos.y, camPos.z, 1.0f);
  m_lightDir = normalize(float4(-0.5f, 1.0f, 0.6f, 0.0f));
  m_fogColor = float4(0.62f, 0.70f, 0.80f, 1.0f);

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::BACK;
  m_pipeline.clearColor = m_fogColor;
}

void InstancedField::Render(int a_width, int a_height, float a_time, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  LiteRS::Clear(m_pipeline);
  LiteRS::DrawIndexedTriangles(m_pipeline, this, &InstancedField::kernelVS_Ground, &InstancedField::kernelPS_Ground,
                               m_groundIndices.data(), uint32_t(m_groundIndices.size()));

  // one draw per mesh: (indices of one instance, number of instances, arguments of the vertex shader)
  //
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &InstancedField::kernelVS_Object, &InstancedField::kernelPS_Object,
                                        m_cubeIndices.data(), uint32_t(m_cubeIndices.size()), m_instanceNum[MESH_CUBE],
                                        m_firstInstance[MESH_CUBE], a_time);
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &InstancedField::kernelVS_Object, &InstancedField::kernelPS_Object,
                                        m_sphereIndices.data(), uint32_t(m_sphereIndices.size()), m_instanceNum[MESH_SPHERE],
                                        m_firstInstance[MESH_SPHERE], a_time);
  LiteRS::DrawIndexedInstancedTriangles(m_pipeline, this, &InstancedField::kernelVS_Object, &InstancedField::kernelPS_Object,
                                        m_torusIndices.data(), uint32_t(m_torusIndices.size()), m_instanceNum[MESH_TORUS],
                                        m_firstInstance[MESH_TORUS], a_time);

  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

/// a_instId is 0, 1, ... inside one draw; data of the instance are at a_firstInstance + a_instId
float4 InstancedField::kernelVS_Object(uint32_t a_vertId, uint32_t a_instId, uint32_t a_firstInstance, float a_time, ShadeVaryings* a_out) const
{
  const uint32_t instId   = a_firstInstance + a_instId;
  const float4   posScale = m_instPosScale[instId];
  const float4   anim     = m_instAnim[instId];

  const float    angle    = anim.x*a_time + anim.y;
  const float    lift     = anim.z*std::abs(std::sin(2.0f*a_time + anim.y));
  const float4x4 rotation = LiteMath::rotate4x4Y(angle)*LiteMath::rotate4x4X(0.5f*angle);

  const float4   pos      = m_positions[a_vertId];
  const float4   worldPos = rotation*float4(pos.x*posScale.w, pos.y*posScale.w, pos.z*posScale.w, 1.0f) + float4(posScale.x, posScale.y + lift, posScale.z, 0.0f);

  a_out->normal   = rotation*m_normals[a_vertId];
  a_out->worldPos = worldPos;
  a_out->color    = m_instColor[instId];
  return m_viewProj*worldPos;
}

float4 InstancedField::kernelVS_Ground(uint32_t a_vertId, ShadeVaryings* a_out) const
{
  const float4 worldPos = m_positions[a_vertId];
  a_out->normal   = m_normals[a_vertId];
  a_out->worldPos = worldPos;
  a_out->color    = float4(0.85f, 0.85f, 0.80f, 1.0f);
  return m_viewProj*worldPos;
}

float4 InstancedField::kernelPS_Object(const ShadeVaryings& a_in) const
{
  const float4 n        = normalize(a_in.normal);
  const float4 v        = normalize(m_camPos - a_in.worldPos);
  const float4 h        = normalize(m_lightDir + v);
  const float  diffuse  = std::max(dot(n, m_lightDir), 0.0f);
  const float  specular = 0.35f*std::pow(std::max(dot(n, h), 0.0f), 32.0f);
  const float  ambient  = 0.25f + 0.10f*n.y;
  const float4 color    = a_in.color*(ambient + 0.75f*diffuse) + float4(specular, specular, specular, 0.0f);
  const float  fog      = clamp((length(m_camPos - a_in.worldPos) - 15.0f)/25.0f, 0.0f, 1.0f);
  const float4 res      = color*(1.0f - fog) + m_fogColor*fog;
  return float4(res.x, res.y, res.z, 1.0f);
}

float4 InstancedField::kernelPS_Ground(const ShadeVaryings& a_in) const
{
  const int    cx      = int(std::floor(a_in.worldPos.x/1.1f + 0.5f));
  const int    cz      = int(std::floor(a_in.worldPos.z/1.1f + 0.5f));
  const float  checker = ((cx + cz) & 1) == 0 ? 1.0f : 0.8f;
  const float  diffuse = std::max(dot(normalize(a_in.normal), m_lightDir), 0.0f);
  const float4 color   = a_in.color*checker*(0.3f + 0.7f*diffuse);
  const float  fog     = clamp((length(m_camPos - a_in.worldPos) - 15.0f)/25.0f, 0.0f, 1.0f);
  const float4 res     = color*(1.0f - fog) + m_fogColor*fog;
  return float4(res.x, res.y, res.z, 1.0f);
}

void InstancedField::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = m_color[int2(x, y)] & 0x00FFFFFF;
  }
}

void InstancedField::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
