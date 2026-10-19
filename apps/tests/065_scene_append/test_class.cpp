#include "test_class.h"
#include <cfloat>

SceneAppend::SceneAppend()
{
  m_pAccelStruct = std::shared_ptr<ISceneObject>(CreateSceneRT("BVH2Common"), [](ISceneObject* p) { DeleteSceneRT(p); });
}

uint32_t SceneAppend::AddMesh(const std::vector<float4>& a_verts, const std::vector<uint32_t>& a_indices)
{
  return m_pAccelStruct->AddGeom_Triangles3f((const float*)a_verts.data(), a_verts.size(), a_indices.data(), a_indices.size(), BUILD_HIGH, sizeof(float)*4);
}

void SceneAppend::SetInstances(const std::vector<uint32_t>& a_geomIds, const std::vector<float4x4>& a_matrices)
{
  m_pAccelStruct->ClearScene();
  for(size_t i = 0; i < a_geomIds.size(); i++)
    m_pAccelStruct->AddInstance(a_geomIds[i], a_matrices[i]);
  m_pAccelStruct->CommitScene();
}

void SceneAppend::kernel2D_Trace(uint32_t a_width, uint32_t a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(uint32_t y = 0; y < a_height; y++)
  {
    for(uint32_t x = 0; x < a_width; x++)
    {
      const float sceneSize = m_sceneSize;
      const float u = (float(x) + 0.5f)/float(a_width);
      const float v = (float(y) + 0.5f)/float(a_height);
      const float4 rayPos = float4((2.0f*u - 1.0f)*sceneSize, (2.0f*v - 1.0f)*sceneSize, -100.0f, 0.0f);
      const float4 rayDir = float4(0.0f, 0.0f, 1.0f, FLT_MAX);
      const CRT_Hit hit = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);
      a_out[y*a_width + x] = (hit.primId == uint32_t(-1)) ? 0xFFFFFFFFu : ((hit.instId << 20) | (hit.primId & 0xFFFFFu));
    }
  }
}

void SceneAppend::Render(uint32_t a_width, uint32_t a_height, uint32_t* a_out)
{
  kernel2D_Trace(a_width, a_height, a_out);
}
