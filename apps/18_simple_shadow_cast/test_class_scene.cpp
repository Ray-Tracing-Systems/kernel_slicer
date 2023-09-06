#include "test_class.h"
#include "include/crandom.h"

#include "cmesh4.h"
using cmesh4::SimpleMesh;

void TestClass::InitSceneMaterials(int a_numSpheres, int a_seed)
{ 
  ///// create simple materials
  //
  m_materials.resize(10);
  m_materials[0] = float4(0.5f, 0.5f, 0.5f, 0.0f);
  m_materials[1] = float4(0.6f, 0.0235294f, 0.0235294f, 0.0f);
  m_materials[2] = float4(0.0235294f, 0.6f, 0.0235294f, 0.0f);
  m_materials[3] = float4(0.6f, 0.6f, 0.1f, 0.0f);
  m_materials[4] = float4(0.0847059, 0.144706,0.265882, 0.0f);
  m_materials[5] = float4(0.5f, 0.5f, 0.75f, 0.0f);
  m_materials[6] = float4(0.25f, 0.0f, 0.5f, 0.0f);
  m_materials[7] = m_materials[5];
  m_materials[8] = m_materials[5];
  m_materials[9] = float4(0.0, 0.0f, 0.0f, 20.0f); // emissive material
  m_emissiveMaterialId = 9;
}

int TestClass::LoadScene(const char* meshPath)
{ 
  SimpleMesh m_mesh = cmesh4::LoadMeshFromVSGF(meshPath);
  
  std::cout << "[LoadScene]: create accel struct " << std::endl;
  
  m_materialIds = m_mesh.matIndices;
  InitSceneMaterials(10);

  std::cout << "IndicesNum   = " << m_mesh.indices.size() << std::endl;
  std::cout << "TrianglesNum = " << m_mesh.TrianglesNum() << std::endl;
  std::cout << "MateriaIdNum = " << m_mesh.matIndices.size() << std::endl;

  m_pAccelStruct->ClearGeom();
  auto geomId = m_pAccelStruct->AddGeom_Triangles3f((const float*)m_mesh.vPos4f.data(), m_mesh.vPos4f.size(), m_mesh.indices.data(), m_mesh.indices.size(), BUILD_HIGH, sizeof(float)*4);
  
  m_pAccelStruct->ClearScene();
  float4x4 identitiMatrix;
  auto instId = m_pAccelStruct->AddInstance(geomId, identitiMatrix);
  m_pAccelStruct->CommitScene();

  return 0;

}