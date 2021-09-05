#include "test_class.h"
#include "include/crandom.h"

#include "cmesh.h"
using cmesh::SimpleMesh;

#include "hydraxml.h"

//void TestClass::InitSceneMaterials(int a_numSpheres, int a_seed)
//{ 
//  ///// create simple materials
//  //
//  m_materials.resize(10);
//  m_materials[0] = float4(0.5f, 0.5f, 0.5f, 0.0f);
//  m_materials[1] = float4(0.6f, 0.0235294f, 0.0235294f, 0.0f);
//  m_materials[2] = float4(0.0235294f, 0.6f, 0.0235294f, 0.0f);
//  m_materials[3] = float4(0.6f, 0.6f, 0.1f, 0.0f);
//  m_materials[4] = float4(0.0847059, 0.144706,0.265882, 0.0f);
//  m_materials[5] = float4(0.5f, 0.5f, 0.75f, 0.0f);
//  m_materials[6] = float4(0.25f, 0.0f, 0.5f, 0.0f);
//  m_materials[7] = m_materials[5];
//  m_materials[8] = m_materials[5];
//  m_materials[9] = float4(0.0, 0.0f, 0.0f, 20.0f); // emissive material
//  m_emissiveMaterialId = 9;
//}

int TestClass::LoadScene(const char* scehePath, const char* meshPath, bool a_needReorder)
{   
  hydra_xml::HydraScene scene;
  
  scene.LoadState(scehePath);
  
  //// (1) load materials
  //
  m_materials.resize(0);
  m_materials.reserve(100);
  for(auto materialNode : scene.MaterialNodes())
  {
    float4 color(0.5f, 0.5f, 0.75f, 0.0f);
    auto node = materialNode.child(L"diffuse").child(L"color");
    if(node != nullptr)
      color = to_float4(hydra_xml::read3f(node.attribute(L"val")), 0.0f);
    
    m_materials.push_back(color);
  }
  
  //// (2) load meshes
  //
  m_pAccelStruct->ClearGeom();
  SimpleMesh m_mesh;
  for(const auto& meshPath : scene.MeshFiles())
  {
    std::cout << meshPath.c_str() << std::endl;
    m_mesh      = cmesh::LoadMeshFromVSGF(meshPath.c_str());
    auto geomId = m_pAccelStruct->AddGeom_Triangles4f(m_mesh.vPos4f.data(), m_mesh.vPos4f.size(), m_mesh.indices.data(), m_mesh.indices.size());
  }

  //// (2.1) we actually forgot to build table to get material id by (mesh_id,prim_id) pair
  //
  
  //// (3) make instances of created meshes
  //
  m_pAccelStruct->BeginScene();
  for(const auto& inst : scene.InstancesGeom())
    m_pAccelStruct->AddInstance(inst.geomId, inst.matrix);
  m_pAccelStruct->EndScene();

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  
  /*
  SimpleMesh m_mesh = cmesh::LoadMeshFromVSGF(meshPath);

  m_vPos4f  = m_mesh.vPos4f;
  m_vNorm4f = m_mesh.vNorm4f;
  
  std::cout << "[LoadScene]: create accel struct " << std::endl;
  
  m_materialIds      = m_mesh.matIndices;
  m_indicesReordered = m_mesh.indices;

  InitSceneMaterials(10);

  std::cout << "IndicesNum   = " << m_mesh.indices.size() << std::endl;
  std::cout << "TrianglesNum = " << m_mesh.TrianglesNum() << std::endl;
  std::cout << "MateriaIdNum = " << m_mesh.matIndices.size() << std::endl;

  m_pAccelStruct->ClearGeom();
  auto geomId = m_pAccelStruct->AddGeom_Triangles4f(m_vPos4f.data(), m_vPos4f.size(), m_indicesReordered.data(), m_indicesReordered.size());
  
  m_pAccelStruct->BeginScene();
  float4x4 identitiMatrix;
  auto instId = m_pAccelStruct->AddInstance(geomId, (const float*)&identitiMatrix);
  m_pAccelStruct->EndScene();
  */

  return 0;
}