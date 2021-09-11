#include "test_class.h"
#include "include/crandom.h"

#include "cmesh.h"
using cmesh::SimpleMesh;

#define LAYOUT_STD140 // !!! PLEASE BE CAREFUL WITH THIS !!!
#include "hydraxml.h"

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
    if(materialNode.attribute(L"light_id") != nullptr)
    {
      auto node = materialNode.child(L"emission").child(L"color");
      color   = to_float4(hydra_xml::readval3f(node), 0.0f);
      color.w = 0.333334f*(color.x + color.y + color.z);
    }

    auto node = materialNode.child(L"diffuse").child(L"color");
    if(node != nullptr)
      color = to_float4(hydra_xml::readval3f(node), 0.0f);

    m_materials.push_back(color);
  }

  auto meshes    = scene.MeshFiles();     // need to get them before use because embree crap break memory
  auto instances = scene.InstancesGeom(); // --//-- (same)
  auto cams      = scene.Cameras();

  // load first camera and update matrix
  //
  assert(cams.size() > 0);
  auto cam       = cams[0];
  float aspect   = 1.0f;
  auto proj      = perspectiveMatrix(cam.fov, aspect, cam.nearPlane, cam.farPlane);
  auto worldView = lookAt(float3(cam.pos), float3(cam.lookAt), float3(cam.up));
    
  m_projInv      = inverse4x4(proj);
  m_worldViewInv = inverse4x4(worldView);

  //// (2) load meshes
  //
  m_matIdOffsets.reserve(1024);
  m_matIdByPrimId.reserve(128000);

  m_pAccelStruct->ClearGeom();
  for(const auto& meshPath : meshes)
  {
    std::cout << meshPath.c_str() << std::endl;
    auto currMesh = cmesh::LoadMeshFromVSGF(meshPath.c_str());
    auto geomId   = m_pAccelStruct->AddGeom_Triangles4f(currMesh.vPos4f.data(), currMesh.vPos4f.size(), currMesh.indices.data(), currMesh.indices.size());
    
    m_matIdOffsets.push_back(m_matIdByPrimId.size());
    m_matIdByPrimId.insert(m_matIdByPrimId.end(), currMesh.matIndices.begin(), currMesh.matIndices.end() );
  }
  
  //// (3) make instances of created meshes
  //
  m_pAccelStruct->ClearScene();
  for(const auto& inst : instances)
    m_pAccelStruct->AddInstance(inst.geomId, inst.matrix);
  m_pAccelStruct->CommitScene();

  return 0;
}