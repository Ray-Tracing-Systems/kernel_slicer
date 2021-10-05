#include "test_class.h"
#include "include/crandom.h"

#include "cmesh.h"
using cmesh::SimpleMesh;

#define LAYOUT_STD140 // !!! PLEASE BE CAREFUL WITH THIS !!!
#include "hydraxml.h"

#include <string>

int TestClass::LoadScene(const char* scehePath)
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

  // load first camera and update matrix
  //
  for(auto cam : scene.Cameras())
  {
    float aspect   = 1.0f;
    auto proj      = perspectiveMatrix(cam.fov, aspect, cam.nearPlane, cam.farPlane);
    auto worldView = lookAt(float3(cam.pos), float3(cam.lookAt), float3(cam.up));
      
    m_projInv      = inverse4x4(proj);
    m_worldViewInv = inverse4x4(worldView);
    break; // take first cam
  }

  // load first light
  //
  for(auto lightInst : scene.InstancesLights())
  {
    const std::wstring shape = lightInst.lightNode.attribute(L"shape").as_string();
    const float sizeX = lightInst.lightNode.child(L"size").attribute(L"half_width").as_float();
    const float sizeZ = lightInst.lightNode.child(L"size").attribute(L"half_length").as_float();
    const float power = lightInst.lightNode.child(L"intensity").child(L"multiplier").text().as_float();
    if(shape != L"rect")
    {
      std::cout << "WARNING: not supported shape of the light object!" << std::endl;
      continue;
    }
    
    m_light.pos       = lightInst.matrix*float4(0,0,0,1);
    m_light.size      = float2(sizeX, sizeZ);
    m_light.intensity = float4(power,power,power,0);
    break;
  }

  //// (2) load meshes
  //
  m_matIdOffsets.reserve(1024);
  m_vertOffset.reserve(1024);
  m_matIdByPrimId.reserve(128000);
  m_triIndices.reserve(128000*3);

  m_pAccelStruct->ClearGeom();
  for(auto meshPath : scene.MeshFiles())
  {
    std::cout << "[LoadScene]: mesh = " << meshPath.c_str() << std::endl;
    auto currMesh = cmesh::LoadMeshFromVSGF(meshPath.c_str());
    auto geomId   = m_pAccelStruct->AddGeom_Triangles4f(currMesh.vPos4f.data(), currMesh.vPos4f.size(), currMesh.indices.data(), currMesh.indices.size());
    
    m_matIdOffsets.push_back(m_matIdByPrimId.size());
    m_vertOffset.push_back(m_vPos4f.size());

    m_matIdByPrimId.insert(m_matIdByPrimId.end(), currMesh.matIndices.begin(), currMesh.matIndices.end() );
    m_triIndices.insert(m_triIndices.end(), currMesh.indices.begin(), currMesh.indices.end());

    m_vPos4f.insert(m_vPos4f.end(),   currMesh.vPos4f.begin(),  currMesh.vPos4f.end());
    m_vNorm4f.insert(m_vNorm4f.end(), currMesh.vNorm4f.begin(), currMesh.vNorm4f.end());
  }
  
  //// (3) make instances of created meshes
  //
  m_normMatrices.clear();

  m_pAccelStruct->ClearScene();
  for(auto inst : scene.InstancesGeom())
  {
    m_pAccelStruct->AddInstance(inst.geomId, inst.matrix);
    m_normMatrices.push_back(transpose(inverse4x4(inst.matrix)));
  }
  m_pAccelStruct->CommitScene();

  return 0;
}