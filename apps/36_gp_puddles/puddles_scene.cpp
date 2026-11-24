#include "puddles.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>

#include "hydraxml.h"
#include "cmesh4.h"

static float4 MaterialColor(pugi::xml_node a_mat) // diffuse color, or reflection color for pure mirrors and glass
{
  if(a_mat.child(L"diffuse").child(L"color"))
  {
    const float3 c = hydra_xml::readval3f(a_mat.child(L"diffuse").child(L"color"));
    return float4(c.x, c.y, c.z, 0.0f);
  }
  if(a_mat.child(L"reflectivity").child(L"color"))
  {
    const float3 c = hydra_xml::readval3f(a_mat.child(L"reflectivity").child(L"color"))*0.7f;
    return float4(c.x, c.y, c.z, 0.0f);
  }
  return float4(0.5f, 0.5f, 0.5f, 0.0f);
}

bool PuddleRender::LoadScene(const char* a_scenePath)
{
  hydra_xml::HydraScene scene;
  if(scene.LoadState(a_scenePath) != 0)
  {
    std::cout << "[PuddleRender::LoadScene]: can't load scene '" << a_scenePath << "'" << std::endl;
    return false;
  }

  std::unordered_map<uint32_t, float4> materials;
  for(auto matNode : scene.MaterialNodes())
    materials[matNode.attribute(L"id").as_uint()] = MaterialColor(matNode);

  std::vector< std::vector<int32_t> > remapLists;
  for(auto remap : scene.RemapLists())
    remapLists.push_back(remap);

  // meshes: common vertex and index buffers, indices stay local to each mesh
  //
  m_pAccelStruct->ClearGeom();
  std::vector<std::string> meshNames;
  std::vector<uint32_t>    meshMaterial;
  for(auto geomNode : scene.GeomNodes())
    meshNames.push_back(hydra_xml::ws2s(geomNode.attribute(L"name").as_string()));

  m_positions.clear();
  m_normals.clear();
  m_indices.clear();
  m_meshNum = 0;
  for(auto meshPath : scene.MeshFiles())
  {
    if(m_meshNum >= MAX_MESHES)
    {
      std::cout << "[PuddleRender::LoadScene]: too many meshes, max is " << MAX_MESHES << std::endl;
      return false;
    }
    const cmesh4::SimpleMesh mesh = cmesh4::LoadMeshFromVSGF(meshPath.c_str());
    if(mesh.VerticesNum() == 0)
    {
      std::cout << "[PuddleRender::LoadScene]: can't load mesh '" << meshPath << "'" << std::endl;
      return false;
    }
    m_meshFirstIndex[m_meshNum] = uint32_t(m_indices.size());
    m_meshIndexNum  [m_meshNum] = uint32_t(mesh.IndicesNum());
    m_meshVertOffset[m_meshNum] = uint32_t(m_positions.size());
    m_positions.insert(m_positions.end(), mesh.vPos4f.begin(),  mesh.vPos4f.end());
    m_normals.insert  (m_normals.end(),   mesh.vNorm4f.begin(), mesh.vNorm4f.end());
    m_indices.insert  (m_indices.end(),   mesh.indices.begin(), mesh.indices.end());
    meshMaterial.push_back(mesh.matIndices.empty() ? 0u : mesh.matIndices[0]); // one material per mesh is enough here

    const uint32_t geomId = m_pAccelStruct->AddGeom_Triangles3f((const float*)mesh.vPos4f.data(), mesh.vPos4f.size(), mesh.indices.data(), mesh.indices.size(),
                                                                BUILD_HIGH, sizeof(float)*4);
    if(geomId != m_meshNum)
      std::cout << "[PuddleRender::LoadScene]: unexpected geom id " << geomId << std::endl;
    m_meshNum++;
  }

  // instances, sorted by meshes: instances of one mesh are drawn by one instanced draw call
  //
  std::vector<hydra_xml::Instance> instances;
  for(auto inst : scene.InstancesGeom())
    if(inst.geomId < m_meshNum)
      instances.push_back(inst);
  std::stable_sort(instances.begin(), instances.end(), [](const auto& a, const auto& b) { return a.geomId < b.geomId; });

  for(uint32_t mesh = 0; mesh < m_meshNum; mesh++)
    m_meshInstNum[mesh] = 0;

  m_instMatrix.resize(instances.size());
  m_instNormalMatrix.resize(instances.size());
  m_instColor.resize(instances.size());
  m_instMesh.resize(instances.size());
  m_pAccelStruct->ClearScene();
  for(size_t i = 0; i < instances.size(); i++)
  {
    const auto&    inst = instances[i];
    const uint32_t mesh = inst.geomId;
    if(m_meshInstNum[mesh] == 0)
      m_meshFirstInst[mesh] = uint32_t(i);
    m_meshInstNum[mesh]++;

    uint32_t matId = meshMaterial[mesh];
    if(inst.rmapId < remapLists.size())
    {
      const auto& remap = remapLists[inst.rmapId];
      for(size_t j = 0; j + 1 < remap.size(); j += 2)
        if(remap[j] == int32_t(matId))
          matId = uint32_t(remap[j + 1]);
    }

    const bool isGround = (meshNames[mesh].find("plane") != std::string::npos);
    float4 color        = materials.count(matId) ? materials[matId] : float4(0.5f, 0.5f, 0.5f, 0.0f);
    color.w             = isGround ? 1.0f : 0.0f;

    m_instMatrix      [i] = inst.matrix;
    m_instNormalMatrix[i] = LiteMath::transpose(LiteMath::inverse4x4(inst.matrix));
    m_instColor       [i] = color;
    m_instMesh        [i] = mesh;
    m_pAccelStruct->AddInstance(mesh, inst.matrix);
  }
  m_pAccelStruct->CommitScene();

  for(uint32_t mesh = 0; mesh < m_meshNum; mesh++)
  {
    if(m_meshInstNum[mesh] == 0)
      m_meshFirstInst[mesh] = 0;
    std::cout << "[PuddleRender::LoadScene]: mesh " << mesh << " '" << meshNames[mesh].c_str() << "': " << m_meshIndexNum[mesh]/3 << " triangles, "
              << m_meshInstNum[mesh] << " instances" << std::endl;
  }

  // camera: the first one in the file
  //
  for(auto cam : scene.Cameras())
  {
    m_camPos   = float4(cam.pos[0], cam.pos[1], cam.pos[2], 1.0f);
    m_camView  = LiteMath::lookAt(float3(cam.pos[0], cam.pos[1], cam.pos[2]), float3(cam.lookAt[0], cam.lookAt[1], cam.lookAt[2]),
                                  float3(cam.up[0], cam.up[1], cam.up[2]));
    m_camFov   = cam.fov;
    m_camNear  = cam.nearPlane;
    m_camFar   = cam.farPlane;
    break;
  }

  const LiteMath::Box4f box = scene.GetSceneBBox();
  const float size = length(float3(box.boxMax.x - box.boxMin.x, box.boxMax.y - box.boxMin.y, box.boxMax.z - box.boxMin.z));
  if(std::isfinite(size) && size > 0.0f)
    m_sceneSize = size;
  m_sunDir    = normalize(m_sunDir);
  return true;
}
