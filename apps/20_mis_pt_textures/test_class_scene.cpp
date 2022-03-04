#include "test_class.h"
#include "include/crandom.h"

#include "cmesh.h"
using cmesh::SimpleMesh;

#define LAYOUT_STD140 // !!! PLEASE BE CAREFUL WITH THIS !!!
#include "hydraxml.h"

#include <string>

struct TextureInfo
{
  std::wstring path;   ///< path to file with texture data
  uint32_t     width;  ///< assumed texture width
  uint32_t     height; ///< assumed texture height
  uint32_t     bpp;    ///< assumed texture bytes per pixel, we support 4 (LDR) or 16 (HDR) during loading; Note that HDR texture could be compressed to 8 bytes (half4) on GPU.
};

int Integrator::LoadScene(const char* scehePath)
{   
  hydra_xml::HydraScene scene;
  scene.LoadState(scehePath);
  
  std::vector<TextureInfo> texturesInfo;
  texturesInfo.resize(0);
  texturesInfo.reserve(100);

  //// (0) load textures info
  //
  for(auto texNode : scene.TextureNodes())
  {
    TextureInfo tex;
    tex.path   = texNode.attribute(L"loc").as_string();
    tex.width  = texNode.attribute(L"width").as_uint();
    tex.height = texNode.attribute(L"height").as_uint();
    if(tex.width != 0 && tex.height != 0)
    {
      size_t byteSize = texNode.attribute(L"bytesize").as_ullong();
      tex.bpp    = byteSize / size_t(tex.width*tex.height);
    }
    texturesInfo.push_back(tex);
  }

  //// (1) load materials
  //
  m_materials.resize(0);
  m_materials.reserve(100);
  for(auto materialNode : scene.MaterialNodes())
  {
    // read Hydra or GLTF materials
    //
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
    
    float3 reflColor = float3(0,0,0);
    float glosiness  = 1.0f;
    auto nodeRefl = materialNode.child(L"reflectivity");
    if(nodeRefl != nullptr)
    {
      reflColor = hydra_xml::readval3f(nodeRefl.child(L"color"));
      glosiness = nodeRefl.child(L"glossiness").text().as_float(); // #TODO: read in different way ... 
    }

    const bool hasFresnel  = (nodeRefl.child(L"fresnel").attribute(L"val").as_int() != 0);
    const float fresnelIOR = nodeRefl.child(L"fresnel_ior").attribute(L"val").as_float();

    GLTFMaterial mat = {};
    mat.brdfType     = BRDF_TYPE_LAMBERT;
    mat.baseColor    = color;
    mat.coatColor    = float4(0,0,0,0); 
    mat.alpha        = 0.0f;
    
    if(length(reflColor) > 1e-5f && length(to_float3(color)) > 1e-5f)
    {
      mat.brdfType   = BRDF_TYPE_GLTF;
      mat.baseColor  = color;
      mat.coatColor  = to_float4(reflColor, 0.0f);
      mat.metalColor = to_float4(reflColor, 0.0f);
      if(hasFresnel)
      {
        mat.alpha = 0.0f;
      }
      else
      {
        mat.alpha     = length(reflColor)/( length(reflColor) + length3f(color) );
        mat.coatColor = float4(0,0,0,0);                                            // disable coating for such blend type
      }
    }
    else if(length(reflColor) > 1e-5f)
    {
      mat.brdfType   = BRDF_TYPE_GGX;
      mat.metalColor = to_float4(reflColor, 0.0f);
      mat.alpha      = 1.0f;
    }
    mat.glosiness  = glosiness;
    if(color[3] > 1e-5f)
    {
      mat.brdfType = BRDF_TYPE_LIGHT_SOURCE;
    }
    m_materials.push_back(mat);
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