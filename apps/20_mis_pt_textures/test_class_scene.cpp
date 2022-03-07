#include "test_class.h"
#include "include/crandom.h"

#include "cmesh.h"
using cmesh::SimpleMesh;

#define LAYOUT_STD140 // !!! PLEASE BE CAREFUL WITH THIS !!!
#include "hydraxml.h"

#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

struct TextureInfo
{
  std::wstring path;   ///< path to file with texture data
  uint32_t     width;  ///< assumed texture width
  uint32_t     height; ///< assumed texture height
  uint32_t     bpp;    ///< assumed texture bytes per pixel, we support 4 (LDR) or 16 (HDR) during loading; Note that HDR texture could be compressed to 8 bytes (half4) on GPU.
};

struct HydraSampler
{
  float4    row0       = float4(1,0,0,0);
  float4    row1       = float4(0,1,0,0);
  float     inputGamma = 2.2f;
  bool      alphaFromRGB = true;
  
  uint32_t  texId = 0;
  Sampler   sampler;

  bool operator==(const HydraSampler& a_rhs) const
  {
    const bool addrAreSame     = (sampler.addressU == a_rhs.sampler.addressU) && (sampler.addressV == a_rhs.sampler.addressV) && (sampler.addressW == a_rhs.sampler.addressW);
    const bool filtersAreSame  = (sampler.filter == a_rhs.sampler.filter);
    const bool hasBorderSam    = (sampler.addressU == Sampler::AddressMode::BORDER || sampler.addressV == Sampler::AddressMode::BORDER || sampler.addressW == Sampler::AddressMode::BORDER);
    const bool sameBorderColor = (length3f(sampler.borderColor - a_rhs.sampler.borderColor) < 1e-5f);
    const bool sameTexId       = (texId == a_rhs.texId);
    return (addrAreSame && filtersAreSame) && (!hasBorderSam || sameBorderColor) && sameTexId;
  }
};

class HydraSamplerHash 
{
public:
  size_t operator()(const HydraSampler& sam) const
  {
    const size_t addressMode1 = size_t(sam.sampler.addressU);
    const size_t addressMode2 = size_t(sam.sampler.addressV) << 4;
    const size_t addressMode3 = size_t(sam.sampler.addressW) << 8;
    const size_t filterMode   = size_t(sam.sampler.filter)   << 12;
    return addressMode1 | addressMode2 | addressMode3 | filterMode | (size_t(sam.texId) << 16);
  }
};

Sampler::AddressMode GetAddrModeFromString(const std::wstring& a_mode)
{
  if(a_mode == L"clamp")
    return Sampler::AddressMode::CLAMP;
  else if(a_mode == L"wrap")
    return Sampler::AddressMode::WRAP;
  else if(a_mode == L"mirror")
    return Sampler::AddressMode::MIRROR;
  else if(a_mode == L"border")
    return Sampler::AddressMode::BORDER;
  else if(a_mode == L"mirror_once")
    return Sampler::AddressMode::MIRROR_ONCE;
  else
    return Sampler::AddressMode::WRAP;
}

HydraSampler ReadSamplerFromColorNode(const pugi::xml_node a_colorNodes)
{
  HydraSampler res;
  auto texNode = a_colorNodes.child(L"texture");
  if(texNode == nullptr)
    return res;
  
  res.texId = texNode.attribute(L"id").as_uint();
  
  if(texNode.attribute(L"addressing_mode_u") != nullptr)
  {
    std::wstring addModeU = texNode.attribute(L"addressing_mode_u").as_string();
    res.sampler.addressU  = GetAddrModeFromString(addModeU);
  } 

  if(texNode.attribute(L"addressing_mode_v") != nullptr)
  {
    std::wstring addModeV = texNode.attribute(L"addressing_mode_v").as_string();
    res.sampler.addressV  = GetAddrModeFromString(addModeV);
  }

  if(texNode.attribute(L"addressing_mode_w") == nullptr)
    res.sampler.addressW  = res.sampler.addressV;
  else
  {
    std::wstring addModeW = texNode.attribute(L"addressing_mode_w").as_string();
    res.sampler.addressW  = GetAddrModeFromString(addModeW);
  }

  res.sampler.filter = Sampler::Filter::LINEAR;

  if(texNode.attribute(L"input_gamma") != nullptr)
    res.inputGamma = texNode.attribute(L"input_gamma").as_float();

  const std::wstring inputAlphaMode = texNode.attribute(L"input_alpha").as_string();
  if(inputAlphaMode == L"alpha")
    res.alphaFromRGB = false;
  
  // read texture matrix
  //
  std::wstringstream inputStream(texNode.attribute(L"matrix").as_string()); // in HydraXML we store matrices by rows
  for(int i=0;i<4;i++)
    inputStream >> res.row0[i];
  for(int i=0;i<4;i++)
    inputStream >> res.row1[i];
  return res;
}

//bool LoadHDRImageFromFile(const wchar_t* a_fileName, int* pW, int* pH, std::vector<float>& a_data);
//bool LoadLDRImageFromFile(const wchar_t* a_fileName, int* pW, int* pH, std::vector<uint32_t>& a_data);
//bool SaveLDRImageToFile  (const wchar_t* a_fileName, int w, int h, uint32_t* data);

std::shared_ptr<ITexture2DCombined> MakeWhiteDummy()
{
  constexpr uint32_t WHITE = 0x00FFFFFF;
  std::shared_ptr< Texture2D<uint32_t> > pTexture1 = std::make_shared< Texture2D<uint32_t> >(1, 1, &WHITE);
  Sampler sampler;
  sampler.filter   = Sampler::Filter::NEAREST; 
  sampler.addressU = Sampler::AddressMode::CLAMP;
  sampler.addressV = Sampler::AddressMode::CLAMP;
  return MakeCombinedTexture2D(pTexture1, sampler);
}

std::shared_ptr<ITexture2DCombined> LoadTextureAndMakeCombined(const TextureInfo& a_texInfo, const Sampler& a_sampler)
{
  std::shared_ptr<ITexture2DCombined> pResult = nullptr;
  int wh[2] = {0,0};
  
  #ifdef WIN32
  std::ifstream fin(a_texInfo.path.c_str(), std::ios::binary);
  #else
  std::string   fnameA(a_texInfo.path.begin(), a_texInfo.path.end());
  std::ifstream fin(fnameA.c_str(), std::ios::binary);
  if(!fin.is_open())
    std::cout << "[LoadTextureAndMakeCombined]: can't open '" << fnameA << "'" << std::endl;
  #endif

  fin.read((char*)wh, sizeof(int)*2);
  if(a_texInfo.bpp == 16)
  {
    std::vector<float> data(wh[0]*wh[1]*4);
    fin.read((char*)data.data(), sizeof(float)*4*data.size());
    fin.close();

    auto pTexture = std::make_shared< Texture2D<float4> >(wh[0], wh[1], (const float4*)data.data());
    pResult = MakeCombinedTexture2D(pTexture, a_sampler);
  }
  else
  {
    std::vector<uint32_t> data(wh[0]*wh[1]);
    fin.read((char*)data.data(), sizeof(uint32_t)*data.size());
    fin.close();

    auto pTexture = std::make_shared< Texture2D<uint32_t> >(wh[0], wh[1], data.data());
    pTexture->setSRGB(true);
    pResult = MakeCombinedTexture2D(pTexture, a_sampler);
  }
 
  return pResult;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int Integrator::LoadScene(const char* scehePath)
{   
  hydra_xml::HydraScene scene;
  scene.LoadState(scehePath);
  
  std::vector<TextureInfo> texturesInfo;
  texturesInfo.resize(0);
  texturesInfo.reserve(100);

  std::string scenePathStr(scehePath);
  #ifdef WIN32
  size_t endPos = scenePathStr.find_last_of("\\");
  if(endPos == std::string::npos)
    endPos = scenePathStr.find_last_of("/");
  #else
  size_t endPos = scenePathStr.find_last_of("/");
  #endif

  std::string sceneFolder = scenePathStr.substr(0, endPos);

  //// (0) load textures info
  //
  for(auto texNode : scene.TextureNodes())
  {
    TextureInfo tex;
    tex.path   = std::wstring(sceneFolder.begin(), sceneFolder.end()) + L"/" + texNode.attribute(L"loc").as_string();
    tex.width  = texNode.attribute(L"width").as_uint();
    tex.height = texNode.attribute(L"height").as_uint();
    if(tex.width != 0 && tex.height != 0)
    {
      size_t byteSize = texNode.attribute(L"bytesize").as_ullong();
      tex.bpp    = byteSize / size_t(tex.width*tex.height);
    }
    texturesInfo.push_back(tex);
  }

  std::unordered_map<HydraSampler, uint32_t, HydraSamplerHash> texCache;
  texCache[HydraSampler()] = 0; // zero white texture
  
  m_textures.resize(0);
  m_textures.reserve(256);
  m_textures.push_back(MakeWhiteDummy());

  //// (1) load materials
  //
  m_materials.resize(0);
  m_materials.reserve(100);
  for(auto materialNode : scene.MaterialNodes())
  {
    GLTFMaterial mat = {};
    mat.brdfType     = BRDF_TYPE_LAMBERT;
    mat.alpha        = 0.0f;
    mat.coatColor    = float4(0,0,0,0); 
    mat.metalColor   = float4(0,0,0,0); 

    // read Hydra or GLTF materials
    //
    float4 color(0.5f, 0.5f, 0.75f, 0.0f);
    if(materialNode.attribute(L"light_id") != nullptr)
    {
      auto node = materialNode.child(L"emission").child(L"color");
      color   = to_float4(hydra_xml::readval3f(node), 0.0f);
      color.w = 0.333334f*(color.x + color.y + color.z);

      //TODO: process emissive texture
      HydraSampler emissiveSampler = ReadSamplerFromColorNode(materialNode.child(L"emission"));
    }

    auto nodeDiff = materialNode.child(L"diffuse").child(L"color");
    if(nodeDiff != nullptr)
    {
      color = to_float4(hydra_xml::readval3f(nodeDiff), 0.0f);
      HydraSampler diffSampler = ReadSamplerFromColorNode(nodeDiff);
      auto p = texCache.find(diffSampler);
      if(p == texCache.end())
      {
        texCache[diffSampler] = uint(m_textures.size());
        const uint32_t texId  = nodeDiff.child(L"texture").attribute(L"id").as_uint();
        m_textures.push_back(LoadTextureAndMakeCombined(texturesInfo[texId], diffSampler.sampler));
        p = texCache.find(diffSampler);
      }
      
      mat.row0 [0]  = diffSampler.row0;
      mat.row1 [0]  = diffSampler.row1;
      mat.texId[0]  = p->second;
    }

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
    else if(length(to_float3(color)) > 1e-5f)
    {
      mat.brdfType   = BRDF_TYPE_LAMBERT;
      mat.baseColor  = color;
      mat.alpha      = 0.0f;
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
    float power = lightInst.lightNode.child(L"intensity").child(L"multiplier").text().as_float();
    if(power == 0.0f)
      power = lightInst.lightNode.child(L"intensity").child(L"multiplier").attribute(L"val").as_float();

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

  m_vNorm4f.resize(0);
  m_vTexc2f.resize(0);

  m_pAccelStruct->ClearGeom();
  for(auto meshPath : scene.MeshFiles())
  {
    std::cout << "[LoadScene]: mesh = " << meshPath.c_str() << std::endl;
    auto currMesh = cmesh::LoadMeshFromVSGF(meshPath.c_str());
    auto geomId   = m_pAccelStruct->AddGeom_Triangles4f(currMesh.vPos4f.data(), currMesh.vPos4f.size(), currMesh.indices.data(), currMesh.indices.size());
    
    m_matIdOffsets.push_back(m_matIdByPrimId.size());
    m_vertOffset.push_back(m_vNorm4f.size());

    m_matIdByPrimId.insert(m_matIdByPrimId.end(), currMesh.matIndices.begin(), currMesh.matIndices.end() );
    m_triIndices.insert(m_triIndices.end(), currMesh.indices.begin(), currMesh.indices.end());

    //for(size_t i=0;i<currMesh.vPos4f.size();i++) // pack texture coords to fourth components
    //{
    //  currMesh.vPos4f [i].w = currMesh.vTexCoord2f[i].x;
    //  currMesh.vNorm4f[i].w = currMesh.vTexCoord2f[i].y;
    //}

    m_vNorm4f.insert(m_vNorm4f.end(), currMesh.vNorm4f.begin(),     currMesh.vNorm4f.end());     // #TODO: store compressed normal and tangent together
    m_vTexc2f.insert(m_vTexc2f.end(), currMesh.vTexCoord2f.begin(), currMesh.vTexCoord2f.end()); // #TODO: store quantized texture coordinates
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