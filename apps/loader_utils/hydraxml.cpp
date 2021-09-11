#include "hydraxml.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <locale>
#include <codecvt>

#if defined(__ANDROID__)
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "HydraXML", __VA_ARGS__))
#endif

namespace hydra_xml
{
  std::wstring s2ws(const std::string& str)
  {
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;
    return converterX.from_bytes(str);
  }

  std::string ws2s(const std::wstring& wstr)
  {
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;
    return converterX.to_bytes(wstr);
  }

  void HydraScene::LogError(const std::string &msg)
  {
    std::cout << "HydraScene ERROR: " << msg << std::endl;
  }

#if defined(__ANDROID__)
  int HydraScene::LoadState(AAssetManager* mgr, const std::string &path)
  {
    AAsset* asset = AAssetManager_open(mgr, path.c_str(), AASSET_MODE_STREAMING);
    if (!asset)
    {
      LOGE("Could not load scene from \"%s\"!", path.c_str());
    }
    assert(asset);

    size_t asset_size = AAsset_getLength(asset);

    assert(asset_size > 0);

    void* data = malloc(asset_size);
    AAsset_read(asset, data, asset_size);
    AAsset_close(asset);

    pugi::xml_document xmlDoc;

    auto loaded = xmlDoc.load_buffer(data, asset_size);

    if(!loaded)
    {
      std::string str(loaded.description());
      LOGE("pugixml error loading scene: %s", str.c_str());

      return -1;
    }

    auto pos = path.find_last_of(L'/');
    m_libraryRootDir = path.substr(0, pos);

    auto texturesLib  = xmlDoc.child(L"textures_lib");
    auto materialsLib = xmlDoc.child(L"materials_lib");
    auto geometryLib  = xmlDoc.child(L"geometry_lib");
    auto lightsLib    = xmlDoc.child(L"lights_lib");

    auto cameraLib    = xmlDoc.child(L"cam_lib");
    auto settingsNode = xmlDoc.child(L"render_lib");
    auto sceneNode    = xmlDoc.child(L"scenes");

    if (texturesLib == nullptr || materialsLib == nullptr || lightsLib == nullptr || cameraLib == nullptr ||
        geometryLib == nullptr || settingsNode == nullptr || sceneNode == nullptr)
    {
      std::string errMsg = "Loaded state (" +  path + ") doesn't have one of (textures_lib, materials_lib, lights_lib, cam_lib, geometry_lib, render_lib, scenes";
      LogError(errMsg);
      return -1;
    }

    parseInstancedMeshes(sceneNode, geometryLib);

    return 0;
  }
#else
  int HydraScene::LoadState(const std::string &path)
  {
    auto loaded = m_xmlDoc.load_file(path.c_str());

    if(!loaded)
    {
      std::string  str(loaded.description());
      std::wstring errorMsg(str.begin(), str.end());

      LogError("Error loading scene from: " + path);
      LogError(ws2s(errorMsg));

      return -1;
    }

    auto pos = path.find_last_of(L'/');
    m_libraryRootDir = path.substr(0, pos);

    m_texturesLib  = m_xmlDoc.child(L"textures_lib");
    m_materialsLib = m_xmlDoc.child(L"materials_lib");
    m_geometryLib  = m_xmlDoc.child(L"geometry_lib");
    m_lightsLib    = m_xmlDoc.child(L"lights_lib");

    m_cameraLib    = m_xmlDoc.child(L"cam_lib");
    m_settingsNode = m_xmlDoc.child(L"render_lib");
    m_sceneNode    = m_xmlDoc.child(L"scenes");

    if (m_texturesLib == nullptr || m_materialsLib == nullptr || m_lightsLib == nullptr || m_cameraLib == nullptr || m_geometryLib == nullptr || m_settingsNode == nullptr || m_sceneNode == nullptr)
    {
      std::string errMsg = "Loaded state (" +  path + ") doesn't have one of (textures_lib, materials_lib, lights_lib, cam_lib, geometry_lib, render_lib, scenes";
      LogError(errMsg);
      return -1;
    }

    parseInstancedMeshes(m_sceneNode, m_geometryLib);

    return 0;
  }
#endif

  void HydraScene::parseInstancedMeshes(pugi::xml_node a_scenelib, pugi::xml_node a_geomlib)
  {
    auto scene = a_scenelib.first_child();
    for (pugi::xml_node inst = scene.first_child(); inst != nullptr; inst = inst.next_sibling())
    {
      if (std::wstring(inst.name()) == L"instance_light")
        break;

      auto mesh_id = inst.attribute(L"mesh_id").as_string();
      auto matrix = std::wstring(inst.attribute(L"matrix").as_string());

      auto meshNode = a_geomlib.find_child_by_attribute(L"id", mesh_id);

      if(meshNode != nullptr)
      {
        auto meshLoc = ws2s(std::wstring(meshNode.attribute(L"loc").as_string()));
        meshLoc = m_libraryRootDir + "/" + meshLoc;

#if not defined(__ANDROID__)
        std::ifstream checkMesh(meshLoc);
        if(!checkMesh.good())
        {
          LogError("Mesh not found at: " + meshLoc + ". Loader will skip it.");
          continue;
        }
        else
        {
          checkMesh.close();
        }
#endif

        if(unique_meshes.find(meshLoc) == unique_meshes.end())
        {
          unique_meshes.emplace(meshLoc);
          //m_meshloc.push_back(meshLoc);
        }


        if(m_instancesPerMeshLoc.find(meshLoc) != m_instancesPerMeshLoc.end())
        {
          m_instancesPerMeshLoc[meshLoc].push_back(float4x4FromString(matrix));
        }
        else
        {
          std::vector<LiteMath::float4x4> tmp = { float4x4FromString(matrix) };
          m_instancesPerMeshLoc[meshLoc] = tmp;
        }
      }
    }


  }

  LiteMath::float4x4 float4x4FromString(const std::wstring &matrix_str)
  {
    LiteMath::float4x4 result;
    std::wstringstream inputStream(matrix_str);
    
    float data[16];
    for(int i=0;i<16;i++)
      inputStream >> data[i];
    
    result.set_row(0, LiteMath::float4(data[0],data[1], data[2], data[3]));
    result.set_row(1, LiteMath::float4(data[4],data[5], data[6], data[7]));
    result.set_row(2, LiteMath::float4(data[8],data[9], data[10], data[11]));
    result.set_row(3, LiteMath::float4(data[12],data[13], data[14], data[15])); 

    return result;
  }

  LiteMath::float3 read3f(pugi::xml_attribute a_attr)
  {
    LiteMath::float3 res(0, 0, 0);
    const wchar_t* camPosStr = a_attr.as_string();
    if (camPosStr != nullptr)
    {
      std::wstringstream inputStream(camPosStr);
      inputStream >> res.x >> res.y >> res.z;
    }
    return res;
  }

  LiteMath::float3 read3f(pugi::xml_node a_node)
  {
    LiteMath::float3 res(0,0,0);
    const wchar_t* camPosStr = a_node.text().as_string();
    if (camPosStr != nullptr)
    {
      std::wstringstream inputStream(camPosStr);
      inputStream >> res.x >> res.y >> res.z;
    }
    return res;
  }

  LiteMath::float3 readval3f(pugi::xml_node a_node)
  {
    float3 color;
    if(a_node.attribute(L"val") != nullptr)
      color = hydra_xml::read3f(a_node.attribute(L"val"));
    else
      color = hydra_xml::read3f(a_node);
    return color;
  }

  std::vector<LightInstance> HydraScene::InstancesLights(uint32_t a_sceneId) 
  {
    auto sceneNode = m_sceneNode.child(L"scene");
    if(a_sceneId != 0)
    {
      std::wstringstream temp;
      temp << a_sceneId;
      std::wstring tempStr = temp.str();
      sceneNode = m_sceneNode.find_child_by_attribute(L"id", tempStr.c_str());
    }

    std::vector<pugi::xml_node> lights; 
    lights.reserve(256);
    for(auto lightNode : m_lightsLib.children())
      lights.push_back(lightNode);

    std::vector<LightInstance> result;
    result.reserve(256);

    LightInstance inst;
    for(auto instNode = sceneNode.child(L"instance_light"); instNode != nullptr; instNode = instNode.next_sibling())
    {
      std::wstring nameStr = instNode.name();
      if(nameStr != L"instance_light")
        continue;
      inst.instNode  = instNode;
      inst.instId    = instNode.attribute(L"id").as_uint();
      inst.lightId   = instNode.attribute(L"light_id").as_uint(); 
      inst.lightNode = lights[inst.lightId];
    }
    return result;
  }

}

