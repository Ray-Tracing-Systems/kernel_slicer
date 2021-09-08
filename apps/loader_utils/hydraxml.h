#ifndef HYDRAXML_H
#define HYDRAXML_H

#include "pugixml.hpp"
#include "LiteMath.h"
using namespace LiteMath;

#include <vector>
#include <set>
#include <unordered_map>
//#include <iostream>

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#include <android/log.h>
#endif

namespace hydra_xml
{
  std::wstring s2ws(const std::string& str);
  std::string  ws2s(const std::wstring& wstr);
  LiteMath::float4x4 float4x4FromString(const std::wstring &matrix_str);
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct Instance
  {
    uint32_t           geomId = uint32_t(-1); ///< geom id
    uint32_t           rmapId = uint32_t(-1); ///< remap list id, todo: add function to get real remap list by id
    LiteMath::float4x4 matrix; ///< trannform matrix
  };

  struct LightInstance
  {
    uint32_t           instId  = uint32_t(-1);
    uint32_t           lightId = uint32_t(-1);
    pugi::xml_node     instNode;
    pugi::xml_node     lightNode;
    LiteMath::float4x4 matrix;
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // //
  // //
	// class LocIterator // (!!!) does not works, mem issues
	// {
	// 	 friend class pugi::xml_node;
  //   friend class pugi::xml_node_iterator;
  // 
	// public:
  // 
	// 	// Default constructor
	// 	LocIterator() : m_libraryRootDir("") {}
	// 	LocIterator(const pugi::xml_node_iterator& a_iter, const std::string& a_str) : m_iter(a_iter), m_libraryRootDir(a_str) {}
  // 
	// 	// Iterator operators
	// 	bool operator==(const LocIterator& rhs) const { return m_iter == rhs.m_iter;}
	// 	bool operator!=(const LocIterator& rhs) const { return (m_iter != rhs.m_iter); }
  // 
  //   std::string operator*() const // (!!!) does not works, mem issues for 'm_iter->attribute'
  //   { 
  //     auto attr    = m_iter->attribute(L"loc");
  //     auto meshLoc = ws2s(std::wstring(attr.as_string()));
  //     return m_libraryRootDir + "/" + meshLoc;
  //   }
  // 
	// 	const LocIterator& operator++() { ++m_iter; return *this; }
	// 	LocIterator operator++(int)     { m_iter++; return *this; }
  // 
	// 	const LocIterator& operator--() { --m_iter; return *this; }
	// 	LocIterator operator--(int)     { m_iter--; return *this; }
  // 
  // private:
  //   pugi::xml_node_iterator m_iter;
  //   std::string m_libraryRootDir;
	// };

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct HydraScene
  {
  public:
    HydraScene() = default;
    ~HydraScene() = default;  
    
    #if defined(__ANDROID__)
    int LoadState(AAssetManager* mgr, const std::string &path);
    #else
    int LoadState(const std::string &path);
    #endif  

    //// use this functions with C++11 range for 
    //
    pugi::xml_object_range<pugi::xml_node_iterator> TextureNodes()  { return m_texturesLib.children();  } 
    pugi::xml_object_range<pugi::xml_node_iterator> MaterialNodes() { return m_materialsLib.children(); }
    pugi::xml_object_range<pugi::xml_node_iterator> GeomNodes()     { return m_geometryLib.children();  }
    pugi::xml_object_range<pugi::xml_node_iterator> LightNodes()    { return m_lightsLib.children();    }
    pugi::xml_object_range<pugi::xml_node_iterator> CameraNodes()   { return m_cameraLib.children();    }
    
    //// please also use this functions with C++11 range for 
    //
    std::vector<std::string>   MeshFiles();
    std::vector<std::string>   TextureFiles();
    std::vector<Instance>      InstancesGeom(uint32_t a_sceneId = 0);
    std::vector<LightInstance> InstancesLights(uint32_t a_sceneId = 0);

    std::vector<LiteMath::float4x4> GetAllInstancesOfMeshLoc(const std::string& a_loc) const 
    { 
      auto pFound = m_instancesPerMeshLoc.find(a_loc);
      if(pFound == m_instancesPerMeshLoc.end())
        return std::vector<LiteMath::float4x4>();
      else
        return pFound->second; 
    }
    
  private:
    void parseInstancedMeshes(pugi::xml_node a_scenelib, pugi::xml_node a_geomlib);
    void LogError(const std::string &msg);  
    
    //std::vector<pugi::xml_node> ListChildren(pugi::xml_node a_nodes, size_t a_reserveSize = 256);
    std::set<std::string> unique_meshes;
    std::string m_libraryRootDir;
    pugi::xml_node m_texturesLib ; 
    pugi::xml_node m_materialsLib; 
    pugi::xml_node m_geometryLib ; 
    pugi::xml_node m_lightsLib   ;
    pugi::xml_node m_cameraLib   ; 
    pugi::xml_node m_settingsNode; 
    pugi::xml_node m_sceneNode   ; 

    std::unordered_map<std::string, std::vector<LiteMath::float4x4> > m_instancesPerMeshLoc;
  };

  LiteMath::float3 read3f(pugi::xml_attribute a_attr);
}

#endif //HYDRAXML_H
