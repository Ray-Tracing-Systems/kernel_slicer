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
  LiteMath::float3   read3f(pugi::xml_attribute a_attr);
  LiteMath::float3   read3f(pugi::xml_node a_node);
  LiteMath::float3   readval3f(pugi::xml_node a_node);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct Instance
  {
    uint32_t           geomId = uint32_t(-1); ///< geom id
    uint32_t           rmapId = uint32_t(-1); ///< remap list id, todo: add function to get real remap list by id
    LiteMath::float4x4 matrix;                ///< transform matrix
  };

  struct LightInstance
  {
    uint32_t           instId  = uint32_t(-1);
    uint32_t           lightId = uint32_t(-1);
    pugi::xml_node     instNode;
    pugi::xml_node     lightNode;
    LiteMath::float4x4 matrix;
  };

  struct Camera
  {
    float pos[3];
    float lookAt[3];
    float up[3];
    float fov;
    float nearPlane;
    float farPlane;
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class LocIterator //
	{
	  friend class pugi::xml_node;
    friend class pugi::xml_node_iterator;
  
	public:
  
		// Default constructor
		LocIterator() : m_libraryRootDir("") {}
		LocIterator(const pugi::xml_node_iterator& a_iter, const std::string& a_str) : m_iter(a_iter), m_libraryRootDir(a_str) {}
  
		// Iterator operators
		bool operator==(const LocIterator& rhs) const { return m_iter == rhs.m_iter;}
		bool operator!=(const LocIterator& rhs) const { return (m_iter != rhs.m_iter); }
  
    std::string operator*() const 
    { 
      auto attr    = m_iter->attribute(L"loc");
      auto meshLoc = ws2s(std::wstring(attr.as_string()));
      return m_libraryRootDir + "/" + meshLoc;
    }
  
		const LocIterator& operator++() { ++m_iter; return *this; }
		LocIterator operator++(int)     { m_iter++; return *this; }
  
		const LocIterator& operator--() { --m_iter; return *this; }
		LocIterator operator--(int)     { m_iter--; return *this; }
  
  private:
    pugi::xml_node_iterator m_iter;
    std::string m_libraryRootDir;
	};

  class InstIterator //
	{
	  friend class pugi::xml_node;
    friend class pugi::xml_node_iterator;
  
	public:
  
		// Default constructor
		InstIterator() {}
		InstIterator(const pugi::xml_node_iterator& a_iter, const pugi::xml_node_iterator& a_end) : m_iter(a_iter), m_end(a_end) {}
  
		// Iterator operators
		bool operator==(const InstIterator& rhs) const { return m_iter == rhs.m_iter;}
		bool operator!=(const InstIterator& rhs) const { return (m_iter != rhs.m_iter); }
  
    Instance operator*() const 
    { 
      Instance inst;
      inst.geomId = m_iter->attribute(L"mesh_id").as_uint();
      inst.rmapId = m_iter->attribute(L"rmap_id").as_uint();
      inst.matrix = float4x4FromString(m_iter->attribute(L"matrix").as_string());
      return inst;
    }
  
		const InstIterator& operator++() { do ++m_iter; while(m_iter != m_end && std::wstring(m_iter->name()) != L"instance"); return *this; }
		InstIterator operator++(int)     { do m_iter++; while(m_iter != m_end && std::wstring(m_iter->name()) != L"instance"); return *this; }
  
		const InstIterator& operator--() { do --m_iter; while(m_iter != m_end && std::wstring(m_iter->name()) != L"instance"); return *this; }
		InstIterator operator--(int)     { do m_iter--; while(m_iter != m_end && std::wstring(m_iter->name()) != L"instance"); return *this; }
  
  private:
    pugi::xml_node_iterator m_iter;
    pugi::xml_node_iterator m_end;
	};

  class CamIterator //
	{
	  friend class pugi::xml_node;
    friend class pugi::xml_node_iterator;
  
	public:
  
		// Default constructor
		CamIterator() {}
		CamIterator(const pugi::xml_node_iterator& a_iter) : m_iter(a_iter) {}
  
		// Iterator operators
		bool operator==(const CamIterator& rhs) const { return m_iter == rhs.m_iter;}
		bool operator!=(const CamIterator& rhs) const { return (m_iter != rhs.m_iter); }
  
    Camera operator*() const 
    { 
      Camera cam;
      cam.fov       = m_iter->child(L"fov").text().as_float(); 
      cam.nearPlane = m_iter->child(L"nearClipPlane").text().as_float();
      cam.farPlane  = m_iter->child(L"farClipPlane").text().as_float();  
      
      LiteMath::float3 pos    = hydra_xml::read3f(m_iter->child(L"position"));
      LiteMath::float3 lookAt = hydra_xml::read3f(m_iter->child(L"look_at"));
      LiteMath::float3 up     = hydra_xml::read3f(m_iter->child(L"up"));
      for(int i=0;i<3;i++)
      {
        cam.pos   [i] = pos[i];
        cam.lookAt[i] = lookAt[i];
        cam.up    [i] = up[i];
      }
      return cam;
    }
  
		const CamIterator& operator++() { ++m_iter; return *this; }
		CamIterator operator++(int)     { m_iter++; return *this; }
  
		const CamIterator& operator--() { --m_iter; return *this; }
		CamIterator operator--(int)     { m_iter--; return *this; }
  
  private:
    pugi::xml_node_iterator m_iter;
	};

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
    pugi::xml_object_range<LocIterator> MeshFiles()    { return pugi::xml_object_range(LocIterator(m_geometryLib.begin(), m_libraryRootDir), 
                                                                                       LocIterator(m_geometryLib.end(), m_libraryRootDir)
                                                                                       ); }

    pugi::xml_object_range<LocIterator> TextureFiles() { return pugi::xml_object_range(LocIterator(m_texturesLib.begin(), m_libraryRootDir), 
                                                                                       LocIterator(m_texturesLib.end(), m_libraryRootDir)
                                                                                       ); }

    pugi::xml_object_range<InstIterator> InstancesGeom() { return pugi::xml_object_range(InstIterator(m_sceneNode.child(L"scene").child(L"instance"), m_sceneNode.child(L"scene").end()), 
                                                                                         InstIterator(m_sceneNode.child(L"scene").end(), m_sceneNode.child(L"scene").end())
                                                                                         ); }
    
    std::vector<LightInstance> InstancesLights(uint32_t a_sceneId = 0);

    pugi::xml_object_range<CamIterator> Cameras() { return pugi::xml_object_range(CamIterator(m_cameraLib.begin()), 
                                                                                  CamIterator(m_cameraLib.end())); }

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
    
    std::set<std::string> unique_meshes;
    std::string m_libraryRootDir;
    pugi::xml_node m_texturesLib ; 
    pugi::xml_node m_materialsLib; 
    pugi::xml_node m_geometryLib ; 
    pugi::xml_node m_lightsLib   ;
    pugi::xml_node m_cameraLib   ; 
    pugi::xml_node m_settingsNode; 
    pugi::xml_node m_sceneNode   ; 
    pugi::xml_document m_xmlDoc;

    std::unordered_map<std::string, std::vector<LiteMath::float4x4> > m_instancesPerMeshLoc;
  };

  
}

#endif //HYDRAXML_H
