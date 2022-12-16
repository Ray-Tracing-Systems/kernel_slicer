#ifndef HYDRAXML_H
#define HYDRAXML_H

#include "pugixml.hpp"
#include "LiteMath.h"
using namespace LiteMath;

#include <vector>
#include <string>
#include <sstream>
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
  //LiteMath::float3   read3f(pugi::xml_attribute a_attr);
  //LiteMath::float3   read3f(pugi::xml_node a_node);
  LiteMath::float3   readval3f(pugi::xml_node a_node);
  float              readval1f(const pugi::xml_node a_color);
  int                readval1i(const pugi::xml_node a_color);
  unsigned int       readval1u(const pugi::xml_node a_color);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct Instance
  {
    uint32_t           geomId = uint32_t(-1); ///< geom id
    uint32_t           rmapId = uint32_t(-1); ///< remap list id, todo: add function to get real remap list by id
    LiteMath::float4x4 matrix;                ///< transform matrix
    pugi::xml_node     node;
  };

  struct LightInstance
  {
    uint32_t           instId  = uint32_t(-1);
    uint32_t           lightId = uint32_t(-1);
    pugi::xml_node     instNode;
    pugi::xml_node     lightNode;
    LiteMath::float4x4 matrix;
    pugi::xml_node     node;
  };

  struct Camera
  {
    float pos[3];
    float lookAt[3];
    float up[3];
    float fov;
    float nearPlane;
    float farPlane;
    pugi::xml_node node;
  };

  struct Settings
  {
    uint32_t width;
    uint32_t height;
    uint32_t spp;
    uint32_t depth;
    uint32_t depthDiffuse;
    pugi::xml_node node;
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
      inst.node   = (*m_iter);
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
      cam.fov       = hydra_xml::readval1f(m_iter->child(L"fov")); 
      cam.nearPlane = hydra_xml::readval1f(m_iter->child(L"nearClipPlane"));
      cam.farPlane  = hydra_xml::readval1f(m_iter->child(L"farClipPlane"));  
      
      LiteMath::float3 pos    = hydra_xml::readval3f(m_iter->child(L"position"));
      LiteMath::float3 lookAt = hydra_xml::readval3f(m_iter->child(L"look_at"));
      LiteMath::float3 up     = hydra_xml::readval3f(m_iter->child(L"up"));
      for(int i=0;i<3;i++)
      {
        cam.pos   [i] = pos[i];
        cam.lookAt[i] = lookAt[i];
        cam.up    [i] = up[i];
      }
      cam.node = (*m_iter);
      return cam;
    }
  
		const CamIterator& operator++() { ++m_iter; return *this; }
		CamIterator operator++(int)     { m_iter++; return *this; }
  
		const CamIterator& operator--() { --m_iter; return *this; }
		CamIterator operator--(int)     { m_iter--; return *this; }
  
  private:
    pugi::xml_node_iterator m_iter;
	};

  class RemapListIterator //
	{
	  friend class pugi::xml_node;
    friend class pugi::xml_node_iterator;
  
	public:
  
		// Default constructor
		RemapListIterator() {}
		RemapListIterator(const pugi::xml_node_iterator& a_iter) : m_iter(a_iter) {}
  
		// Iterator operators
		bool operator==(const RemapListIterator& rhs) const { return m_iter == rhs.m_iter;}
		bool operator!=(const RemapListIterator& rhs) const { return (m_iter != rhs.m_iter); }
  
    std::vector<int32_t> operator*() const 
    { 
      int size = m_iter->attribute(L"size").as_int();
      std::vector<int32_t> remapList(size); 
      std::string strData = ws2s(m_iter->attribute(L"val").as_string());
      std::stringstream strIn(strData.data());
      for(int i=0;i<size;i++)
        strIn >> remapList[i]; 
      return remapList;
    }
  
		const RemapListIterator& operator++() { ++m_iter; return *this; }
		RemapListIterator operator++(int)     { m_iter++; return *this; }
  
		const RemapListIterator& operator--() { --m_iter; return *this; }
		RemapListIterator operator--(int)     { m_iter--; return *this; }
  
  private:
    pugi::xml_node_iterator m_iter;
	};
  
  class SettingsIterator //
	{
	  friend class pugi::xml_node;
    friend class pugi::xml_node_iterator;
  
	public:
  
		// Default constructor
		SettingsIterator() {}
		SettingsIterator(const pugi::xml_node_iterator& a_iter) : m_iter(a_iter) {}
  
		// Iterator operators
		bool operator==(const SettingsIterator& rhs) const { return m_iter == rhs.m_iter;}
		bool operator!=(const SettingsIterator& rhs) const { return (m_iter != rhs.m_iter); }
  
    Settings operator*() const 
    { 
      Settings settings;
      settings.width  = hydra_xml::readval1u(m_iter->child(L"width"));
      settings.height = hydra_xml::readval1u(m_iter->child(L"height"));
      settings.depth  = hydra_xml::readval1u(m_iter->child(L"trace_depth"));
      settings.depthDiffuse = hydra_xml::readval1u(m_iter->child(L"diff_trace_depth"));
      settings.spp    = hydra_xml::readval1u(m_iter->child(L"maxRaysPerPixel"));
      settings.node   = (*m_iter);
      return settings;
    }
  
		const SettingsIterator& operator++() { ++m_iter; return *this; }
		SettingsIterator operator++(int)     { m_iter++; return *this; }
  
		const SettingsIterator& operator--() { --m_iter; return *this; }
		SettingsIterator operator--(int)     { m_iter--; return *this; }
  
  private:
    pugi::xml_node_iterator m_iter;
	};


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

    pugi::xml_object_range<InstIterator> InstancesGeom() { return pugi::xml_object_range(InstIterator(m_scenesNode.child(L"scene").child(L"instance"), m_scenesNode.child(L"scene").end()), 
                                                                                         InstIterator(m_scenesNode.child(L"scene").end(), m_scenesNode.child(L"scene").end())
                                                                                         ); }
    
    std::vector<LightInstance> InstancesLights(uint32_t a_sceneId = 0);

    pugi::xml_object_range<RemapListIterator> RemapLists()  { return pugi::xml_object_range(RemapListIterator(m_scenesNode.child(L"scene").child(L"remap_lists").begin()), 
                                                                                            RemapListIterator(m_scenesNode.child(L"scene").child(L"remap_lists").end())
                                                                                            ); }

    pugi::xml_object_range<CamIterator>      Cameras()  { return pugi::xml_object_range(CamIterator(m_cameraLib.begin()), CamIterator(m_cameraLib.end())); }
    pugi::xml_object_range<SettingsIterator> Settings() { return pugi::xml_object_range(SettingsIterator(m_settingsNode.begin()), SettingsIterator(m_settingsNode.end())); }

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
    std::string        m_libraryRootDir;
    pugi::xml_node     m_texturesLib; 
    pugi::xml_node     m_materialsLib; 
    pugi::xml_node     m_geometryLib; 
    pugi::xml_node     m_lightsLib;
    pugi::xml_node     m_cameraLib; 
    pugi::xml_node     m_settingsNode; 
    pugi::xml_node     m_scenesNode; 
    pugi::xml_document m_xmlDoc;

    std::unordered_map<std::string, std::vector<LiteMath::float4x4> > m_instancesPerMeshLoc;
  };

  
}

#endif //HYDRAXML_H
