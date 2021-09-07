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
    uint32_t           rmapId = uint32_t(-1); ///< remap list id, todo: add function to ger real remp list
    LiteMath::float4x4 matrix; ///< trannform matrix
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //class InstanceIterator 
  //{
  // public:
  //  InstanceIterator(InstanceIterator& a_iter) : m_currNode(a_iter.m_currNode){}
  //  InstanceIterator(pugi::xml_node start) : m_currNode(start) { }
  //
  //  Instance  operator*() const 
  //  { 
  //    Instance result;
  //    result.geomId = m_currNode.attribute(L"mesh_id").as_uint();
  //    result.rmapId = m_currNode.attribute(L"rmap_id").as_uint();
  //    result.matrix = float4x4FromString(m_currNode.attribute(L"matrix").as_string());
  //    return result; 
  //  }
  //  const InstanceIterator& operator++() { m_currNode = m_currNode.next_sibling(L"instance"); return *this; }
  //  bool operator !=(const InstanceIterator &other) const { return m_currNode != other.m_currNode; }  
  //    
  // private:
  //   pugi::xml_node m_currNode;
  //};

  //
  //
	class LocIterator
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
    
    std::vector<std::string> m_meshloc;                                                      // #TODO: replace with ListGeom or ListMeshes
    std::unordered_map<std::string, std::vector<LiteMath::float4x4> > m_instancesPerMeshLoc; // #TODO: put this inside
    
    //// use this functions with C++11 range for 
    //
    //pugi::xml_object_range<MyIterator> MaterialNodes() { return pugi::xml_object_range<MyIterator>(m_materialsLib.begin(), m_materialsLib.end()); } //m_materialsLib.children(); } // #TODO: skip unused materials
    pugi::xml_object_range<pugi::xml_node_iterator> MaterialNodes() { return m_materialsLib.children(); }
    pugi::xml_object_range<pugi::xml_node_iterator> LightNodes()    { return m_lightsLib.children();    } // #TODO: skip unused lights

    pugi::xml_object_range<LocIterator>        MeshFiles()     { return pugi::xml_object_range(LocIterator(m_geometryLib.begin(), m_libraryRootDir), 
                                                                                               LocIterator(m_geometryLib.end(), "")); }

    //pugi::xml_object_range<MyIterator>        MeshFiles()     { return pugi::xml_object_range(MyIterator(m_geometryLib.begin(), m_libraryRootDir), 
    //                                                                                          MyIterator(m_geometryLib.end(), "")); }

    //range_obj<LocIterator>      TextureFiles()  { return range_obj(LocIterator(m_texturesLib.first_child(),  m_libraryRootDir), LocIterator(m_texturesLib.end())); }
    //range_obj<InstanceIterator> InstancesGeom() { return range_obj(InstanceIterator(m_sceneNode.child(L"instance")), InstanceIterator(pugi::xml_node()));}
    
  //private:
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
  };

  LiteMath::float3 read3f(pugi::xml_attribute a_attr);
  

}

#endif //HYDRAXML_H
