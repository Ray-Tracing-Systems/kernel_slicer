#ifndef HYDRAXML_H
#define HYDRAXML_H

#include "pugixml.hpp"
#include "LiteMath.h"
using namespace LiteMath;

#include <vector>
#include <set>
#include <unordered_map>

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
  
  // Range-based for loop support
  template <typename It> 
  class range_obj
  {
  public:
  	typedef It const_iterator;
  	typedef It iterator;  
  	range_obj(It b, It e): _begin(b), _end(e){}
  	It begin() { return _begin; }
  	It end() { return _end; }  
  private:
  	It _begin, _end;
  };

  class LocIterator 
  {
   public:
     LocIterator(pugi::xml_node start, const std::string& a_libPath) : m_currNode(start), m_libraryRootDir(a_libPath) { m_isNull = (m_currNode == nullptr);  }  
     LocIterator(LocIterator& a_iter) : m_currNode(a_iter.m_currNode), m_libraryRootDir(a_iter.m_libraryRootDir) {}

     std::string operator*() const 
     { 
       auto attr = m_currNode.attribute(L"loc");
       if(attr == nullptr)
         return "";
       auto meshLoc = ws2s(std::wstring(attr.as_string()));
       return m_libraryRootDir + "/" + meshLoc;
     }
     
     const LocIterator& operator++() 
     { 
       m_currNode = m_currNode.next_sibling(); 
       m_isNull = (m_currNode == nullptr); 
       return *this; 
     }

     bool operator !=(const LocIterator &other) const { return (m_currNode != other.m_currNode) || (m_isNull != other.m_isNull); }
     bool operator ==(const LocIterator &other) const { return (m_currNode == other.m_currNode) && (m_isNull == other.m_isNull); }  
     
   private:
     pugi::xml_node      m_currNode;
     const std::string&  m_libraryRootDir;
     bool                m_isNull = false;
  };

  class InstanceIterator 
  {
   public:
    InstanceIterator(InstanceIterator& a_iter) : m_currNode(a_iter.m_currNode){}
    InstanceIterator(pugi::xml_node start) : m_currNode(start) { }

    Instance  operator*() const 
    { 
      Instance result;
      result.geomId = m_currNode.attribute(L"mesh_id").as_uint();
      result.rmapId = m_currNode.attribute(L"rmap_id").as_uint();
      result.matrix = float4x4FromString(m_currNode.attribute(L"matrix").as_string());
      return result; 
    }
    const InstanceIterator& operator++() { m_currNode = m_currNode.next_sibling(L"instance"); return *this; }
    bool operator !=(const InstanceIterator &other) const { return m_currNode != other.m_currNode; }  
      
   private:
     pugi::xml_node m_currNode;
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
    pugi::xml_object_range<pugi::xml_node_iterator> MaterialNodes() { return m_materialsLib.children(); } // #TODO: skip unused materials
    pugi::xml_object_range<pugi::xml_node_iterator> LightNodes()    { return m_lightsLib.children();    } // #TODO: skip unused lights

    range_obj<LocIterator>      MeshFiles()     { return range_obj(LocIterator(m_geometryLib.child(L"mesh"), m_libraryRootDir), LocIterator(pugi::xml_node(), m_libraryRootDir)); }
    range_obj<LocIterator>      TextureFiles()  { return range_obj(LocIterator(m_texturesLib.first_child(),  m_libraryRootDir), LocIterator(pugi::xml_node(), m_libraryRootDir)); }
    range_obj<InstanceIterator> InstancesGeom() { return range_obj(InstanceIterator(m_sceneNode.child(L"instance")), InstanceIterator(pugi::xml_node()));}
    
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
  };

  LiteMath::float3 read3f(pugi::xml_attribute a_attr);
  

}

#endif //HYDRAXML_H
