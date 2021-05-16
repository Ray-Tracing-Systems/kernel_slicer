#ifndef HYDRAXML_H
#define HYDRAXML_H

#include "pugixml.hpp"
#include "OpenCLMath.h" 
#include <vector>
#include <set>
#include <unordered_map>

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#include <android/log.h>
#endif

namespace hydra_xml
{
    struct HydraScene
    {
        HydraScene() = default;
        ~HydraScene() = default;

#if defined(__ANDROID__)
        int LoadState(AAssetManager* mgr, const std::string &path);
#else
        int LoadState(const std::string &path);
#endif

        std::vector<std::string> m_meshloc;
        //std::vector<std::string> m_texloc;
        std::unordered_map<std::string, std::vector<LiteMath::float4x4> > m_instancesPerMeshLoc;

    private:
        void parseInstancedMeshes(pugi::xml_node a_scenelib, pugi::xml_node a_geomlib);
        void LogError(const std::string &msg);

        std::set<std::string> unique_meshes;
        std::string m_libraryRootDir;
    };

    LiteMath::float4x4 float4x4FromString(const std::wstring &matrix_str);

}




#endif //HYDRAXML_H
