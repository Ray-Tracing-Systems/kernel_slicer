#ifndef KSLICER_TEMPLATE_RENDERING_H
#define KSLICER_TEMPLATE_RENDERING_H

#include <iostream>
#include <string>

#include "kslicer.h"

namespace kslicer
{
  nlohmann::json PrepareJsonForAllCPP(const MainClassInfo& a_classInfo, const clang::CompilerInstance& compiler,
                                      const std::vector<MainFuncInfo>& a_methodsToGenerate, const std::vector<kslicer::DeclInClass>& usedDecl,
                                      const std::string& a_genIncude, 
                                      const uint32_t threadsOrder[3],
                                      const std::string& a_composImplName, 
                                      const nlohmann::json& uboJson, 
                                      const TextGenSettings& a_settings,
                                      const IntersectionShader2& a_foundIS2);

  nlohmann::json PrepareJsonForKernels(MainClassInfo& a_classInfo,
                                       const std::vector<kslicer::FuncData>& usedFunctions,
                                       const std::vector<kslicer::DeclInClass>& usedDecl,
                                       const clang::CompilerInstance& compiler,
                                       const uint32_t    threadsOrder[3],
                                       const nlohmann::json& uboJson,
                                       const nlohmann::json& kernelOptions,
                                       const std::vector<std::string>& usedDefines,
                                       const TextGenSettings& a_settings);

  nlohmann::json PrepareUBOJson(MainClassInfo& a_classInfo,
                                const std::vector<kslicer::DataMemberInfo>& a_dataMembers,
                                const clang::CompilerInstance& compiler,
                                const TextGenSettings& a_settings);

  void ApplyJsonToTemplate(const std::filesystem::path& a_declTemplateFilePath, const std::filesystem::path& a_suffix, const nlohmann::json& a_data);

  nlohmann::json PutHierarchyToJson(const kslicer::MainClassInfo::VFHHierarchy& h, 
                                    const clang::CompilerInstance& compiler,
                                    const MainClassInfo& a_classInfo,
                                    size_t& fnGroupOffset);

  nlohmann::json PutHierarchiesDataToJson(const std::unordered_map<std::string, kslicer::MainClassInfo::VFHHierarchy>& hierarchies,
                                          const clang::CompilerInstance& compiler,
                                          const MainClassInfo& a_classInfo);

  nlohmann::json ListCallableStructures(const std::unordered_map<std::string, kslicer::MainClassInfo::VFHHierarchy>& hierarchies,
                                        const clang::CompilerInstance& compiler,
                                        const MainClassInfo& a_classInfo, 
                                        uint32_t& a_totalShaders);                                         

  nlohmann::json FindIntersectionHierarchy(nlohmann::json a_hierarchies);

  nlohmann::json GetOriginalKernelJson(const KernelInfo& k, const MainClassInfo& a_classInfo);

  std::vector<std::pair<std::string, std::string> > SortByKeysByLen(const std::unordered_map<std::string, std::string>& a_map);
}

#endif