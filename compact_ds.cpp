#include "kslicer.h"

#include <iostream>
#include <cctype>

nlohmann::json kslicer::NonConstantDataOfKernel(const nlohmann::json& a_kernelOptions, const std::string& a_kernelName, bool a_entryWillBeAdded)
{
  if(!a_kernelOptions.is_object())
    return nullptr;

  auto pKernel = a_kernelOptions.find(a_kernelName);
  if(pKernel == a_kernelOptions.end() && !a_entryWillBeAdded)
    return nullptr;

  nlohmann::json options = (pKernel == a_kernelOptions.end()) ? nlohmann::json() : (*pKernel);
  if(options == nullptr) // empty entry of the kernel ==> options of "all"/"default"
  {
    auto pDefault = a_kernelOptions.find("all");
    if(pDefault == a_kernelOptions.end())
      pDefault = a_kernelOptions.find("default");
    if(pDefault != a_kernelOptions.end())
      options = (*pDefault);
  }

  if(!options.is_object() || !options.contains("nonConstantData"))
    return nullptr;
  return options["nonConstantData"];
}

const kslicer::CompactBufferGroup* kslicer::MainClassInfo::FindCompactGroup(const std::string& a_memberName) const
{
  auto p = compactGroupOfMember.find(a_memberName);
  if(p == compactGroupOfMember.end())
    return nullptr;
  return &compactGroups[p->second];
}

// 'float' ==> '1f', 'uint2' ==> '2ui', 'int4' ==> '4i', 'MaterialData' ==> '_MaterialData'
//
static std::string CompactBufferSuffix(const std::string& a_shaderType)
{
  static const std::vector< std::pair<std::string, std::string> > baseTypes = { {"float", "f"}, {"uint", "ui"}, {"int", "i"}, {"half", "h"}, {"double", "d"} };
  for(const auto& base : baseTypes)
  {
    if(a_shaderType == base.first)
      return "1" + base.second;
    if(a_shaderType.size() == base.first.size() + 1 && a_shaderType.find(base.first) == 0 && a_shaderType.back() >= '2' && a_shaderType.back() <= '4')
      return std::string(1, a_shaderType.back()) + base.second;
  }

  std::string res = "_";
  for(char c : a_shaderType)
    res += std::isalnum((unsigned char)c) ? c : '_';
  return res;
}

void kslicer::MainClassInfo::MakeCompactBufferGroups(const nlohmann::json& a_kernelOptions)
{
  compactGroups.clear();
  compactGroupOfMember.clear();
  if(!compactDS)
    return;

  kslicer::umap<std::string, const DataMemberInfo*> membersByName;
  for(const auto& member : dataMembers)
    if(member.isContainer)
      membersByName[member.name] = &member;

  // (1) vectors used in shaders and vectors which are written at least in one shader;
  //     constness is the same as for shader buffers in PrepareJsonForKernels: "nonConstantData" of kernel options or 'const' of member;
  //     kernels without entry in options get an empty entry before shaders are generated (see kslicer_main.cpp), except block expansion kernels
  //
  kslicer::uset<std::string> usedInShaders, notCompactable;
  auto processKernel = [&](const std::string& a_kernelName, bool a_entryWillBeAdded, const std::map<std::string, UsedContainerInfo>& a_containers)
  {
    const nlohmann::json nonConstData = kslicer::NonConstantDataOfKernel(a_kernelOptions, a_kernelName, a_entryWillBeAdded);
    for(const auto& c : a_containers)
    {
      usedInShaders.insert(c.first);
      if(c.second.isSetter || c.second.bindWithRef)
      {
        notCompactable.insert(c.first);
        continue;
      }
      auto pMember = membersByName.find(c.first);
      if(pMember == membersByName.end())
        continue;
      bool isConst = pMember->second->isConst;
      if(nonConstData != nullptr)
      {
        auto pFlag = nonConstData.find(c.first);
        isConst = (pFlag == nonConstData.end() || pFlag->is_null() || pFlag->get<int>() == 0);
      }
      if(!isConst)
        notCompactable.insert(c.first);
    }
  };

  for(const auto& k : kernels)
  {
    if(megakernelRTV && k.second.pattern == PATTERN_TP::PATTERN_RTV) // in megakernel mode RTV kernels are joined to megakernel of control function
      continue;
    processKernel(k.first, !k.second.be.enabled, k.second.usedContainers);
  }

  if(megakernelRTV)
  {
    for(const auto& cf : mainFunc)
    {
      if(cf.pattern != PATTERN_TP::PATTERN_RTV)
        continue;
      std::map<std::string, UsedContainerInfo> containers;
      for(const auto& kernelName : cf.UsedKernels)
      {
        auto pKernel = kernels.find(kernelName);
        if(pKernel != kernels.end())
          for(const auto& c : pKernel->second.usedContainers)
            containers[c.first] = c.second;
      }
      processKernel(cf.Name + "Mega", true, containers);
    }
  }

  // (2) group of 'CommitAppendedData()' for each vector, as in PrepareJsonForAllCPP: all vectors of a joined buffer must be in the same group
  //
  const bool hasCommitAppended = (allMemberFunctions.find("CommitAppendedData") != allMemberFunctions.end());
  kslicer::umap<std::string, std::string> appendGroupOfVector;
  for(const auto& group : appendGroups)
    for(const auto& vecName : group.second)
      appendGroupOfVector[vecName] = group.first;

  // (3) split read-only vectors to groups by type of elements in shaders (and by append group)
  //
  std::vector<std::string> names;
  for(const auto& member : membersByName)
    names.push_back(member.first);
  std::sort(names.begin(), names.end());

  std::map<std::string, CompactBufferGroup> groupByKey;
  std::vector<std::string>                  keysInOrder;
  for(const auto& name : names)
  {
    const DataMemberInfo& member = *(membersByName[name]);
    const bool isLayers = std::any_of(nnLayersOfWeights.begin(), nnLayersOfWeights.end(), [&](const auto& nn) { return nn.second == name; });
    const bool plainVector = member.kind == DATA_KIND::KIND_VECTOR && kslicer::IsVectorContainer(member.containerType) && member.usage == DATA_USAGE::USAGE_USER &&
                             !member.hasPrefix && !member.isSingle && !member.bindWithRef && !member.isPointer && !IsVFHBuffer(name) &&
                             nnLayersOfWeights.find(name) == nnLayersOfWeights.end() && !isLayers && graphics.indexBuffers.find(name) == graphics.indexBuffers.end();
    if(!plainVector || usedInShaders.find(name) == usedInShaders.end() || notCompactable.find(name) != notCompactable.end())
      continue;

    std::string appendGroup = "";
    if(hasCommitAppended)
    {
      auto pGroup = appendGroupOfVector.find(name);
      appendGroup = (pGroup == appendGroupOfVector.end()) ? std::string("otherGroup") : pGroup->second;
    }

    const std::string shaderType = pShaderFuncRewriter->RewriteStdVectorTypeStr(pShaderCC->ProcessBufferType(member.containerDataType));
    const std::string key        = appendGroup + "|" + shaderType;
    auto pGroup = groupByKey.find(key);
    if(pGroup == groupByKey.end())
    {
      CompactBufferGroup group;
      group.name         = "kgen_data" + CompactBufferSuffix(shaderType) + (appendGroup.empty() ? std::string("") : "_" + appendGroup);
      group.shaderType   = shaderType;
      group.appendGroup  = appendGroup;
      group.scalarLayout = member.containerScalarLayout;
      pGroup = groupByKey.insert(std::make_pair(key, group)).first;
      keysInOrder.push_back(key);
    }
    pGroup->second.members.push_back(name);
  }

  // (4) only groups of two and more vectors reduce number of bindings
  //
  for(const auto& key : keysInOrder)
  {
    const auto& group = groupByKey[key];
    if(group.members.size() < 2)
      continue;
    if(allDataMembers.find(group.name) != allDataMembers.end())
    {
      std::cout << "  [kslicer]: ERROR, '-compact_ds': class member '" << group.name.c_str() << "' has the same name as joined buffer, rename it" << std::endl;
      continue;
    }
    for(const auto& name : group.members)
      compactGroupOfMember[name] = uint32_t(compactGroups.size());
    compactGroups.push_back(group);
  }

  // (5) offset of each vector inside of its joined buffer (in elements) is placed to class data ('ubo'): 'm_vec_offset'
  //
  for(const auto& group : compactGroups)
  {
    std::cout << "  compact_ds: " << group.name.c_str() << " (" << group.shaderType.c_str() << ") <== ";
    for(const auto& name : group.members)
    {
      std::cout << name.c_str() << " ";
      kslicer::DataMemberInfo offset;
      offset.type            = "uint";
      offset.sizeInBytes     = sizeof(unsigned int);
      offset.name            = name + "_offset";
      offset.usedInKernel    = true;
      offset.isContainerInfo = true;
      offset.kind            = kslicer::DATA_KIND::KIND_POD;
      dataMembers.push_back(offset);
    }
    std::cout << std::endl;
  }
}
