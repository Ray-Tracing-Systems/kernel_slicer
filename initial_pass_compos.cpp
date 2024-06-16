#include "initial_pass.h"
#include <iostream>
#include <vector>
#include <string>

std::string FindPrefixName(const kslicer::ClassInfo& mainClassInfo, const kslicer::ClassInfo& apiClassInfo)
{
  std::string prefixName = "";

  std::vector<std::string> allowedNames = {
    apiClassInfo.name, 
    apiClassInfo.name + "*",
    std::string("std::shared_ptr<") + apiClassInfo.name + ">",
    std::string("std::unique_ptr<") + apiClassInfo.name + ">"
  };

  for(const auto& member : mainClassInfo.dataMembers) {
    for(const auto& possName : allowedNames)
      if(member.second.type == possName)
        prefixName = member.second.name;
  }

  return prefixName;
}


std::string kslicer::PerformClassComposition(kslicer::ClassInfo& mainClassInfo, const kslicer::ClassInfo& apiClassInfo, const kslicer::ClassInfo& implClassInfo)
{
  // (1) find member with type of apiClassInfo.name
  //
  std::string prefixName = FindPrefixName(mainClassInfo, apiClassInfo);
  if(prefixName == "")
    return "";
 
  // (2) merge data and functions (dataMembers, allMemberFunctions)
  //
  for(auto member : implClassInfo.dataMembers) {
    member.second.name       = prefixName + "_" + member.second.name;
    member.second.hasPrefix  = true;
    member.second.prefixName = prefixName;
    mainClassInfo.dataMembers[member.second.name] = member.second;
  }

  for(auto member : implClassInfo.allMemberFunctions) {
    std::string name = prefixName + "_" + member.first;
    mainClassInfo.allMemberFunctions[name] = member.second;
  }

  return prefixName;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> kslicer::GetBaseClassesNames(const clang::CXXRecordDecl* mainClassASTNode)
{
  std::vector<std::string> baseClassNames;
  if(mainClassASTNode == nullptr)
    return baseClassNames;

  // Итерируемся по базовым классам
  for (const auto& base : mainClassASTNode->bases()) {
    const clang::CXXRecordDecl* baseDecl = base.getType()->getAsCXXRecordDecl();
    if (baseDecl) {
      baseClassNames.push_back(baseDecl->getNameAsString());
    }
  }

  return baseClassNames;
}

namespace kslicer 
{
  // Helper function to get the depth of a class in the inheritance hierarchy
  int GetClassDepth(const clang::CXXRecordDecl* derived) 
  {
    int depth = 0;
    while (!derived->bases().empty()) {
      derived = derived->bases_begin()->getType()->getAsCXXRecordDecl();
      depth++;
    }
    return depth;
  }
}

std::vector<const clang::CXXRecordDecl*> kslicer::ExtractAndSortBaseClasses(const std::vector<const clang::CXXRecordDecl*>& classes, const clang::CXXRecordDecl* derived) 
{
  std::vector<const clang::CXXRecordDecl*> result;
  result.reserve(classes.size());
  
  for(size_t i=0;i<classes.size();i++)
    if(derived->isDerivedFrom(classes[i]))
      result.push_back(classes[i]);
  
  std::unordered_map<const clang::CXXRecordDecl*, int> depth;
  for(auto node : result)
    depth[node] = GetClassDepth(node);

  std::sort(result.begin(), result.end(), [&](const clang::CXXRecordDecl* a, const clang::CXXRecordDecl* b) { return depth[a] > depth[b]; });

  return result;
}