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
 
  // (2) merge data (dataMembers) and functions (allMemberFunctions)
  //
  for(auto member : implClassInfo.dataMembers) {
    member.second.name       = prefixName + "_" + member.second.name;
    member.second.hasPrefix  = true;
    member.second.prefixName = prefixName;
    mainClassInfo.dataMembers[member.second.name] = member.second;
  }

  for(auto member : implClassInfo.funMembers) {
    std::string name = prefixName + "_" + member.first;
    mainClassInfo.funMembers[name] = member.second;
  }

  // (3) merge kernels (...)
  //

  // ... 

  return prefixName;
}

void kslicer::PerformInheritanceMerge(kslicer::ClassInfo& mainClassInfo, const kslicer::ClassInfo& baseClassInfo)
{
  // (2) merge data and functions (dataMembers, allMemberFunctions)
  //
  for(auto member : baseClassInfo.dataMembers) 
  {
    member.second.name       = member.second.name;
    member.second.hasPrefix  = false;
    member.second.prefixName = "";
    mainClassInfo.dataMembers[member.second.name] = member.second;
  }
   
  // merge general functions from base class
  //
  for(auto member : baseClassInfo.funMembers) 
  {
    std::string name = member.first;
    auto p = mainClassInfo.funMembers.find(name);
    if(p == mainClassInfo.funMembers.end())             // because implementation in main (derived) class
      mainClassInfo.funMembers[name] = member.second;   // always overrides any implementations in base class
    else                                                             // but we can store overriden function with it's unique name: 'class_function'
    {                                                                // 
      std::string funName = baseClassInfo.name + "_" + member.first; //
      mainClassInfo.funMembers[funName] = member.second;             //
    }
  }
  
  // merge kernels from base class
  //
  for(const auto& f : baseClassInfo.funKernels) 
  {
    auto p = mainClassInfo.funKernels.find(f.first);             // because implementation in main (derived) class
    if(p == mainClassInfo.funKernels.end())                      // always overrides any implementations in base class
      mainClassInfo.funKernels[f.first] = f.second;
    else                                                         // but we can store overriden function with it's unique name: 'class_function'
    {                                                            // 
      std::string funName = baseClassInfo.name + "_" + f.first;  //
      mainClassInfo.funKernels[funName] = f.second;              //
    }
  }
  
  // merge control functions base class
  //
  for(const auto& f : baseClassInfo.funControls) 
  {
    auto p = mainClassInfo.funControls.find(f.first);   // because implementation in main (derived) class
    if(p == mainClassInfo.funControls.end())            // always overrides any implementations in base class
      mainClassInfo.funControls[f.first] = f.second;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> kslicer::GetBaseClassesNames(const clang::CXXRecordDecl* mainClassASTNode)
{
  std::vector<std::string> baseClassNames;
  return baseClassNames;
  
  /*
  if(mainClassASTNode == nullptr)
    return baseClassNames;
  
  std::cout << "[GetBaseClassesNames]: (0): ok, pointer = " << mainClassASTNode << std::endl;
  auto name = mainClassASTNode->getName().str();
  std::cout << "[GetBaseClassesNames]: (1): ok, pointer = " << mainClassASTNode << std::endl;
  std::cout << "[MainFuncSeeker]: find main class " << name.c_str() << ", name = " << name.c_str() << std::endl;

  // Итерируемся по базовым классам
  for (const auto& base : mainClassASTNode->bases()) {
    std::cout << "[GetBaseClassesNames]: (1): ok " << std::endl;
    const auto baseT = base.getType();
    std::cout << "[GetBaseClassesNames]: (2): ok " << std::endl;
    const clang::CXXRecordDecl* baseDecl = baseT->getAsCXXRecordDecl();
     std::cout << "[GetBaseClassesNames]: (3): ok " << std::endl;
    if (baseDecl)
      baseClassNames.push_back(baseDecl->getNameAsString());
  }

  return baseClassNames;
  */
 
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