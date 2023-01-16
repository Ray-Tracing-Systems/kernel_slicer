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
    member.second.name = prefixName + "_" + member.second.name;
    mainClassInfo.dataMembers[member.second.name] = member.second;
  }

  for(auto member : implClassInfo.allMemberFunctions) {
    std::string name = prefixName + "_" + member.first;
    mainClassInfo.allMemberFunctions[name] = member.second;
  }

  return prefixName;
}