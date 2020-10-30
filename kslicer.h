#ifndef KSLICER_H
#define KSLICER_H

#include <string>
#include <unordered_map>

#include "clang/AST/DeclCXX.h"

namespace kslicer
{
  const std::string GetProjPrefix();

  /**
  \brief for each method MainClass::kernel_XXX
  */
  struct KernelInfo 
  {
    struct Arg 
    {
      std::string type;
      std::string name;
      int         size;
    };
    std::string      return_type;
    std::string      name;
    std::vector<Arg> args;
  
    const clang::CXXMethodDecl* astNode = nullptr;
  };

  /**
  \brief for data member of MainClass
  */
  struct DataMemberInfo 
  {
    std::string name;
    std::string type;
    size_t      sizeInBytes;              // may be not needed due to using sizeof in generated code, but it is useful for sorting members by size and making apropriate aligment
    size_t      offsetInTargetBuffer = 0; // offset in bytes in terget buffer that stores all data members
    
    bool isContainer = false;
    bool isArray     = false; // if is array, element type stored incontainerDataType 
    size_t arraySize = 0;
    std::string containerType;
    std::string containerDataType;
  };

  std::vector<DataMemberInfo> MakeClassDataListAndCalcOffsets(std::unordered_map<std::string, DataMemberInfo>& vars);

};

#endif