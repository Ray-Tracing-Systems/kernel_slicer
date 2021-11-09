#include <stdio.h>
#include <vector>
#include <system_error>
#include <iostream>
#include <fstream>

#include <unordered_map>
#include <iomanip>

#include "kslicer.h"





std::unordered_map<std::string, kslicer::SetterStruct> kslicer::ProcessAllSetters(const std::unordered_map<std::string, const clang::CXXMethodDecl*>& a_setterFunc, clang::CompilerInstance& a_compiler)
{
  std::unordered_map<std::string, kslicer::SetterStruct> res;
 
  for(const auto kv : a_setterFunc)
  {
    const clang::CXXMethodDecl* node = kv.second;
    for(unsigned paramId = 0; paramId < node->getNumParams(); paramId++)
    {
      const clang::ParmVarDecl* pParam  = node->getParamDecl(paramId);
      const clang::QualType typeOfParam = pParam->getType();
      const clang::CXXRecordDecl* pDecl = typeOfParam->getAsCXXRecordDecl();
      if(pDecl == nullptr)
        continue;
    
      // process (pParam, typeOfParam, pDecl)
      //
      std::string formalName = pParam->getNameAsString();
      std::string typeName   = typeOfParam.getAsString();
      int a = 2;
      //auto paramInfo = kslicer::GetParamInfo(pParam);
    }
  }
 
  return res;
}