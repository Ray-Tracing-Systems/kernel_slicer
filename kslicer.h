#ifndef KSLICER_H
#define KSLICER_H

#include <string>
#include <vector>
#include <unordered_map>

#include "clang/AST/DeclCXX.h"
#include "clang/Frontend/CompilerInstance.h"

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
    bool usedInMainFunc = false;
  };

  /**
  \brief for data member of MainClass
  */
  struct DataMemberInfo 
  {
    std::string name;
    std::string type;
    size_t      sizeInBytes;              ///<! may be not needed due to using sizeof in generated code, but it is useful for sorting members by size and making apropriate aligment
    size_t      offsetInTargetBuffer = 0; ///<! offset in bytes in terget buffer that stores all data members
    
    bool isContainer  = false;
    bool isArray      = false; ///<! if is array, element type stored incontainerDataType 
    bool usedInKernel = false; ///<! if any kernel use the member --> true; if no one uses --> false.
    size_t arraySize  = 0;
    std::string containerType;
    std::string containerDataType;
  };

  /**
  \brief for local variables of MainFunc
  */
  struct DataLocalVarInfo 
  {
    std::string name;
    std::string type;
    size_t      sizeInBytes;

    bool        isArray   = false;
    size_t      arraySize = 0;
    std::string typeOfArrayElement;
    size_t      sizeInBytesOfArrayElement = 0;
  };
  
  /**
  \brief collector of all information about input main class
  */
  struct MainClassInfo
  {
    std::vector<KernelInfo>      kernels;         ///<! only those kerneles which are called from main function
    std::vector<DataMemberInfo>  classVariables;  ///<! only those member variables which are referenced from kernels 
    std::vector<DataMemberInfo>  containers;      ///<! containers that should be transformed to buffers

    std::unordered_map<std::string, KernelInfo>     allKernels;
    std::unordered_map<std::string, DataMemberInfo> allDataMembers;

    const clang::CXXMethodDecl*                       mainFuncNode;
    std::unordered_map<std::string, DataLocalVarInfo> mainFuncLocals;

    //std::vector<const clang::FunctionDecl*>  localFunctions; ///<! functions from main file that should be generated in .cl file
    //std::vector<const clang::CXXMethodDecl*> localMembers;   ///<! member function of main class that should be decorated and then generated in .cl file 

    std::string mainClassName;
    std::string mainClassFileName;
    std::string mainClassFileInclude;
    std::string mainFuncName;

    std::unordered_map<std::string, bool> allIncludeFiles; // true if we need to include it in to CL, false otherwise
  };

  /**
  \brief select local variables of main class that can be placed in auxilary buffer
  */
  std::vector<DataMemberInfo> MakeClassDataListAndCalcOffsets(std::unordered_map<std::string, DataMemberInfo>& vars);

  
  void ReplaceOpenCLBuiltInTypes(std::string& a_typeName);

  std::string ProcessKernel(const clang::CXXMethodDecl* a_node, clang::CompilerInstance& compiler, const MainClassInfo& a_codeInfo);
  
  std::vector<std::string> GetAllPredefinedThreadIdNames();
};

std::string GetRangeSourceCode(const clang::SourceRange a_range, clang::SourceManager& sm); 

#endif