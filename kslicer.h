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

  // assume there could be only 4 form of kernel arg when kernel is called
  //
  enum class KERN_CALL_ARG_TYPE{
    ARG_REFERENCE_LOCAL         = 0, // Passing the address of a local variable or local array by pointer (for example "& rayPosAndNear" or just randsArray);
    ARG_REFERENCE_ARG           = 1, // Passing the pointer that was supplied to the argument of MainFunc (for example, just "out_color") 
    ARG_REFERENCE_CLASS_VECTOR  = 2, // Passing a pointer to a class member of type std::vector<T>::data() (for example m_materials.data())
    ARG_REFERENCE_CLASS_POD     = 3, // Passing a pointer to a member of the class of type plain old data. For example, "&m_worldViewProjInv"
    ARG_REFERENCE_UNKNOWN_TYPE  = 9  // Unknown type of arument yet. Generaly means we need to furthe process it, for example find among class variables or local variables
    };

  struct ArgReferenceOnCall
  {
    KERN_CALL_ARG_TYPE argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE;
    std::string        varName = "";
    bool umpersanned           = false; // just signal that '&' was applied to this argument, and thus it is likely to be (ARG_REFERENCE_LOCAL or ARG_REFERENCE_CLASS_POD)
  };


  struct KernelCallInfo
  {
    std::string                     kernelName;
    std::vector<ArgReferenceOnCall> allDescriptorSetsInfo;
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
    std::vector<KernelCallInfo>           allDescriptorSetsInfo;
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