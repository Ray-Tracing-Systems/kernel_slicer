#ifndef KSLICER_H
#define KSLICER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "clang/AST/DeclCXX.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

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
      bool needFakeOffset = false;
      bool isThreadID     = false;
    };

    struct LoopIter 
    {
      std::string type;
      std::string name;
      std::string sizeExpr;
    };
    
    std::string      return_type;
    std::string      name;
    std::vector<Arg> args;      ///<! all arguments of a kernel
    std::vector<Arg> loopIters; ///<! info about internal loops inside kernel which should be eliminated (so these loops are transformed to kernel call); For IPV pattern.

    const clang::CXXMethodDecl* astNode = nullptr;
    bool usedInMainFunc = false;
    bool isBoolTyped    = false; ///<! special case: if kernel return boolean, we analyze loop exit (break) or function exit (return) expression
    bool usedInExitExpr = false;
    bool checkThreadFlags = false;

    std::string DeclCmd;
    std::unordered_set<std::string> usedVectors; // list of all std::vector<T> member names which is referenced inside kernel
    std::string rewrittenText;
  };

  /**
  \brief for each data member of MainClass
  */
  struct DataMemberInfo 
  {
    std::string name;
    std::string type;
    size_t      sizeInBytes;              ///<! may be not needed due to using sizeof in generated code, but it is useful for sorting members by size and making apropriate aligment
    size_t      offsetInTargetBuffer = 0; ///<! offset in bytes in terget buffer that stores all data members
    
    bool isContainerInfo = false; ///<! auto generated std::vector<T>::size() or capacity() or some analogue
    bool isContainer     = false; ///<! if std::vector<...>
    bool isArray         = false; ///<! if is array, element type stored incontainerDataType;
    bool usedInKernel    = false; ///<! if any kernel use the member --> true; if no one uses --> false;
    bool usedInMainFn    = false; ///<! if std::vector is used in MainFunction like vector.data().
    size_t arraySize     = 0;
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
  \brief for arguments of MainFunc which are pointers
  */
  struct InOutVarInfo 
  {
    std::string name;
  };

  // assume there could be only 4 form of kernel arg when kernel is called
  //
  enum class KERN_CALL_ARG_TYPE{
    ARG_REFERENCE_LOCAL         = 0, // Passing the address of a local variable or local array by pointer (for example "& rayPosAndNear" or just randsArray);
    ARG_REFERENCE_ARG           = 1, // Passing the pointer that was supplied to the argument of MainFunc (for example, just "out_color") 
    ARG_REFERENCE_CLASS_VECTOR  = 2, // Passing a pointer to a class member of type std::vector<T>::data() (for example m_materials.data())
    ARG_REFERENCE_CLASS_POD     = 3, // Passing a pointer to a member of the class of type plain old data. For example, "&m_worldViewProjInv"
    ARG_REFERENCE_THREAD_ID     = 4, // Passing tidX, tidY or tidZ
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
    std::string                     originKernelName;
    std::string                     callerName;
    std::vector<ArgReferenceOnCall> descriptorSetsInfo;
  };

  struct MainFuncNameInfo
  {
    std::string              name;
    std::vector<std::string> kernelNames;
  };

  enum class KernelStmtKind { CALL_TYPE_SIMPLE          = 1,
                              EXIT_TYPE_FUNCTION_RETURN = 2, 
                              EXIT_TYPE_LOOP_BREAK      = 3};

  struct KernelStatementInfo
  {
    std::string        kernelName;
    clang::SourceRange kernelCallRange;
    clang::SourceRange ifExprRange;
    KernelStmtKind     exprKind = KernelStmtKind::CALL_TYPE_SIMPLE;
    bool               isNegative = false;
  };

  struct MainFuncInfo
  {
    std::string                                       Name;
    const clang::CXXMethodDecl*                       Node;
    std::unordered_map<std::string, DataLocalVarInfo> Locals;
    std::unordered_map<std::string, InOutVarInfo>     InOuts;
    std::unordered_set<std::string>                   ExcludeList;
    std::unordered_set<std::string>                   UsedKernels;
    
    std::string GeneratedDecl;
    std::string CodeGenerated;

    size_t startDSNumber = 0;
    size_t endDSNumber   = 0;

    // RT template specific
    //
    std::unordered_map<uint64_t, KernelStatementInfo> ExitExprIfCond;
    std::unordered_map<uint64_t, KernelStatementInfo> ExitExprIfCall;
    std::unordered_map<uint64_t, KernelStatementInfo> CallsInsideFor;

    bool   needToAddThreadFlags = false;
  };
  
  class VariableAndFunctionFilter;

  /**
  \brief collector of all information about input main class
  */
  struct MainClassInfo
  {
    std::vector<KernelInfo>      kernels;     ///<! only those kernels which are called from 'Main'/'Control' functions
    std::vector<DataMemberInfo>  dataMembers; ///<! only those member variables which are referenced from kernels 
    std::vector<DataMemberInfo>  containers;  ///<! containers that should be transformed to buffers

    std::unordered_map<std::string, KernelInfo>     allKernels;
    std::unordered_map<std::string, DataMemberInfo> allDataMembers;

    std::vector<MainFuncInfo>                       mainFunc;

    std::string mainClassName;
    std::string mainClassFileName;
    std::string mainClassFileInclude;

    std::unordered_map<std::string, bool> allIncludeFiles; // true if we need to include it in to CL, false otherwise
    std::vector<KernelCallInfo>           allDescriptorSetsInfo;

    typedef std::vector<clang::ast_matchers::StatementMatcher>               MList;
    typedef std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback> MHandlerCFPtr;
    typedef std::unique_ptr<kslicer::VariableAndFunctionFilter>              MHandlerKFPtr;

    virtual std::string RemoveKernelPrefix(const std::string& a_funcName) const; ///<! "kernel_XXX" --> "XXX"; 
    virtual bool        IsKernel(const std::string& a_funcName) const;           ///<! return true if function is a kernel
    virtual bool        IsThreadIdArg(const KernelInfo::Arg& arg, const KernelInfo& a_kernel) const = 0;  ///<! return true if argument is used as threadId


    //// Processing Control Functions (CF)
    // 
    virtual MList         ListMatchers_CF(const std::string& mainFuncName) = 0;
    virtual MHandlerCFPtr MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler) = 0;
    virtual std::string   VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler) = 0;

    virtual void AddSpecVars_CF(std::vector<MainFuncInfo>&   a_mainFuncList, 
                                std::vector<KernelInfo>&     a_kernelList) {}

    virtual void PlugSpecVarsInCalls_CF(const std::vector<MainFuncInfo>&   a_mainFuncList, 
                                        const std::vector<KernelInfo>&     a_kernelList,
                                        std::vector<KernelCallInfo>&       a_kernelCalls) {}    
                                     
    //// \\

    //// Processing Kernel Functions (KF)
    //
    virtual MList         ListMatchers_KF(const std::string& mainFuncName) = 0; 
    virtual MHandlerKFPtr MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compiler) = 0;
    virtual std::string   VisitAndRewrite_KF(const KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler);

    virtual void ProcessCallArs_KF(const KernelCallInfo& a_call) { };
    
    //// These methods used for final template text rendering
    //
    virtual uint32_t GetKernelDim(const KernelInfo& a_kernel) const = 0; 

    struct ArgTypeAndNamePair
    {
      std::string typeName;
      std::string argName;
    };
   
    virtual std::vector<ArgTypeAndNamePair> GetKernelTIDArgs(const KernelInfo& a_kernel) const; 
    virtual std::vector<ArgTypeAndNamePair> GetKernelCommonArgs(const KernelInfo& a_kernel) const;
  };


  struct RTV_Pattern : public MainClassInfo
  {
    bool          IsThreadIdArg(const KernelInfo::Arg& arg, const KernelInfo& a_kernel) const override;

    MList         ListMatchers_CF(const std::string& mainFuncName) override;
    MHandlerCFPtr MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler) override;

    std::string   VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler) override;

    void AddSpecVars_CF(std::vector<MainFuncInfo>& a_mainFuncList, 
                        std::vector<KernelInfo>&   a_kernelList) override;

    void PlugSpecVarsInCalls_CF(const std::vector<MainFuncInfo>& a_mainFuncList, 
                                const std::vector<KernelInfo>&   a_kernelList,
                                std::vector<KernelCallInfo>&     a_kernelCalls) override;    
    
    MList         ListMatchers_KF(const std::string& mainFuncName) override;
    MHandlerKFPtr MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compiler) override;
    void          ProcessCallArs_KF(const KernelCallInfo& a_call) override;   

    uint32_t      GetKernelDim(const KernelInfo& a_kernel) const override;                       
  };

  struct IPV_Pattern : public MainClassInfo
  {
    std::string   RemoveKernelPrefix(const std::string& a_funcName) const override; ///<! "kernel2D_XXX" --> "XXX"; 
    bool          IsKernel(const std::string& a_funcName) const override;           ///<! return true if function is a kernel
    bool          IsThreadIdArg(const KernelInfo::Arg& arg, const KernelInfo& a_kernel) const override;

    MList         ListMatchers_CF(const std::string& mainFuncName) override;
    MHandlerCFPtr MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler) override;
  
    std::string   VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler) override; 

    MList         ListMatchers_KF(const std::string& mainFuncName) override;
    MHandlerKFPtr MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compiler) override; 

    uint32_t      GetKernelDim(const KernelInfo& a_kernel) const override;                     
  };



  /**
  \brief select local variables of main class that can be placed in auxilary buffer
  */
  std::vector<DataMemberInfo> MakeClassDataListAndCalcOffsets(std::unordered_map<std::string, DataMemberInfo>& vars);

  
  void ReplaceOpenCLBuiltInTypes(std::string& a_typeName);
  std::vector<std::string> GetAllPredefinedThreadIdNamesRTV();

  std::string GetRangeSourceCode(const clang::SourceRange a_range, const clang::CompilerInstance& compiler);
  std::string CutOffFileExt(const std::string& a_filePath);

  uint64_t GetHashOfSourceRange(const clang::SourceRange& a_range);
};



#endif