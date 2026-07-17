#ifndef MAINCLASS_H_
#define MAINCLASS_H_
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <vector>
#include <string>
#include "clang/AST/DeclCXX.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace kslicer {


struct MainClassInfo
  {
    std::unordered_map<std::string, const clang::CXXRecordDecl*> allASTNodes; ///<! AST nodes for all considered classes in program
    std::unordered_map<std::string, KernelInfo>     allKernels;       ///<! list of all kernels; used only on the second pass to identify Control Functions; it is not recommended to use it anywhere else
    std::unordered_map<std::string, DataMemberInfo> allDataMembers;   ///<! list of all class data members;
    std::unordered_map<std::string, ProbablyUsed>   usedProbably;     ///<! variables which are used in virtual functions and *probably* will be used in *SOME* kernels if they call these virtual functions

    std::unordered_set<std::string>                 usedServiceCalls; ///<! memcpy, memset, scan, sort and e.t.c.
    std::unordered_map<std::string, ServiceCall>    serviceCalls;     ///<! actual list of used service calls

    std::unordered_map<std::string, const clang::CXXMethodDecl*> allMemberFunctions;  ///<! in fact this is used for a specific case, RTV pattern, full impl function, check for user define 'XXXBlock' function for control function 'XXX'
                                                                                      ///<! and we do not support overloading here ...

    std::unordered_map<std::string, KernelInfo> kernels;            ///<! only those kernels which are called from 'Main'/'Control' functions
    std::unordered_map<std::string, KernelInfo> megakernelsByName;  ///<! megakernels for RTV pattern

    std::optional<std::pair<std::string, kslicer::KernelInfo>> FindKernelByName(const std::string& a_name) const;
    
    std::vector<std::string>                    indirectKernels; ///<! list of all kernel names which require indirect dispatch; The order is essential because it is used for indirect buffer offsets
    std::vector<DataMemberInfo>                 dataMembers;     ///<! only those member variables which are referenced from kernels
    std::vector<MainFuncInfo>                   mainFunc;        ///<! list of all control functions
  
    
    std::unordered_map<std::string, ArrayData>      m_threadLocalArrays;
    std::unordered_map<uint64_t, RewrittenFunction> m_functionsDone;

    std::string                                        mainClassName;         ///<! Current main class (derived)
    std::unordered_map<std::string, int>               mainClassNames;        ///<! All main classes (derived + base) 
    std::unordered_set<std::string>                    dataClassNames; 

    std::vector< std::pair<std::string, std::string> > intersectionShaders;
    std::vector< std::pair<std::string, std::string> > intersectionTriangle;
    std::unordered_set<std::string>                    intersectionWhiteList;
    std::unordered_set<std::string>                    intersectionBlackList;

    std::unordered_set<std::string>                    withBufferReference;
    std::unordered_set<std::string>                    withoutBufferReference;
    bool                                               withBufferReferenceAll = false;
    std::vector< std::pair<std::string, std::string> > userTypedefs;

    std::filesystem::path mainClassFileName;
    std::string           mainClassFileInclude;
    std::string           mainClassSuffix;
    
    std::unordered_map<std::string, std::string> composPrefix;
    std::unordered_set<std::string>              composIntersection;
    const clang::CXXRecordDecl* mainClassASTNode = nullptr;
    std::vector<const clang::CXXConstructorDecl* > ctors;
    std::string shaderFolderPrefix = "";
    ShaderFeatures          globalShaderFeatures;
    OptionalDeviceFeatures  globalDeviceFeatures;
    

    std::vector<std::filesystem::path> ignoreFolders;  ///<! in these folders files are ignored
    std::vector<std::filesystem::path> processFolders; ///<! in these folders files are processed to take functions and structures from them to shaders
    std::vector<std::string> ignoreFiles;    ///<! exception to 'processFolders'
    std::vector<std::string> processFiles;   ///<! exception to 'ignoreFolders'
    std::vector<std::string> cppIncudes;     ///<! additional includes which we need to insert in generated cpp file
    bool NeedToProcessDeclInFile(const std::string a_fileName) const;
    bool IsInExcludedFolder(const std::string& fileName);

    std::unordered_set<std::string> GetExcludedNames() const;

    std::unordered_map<std::string, bool> allIncludeFiles; // true if we need to include it in to CL, false otherwise
    std::vector<KernelCallInfo>           allDescriptorSetsInfo;

    std::shared_ptr<IShaderCompiler>            pShaderCC           = nullptr;
    std::shared_ptr<IHostCodeGen>               pHostCC             = nullptr;  
    std::shared_ptr<kslicer::FunctionRewriter>  pShaderFuncRewriter = nullptr;
    uint32_t m_indirectBufferSize = 0;            ///<! size of indirect buffer
    uint32_t m_timestampPoolSize  = uint32_t(-1); ///<! size of timestamp pool for all kernels calls

    typedef std::vector<clang::ast_matchers::StatementMatcher>               MList;
    typedef std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback> MHandlerCFPtr;
    typedef std::unique_ptr<kslicer::UsedCodeFilter>                         MHandlerKFPtr;

    virtual std::string RemoveKernelPrefix(const std::string& a_funcName) const;                          ///<! "kernel_XXX" --> "XXX";
    virtual bool        IsKernel(const std::string& a_funcName) const;                                    ///<! return true if function is a kernel
    virtual void        ProcessKernelArg(KernelInfo::ArgInfo& arg, const KernelInfo& a_kernel) const;     ///<!
    virtual bool        IsIndirect(const KernelInfo& a_kernel) const;
    virtual PATTERN_TP  PatternByKernelName(const std::string& a_kernelName);

    //// Processing Control Functions (CF)
    //
    virtual MList         ListMatchers_CF(const std::string& mainFuncName);
    virtual MHandlerCFPtr MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler);
    virtual void          VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler);

    virtual void AddSpecVars_CF(std::vector<MainFuncInfo>& a_mainFuncList, std::unordered_map<std::string, KernelInfo>& a_kernelList);

    virtual void PlugSpecVarsInCalls_CF(const std::vector<MainFuncInfo>&                      a_mainFuncList,
                                        const std::unordered_map<std::string, KernelInfo>&    a_kernelList,
                                        std::vector<KernelCallInfo>&                          a_kernelCalls);

    virtual void ProcessVFH(const std::vector<const clang::CXXRecordDecl*>& a_decls, const clang::CompilerInstance& a_compiler);
    virtual void ExtractVFHConstants(const clang::CompilerInstance& compiler, clang::tooling::ClangTool& Tool);
    virtual void AppendAllRefsBufferIfNeeded(std::vector<DataMemberInfo>& a_vector);
    virtual void AppendAccelStructForIntersectionShadersIfNeeded(std::vector<DataMemberInfo>& a_vector, std::string composImplName);
    virtual void AppendAccelStructForIntersectionShadersIfNeeded(std::vector<DataMemberInfo>& a_vector, const IntersectionShader2& a_shader);

    //// \\

    //// Processing Kernel Functions (KF)
    //
    virtual MList         ListMatchers_KF(const KernelInfo& a_kernel);
    virtual MHandlerKFPtr MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compiler);

    virtual std::string   VisitAndRewrite_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler,
                                             std::string& a_outLoopInitCode, std::string& a_outLoopFinishCode);
    virtual void          VisitAndPrepare_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler); // additional informational pass, does not rewrite the code!

    virtual void ProcessCallArs_KF(const KernelCallInfo& a_call);

    //// These methods used for final template text rendering
    //
    virtual uint32_t GetKernelDim(const KernelInfo& a_kernel) const;

    virtual std::vector<ArgFinal> GetKernelTIDArgs(const KernelInfo& a_kernel) const;
    virtual std::vector<ArgFinal> GetKernelCommonArgs(const KernelInfo& a_kernel) const;

    virtual void        GetCFSourceCodeCmd(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler, bool a_megakernelRTV);
    virtual std::string GetCFDeclFromSource(const std::string& sourceCode);

    virtual void AddTempBufferToKernel(const std::string a_buffName, const std::string a_elemTypeName, KernelInfo& a_kernel); ///<! if kernel need some additional buffers (for reduction for example) use this function

    struct DImplFunc
    {
      const clang::CXXMethodDecl* decl = nullptr;
      std::string                 name;
      std::string                 nameRewritten;
      std::string                 srcRewritten;
      bool                        isEmpty        = false;
      bool                        isConstMember  = false;
      bool                        isIntersection = false;
    };

    struct DImplClass
    {
      const clang::CXXRecordDecl* decl = nullptr;
      std::string                 name;
      std::vector<DImplFunc>      memberFunctions;
      std::vector<std::string>    fields;
      bool                        isEmpty = false; ///<! empty if all memberFunctions are empty
      std::string                 objBufferName;
      std::string                 interfaceName;
      std::string                 tagName;
      uint32_t                    tagId;
    };

    enum  VFH_LEVEL{ VFH_LEVEL_1 = 1, // all imlementations are same size as interface, switch-based impl. in shader
                     VFH_LEVEL_2 = 2, // implementations of different size, GLSL_EXT_buffer_reference2, switch-based impl. in shader
                     VFH_LEVEL_3 = 3  // implementations of different size, GLSL_EXT_buffer_reference2, callable-shaders based implementation; 
                     };               // select between VFH_LEVEL_2 and VFH_LEVEL_3 is a responsibility of generator option and, there is no difference of them for user

    struct VFHTagInfo
    {
      std::string name;
      uint32_t    id;
    };

    struct VFHHierarchy
    {
      const clang::CXXRecordDecl* interfaceDecl = nullptr;
      std::string                 interfaceName;
      std::string                 objBufferName;
      std::string                 accStructName;
      std::vector<DImplClass>     implementations;
      VFH_LEVEL                   level = VFH_LEVEL_1;
      bool                        hasIntersection = false;

      std::vector<kslicer::DeclInClass>            usedDecls;
      std::unordered_map<std::string, VFHTagInfo>  tagByClassName;
      std::map<std::string, kslicer::FuncData>     virtualFunctions;
    };

    struct BufferReference 
    {
      std::string name;
      std::string typeOfElem;
    };

    bool halfFloatTextures  = false;
    bool megakernelRTV      = false;
    bool persistentRTV      = false; // current implementation for persistent threads on done only for megakernels in RTV
    bool useComplexNumbers  = false;
    bool genGPUAPI          = false;
    bool forceAllBufToRefs  = false;
    bool placeVectorsInUBO  = false;
    bool shitIsAlwaysConst  = false;
    bool hasLocalContainers = false;

    std::unordered_map<std::string, VFHHierarchy> m_vhierarchy;
    std::vector<BufferReference>                  m_allRefsFromVFH;
    bool IsVFHBuffer(const std::string& a_name, VFH_LEVEL* pOutLevel = nullptr, VFHHierarchy* pHierarchy = nullptr) const;

    std::unordered_set<std::string> ExtractTypesFromUsedContainers(const std::unordered_map<std::string, kslicer::DeclInClass>& a_otherDecls);
    void ProcessMemberTypes(const std::unordered_map<std::string, kslicer::DeclInClass>& a_otherDecls, clang::SourceManager& a_srcMgr,
                            std::vector<kslicer::DeclInClass>& generalDecls);

    void ProcessMemberTypesAligment(std::vector<DataMemberInfo>& a_members, const std::unordered_map<std::string, kslicer::DeclInClass>& a_otherDecls, const clang::ASTContext& a_astContext);

    std::unordered_map<std::string, VFHHierarchy> SelectVFHOnlyUsedByKernel(const std::unordered_map<std::string, VFHHierarchy>& a_hierarhices, const KernelInfo& k) const;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<std::string>                           m_setterStructDecls;
    std::vector<std::string>                           m_setterFuncDecls;
    std::unordered_map<std::string, std::string>       m_setterVars;
    std::unordered_map<std::string, DataMemberInfo>    m_setterData;

    void ProcessAllSetters(const std::unordered_map<std::string, const clang::CXXMethodDecl*>& a_setterFunc, clang::CompilerInstance& a_compiler);
    void ProcessBlockExpansionKernel(KernelInfo& a_kernel, const clang::CompilerInstance& compiler);

    std::vector< std::pair<std::string, std::string> > GetFieldsFromStruct(const clang::CXXRecordDecl* recordDecl, size_t* pSummOfFiledsSize = nullptr) const;
    bool HasBufferReferenceBind() const;
  };


}

#endif