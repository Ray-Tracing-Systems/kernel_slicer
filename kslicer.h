#ifndef KSLICER_H
#define KSLICER_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <filesystem>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <inja.hpp>
#pragma GCC diagnostic pop

#include "clang/AST/DeclCXX.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

static constexpr bool SLANG_ELIMINATE_LOCAL_POINTERS    = true;
static constexpr bool SLANG_SUPPORT_POINTER_ADD_IN_ARGS = false;

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);
std::string ToLowerCase(std::string a_str);

namespace kslicer
{
  struct TextGenSettings
  {
    std::string interfaceName;
    uint32_t wgpu_ver = 30;
    bool enableRayGen      = false;
    bool enableRayGenForce = false;
    bool enableMotionBlur  = false;
    bool enableCallable    = false;
    bool enableTimeStamps  = false;
    bool usePipelineCache  = false;
    bool genSeparateGPUAPI = false;
    bool useCUBforCUDA     = true;
    bool skipUBORead       = false;
    bool uboIsAlwaysConst  = false;
    bool uboIsAlwaysUniform = false;
  };

  struct IShaderCompiler;

  enum class DATA_KIND  { KIND_UNKNOWN = 0,
                          KIND_POD,                             ///<! Any Plain Old Data
                          KIND_POINTER,                         ///<! float*
                          KIND_VECTOR,                          ///<! std::vector<float>
                          KIND_TEXTURE,                         ///<! Texture2D<uchar4>
                          KIND_TEXTURE_SAMPLER_COMBINED,        ///<! std::shared_ptr<ICombinedImageSampler>
                          KIND_TEXTURE_SAMPLER_COMBINED_ARRAY,  ///<! std::vector< std::shared_ptr<ICombinedImageSampler> >
                          KIND_ACCEL_STRUCT,                    ///<! std::shared_ptr<ISceneObject>
                          KIND_HASH_TABLE,                      ///<! std::unordered_map<uint32_t,uint32_t>
                          KIND_SAMPLER                          ///<! Sampler
                          };

  enum class DATA_USAGE { USAGE_USER = 0, USAGE_SLICER_REDUCTION = 1 };
  enum class TEX_ACCESS { TEX_ACCESS_NOTHING = 0, TEX_ACCESS_READ = 1, TEX_ACCESS_WRITE = 2, TEX_ACCESS_SAMPLE = 4 };

  enum class CPP11_ATTR { ATTR_UNKNOWN = 0, ATTR_KERNEL = 1, ATTR_SETTER = 2  };

  /**
  \brief for each kernel we collect list of containes accesed by this kernel
  */
  struct UsedContainerInfo
  {
    std::string type;
    std::string name;
    DATA_KIND   kind   = DATA_KIND::KIND_UNKNOWN;

    bool isConst       = false;
    bool isSetter      = false;
    bool bindWithRef   = false;     
    bool isTexture()     const { return (kind == DATA_KIND::KIND_TEXTURE); }
    bool isAccelStruct() const { return (kind == DATA_KIND::KIND_ACCEL_STRUCT); }

    std::string setterPrefix;
    std::string setterSuffix;
  };

  bool  IsTextureContainer(const std::string& a_typeName); ///<! return true for all types of textures
  bool  IsSamplerTypeName(const std::string& a_typeName);  ///<! return true for all types of textures
  bool  IsCombinedImageSamplerTypeName(const std::string& a_typeName);  ///<! return true for all types of image combined samplers
  bool  IsMatrixTypeName(const std::string& a_typeName);

  struct ProbablyUsed
  {
    clang::FieldDecl* astNode = nullptr;
    bool isContainer;
    UsedContainerInfo info;
    std::string interfaceName;
    std::string className;
    std::string objBufferName;
    std::string containerType;
    std::string containerDataType;
  };

  struct ArgMatch
  {
    std::string formal;
    std::string actual;
    std::string type;
    uint32_t    argId;
    bool        isPointer = false;
  };

  /**
  \brief functions with input pointers which access global memory; they should be rewritten for GLSL.

         GLSL don't support pointers or passing buffers inside functions, so ... we have to insert all actual arguments inside function source code
         This greately complicate kernel_slicer work and we support currently only ove level of recursion, but we don't really have a choice.
  */
  struct ShittyFunction
  {
    std::vector<ArgMatch> pointers;
    std::string           originalName;
    std::string           ShittyName() const
    {
      std::string result = originalName;
      for(auto x : pointers)
        result += std::string("_") + x.actual;
      return result;
    }
  };

  /**
  \brief for each kernel we collect list of used shader features. These features should be enabled in GLSL later.
  */
  struct ShaderFeatures
  {
    ShaderFeatures operator||(const ShaderFeatures& rhs)
    {
      useByteType    = useByteType  || rhs.useByteType;
      useShortType   = useShortType || rhs.useShortType;
      useInt64Type   = useInt64Type || rhs.useInt64Type;
      useFloat64Type = useFloat64Type || rhs.useFloat64Type;
      useHalfType    = useHalfType    || rhs.useHalfType;

      useFloatAtomicAdd  = useFloatAtomicAdd  || rhs.useFloatAtomicAdd;
      useDoubleAtomicAdd = useDoubleAtomicAdd || rhs.useDoubleAtomicAdd;
      use8BitStorage     = use8BitStorage     || rhs.use8BitStorage;

      return *this;
    }

    bool useByteType       = false;
    bool useShortType      = false;
    bool useInt64Type      = false;
    bool useFloat64Type    = false;
    bool useHalfType       = false;

    bool useFloatAtomicAdd  = false;
    bool useDoubleAtomicAdd = false;
    bool use8BitStorage     = false;
  };

  /**
  \brief These features (within previous) should be enabled for Device in Vulkan
  */
  struct OptionalDeviceFeatures
  {
    OptionalDeviceFeatures operator||(const OptionalDeviceFeatures& rhs)
    {
      useRTX    = useRTX    || rhs.useRTX;
      useVarPtr = useVarPtr || rhs.useVarPtr;
      return *this;
    }

    bool useRTX    = false;
    bool useVarPtr = false;
  };

  struct ArrayData
  {
    uint32_t    arraySize  = 0;
    std::string arrayName;
    std::string elemType;
  };

  /**
  \brief Both for common functions and member functions called from kernels
  */
  struct FuncData
  {
    const clang::FunctionDecl* astNode;
    std::string        name;
    clang::SourceRange srcRange;
    uint64_t           srcHash;
    bool               isMember = false;
    bool               isKernel = false;
    bool               isVirtual = false;
    int                depthUse = 0;    ///!< depth Of Usage; 0 -- for kernels; 1 -- for functions called from kernel; 2 -- for functions called from functions called from kernels
                                        ///!< please note that if function is called under different depth, maximum depth should be stored in this variable;
    bool hasPrefix = false;
    std::string prefixName;
    std::unordered_set<std::string> calledMembers;
  
    std::string thisTypeName;                                ///!< currently filled for VFH only, TODO: fill for other
    std::string declRewritten;                               ///!< currently filled for VFH only, TODO: fill for other
    std::vector< std::pair<std::string, std::string> > args; ///!< currently filled for VFH only, TODO: fill for other
    
    std::string retTypeName;
    const clang::CXXRecordDecl* retTypeDecl;
  };

  struct RewrittenFunction
  {
    std::string funText() const { return funDecl + funBody; }
    std::string funDecl;
    std::string funBody;
  };

  struct IntersectionShader2
  {
    std::string shaderName; ///<! intersection check    (main intersection shader)
    std::string finishName; ///<! intersection complete (when intersection is found, complete with normal computing and e.t.c)
    std::string triTagName;
    std::string bufferName;
    std::string accObjName;
  };

  struct TemplatedFunctionLM ///<! for templated functions from LiteMath which should be generated per kernel
  {
    std::string name;
    std::string nameOriginal;
    std::string types[4]; ///<! max 4 templated arguments currently
  };

  /**
  \brief for each method MainClass::kernel_XXX
  */
  struct KernelInfo
  {

    struct ArgInfo
    {
      std::string type;
      std::string name;
      int         size;
      int         sizeOf = 0;
      DATA_KIND   kind = DATA_KIND::KIND_UNKNOWN;

      bool needFakeOffset = false;
      bool isThreadID     = false; ///<! used by RTV-like patterns where loop is defined out of kernel
      bool isLoopSize     = false; ///<! used by IPV-like patterns where loop is defined inside kernel

      bool isThreadFlags  = false;
      bool isReference    = false;
      bool isContainer    = false;
      bool isConstant     = false;

      std::string containerType;
      std::string containerDataType;

      bool IsTexture() const { return (kind == DATA_KIND::KIND_TEXTURE) || (isContainer && IsTextureContainer(containerType)); }
      bool IsPointer() const { return (kind == DATA_KIND::KIND_POINTER); }
      bool IsUser()    const { return !isThreadID && !isLoopSize && !needFakeOffset && !IsPointer() && !IsTexture() && !isContainer; }
    };

    enum class IPV_LOOP_KIND  { LOOP_KIND_LESS = 0, LOOP_KIND_LESS_EQUAL = 1};

    struct LoopIter
    {
      std::string type;
      std::string name;
      std::string startText;
      std::string sizeText;
      std::string strideText;

      std::string condTextOriginal;
      std::string iterTextOriginal;

      const clang::Stmt* startNode   = nullptr;
      const clang::Expr* sizeNode    = nullptr;
      const clang::Expr* strideNode  = nullptr;
      const clang::Stmt* bodyNode    = nullptr;

      uint32_t    loopNesting = 0;       ///<!
      uint32_t    id;                    ///<! used to preserve or change loops order
      IPV_LOOP_KIND condKind = IPV_LOOP_KIND::LOOP_KIND_LESS;
    };

    enum class REDUCTION_TYPE {ADD_ONE = 1, ADD = 2, MUL = 3, FUNC = 4, SUB = 5, SUB_ONE,  UNKNOWN = 255};

    struct ReductionAccess
    {
      REDUCTION_TYPE type;
      std::string    rightExpr;
      std::string    leftExpr; // altered left expression (arrays and other ... )
      bool           leftIsArray  = false;
      uint32_t       arraySize    = 0;
      std::string    arrayIndex;
      std::string    arrayName;
      std::vector<std::string> arrayTmpBufferNames;

      std::string    funcName;
      std::string    dataType   = "UnknownType";
      std::string    tmpVarName = "UnknownReductionOutput";
      std::string    GetInitialValue(bool isGLSL, const std::string& a_dataType)  const;
      std::string    GetOp(std::shared_ptr<IShaderCompiler> pShaderCC) const;
      std::string    GetOp2(std::shared_ptr<IShaderCompiler> pShaderCC) const;

      bool           SupportAtomicLastStep()        const;
      size_t         GetSizeOfDataType()            const;
    };

    std::string           return_type;          ///<! func. return type
    std::string           return_class;         ///<! class name of pointer if pointer is returned
    std::string           name;                 ///<! func. name
    std::string           className;            ///<! Class::kernel_XXX --> 'Class'
    std::string           interfaceName;        ///<! Name of the interface if the kernel is virtual
    std::vector<ArgInfo>  args;                 ///<! all arguments of a kernel
    std::vector<LoopIter> loopIters;            ///<! info about internal loops inside kernel which should be eliminated (so these loops are transformed to kernel call); For IPV pattern.
    std::string           debugOriginalText;

    uint32_t GetDim() const
    {
      if(loopIters.size() != 0)
        return uint32_t(loopIters.size());

      uint32_t size = 0;
      for(auto arg : args) {
        if(arg.isThreadID)
          size++;
      }

      if(size == 0)
      {
        std::cout << "  [ReductionAccess::GetDim]: Error, zero kernel dim" << std::endl;
        size = 1;
      }

      return size;
    }

    int loopInsidesOrder        = 100;
    int loopOutsidesInitOrder   = 100;

    clang::SourceRange    loopInsides;          ///<! used by IPV pattern to extract loops insides and make them kernel source
    clang::SourceRange    loopOutsidesInit;     ///<! used by IPV pattern to extract code before loops and then make additional initialization kernel
    clang::SourceRange    loopOutsidesFinish;   ///<! used by IPV pattern to extract code after  loops and then make additional finalization kernel
    bool                  hasInitPass   = false;///<! used by IPV pattern (currently); indicate that we need insert additional single-threaded run before current kernel (for reduction init or indirect dispatch buffer init)
    bool                  hasFinishPass = false;///<! used by IPV pattern (currently); indicate that we need insert additional passes              after  current kernel
    bool                  hasFinishPassSelf = false; ///<! if we need to do some-thing after loop and after generated loop finish pass
    bool                  hasIntersectionShader2 = false;
    IntersectionShader2   intersectionShader2Info;

    const clang::CXXMethodDecl* astNode = nullptr;
    bool usedInMainFunc = false;                ///<! wherther kernel is actually used or just declared
    bool isBoolTyped    = false;                ///<! used by RTV pattern; special case: if kernel return boolean, we analyze loop exit (break) or function exit (return) expression
    bool usedInExitExpr = false;                ///<! used by RTV pattern; if kernel is used in Control Function in if(kernelXXX()) --> break or return extression
    bool checkThreadFlags = false;              ///<! used by RTV pattern; if Kernel.shouldCheckExitFlag --> insert check flags code in kernel
    bool isMega           = false;

    std::string RetType;                         ///<! kernel return type
    std::string DeclCmd;                         ///<! used during class header to print declaration of current 'XXXCmd' for current 'kernel_XXX'
    std::map<std::string, UsedContainerInfo>     usedContainers;      ///<! list of all std::vector<T> member names which is referenced inside kernel
    std::unordered_set<std::string>                        usedMembers;         ///<! list of all other variables used inside kernel
    std::unordered_map<uint64_t, FuncData>                 usedMemberFunctions; ///<! list of all used member functions from this kernel

    std::unordered_map<std::string, ReductionAccess>   subjectedToReduction; ///<! if member is used in reduction expression
    std::unordered_map<std::string, TEX_ACCESS>        texAccessInArgs;
    std::unordered_map<std::string, TEX_ACCESS>        texAccessInMemb;
    std::unordered_map<std::string, std::string>       texAccessSampler;
    std::vector<ShittyFunction>                        shittyFunctions;     ///<! functions with input pointers accesed global memory, they should be rewritten for GLSL
    std::vector<const KernelInfo*>                     subkernels;          ///<! for RTV pattern only, when joing everything to mega-kernel this array store pointers to used kernels
    ShittyFunction                                     currentShit;         ///<!
    std::unordered_map<std::string, ArrayData>         threadLocalArrays;
    std::map<std::string, TemplatedFunctionLM>         templatedFunctionsLM;
   
    struct BEBlock
    {
      bool                  isParallel = false;
      const clang::ForStmt* forLoop    = nullptr;
      const clang::Stmt*    astNode    = nullptr;
    };

    struct BlockExpansionInfo
    {
      std::vector<const clang::DeclStmt*> sharedDecls;
      std::vector<BEBlock>                statements;
      bool enabled = false;
      std::string wgNames[3] = {"unknown", "unknown", "unknown"};
      std::string wgTypes[3] = {"uint", "uint", "uint"};
    };

    BlockExpansionInfo be;

    std::string rewrittenText;                   ///<! rewritten source code of a kernel
    std::string rewrittenInit;                   ///<! rewritten loop initialization code for kernel
    std::string rewrittenFinish;                 ///<! rewritten loop finish         code for kernel

    uint32_t wgSize[3] = {256,1,1};              ///<! workgroup size for the case when setting wgsize with spec constant is not allowed
    uint32_t warpSize  = 32;                     ///<! warp size in which we can rely on to omit sync in reduction and e.t.c.
    bool     enableSubGroups  = false;           ///<! enable subgroup operations for reduction and e.t.c.
    bool     enableRTPipeline = false;

    bool     singleThreadISPC = false;
    bool     openMpAndISPC    = false;
    bool     explicitIdISPC   = false;
    bool     useBlockOperations = false;         ///<! kernel uses ReduceAdd pattern for vector, all threads must be valid (don't add if(runThisThread) {... })

    bool      isIndirect = false;                ///<! IPV pattern; if loop size is defined by class member variable or vector size, we interpret it as indirect dispatching
    uint32_t  indirectBlockOffset = 0;           ///<! IPV pattern; for such kernels we have to know some offset in indirect buffer for thread blocks number (use int4 data for each kernel)
    uint32_t  indirectMakerOffset = 0;           ///<! RTV pattern; kernel-makers have to update size in the indirect buffer

    ShaderFeatures shaderFeatures;               ///<! Shader features which are required by this kernel
  };

  /**
  \brief Arguments of Kernels (And CF?) which are finilazed for the last stage of templated text rendering
  */
  struct ArgFinal
  {
    std::string type;
    std::string name;

    KernelInfo::LoopIter loopIter;     ///<! used if thiscariable is a loopIter
    std::string imageType;
    std::string imageFormat;
    bool        isDefinedInClass = false;
    bool        isThreadFlags    = false;
    bool        isImage          = false;
    bool        isSampler        = false;
    bool        isPointer        = false;
    bool        isConstant       = false;
  };

  /**
  \brief for each data member of MainClass
  */
  struct DataMemberInfo
  {
    std::string name;
    std::string type;
    DATA_KIND   kind = DATA_KIND::KIND_UNKNOWN;

    size_t      sizeInBytes          = 0; ///<! may be not needed due to using sizeof in generated code, but it is useful for sorting members by size and making apropriate aligment
    size_t      alignedSizeInBytes   = 0; ///<! aligned size will be known when data will be placed to a buffer
    size_t      offsetInTargetBuffer = 0; ///<! offset in bytes in terget buffer that stores all data members
    size_t      aligmentGLSL         = sizeof(int); ///<! aligment which GLSL compiler does anyway (we can't control that)

    bool isContainerInfo   = false; ///<! auto generated std::vector<T>::size() or capacity() or some analogue
    bool isContainer       = false; ///<! if std::vector<...>
    bool isArray           = false; ///<! if is array, element type stored in containerDataType;
    bool usedInKernel      = false; ///<! if any kernel use the member --> true; if no one uses --> false;
    bool usedInMainFn      = false; ///<! if std::vector is used in MainFunction like vector.data();
    bool isPointer         = false;
    bool isConst           = false; ///<! const float4 BACKGROUND_COLOR = ... (they should not be read back)
    bool isSingle          = false; ///<! single struct inside buffer, not a vector (vector with size() == 1), special case for all_references and other service needs
    bool bindWithRef       = false; ///<! if bound with buffer reference

    bool hasPrefix = false;
    bool hasIntersectionShader = false; ///<! indicate that this acceleration structure has user-defined intersection procedure
    std::string prefixName;

    DATA_USAGE usage = DATA_USAGE::USAGE_USER;         ///<! if this is service and 'implicit' data which was agged by generator, not by user;
    TEX_ACCESS tmask = TEX_ACCESS::TEX_ACCESS_NOTHING; ///<! store texture access flags if this data member is a texture

    size_t      arraySize = 0;                         ///<! 'N' if data is declared as 'array[N]';
    std::string containerType;                         ///<! std::vector usually
    std::string containerDataType;                     ///<! data type 'T' inside of std::vector<T>
    std::string intersectionClassName;                 ///<! used in the case of user intersection

    clang::TypeDecl* pTypeDeclIfRecord = nullptr;
    clang::TypeDecl* pContainerDataTypeDeclIfRecord = nullptr;

    bool IsUsedTexture() const { return isContainer && IsTextureContainer(containerType); }  // && isContainer && kslicer::IsTexture(containerType); }
  };

  struct DataMemberInfo_ByAligment
  {
    inline bool operator() (const kslicer::DataMemberInfo& struct1, const kslicer::DataMemberInfo& struct2)
    {
      if(struct1.aligmentGLSL != struct2.aligmentGLSL)
        return (struct1.aligmentGLSL > struct2.aligmentGLSL);
      else if(struct1.sizeInBytes != struct2.sizeInBytes)
        return (struct1.sizeInBytes > struct2.sizeInBytes);
      else if(struct1.isContainerInfo && !struct2.isContainerInfo)
        return false;
      else if(!struct1.isContainerInfo && struct2.isContainerInfo)
        return true;
      else
        return struct1.name < struct2.name;
    }
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
    bool        isConst   = false;
    bool        isContainer = false;

    size_t      arraySize = 0;
    std::string typeOfArrayElement;
    size_t      sizeInBytesOfArrayElement = 0;

    std::string containerType;
    std::string containerDataType;
  };

  /**
  \brief for arguments of ControlFunc which are pointers
  */
  struct InOutVarInfo
  {
    std::string name = "";
    std::string type = "";
    DATA_KIND   kind = DATA_KIND::KIND_UNKNOWN;
    bool isConst     = false;
    bool isRef       = false;
    bool isThreadId  = false;
    bool isTexture() const { return (kind == DATA_KIND::KIND_TEXTURE); };
    bool isPointer() const { return (kind == DATA_KIND::KIND_POINTER); };

    const clang::ParmVarDecl* paramNode = nullptr;
    std::vector<std::string> sizeUserAttr;
    std::string containerType;
    std::string containerDataType;
  };

  InOutVarInfo GetParamInfo(const clang::ParmVarDecl* currParam, const clang::CompilerInstance& compiler);
  std::unordered_set<std::string> ListPredefinedMathTypes();

  // assume there could be only 4 form of kernel arg when kernel is called
  //
  enum class KERN_CALL_ARG_TYPE{
    ARG_REFERENCE_LOCAL         = 0, // Passing the address of a local variable or local array by pointer (for example "& rayPosAndNear" or just randsArray);
    ARG_REFERENCE_ARG           = 1, // Passing the pointer (or texture) that was supplied to the argument of MainFunc (for example, just "out_color")
    ARG_REFERENCE_CLASS_VECTOR  = 2, // Passing a pointer to a class member of type std::vector<T>::data() (for example m_materials.data())
    ARG_REFERENCE_CLASS_POD     = 3, // Passing a pointer to a member of the class of type plain old data. For example, "&m_worldViewProjInv"
    ARG_REFERENCE_THREAD_ID     = 4, // Passing tidX, tidY or tidZ
    ARG_REFERENCE_CONST_OR_LITERAL = 5, // Passing const variables or literals (numbers)
    ARG_REFERENCE_SERVICE_DATA  = 6,    // Passing reference to some temp buffer whis should be generated inside class
    ARG_REFERENCE_UNKNOWN_TYPE  = 9     // Unknown type of arument yet. Generaly means we need to furthe process it, for example find among class variables or local variables
    };

  struct ArgReferenceOnCall
  {
    KERN_CALL_ARG_TYPE argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE;
    std::string        name = "";
    std::string        type = "";
    DATA_KIND          kind = DATA_KIND::KIND_UNKNOWN;
    const clang::Expr* node = nullptr;

    bool isConst            = false;
    bool isExcludedRTV      = false;

    bool isTexture    () const { return (kind == DATA_KIND::KIND_TEXTURE); }
    bool isAccelStruct() const { return (kind == DATA_KIND::KIND_ACCEL_STRUCT); }
  };

  struct KernelCallInfo
  {
    std::string                     kernelName;
    std::string                     originKernelName;
    std::string                     callerName;
    std::vector<ArgReferenceOnCall> descriptorSetsInfo;
    bool isService = false; ///<! indicate that this call is added by the slicer itself. It is not user kernel.
    bool isMega = false;
  };

  struct CFNameInfo
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
    std::unordered_map<std::string, DataLocalVarInfo> LocalConst;
    std::vector<InOutVarInfo>                         InOuts;
    std::unordered_set<std::string>                   ExcludeList;
    std::unordered_set<std::string>                   UsedKernels;

    std::unordered_map<std::string, UsedContainerInfo> usedContainers;      ///<! list of all std::vector<T> member names which is referenced inside ControlFunc
    std::unordered_set<std::string>                    usedMembers;         ///<! list of all other variables used inside ControlFunc
    std::unordered_map<uint64_t,          FuncData>    usedMemberFunctions; ///<! list of all used member functions from this kernel

    std::string ReturnType;
    std::string GeneratedDecl;
    std::string CodeGenerated;
    std::string OriginalDecl;
    std::string MegaKernelCall;

    size_t startDSNumber = 0;
    size_t endDSNumber   = 0;
    size_t startTSNumber = 0;
    size_t endTSNumber   = 0;

    // RT template specific
    //
    std::unordered_map<uint64_t, KernelStatementInfo> ExitExprIfCond;
    std::unordered_map<uint64_t, KernelStatementInfo> ExitExprIfCall;
    std::unordered_map<uint64_t, KernelStatementInfo> CallsInsideFor;

    bool   needToAddThreadFlags = false;
    bool   usePersistentThreads = false;
    KernelInfo                     megakernel;     ///<! for RTV pattern only, when joing everything to mega-kernel
    std::vector<const KernelInfo*> subkernels;     ///<! for RTV pattern only, when joing everything to mega-kernel this array store pointers to used kernels
    std::vector<KernelInfo>        subkernelsData; ///<! for RTV pattern only

    std::map<std::string, DataLocalVarInfo> localContainers; ///<! currently for IPV only
  };

  enum class DECL_IN_CLASS{ DECL_STRUCT, DECL_TYPEDEF, DECL_CONSTANT, DECL_UNKNOWN};

  /**
  \brief declaration of types and constant inside main class
  */
  struct DeclInClass
  {
    std::string        name;
    std::string        type;
    clang::SourceRange srcRange;
    uint64_t           srcHash;
    clang::TypeDecl*   astNode = nullptr;
    clang::VarDecl*    varNode = nullptr;
    uint32_t           order = 0; ///<! to sort them before put in generated kernels source code
    DECL_IN_CLASS      kind  = DECL_IN_CLASS::DECL_UNKNOWN;
    bool               extracted = false;
    bool               isArray   = false;
    bool               inClass   = false;
    uint32_t           arraySize = 0;
    int64_t            constVal  = 0;
    std::string        lostValue;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////  FunctionRewriter  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  class MainClassInfo;
  void MarkRewrittenRecursive(const clang::Stmt* currNode, std::unordered_set<uint64_t>& a_rewrittenNodes);
  void MarkRewrittenRecursive(const clang::Decl* currNode, std::unordered_set<uint64_t>& a_rewrittenNodes);
  bool IsVectorContructorNeedsReplacement(const std::string& a_typeName);

  /**
  \brief process local functions (data["LocalFunctions"]), float3 --> make_float3, std::max --> fmax and e.t.c.
  */
  class FunctionRewriter : public clang::RecursiveASTVisitor<FunctionRewriter> //
  {
  public:

    FunctionRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo) :
                     m_rewriter(R), m_compiler(a_compiler), m_codeInfo(a_codeInfo)
    {
      m_pRewrittenNodes = std::make_shared< std::unordered_set<uint64_t> >();
    }

    virtual ~FunctionRewriter(){}

    bool VisitCallExpr(clang::CallExpr* f)                    { return VisitCallExpr_Impl(f); }
    bool VisitCXXConstructExpr(clang::CXXConstructExpr* call) { return VisitCXXConstructExpr_Impl(call); }

    bool VisitFunctionDecl(clang::FunctionDecl* fDecl)       { return VisitFunctionDecl_Impl(fDecl); }
    bool VisitCXXMethodDecl(clang::CXXMethodDecl* fDecl)     { return VisitCXXMethodDecl_Impl(fDecl); }

    bool VisitVarDecl(clang::VarDecl* decl)                  { return VisitVarDecl_Impl(decl);        }
    bool VisitDeclStmt(clang::DeclStmt* decl)                { return VisitDeclStmt_Impl(decl);       } // for multiple vars in line like int i,j,k=2;

    bool VisitMemberExpr(clang::MemberExpr* expr)            { return VisitMemberExpr_Impl(expr);     }
    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* f) { return VisitCXXMemberCallExpr_Impl(f); }
    bool VisitFieldDecl(clang::FieldDecl* decl)              { return VisitFieldDecl_Impl(decl);      }
    bool VisitUnaryOperator(clang::UnaryOperator* op)        { return VisitUnaryOperator_Impl(op);    }
    bool VisitCStyleCastExpr(clang::CStyleCastExpr* cast)    { return VisitCStyleCastExpr_Impl(cast); }
    bool VisitImplicitCastExpr(clang::ImplicitCastExpr* cast){ return VisitImplicitCastExpr_Impl(cast); }

    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr* arrayExpr)            { return VisitArraySubscriptExpr_Impl(arrayExpr); }
    bool VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr* szOfExpr) { return VisitUnaryExprOrTypeTraitExpr_Impl(szOfExpr); }
    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* expr)               { return VisitCXXOperatorCallExpr_Impl(expr); }

    bool VisitCompoundAssignOperator(clang::CompoundAssignOperator* expr) { return VisitCompoundAssignOperator_Impl(expr); }
    bool VisitBinaryOperator(clang::BinaryOperator* expr)                 { return VisitBinaryOperator_Impl(expr); }
    bool VisitDeclRefExpr(clang::DeclRefExpr* expr)                       { return VisitDeclRefExpr_Impl(expr); }
    bool VisitFloatingLiteral(clang::FloatingLiteral* expr)               
    { 
      return VisitFloatingLiteral_Impl(expr);
    } 

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual std::string RecursiveRewrite(const clang::Stmt* expr);
    virtual std::string RewriteFuncDecl(clang::FunctionDecl* fDecl);

    virtual kslicer::RewrittenFunction RewriteFunction(clang::FunctionDecl* fDecl);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual std::string RewriteStdVectorTypeStr(const std::string& a_str) const;
    virtual std::string RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const { return RewriteStdVectorTypeStr(a_typeName); }
    virtual std::string RewriteImageType(const std::string& a_containerType, const std::string& a_containerDataType, TEX_ACCESS a_accessType, std::string& outImageFormat) const { return "readonly image2D"; }

    mutable ShaderFeatures sFeatures;
    virtual ShaderFeatures GetShaderFeatures() const { return sFeatures; }
    std::shared_ptr< std::unordered_set<uint64_t> > m_pRewrittenNodes = nullptr;

    virtual void SetCurrFuncInfo  (kslicer::FuncData* a_pInfo) { m_pCurrFuncInfo = a_pInfo; }
    virtual void ResetCurrFuncInfo()                           { m_pCurrFuncInfo = nullptr; }
    
    virtual bool NeedsVectorTypeRewrite(const std::string& a_str) { return false; }

    std::unordered_map<uint64_t, std::string>& WorkAroundRef() { return m_workAround; }
  protected:
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    clang::Rewriter&               m_rewriter;
    const clang::CompilerInstance& m_compiler;
    MainClassInfo*                 m_codeInfo;
    kslicer::FuncData*             m_pCurrFuncInfo = nullptr;
    std::unordered_map<uint64_t, std::string> m_workAround;
    ///////////////////////////////////////////////////////////////////////////////////////////////////
  
    void ReplaceTextOrWorkAround(clang::SourceRange a_range, const std::string& a_text);

    void MarkRewritten(const clang::Stmt* expr);
    bool WasNotRewrittenYet(const clang::Stmt* expr);

    std::string FunctionCallRewrite(const clang::CallExpr* call);
    std::string FunctionCallRewrite(const clang::CXXConstructExpr* call);
    std::string FunctionCallRewriteNoName(const clang::CXXConstructExpr* call);

    virtual std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText);
    virtual std::string RewriteConstructCall(clang::CXXConstructExpr* call);

    virtual void CkeckAndProcessForThreadLocalVarDecl(clang::VarDecl* decl);

  public:
    virtual bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl);
    virtual bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)     { return true; } // override this in Derived class

    virtual bool VisitVarDecl_Impl(clang::VarDecl* decl)                  { return true; } // override this in Derived class
    virtual bool VisitDeclStmt_Impl(clang::DeclStmt* decl)                { return true; } // override this in Derived class

    virtual bool VisitMemberExpr_Impl(clang::MemberExpr* expr)            { return true; } // override this in Derived class
    virtual bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f) { return true; } // override this in Derived class 
    virtual bool VisitFieldDecl_Impl(clang::FieldDecl* decl)              { return true; } // override this in Derived class
    virtual bool VisitUnaryOperator_Impl(clang::UnaryOperator* op)        { return true; } // override this in Derived class
    virtual bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)    { return true; } // override this in Derived class
    virtual bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast){ return true; } // override this in Derived class
    virtual bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call);                // override this in Derived class
    virtual bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr);

    virtual bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) { return true; }
    virtual bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) { return true; }
    virtual bool VisitCallExpr_Impl(clang::CallExpr* f);

    virtual bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) { return true; }
    virtual bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr) { return true; }
    virtual bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) { return true; }
    virtual bool VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr) { return true; }
    virtual bool VisitReturnStmt_Impl(clang::ReturnStmt* ret) { return true; }
    
    kslicer::ShittyFunction m_shit;
  };

  class FunctionRewriter2 : public FunctionRewriter ///!< BASE CLASS FOR ALL NEW BACKENDS
  {
  public:
    FunctionRewriter2(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo) : FunctionRewriter(R,a_compiler,a_codeInfo)  {}
    ~FunctionRewriter2(){}

    bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)   override;
    bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl) override;

    bool VisitVarDecl_Impl(clang::VarDecl* decl)                  override;
    bool VisitDeclStmt_Impl(clang::DeclStmt* decl)                override;

    bool VisitMemberExpr_Impl(clang::MemberExpr* expr)             override;
    bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  override; 
    bool VisitFieldDecl_Impl(clang::FieldDecl* decl)               override;
    bool VisitUnaryOperator_Impl(clang::UnaryOperator* op)         override;
    bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)     override;
    bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
    bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) override; 
    bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;

    bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)            override;
    bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;
    bool VisitCallExpr_Impl(clang::CallExpr* f)                                        override;

    bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) override;
    bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr)                 override;
    bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)                       override;
    bool VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr)               override;

    bool VisitReturnStmt_Impl(clang::ReturnStmt* ret) override;

    // Also important functions to use(!)
    //
    std::string RecursiveRewrite(const clang::Stmt* expr) override;
    
    virtual bool DetectAndRewriteShallowPattern(const clang::Stmt* expr, std::string& a_out);
    
    // DetectAndRewriteExpression --> DARExpr
    //
    virtual bool NeedToRewriteReductionOp(const std::string& op, const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr, std::string& outStr);
    virtual void DARExpr_ReductionFunc(const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr);
    
    virtual void DARExpr_RWTexture(clang::CXXOperatorCallExpr* expr, bool write);
    virtual void DARExpr_TextureAccess(clang::CXXOperatorCallExpr* expr);
    virtual void DARExpr_TextureAccess(clang::BinaryOperator* expr);
    virtual void DARExpr_TextureAccess(clang::CXXMemberCallExpr* call);

    std::unordered_set<uint64_t> m_visitedTexAccessNodes;

    // for kernel processing only
    //
    void InitKernelData(kslicer::KernelInfo& a_kernelRef, const std::string& a_fakeOffsetExp);
    bool                            m_kernelMode = false; ///!< if proccesed function is kernel or nor
    kslicer::KernelInfo*            m_pCurrKernel = nullptr;
    std::unordered_set<std::string> m_kernelUserArgs;
    std::string                     m_fakeOffsetExp;
    std::unordered_map<std::string, kslicer::DataMemberInfo> m_variables;
    bool                            processFuncMember = false; ///<! when process function members in the same way as kernels

    // general rewrite functions, same for all new backends
    //
    virtual bool NeedToRewriteMemberExpr(const clang::MemberExpr* expr, std::string& out_text);
    virtual bool NeedToRewriteDeclRefExpr(const clang::DeclRefExpr* expr, std::string& out_text);
    
    virtual bool CheckIfExprHasArgumentThatNeedFakeOffset(const std::string& exprStr);
    virtual bool NameNeedsFakeOffset(const std::string& a_name) const;
    virtual std::string CompleteFunctionCallRewrite(clang::CallExpr* call);
    virtual std::string KGenArgsName() const { return "kgenArgs."; }

    RewrittenFunction RewriteFunction(clang::FunctionDecl* fDecl);
    std::string       RewriteFuncDecl(clang::FunctionDecl* fDecl);

    virtual bool IsISPC() const { return false; }
  };
  
  struct IRecursiveRewriteOverride
  {
    virtual std::string RecursiveRewriteImpl(const clang::Stmt* expr) = 0;
    virtual kslicer::ShaderFeatures GetShaderFeatures() const { return kslicer::ShaderFeatures(); }
    virtual std::unordered_set<uint64_t> GetVisitedNodes() const = 0;
  };

  /**
  \brief process local functions
  */
  class GLSLFunctionRewriter : public FunctionRewriter //
  {
  public:
  
    GLSLFunctionRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit);
    ~GLSLFunctionRewriter(){}
  
    bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl) override;
    bool VisitCallExpr_Impl(clang::CallExpr* f)             override;
    bool VisitVarDecl_Impl(clang::VarDecl* decl)            override;
    bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast) override;
    bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
    bool VisitMemberExpr_Impl(clang::MemberExpr* expr)         override;
    bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr)   override;
    bool VisitDeclStmt_Impl(clang::DeclStmt* decl)             override;
    bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)  override;
    bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;
  
    bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f) override;
    bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;
  
    std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override;
    IRecursiveRewriteOverride* m_pKernelRewriter = nullptr;
  
    std::string RewriteStdVectorTypeStr(const std::string& a_str) const override;
    std::string RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const override;
    std::string RewriteImageType(const std::string& a_containerType, const std::string& a_containerDataType, kslicer::TEX_ACCESS a_accessType, std::string& outImageFormat) const override;
  
    std::unordered_map<std::string, std::string> m_vecReplacements;
    std::unordered_map<std::string, std::string> m_funReplacements;
    std::vector<std::pair<std::string, std::string> > m_vecReplacements2;

  
    std::string RewriteFuncDecl(clang::FunctionDecl* fDecl) override;
    std::string RecursiveRewrite(const clang::Stmt* expr) override;
    void        ApplyDefferedWorkArounds();
    
    struct BadRewqriteResult
    {
      std::string text;
      bool        isSingle;
      bool        isRewritten;
    };

    void        Get2DIndicesOfFloat4x4(const clang::CXXOperatorCallExpr* expr, const clang::Expr* out[3]);
  
    bool        NeedsVectorTypeRewrite(const std::string& a_str) override;
    std::string CompleteFunctionCallRewrite(clang::CallExpr* call);  
  };

  class SlangRewriter : public FunctionRewriter2 ///!< BASE CLASS FOR ALL NEW BACKENDS
  {
  public:
    SlangRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo) : FunctionRewriter2(R,a_compiler,a_codeInfo) { Init();}
    ~SlangRewriter(){ }

    bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)   override;
    bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl) override;

    bool VisitVarDecl_Impl(clang::VarDecl* decl)                  override;
    bool VisitDeclStmt_Impl(clang::DeclStmt* decl)                override;
    bool VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr)  override;

    bool VisitMemberExpr_Impl(clang::MemberExpr* expr)             override;
    bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  override; 
    bool VisitFieldDecl_Impl(clang::FieldDecl* decl)               override;
    bool VisitUnaryOperator_Impl(clang::UnaryOperator* op)         override;
    bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)     override;
    bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
    bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) override; 
    bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;

    bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)            override;
    bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;
    bool VisitCallExpr_Impl(clang::CallExpr* f)                                        override;

    bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) override;
    bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr)                 override;
    bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)                       override;

    // Also important functions to use(!)
    //
    bool        NeedsVectorTypeRewrite(const std::string& a_str) override;
    std::string RewriteStdVectorTypeStr(const std::string& a_str) const override;
    std::string RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const override;
    
    //
    //
    std::string RecursiveRewrite(const clang::Stmt* expr) override;
    std::string RewriteFuncDecl(clang::FunctionDecl* fDecl) override;
    //void MarkRewritten(const clang::Stmt* expr);
    //bool WasNotRewrittenYet(const clang::Stmt* expr);

    std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override;
  private:
    void Init();
    std::unordered_map<std::string, std::string> m_typesReplacement;
    std::unordered_map<std::string, std::string> m_funReplacements;
  };

  class CudaRewriter : public FunctionRewriter2 ///!< BASE CLASS FOR ALL NEW BACKENDS
  {
  public:
    CudaRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo) : FunctionRewriter2(R,a_compiler,a_codeInfo) { Init();}
    ~CudaRewriter(){ }

    bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)   override;
    bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl) override;

    bool VisitVarDecl_Impl(clang::VarDecl* decl)                  override;
    bool VisitDeclStmt_Impl(clang::DeclStmt* decl)                override;
    bool VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr)  override;

    bool VisitMemberExpr_Impl(clang::MemberExpr* expr)             override;
    bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  override; 
    bool VisitFieldDecl_Impl(clang::FieldDecl* decl)               override;
    bool VisitUnaryOperator_Impl(clang::UnaryOperator* op)         override;
    bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)     override;
    bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
    bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) override; 
    bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;

    bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)            override;
    bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;
    bool VisitCallExpr_Impl(clang::CallExpr* f)                                        override;

    bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) override;
    bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr)                 override;
    bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)                       override;

    // Also important functions to use(!)
    //
    bool        NeedsVectorTypeRewrite(const std::string& a_str) override;
    std::string RewriteStdVectorTypeStr(const std::string& a_str) const override;
    std::string RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const override;
    
    //
    //
    std::string RecursiveRewrite(const clang::Stmt* expr) override;
    std::string RewriteFuncDecl(clang::FunctionDecl* fDecl) override;

    std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override;
  private:
    void Init();
    std::unordered_map<std::string, std::string> m_typesReplacement;
    std::unordered_map<std::string, std::string> m_funReplacements;
  };

  class ISPCRewriter : public FunctionRewriter2 ///!< BASE CLASS FOR ALL NEW BACKENDS
  {
  public:
    ISPCRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo) : FunctionRewriter2(R,a_compiler,a_codeInfo) { Init();}
    ~ISPCRewriter(){ }

    bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)   override;
    bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl) override;

    bool VisitVarDecl_Impl(clang::VarDecl* decl)                  override;
    bool VisitDeclStmt_Impl(clang::DeclStmt* decl)                override;
    bool VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr)  override;

    bool VisitMemberExpr_Impl(clang::MemberExpr* expr)             override;
    bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  override; 
    bool VisitFieldDecl_Impl(clang::FieldDecl* decl)               override;
    bool VisitUnaryOperator_Impl(clang::UnaryOperator* op)         override;
    bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)     override;
    bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
    bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) override; 
    bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;

    bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)            override;
    bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;
    bool VisitCallExpr_Impl(clang::CallExpr* f)                                        override;

    bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) override;
    bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr)                 override;
    bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)                       override;

    // Also important functions to use(!)
    //
    bool        NeedsVectorTypeRewrite(const std::string& a_str) override;
    std::string RewriteStdVectorTypeStr(const std::string& a_str) const override;
    std::string RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const override;
    
    std::string RecursiveRewrite(const clang::Stmt* expr) override;
    std::string RewriteFuncDecl(clang::FunctionDecl* fDecl) override;

    std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override;
  private:
    void Init();
    std::unordered_map<std::string, std::string> m_typesReplacement;
    std::unordered_map<std::string, std::string> m_funReplacements;
  };

  std::unordered_map<std::string, std::string> ListSlangStandartTypeReplacements(bool a_NeedConstCopy = true);
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////  KernelRewriter  //////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  class KernelInfoVisitor : public clang::RecursiveASTVisitor<KernelInfoVisitor> // replace all expressions with class variables to kgen_data buffer access
  {
  public:
  
    KernelInfoVisitor(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, bool a_onlyShaderFeatures = false);
    virtual ~KernelInfoVisitor() {}
  
    bool VisitForStmt(clang::ForStmt* forLoop);
    bool VisitMemberExpr(clang::MemberExpr* expr);
    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* f);
    bool VisitReturnStmt(clang::ReturnStmt* ret);
    bool VisitUnaryOperator(clang::UnaryOperator* expr);
  
    bool VisitCompoundAssignOperator(clang::CompoundAssignOperator* expr);
    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* expr);
    bool VisitBinaryOperator(clang::BinaryOperator* expr);
    bool VisitCallExpr(clang::CallExpr* call);
    bool VisitVarDecl(clang::VarDecl* decl);
  
  protected:
  
    void DetectTextureAccess(clang::CXXMemberCallExpr* call);
    void DetectTextureAccess(clang::CXXOperatorCallExpr* expr);
    void DetectTextureAccess(clang::BinaryOperator* expr);
    void ProcessReadWriteTexture(clang::CXXOperatorCallExpr* expr, bool write);
  
    void ProcessReductionOp(const std::string& op, const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr);
    void DetectFuncReductionAccess(const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr);
    bool NameNeedsFakeOffset(const std::string& a_name) const;
  
    clang::Rewriter&               m_rewriter;
    const clang::CompilerInstance& m_compiler;
    kslicer::MainClassInfo*        m_codeInfo;
    kslicer::KernelInfo&           m_currKernel;
    bool                           m_onlyShaderFeatures;
  
    std::unordered_set<uint64_t> m_visitedTexAccessNodes;
  };
  
  void DisplayVisitedNodes(const std::unordered_set<uint64_t>& a_nodes);
  bool CheckSettersAccess(const clang::MemberExpr* expr, const MainClassInfo* a_codeInfo, const clang::CompilerInstance& a_compiler,
                          std::string* setterS, std::string* setterM);

  class KernelRewriter : public clang::RecursiveASTVisitor<KernelRewriter> // replace all expressions with class variables to kgen_data buffer access
  {
  public:

    KernelRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, const std::string& a_fakeOffsetExpr);
    virtual ~KernelRewriter() {}

    bool VisitVarDecl(clang::VarDecl* decl)                    { return VisitVarDecl_Impl(decl); }
    bool VisitMemberExpr(clang::MemberExpr* expr)              { if(WasRewritten(expr)) return true; else return VisitMemberExpr_Impl(expr); }
    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* f)   { if(WasRewritten(f))    return true; else return VisitCXXMemberCallExpr_Impl(f); }
    bool VisitCallExpr(clang::CallExpr* f)                     { if(WasRewritten(f))    return true; else return VisitCallExpr_Impl(f); }
    bool VisitCXXConstructExpr(clang::CXXConstructExpr* call)  { if(WasRewritten(call)) return true; else return VisitCXXConstructExpr_Impl(call); }
    bool VisitReturnStmt(clang::ReturnStmt* ret)               { if(WasRewritten(ret))  return true; else return VisitReturnStmt_Impl(ret); }
    bool VisitUnaryOperator(clang::UnaryOperator* expr)        { if(WasRewritten(expr)) return true; else return VisitUnaryOperator_Impl(expr);  }
    bool VisitBinaryOperator(clang::BinaryOperator* expr)      { if(WasRewritten(expr)) return true; else return VisitBinaryOperator_Impl(expr); }

    bool VisitCompoundAssignOperator(clang::CompoundAssignOperator* expr) { if(WasRewritten(expr)) return true; else return VisitCompoundAssignOperator_Impl(expr); }
    bool VisitCXXOperatorCallExpr   (clang::CXXOperatorCallExpr* expr)    { if(WasRewritten(expr)) return true; else return VisitCXXOperatorCallExpr_Impl(expr); }
    bool VisitCStyleCastExpr(clang::CStyleCastExpr* cast)                 { if(WasRewritten(cast)) return true; else return VisitCStyleCastExpr_Impl(cast); }
    bool VisitImplicitCastExpr(clang::ImplicitCastExpr* cast)             { if(WasRewritten(cast)) return true; else return VisitImplicitCastExpr_Impl(cast); }
    bool VisitDeclRefExpr(clang::DeclRefExpr* expr)                       { if(WasRewritten(expr)) return true; else return VisitDeclRefExpr_Impl(expr); }
    bool VisitFloatingLiteral(clang::FloatingLiteral* expr)               { if(WasRewritten(expr)) return true; else return VisitFloatingLiteral_Impl(expr); }
    bool VisitDeclStmt(clang::DeclStmt* stmt)                             { if(WasRewritten(stmt)) return true; else return VisitDeclStmt_Impl(stmt); }
    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr* arrayExpr)    { if(WasRewritten(arrayExpr))        return true; else return VisitArraySubscriptExpr_Impl(arrayExpr);  }
    bool VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr* szOfExpr) { if(WasRewritten(szOfExpr)) return true; else return VisitUnaryExprOrTypeTraitExpr_Impl(szOfExpr); }

    std::shared_ptr<std::unordered_set<uint64_t> > m_pRewrittenNodes = nullptr;
    virtual std::string RecursiveRewrite (const clang::Stmt* expr);
    virtual void ReplaceTextOrWorkAround(clang::SourceRange a_range, const std::string& a_text);
    virtual void ApplyDefferedWorkArounds();
    
    std::unordered_map<uint64_t, std::string>  m_workAround;

    virtual void ClearUserArgs() { }
    virtual ShaderFeatures GetKernelShaderFeatures() const { return ShaderFeatures(); }
    bool NameNeedsFakeOffset(const std::string& a_name) const;

    bool processFuncMember = false;
    virtual void SetCurrFuncInfo  (kslicer::FuncData* a_pInfo) { m_pCurrFuncInfo = a_pInfo; }
    virtual void ResetCurrFuncInfo()                           { m_pCurrFuncInfo = nullptr; }

    virtual void SetCurrKernelInfo  (kslicer::KernelInfo* a_pInfo) { m_pCurrKernelInfo = a_pInfo; }
    virtual void ResetCurrKernelInfo()                             { m_pCurrKernelInfo = nullptr; }
    
    const clang::CompilerInstance& GetCompiler() { return m_compiler; }

  protected:

    kslicer::FuncData* m_pCurrFuncInfo   = nullptr;
    KernelInfo*        m_pCurrKernelInfo = nullptr;

    bool CheckIfExprHasArgumentThatNeedFakeOffset(const std::string& exprStr);
    void ProcessReductionOp(const std::string& op, const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr);

    virtual bool IsGLSL() const { return false; }

    virtual std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText);
    void DetectTextureAccess(clang::CXXOperatorCallExpr* expr);
    void DetectTextureAccess(clang::CXXMemberCallExpr*   call);
    void DetectTextureAccess(clang::BinaryOperator* expr);
    void DetectFuncReductionAccess(const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr);
    void ProcessReadWriteTexture(clang::CXXOperatorCallExpr* expr, bool write);

    clang::Rewriter&                                         m_rewriter;
    const clang::CompilerInstance&                           m_compiler;
    MainClassInfo*                                           m_codeInfo;
    std::string                                              m_mainClassName;
    std::unordered_map<std::string, kslicer::DataMemberInfo> m_variables;
    const std::vector<kslicer::KernelInfo::ArgInfo>&         m_args;
    const std::string&                                       m_fakeOffsetExp;
    std::vector<std::string>                                 m_threadIdArgs;
    std::string                                              m_threadIdExplicitIndexISPC = "";
    bool                                                     m_kernelIsBoolTyped;
    kslicer::KernelInfo&                                     m_currKernel;
    bool                                                     m_explicitIdISPC = false;

    std::unordered_set<uint64_t>                             m_visitedTexAccessNodes;

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    std::string FunctionCallRewrite(const clang::CallExpr* call);
    std::string FunctionCallRewrite(const clang::CXXConstructExpr* call);
    std::string FunctionCallRewriteNoName(const clang::CXXConstructExpr* call);

    bool WasNotRewrittenYet(const clang::Stmt* expr);
    bool WasRewritten(const clang::Stmt* expr) { return !WasNotRewrittenYet(expr); }
    void MarkRewritten(const clang::Stmt* expr);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    virtual bool VisitMemberExpr_Impl(clang::MemberExpr* expr);
    virtual bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f);
    virtual bool VisitCallExpr_Impl(clang::CallExpr* f);
    virtual bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call);
    virtual bool VisitReturnStmt_Impl(clang::ReturnStmt* ret);

    // to detect reduction inside IPV programming template
    //
    virtual bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr);                   // ++, --, (*var) =  ...
    virtual bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr); // +=, *=, -=; to detect reduction
    virtual bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr);       // +=, *=, -=; to detect reduction for custom data types (float3/float4 for example)
    virtual bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr);                 // m_var = f(m_var, expr)

    virtual bool VisitVarDecl_Impl(clang::VarDecl* decl)                   { return true; } // override this in Derived class
    virtual bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)     { return true; } // override this in Derived class
    virtual bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) { return true; } // override this in Derived class
    virtual bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr);
    virtual bool VisitDeclStmt_Impl(clang::DeclStmt* decl)                 { return true; } // override this in Derived class
    virtual bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)  { return true; } // override this in Derived class
    virtual bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) { return true; }
    virtual bool VisitFloatingLiteral_Impl(clang::FloatingLiteral* f);
    virtual bool NeedToRewriteMemberExpr(const clang::MemberExpr* expr, std::string& out_text);

  };
  
  ///!< Base class for rewrite kernels in new back-ends, but in general should be implemented via 'FunctionRewriter2' (m_pFunRW2)
  ///!< So, don't override it unless you really need
  class KernelRewriter2 : public KernelRewriter 
  {
  public:

    KernelRewriter2(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                    MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, 
                    const std::string& a_fakeOffsetExpr, std::shared_ptr<FunctionRewriter2> a_pFunImpl)
                    : KernelRewriter(R,a_compiler,a_codeInfo,a_kernel,a_fakeOffsetExpr), m_pFunRW2(a_pFunImpl)
    
    {

    }

    virtual ~KernelRewriter2() {}

    bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr)                   override; // ++, --, (*var) =  ...
    bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) override; // +=, *=, -=; to detect reduction
    bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)       override; // +=, *=, -=; to detect reduction for custom data types (float3/float4 for example)
    bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr)                 override; // m_var = f(m_var, expr)

    bool VisitVarDecl_Impl(clang::VarDecl* decl)                               override; // override this in Derived class
    bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)                 override; // override this in Derived class
    bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast)             override; // override this in Derived class

    bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)                               override;
    bool VisitDeclStmt_Impl(clang::DeclStmt* decl)                                     override;
    bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)            override;
    bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;

    bool VisitMemberExpr_Impl(clang::MemberExpr* expr)      override;
    bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* call) override;
    bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) override;
    bool VisitCallExpr_Impl(clang::CallExpr* f)               override;
    bool VisitFloatingLiteral_Impl(clang::FloatingLiteral* f) override; 
   
    bool VisitReturnStmt_Impl(clang::ReturnStmt* ret) override;

    std::string RecursiveRewrite(const clang::Stmt* expr) override;

    void SetCurrFuncInfo(kslicer::FuncData* a_pInfo) override 
    { 
      m_pCurrFuncInfo = a_pInfo; 
      if(m_pFunRW2 != nullptr)
        m_pFunRW2->SetCurrFuncInfo(a_pInfo);
    }
    
    void ResetCurrFuncInfo() override 
    { 
      m_pCurrFuncInfo = nullptr; 
      if(m_pFunRW2 != nullptr)
        m_pFunRW2->ResetCurrFuncInfo();
    }

  protected:
    std::shared_ptr<FunctionRewriter2> m_pFunRW2 = nullptr;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::string CleanTypeName(const std::string& a_str);

  struct IShaderCompiler
  {
    IShaderCompiler(){}
    virtual ~IShaderCompiler(){}
    virtual std::string UBOAccess(const std::string& a_name) const = 0;
    virtual std::string ReplaceSizeCapacityExpr(const std::string& a_str) const;
    virtual std::string ProcessBufferType(const std::string& a_typeName) const { return a_typeName; };
  
    virtual bool        IsSingleShader()   const = 0;
    virtual std::string ShaderSingleFile() const = 0;
    virtual std::string ShaderFolder()     const = 0;
   
    virtual bool        MemberFunctionsAreSupported() const { return false; }
    virtual bool        BuffersAsPointersInShaders()  const { return false; }
    virtual bool        IsGLSL() const { return !IsSingleShader(); }
    virtual bool        IsISPC() const { return false; }
    virtual bool        IsCUDA() const { return false; }
    virtual bool        IsWGPU() const { return false; }

    virtual void        GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) = 0;

    virtual std::string LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const = 0;

    virtual std::string ReplaceCallFromStdNamespace(const std::string& a_call, const std::string& a_typeName) const 
    { 
      std::string call = a_call;
      ReplaceFirst(call, "std::", "");
      return call;
    }

    virtual std::string IndirectBufferDataType() const { return "uint4* "; }

    virtual bool UseSeparateUBOForArguments() const { return false; }
    virtual bool UseSpecConstForWgSize() const { return false; }
    virtual void GetThreadSizeNames(std::string a_strs[3]) const = 0;
    
    virtual bool        SupportAtomicGlobal(const KernelInfo::ReductionAccess& acc)             const { return acc.SupportAtomicLastStep(); } 
    virtual std::string GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const { return "unknownSubgroup"; }
    virtual std::string GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const { return "unknownAtomic";}

    virtual std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                        kslicer::ShittyFunction a_shit = kslicer::ShittyFunction()) = 0;
    virtual std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                        kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) = 0;

    virtual std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) = 0;
    virtual std::string Name() const { return "unknown shader compiler"; }

    virtual std::string RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const = 0;
    
    // for block expansion
    //
    virtual std::string RewriteBESharedDecl(const clang::DeclStmt* decl, std::shared_ptr<KernelRewriter> pRewriter);
    virtual std::string RewriteBEParallelFor(const clang::ForStmt* forExpr, std::shared_ptr<KernelRewriter> pRewriter);
    virtual std::string RewriteBEStmt(const clang::Stmt* stmt, std::shared_ptr<KernelRewriter> pRewriter);
    
    // for RTV only
    //
    virtual std::string RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds); 
  };

  struct ClspvCompiler : IShaderCompiler
  {
    ClspvCompiler(bool a_useCPP, const std::string& a_prefix);
    std::string UBOAccess(const std::string& a_name) const override { return std::string("ubo->") + a_name; };
    bool        IsSingleShader()   const override { return true; }
    std::string ShaderFolder()     const override { return "clspv_shaders_aux"; }
    std::string ShaderSingleFile() const override { return "z_generated.cl"; }
    bool        BuffersAsPointersInShaders() const override { return true; }

    void        GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;

    bool        UseSeparateUBOForArguments() const override { return m_useCpp; }
    bool        UseSpecConstForWgSize()      const override { return m_useCpp; }

    std::string LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3])                               const override;
    std::string ReplaceCallFromStdNamespace(const std::string& a_call, const std::string& a_typeName) const override;
    void        GetThreadSizeNames(std::string a_strs[3])                                             const override;
    std::string GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    std::string GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;

    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string Name() const override { return "OpenCL"; }

    std::string RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const override;

  protected:
    virtual std::string BuildCommand(const std::string& a_inputFile = "") const;
    bool m_useCpp;
    const std::string& m_suffix;
  };

  struct ISPCCompiler : ClspvCompiler
  {
    ISPCCompiler(bool a_useCPP, const std::string& a_prefix);
    std::string UBOAccess(const std::string& a_name) const override { return std::string("ubo[0].") + a_name; };
    std::string Name() const override { return "ISPC"; }
    void        GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;
    bool        IsISPC() const override { return true; }
    std::string BuildCommand(const std::string& a_inputFile) const override;
    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string ReplaceCallFromStdNamespace(const std::string& a_call, const std::string& a_typeName) const override;
    bool        BuffersAsPointersInShaders() const override { return false; }
    bool        SupportAtomicGlobal(const KernelInfo::ReductionAccess& acc) const override { return true; }

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;
  };

  struct GLSLCompiler : IShaderCompiler
  {
    GLSLCompiler(const std::string& a_prefix);
    std::string UBOAccess(const std::string& a_name) const override { return std::string("ubo.") + a_name; };
    std::string ProcessBufferType(const std::string& a_typeName) const override;
    
    bool        IsSingleShader()                     const override { return false;}
    bool        MemberFunctionsAreSupported()        const override { return true; }
    std::string ShaderFolder()                       const override { return std::string("shaders") + ToLowerCase(m_suffix); }
    std::string ShaderSingleFile()                   const override { return ""; }

    void GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;

    std::string LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const override;
    void        GetThreadSizeNames(std::string a_strs[3])               const override;
    std::string GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    std::string GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;

    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string Name() const override { return "GLSL"; }

    std::string RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const override;
    std::string RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds) override; 
    
    std::string IndirectBufferDataType() const override { return "uvec4 "; }

  private:
    const std::string& m_suffix;
    void ProcessVectorTypesString(std::string& a_str);
  };

  struct SlangCompiler : IShaderCompiler
  {
    SlangCompiler(const std::string& a_prefix, bool a_wgpuEnabled = false);
    std::string UBOAccess(const std::string& a_name) const override { return std::string("ubo[0].") + a_name; };
    std::string ProcessBufferType(const std::string& a_typeName) const override;

    bool        IsSingleShader()                     const override { return false; }
    bool        MemberFunctionsAreSupported()        const override { return true; }
    std::string ShaderFolder()                       const override { return std::string("shaders") + ToLowerCase(m_suffix); }
    std::string ShaderSingleFile()                   const override { return ""; }
    
    bool        IsGLSL() const override { return false; }
    bool        IsISPC() const override { return false; }
    bool        IsWGPU() const override { return m_wgpuEnabled; }

    void GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;

    std::string LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const override;
    void        GetThreadSizeNames(std::string a_strs[3])               const override;
    std::string GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    std::string GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;

    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string Name() const override { return "Slang"; }

    std::string RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const override;
    std::string RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds) override; 

    std::string IndirectBufferDataType() const override { return "uint4 "; }

  private:
    void ProcessVectorTypesString(std::string& a_str);
    const std::string& m_suffix;
    std::unordered_map<std::string, std::string> m_typesReplacement;
    bool m_wgpuEnabled;
  };

  struct CudaCompiler : IShaderCompiler
  {
    CudaCompiler(const std::string& a_prefix);
    std::string UBOAccess(const std::string& a_name) const override 
    {
      if(a_name.find(".size()") != std::string::npos) // kernelJson["IndirectSizeX"]  = a_classInfo.pShaderCC->UBOAccess(exprContent);
        return a_name;
      else
        return std::string("ubo.") + a_name; 
    } //  { return a_name; }
    std::string ReplaceSizeCapacityExpr(const std::string& a_str) const override { return a_str; }
    std::string ProcessBufferType(const std::string& a_typeName) const override;

    bool        IsSingleShader()                     const override { return true; }
    bool        MemberFunctionsAreSupported()        const override { return true; }
    std::string ShaderFolder()                       const override { return ""; }
    std::string ShaderSingleFile()                   const override { return ""; }
    
    bool        IsGLSL() const override { return false; }
    bool        IsISPC() const override { return false; }
    bool        IsCUDA() const override { return true;  }

    void GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings) override;

    std::string LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const override;
    void        GetThreadSizeNames(std::string a_strs[3])               const override;
    std::string GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    std::string GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const override;
    bool        SupportAtomicGlobal(const KernelInfo::ReductionAccess& acc) const override { return true; }

    std::shared_ptr<kslicer::FunctionRewriter> MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) override;
    std::shared_ptr<KernelRewriter>            MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                kslicer::KernelInfo& a_kernel, const std::string& fakeOffs) override;

    std::string PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter) override;
    std::string Name() const override { return "CUDA"; }

    std::string RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const override;
    std::string RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds) override; 

    std::string IndirectBufferDataType() const override { return "uint4 "; }

  private:
    const std::string& m_suffix;
    std::unordered_map<std::string, std::string> m_typesReplacement;
  };

  struct IHostCodeGen
  {
    IHostCodeGen(){}
    virtual ~IHostCodeGen(){}

    virtual std::string Name() const { return ""; } 
    virtual void GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) {}
    virtual void GenerateHostDevFeatures(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) {}
    virtual bool IsCUDA() const { return false; }
    virtual bool IsWGPU() const { return false; }
    virtual bool HasSpecConstants() const { return false; }
  };

  struct VulkanCodeGen : public IHostCodeGen
  {
    std::string Name() const override { return "Vulkan"; }
    void GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) override;
    void GenerateHostDevFeatures(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) override;
    bool HasSpecConstants() const override { return true; }
  };

  struct WGPUCodeGen : public IHostCodeGen
  {
    std::string Name() const override { return "WebGPU"; }
    bool IsWGPU()      const override { return true; }
    void GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) override;
    void GenerateHostDevFeatures(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) override;
  };

  struct CudaCodeGen : public IHostCodeGen
  {
    CudaCodeGen(const std::string& a_actualCUDAImpl) : m_actualCUDAImpl(a_actualCUDAImpl) {}
    std::string Name() const override { return m_actualCUDAImpl; }
    void GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) override;
    bool IsCUDA() const override { return true; }
    std::string m_actualCUDAImpl;
  };

  struct ISPCCodeGen : public IHostCodeGen
  {
    std::string Name() const override { return "ISPC"; }
    void GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings) override;
  };

  struct ServiceCall
  {
    std::string opName;
    std::string dataTypeName;
    std::string lambdaSource;
    std::string key() const { return opName + "_" + dataTypeName; }
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  class UsedCodeFilter;

  /**
  \brief collector of all information about input main class
  */
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

    std::unordered_map<std::string, KernelInfo>::iterator       FindKernelByName(const std::string& a_name);
    std::unordered_map<std::string, KernelInfo>::const_iterator FindKernelByName(const std::string& a_name) const;

    std::vector<std::string>                    indirectKernels; ///<! list of all kernel names which require indirect dispatch; The order is essential because it is used for indirect buffer offsets
    std::vector<DataMemberInfo>                 dataMembers;     ///<! only those member variables which are referenced from kernels
    std::vector<MainFuncInfo>                   mainFunc;        ///<! list of all control functions
  
    std::unordered_map<std::string, ArrayData>      m_threadLocalArrays;
    std::unordered_map<uint64_t, RewrittenFunction> m_functionsDone;

    std::string                                        mainClassName;         ///<! Current main class (derived)
    std::unordered_map<std::string, int>               mainClassNames;        ///<! All main classes (derived + base)
    std::unordered_set<std::string>                    composClassNames; 
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
    virtual void        ProcessKernelArg(KernelInfo::ArgInfo& arg, const KernelInfo& a_kernel) const { }  ///<!
    virtual bool        IsIndirect(const KernelInfo& a_kernel) const;
    virtual bool        IsRTV() const { return false; }

    //// Processing Control Functions (CF)
    //
    virtual MList         ListMatchers_CF(const std::string& mainFuncName) = 0;
    virtual MHandlerCFPtr MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler) = 0;
    virtual void          VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler);

    virtual void AddSpecVars_CF(std::vector<MainFuncInfo>& a_mainFuncList, std::unordered_map<std::string, KernelInfo>& a_kernelList) {}

    virtual void PlugSpecVarsInCalls_CF(const std::vector<MainFuncInfo>&                      a_mainFuncList,
                                        const std::unordered_map<std::string, KernelInfo>&    a_kernelList,
                                        std::vector<KernelCallInfo>&                          a_kernelCalls) {}

    virtual void ProcessVFH(const std::vector<const clang::CXXRecordDecl*>& a_decls, const clang::CompilerInstance& a_compiler);
    virtual void ExtractVFHConstants(const clang::CompilerInstance& compiler, clang::tooling::ClangTool& Tool);
    virtual void AppendAllRefsBufferIfNeeded(std::vector<DataMemberInfo>& a_vector);
    virtual void AppendAccelStructForIntersectionShadersIfNeeded(std::vector<DataMemberInfo>& a_vector, std::string composImplName);
    virtual void AppendAccelStructForIntersectionShadersIfNeeded(std::vector<DataMemberInfo>& a_vector, const IntersectionShader2& a_shader);

    //// \\

    //// Processing Kernel Functions (KF)
    //
    virtual MList         ListMatchers_KF(const std::string& mainFuncName) = 0;
    virtual MHandlerKFPtr MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compiler) = 0;
    virtual std::string   VisitAndRewrite_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler,
                                             std::string& a_outLoopInitCode, std::string& a_outLoopFinishCode);
    virtual void          VisitAndPrepare_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler); // additional informational pass, does not rewrite the code!

    virtual void ProcessCallArs_KF(const KernelCallInfo& a_call);

    //// These methods used for final template text rendering
    //
    virtual uint32_t GetKernelDim(const KernelInfo& a_kernel) const = 0;

    virtual std::vector<ArgFinal> GetKernelTIDArgs(const KernelInfo& a_kernel) const;
    virtual std::vector<ArgFinal> GetKernelCommonArgs(const KernelInfo& a_kernel) const;

    virtual void        GetCFSourceCodeCmd(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler, bool a_megakernelRTV);
    virtual std::string GetCFDeclFromSource(const std::string& sourceCode);

    virtual bool NeedThreadFlags() const { return false; }
    virtual bool NeedFakeOffset() const { return false; }
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

    bool halfFloatTextures = false;
    bool megakernelRTV     = false;
    bool persistentRTV     = false; // current implementation for persistent threads on done only for megakernels in RTV
    bool useComplexNumbers = false;
    bool genGPUAPI         = false;
    bool forceAllBufToRefs = false;
    bool placeVectorsInUBO = false;
    bool shitIsAlwaysConst = false;

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


  struct RTV_Pattern : public MainClassInfo
  {
    MList         ListMatchers_CF(const std::string& mainFuncName) override;
    MHandlerCFPtr MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler) override;

    void AddSpecVars_CF(std::vector<MainFuncInfo>& a_mainFuncList, std::unordered_map<std::string, KernelInfo>&  a_kernelList) override;

    void PlugSpecVarsInCalls_CF(const std::vector<MainFuncInfo>&                   a_mainFuncList,
                                const std::unordered_map<std::string, KernelInfo>& a_kernelList,
                                std::vector<KernelCallInfo>&                       a_kernelCalls) override;

    MList         ListMatchers_KF(const std::string& mainFuncName) override;
    MHandlerKFPtr MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compiler) override;
    void          ProcessCallArs_KF(const KernelCallInfo& a_call) override;

    uint32_t      GetKernelDim(const KernelInfo& a_kernel) const override;
    void          ProcessKernelArg(KernelInfo::ArgInfo& arg, const KernelInfo& a_kernel) const override;

    bool NeedThreadFlags() const override { return true; }
    bool NeedFakeOffset () const override { return true; }
    bool IsRTV          () const override { return true; }

  private:

  };

  struct IPV_Pattern : public MainClassInfo
  {
    MList         ListMatchers_CF(const std::string& mainFuncName) override;
    MHandlerCFPtr MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler) override;

    MList         ListMatchers_KF(const std::string& mainFuncName) override;
    MHandlerKFPtr MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compiler) override;
    std::string   VisitAndRewrite_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler,
                                     std::string& a_outLoopInitCode, std::string& a_outLoopFinishCode) override;

    uint32_t      GetKernelDim(const KernelInfo& a_kernel) const override;
    void          ProcessKernelArg(KernelInfo::ArgInfo& arg, const KernelInfo& a_kernel) const override;

    std::vector<ArgFinal> GetKernelTIDArgs(const KernelInfo& a_kernel) const override;
    bool NeedThreadFlags() const override { return false; }
  };


  /**
  \brief select local variables of main class that can be placed in auxilary buffer
  */
  std::vector<DataMemberInfo> MakeClassDataListAndCalcOffsets(std::unordered_map<std::string, DataMemberInfo>& vars, const std::unordered_set<std::string>& a_forceUsed);
  std::vector<kslicer::KernelInfo::ArgInfo> GetUserKernelArgs(const std::vector<kslicer::KernelInfo::ArgInfo>& a_allArgs);

  std::vector<std::string> GetAllPredefinedThreadIdNamesRTV();

  std::string GetRangeSourceCode(const clang::SourceRange a_range, const clang::CompilerInstance& compiler);
  std::string GetRangeSourceCode(const clang::SourceRange a_range, const clang::SourceManager& sm);
  std::string CutOffFileExt(const std::string& a_filePath);
  std::string CutOffStructClass(const std::string& a_typeName);
  
  FuncData FuncDataFromKernel(const kslicer::KernelInfo& k);
  uint64_t GetHashOfSourceRange(const clang::SourceRange& a_range);
  static constexpr size_t READ_BEFORE_USE_THRESHOLD = sizeof(float)*4;

  void PrintError(const std::string& a_msg, const clang::SourceRange& a_range, const clang::SourceManager& a_sm);
  void PrintWarning(const std::string& a_msg, const clang::SourceRange& a_range, const clang::SourceManager& a_sm);
  void ExtractTypeAndVarNameFromConstructor(clang::CXXConstructExpr* constructExpr, clang::ASTContext* astContext, std::string& varName, std::string& typeName);


  bool IsTexture(clang::QualType a_qt);
  bool IsAccelStruct(const std::string& a_typeName);
  bool IsVectorContainer(const std::string& a_typeName);
  bool IsPointerContainer(const std::string& a_typeName);

  clang::TypeDecl* SplitContainerTypes(const clang::ClassTemplateSpecializationDecl* specDecl, std::string& a_containerType, std::string& a_containerDataType);
  std::string GetDSArgName(const std::string& a_mainFuncName, const kslicer::ArgReferenceOnCall& a_arg, bool a_megakernel);
  std::string GetDSVulkanAccessLayout(TEX_ACCESS a_accessMask);
  std::string GetDSVulkanAccessMask(TEX_ACCESS a_accessMask);

  DataMemberInfo ExtractMemberInfo(clang::FieldDecl* fd, const clang::ASTContext& astContext);
  std::string InferenceVulkanTextureFormatFromTypeName(const std::string& a_typeName, bool a_useHalFloat);

  std::vector<kslicer::ArgMatch> MatchCallArgsForKernel(clang::CallExpr* call, const KernelInfo& k, const clang::CompilerInstance& a_compiler);

  std::vector<const KernelInfo*> extractUsedKernelsByName(const std::unordered_set<std::string>& a_usedNames, const std::unordered_map<std::string, KernelInfo>& a_kernels);
  KernelInfo                     joinToMegaKernel        (const std::vector<const KernelInfo*>& a_kernels, const MainFuncInfo& cf);
  std::string                    GetCFMegaKernelCall     (const MainFuncInfo& a_mainFunc);

  DATA_KIND     GetKindOfType(const clang::QualType qt);
  DECL_IN_CLASS GetKindOfDecl(const clang::TypeDecl* node);
  CPP11_ATTR GetMethodAttr(const clang::CXXMethodDecl* f, clang::CompilerInstance& a_compiler);

  KernelInfo::ArgInfo ProcessParameter(const clang::ParmVarDecl *p);
  void CheckInterlanIncInExcludedFolders(const std::vector<std::filesystem::path>& a_folders);

  ShaderFeatures GetUsedShaderFeaturesFromTypeName(const std::string& a_str);

  std::unordered_set<std::string> GetAllServiceKernels();

  std::string GetRangeSourceCode(const clang::SourceRange a_range, const clang::CompilerInstance& compiler);
  std::string GetRangeSourceCode(const clang::SourceRange a_range, const clang::SourceManager& sm);
  void PrintError(const std::string& a_msg, const clang::SourceRange& a_range, const clang::SourceManager& a_sm);
  void PrintWarning(const std::string& a_msg, const clang::SourceRange& a_range, const clang::SourceManager& a_sm);

  std::string CutOffFileExt(const std::string& a_filePath);
  std::string CutOffStructClass(const std::string& a_typeName);

  std::unordered_map<std::string, std::string> ListGLSLVectorReplacements();
  const clang::Expr* RemoveImplicitCast(const clang::Expr* a_expr);
  clang::Expr* RemoveImplicitCast(clang::Expr* a_expr);

  std::vector<std::string> GetBaseClassesNames(const clang::CXXRecordDecl* mainClassASTNode);
  std::vector<const clang::CXXRecordDecl*> ExtractAndSortBaseClasses(const std::vector<const clang::CXXRecordDecl*>& classes, const clang::CXXRecordDecl* derived);

  void ExtractBlockSizeFromCall(clang::CXXMemberCallExpr* f, kslicer::KernelInfo& kernel, const clang::CompilerInstance& compiler);

  void ProcessFunctionsInQueueBFS(kslicer::MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler, std::queue<kslicer::FuncData>& functionsToProcess, std::unordered_map<uint64_t, kslicer::FuncData>& usedFunctions);
  std::vector<kslicer::FuncData> SortByDepthInUse(const std::unordered_map<uint64_t, kslicer::FuncData>& usedFunctions);

  struct VFHAccessNodes 
  {
    VFHAccessNodes(){}
    std::string interfaceName;
    std::string interfaceTypeName;
    std::string buffName;
    std::string offsetName;
  };

  VFHAccessNodes GetVFHAccessNodes(const clang::CXXMemberCallExpr* f, const clang::CompilerInstance& a_compiler);
  bool IsCalledWithArrowAndVirtual(const clang::CXXMemberCallExpr* f);

  bool NeedRewriteTextureArray(clang::CXXMemberCallExpr* a_call, std::string& objName, int& texCoordId);

  std::string ExtractSizeFromArgExpression(const std::string& a_str);
  std::string ClearNameFromBegin(const std::string& a_str);
  std::string FixLamdbaSourceCode(std::string a_str);
  std::string SubstrBetween(const std::string& a_str, const std::string& first, const std::string& second);
  
  struct NameFlagsPair
  {
    std::string         name;
    kslicer::TEX_ACCESS flags;
    uint32_t            argId = 0;
    bool                isArg = false;
  };
  std::vector<NameFlagsPair> ListAccessedTextures(const std::vector<kslicer::ArgReferenceOnCall>& args, const kslicer::KernelInfo& kernel);
  
  /**\brief put all args together with comma or ',' to gave unique key for any concrete argument sequence.
    \return unique strig key which you can pass in std::unordered_map for example 
  */
  std::string MakeKernellCallSignature(const std::string& a_mainFuncName, const std::vector<ArgReferenceOnCall>& a_args, const std::map<std::string, UsedContainerInfo>& a_usedContainers);
}

std::unordered_map<std::string, std::string> ReadCommandLineParams(int argc, const char** argv, 
                                                                   std::unordered_map<std::string, std::string>& defines,
                                                                   std::filesystem::path& fileName,
                                                                   std::vector<std::string>& allFiles,
                                                                   std::vector<std::string>& ignoreFiles,
                                                                   std::vector<std::string>& processFiles,
                                                                   std::vector<std::string>& cppIncludes);

std::vector<const char*> ExcludeSlicerParams(int argc, const char** argv, const std::unordered_map<std::string,std::string>& params, const char* a_mainFileName,  const std::unordered_map<std::string,std::string>& defines);

void MakeAbsolutePathRelativeTo(std::filesystem::path& a_filePath, const std::filesystem::path& a_folderPath);

const char* GetClangToolingErrorCodeMessage(int code);
void ReadThreadsOrderFromStr(const std::string& threadsOrderStr, uint32_t  threadsOrder[3]);

template <typename Cont, typename Pred>
Cont filter(const Cont &container, Pred predicate)
{
  Cont result;
  std::copy_if(container.begin(), container.end(), std::back_inserter(result), predicate);
  return result;
}

#endif