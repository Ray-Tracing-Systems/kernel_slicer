#include "template_rendering.h"
#include "class_gen.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <inja.hpp>
#pragma GCC diagnostic pop

#include <algorithm>

#ifdef WIN32
typedef unsigned int uint;
#endif

// Just for convenience
using namespace inja;
using json = nlohmann::json;

static std::string GetFolderPath(const std::string& a_filePath)
{
  #ifdef WIN32
  const std::string slash = "\\";
  #else
  const std::string slash = "/";
  #endif

  size_t lastindex = a_filePath.find_last_of(slash); 
  assert(lastindex != std::string::npos);   
  return a_filePath.substr(0, lastindex); 
}

static void MakeAbsolutePathRelativeTo(std::string& a_filePath, const std::string& a_folderPath)
{
  if(a_filePath.find(a_folderPath) != std::string::npos)  // cut off folder path
    a_filePath = a_filePath.substr(a_folderPath.size() + 1);
}

static std::unordered_map<std::string, std::string> MakeMapForKernelsDeclByName(const std::vector<std::string>& kernelsCallCmdDecl)
{
  std::unordered_map<std::string,std::string> kernelDeclByName;
  for(size_t i=0;i<kernelsCallCmdDecl.size();i++)
  {
    std::string kernDecl = kernelsCallCmdDecl[i];
    size_t      rbPos    = kernDecl.find("Cmd(");
    assert(rbPos    != std::string::npos);    
    
    std::string kernName       = kernDecl.substr(0, rbPos);
    kernelDeclByName[kernName] = kernDecl;
  }
  return kernelDeclByName;
}

static inline size_t AlignedSize(const size_t a_size)
{
  size_t currSize = 4;
  while(a_size > currSize)
    currSize = currSize*2;
  return currSize;
}

static json PutHierarchyToJson(const kslicer::MainClassInfo::DHierarchy& h, const clang::CompilerInstance& compiler)
{
  json hierarchy;
  hierarchy["Name"]             = h.interfaceName;
  hierarchy["IndirectDispatch"] = (h.dispatchType == kslicer::VKERNEL_IMPL_TYPE::VKERNEL_INDIRECT_DISPATCH);
  hierarchy["IndirectOffset"]   = h.indirectBlockOffset;
  
  hierarchy["Constants"]        = std::vector<std::string>();
  for(const auto& decl : h.usedDecls)
  {
    if(decl.kind == kslicer::DECL_IN_CLASS::DECL_CONSTANT)
    {
      std::string typeInCL = decl.type;
      ReplaceFirst(typeInCL, "const", "__constant static");
      json currConstant;
      currConstant["Type"]  = typeInCL;
      currConstant["Name"]  = decl.name;
      currConstant["Value"] = kslicer::GetRangeSourceCode(decl.srcRange, compiler);
      hierarchy["Constants"].push_back(currConstant);
    }
  }
  
  hierarchy["Implementations"] = std::vector<std::string>();
  for(const auto& impl : h.implementations)
  {
    if(impl.isEmpty)
      continue;
    const auto p2 = h.tagByClassName.find(impl.name);
    assert(p2 != h.tagByClassName.end());
    json currImpl;
    currImpl["ClassName"] = impl.name;
    currImpl["TagName"]   = p2->second;
    currImpl["MemberFunctions"] = std::vector<std::string>();
    for(const auto& member : impl.memberFunctions)
    {
      currImpl["MemberFunctions"].push_back(member.srcRewritten);
    }
    currImpl["Fields"] = std::vector<std::string>();
    for(const auto& field : impl.fields)
      currImpl["Fields"].push_back(field);
      
    hierarchy["Implementations"].push_back(currImpl);
  }
  hierarchy["ImplAlignedSize"] = AlignedSize(h.implementations.size()+1);
  
  return hierarchy;  
}

static json PutHierarchiesDataToJson(const std::unordered_map<std::string, kslicer::MainClassInfo::DHierarchy>& hierarchies, 
                                     const clang::CompilerInstance& compiler)
{
  json data = std::vector<std::string>();
  for(const auto& p : hierarchies)
    data.push_back(PutHierarchyToJson(p.second, compiler));
  return data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct DSTextureAccess
{
  std::string accessLayout;
  std::string accessDSType;
  std::string SamplerName;
};

typedef typename std::unordered_map<std::string, kslicer::TEX_ACCESS>::const_iterator FlagsPointerType;
typedef typename std::unordered_map<std::string, std::string>::const_iterator         SamplerPointerType;

DSTextureAccess ObtainDSTextureAccess(const kslicer::KernelInfo& kernel, FlagsPointerType pAccessFlags, SamplerPointerType pSampler, bool isConstAccess)
{
  DSTextureAccess result;
  std::string samplerName = (pSampler == kernel.texAccessSampler.end()) ? "VK_NULL_HANDLE" : std::string("m_vdata.") + pSampler->second;
  if(pAccessFlags != kernel.texAccessInArgs.end())
  {
    result.accessLayout = "VK_IMAGE_LAYOUT_GENERAL";
    result.accessDSType = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
    result.SamplerName  = samplerName;
    if(pAccessFlags->second == kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE)
    {
      result.accessLayout = isConstAccess ? "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL" : "VK_IMAGE_LAYOUT_GENERAL";
      result.accessDSType = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
      result.SamplerName  = samplerName;
    }
  }
  else if(isConstAccess)
  {
    result.accessLayout = "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
    result.accessDSType = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
    result.SamplerName  = samplerName;
  }
  else
  {
    result.accessLayout = "VK_IMAGE_LAYOUT_GENERAL";
    result.accessDSType = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
    result.SamplerName  = samplerName;
  }

  return result;
}

DSTextureAccess ObtainDSTextureAccessArg(const kslicer::KernelInfo& kernel, int argId, bool isConstAccess)
{
  std::string argNameInKernel = kernel.args[argId].name;
  const auto pAccessFlags     = kernel.texAccessInArgs.find(argNameInKernel);
  const auto pSampler         = kernel.texAccessSampler.find(argNameInKernel); 
  return ObtainDSTextureAccess(kernel, pAccessFlags, pSampler, isConstAccess);
}

DSTextureAccess ObtainDSTextureAccessMemb(const kslicer::KernelInfo& kernel, const std::string& varName, bool isConstAccess)
{
  const auto pAccessFlags     = kernel.texAccessInMemb.find(varName);
  const auto pSampler         = kernel.texAccessSampler.find(varName); 
  return ObtainDSTextureAccess(kernel, pAccessFlags, pSampler, isConstAccess);
}


std::string kslicer::InferenceVulkanTextureFormatFromTypeName(const std::string& a_typeName, bool a_useHalFloat) // TODO: OMG, please make this with hash tables
{
  if(a_typeName == "uint" || a_typeName == "unsigned int" || a_typeName == "uint32_t")
    return "VK_FORMAT_R32_UINT";
  else if(a_typeName == "ushort " || a_typeName == "unsigned short" || a_typeName == "uint16_t")
    return "VK_FORMAT_R16_UNORM"; // assume sample floats in [0,1]
  else if(a_typeName == "uchar " || a_typeName == "unsigned char" || a_typeName == "uint8_t")
    return "VK_FORMAT_R8_UNORM";  // assume sample floats in [0,1]
  else if(a_typeName == "int" || a_typeName == "int" || a_typeName == "int32_t")
    return "VK_FORMAT_R32_SINT";
  else if(a_typeName == "short"|| a_typeName == "int16_t")
    return "VK_FORMAT_R16_SNORM"; // assume sample floats in [-1,1]
  else if(a_typeName == "char " || a_typeName == "int8_t")
    return "VK_FORMAT_R8_SNORM";  // assume sample floats in [-1,1]
  
  else if(a_typeName == "uint2" || a_typeName == "uvec2")
    return "VK_FORMAT_R32G32_UINT";
  else if(a_typeName == "ushort2")
    return "VK_FORMAT_R16G16_UNORM";
  else if(a_typeName == "uchar2")
    return "VK_FORMAT_R8G8_UNORM";
  else if(a_typeName == "int2" || a_typeName == "ivec2")
    return "VK_FORMAT_R32G32_SINT";
  else if(a_typeName == "short2")
    return "VK_FORMAT_R16G16_SNORM";
  else if(a_typeName == "char2")
    return "VK_FORMAT_R8G8_SNORM";
  
  else if(a_typeName == "uint3" || a_typeName == "uvec3")
    return "VK_FORMAT_R32G32B32_UINT";
  else if(a_typeName == "ushort3")
    return "VK_FORMAT_R16G16B16_UNORM";
  else if(a_typeName == "uchar3")
    return "VK_FORMAT_R8G8B8_UNORM";
  else if(a_typeName == "int3" || a_typeName == "ivec3")
    return "VK_FORMAT_R32G32B32_SINT";
  else if(a_typeName == "short3")
    return "VK_FORMAT_R16G16B16_SNORM";
  else if(a_typeName == "char3")
    return "VK_FORMAT_R8G8B8_SNORM";
  
  else if(a_typeName == "uint4" || a_typeName == "uvec4")
    return "VK_FORMAT_R32G32B32A32_UINT";
  else if(a_typeName == "ushort4")
    return "VK_FORMAT_R16G16B16A16_UNORM";
  else if(a_typeName == "uchar4")
    return "VK_FORMAT_R8G8B8A8_UNORM";
  else if(a_typeName == "int4" || a_typeName == "ivec4")
    return "VK_FORMAT_R32G32B32A32_SINT";
  else if(a_typeName == "short4")
    return "VK_FORMAT_R16G16B16A16_SNORM";
  else if(a_typeName == "char4")
    return "VK_FORMAT_R8G8B8A8_SNORM";

  if(a_useHalFloat)
  {
    if(a_typeName == "float4" || a_typeName == "vec4" || a_typeName == "half4")
      return "VK_FORMAT_R16G16B16A16_SFLOAT";
    else if(a_typeName == "float3" || a_typeName == "vec3" || a_typeName == "half3")
      return "VK_FORMAT_R16G16B16_SFLOAT";
    else if(a_typeName == "float2" || a_typeName == "vec2" || a_typeName == "half2")
      return "VK_FORMAT_R16G16_SFLOAT";
    else if(a_typeName == "float" || a_typeName == "half")
      return "VK_FORMAT_R16_SFLOAT";
  }
  else
  {
    if(a_typeName == "float4" || a_typeName == "vec4")
      return "VK_FORMAT_R32G32B32A32_SFLOAT";
    else if(a_typeName == "float3" || a_typeName == "vec3")
      return "VK_FORMAT_R32G32B32_SFLOAT";
    else if(a_typeName == "float2" || a_typeName == "vec2")
      return "VK_FORMAT_R32G32_SFLOAT";
    else if(a_typeName == "float")
      return "VK_FORMAT_R32_SFLOAT";

    else if(a_typeName == "half4")
      return "VK_FORMAT_R16G16B16A16_SFLOAT";
    else if(a_typeName == "half3")
      return "VK_FORMAT_R16G16B16_SFLOAT";
    else if(a_typeName == "half2")
      return "VK_FORMAT_R16G16_SFLOAT";
    else if(a_typeName == "half")
      return "VK_FORMAT_R16_SFLOAT";
  }

  return "VK_FORMAT_R32G32B32A32_SFLOAT";
}


static bool IsZeroStartLoopStatement(const clang::Stmt* a_stmt, const clang::CompilerInstance& compiler)
{
  if(a_stmt == nullptr)
    return true;

  auto text = kslicer::GetRangeSourceCode(a_stmt->getSourceRange(), compiler);
  return (text == "0");
}

static bool HaveToBeOverriden(const kslicer::MainFuncInfo& a_func, const kslicer::MainClassInfo& a_classInfo)
{
  assert(a_func.Node != nullptr);
  if(!a_func.Node->isVirtual() && !a_classInfo.IsRTV())
  {
    for(const auto& var : a_func.InOuts)
    {
      if(var.sizeUserAttr.size() != 0)
      {
        std::cout << "  [kslicer]: attribute size(...) is specified for argument '" << var.name.c_str() << "', but Control Function " << a_func.Name.c_str() << " is not virtual" << std::endl;  
        std::cout << "  [kslicer]: the Control Function which is supposed to be overriden must be virtual " << std::endl;
      }
    }

    return false;
  }

  for(const auto& var : a_func.InOuts)
  {
    if(var.kind == kslicer::DATA_KIND::KIND_POINTER)
    {
      if(var.sizeUserAttr.size() == 0) {
        std::cout << "  [kslicer]: warning, unknown data size for param " << var.name.c_str() << " of Control Function " << a_func.Name.c_str() << std::endl;
        std::cout << "  [kslicer]: the Control Function is declared virual, but kslicer can't generate implementation due to unknown data size of a pointer " << std::endl;
        return false;
      }
    }
  }

  if(a_classInfo.IsRTV())
  {
    auto p = a_classInfo.allMemberFunctions.find(a_func.Name + "Block");
    if(p == a_classInfo.allMemberFunctions.end()) 
    {
      std::stringstream strOut;
      strOut << "virtual " << a_func.ReturnType << " " << a_func.Name << "Block(";
      for(uint32_t i=0; i < a_func.Node->getNumParams(); i++)
      {
        const clang::ParmVarDecl* pParam = a_func.Node->getParamDecl(i);
        const clang::QualType paramType = pParam->getType();
        strOut << paramType.getAsString() << " " << pParam->getNameAsString();
        if(i != a_func.Node->getNumParams()-1)
          strOut << ", ";
      }
      if(a_func.Node->getNumParams() != 0)
        strOut << ", ";
      strOut << "uint32_t a_numPasses = 1)";

      std::cout << "[kslicer]: warning, can't find virtual fuction '" << a_func.Name << "Block' in main class '" << a_classInfo.mainClassName.c_str() << "'" << std::endl;
      std::cout << "[kslicer]: When RTV pattern is used, for each Control Function 'XXX' you should define 'XXXBlock' virtual function with additional parameter for passes num. " << std::endl;
      std::cout << "[kslicer]: In your case it should be: '" << strOut.str() << "'" << std::endl;
      std::cout << "[kslicer]: This function will be overriden in the generated class. " << std::endl;
    }
    
    if(p != a_classInfo.allMemberFunctions.end()) 
    {
      if(!p->second->isVirtual())
        std::cout << "[kslicer]: warning, function '" << a_func.Name << "Block' should be virtual" << std::endl;  
      //#TODO: check function prototype
    }
  }

  return true;
}

static std::string GetSizeExpression(const std::vector<std::string> a_sizeVars)
{
  if(a_sizeVars.size() == 0)
    return "0";

  if(a_sizeVars.size() == 1)
    return a_sizeVars[0];

  std::string exprText = a_sizeVars[0];
  for(size_t i=1;i<a_sizeVars.size();i++)
    exprText += ("*" + a_sizeVars[i]);
  return exprText;
}

static nlohmann::json GetJsonForFullCFImpl(const kslicer::MainFuncInfo& a_func, const kslicer::MainClassInfo& a_classInfo, const clang::CompilerInstance& compiler)
{
  nlohmann::json res;

  res["InputData"]  = std::vector<std::string>();
  res["OutputData"] = std::vector<std::string>();

  bool hasImages = false;
  
  for(const auto& var : a_func.InOuts)
  {
    if(!var.isTexture() && !var.isPointer())
      continue;
    nlohmann::json varData;
    varData["Name"]      = var.name;
    varData["IsTexture"] = var.isTexture();
    varData["IsTextureArray"] = (var.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY);
    
    std::string type = var.type;
    if(var.isPointer())
      type.erase(std::remove(type.begin(), type.end(), '*'), type.end());
    else if(var.isTexture())
    {
      varData["Format"] = kslicer::InferenceVulkanTextureFormatFromTypeName(a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(var.containerDataType), false);
      hasImages = true;
    }
  
    varData["DataType"] = type; // TODO: if texture, get data type
    varData["DataSize"] = GetSizeExpression(var.sizeUserAttr);

    if(var.isConst)
      res["InputData"].push_back(varData);
    else 
      res["OutputData"].push_back(varData);
  }
  
  // both SetVulkanInOutFor_ControlFunc(...) and ControlFuncCmd;
  //
  bool useBufferOffsets = false;
  std::stringstream callsOut;
  std::stringstream commandInOut;
  bool unclosedComma = false;

  for(uint32_t i=0; i < a_func.Node->getNumParams(); i++)
  {
    const clang::ParmVarDecl* pParam = a_func.Node->getParamDecl(i);
    const clang::QualType paramType = pParam->getType();
    callsOut << pParam->getNameAsString();
    if(i!=a_func.Node->getNumParams()-1)
      callsOut << ", ";

    if(paramType->isPointerType())                                       //#TODO: implement for textures also
    {
      if(useBufferOffsets)
        commandInOut << "tempBuffer, " << pParam->getNameAsString() << "Offset";
      else
        commandInOut << pParam->getNameAsString() << "GPU, 0";
      
      if(i!=a_func.Node->getNumParams()-1)
      {
        commandInOut << ", ";
        unclosedComma = true;
      }
      else
        unclosedComma = false;
    }
    else if(a_func.InOuts[i].isTexture())
    {
      commandInOut << pParam->getNameAsString() << "Img.image, " << pParam->getNameAsString() << "Img.view";
      if(i!=a_func.Node->getNumParams()-1)
      {
        commandInOut << ", ";
        unclosedComma = true;
      }
      else
        unclosedComma = false;
    }
  }
  

  if(unclosedComma)
    commandInOut << "0"; 
  res["ArgsOnCall"]     = callsOut.str();
  res["ArgsOnSetInOut"] = commandInOut.str();
  res["HasImages"]      = hasImages;

  return res;
}

static bool IgnoreArgForDS(size_t j, const std::vector<kslicer::ArgReferenceOnCall>& argsOnCall, const std::vector<kslicer::KernelInfo::ArgInfo>& args, const std::string& kernelName, bool isRTV)
{
  if(argsOnCall[j].name == "this") // if this pointer passed to kernel (used for virtual kernels), ignore it because it passe there anyway
    return true;
  bool ignoreArg = false; 
  
  if(isRTV)
  {
    if(argsOnCall[j].argType == kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_CONST_OR_LITERAL)
      return true;
    
    if(argsOnCall[j].isExcludedRTV)
      return true;

    bool found     = false;   
    size_t foundId = size_t(-1);
    for(size_t k=0;k<args.size();k++)
    {
      if(argsOnCall[j].name == args[k].name)
      {
        foundId = k;
        break;
      }
    }
    
    if(foundId != size_t(-1))
      ignoreArg = (args[foundId].isThreadID || args[foundId].isLoopSize || args[foundId].IsUser());
  }
  else if(j < args.size())
    ignoreArg = (args[j].isThreadID || args[j].isLoopSize || args[j].IsUser());
  
  return ignoreArg;      
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json kslicer::PrepareJsonForAllCPP(const MainClassInfo& a_classInfo, const clang::CompilerInstance& compiler,
                                             const std::vector<MainFuncInfo>& a_methodsToGenerate, const std::vector<kslicer::DeclInClass>& usedDecl,
                                             const std::string& a_genIncude,
                                             const uint32_t    threadsOrder[3],
                                             const std::string& uboIncludeName, const nlohmann::json& uboJson)
{
  std::string folderPath           = GetFolderPath(a_classInfo.mainClassFileName);
  std::string shaderPath           = "./" + a_classInfo.pShaderCC->ShaderFolder();
  std::string mainInclude          = a_classInfo.mainClassFileInclude;
  std::string mainIncludeGenerated = a_genIncude;

  MakeAbsolutePathRelativeTo(mainInclude, folderPath);
  MakeAbsolutePathRelativeTo(mainIncludeGenerated, folderPath);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::stringstream strOut;
  strOut << "#include \"" << mainInclude.c_str() << "\"" << std::endl;


  json data;
  data["Includes"]           = strOut.str();
  data["UBOIncl"]            = uboIncludeName;
  data["MainClassName"]      = a_classInfo.mainClassName;
  data["ShaderSingleFile"]   = a_classInfo.pShaderCC->ShaderSingleFile();
  data["ShaderGLSL"]         = a_classInfo.pShaderCC->IsGLSL();
  data["UseSeparateUBO"]     = a_classInfo.pShaderCC->UseSeparateUBOForArguments();
  data["UseSpecConstWgSize"] = a_classInfo.pShaderCC->UseSpecConstForWgSize();
  data["UseServiceMemCopy"]  = (a_classInfo.usedServiceCalls.find("memcpy") != a_classInfo.usedServiceCalls.end());
  data["IsRTV"]              = a_classInfo.IsRTV();
  data["IsMega"]             = a_classInfo.megakernelRTV;

  data["PlainMembersUpdateFunctions"]  = "";
  data["VectorMembersUpdateFunctions"] = "";
  data["KernelsDecls"]                 = std::vector<std::string>();
  if(a_classInfo.megakernelRTV)
  {
    for(const auto& cf: a_classInfo.mainFunc)
      data["KernelsDecls"].push_back("virtual void " + cf.megakernel.DeclCmd + ";");
  }
  else
  {
    for(const auto& k : a_classInfo.kernels)
      data["KernelsDecls"].push_back("virtual void " + k.second.DeclCmd + ";");
  } 

  data["TotalDSNumber"]   = a_classInfo.allDescriptorSetsInfo.size();
  data["VectorMembers"]   = std::vector<std::string>();
  data["TextureMembers"]  = std::vector<std::string>();
  data["TexArrayMembers"] = std::vector<std::string>();
  data["SceneMembers"]    = std::vector<std::string>(); // ray tracing specific objects
  for(const auto var : a_classInfo.dataMembers)
  {
    if(var.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED || var.IsUsedTexture())
      data["TextureMembers"].push_back(var.name);
    else if(var.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY)
      data["TexArrayMembers"].push_back(var.name);
    else if(var.isContainer && kslicer::IsVectorContainer(var.containerType))
      data["VectorMembers"].push_back(var.name);
    else if(var.isContainer && kslicer::IsPointerContainer(var.containerType) && 
                               ((var.containerDataType == "struct ISceneObject") || 
                                (var.containerDataType == "class ISceneObject")))
      data["SceneMembers"].push_back(var.name);
  }

  data["SamplerMembers"] = std::vector<std::string>();
  for(const auto& member : a_classInfo.allDataMembers)
  {
    if(member.second.type == "struct Sampler")
      data["SamplerMembers"].push_back(member.second.name);
  }

  data["Constructors"] = std::vector<std::string>();
  for(auto ctor : a_classInfo.ctors)
  {
    std::string fNameGented = ""; 
    std::string fNameOrigin = "";

    for(unsigned i=0;i<ctor->getNumParams();i++)
    {
      auto pParam = ctor->getParamDecl(i);
      auto qt     = pParam->getType();
      
      fNameGented += qt.getAsString() + " " + pParam->getNameAsString();
      fNameOrigin += pParam->getNameAsString();
    
      if(i < ctor->getNumParams()-1)
      {
        fNameOrigin += ", ";
        fNameGented += ", ";
      }
    }

    json local;
    local["ClassName"] = ctor->getNameInfo().getName().getAsString();
    local["NumParams"] = ctor->getNumParams();
    local["Params"]    = fNameGented;
    local["PrevCall"]  = fNameOrigin;      
    data["Constructors"].push_back(local);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  data["IncludeClassDecl"]    = mainIncludeGenerated;
  data["TotalDescriptorSets"] = a_classInfo.allDescriptorSetsInfo.size(); // #TODO: REFACTOR THIS !!!
  data["TotalDSNumber"]       = a_classInfo.allDescriptorSetsInfo.size(); // #TODO: REFACTOR THIS !!! 

  size_t allClassVarsSizeInBytes = 0;
  for(const auto& var : a_classInfo.dataMembers)
    allClassVarsSizeInBytes += var.sizeInBytes;
  
  data["AllClassVarsSize"]  = allClassVarsSizeInBytes;

  std::unordered_map<std::string, DataMemberInfo> containersInfo; 
  containersInfo.reserve(a_classInfo.dataMembers.size());

  data["ClassVars"] = std::vector<std::string>();
  for(const auto& v : a_classInfo.dataMembers)
  {
    if(v.isContainerInfo)
    {
      containersInfo[v.name] = v;
      continue;
    }

    if(v.isContainer)
      continue;

    json local;
    local["Name"]      = v.name;
    local["Offset"]    = v.offsetInTargetBuffer;
    local["Size"]      = v.sizeInBytes;
    local["IsArray"]   = v.isArray;
    local["ArraySize"] = v.arraySize;
    local["IsConst"]   = v.isConst;
    data["ClassVars"].push_back(local);
  }

  data["ClassVectorVars"]   = std::vector<std::string>();
  data["ClassTextureVars"]  = std::vector<std::string>();
  data["ClassTexArrayVars"] = std::vector<std::string>();
  for(const auto& v : a_classInfo.dataMembers)
  {
    if(!v.isContainer || v.usage != kslicer::DATA_USAGE::USAGE_USER)
      continue;
    
    if(v.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED)
    {
      json local;
      local["Name"] = v.name; 
      local["Usage"]       = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT";
      local["NeedUpdate"]  = true; 
      local["Format"]      = v.name + "->format()";
      local["AccessSymb"]  = "->";
      local["NeedSampler"] = true;
      data["ClassTextureVars"].push_back(local);  
    }
    else if(v.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY)
    {
      json local;
      local["Name"] = v.name; 
      local["Usage"]       = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT";
      local["NeedUpdate"]  = true; 
      local["Format"]      = v.name + "->format()";
      local["AccessSymb"]  = "->";
      local["NeedSampler"] = true;
      data["ClassTexArrayVars"].push_back(local); 
    }
    else if(v.IsUsedTexture())
    {
      json local;
      local["Name"]       = v.name;
      local["Format"]     = kslicer::InferenceVulkanTextureFormatFromTypeName(a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(v.containerDataType), a_classInfo.halfFloatTextures);
      local["Usage"]      = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT";
      local["NeedUpdate"] = false;
      local["NeedSampler"] = false;
      local["AccessSymb"] = ".";

      if(v.tmask == TEX_ACCESS::TEX_ACCESS_SAMPLE || 
         v.tmask == TEX_ACCESS::TEX_ACCESS_READ   || 
         v.tmask == TEX_ACCESS::TEX_ACCESS_NOTHING) // TEX_ACCESS_NOTHING arises due to passing textures to functions
      {
        local["Usage"]      = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT";
        local["NeedUpdate"] = true;
      }
      else if( int(v.tmask) == (int(TEX_ACCESS::TEX_ACCESS_READ) | int(TEX_ACCESS::TEX_ACCESS_SAMPLE)))
      {
        local["Usage"]      = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT";
        local["NeedUpdate"] = true;
      }
      else if( int(v.tmask) == (int(TEX_ACCESS::TEX_ACCESS_READ)   | int(TEX_ACCESS::TEX_ACCESS_WRITE)) )
      {
        local["Usage"]      = "VK_IMAGE_USAGE_STORAGE_BIT";
        local["NeedUpdate"] = false;
      }
      else if( int(v.tmask) == (int(TEX_ACCESS::TEX_ACCESS_SAMPLE) | int(TEX_ACCESS::TEX_ACCESS_WRITE)))
      {
        local["Usage"]      = "VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT";
        local["NeedUpdate"] = false;
      }

      data["ClassTextureVars"].push_back(local);     
    }
    else if(v.isContainer && kslicer::IsVectorContainer(v.containerType))
    {
      std::string sizeName     = v.name + "_size";
      std::string capacityName = v.name + "_capacity";
  
      auto p1 = containersInfo.find(sizeName);
      auto p2 = containersInfo.find(capacityName);
  
      assert(p1 != containersInfo.end() && p2 != containersInfo.end());
  
      json local;
      local["Name"]           = v.name;
      local["SizeOffset"]     = p1->second.offsetInTargetBuffer;
      local["CapacityOffset"] = p2->second.offsetInTargetBuffer;
      local["TypeOfData"]     = v.containerDataType;
      local["AccessSymb"]     = ".";
      local["NeedSampler"]    = false;
      data["ClassVectorVars"].push_back(local);     
    }
    // TODO: add processing for Scene/Acceleration structures

  }

  data["RedVectorVars"] = std::vector<std::string>();
  for(const auto& v : a_classInfo.dataMembers)
  {
    if(!v.isContainer || v.usage != kslicer::DATA_USAGE::USAGE_SLICER_REDUCTION)
      continue;
    
    json local;
    local["Name"] = v.name;
    local["Type"] = v.containerDataType;

    data["RedVectorVars"].push_back(local);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<kslicer::KernelInfo> currKernels;
  if(a_classInfo.megakernelRTV)
  {
    for(auto& cf : a_classInfo.mainFunc)
      currKernels.push_back(cf.megakernel);
  }
  else
  {
    for(const auto& nk : a_classInfo.kernels)
      currKernels.push_back(nk.second);
  }

  std::vector<std::string> kernelsCallCmdDecl; 
  kernelsCallCmdDecl.reserve(a_classInfo.kernels.size());
  for(const auto& k : currKernels)
    kernelsCallCmdDecl.push_back(k.DeclCmd);  

  auto kernelDeclByName = MakeMapForKernelsDeclByName(kernelsCallCmdDecl);

  data["MultipleSourceShaders"] = !a_classInfo.pShaderCC->IsSingleSource();
  data["ShaderFolder"]          = a_classInfo.pShaderCC->ShaderFolder();
  
  auto dhierarchies             = a_classInfo.GetDispatchingHierarchies();
  data["DispatchHierarchies"]   = PutHierarchiesDataToJson(dhierarchies, compiler);
  
  data["IndirectBufferSize"] = a_classInfo.m_indirectBufferSize;
  data["IndirectDispatches"] = std::vector<std::string>();
  data["Kernels"]            = std::vector<std::string>();  

  bool useSubgroups = false;
  int subgroupMaxSize = 0;

  for(const auto& k : currKernels)
  {    
    std::string kernName = a_classInfo.RemoveKernelPrefix(k.name);
    const auto auxArgs   = GetUserKernelArgs(k.args);
    
    std::string outFileName = k.name + "_UpdateIndirect" + ".cl.spv";
    std::string outFilePath = shaderPath + "/" + outFileName;

    useSubgroups    = useSubgroups || k.enableSubGroups;
    subgroupMaxSize = std::max(subgroupMaxSize, int(k.warpSize));

    if(k.isIndirect)
    {
      json indirectJson;
      indirectJson["KernelName"]   = kernName;
      indirectJson["OriginalName"] = k.name;
      indirectJson["ShaderPath"]   = outFilePath.c_str();
      indirectJson["Offset"]       = k.indirectBlockOffset;
      data["IndirectDispatches"].push_back(indirectJson);
    }
    else if(k.isMaker)
    {
      auto p = a_classInfo.m_vhierarchy.find(k.interfaceName);
      if(p != a_classInfo.m_vhierarchy.end() && p->second.dispatchType == kslicer::VKERNEL_IMPL_TYPE::VKERNEL_INDIRECT_DISPATCH)
      {
        json indirectJson;
        indirectJson["KernelName"]   = kernName;
        indirectJson["OriginalName"] = k.name;
        indirectJson["ShaderPath"]   = outFilePath.c_str();
        indirectJson["Offset"]       = k.indirectMakerOffset;
        data["IndirectDispatches"].push_back(indirectJson);
      }
    }

    json kernelJson;
    kernelJson["Name"]           = kernName;
    kernelJson["OriginalName"]   = k.name;
    kernelJson["IsIndirect"]     = k.isIndirect;
    kernelJson["IndirectOffset"] = k.indirectBlockOffset;
    kernelJson["IsMaker"]        = k.isMaker;
    kernelJson["IsVirtual"]      = k.isVirtual;
    kernelJson["ArgCount"]       = k.args.size();
    kernelJson["HasLoopInit"]    = k.hasInitPass;
    kernelJson["HasLoopFinish"]  = k.hasFinishPassSelf;
    kernelJson["Decl"]           = kernelDeclByName[kernName];
    kernelJson["Args"]           = std::vector<std::string>();
    kernelJson["threadDim"]      = a_classInfo.GetKernelTIDArgs(k).size();
    size_t actualSize     = 0;
    for(const auto& arg : k.args)
    {
      const auto pos1 = arg.type.find(std::string("class ")  + a_classInfo.mainClassName);
      const auto pos2 = arg.type.find(std::string("struct ") + a_classInfo.mainClassName);
      const auto pos3 = arg.type.find(a_classInfo.mainClassName);
      if(arg.isThreadID || arg.isLoopSize || arg.IsUser() ||                                  // exclude TID and loopSize args bindings
         pos1 != std::string::npos || pos2 != std::string::npos || pos3 != std::string::npos) // exclude special case of passing MainClass to virtual kernels
        continue;
      
      json argData;
      argData["Name"]  = arg.name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Count"] = "1";
      argData["Id"]    = actualSize;
      argData["IsTextureArray"] = false;

      if(arg.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY)
      {
        argData["Count"] = arg.name + ".size()";
        argData["Type"]  = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
        argData["IsTextureArray"] = true;
      }
      else if(arg.IsTexture())
      {
        auto pAccessFlags = k.texAccessInArgs.find(arg.name);
        if(pAccessFlags->second == TEX_ACCESS::TEX_ACCESS_SAMPLE)
          argData["Type"] = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
        else
          argData["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
      }
      else if(arg.isContainer && arg.containerDataType == "struct ISceneObject")
      {
        argData["Type"] = "VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR";
      }
      else
        argData["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";

      kernelJson["Args"].push_back(argData);
      actualSize++;
    }

    for(const auto& container : k.usedContainers) // TODO: add support fo textures (!!!)
    {
      json argData;
      argData["Name"]  = container.second.name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
      argData["Count"] = "1";
      argData["IsTextureArray"] = false;
      
      if(container.second.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY)
      {
        argData["Type"]  = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
        argData["Count"] = container.second.name + ".size()";
        argData["IsTextureArray"] = true;
      }
      else if(container.second.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED)
      {
        argData["Type"] = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
      }
      else if(container.second.isTexture())
      {
        auto pAccessFlags = k.texAccessInMemb.find(container.second.name);
        if(pAccessFlags == k.texAccessInMemb.end() || pAccessFlags->second == TEX_ACCESS::TEX_ACCESS_SAMPLE)
          argData["Type"] = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
        else
          argData["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
      }
      else if(container.second.isAccelStruct())
      {
        argData["Type"] = "VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR";
      }
      else
        argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
      
      kernelJson["Args"].push_back(argData);
      actualSize++;
    }

    if(k.isMaker || k.isVirtual)
    {
      json argData;
      argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
      argData["Name"]  = "SomeInterfaceObjPointerData";
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
      argData["Count"] = "1";
      argData["IsTextureArray"] = false;
      kernelJson["Args"].push_back(argData);
      actualSize++;

      json hierarchy = PutHierarchyToJson(dhierarchies[k.interfaceName], compiler); 
      
      //std::cout << std::endl << "--------------------------" << std::endl;
      //std::cout << hierarchy.dump(2) << std::endl;
      //std::cout << std::endl << "--------------------------" << std::endl;

      hierarchy["RedLoop1"] = std::vector<std::string>();
      hierarchy["RedLoop2"] = std::vector<std::string>();
      const uint32_t blockSize = k.wgSize[0]*k.wgSize[1]*k.wgSize[2];
      for (uint c = blockSize/2; c>k.warpSize; c/=2)
        hierarchy["RedLoop1"].push_back(c);
      for (uint c = k.warpSize; c>0; c/=2)
        hierarchy["RedLoop2"].push_back(c);
      kernelJson["Hierarchy"] = hierarchy; 
    }
    else
    {
      json temp;
      temp["IndirectDispatch"] = false; // because of 'Kernel.Hierarchy.IndirectDispatch' check
      kernelJson["Hierarchy"] = temp;
    }

    kernelJson["ArgCount"] = actualSize;
  
    auto tidArgs = a_classInfo.GetKernelTIDArgs(k);
    std::vector<std::string> threadIdNamesList(tidArgs.size());
    assert(threadIdNamesList.size() <= 3);
    assert(threadIdNamesList.size() > 0);

    // change threads/loops order if required
    //
    for(size_t i=0;i<tidArgs.size();i++)
    {
      uint32_t tid = std::min<uint32_t>(threadsOrder[i], tidArgs.size()-1);
      if(tidArgs[tid].loopIter.condKind == kslicer::KernelInfo::IPV_LOOP_KIND::LOOP_KIND_LESS_EQUAL)
        threadIdNamesList[i] = tidArgs[tid].loopIter.sizeText + "+1";
      else
        threadIdNamesList[i] = tidArgs[tid].loopIter.sizeText;
    }

    if(threadIdNamesList.size() > 0)
    {
      kernelJson["tidX"] = threadIdNamesList[0];
      kernelJson["begX"] = tidArgs[0].loopIter.startText == "" ? "0" : tidArgs[0].loopIter.startText;  
      kernelJson["SmplX"] = IsZeroStartLoopStatement(tidArgs[0].loopIter.startNode, compiler);
    }
    else
    {
      kernelJson["tidX"] = 1;
      kernelJson["begX"] = 0;
      kernelJson["SmplX"] = true;
    }

    if(threadIdNamesList.size() > 1)
    {
      kernelJson["tidY"] = threadIdNamesList[1];
      kernelJson["begY"] = tidArgs[1].loopIter.startText;  
      kernelJson["SmplY"] = IsZeroStartLoopStatement(tidArgs[1].loopIter.startNode, compiler);
    }
    else
    {
      kernelJson["tidY"] = 1;
      kernelJson["begY"] = 0;
      kernelJson["SmplY"] = true;
    }

    if(threadIdNamesList.size() > 2)
    {
      kernelJson["tidZ"] = threadIdNamesList[2];
      kernelJson["begZ"] = tidArgs[2].loopIter.startText;  
      kernelJson["SmplZ"] = IsZeroStartLoopStatement(tidArgs[2].loopIter.startNode, compiler);
    }
    else
    {
      kernelJson["tidZ"] = 1;
      kernelJson["begZ"] = 0;
      kernelJson["SmplZ"] = true;
    }

    // put auxArgs to push constants
    //
    int sizeCurr = 0;
    kernelJson["AuxArgs"] = std::vector<std::string>();
    for(auto arg : auxArgs)
    {
      std::string typestr = kslicer::CleanTypeName(arg.type);
      json argData;
      argData["Name"] = arg.name;
      argData["Type"] = typestr;
      kernelJson["AuxArgs"].push_back(argData);
      sizeCurr += arg.size;
    }
    
    // identify wherther we nedd to add reduction pass after current kernel
    //
    json reductionVarNames = std::vector<std::string>();
    for(const auto& var : k.subjectedToReduction)
    {
      if(!var.second.SupportAtomicLastStep())
      {
        json varData;
        varData["Name"] = var.second.tmpVarName;
        varData["Type"] = var.second.dataType;
        reductionVarNames.push_back(varData);
      }
    }
      
    kernelJson["FinishRed"]    = (reductionVarNames.size() !=0);
    kernelJson["RedVarsFPNum"] = reductionVarNames.size();
    kernelJson["RedVarsFPArr"] = reductionVarNames;

    kernelJson["WGSizeX"]      = k.wgSize[0]; //
    kernelJson["WGSizeY"]      = k.wgSize[1]; // 
    kernelJson["WGSizeZ"]      = k.wgSize[2]; // 

    data["Kernels"].push_back(kernelJson);
  }
  
  data["UseSubGroups"] = useSubgroups;
  data["SubGroupSize"] = subgroupMaxSize;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  // check for CommitDeviceData and GetExecutionTime and other specific functions
  { 
    auto pCommit  = a_classInfo.allMemberFunctions.find("CommitDeviceData");
    auto pGetTime = a_classInfo.allMemberFunctions.find("GetExecutionTime");
    auto pUpdPOD  = a_classInfo.allMemberFunctions.find("UpdateMembersPlainData");
    auto pUpdVec  = a_classInfo.allMemberFunctions.find("UpdateMembersVectorData");
    auto pUpdTex  = a_classInfo.allMemberFunctions.find("UpdateMembersTexureData");
    auto pScnRstr = a_classInfo.allMemberFunctions.find("SceneRestrictions");
    
    if(pCommit == a_classInfo.allMemberFunctions.end())
      std::cout << "  [kslicer]: warning, can't find fuction 'CommitDeviceData', you should define it: 'virtual void CommitDeviceData(){}'" << std::endl; 
    else
    {
      if(!pCommit->second->isVirtual())
        std::cout << "  [kslicer]: warning, function 'CommitDeviceData' should be virtual" << std::endl;  
    }
  
    if(pGetTime == a_classInfo.allMemberFunctions.end())
      std::cout << "  [kslicer]: warning, can't find fuction 'GetExecutionTime', you should define it: 'virtual void GetExecutionTime(const char* a_funcName, float a_out[4]){}'" << std::endl; 
    else
    {
      if(!pGetTime->second->isVirtual())
        std::cout << "  [kslicer]: warning, function 'GetExecutionTime' should be virtual" << std::endl;  
    }

    if(pUpdPOD != a_classInfo.allMemberFunctions.end())
    {
      if(!pUpdPOD->second->isVirtual())
        std::cout << "  [kslicer]: warning, function 'UpdateMembersPlainData' should be virtual" << std::endl;  
    }

    if(pUpdVec != a_classInfo.allMemberFunctions.end())
    {
      if(!pUpdVec->second->isVirtual())
        std::cout << "  [kslicer]: warning, function 'UpdateMembersVectorData' should be virtual" << std::endl;  
    }

    if(pUpdTex != a_classInfo.allMemberFunctions.end())
    {
      if(!pUpdTex->second->isVirtual())
        std::cout << "  [kslicer]: warning, function 'UpdateMembersTexureData' should be virtual" << std::endl;  
    }

    if(pScnRstr == a_classInfo.allMemberFunctions.end() && a_classInfo.IsRTV())
    {
      std::cout << "  [kslicer]: warning, function 'SceneRestrictions' is not found. It would be generated by kernel_slicer." << std::endl;  
      std::cout << "  [kslicer]: you may add this function in base class or override it in derived class (derived from generated class)." << std::endl;
    }

    data["HasCommitDeviceFunc"]       = (pCommit  != a_classInfo.allMemberFunctions.end());
    data["HasGetTimeFunc"]            = (pGetTime != a_classInfo.allMemberFunctions.end());

    data["UpdateMembersPlainData"]    = (pUpdPOD  != a_classInfo.allMemberFunctions.end());
    data["UpdateMembersVectorData"]   = (pUpdVec  != a_classInfo.allMemberFunctions.end());
    data["UpdateMembersTextureData"]  = (pUpdTex  != a_classInfo.allMemberFunctions.end());
    data["GenerateSceneRestrictions"] = (pScnRstr == a_classInfo.allMemberFunctions.end() && a_classInfo.IsRTV());
  }

  data["HasRTXAccelStruct"] = false;
  data["MainFunctions"] = std::vector<std::string>();
  bool atLeastOneFullOverride = false;
  for(const auto& mainFunc : a_methodsToGenerate)
  {
    json data2;
    data2["Name"]                 = mainFunc.Name;
    data2["DescriptorSets"]       = std::vector<std::string>();
    data2["Decl"]                 = mainFunc.GeneratedDecl;
    data2["DeclOrig"]             = mainFunc.OriginalDecl;
    data2["LocalVarsBuffersDecl"] = std::vector<std::string>();

    bool HasCPUOverride = HaveToBeOverriden(mainFunc, a_classInfo); 
    data2["OverrideMe"] = HasCPUOverride;
    if(HasCPUOverride)
    {
      data2["FullImpl"] = GetJsonForFullCFImpl(mainFunc, a_classInfo, compiler);
      atLeastOneFullOverride = true;
    }
    data2["IsRTV"] = a_classInfo.IsRTV();

    for(const auto& v : mainFunc.Locals)
    {
      json local;
      local["Name"] = v.second.name;
      local["Type"] = v.second.type;
      local["TransferDST"] = (v.second.name == "threadFlags"); // rtv thread flags
      data2["LocalVarsBuffersDecl"].push_back(local);
    }

    data2["InOutVars"] = std::vector<std::string>();
    for(const auto& v : mainFunc.InOuts)
    {
      if(v.isThreadId || v.kind == DATA_KIND::KIND_POD || v.kind == DATA_KIND::KIND_UNKNOWN)
        continue;
      json controlArg;
      controlArg["Name"]      = v.name;
      controlArg["IsTexture"] = v.isTexture();
      data2["InOutVars"].push_back(controlArg);
    }

    // for impl, ds bindings
    //
    //std::ofstream debug("z_problems.txt", std::ios::app);
    for(size_t i=mainFunc.startDSNumber; i<mainFunc.endDSNumber; i++)
    {
      auto& dsArgs = a_classInfo.allDescriptorSetsInfo[i];
      //std::cout << "[ds bindings] kname = " << dsArgs.originKernelName.c_str() << std::endl;

      const auto pFoundKernel   = a_classInfo.FindKernelByName(dsArgs.originKernelName);
      const bool internalKernel = (pFoundKernel == a_classInfo.kernels.end());
      const bool isMegaKernel   = internalKernel ? false : pFoundKernel->second.isMega;
      
      json local;
      local["Id"]         = i;
      local["KernelName"] = dsArgs.kernelName;
      local["Layout"]     = dsArgs.kernelName + "DSLayout";
      local["Args"]       = std::vector<std::string>();
      local["ArgNames"]   = std::vector<std::string>();
      local["IsServiceCall"] = dsArgs.isService;
      local["IsVirtual"]     = false;

      uint32_t realId = 0; 
      for(size_t j=0;j<dsArgs.descriptorSetsInfo.size();j++)
      {  
        if(!internalKernel)
        {
          const bool ignoreArg = IgnoreArgForDS(j, dsArgs.descriptorSetsInfo, pFoundKernel->second.args, pFoundKernel->second.name, a_classInfo.IsRTV()); 
          if(ignoreArg && !isMegaKernel) 
            continue;
        }
        
        const std::string dsArgName = kslicer::GetDSArgName(mainFunc.Name, dsArgs.descriptorSetsInfo[j], a_classInfo.megakernelRTV);

        json arg;
        arg["Id"]            = realId;
        arg["Name"]          = dsArgName;
        arg["NameOriginal"]  = dsArgs.descriptorSetsInfo[j].name;
        arg["Offset"]        = 0;
        arg["IsTexture"]     = dsArgs.descriptorSetsInfo[j].isTexture();
        arg["IsAccelStruct"] = dsArgs.descriptorSetsInfo[j].isAccelStruct();
        arg["IsTextureArray"]= (dsArgs.descriptorSetsInfo[j].kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY);

        if(dsArgs.descriptorSetsInfo[j].kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED)
        {
          arg["IsTexture"]     = true;
          arg["IsAccelStruct"] = false;
          arg["AccessLayout"]  = "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
          arg["AccessDSType"]  = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
          arg["SamplerName"]   = std::string("m_vdata.") + dsArgs.descriptorSetsInfo[j].name + "Sampler";
        }
        else if(dsArgs.descriptorSetsInfo[j].kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY)
        {
          arg["IsTexture"]     = false;
          arg["IsTextureArray"]= true;
          arg["IsAccelStruct"] = false;
          arg["AccessLayout"]  = "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
          arg["AccessDSType"]  = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
          arg["SamplerName"]   = std::string("m_vdata.") + dsArgs.descriptorSetsInfo[j].name + "ArraySampler";
        }
        else if(dsArgs.descriptorSetsInfo[j].isTexture())
        {
          bool isConst = dsArgs.descriptorSetsInfo[j].isConst;
          auto pMember = a_classInfo.allDataMembers.find(dsArgName);
          if(pMember != a_classInfo.allDataMembers.end() && pMember->second.IsUsedTexture())
            isConst = (pMember->second.tmask == TEX_ACCESS::TEX_ACCESS_SAMPLE) || 
                      (pMember->second.tmask == TEX_ACCESS::TEX_ACCESS_READ)   ||  
                      (pMember->second.tmask == TEX_ACCESS::TEX_ACCESS_NOTHING);

          auto texDSInfo = ObtainDSTextureAccessArg(pFoundKernel->second, j, isConst);
          arg["AccessLayout"] = texDSInfo.accessLayout;
          arg["AccessDSType"] = texDSInfo.accessDSType;
          arg["SamplerName"]  = texDSInfo.SamplerName;
        }
        else if(dsArgs.descriptorSetsInfo[j].isAccelStruct())
        {
          //std::cout << "[kslicer error]: passing acceleration structures to kernel arguments is not yet implemented" << std::endl; 
          data["HasRTXAccelStruct"] = true;
        } 

        local["Args"].push_back(arg);
        local["ArgNames"].push_back(dsArgs.descriptorSetsInfo[j].name);
        realId++;
      }
      
      if(pFoundKernel != a_classInfo.kernels.end() && !isMegaKernel) // seems for MegaKernel these containers are already in 'dsArgs.descriptorSetsInfo' 
      {
        for(const auto& container : pFoundKernel->second.usedContainers) // add all class-member vectors bindings
        {
          json arg;
          arg["Id"]            = realId;
          arg["Name"]          = "m_vdata." + container.second.name;
          arg["NameOriginal"]  = container.second.name;
          arg["IsTexture"]     = container.second.isTexture();
          arg["IsAccelStruct"] = container.second.isAccelStruct();
          arg["IsTextureArray"]= false;
          
          if(container.second.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED)
          {
            arg["IsTexture"]     = true;
            arg["IsAccelStruct"] = false;
            arg["AccessLayout"]  = "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
            arg["AccessDSType"]  = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
            arg["SamplerName"]   = std::string("m_vdata.") + container.second.name + "Sampler";
          }
          else if(container.second.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY)
          {
            arg["IsTexture"]     = false;
            arg["IsTextureArray"]= true;
            arg["IsAccelStruct"] = false;
            arg["AccessLayout"]  = "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
            arg["AccessDSType"]  = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
            arg["SamplerName"]   = std::string("m_vdata.") + container.second.name + "ArraySampler";
          }
          else if(container.second.isTexture())
          {
            auto pMember = a_classInfo.allDataMembers.find(container.second.name);
            bool isConst = (pMember->second.tmask == TEX_ACCESS::TEX_ACCESS_SAMPLE) || 
                           (pMember->second.tmask == TEX_ACCESS::TEX_ACCESS_READ)   ||  
                           (pMember->second.tmask == TEX_ACCESS::TEX_ACCESS_NOTHING);
            auto texDSInfo = ObtainDSTextureAccessMemb(pFoundKernel->second, container.second.name, isConst);
            arg["AccessLayout"] = texDSInfo.accessLayout;
            arg["AccessDSType"] = texDSInfo.accessDSType;
            arg["SamplerName"]  = texDSInfo.SamplerName;
          }
          else if(container.second.isAccelStruct())
          {
            data["HasRTXAccelStruct"] = true;
            arg["Name"] = container.second.name; // remove m_vdata."
          }
          else // buffer
          {
            if(container.second.isSetter)
            {
              arg["Name"]   = container.second.setterPrefix + "Vulkan." + container.second.setterSuffix;
            }
          }

          local["Args"].push_back(arg);
          local["ArgNames"].push_back(container.second.name);
          realId++;
        }

        if(pFoundKernel->second.isMaker || pFoundKernel->second.isVirtual)
        {
          auto hierarchy = dhierarchies[pFoundKernel->second.interfaceName];

          json arg;
          arg["Id"]        = realId;
          arg["Name"]      = std::string("m_") + hierarchy.interfaceName + "ObjPtr";
          arg["IsTexture"] = false;
          arg["IsTextureArray"]= false;
          arg["IsAccelStruct"] = false;

          local["Args"].push_back(arg);
          local["ArgNames"].push_back(hierarchy.interfaceName + "ObjPtrData");
          realId++;          
        }
        
        local["IsVirtual"] = pFoundKernel->second.isVirtual;
        if(pFoundKernel->second.isVirtual)
        {
          auto hierarchy            = dhierarchies[pFoundKernel->second.interfaceName];
          local["ObjectBufferName"] = hierarchy.objBufferName;
        }
      }
      
      local["ArgNumber"] = realId;
      data2["DescriptorSets"].push_back(local);
    }
    //debug.close();

    data2["MainFuncDeclCmd"]      = mainFunc.GeneratedDecl;
    data2["MainFuncTextCmd"]      = mainFunc.CodeGenerated;
    data2["ReturnType"]           = mainFunc.ReturnType;
    data2["IsRTV"]                = a_classInfo.IsRTV();
    data2["IsMega"]               = a_classInfo.megakernelRTV;
    data2["NeedThreadFlags"]      = a_classInfo.NeedThreadFlags();
    data2["NeedToAddThreadFlags"] = mainFunc.needToAddThreadFlags;
    data2["DSId"]                 = mainFunc.startDSNumber;
    data2["MegaKernelCall"]       = mainFunc.MegaKernelCall;
    data["MainFunctions"].push_back(data2);
  }
  
  data["HasFullImpl"] = atLeastOneFullOverride;
  if(atLeastOneFullOverride && a_classInfo.ctors.size() == 0)
  {
    std::cout << "[kslicer warning]: " << "class '" << a_classInfo.mainClassName << "' has Control Functions to override, but does not have any declared constructots" << std::endl;
    std::cout << "[kslicer warning]: " << "at least one constructor should be declared to succsesfully generate factory function for generated class" << std::endl;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  data["DescriptorSetsAll"] = std::vector<std::string>();
  for(size_t i=0; i< a_classInfo.allDescriptorSetsInfo.size();i++)
  {
    const auto& dsArgs = a_classInfo.allDescriptorSetsInfo[i];
    json local;
    local["Id"]         = i;
    local["Layout"]     = dsArgs.kernelName + "DSLayout";
    data["DescriptorSetsAll"].push_back(local);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  data["SettersDecl"] = std::vector<std::string>();
  data["SetterFuncs"] = std::vector<std::string>();
  data["SetterVars"]  = std::vector<std::string>();

  for(const auto& s : a_classInfo.m_setterStructDecls)
    data["SettersDecl"].push_back(s);

  for(const auto& s : a_classInfo.m_setterFuncDecls)
    data["SetterFuncs"].push_back(s);

  for(const auto& kv : a_classInfo.m_setterVars)
  { 
    json local;
    local["Type"] = kv.second;
    local["Name"] = kv.first;
    data["SetterVars"].push_back(local);
  }

  // declarations of struct, constants and typedefs inside class
  //
  std::unordered_set<std::string> excludedNames;
  for(auto pair : a_classInfo.m_setterVars)
    excludedNames.insert(kslicer::CleanTypeName(pair.second));

  data["ClassDecls"] = std::vector<std::string>();
  for(const auto decl : usedDecl)
  {
    if(!decl.extracted)
      continue;
    if(excludedNames.find(decl.type) != excludedNames.end())
      continue;

    json cdecl;
    cdecl["InClass"] = decl.inClass;
    cdecl["IsType"]  = (decl.kind == DECL_IN_CLASS::DECL_STRUCT);
    cdecl["Type"]    = kslicer::CleanTypeName(decl.type);
    data["ClassDecls"].push_back(cdecl);
  }

  return data;
}

namespace kslicer
{
  std::string GetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, 
                                      const std::vector<kslicer::ArgFinal>& threadIds,
                                      const std::string a_names[3]);
}

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);


nlohmann::json kslicer::PrepareUBOJson(MainClassInfo& a_classInfo, const std::vector<kslicer::DataMemberInfo>& a_dataMembers, const clang::CompilerInstance& compiler)
{
  nlohmann::json data;
  auto pShaderRewriter = a_classInfo.pShaderFuncRewriter;

  auto podMembers = filter(a_classInfo.dataMembers, [](auto& memb) { return !memb.isContainer; });
  uint32_t dummyCounter = 0;
  data["MainClassName"]   = a_classInfo.mainClassName;
  data["UBOStructFields"] = std::vector<std::string>();
  data["ShaderGLSL"]      = a_classInfo.pShaderCC->IsGLSL();
  data["Hierarchies"]     = PutHierarchiesDataToJson(a_classInfo.GetDispatchingHierarchies(), compiler);

  for(auto member : podMembers)
  {
    std::string typeStr = member.type;
    if(member.isArray)
      typeStr = typeStr.substr(0, typeStr.find("["));
    typeStr = pShaderRewriter->RewriteStdVectorTypeStr(typeStr); 
    ReplaceFirst(typeStr, "const ", "");

    size_t sizeO = member.sizeInBytes;
    size_t sizeA = member.alignedSizeInBytes;

    const bool isVec3Member = ((typeStr == "vec3") || (typeStr == "ivec3") || (typeStr == "uvec3")) && a_classInfo.pShaderCC->IsGLSL();

   
    json uboField;
    uboField["Type"]      = typeStr;
    uboField["Name"]      = member.name;
    uboField["IsArray"]   = member.isArray;
    uboField["ArraySize"] = member.arraySize;
    uboField["IsDummy"]   = false; 
    uboField["IsVec3"]    = isVec3Member; 
    data["UBOStructFields"].push_back(uboField);
    
    while(sizeO < sizeA) // TODO: make this more effitient
    {
      std::stringstream strOut;
      strOut << "dummy" << dummyCounter;
      dummyCounter++;
      sizeO += sizeof(uint32_t);
      uboField["Type"]    = "uint";
      uboField["Name"]    = strOut.str();
      uboField["IsDummy"] = true;
      uboField["IsVec3"]  = false;
      data["UBOStructFields"].push_back(uboField);
    }

  }


  return data;
}
