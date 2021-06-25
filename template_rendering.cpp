#include "template_rendering.h"
#include "class_gen.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include <inja.hpp>
#include <algorithm>

// Just for convenience
using namespace inja;
using json = nlohmann::json;

std::string GetFolderPath(const std::string& a_filePath)
{
  size_t lastindex = a_filePath.find_last_of("/"); 
  assert(lastindex != std::string::npos);   
  return a_filePath.substr(0, lastindex); 
}

void MakeAbsolutePathRelativeTo(std::string& a_filePath, const std::string& a_folderPath)
{
  if(a_filePath.find(a_folderPath) != std::string::npos)  // cut off folder path
    a_filePath = a_filePath.substr(a_folderPath.size() + 1);
}

void kslicer::PrintVulkanBasicsFile(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo)
{
  json data;
  inja::Environment env;
  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, data);
  
  std::string folderPath = GetFolderPath(a_classInfo.mainClassFileName);

  std::ofstream fout(folderPath + "/vulkan_basics.h");
  fout << result.c_str() << std::endl;
  fout.close();
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

std::string kslicer::GetDSArgName(const std::string& a_mainFuncName, const kslicer::ArgReferenceOnCall& a_arg)
{
  switch(a_arg.argType)
  {
    case  kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG:
    return a_mainFuncName + "_local." + a_arg.varName; 

    case  kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL:
    case  kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_VECTOR:
    {
      auto posOfData = a_arg.varName.find(".data()");
      if(posOfData != std::string::npos)
        return std::string("m_vdata.") + a_arg.varName.substr(0, posOfData);
      else
        return a_mainFuncName + "_local." + a_arg.varName; 
    }
    
    default:
    return std::string("m_vdata.") + a_arg.varName;
  };
}


std::string kslicer::GetDSVulkanAccessMask(kslicer::TEX_ACCESS a_accessMask)
{
  switch(a_accessMask)
  {
    case kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE:
    case kslicer::TEX_ACCESS::TEX_ACCESS_READ:
    return "VK_ACCESS_SHADER_READ_BIT";

    case kslicer::TEX_ACCESS::TEX_ACCESS_WRITE:
    return "VK_ACCESS_SHADER_WRITE_BIT";

    default:
    return "VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT";
  }
}

std::vector<kslicer::KernelInfo::Arg> kslicer::GetUserKernelArgs(const std::vector<kslicer::KernelInfo::Arg>& a_allArgs)
{
  std::vector<kslicer::KernelInfo::Arg> result;
  result.reserve(a_allArgs.size());

  for(const auto& arg : a_allArgs)
  {
    if(arg.IsUser())
      result.push_back(arg);
  }

  return result;
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
  hierarchy["IndirectDispatch"] = (h.dispatchType == kslicer::VKERNEL_INDIRECT_DISPATCH);
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
  json data;
  data = std::vector<std::string>();
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
  if(pAccessFlags != kernel.texAccessInArgs.end())
  {
    result.accessLayout = "VK_IMAGE_LAYOUT_GENERAL";
    result.accessDSType = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
    result.SamplerName  = "VK_NULL_HANDLE";
    if(pAccessFlags->second == kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE)
    {
      result.accessDSType = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
      result.SamplerName  = (pSampler == kernel.texAccessSampler.end()) ? "VK_NULL_HANDLE" : std::string("m_vdata.") + pSampler->second;
    }
  }
  else if(isConstAccess)
  {
    result.accessLayout = "VK_IMAGE_LAYOUT_GENERAL";
    result.accessDSType = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
    result.SamplerName  = "VK_NULL_HANDLE";
  }
  else
  {
    result.accessLayout = "VK_IMAGE_LAYOUT_GENERAL";
    result.accessDSType = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
    result.SamplerName  = "VK_NULL_HANDLE";
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json kslicer::PrepareJsonForAllCPP(const MainClassInfo& a_classInfo, const clang::CompilerInstance& compiler,
                                             const std::vector<MainFuncInfo>& a_methodsToGenerate, 
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

  std::stringstream strOut2;
  for(const auto& k : a_classInfo.kernels)
    strOut2 << "virtual void " << k.second.DeclCmd.c_str() << ";\n"; // << k.second.RetType.c_str()

  json data;
  data["Includes"]           = strOut.str();
  data["UBOIncl"]            = uboIncludeName;
  data["MainClassName"]      = a_classInfo.mainClassName;
  data["ShaderSingleFile"]   = a_classInfo.pShaderCC->ShaderSingleFile();
  data["ShaderGLSL"]         = a_classInfo.pShaderCC->IsGLSL();
  data["UseSeparateUBO"]     = a_classInfo.pShaderCC->UseSeparateUBOForArguments();
  data["UseSpecConstWgSize"] = a_classInfo.pShaderCC->UseSpecConstForWgSize();
  data["UseServiceMemCopy"]  = (a_classInfo.usedServiceCalls.find("memcpy") != a_classInfo.usedServiceCalls.end());

  data["PlainMembersUpdateFunctions"]  = "";
  data["VectorMembersUpdateFunctions"] = "";
  data["KernelsDecl"]                  = strOut2.str();   
  data["TotalDSNumber"]                = a_classInfo.allDescriptorSetsInfo.size();

  data["VectorMembers"]  = std::vector<std::string>();
  data["TextureMembers"] = std::vector<std::string>();
  for(const auto var : a_classInfo.dataMembers)
  {
    if(var.IsUsedTexture())
      data["TextureMembers"].push_back(var.name);
    else if(var.isContainer)
      data["VectorMembers"].push_back(var.name);
  }

  data["SamplerMembers"] = std::vector<std::string>();
  for(const auto& member : a_classInfo.allDataMembers)
  {
    if(member.second.type == "struct Sampler")
      data["SamplerMembers"].push_back(member.second.name);
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
    data["ClassVars"].push_back(local);
  }

  data["ClassVectorVars"]  = std::vector<std::string>();
  data["ClassTextureVars"] = std::vector<std::string>();
  for(const auto& v : a_classInfo.dataMembers)
  {
    if(!v.isContainer || v.usage != kslicer::DATA_USAGE::USAGE_USER)
      continue;
    
    if(v.IsUsedTexture())
    {
      json local;
      local["Name"]       = v.name;
      local["Format"]     = kslicer::InferenceVulkanTextureFormatFromTypeName(a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(v.containerDataType), a_classInfo.halfFloatTextures);
      local["Usage"]      = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT";
      local["NeedUpdate"] = false;

      if(v.tmask == TEX_ACCESS::TEX_ACCESS_SAMPLE)
      {
        local["Usage"]      = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT";
        local["NeedUpdate"] = true;
      }
      else if(v.tmask == TEX_ACCESS::TEX_ACCESS_READ)
      {
        local["Usage"]      = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT";
        local["NeedUpdate"] = true;
      }
      else if( int(v.tmask) == (int(TEX_ACCESS::TEX_ACCESS_READ) | int(TEX_ACCESS::TEX_ACCESS_SAMPLE)))
      {
        local["Usage"]      = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT";
        local["NeedUpdate"] = true;
      }
      else if( int(v.tmask) == (int(TEX_ACCESS::TEX_ACCESS_READ)   | int(TEX_ACCESS::TEX_ACCESS_WRITE)) || 
               int(v.tmask) == (int(TEX_ACCESS::TEX_ACCESS_SAMPLE) | int(TEX_ACCESS::TEX_ACCESS_WRITE)))
      {
        local["Usage"]      = "VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT";
        local["NeedUpdate"] = false;
      }

      local["Usage"]  = "VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT"; // TODO: inference usage (!!!)
      data["ClassTextureVars"].push_back(local);     
    }
    else
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
      data["ClassVectorVars"].push_back(local);     
    }
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

  std::vector<std::string> kernelsCallCmdDecl; 
  kernelsCallCmdDecl.reserve(a_classInfo.kernels.size());
  for(const auto& k : a_classInfo.kernels)
    kernelsCallCmdDecl.push_back(k.second.DeclCmd);  

  auto kernelDeclByName = MakeMapForKernelsDeclByName(kernelsCallCmdDecl);

  data["MultipleSourceShaders"] = !a_classInfo.pShaderCC->IsSingleSource();
  data["ShaderFolder"]          = a_classInfo.pShaderCC->ShaderFolder();
  
  auto dhierarchies             = a_classInfo.GetDispatchingHierarchies();
  data["DispatchHierarchies"]   = PutHierarchiesDataToJson(dhierarchies, compiler);
  
  data["IndirectBufferSize"] = a_classInfo.m_indirectBufferSize;
  data["IndirectDispatches"] = std::vector<std::string>();
  data["Kernels"]            = std::vector<std::string>();  

  for(const auto& nk : a_classInfo.kernels)
  {
    const auto& k        = nk.second;
    std::string kernName = a_classInfo.RemoveKernelPrefix(k.name);
    const auto auxArgs   = GetUserKernelArgs(k.args);
    
    std::string outFileName = k.name + "_UpdateIndirect" + ".cl.spv";
    std::string outFilePath = shaderPath + "/" + outFileName;

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
      if(p != a_classInfo.m_vhierarchy.end() && p->second.dispatchType == kslicer::VKERNEL_INDIRECT_DISPATCH)
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
      if(arg.isThreadID || arg.isLoopSize || arg.IsUser() ||     // exclude TID and loopSize args bindings
         pos1 != std::string::npos || pos2 != std::string::npos) // exclude special case of passing MainClass to virtual kernels
        continue;
      
      json argData;
      if(arg.IsTexture())
      {
        auto pAccessFlags = k.texAccessInArgs.find(arg.name);
        if(pAccessFlags->second == TEX_ACCESS::TEX_ACCESS_SAMPLE)
          argData["Type"] = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
        else
          argData["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
      }
      else
        argData["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";

      argData["Name"]  = arg.name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
      kernelJson["Args"].push_back(argData);
      actualSize++;
    }

    for(const auto& container : k.usedContainers) // TODO: add support fo textures (!!!)
    {
      json argData;

      if(container.second.isTexture)
      {
        auto pAccessFlags = k.texAccessInMemb.find(container.second.name);
        if(pAccessFlags == k.texAccessInMemb.end() || pAccessFlags->second == TEX_ACCESS::TEX_ACCESS_SAMPLE)
          argData["Type"] = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
        else
          argData["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
      }
      else
        argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";

      argData["Name"]  = container.second.name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
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
      threadIdNamesList[i] = tidArgs[tid].sizeName;
    }

    if(threadIdNamesList.size() > 0)
      kernelJson["tidX"] = threadIdNamesList[0];
    else
      kernelJson["tidX"] = 1;

    if(threadIdNamesList.size() > 1)
      kernelJson["tidY"] = threadIdNamesList[1];
    else
      kernelJson["tidY"] = 1;

    if(threadIdNamesList.size() > 2)
      kernelJson["tidZ"] = threadIdNamesList[2];
    else
      kernelJson["tidZ"] = 1;

    // put auxArgs to push constants
    //
    int sizeCurr = 0;
    kernelJson["AuxArgs"] = std::vector<std::string>();
    for(auto arg : auxArgs)
    {
      std::string typestr = arg.type;
      ReplaceFirst(typestr, "const ", "");
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
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  data["MainFunctions"] = std::vector<std::string>();
  for(const auto& mainFunc : a_methodsToGenerate)
  {
    json data2;
    data2["Name"]                 = mainFunc.Name;
    data2["DescriptorSets"]       = std::vector<std::string>();
    
    // for decl
    //
    data2["Decl"]                 = mainFunc.GeneratedDecl;
    data2["LocalVarsBuffersDecl"] = std::vector<std::string>();
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
      json controlArg;
      controlArg["Name"]      = v.name;
      controlArg["IsTexture"] = v.isTexture;
      data2["InOutVars"].push_back(controlArg);
    }

    // for impl, ds bindings
    //
    for(size_t i=mainFunc.startDSNumber; i<mainFunc.endDSNumber; i++)
    {
      auto& dsArgs               = a_classInfo.allDescriptorSetsInfo[i];
      const auto pFoundKernel    = a_classInfo.kernels.find(dsArgs.originKernelName);
      const bool handMadeKernels = (pFoundKernel == a_classInfo.kernels.end());

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
        if(!handMadeKernels && (pFoundKernel->second.args[j].isThreadID || pFoundKernel->second.args[j].isLoopSize || pFoundKernel->second.args[j].IsUser() ||
                                dsArgs.descriptorSetsInfo[j].varName == "this")) // if this pointer passed to kernel (used for virtual kernels), ignore it because it passe there anyway
          continue;

        const std::string dsArgName = kslicer::GetDSArgName(mainFunc.Name, dsArgs.descriptorSetsInfo[j]);

        json arg;
        arg["Id"]        = realId;
        arg["Name"]      = dsArgName;
        arg["IsTexture"] = dsArgs.descriptorSetsInfo[j].isTexture;
        if(dsArgs.descriptorSetsInfo[j].isTexture)
        {
          auto texDSInfo = ObtainDSTextureAccessArg(pFoundKernel->second, j, dsArgs.descriptorSetsInfo[j].isConst);
          arg["AccessLayout"] = texDSInfo.accessLayout;
          arg["AccessDSType"] = texDSInfo.accessDSType;
          arg["SamplerName"]  = texDSInfo.SamplerName;
        }
        local["Args"].push_back(arg);
        local["ArgNames"].push_back(dsArgs.descriptorSetsInfo[j].varName);
        realId++;
      }
      
      if(pFoundKernel != a_classInfo.kernels.end())
      {
        for(const auto& container : pFoundKernel->second.usedContainers) // add all class-member vectors bindings
        {
          json arg;
          arg["Id"]        = realId;
          arg["Name"]      = "m_vdata." + container.second.name;
          arg["IsTexture"] = container.second.isTexture;
          if(container.second.isTexture)
          {
            auto texDSInfo = ObtainDSTextureAccessMemb(pFoundKernel->second, container.second.name, container.second.isConst);
            arg["AccessLayout"] = texDSInfo.accessLayout;
            arg["AccessDSType"] = texDSInfo.accessDSType;
            arg["SamplerName"]  = texDSInfo.SamplerName;
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

    // for impl, other
    //
    data2["MainFuncCmd"] = mainFunc.CodeGenerated;

    data["MainFunctions"].push_back(data2);
  }

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

  return data;
}

void kslicer::ApplyJsonToTemplate(const std::string& a_declTemplateFilePath, const std::string& a_outFilePath, const nlohmann::json& a_data)
{
  inja::Environment env;
  env.set_trim_blocks(true);
  env.set_lstrip_blocks(true);

  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, a_data);
  
  std::ofstream fout(a_outFilePath);
  fout << result.c_str() << std::endl;
  fout.close();
}

namespace kslicer
{
  std::string GetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, 
                                      const std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair>& threadIds,
                                      const std::string a_names[3]);
};
bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);


nlohmann::json kslicer::PrepareUBOJson(MainClassInfo& a_classInfo, const std::vector<kslicer::DataMemberInfo>& a_dataMembers, const clang::CompilerInstance& compiler)
{
  nlohmann::json data;
  auto pShaderRewriter = a_classInfo.pShaderFuncRewriter;

  auto podMembers = filter(a_classInfo.dataMembers, [](auto& memb) { return !memb.isContainer; });
  uint32_t dummyCounter = 0;
  data["MainClassName"]   = a_classInfo.mainClassName;
  data["UBOStructFields"] = std::vector<std::string>();
  data["ShaderGLSL"]      = !a_classInfo.pShaderCC->IsSingleSource();

  for(auto member : podMembers)
  {
    std::string typeStr = member.type;
    if(member.isArray)
      typeStr = typeStr.substr(0, typeStr.find("["));
    typeStr = pShaderRewriter->RewriteStdVectorTypeStr(typeStr); 

    size_t sizeO = member.sizeInBytes;
    size_t sizeA = member.alignedSizeInBytes;

    json uboField;
    uboField["Type"]      = typeStr;
    uboField["Name"]      = member.name;
    uboField["IsArray"]   = member.isArray;
    uboField["ArraySize"] = member.arraySize;
    data["UBOStructFields"].push_back(uboField);
    
    while(sizeO < sizeA)
    {
      std::stringstream strOut;
      strOut << "dummy" << dummyCounter;
      dummyCounter++;
      sizeO += sizeof(uint32_t);
      uboField["Type"] = "uint";
      uboField["Name"] = strOut.str();
      data["UBOStructFields"].push_back(uboField);
    }

    assert(sizeO == sizeA);
   
   data["Hierarchies"] = PutHierarchiesDataToJson(a_classInfo.GetDispatchingHierarchies(), compiler);
  }

  return data;
}

static json ReductionAccessFill(const kslicer::KernelInfo::ReductionAccess& second, std::shared_ptr<kslicer::IShaderCompiler> pShaderCC)
{
  json varJ;
  varJ["Type"]          = second.dataType;
  varJ["Name"]          = second.leftExpr;
  varJ["Init"]          = second.GetInitialValue();
  varJ["Op"]            = second.GetOp(pShaderCC);
  varJ["NegLastStep"]   = (second.type == kslicer::KernelInfo::REDUCTION_TYPE::SUB || second.type == kslicer::KernelInfo::REDUCTION_TYPE::SUB_ONE);
  varJ["BinFuncForm"]   = (second.type == kslicer::KernelInfo::REDUCTION_TYPE::FUNC);
  varJ["OutTempName"]   = second.tmpVarName;
  varJ["SupportAtomic"] = second.SupportAtomicLastStep();
  varJ["AtomicOp"]      = second.GetAtomicImplCode();
  varJ["IsArray"]       = second.leftIsArray;
  varJ["ArraySize"]     = second.arraySize;
  if(second.leftIsArray)
  {
    varJ["Name"]         = second.arrayName;
    varJ["OutTempNameA"] = second.arrayTmpBufferNames;
  }
  return varJ;
}


json kslicer::PrepareJsonForKernels(MainClassInfo& a_classInfo, 
                                    const std::vector<kslicer::FuncData>& usedFunctions,
                                    const std::vector<kslicer::DeclInClass>& usedDecl,
                                    const clang::CompilerInstance& compiler,
                                    const uint32_t  threadsOrder[3],
                                    const std::string& uboIncludeName, const nlohmann::json& uboJson)
{
  auto pShaderRewriter = a_classInfo.pShaderFuncRewriter;

  std::unordered_map<std::string, DataMemberInfo> dataMembersCached;
  dataMembersCached.reserve(a_classInfo.dataMembers.size());
  for(const auto& member : a_classInfo.dataMembers)
    dataMembersCached[member.name] = member;

  json data;
  data["MainClassName"]      = a_classInfo.mainClassName;
  data["UseSpecConstWgSize"] = a_classInfo.pShaderCC->UseSpecConstForWgSize();
  data["UseServiceMemCopy"]  = (a_classInfo.usedServiceCalls.find("memcpy") != a_classInfo.usedServiceCalls.end());

  // (1) put includes
  //
  data["Includes"] = std::vector<std::string>();
  for(auto keyVal : a_classInfo.allIncludeFiles) // we will search for only used include files among all of them (quoted, angled were excluded earlier)
  { 
    bool found = false;
    for(auto f : a_classInfo.includeToShadersFolders)  // exclude everything from "shader" folders
    {
      if(keyVal.first.find(f) != std::string::npos)
      {
        found = true;
        break;
      }
    }
    if(!found)
      continue;
   
    if(a_classInfo.mainClassFileInclude.find(keyVal.first) == std::string::npos)
      data["Includes"].push_back(keyVal.first);
  }
  data["UBOIncl"] = uboIncludeName;

  // (2) declarations of struct, constants and typedefs inside class
  //
  data["ClassDecls"] = std::vector<std::string>();
  for(const auto decl : usedDecl)
  {
    if(!decl.extracted)
      continue;

    data["ClassDecls"].push_back(a_classInfo.pShaderCC->PrintHeaderDecl(decl,compiler));
  }

  // (3) local functions
  //
  ShaderFeatures shaderFeatures;
  data["LocalFunctions"] = std::vector<std::string>();
  std::vector<kslicer::FuncData> funcMembers;
  {
    clang::Rewriter rewrite2;
    rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
    auto pVisitorF = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);

    for (const auto& f : usedFunctions) 
    { 
      if(a_classInfo.IsExcludedLocalFunction(f.name)) // check exclude list here, don't put such functions in cl file
        continue;

      if(!f.isMember)
      {
        pVisitorF->TraverseDecl(const_cast<clang::FunctionDecl*>(f.astNode));
        data["LocalFunctions"].push_back(rewrite2.getRewrittenText(f.srcRange));
        shaderFeatures = shaderFeatures || pVisitorF->GetShaderFeatures();
      }
      else
        funcMembers.push_back(f);
    }
  }

  if(a_classInfo.NeedFakeOffset())
  {
    data["LocalFunctions"].push_back("uint fakeOffset(uint x, uint y, uint pitch) { return y*pitch + x; }  // RTV pattern, for 2D threading"); // todo: ckeck if RTV pattern is used here!
    //data["LocalFunctions"].push_back("uint fakeOffset3(uint x, uint y, uint z, uint sizeY, uint sizeX) { return z*sizeY*sizeX + y*sizeX + x; } // for 3D threading");
  }

  data["GlobalUseInt8"]  = shaderFeatures.useByteType;
  data["GlobalUseInt16"] = shaderFeatures.useShortType;
  data["GlobalUseInt64"] = shaderFeatures.useInt64Type;

  auto dhierarchies   = a_classInfo.GetDispatchingHierarchies();
  data["Hierarchies"] = PutHierarchiesDataToJson(dhierarchies, compiler);

  // (4) put kernels
  //
  clang::SourceManager& sm = compiler.getSourceManager();
  data["Kernels"] = std::vector<std::string>();
  
  for (const auto& nk : a_classInfo.kernels)  
  {
    const auto& k = nk.second;
    std::cout << "  processing " << k.name << std::endl;
    
    auto commonArgs = a_classInfo.GetKernelCommonArgs(k);
    auto tidArgs    = a_classInfo.GetKernelTIDArgs(k);
    
    uint VArgsSize = 0;

    json args = std::vector<std::string>();
    for(auto commonArg : commonArgs)
    {
      json argj;
      std::string buffType1 = a_classInfo.pShaderCC->ProcessBufferType(commonArg.typeName);
      std::string buffType2 = pShaderRewriter->RewriteStdVectorTypeStr(buffType1); 
      argj["Type"]     = commonArg.isImage ? commonArg.imageType : buffType2;
      argj["Name"]     = commonArg.argName;
      argj["IsUBO"]    = commonArg.isUBO;
      argj["IsImage"]  = commonArg.isImage;
      argj["NeedFmt"]  = !commonArg.isSampler;
      argj["ImFormat"] = commonArg.imageFormat;

      args.push_back(argj);
      if(!commonArg.isThreadFlags)
        VArgsSize++;
    }

    assert(tidArgs.size() <= 3);

    // now add all std::vector members
    //
    json vecs = std::vector<std::string>();
    for(const auto& container : k.usedContainers)
    {
      auto pVecMember     = dataMembersCached.find(container.second.name);
      auto pVecSizeMember = dataMembersCached.find(container.second.name + "_size");

      assert(pVecMember     != dataMembersCached.end());
      assert(pVecMember->second.isContainer);

      std::string buffType1 = a_classInfo.pShaderCC->ProcessBufferType(pVecMember->second.containerDataType);
      std::string buffType2 = pShaderRewriter->RewriteStdVectorTypeStr(buffType1);
      if(!a_classInfo.pShaderCC->IsGLSL())
        buffType2 += "*";
      
      json argj;
      argj["Type"]       = buffType2;
      argj["Name"]       = pVecMember->second.name;
      argj["IsUBO"]      = false;
      argj["IsImage"]    = false;
      if(pVecMember->second.isContainer && kslicer::IsTextureContainer(pVecMember->second.containerType))
      {
        std::string imageFormat;
        auto pMemberAccess = k.texAccessInMemb.find(pVecMember->second.name); 
        auto accessFlags   = (pMemberAccess == k.texAccessInMemb.end()) ? kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE : pMemberAccess->second; //pVecMember->second.tmask; 
        argj["IsImage"]  = true;
        argj["Type"]     = a_classInfo.pShaderFuncRewriter->RewriteImageType(pVecMember->second.containerType, pVecMember->second.containerDataType, accessFlags, imageFormat);
        argj["NeedFmt"]  = (accessFlags != kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE);
        argj["ImFormat"] = imageFormat;
        argj["SizeOffset"] = 0;
      }
      else
        argj["SizeOffset"] = pVecSizeMember->second.offsetInTargetBuffer / sizeof(uint32_t);
      
      args.push_back(argj);
      vecs.push_back(argj);
    }

    if(k.isMaker) // add to kernel ObjPtr buffer
    {
      json argj; 
      argj["Type"]       = "uint2       *";
      argj["Name"]       = "kgen_objPtrData";
      argj["IsUBO"]      = false;
      args.push_back(argj);
    }
    
    std::vector<kslicer::DataMemberInfo> membersToRead;
    for(const auto& name : k.usedMembers)
    {
      auto pCachedMember = dataMembersCached.find(name);
      if(pCachedMember != dataMembersCached.end())
        membersToRead.push_back(pCachedMember->second);
    }

    std::sort(membersToRead.begin(), membersToRead.end(), [](const auto& a, const auto& b) { return a.offsetInTargetBuffer < b.offsetInTargetBuffer; });

    json members = std::vector<std::string>();
    for(const auto member : membersToRead)
    {
      if(member.isArray || member.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD) // read large data structures directly inside kernel code, don't read them at the beggining of kernel.
        continue;

      if(k.subjectedToReduction.find(member.name) != k.subjectedToReduction.end())  // exclude this opt for members which subjected to reduction
        continue;

      json memberData;
      memberData["Type"]   = pShaderRewriter->RewriteStdVectorTypeStr(member.type);
      memberData["Name"]   = member.name;
      memberData["Offset"] = member.offsetInTargetBuffer / sizeof(uint32_t);
      members.push_back(memberData);
    }

    const auto userArgsArr = GetUserKernelArgs(k.args);
    json userArgs = std::vector<std::string>();
    for(const auto& arg : userArgsArr)
    {
      json argj;
      argj["Type"]  = pShaderRewriter->RewriteStdVectorTypeStr(arg.type);
      argj["Name"]  = arg.name;
      argj["IsUBO"] = false;
      userArgs.push_back(argj);
    }
    
    // extract all arrays access in seperate map
    //
    std::unordered_map<std::string, KernelInfo::ReductionAccess> subjToRedCopy; subjToRedCopy.reserve(k.subjectedToReduction.size());
    std::unordered_map<std::string, KernelInfo::ReductionAccess> subjToRedArray;
    for(const auto& var : k.subjectedToReduction)
    {
      if(!var.second.leftIsArray)
      {
        subjToRedCopy[var.first] = var.second;
        continue;
      }
      
      auto p = subjToRedArray.find(var.second.arrayName);
      if(p != subjToRedArray.end())
        p->second.arrayTmpBufferNames.push_back(var.second.tmpVarName);
      else
      {
        subjToRedArray[var.second.arrayName] = var.second;
        subjToRedArray[var.second.arrayName].arrayTmpBufferNames.push_back(var.second.tmpVarName);
      }
    } 

    bool needFinishReductionPass = false;
    json reductionVars = std::vector<std::string>();
    json reductionArrs = std::vector<std::string>();
    for(const auto& var : subjToRedCopy)
    {
      json varJ = ReductionAccessFill(var.second, a_classInfo.pShaderCC);
      needFinishReductionPass = needFinishReductionPass || !varJ["SupportAtomic"];
      reductionVars.push_back(varJ);
    }

    for(const auto& var : subjToRedArray)
    {
      json varJ = ReductionAccessFill(var.second, a_classInfo.pShaderCC);
      needFinishReductionPass = needFinishReductionPass || !varJ["SupportAtomic"];
      reductionArrs.push_back(varJ);
    }
    
    json kernelJson;
    kernelJson["RedLoop1"] = std::vector<std::string>();
    kernelJson["RedLoop2"] = std::vector<std::string>();

    const uint32_t blockSize = k.wgSize[0]*k.wgSize[1]*k.wgSize[2];
    for (uint c = blockSize/2; c>k.warpSize; c/=2)
      kernelJson["RedLoop1"].push_back(c);
    for (uint c = k.warpSize; c>0; c/=2)
      kernelJson["RedLoop2"].push_back(c);
    
    kernelJson["LastArgNF"]  = VArgsSize; // Last Argument No Flags
    kernelJson["Args"]       = args;
    kernelJson["Vecs"]       = vecs;
    kernelJson["UserArgs"]   = userArgs;
    kernelJson["Members"]    = members;
    kernelJson["Name"]       = k.name;
    kernelJson["UBOBinding"] = args.size(); // for circle
    kernelJson["HasEpilog"]  = k.isBoolTyped || reductionVars.size() != 0 || reductionArrs.size() != 0 || k.isMaker;
    kernelJson["IsBoolean"]  = k.isBoolTyped;
    kernelJson["IsMaker"]    = k.isMaker;
    kernelJson["IsVirtual"]  = k.isVirtual;
    kernelJson["SubjToRed"]  = reductionVars;
    kernelJson["ArrsToRed"]  = reductionArrs;
    kernelJson["FinishRed"]  = needFinishReductionPass;

    std::string sourceCodeCut = k.rewrittenText.substr(k.rewrittenText.find_first_of('{')+1);
    kernelJson["Source"]      = sourceCodeCut.substr(0, sourceCodeCut.find_last_of('}'));
    
    //////////////////////////////////////////////////////////////////////////////////////////
  
    {
      kernelJson["ThreadIds"] = std::vector<std::string>();

      std::vector<std::string> threadIdNames(tidArgs.size());
      for(size_t i=0;i<tidArgs.size();i++)
      {
        uint32_t tid = std::min<uint32_t>(threadsOrder[i], tidArgs.size()-1);
        threadIdNames[i] = tidArgs[tid].argName;

        json threadId;
        threadId["Name"] = tidArgs[tid].argName;
        threadId["Type"] = tidArgs[tid].typeName;
        kernelJson["ThreadIds"].push_back(threadId);
      }

      kernelJson["threadDim"]   = tidArgs.size();
      kernelJson["threadNames"] = threadIdNames;
      if(threadIdNames.size() >= 1)
        kernelJson["threadName1"] = threadIdNames[0];
      if(threadIdNames.size() >= 2)
        kernelJson["threadName2"] = threadIdNames[1];
      if(threadIdNames.size() == 3)
        kernelJson["threadName3"] = threadIdNames[2];
    }

    std::string tidNames[3] = {"kgen_iNumElementsX", "kgen_iNumElementsY", "kgen_iNumElementsZ"};
 
    if(k.loopIters.size() != 0) 
    {
      for(const auto& iter : k.loopIters)
      {
        uint32_t loopIdReorderd  = threadsOrder[iter.loopNesting];
        tidNames[loopIdReorderd] = iter.sizeExpr;                   // #TODO: assert that this expression does not contain .size(); if it does
      }                                                             // we must change it to 'vec_size2' for example 
    }

    kernelJson["threadIdName1"] = tidNames[0]; 
    kernelJson["threadIdName2"] = tidNames[1]; 
    kernelJson["threadIdName3"] = tidNames[2]; 

    kernelJson["WGSizeX"]       = k.wgSize[0]; //
    kernelJson["WGSizeY"]       = k.wgSize[1]; // 
    kernelJson["WGSizeZ"]       = k.wgSize[2]; // 

    //////////////////////////////////////////////////////////////////////////////////////////
    std::string names[3];
    a_classInfo.pShaderCC->GetThreadSizeNames(names);

    kernelJson["shouldCheckExitFlag"] = k.checkThreadFlags;
    kernelJson["checkFlagsExpr"]      = "//xxx//";
    kernelJson["ThreadOffset"]        = kslicer::GetFakeOffsetExpression(k, a_classInfo.GetKernelTIDArgs(k), names);
    kernelJson["InitKPass"]           = false;
    kernelJson["IsIndirect"]          = k.isIndirect;
    if(k.isIndirect)
    {
      kernelJson["IndirectSizeX"]  = "0";
      kernelJson["IndirectSizeY"]  = "0";
      kernelJson["IndirectSizeZ"]  = "0";
      
      if(k.loopIters.size() > 0)
      {
        std::string exprContent     = kslicer::ReplaceSizeCapacityExpr(k.loopIters[0].sizeExpr);
        kernelJson["IndirectSizeX"] = a_classInfo.pShaderCC->UBOAccess(exprContent); 
      }

      if(k.loopIters.size() > 1)
      {
        std::string exprContent     = kslicer::ReplaceSizeCapacityExpr(k.loopIters[1].sizeExpr);
        kernelJson["IndirectSizeY"] = a_classInfo.pShaderCC->UBOAccess(exprContent); 
      }

      if(k.loopIters.size() > 2)
      {
        std::string exprContent     = kslicer::ReplaceSizeCapacityExpr(k.loopIters[2].sizeExpr);
        kernelJson["IndirectSizeZ"] = a_classInfo.pShaderCC->UBOAccess(exprContent); 
      }

      kernelJson["IndirectOffset"] = k.indirectBlockOffset; 
      kernelJson["threadIdName1"]  = "kgen_iNumElementsX";
      kernelJson["threadIdName2"]  = "kgen_iNumElementsY";
      kernelJson["threadIdName3"]  = "kgen_iNumElementsZ";
    }
    else
    {
      kernelJson["IndirectSizeX"] = tidNames[0]; 
      kernelJson["IndirectSizeY"] = tidNames[1]; 
      kernelJson["IndirectSizeZ"] = tidNames[2]; 
    }
    
    if(k.isVirtual || k.isMaker)
    {
      json hierarchy = PutHierarchyToJson(dhierarchies[k.interfaceName], compiler);
      hierarchy["RedLoop1"] = std::vector<std::string>();
      hierarchy["RedLoop2"] = std::vector<std::string>();
      const uint32_t blockSize = k.wgSize[0]*k.wgSize[1]*k.wgSize[2];
      for (uint c = blockSize/2; c>k.warpSize; c/=2)
        hierarchy["RedLoop1"].push_back(c);
      for (uint c = k.warpSize; c>0; c/=2)
        hierarchy["RedLoop2"].push_back(c);
      kernelJson["Hierarchy"] = hierarchy; 
      
      bool isConstObj = false;
      if(k.isVirtual)
      {
        for(const auto& impl : dhierarchies[k.interfaceName].implementations) // TODO: refactor this code to function
        {
          for(const auto& member : impl.memberFunctions)
          {
            if(member.name == k.name)
            {
              isConstObj = member.isConstMember || isConstObj;
              if(isConstObj)
                break;
            }
          }
          break; // it is enough to process only one of impl, because function interface is the same for all of them
        }
      }
      kernelJson["IsConstObj"] = isConstObj;
    }
    else
    {
      json temp;
      temp["IndirectDispatch"] = false; // because of 'Kernel.Hierarchy.IndirectDispatch' check could happen
      kernelJson["Hierarchy"]  = temp;
      kernelJson["IsConstObj"] = false;
    }

    kernelJson["MemberFunctions"] = std::vector<std::string>();
    if(funcMembers.size() > 0)
    {
      clang::Rewriter rewrite2;
      rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
      auto pVisitorF = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);
      auto pVisitorK = a_classInfo.pShaderCC->MakeKernRewriter(rewrite2, compiler, &a_classInfo, const_cast<kslicer::KernelInfo&>(k), "", false);
      pVisitorK->ClearUserArgs();
    
      for(auto& f : funcMembers)
      {
        auto funcNode = const_cast<clang::FunctionDecl*>(f.astNode);
        const std::string funcDeclText = pVisitorF->RewriteFuncDecl(funcNode);
        const std::string funcBodyText = pVisitorK->RecursiveRewrite(funcNode->getBody());
        kernelJson["MemberFunctions"].push_back(funcDeclText + funcBodyText);
      }
    }

    auto original = kernelJson;
    
    // if we have additional init statements we should add additional init kernel before our kernel
    //
    if(k.hasInitPass)
    {      
      kernelJson["Name"]      = k.name + "_Init";
      kernelJson["Source"]    = k.rewrittenInit.substr(k.rewrittenInit.find_first_of('{')+1);
      kernelJson["Members"]   = std::vector<json>();
      kernelJson["HasEpilog"] = false;
      kernelJson["FinishRed"] = false;
      kernelJson["InitKPass"] = true;
      kernelJson["WGSizeX"]   = 1;
      kernelJson["WGSizeY"]   = 1;
      kernelJson["WGSizeZ"]   = 1;
      data["Kernels"].push_back(kernelJson);
    }

    data["Kernels"].push_back(original);

    if(k.hasFinishPassSelf)
    {
      kernelJson["Name"]      = k.name + "_Finish";
      kernelJson["Source"]    = k.rewrittenFinish;
      kernelJson["Members"]   = std::vector<json>();
      kernelJson["HasEpilog"] = false;
      kernelJson["FinishRed"] = false;
      kernelJson["InitKPass"] = true;
      kernelJson["WGSizeX"]   = 1;
      kernelJson["WGSizeY"]   = 1;
      kernelJson["WGSizeZ"]   = 1;
      data["Kernels"].push_back(kernelJson);
    }
  
  }

  return data;
}

std::string kslicer::ReplaceSizeCapacityExpr(const std::string& a_str)
{
  const auto posOfPoint = a_str.find(".");
  if(posOfPoint != std::string::npos)
  {
    const std::string memberNameA = a_str.substr(0, posOfPoint);
    const std::string fname       = a_str.substr(posOfPoint+1);
    return memberNameA + "_" + fname.substr(0, fname.find("("));
  }
  else
    return a_str;
}

