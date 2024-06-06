#include "kslicer.h"
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

#ifdef _WIN32
typedef unsigned int uint;
#endif

// Just for convenience
using namespace inja;
using json = nlohmann::json;

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

std::string kslicer::GetDSArgName(const std::string& a_mainFuncName, const kslicer::ArgReferenceOnCall& a_arg, bool a_megakernel)
{
  switch(a_arg.argType)
  {
    case  kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG:
    return a_mainFuncName + "_local." + a_arg.name;

    case  kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL:
    case  kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_VECTOR:
    {
      auto posOfData = a_arg.name.find(".data()");
      if(posOfData != std::string::npos)
        return std::string("m_vdata.") + a_arg.name.substr(0, posOfData);
      else if(a_arg.kind == DATA_KIND::KIND_ACCEL_STRUCT)
        return a_arg.name;
      else if(a_megakernel)
        return std::string("m_vdata.") + a_arg.name;
      else
        return a_mainFuncName + "_local." + a_arg.name;
    }
    break;

    case kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_SERVICE_DATA:
    return a_arg.name;
    break;

    default:
    return std::string("m_vdata.") + a_arg.name;
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

std::vector<kslicer::KernelInfo::ArgInfo> kslicer::GetUserKernelArgs(const std::vector<kslicer::KernelInfo::ArgInfo>& a_allArgs)
{
  std::vector<kslicer::KernelInfo::ArgInfo> result;
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

nlohmann::json kslicer::PutHierarchyToJson(const kslicer::MainClassInfo::DHierarchy& h, 
                                           const clang::CompilerInstance& compiler,
                                           const MainClassInfo& a_classInfo)
{
  json hierarchy;
  hierarchy["Name"]             = h.interfaceName;
  hierarchy["ObjBufferName"]    = h.objBufferName;
  hierarchy["IndirectDispatch"] = 0;
  hierarchy["IndirectOffset"]   = 0;
  
  hierarchy["InterfaceFields"]  = std::vector<json>();
  auto fields = a_classInfo.GetFieldsFromStruct(h.interfaceDecl);
  for(auto field : fields) 
  {
    json local;
    local["Type"] = field.first;
    local["Name"] = field.second;
    hierarchy["InterfaceFields"].push_back(local);
  }


  hierarchy["Constants"] = std::vector<json>();
  for(const auto& decl : h.usedDecls)
  {
    if(decl.kind == kslicer::DECL_IN_CLASS::DECL_CONSTANT)
    {
      json currConstant;
      currConstant["Type"]  = a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(decl.type);
      currConstant["Name"]  = decl.name;
      currConstant["Value"] = kslicer::GetRangeSourceCode(decl.srcRange, compiler);
      hierarchy["Constants"].push_back(currConstant);
    }
  }
  
  bool emptyIsFound = false;
  hierarchy["Implementations"] = std::vector<json>();
  for(const auto& impl : h.implementations)
  {  
    const auto p2 = h.tagByClassName.find(impl.name);
    assert(p2 != h.tagByClassName.end());
    json currImpl;
    currImpl["ClassName"] = impl.name;
    currImpl["TagName"]   = p2->second;
    currImpl["MemberFunctions"] = std::vector<json>();
    currImpl["ObjBufferName"]   = h.objBufferName;
    for(const auto& member : impl.memberFunctions)
    {
      currImpl["MemberFunctions"].push_back(member.srcRewritten);
    }
    currImpl["Fields"] = std::vector<json>();
    for(const auto& field : impl.fields)
      currImpl["Fields"].push_back(field);
   
    if(impl.isEmpty) {
      hierarchy["EmptyImplementation"] = currImpl;
      emptyIsFound = true;
    }
    else
      hierarchy["Implementations"].push_back(currImpl);
  }
  hierarchy["ImplAlignedSize"] = AlignedSize(h.implementations.size()+1);
  if(h.implementations.size()!= 0 && !emptyIsFound)
    std::cout << "  VFH::ALERT! Empty implementation is not found! Don't add any functions except 'GetTag()' to EmptyImpl class" << std::endl;

  hierarchy["VirtualFunctions"] = std::vector<json>();
  for(const auto& vf : h.virtualFunctions)
  {
    json virtualFunc;
    virtualFunc["Name"] = vf.second.name;
    virtualFunc["Decl"] = vf.second.declRewritten;
    virtualFunc["Args"] = std::vector<json>();
    {
      json argJ;
      argJ["Type"] = "uint";
      argJ["Name"] = "selfId";
      virtualFunc["Args"].push_back(argJ);
    }
    for(const auto arg : vf.second.args) {
      json argJ;
      argJ["Type"] = arg.first;
      argJ["Name"] = arg.second;
      virtualFunc["Args"].push_back(argJ);
    }
    virtualFunc["ArgLen"] = vf.second.args.size();
    //virtualFunc["ThisTypeName"]  = vf.second.thisTypeName;
    hierarchy["VirtualFunctions"].push_back(virtualFunc);
  }

  return hierarchy;
}

nlohmann::json kslicer::PutHierarchiesDataToJson(const std::unordered_map<std::string, kslicer::MainClassInfo::DHierarchy>& hierarchies,
                                                 const clang::CompilerInstance& compiler,
                                                 const MainClassInfo& a_classInfo)
{
  json data = std::vector<json>();
  for(const auto& p : hierarchies)
    data.push_back(PutHierarchyToJson(p.second, compiler, a_classInfo));
  return data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void kslicer::ApplyJsonToTemplate(const std::filesystem::path& a_declTemplateFilePath, const std::filesystem::path& a_outFilePath, const nlohmann::json& a_data)
{
  inja::Environment env;
  env.set_trim_blocks(true);
  env.set_lstrip_blocks(true);

  const std::string declTemplateFilePath = a_declTemplateFilePath.u8string();
  inja::Template temp = env.parse_template(declTemplateFilePath.c_str());
  std::string result  = env.render(temp, a_data);

  std::ofstream fout(a_outFilePath);
  fout << result.c_str() << std::endl;
  fout.close();
}

namespace kslicer
{
  std::string GetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo,
                                      const std::vector<kslicer::ArgFinal>& threadIds,
                                      const std::string a_names[3]);
}

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);

static json ReductionAccessFill(const kslicer::KernelInfo::ReductionAccess& second, std::shared_ptr<kslicer::IShaderCompiler> pShaderCC, std::shared_ptr<kslicer::FunctionRewriter> pShaderFuncRewriter)
{
  const std::string rewrtittenType = pShaderFuncRewriter->RewriteStdVectorTypeStr(second.dataType);
  json varJ;
  varJ["Type"]          = rewrtittenType;
  varJ["Name"]          = second.leftExpr;
  varJ["Init"]          = second.GetInitialValue(pShaderCC->IsGLSL(), rewrtittenType);
  varJ["Op"]            = second.GetOp(pShaderCC);
  varJ["Op2"]           = second.GetOp2(pShaderCC);
  varJ["NegLastStep"]   = (second.type == kslicer::KernelInfo::REDUCTION_TYPE::SUB || second.type == kslicer::KernelInfo::REDUCTION_TYPE::SUB_ONE);
  varJ["BinFuncForm"]   = (second.type == kslicer::KernelInfo::REDUCTION_TYPE::FUNC);
  varJ["OutTempName"]   = second.tmpVarName;
  varJ["SupportAtomic"] = second.SupportAtomicLastStep();
  varJ["AtomicOp"]      = second.GetAtomicImplCode(pShaderCC->IsGLSL());
  varJ["SubgroupOp"]    = second.GetSubgroupOpCode(pShaderCC->IsGLSL());
  //varJ["UseSubgroups"]  = second.useSubGroups;
  varJ["IsArray"]       = second.leftIsArray;
  varJ["ArraySize"]     = second.arraySize;
  if(second.leftIsArray)
  {
    varJ["Name"]         = second.arrayName;
    varJ["OutTempNameA"] = second.arrayTmpBufferNames;
  }
  return varJ;
}

std::unordered_map<std::string, std::string> ListISPCVectorReplacements();

const std::string ConvertVecTypesToISPC(const std::string& a_typeName,
                                        const std::string& a_argName);

static bool isConvertibleToInt(const std::string& str) {
    bool result = false;
    try {
        // Attempt to convert the string to
        // an integer using std::stoi
        std::stoi(str);
        // Conversion successful,
        // string can be converted to an integer
        result = true;
    } catch (...) {
        // Conversion failed, string cannot
        // be converted to an integer
    }
    return result;
}


json kslicer::PrepareJsonForKernels(MainClassInfo& a_classInfo,
                                    const std::vector<kslicer::FuncData>& usedFunctions,
                                    const std::vector<kslicer::DeclInClass>& usedDecl,
                                    const clang::CompilerInstance& compiler,
                                    const uint32_t  threadsOrder[3],
                                    const std::string& uboIncludeName,
                                    const nlohmann::json& uboJson,
                                    const std::vector<std::string>& usedDefines,
                                    const TextGenSettings& a_settings)
{
  auto pShaderRewriter = a_classInfo.pShaderFuncRewriter;

  std::unordered_map<std::string, DataMemberInfo> dataMembersCached;
  dataMembersCached.reserve(a_classInfo.dataMembers.size());
  for(const auto& member : a_classInfo.dataMembers)
    dataMembersCached[member.name] = member;

  std::unordered_map<std::string, kslicer::ShittyFunction> shittyFunctions;
  if(a_classInfo.pShaderCC->IsGLSL())
  {
    for(const auto& k : a_classInfo.kernels)
    {
      for(auto f : k.second.shittyFunctions)
        shittyFunctions[f.originalName] = f;
    }
  }

  json data;
  data["MainClassName"]   = a_classInfo.mainClassName;
  data["MainClassSuffix"] = a_classInfo.mainClassSuffix;
  data["MainClassSuffixLowerCase"] = ToLowerCase(a_classInfo.mainClassSuffix);
  data["UseSpecConstWgSize"] = a_classInfo.pShaderCC->UseSpecConstForWgSize();

  data["UseServiceMemCopy"]  = (a_classInfo.usedServiceCalls.find("memcpy") != a_classInfo.usedServiceCalls.end());
  data["UseServiceScan"]     = (a_classInfo.usedServiceCalls.find("exclusive_scan") != a_classInfo.usedServiceCalls.end()) || (a_classInfo.usedServiceCalls.find("inclusive_scan") != a_classInfo.usedServiceCalls.end());
  data["UseServiceSort"]     = (a_classInfo.usedServiceCalls.find("sort") != a_classInfo.usedServiceCalls.end());
  data["UseMatMult"]         = (a_classInfo.usedServiceCalls.find("MatMulTranspose") != a_classInfo.usedServiceCalls.end());
  data["UseComplex"]         = true; // a_classInfo.useComplexNumbers; does not works in appropriate way ...
  data["UseRayGen"]          = a_settings.enableRayGen;
  data["UseMotionBlur"]      = a_settings.enableMotionBlur;

  data["Defines"] = std::vector<std::string>();
  for(const auto& def : usedDefines)
    data["Defines"].push_back(def);

  // (1) put includes
  //
  data["Includes"] = std::vector<std::string>();
  for(auto keyVal : a_classInfo.allIncludeFiles) // we will search for only used include files among all of them (quoted, angled were excluded earlier)
  {
    if(!a_classInfo.IsInExcludedFolder(keyVal.first))
      continue;

    if(a_classInfo.mainClassFileInclude.find(keyVal.first) == std::string::npos)
      data["Includes"].push_back(keyVal.first);
  }
  data["UBOIncl"] = uboIncludeName;

  // (2) declarations of struct, constants and typedefs inside class
  //
  std::unordered_set<std::string> excludedNames;
  for(auto pair : a_classInfo.m_setterVars)
    excludedNames.insert(kslicer::CleanTypeName(pair.second));

  data["ClassDecls"] = std::vector<std::string>();
  std::map<std::string, kslicer::DeclInClass> specConsts;
  for(const auto decl : usedDecl)
  {
    if(!decl.extracted)
      continue;
    if(excludedNames.find(decl.type) != excludedNames.end())
      continue;

    if(a_classInfo.pShaderCC->IsGLSL() && decl.name.find("KSPEC_") != std::string::npos) { // process specialization constants, remove them from normal constants
      std::string val = kslicer::GetRangeSourceCode(decl.srcRange, compiler);
      specConsts[val] = decl;
      continue;
    }

    json c_decl;
    c_decl["Text"]    = a_classInfo.pShaderCC->PrintHeaderDecl(decl, compiler);
    c_decl["InClass"] = decl.inClass;
    c_decl["IsType"]  = (decl.kind == DECL_IN_CLASS::DECL_STRUCT); // || (decl.kind == DECL_IN_CLASS::DECL_TYPEDEF);
    c_decl["Type"]    = kslicer::CleanTypeName(decl.type);
    data["ClassDecls"].push_back(c_decl);
  }

  // (3) local functions preprocess
  //
  std::vector<kslicer::FuncData> funcMembers;
  std::unordered_map<std::string, kslicer::FuncData> cachedFunc;
  {
    for (const auto& f : usedFunctions)
    {
      cachedFunc[f.name] = f;
      auto pShit = shittyFunctions.find(f.name);      // exclude shittyFunctions from 'LocalFunctions'
      if(pShit != shittyFunctions.end())
        continue;

      if(f.isMember)
        funcMembers.push_back(f);
    }
  }

  ShaderFeatures shaderFeatures = a_classInfo.globalShaderFeatures;
  for(auto k : a_classInfo.kernels)
    shaderFeatures = shaderFeatures || k.second.shaderFeatures;

  data["GlobalUseInt8"]    = shaderFeatures.useByteType;
  data["GlobalUseInt16"]   = shaderFeatures.useShortType;
  data["GlobalUseInt64"]   = shaderFeatures.useInt64Type;
  data["GlobalUseFloat64"] = shaderFeatures.useFloat64Type;
  data["GlobalUseHalf"]    = shaderFeatures.useHalfType;

  auto dhierarchies   = a_classInfo.GetDispatchingHierarchies();
  data["Hierarchies"] = PutHierarchiesDataToJson(dhierarchies, compiler, a_classInfo);

  // (4) put kernels
  //
  std::unordered_map<std::string, KernelInfo> kernels; // #TODO: Put this to virtual function and override it for RTV
  {
    if(a_classInfo.megakernelRTV)
    {
      for(const auto& cf : a_classInfo.mainFunc)
      {
        kernels[cf.megakernel.name]            = cf.megakernel;
        kernels[cf.megakernel.name].subkernels = cf.subkernels;
      }
    }
    else
      kernels = a_classInfo.kernels;
  }

  data["Kernels"] = std::vector<std::string>();
  for (const auto& nk : kernels)
  {
    const auto& k = nk.second;
    std::cout << "  processing " << k.name << std::endl;

    auto commonArgs = a_classInfo.GetKernelCommonArgs(k);
    auto tidArgs    = a_classInfo.GetKernelTIDArgs(k);
    uint VArgsSize  = 0;
    uint MArgsSize  = 0;
    bool isTextureArrayUsedInThisKernel = false;

    json args = std::vector<std::string>();
    for(auto commonArg : commonArgs)
    {
      json argj;
      std::string buffType1 = a_classInfo.pShaderCC->ProcessBufferType(commonArg.type);
      std::string buffType2 = pShaderRewriter->RewriteStdVectorTypeStr(buffType1);
      argj["Type"]     = commonArg.isImage ? commonArg.imageType : buffType2;
      argj["Name"]     = commonArg.name;
      argj["IsUBO"]    = commonArg.isDefinedInClass;
      argj["IsImage"]  = commonArg.isImage;
      argj["IsAccelStruct"] = false;
      argj["NeedFmt"]       = !commonArg.isSampler;
      argj["ImFormat"]      = commonArg.imageFormat;
      argj["IsPointer"]     = commonArg.isPointer;
      argj["IsMember"]      = false;

      std::string ispcConverted = argj["Name"];
      if(argj["IsPointer"])
        ispcConverted = ConvertVecTypesToISPC(argj["Type"], argj["Name"]);
      argj["NameISPC"] = ispcConverted;

      args.push_back(argj);
      if(!commonArg.isThreadFlags)
        VArgsSize++;
    }

    assert(tidArgs.size() <= 3);

    // now add all std::vector members
    //
    json rtxNames = std::vector<std::string>();
    json vecs = std::vector<std::string>();
    for(const auto& container : k.usedContainers)
    {
      auto pVecMember     = dataMembersCached.find(container.second.name);
      auto pVecSizeMember = dataMembersCached.find(container.second.name + "_size");

      size_t bufferSizeOffset = 0;

      if(container.second.isSetter)
      {
        //std::cout << "kslicer::PrepareJsonForKernel, setter: " << container.second.name << std::endl;
        //continue;
        pVecMember = a_classInfo.m_setterData.find(container.second.name);
      }
      else if(pVecSizeMember != dataMembersCached.end())
      {
        bufferSizeOffset = pVecSizeMember->second.offsetInTargetBuffer / sizeof(uint32_t);
      }

      assert(pVecMember != dataMembersCached.end());
      assert(pVecMember->second.isContainer);

      std::string buffType1 = a_classInfo.pShaderCC->ProcessBufferType(pVecMember->second.containerDataType);
      std::string buffType2 = pShaderRewriter->RewriteStdVectorTypeStr(buffType1);
      if(!a_classInfo.pShaderCC->IsGLSL() && !a_classInfo.pShaderCC->IsISPC())
        buffType2 += "*";

      json argj;
      argj["Type"]       = buffType2;
      argj["Name"]       = pVecMember->second.name;
      argj["IsUBO"]      = false;
      argj["IsImage"]    = false;
      argj["IsAccelStruct"] = false;
      argj["IsPointer"]     = (pVecMember->second.kind == kslicer::DATA_KIND::KIND_VECTOR);
      argj["IsMember"]      = true;
      std::string ispcConverted = argj["Name"];
      if(argj["IsPointer"])
        ispcConverted = ConvertVecTypesToISPC(argj["Type"], argj["Name"]);
      argj["NameISPC"] = ispcConverted;
      MArgsSize++;

      if(pVecMember->second.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED)
      {
        argj["IsImage"]  = true;
        argj["Type"]     = "sampler2D";
        argj["NeedFmt"]  = false;
        argj["ImFormat"] = "";
        argj["SizeOffset"] = 0;
      }
      else if(pVecMember->second.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY)
      {
        argj["Name"]     = pVecMember->second.name + "[]";
        argj["IsImage"]  = true;
        argj["Type"]     = "sampler2D";
        argj["NeedFmt"]  = false;
        argj["ImFormat"] = "";
        argj["SizeOffset"] = 0;
        isTextureArrayUsedInThisKernel = true;
      }
      else if(pVecMember->second.isContainer && kslicer::IsTextureContainer(pVecMember->second.containerType))
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
      else if(container.second.isAccelStruct())
      {
        argj["IsAccelStruct"] = true;
        rtxNames.push_back(container.second.name);
      }
      else
        argj["SizeOffset"] = bufferSizeOffset; // pVecSizeMember->second.offsetInTargetBuffer / sizeof(uint32_t);

      args.push_back(argj);
      vecs.push_back(argj);
    }

    if(k.isIndirect && !a_classInfo.pShaderCC->IsISPC()) // add indirect buffer to shaders
    {
      json argj;
      argj["Type"]       = a_classInfo.pShaderCC->IsGLSL() ? "uvec4 " : "uint4* ";
      argj["Name"]       = "m_indirectBuffer";
      argj["IsUBO"]      = false;
      argj["IsPointer"]  = false;
      argj["IsImage"]    = false;
      argj["IsAccelStruct"] = false;
      argj["IsMember"]   = false;
      argj["NameISPC"] = argj["Name"];
      args.push_back(argj);
    }

    const auto userArgsArr = GetUserKernelArgs(k.args);
    json userArgs = std::vector<std::string>();
    for(const auto& arg : userArgsArr)
    {
      std::string typeName = pShaderRewriter->RewriteStdVectorTypeStr(arg.type);
      ReplaceFirst(typeName, "const ", "");
      json argj;
      argj["Type"]  = typeName;
      argj["Name"]  = arg.name;
      argj["IsUBO"] = false;
      argj["IsPointer"] = false;
      argj["IsMember"]  = false;
      argj["NameISPC"]  = argj["Name"];
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
      json varJ = ReductionAccessFill(var.second, a_classInfo.pShaderCC, a_classInfo.pShaderFuncRewriter);
      needFinishReductionPass = needFinishReductionPass || !varJ["SupportAtomic"];
      reductionVars.push_back(varJ);
    }

    for(const auto& var : subjToRedArray)
    {
      json varJ = ReductionAccessFill(var.second, a_classInfo.pShaderCC, a_classInfo.pShaderFuncRewriter);
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

    kernelJson["UseSubGroups"] = k.enableSubGroups;

    kernelJson["LastArgNF1"]   = VArgsSize + MArgsSize;
    kernelJson["LastArgNF"]    = VArgsSize; // Last Argument No Flags
    kernelJson["Args"]         = args;
    kernelJson["Vecs"]         = vecs;
    kernelJson["RTXNames"]     = rtxNames;
    kernelJson["UserArgs"]     = userArgs;
    kernelJson["Name"]         = k.name;
    kernelJson["UBOBinding"]   = args.size(); // for circle
    kernelJson["HasEpilog"]    = k.isBoolTyped || reductionVars.size() != 0 || reductionArrs.size() != 0;
    kernelJson["IsBoolean"]    = k.isBoolTyped;
    kernelJson["SubjToRed"]    = reductionVars;
    kernelJson["ArrsToRed"]    = reductionArrs;
    kernelJson["FinishRed"]    = needFinishReductionPass;
    kernelJson["NeedTexArray"] = isTextureArrayUsedInThisKernel;
    kernelJson["WarpSize"]     = k.warpSize;
    kernelJson["InitSource"]   = "";

    kernelJson["SingleThreadISPC"] = k.singleThreadISPC;
    kernelJson["OpenMPAndISPC"]    = k.openMpAndISPC;
    kernelJson["ExplicitIdISPC"]   = k.explicitIdISPC;
    kernelJson["InitKPass"]        = false;

    kernelJson["UseRayGen"]      = k.enableRTPipeline && a_settings.enableRayGen;       // duplicate these options for kernels so we can
    kernelJson["UseMotionBlur"]  = k.enableRTPipeline && a_settings.enableMotionBlur;   // generate some kernels in comute and some in ray tracing mode
    
    kernelJson["EnableBlockExpansion"] = k.be.enabled;
    if(k.be.enabled) // process separate statements inside for loop for Block Expansion
    {
      kernelJson["Source"]   = "";
      kernelJson["SourceBE"] = std::vector<std::string>(); 
      kernelJson["SharedBE"] = std::vector<std::string>();
      
      clang::Rewriter rewrite2;
      rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
      std::shared_ptr<KernelRewriter> pRewriter = a_classInfo.pShaderCC->MakeKernRewriter(rewrite2, compiler, &a_classInfo, const_cast<kslicer::KernelInfo&>(k), std::string(""), false);

      for(const auto var : k.be.sharedDecls)
        kernelJson["SharedBE"].push_back(a_classInfo.pShaderCC->RewriteBESharedDecl(var, pRewriter));
      
      for(const auto stmt : k.be.statements) 
      {
        json statement;
        if(stmt.isParallel && stmt.forLoop != nullptr)
        {
          statement["IsParallel"] = true;
          statement["Text"]     = a_classInfo.pShaderCC->RewriteBEParallelFor(stmt.forLoop, pRewriter);
        }
        else
        {
          statement["IsParallel"] = false;
          statement["Text"]     = a_classInfo.pShaderCC->RewriteBEStmt(stmt.astNode, pRewriter);
        }
        kernelJson["SourceBE"].push_back(statement);
      }
    }
    else             // process the whole code in single pass 
    {
      std::string sourceCodeCut = k.rewrittenText.substr(k.rewrittenText.find_first_of('{')+1);
      kernelJson["Source"]      = sourceCodeCut.substr(0, sourceCodeCut.find_last_of('}'));
      kernelJson["SourceBE"]    = std::vector<std::string>();
      kernelJson["SharedBE"]    = std::vector<std::string>();
    }

    kernelJson["SpecConstants"] = std::vector<std::string>();
    for(auto keyval : specConsts)
    {
      json kspec;
      kspec["Name"] = keyval.second.name;
      kspec["Id"]   = keyval.first;
      kernelJson["SpecConstants"].push_back(kspec);
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    {
      clang::Rewriter rewrite2;
      rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
      auto pVisitorK = a_classInfo.pShaderCC->MakeKernRewriter(rewrite2, compiler, &a_classInfo, const_cast<kslicer::KernelInfo&>(k), "", false);
      //pVisitorK->ClearUserArgs();

      kernelJson["ThreadIds"] = std::vector<std::string>();
      kernelJson["ThreadId0"] = "";
      kernelJson["ThreadId1"] = "";
      kernelJson["ThreadId2"] = "";

      kernelJson["ThreadSizeBE"] = std::vector<std::string>();

      std::vector<std::string> threadIdNames(tidArgs.size());
      for(size_t i=0;i<tidArgs.size();i++)
      {
        uint32_t tid = std::min<uint32_t>(threadsOrder[i], tidArgs.size()-1);
        threadIdNames[i] = tidArgs[tid].name;

        std::string loopSize   = tidArgs[tid].loopIter.sizeText;
        std::string loopStart  = tidArgs[tid].loopIter.startText;
        std::string loopStride = tidArgs[tid].loopIter.strideText;

        if(loopStart == "")
          loopStart = "0";

        if(a_classInfo.pShaderCC->IsISPC())
        {
          if(a_classInfo.allDataMembers.find(loopStart) != a_classInfo.allDataMembers.end())
            loopStart  = a_classInfo.pShaderCC->UBOAccess(loopStart);
          if(a_classInfo.allDataMembers.find(loopSize) != a_classInfo.allDataMembers.end())
            loopSize  = a_classInfo.pShaderCC->UBOAccess(loopSize);
          if(a_classInfo.allDataMembers.find(loopStride) != a_classInfo.allDataMembers.end())
            loopStride  = a_classInfo.pShaderCC->UBOAccess(loopStride);
        }

        const bool noStride = (loopStride == "1") && ((loopStart == "0") ||
                                                      a_classInfo.pShaderCC->IsISPC());

        json threadId;
        if(tidArgs[tid].loopIter.startNode != nullptr && !noStride)
        {
          loopStart  = pVisitorK->RecursiveRewrite(tidArgs[tid].loopIter.startNode);
          //loopSize   = pVisitorK->RecursiveRewrite(tidArgs[tid].sizeNode);
          //loopStride = pVisitorK->RecursiveRewrite(tidArgs[tid].strideNode);
          threadId["Simple"] = 0;
        }
        else
          threadId["Simple"] = 1;

        threadId["Name"]   = tidArgs[tid].name;
        threadId["Type"]   = tidArgs[tid].type;
        threadId["Size"]   = loopSize;
        threadId["Start"]  = loopStart;
        threadId["Stride"] = loopStride;
        if(i == 0)
          kernelJson["ThreadId0"] = threadId;
        else if(i == 1)
          kernelJson["ThreadId1"] = threadId;
        else
          kernelJson["ThreadId2"] = threadId;
        
        json threadIdBE = threadId;
        threadIdBE["Name"]  = k.be.wgNames[i];
        threadIdBE["Type"]  = k.be.wgTypes[i];
        threadIdBE["Value"] = k.wgSize[i];

        kernelJson["ThreadIds"].push_back(threadId);
        kernelJson["ThreadSizeBE"].push_back(threadIdBE);
      }

      kernelJson["threadDim"]   = tidArgs.size();
      kernelJson["threadNames"] = threadIdNames;
      if(threadIdNames.size() >= 1)
      {
        kernelJson["threadName1"] = threadIdNames[0];
        kernelJson["CondLE1"]     = (tidArgs[0].loopIter.condKind == kslicer::KernelInfo::IPV_LOOP_KIND::LOOP_KIND_LESS_EQUAL) ? 1 : 0;
      }
      if(threadIdNames.size() >= 2)
      {
        kernelJson["threadName2"] = threadIdNames[1];
        kernelJson["CondLE2"]     = (tidArgs[1].loopIter.condKind == kslicer::KernelInfo::IPV_LOOP_KIND::LOOP_KIND_LESS_EQUAL) ? 1 : 0;
      }
      if(threadIdNames.size() == 3)
      {
        kernelJson["threadName3"] = threadIdNames[2];
        kernelJson["CondLE3"]     = (tidArgs[2].loopIter.condKind == kslicer::KernelInfo::IPV_LOOP_KIND::LOOP_KIND_LESS_EQUAL) ? 1 : 0;
      }
    }

    //////////////////////////////////////////////////////////////////////////////////  TODO: refactor this code
    std::string tidNames[3];
    std::string tidTypes[3] = {"uint", "uint", "uint"};
    a_classInfo.pShaderCC->GetThreadSizeNames(tidNames);

    if(k.loopIters.size() != 0)
    {
      std::unordered_set<std::string> usedVars;
      for(const auto& iter : k.loopIters)
      {
        uint32_t loopIdReorderd  = threadsOrder[iter.loopNesting];
        auto pFound = usedVars.find(iter.sizeText);
        if(pFound == usedVars.end())
        {
          std::string typeName = pShaderRewriter->RewriteStdVectorTypeStr(iter.type);
          ReplaceFirst(typeName, "const ", "");

          tidNames[loopIdReorderd] = iter.sizeText;                   // #TODO: assert that this expression does not contain .size(); if it does
          tidTypes[loopIdReorderd] = typeName;
          usedVars.insert(iter.sizeText);
        }
      }                                                                // we must change it to 'vec_size2' for example
    }
    //////////////////////////////////////////////////////////////////////////////////  TODO: refactor this code

    kernelJson["threadSZName1"] = tidNames[0];
    kernelJson["threadSZName2"] = tidNames[1];
    kernelJson["threadSZName3"] = tidNames[2];

    kernelJson["threadSZType1"] = tidTypes[0];
    kernelJson["threadSZType2"] = tidTypes[1];
    kernelJson["threadSZType3"] = tidTypes[2];

    kernelJson["WGSizeX"]       = k.wgSize[0]; //
    kernelJson["WGSizeY"]       = k.wgSize[1]; //
    kernelJson["WGSizeZ"]       = k.wgSize[2]; //

    //////////////////////////////////////////////////////////////////////////////////////////
    std::string names[3];
    a_classInfo.pShaderCC->GetThreadSizeNames(names);
    if(a_classInfo.pShaderCC->IsGLSL())
    {
      names[0] = std::string("kgenArgs.") + names[0];
      names[1] = std::string("kgenArgs.") + names[1];
      names[2] = std::string("kgenArgs.") + names[2];
    }

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
      kernelJson["IndirectStartX"] = "0";
      kernelJson["IndirectStartY"] = "0";
      kernelJson["IndirectStartZ"] = "0";

      if(k.loopIters.size() > 0)
      {
        std::string exprContent      = kslicer::ReplaceSizeCapacityExpr(k.loopIters[0].sizeText);
        kernelJson["IndirectSizeX"]  = a_classInfo.pShaderCC->UBOAccess(exprContent);
        kernelJson["IndirectStartX"] = kernelJson["ThreadIds"][0]["Start"];
      }

      if(k.loopIters.size() > 1)
      {
        std::string exprContent     = kslicer::ReplaceSizeCapacityExpr(k.loopIters[1].sizeText);
        kernelJson["IndirectSizeY"] = a_classInfo.pShaderCC->UBOAccess(exprContent);
        kernelJson["IndirectStartY"] = kernelJson["ThreadIds"][1]["Start"];
      }

      if(k.loopIters.size() > 2)
      {
        std::string exprContent     = kslicer::ReplaceSizeCapacityExpr(k.loopIters[2].sizeText);
        kernelJson["IndirectSizeZ"] = a_classInfo.pShaderCC->UBOAccess(exprContent);
        kernelJson["IndirectStartZ"] = kernelJson["ThreadIds"][2]["Start"];
      }

      kernelJson["IndirectOffset"] = k.indirectBlockOffset;
      kernelJson["threadSZName1"]  = "kgen_iNumElementsX";
      kernelJson["threadSZName2"]  = "kgen_iNumElementsY";
      kernelJson["threadSZName3"]  = "kgen_iNumElementsZ";
    }
    else
    {
      kernelJson["IndirectSizeX"] = tidNames[0];
      kernelJson["IndirectSizeY"] = tidNames[1];
      kernelJson["IndirectSizeZ"] = tidNames[2];
    }

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
      pVisitorK->processFuncMember = true; // signal that we process function member, not the kernel itself

      for(auto& f : funcMembers)
      {
        if(f.astNode->isVirtualAsWritten()) // skip virtual functions because they are proccesed else-where
          continue;

        auto funcNode = const_cast<clang::FunctionDecl*>(f.astNode);
        pVisitorF->SetCurrFuncInfo(&f);    // pass auxilary function data inside pVisitorF
        pVisitorK->SetCurrFuncInfo(&f);
        const std::string funcDeclText = pVisitorF->RewriteFuncDecl(funcNode);
        const std::string funcBodyText = pVisitorK->RecursiveRewrite(funcNode->getBody());
        pVisitorF->ResetCurrFuncInfo();
        pVisitorK->ResetCurrFuncInfo();
        kernelJson["MemberFunctions"].push_back(funcDeclText + funcBodyText);
      }
    }

    kernelJson["ShityFunctions"] = std::vector<std::string>();
    std::unordered_map<std::string, kslicer::ShittyFunction> shitByName;
    for(auto shit : k.shittyFunctions) {
      shitByName[shit.ShittyName()] = shit;
      shittyFunctions[shit.originalName] = shit;
    }

    for(auto shit : shitByName)
    {
      auto pFunc = cachedFunc.find(shit.second.originalName);
      if(pFunc == cachedFunc.end())
        continue;

      clang::Rewriter rewrite2;                                                    // It is important to have clear rewriter for each function because here we access same node several times!!!
      rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());  //
      auto pVisitorF = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo, shit.second);
      auto funcNode  = const_cast<clang::FunctionDecl*>(pFunc->second.astNode);

      const std::string funcDeclText = pVisitorF->RewriteFuncDecl(funcNode);
      const std::string funcBodyText = pVisitorF->RecursiveRewrite(funcNode->getBody());

      kernelJson["ShityFunctions"].push_back(funcDeclText + funcBodyText);
    }

    kernelJson["Subkernels"]  = std::vector<std::string>();
    if(a_classInfo.megakernelRTV)
    {
      for(auto pSubkernel : k.subkernels)
      {
        auto& subkernel = (*pSubkernel);

        std::string funcDeclText = "...";
        {
          kslicer::ShittyFunction shit;
          for(const auto& candidate : k.shittyFunctions)
          {
            if(candidate.originalName == subkernel.name)
            {
              shit = candidate;
              break;
            }
          }

          clang::Rewriter rewrite2;                                                    // It is important to have clear rewriter for each function because here we access same node several times!!!
          rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());  //
          auto pVisitorF = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo, shit);
          auto funcNode  = const_cast<clang::CXXMethodDecl*>(subkernel.astNode);
          funcDeclText   = pVisitorF->RewriteFuncDecl(funcNode);
        }

        json subJson;
        subJson["Name"] = subkernel.name;
        subJson["RetT"] = subkernel.isBoolTyped ? "bool" : "void";
        subJson["Decl"] = funcDeclText;
        std::string sourceCodeCut = subkernel.rewrittenText.substr(subkernel.rewrittenText.find_first_of('{')+1);
        subJson["Source"] = sourceCodeCut.substr(0, sourceCodeCut.find_last_of('}'));

        kernelJson["Subkernels"].push_back(subJson);
      }
    }

    kernelJson["ThreadLocalArrays"] = std::vector<json>();
    for(const auto& array : k.threadLocalArrays)
    {
      json local;
      local["Type"] = array.second.elemType;
      local["Name"] = array.second.arrayName;
      local["Size"] = array.second.arraySize;
      kernelJson["ThreadLocalArrays"].push_back(local);
    }

    auto original = kernelJson;

    // if we have additional init statements we should add additional init kernel before our kernel
    //
    if(k.hasInitPass)
    {
      std::string initSourceCode = k.rewrittenInit.substr(k.rewrittenInit.find_first_of('{')+1);
      if(a_classInfo.pShaderCC->IsISPC())
      {
        original["InitSource"] = initSourceCode;
      }
      else
      {
        kernelJson["Name"]      = k.name + "_Init";
        kernelJson["Source"]    = initSourceCode;
        kernelJson["HasEpilog"] = false;
        kernelJson["FinishRed"] = false;
        kernelJson["InitKPass"] = true;
        kernelJson["WGSizeX"]   = 1;
        kernelJson["WGSizeY"]   = 1;
        kernelJson["WGSizeZ"]   = 1;
        data["Kernels"].push_back(kernelJson);
      }
    }

    data["Kernels"].push_back(original);

    if(k.hasFinishPassSelf)
    {
      kernelJson["Name"]      = k.name + "_Finish";
      kernelJson["Source"]    = k.rewrittenFinish;
      kernelJson["HasEpilog"] = false;
      kernelJson["FinishRed"] = false;
      kernelJson["InitKPass"] = true;
      kernelJson["WGSizeX"]   = 1;
      kernelJson["WGSizeY"]   = 1;
      kernelJson["WGSizeZ"]   = 1;
      data["Kernels"].push_back(kernelJson);
    }

  } // for (const auto& nk : kernels)

  // (5) generate local functions
  //
  data["LocalFunctions"] = std::vector<std::string>();
  {
    clang::Rewriter rewrite2;
    rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
    auto pVisitorF = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);

    for (const auto& f : usedFunctions)
    {
      cachedFunc[f.name] = f;
      auto pShit = shittyFunctions.find(f.name);      // exclude shittyFunctions from 'LocalFunctions'
      if(pShit != shittyFunctions.end())
        continue;

      if(!f.isMember)
      {
        pVisitorF->TraverseDecl(const_cast<clang::FunctionDecl*>(f.astNode));
        const std::string funDecl  = rewrite2.getRewrittenText(f.srcRange);             // func body rewrite does not works correctly in this way some-times (see float4x4 indices)
        const std::string funBody  = pVisitorF->RecursiveRewrite(f.astNode->getBody()); // but works in this way ... 
        const std::string declHead = funDecl.substr(0,funDecl.find("{"));               // therefore we join function head and body
    
        data["LocalFunctions"].push_back(declHead + funBody);
        shaderFeatures = shaderFeatures || pVisitorF->GetShaderFeatures();
      }
    }
  }

  if(a_classInfo.NeedFakeOffset())
  {
    data["LocalFunctions"].push_back("uint fakeOffset(uint x, uint y, uint pitch) { return y*pitch + x; }  // RTV pattern, for 2D threading"); // todo: ckeck if RTV pattern is used here!
    //data["LocalFunctions"].push_back("uint fakeOffset3(uint x, uint y, uint z, uint sizeY, uint sizeX) { return z*sizeY*sizeX + y*sizeX + x; } // for 3D threading");
  }

  data["ThreadLocalArrays"] = std::vector<json>();
  for(const auto& array : a_classInfo.m_threadLocalArrays)
  {
    json local;
    local["Type"] = array.second.elemType;
    local["Name"] = array.second.arrayName;
    local["Size"] = array.second.arraySize;
    data["ThreadLocalArrays"].push_back(local);
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

