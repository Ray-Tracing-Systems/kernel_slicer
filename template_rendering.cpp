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

std::string GetFolderPath(const std::string& a_filePath)
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

void MakeAbsolutePathRelativeTo(std::string& a_filePath, const std::string& a_folderPath)
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
                                      const std::vector<kslicer::ArgFinal>& threadIds,
                                      const std::string a_names[3]);
}

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);

static json ReductionAccessFill(const kslicer::KernelInfo::ReductionAccess& second, std::shared_ptr<kslicer::IShaderCompiler> pShaderCC)
{
  json varJ;
  varJ["Type"]          = second.dataType;
  varJ["Name"]          = second.leftExpr;
  varJ["Init"]          = second.GetInitialValue(pShaderCC->IsGLSL());
  varJ["Op"]            = second.GetOp(pShaderCC);
  varJ["NegLastStep"]   = (second.type == kslicer::KernelInfo::REDUCTION_TYPE::SUB || second.type == kslicer::KernelInfo::REDUCTION_TYPE::SUB_ONE);
  varJ["BinFuncForm"]   = (second.type == kslicer::KernelInfo::REDUCTION_TYPE::FUNC);
  varJ["OutTempName"]   = second.tmpVarName;
  varJ["SupportAtomic"] = second.SupportAtomicLastStep();
  varJ["AtomicOp"]      = second.GetAtomicImplCode(pShaderCC->IsGLSL());
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
  data["MainClassName"]      = a_classInfo.mainClassName;
  data["UseSpecConstWgSize"] = a_classInfo.pShaderCC->UseSpecConstForWgSize();
  data["UseServiceMemCopy"]  = (a_classInfo.usedServiceCalls.find("memcpy") != a_classInfo.usedServiceCalls.end());

  // (1) put includes
  //
  data["Includes"] = std::vector<std::string>();
  for(auto keyVal : a_classInfo.allIncludeFiles) // we will search for only used include files among all of them (quoted, angled were excluded earlier)
  { 
    bool found = false;
    for(auto f : a_classInfo.ignoreFolders)  // exclude everything from "shader" folders
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
    cdecl["Text"]    = a_classInfo.pShaderCC->PrintHeaderDecl(decl, compiler);
    cdecl["InClass"] = decl.inClass;
    cdecl["IsType"]  = (decl.kind == DECL_IN_CLASS::DECL_STRUCT); // || (decl.kind == DECL_IN_CLASS::DECL_TYPEDEF);
    cdecl["Type"]    = kslicer::CleanTypeName(decl.type);
    data["ClassDecls"].push_back(cdecl);
  }

  // (3) local functions
  //
  ShaderFeatures shaderFeatures;
  data["LocalFunctions"] = std::vector<std::string>();
  std::vector<kslicer::FuncData> funcMembers;
  std::unordered_map<std::string, kslicer::FuncData> cachedFunc;
  {
    clang::Rewriter rewrite2;
    rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
    auto pVisitorF = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);

    for (const auto& f : usedFunctions) 
    { 
      if(a_classInfo.IsExcludedLocalFunction(f.name)) // check exclude list here, don't put such functions in cl file
        continue;
      
      cachedFunc[f.name] = f;
      auto pShit = shittyFunctions.find(f.name);      // exclude shittyFunctions frol 'LocalFunctions'
      if(pShit != shittyFunctions.end())
        continue;

      if(!f.isMember)
      {
        //f.astNode->dump();
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
  std::unordered_map<std::string, KernelInfo> kernels; // #TODO: Put this to virtual function and override it for RTV
  {
    if(a_classInfo.megakernelRTV)
    {
      for(const auto& cf : a_classInfo.mainFunc)
      {
        kernels[cf.megakernel.name] = cf.megakernel;
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
      argj["NeedFmt"]  = !commonArg.isSampler;
      argj["ImFormat"] = commonArg.imageFormat;

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
      if(!a_classInfo.pShaderCC->IsGLSL())
        buffType2 += "*";
      
      json argj;
      argj["Type"]       = buffType2;
      argj["Name"]       = pVecMember->second.name;
      argj["IsUBO"]      = false;
      argj["IsImage"]    = false;
      argj["IsAccelStruct"] = false; 

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

    if(k.isMaker) // add to kernel ObjPtr buffer
    {
      json argj; 
      argj["Type"]       = "uint2       *";
      argj["Name"]       = "kgen_objPtrData";
      argj["IsUBO"]      = false;
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
    kernelJson["RTXNames"]   = rtxNames;
    kernelJson["UserArgs"]   = userArgs;
    kernelJson["Name"]       = k.name;
    kernelJson["UBOBinding"] = args.size(); // for circle
    kernelJson["HasEpilog"]  = k.isBoolTyped || reductionVars.size() != 0 || reductionArrs.size() != 0 || k.isMaker;
    kernelJson["IsBoolean"]  = k.isBoolTyped;
    kernelJson["IsMaker"]    = k.isMaker;
    kernelJson["IsVirtual"]  = k.isVirtual;
    kernelJson["SubjToRed"]  = reductionVars;
    kernelJson["ArrsToRed"]  = reductionArrs;
    kernelJson["FinishRed"]  = needFinishReductionPass;
    kernelJson["NeedTexArray"] = isTextureArrayUsedInThisKernel;

    std::string sourceCodeCut = k.rewrittenText.substr(k.rewrittenText.find_first_of('{')+1);
    kernelJson["Source"]      = sourceCodeCut.substr(0, sourceCodeCut.find_last_of('}'));

    //////////////////////////////////////////////////////////////////////////////////////////
    {
      clang::Rewriter rewrite2; 
      rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
      auto pVisitorK = a_classInfo.pShaderCC->MakeKernRewriter(rewrite2, compiler, &a_classInfo, const_cast<kslicer::KernelInfo&>(k), "", false);
      //pVisitorK->ClearUserArgs();

      kernelJson["ThreadIds"] = std::vector<std::string>();

      std::vector<std::string> threadIdNames(tidArgs.size());
      for(size_t i=0;i<tidArgs.size();i++)
      {
        uint32_t tid = std::min<uint32_t>(threadsOrder[i], tidArgs.size()-1);
        threadIdNames[i] = tidArgs[tid].name;
        
        std::string loopSize   = tidArgs[tid].loopIter.sizeText;   
        std::string loopStart  = tidArgs[tid].loopIter.startText;  
        std::string loopStride = tidArgs[tid].loopIter.strideText; 
        
        json threadId;
        if(tidArgs[tid].loopIter.startNode != nullptr)
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
        //threadId["Size"]   = loopSize;
        threadId["Start"]  = loopStart;
        threadId["Stride"] = loopStride;
        //if(tidArgs[tid].loopIter.condKind == kslicer::KernelInfo::IPV_LOOP_KIND::LOOP_KIND_LESS_EQUAL)  
        //  threadId["CondLE"] = 1;
        //else
        //  threadId["CondLE"] = 0;
        kernelJson["ThreadIds"].push_back(threadId);
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
      pVisitorK->processFuncMember = true; // signal that we process function member, not the kernel itself
    
      for(auto& f : funcMembers)
      {
        auto funcNode = const_cast<clang::FunctionDecl*>(f.astNode);
        const std::string funcDeclText = pVisitorF->RewriteFuncDecl(funcNode);
        const std::string funcBodyText = pVisitorK->RecursiveRewrite(funcNode->getBody());
        kernelJson["MemberFunctions"].push_back(funcDeclText + funcBodyText);
      }
    }
    
    kernelJson["ShityFunctions"] = std::vector<std::string>();
    std::unordered_map<std::string, kslicer::ShittyFunction> shitByName;
    for(auto shit : k.shittyFunctions)
      shitByName[shit.ShittyName()] = shit;

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

    auto original = kernelJson;
    
    // if we have additional init statements we should add additional init kernel before our kernel
    //
    if(k.hasInitPass)
    {      
      kernelJson["Name"]      = k.name + "_Init";
      kernelJson["Source"]    = k.rewrittenInit.substr(k.rewrittenInit.find_first_of('{')+1);
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

