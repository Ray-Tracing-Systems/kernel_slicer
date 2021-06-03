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

std::string GetDSArgName(const std::string& a_mainFuncName, const std::string& a_dsVarName)
{
  auto posOfData = a_dsVarName.find(".data()");
  if(posOfData != std::string::npos)
    return std::string("m_vdata.") + a_dsVarName.substr(0, posOfData);
  else
    return a_mainFuncName + "_local." + a_dsVarName; 
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
  data["ShaderGLSL"]         = !a_classInfo.pShaderCC->IsSingleSource();
  data["UseSeparateUBO"]     = a_classInfo.pShaderCC->UseSeparateUBOForArguments();
  data["UseSpecConstWgSize"] = a_classInfo.pShaderCC->UseSpecConstForWgSize();
  data["UseServiceMemCopy"]  = (a_classInfo.usedServiceCalls.find("memcpy") != a_classInfo.usedServiceCalls.end());

  data["PlainMembersUpdateFunctions"]  = "";
  data["VectorMembersUpdateFunctions"] = "";
  data["KernelsDecl"]                  = strOut2.str();   
  data["TotalDSNumber"]                = a_classInfo.allDescriptorSetsInfo.size();

  data["VectorMembers"] = std::vector<std::string>();
  for(const auto var : a_classInfo.dataMembers)
  {
    if(var.isContainer)
      data["VectorMembers"].push_back(var.name);
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

  data["ClassVectorVars"] = std::vector<std::string>();
  for(const auto& v : a_classInfo.dataMembers)
  {
    if(!v.isContainer || v.usage != kslicer::DATA_USAGE::USAGE_USER)
      continue;
    
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
      argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
      argData["Name"]  = arg.name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
      kernelJson["Args"].push_back(argData);
      actualSize++;
    }

    for(const auto& name : k.usedVectors)
    {
      json argData;
      argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
      argData["Name"]  = name;
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
      data2["InOutVars"].push_back(v.name);

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

        const std::string dsArgName = GetDSArgName(mainFunc.Name, dsArgs.descriptorSetsInfo[j].varName);

        json arg;
        arg["Id"]   = realId;
        arg["Name"] = dsArgName;
        local["Args"].push_back(arg);
        local["ArgNames"].push_back(dsArgs.descriptorSetsInfo[j].varName);
        realId++;
      }
      
      if(pFoundKernel != a_classInfo.kernels.end())
      {
        for(const auto& vecName : pFoundKernel->second.usedVectors) // add all class-member vectors bindings
        {
          json arg;
          arg["Id"]   = realId;
          arg["Name"] = "m_vdata." + vecName;
          local["Args"].push_back(arg);
          local["ArgNames"].push_back(vecName);
          realId++;
        }

        if(pFoundKernel->second.isMaker || pFoundKernel->second.isVirtual)
        {
          auto hierarchy = dhierarchies[pFoundKernel->second.interfaceName];

          json arg;
          arg["Id"]   = realId;
          arg["Name"] = std::string("m_") + hierarchy.interfaceName + "ObjPtr";
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
  
  clang::Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
  auto pShaderRewriter = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);

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
    typeStr = pShaderRewriter->RewriteVectorTypeStr(typeStr); //a_classInfo.RemoveTypeNamespaces(typeStr);

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
  clang::Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
  auto pShaderRewriter = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);

  std::unordered_map<std::string, DataMemberInfo> dataMembersCached;
  dataMembersCached.reserve(a_classInfo.dataMembers.size());
  for(const auto& member : a_classInfo.dataMembers)
    dataMembersCached[member.name] = member;

  json data;
  data["MainClassName"]      = a_classInfo.mainClassName;
  data["UseSpecConstWgSize"] = a_classInfo.pShaderCC->UseSpecConstForWgSize();

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
  {
    clang::Rewriter rewrite2;
    rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
    auto pVisitor = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);

    for (const auto& f : usedFunctions) 
    { 
      if(a_classInfo.IsExcludedLocalFunction(f.name)) // check exclude list here, don't put such functions in cl file
        continue;
      
      //if(f.name == "RealColorToUint32_f3")
      //  f.astNode->dump();   
      pVisitor->TraverseDecl(const_cast<clang::FunctionDecl*>(f.astNode));
      data["LocalFunctions"].push_back(rewrite2.getRewrittenText(f.srcRange));
      shaderFeatures = shaderFeatures || pVisitor->GetShaderFeatures();
    }
  }
  data["LocalFunctions"].push_back("uint fakeOffset(uint x, uint y, uint pitch) { return y*pitch + x; }                                      // for 2D threading");
  //data["LocalFunctions"].push_back("uint fakeOffset3(uint x, uint y, uint z, uint sizeY, uint sizeX) { return z*sizeY*sizeX + y*sizeX + x; } // for 3D threading");

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
      std::string buffType2 = pShaderRewriter->RewriteVectorTypeStr(buffType1); 
      argj["Type"]  = buffType2;
      argj["Name"]  = commonArg.argName;
      argj["IsUBO"] = commonArg.isUBO;
      args.push_back(argj);
      if(!commonArg.isThreadFlags)
        VArgsSize++;
    }

    assert(tidArgs.size() <= 3);

    std::vector<std::string> threadIdNames(tidArgs.size());
    for(size_t i=0;i<tidArgs.size();i++)
    {
      uint32_t tid = std::min<uint32_t>(threadsOrder[i], tidArgs.size()-1);
      threadIdNames[i] = tidArgs[tid].argName;
    }

    // now add all std::vector members
    //
    json vecs = std::vector<std::string>();
    for(const auto& name : k.usedVectors)
    {
      auto pVecMember     = dataMembersCached.find(name);
      auto pVecSizeMember = dataMembersCached.find(name + "_size");

      assert(pVecMember     != dataMembersCached.end());
      assert(pVecSizeMember != dataMembersCached.end());

      assert(pVecMember->second.isContainer);
      assert(pVecSizeMember->second.isContainerInfo);

      std::string buffType1 = a_classInfo.pShaderCC->ProcessBufferType(pVecMember->second.containerDataType);
      std::string buffType2 = pShaderRewriter->RewriteVectorTypeStr(buffType1);
      if(!a_classInfo.pShaderCC->IsGLSL())
        buffType2 += "*";
      
      json argj;
      argj["Type"]       = buffType2;
      argj["Name"]       = pVecMember->second.name;
      argj["SizeOffset"] = pVecSizeMember->second.offsetInTargetBuffer / sizeof(uint32_t);
      argj["IsUBO"]      = false;
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
      membersToRead.push_back(dataMembersCached[name]);
    
    std::sort(membersToRead.begin(), membersToRead.end(), [](const auto& a, const auto& b) { return a.offsetInTargetBuffer < b.offsetInTargetBuffer; });

    json members = std::vector<std::string>();
    for(const auto member : membersToRead)
    {
      if(member.isArray || member.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD) // read large data structures directly inside kernel code, don't read them at the beggining of kernel.
        continue;

      if(k.subjectedToReduction.find(member.name) != k.subjectedToReduction.end())  // exclude this opt for members which subjected to reduction
        continue;

      json memberData;
      memberData["Type"]   = pShaderRewriter->RewriteVectorTypeStr(member.type);
      memberData["Name"]   = member.name;
      memberData["Offset"] = member.offsetInTargetBuffer / sizeof(uint32_t);
      members.push_back(memberData);
    }

    const auto userArgsArr = GetUserKernelArgs(k.args);
    json userArgs = std::vector<std::string>();
    for(const auto& arg : userArgsArr)
    {
      json argj;
      argj["Type"]  = pShaderRewriter->RewriteVectorTypeStr(arg.type);
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

    kernelJson["threadDim"]   = threadIdNames.size();
    kernelJson["threadNames"] = threadIdNames;
    if(threadIdNames.size() >= 1)
      kernelJson["threadName1"] = threadIdNames[0];
    if(threadIdNames.size() >= 2)
      kernelJson["threadName2"] = threadIdNames[1];
    if(threadIdNames.size() == 3)
      kernelJson["threadName3"] = threadIdNames[2];

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

