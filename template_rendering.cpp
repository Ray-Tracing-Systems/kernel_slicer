#include "template_rendering.h"
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
    size_t voidPos = kernDecl.find("void ");
    size_t boolPos = kernDecl.find("bool ");
    size_t rbPos   = kernDecl.find("Cmd(");

    assert(voidPos != std::string::npos || boolPos != std::string::npos);
    assert(rbPos != std::string::npos);
    
    if(voidPos == std::string::npos)
      voidPos = boolPos;

    std::string kernName       = kernDecl.substr(voidPos + 5, rbPos - 5);
    kernelDeclByName[kernName] = kernDecl.substr(voidPos + 5);
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

nlohmann::json kslicer::PrepareJsonForAllCPP(const MainClassInfo& a_classInfo, 
                                             const std::vector<MainFuncInfo>& a_methodsToGenerate, 
                                             const std::string& a_genIncude)
{
  std::string folderPath           = GetFolderPath(a_classInfo.mainClassFileName);
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
  for(size_t i=0;i<a_classInfo.kernels.size();i++)
  {
    if(i != 0)
      strOut2 << "  ";
    strOut2 << "virtual " << a_classInfo.kernels[i].DeclCmd.c_str() << ";\n";
  }

  json data;
  data["Includes"]      = strOut.str();
  data["MainClassName"] = a_classInfo.mainClassName;

  data["PlainMembersUpdateFunctions"]  = "";
  data["VectorMembersUpdateFunctions"] = "";
  data["KernelsDecl"]                  = strOut2.str();   
  data["TotalDSNumber"]                = a_classInfo.allDescriptorSetsInfo.size();

  data["KernelNames"] = std::vector<std::string>();  
  for(const auto& k : a_classInfo.kernels)
  {
    std::string kernName = a_classInfo.RemoveKernelPrefix(k.name);
    data["KernelNames"].push_back(kernName);
  }

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
    local["Name"]   = v.name;
    local["Offset"] = v.offsetInTargetBuffer;
    local["Size"]   = v.sizeInBytes;
    data["ClassVars"].push_back(local);
  }

  data["ClassVectorVars"] = std::vector<std::string>();
  for(const auto& v : a_classInfo.dataMembers)
  {
    if(!v.isContainer)
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

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::vector<std::string> kernelsCallCmdDecl(a_classInfo.kernels.size());
  for(size_t i=0;i<kernelsCallCmdDecl.size();i++)
    kernelsCallCmdDecl[i] = a_classInfo.kernels[i].DeclCmd;
 
  auto kernelDeclByName = MakeMapForKernelsDeclByName(kernelsCallCmdDecl);

  data["Kernels"] = std::vector<std::string>();  
  for(const auto& k : a_classInfo.kernels)
  {
    std::string kernName = a_classInfo.RemoveKernelPrefix(k.name);
    
    json local;
    local["Name"]         = kernName;
    local["OriginalName"] = k.name;
    local["ArgCount"]     = k.args.size();
    local["Decl"]         = kernelDeclByName[kernName];

    std::vector<std::string> threadIdNamesList;

    local["Args"]         = std::vector<std::string>();
    size_t actualSize     = 0;
    for(const auto& arg : k.args)
    {
      if(arg.isThreadID) // exclude TID args bindings
      {
        threadIdNamesList.push_back(arg.name);
        continue;
      }
      json argData;
      argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
      argData["Name"]  = arg.name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
      local["Args"].push_back(argData);
      actualSize++;
    }

    for(const auto& name : k.usedVectors)
    {
      json argData;
      argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
      argData["Name"]  = name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
      local["Args"].push_back(argData);
      actualSize++;
    }

    local["ArgCount"] = actualSize;

    std::sort(threadIdNamesList.begin(), threadIdNamesList.end());
    
    assert(threadIdNamesList.size() > 0);

    if(threadIdNamesList.size() > 0)
      local["tidX"] = threadIdNamesList[0];
    else
      local["tidX"] = 1;

    if(threadIdNamesList.size() > 1)
      local["tidY"] = threadIdNamesList[1];
    else
      local["tidY"] = 1;

    if(threadIdNamesList.size() > 2)
      local["tidZ"] = threadIdNamesList[2];
    else
      local["tidZ"] = 1;

    data["Kernels"].push_back(local);
  }
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  auto predefinedNamesId = GetAllPredefinedThreadIdNamesRTV();

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
      data2["LocalVarsBuffersDecl"].push_back(local);
    }

    data2["InOutVars"] = std::vector<std::string>();
    for(const auto& v : mainFunc.InOuts)
      data2["InOutVars"].push_back(v.first);

    // for impl, ds bindings
    //
    for(size_t i=mainFunc.startDSNumber; i<mainFunc.endDSNumber; i++)
    {
      auto& dsArgs = a_classInfo.allDescriptorSetsInfo[i];
  
      json local;
      local["Id"]         = i;
      local["KernelName"] = dsArgs.kernelName;
      local["Layout"]     = dsArgs.kernelName + "DSLayout";
      local["Args"]       = std::vector<std::string>();
      local["ArgNames"]   = std::vector<std::string>();

      uint32_t realId = 0; 
      for(size_t j=0;j<dsArgs.descriptorSetsInfo.size();j++)
      {
        auto elementId = std::find(predefinedNamesId.begin(), predefinedNamesId.end(), dsArgs.descriptorSetsInfo[j].varName); // exclude predefined names from arguments
        if(elementId != predefinedNamesId.end())
          continue;

        const std::string dsArgName = GetDSArgName(mainFunc.Name, dsArgs.descriptorSetsInfo[j].varName);

        json arg;
        arg["Id"]   = realId;
        arg["Name"] = dsArgName;
        local["Args"].push_back(arg);
        local["ArgNames"].push_back(dsArgs.descriptorSetsInfo[j].varName);
        realId++;
      }

      size_t kernelId = size_t(-1);
      for(size_t j=0;j<a_classInfo.kernels.size();j++)
      {
        if(a_classInfo.kernels[j].name == dsArgs.originKernelName)
        {
          kernelId = j;
          break;
        }
      }

      if(kernelId != size_t(-1))
      {
        const auto& kernel = a_classInfo.kernels[kernelId];
        for(const auto& vecName : kernel.usedVectors)
        {
          json arg;
          arg["Id"]   = realId;
          arg["Name"] = "m_vdata." + vecName;
          local["Args"].push_back(arg);
          realId++;
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
  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, a_data);
  
  std::ofstream fout(a_outFilePath);
  fout << result.c_str() << std::endl;
  fout.close();
}

std::string GetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo);

void kslicer::PrintGeneratedCLFile(const std::string& a_inFileName, const std::string& a_outFolder, const MainClassInfo& a_classInfo, 
                                   const std::unordered_map<std::string, bool>& usedFiles, 
                                   const std::unordered_map<std::string, clang::SourceRange>& usedFunctions,
                                   const clang::CompilerInstance& compiler)
{
  std::unordered_map<std::string, DataMemberInfo> dataMembersCached;
  dataMembersCached.reserve(a_classInfo.dataMembers.size());
  for(const auto& member : a_classInfo.dataMembers)
    dataMembersCached[member.name] = member;

  const std::string& a_outFileName = a_outFolder + "/" + "z_generated.cl";
  json data;

  // (1) put includes
  //
  data["Includes"] = std::vector<std::string>();
  for(auto keyVal : a_classInfo.allIncludeFiles) // we will search for only used include files among all of them (quoted, angled were excluded earlier)
  {
    for(auto keyVal2 : usedFiles)
    {
      if(keyVal2.first.find(keyVal.first)                   != std::string::npos && 
        a_classInfo.mainClassFileInclude.find(keyVal.first) == std::string::npos)
        data["Includes"].push_back(keyVal.first);
    }
  }

  // (2) local functions
  //
  data["LocalFunctions"] = std::vector<std::string>();
  for (const auto& f : usedFunctions)  
    data["LocalFunctions"].push_back(kslicer::GetRangeSourceCode(f.second, compiler));

  data["LocalFunctions"].push_back("uint fakeOffset(uint x, uint y, uint pitch) { return y*pitch + x; }                                      // for 2D threading");
  data["LocalFunctions"].push_back("uint fakeOffset3(uint x, uint y, uint z, uint sizeY, uint sizeX) { return z*sizeY*sizeX + y*sizeX + x; } // for 3D threading");

  // (3) put kernels
  //
  data["Kernels"] = std::vector<std::string>();
  
  clang::SourceManager& sm = compiler.getSourceManager();
  
  for (const auto& k : a_classInfo.kernels)  
  {
    std::cout << "  processing " << k.name << std::endl;
    
    auto commonArgs = a_classInfo.GetKernelCommonArgs(k);
    auto tidArgs    = a_classInfo.GetKernelTIDArgs(k);
    
    json args = std::vector<std::string>();
    for(auto commonArg : commonArgs)
    {
      json argj;
      argj["Type"] = commonArg.typeName;
      argj["Name"] = commonArg.argName;
      args.push_back(argj);
    }

    std::vector<std::string> threadIdNames;
    for(auto tid : tidArgs)
      threadIdNames.push_back(tid.argName);

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
      
      std::string typeStr = pVecMember->second.containerDataType;
      kslicer::ReplaceOpenCLBuiltInTypes(typeStr);

      json argj;
      argj["Type"] = typeStr + "*";
      argj["Name"] = pVecMember->second.name;
      argj["SizeOffset"] = pVecSizeMember->second.offsetInTargetBuffer / sizeof(uint32_t);
      args.push_back(argj);
      vecs.push_back(argj);
    }

    // add kgen_data buffer and skiped predefined ThreadId back
    //
    {
      json argj;
      argj["Type"] = "uint*";
      argj["Name"] = kslicer::GetProjPrefix() + "data";
      args.push_back(argj);
    }
    
    json kernelJson;
    kernelJson["Args"]        = args;
    kernelJson["Vecs"]        = vecs;
    kernelJson["Name"]        = k.name;

    std::string sourceCodeCut = k.rewrittenText.substr(k.rewrittenText.find_first_of('{')+1);
    kernelJson["Source"]      = sourceCodeCut.substr(0, sourceCodeCut.find_last_of('}'));
    kernelJson["threadDim"]   = threadIdNames.size();
    kernelJson["threadNames"] = threadIdNames;
    if(threadIdNames.size() >= 1)
    {
      kernelJson["threadName1"] = threadIdNames[0];
    }
    if(threadIdNames.size() >= 2)
    {
      kernelJson["threadName2"] = threadIdNames[1];
    }
    if(threadIdNames.size() == 3)
    {
      kernelJson["threadName3"] = threadIdNames[2];
    }

    kernelJson["shouldCheckExitFlag"] = k.checkThreadFlags;
    kernelJson["checkFlagsExpr"]      = "//xxx//";
    kernelJson["ThreadOffset"]        = GetFakeOffsetExpression(k);

    data["Kernels"].push_back(kernelJson);
  }

  inja::Environment env;
  inja::Template temp = env.parse_template(a_inFileName.c_str());
  std::string result  = env.render(temp, data);
  
  std::ofstream fout(a_outFileName);
  fout << result.c_str() << std::endl;
  fout.close();
}