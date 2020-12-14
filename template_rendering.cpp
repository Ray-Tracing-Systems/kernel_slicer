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
    size_t rbPos   = kernDecl.find("Cmd(");

    assert(voidPos != std::string::npos);
    assert(rbPos != std::string::npos);
    
    std::string kernName       = kernDecl.substr(voidPos + 5, rbPos - 5);
    kernelDeclByName[kernName] = kernDecl.substr(voidPos + 5);
  }
  return kernelDeclByName;
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
    std::string kernName = k.name;
    auto pos = kernName.find("kernel_");
    if(pos != std::string::npos)
      kernName = kernName.substr(7);
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

  auto predefinedNames = kslicer::GetAllPredefinedThreadIdNames();

  std::vector<std::string> kernelsCallCmdDecl(a_classInfo.kernels.size());
  for(size_t i=0;i<kernelsCallCmdDecl.size();i++)
    kernelsCallCmdDecl[i] = a_classInfo.kernels[i].DeclCmd;
 
  auto kernelDeclByName = MakeMapForKernelsDeclByName(kernelsCallCmdDecl);

  data["Kernels"] = std::vector<std::string>();  
  for(const auto& k : a_classInfo.kernels)
  {
    std::string kernName = k.name;
    auto pos = kernName.find("kernel_");
    if(pos != std::string::npos)
      kernName = kernName.substr(7);
    
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
      auto elementId = std::find(predefinedNames.begin(), predefinedNames.end(), arg.name);
      if(elementId != predefinedNames.end()) // exclude predefined names from bindings
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
  auto predefinedNamesId = GetAllPredefinedThreadIdNames();

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

        json arg;
        arg["Id"]   = realId;
        arg["Name"] = mainFunc.Name + "_local." + dsArgs.descriptorSetsInfo[j].varName;
        local["Args"].push_back(arg);
        local["ArgNames"].push_back(dsArgs.descriptorSetsInfo[j].varName);
        realId++;
      }

      // #TODO: Add Vector Members bindings here ... 
      //
      size_t kernelId = size_t(-1);
      for(size_t j=0;j<a_classInfo.kernels.size();j++)
      {
        if(a_classInfo.kernels[j].name == std::string("kernel_") + dsArgs.kernelName)
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
    bool foundThreadIdX = false; std::string tidXName = "tid";
    bool foundThreadIdY = false; std::string tidYName = "tid2";
    bool foundThreadIdZ = false; std::string tidZName = "tid3";

    json args = std::vector<std::string>();
    json vecs = std::vector<std::string>();
    for (const auto& arg : k.args) 
    {
      std::string typeStr = arg.type;
      kslicer::ReplaceOpenCLBuiltInTypes(typeStr);
      
      json argj;
      argj["Type"] = typeStr;
      argj["Name"] = arg.name;

      bool skip = false;
      
      if(arg.name == "tid" || arg.name == "tidX") // todo: check several names ... 
      {
        skip           = true;
        foundThreadIdX = true;
        tidXName       = arg.name;
      }
  
      if(arg.name == "tidY") // todo: check several names ... 
      {
        skip           = true;
        foundThreadIdY = true;
        tidYName       = arg.name;
      }
  
      if(arg.name == "tidZ") // todo: check several names ... 
      {
        skip           = true;
        foundThreadIdZ = true;
        tidZName       = arg.name;
      }

      if(!skip)
        args.push_back(argj);
    }

    // now add all std::vector members
    //
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

    std::vector<std::string> threadIdNames;
    {
      if(foundThreadIdX)
        threadIdNames.push_back(tidXName);
      if(foundThreadIdY)
        threadIdNames.push_back(tidYName);
      if(foundThreadIdZ)
        threadIdNames.push_back(tidZName);
    }

    const std::string numThreadsName = kslicer::GetProjPrefix() + "iNumElements";
    json argSizes;
    const char* XYZ[] = {"X","Y","Z"};
    for(size_t i=0;i<threadIdNames.size();i++)
      argSizes.push_back(numThreadsName + XYZ[i]);

    
    json kernelJson;
    kernelJson["Args"]        = args;
    kernelJson["ArgSizes"]    = argSizes;
    kernelJson["Vecs"]        = vecs;
    kernelJson["Name"]        = k.name;

    std::string sourceCodeFull = kslicer::ProcessKernel(k, compiler, a_classInfo);
    std::string sourceCodeCut  = sourceCodeFull.substr(sourceCodeFull.find_first_of('{')+1);
    
    kernelJson["Source"]      = sourceCodeCut;

    kernelJson["threadDim"]   = threadIdNames.size();
    kernelJson["threadNames"] = threadIdNames;
    if(threadIdNames.size() == 1)
      kernelJson["threadName1"] = threadIdNames[0];
    if(threadIdNames.size() == 2)
      kernelJson["threadName2"] = threadIdNames[1];
    if(threadIdNames.size() == 3)
      kernelJson["threadName3"] = threadIdNames[2];

    std::stringstream strOut;
    {
      for(size_t i=0;i<threadIdNames.size();i++)
        strOut << "  const uint " << threadIdNames[i].c_str() << " = get_global_id(" << i << ");"<< std::endl; 
      
      if(threadIdNames.size() == 1)
      {
        strOut << "  if (" << threadIdNames[0].c_str() << " >= " << numThreadsName.c_str() << "X" << ")" << std::endl;                          
        strOut << "    return;";
      }
      else if(threadIdNames.size() == 2)
      {
        strOut << "  if (" << threadIdNames[0].c_str() << " >= " << numThreadsName.c_str() << "X";
        strOut << " || "   << threadIdNames[1].c_str() << " >= " << numThreadsName.c_str() << "Y" <<  ")" << std::endl;                          
        strOut << "    return;";
      }
      else if(threadIdNames.size() == 3)
      {
        strOut << "  if (" << threadIdNames[0].c_str() << " >= " << numThreadsName.c_str() << "X";
        strOut << " || "   << threadIdNames[1].c_str() << " >= " << numThreadsName.c_str() << "Y";  
        strOut << " || "   << threadIdNames[2].c_str() << " >= " << numThreadsName.c_str() << "Z" <<  ")" << std::endl;                        
        strOut << "    return;";
      }
      else
      {
        assert(false);
      }
    }
    
    kernelJson["Prolog"] = strOut.str();

    data["Kernels"].push_back(kernelJson);
  }

  inja::Environment env;
  inja::Template temp = env.parse_template(a_inFileName.c_str());
  std::string result  = env.render(temp, data);
  
  std::ofstream fout(a_outFileName);
  fout << result.c_str() << std::endl;
  fout.close();
}