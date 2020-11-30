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

std::vector<std::string> GetVarNames(const std::unordered_map<std::string, kslicer::DataLocalVarInfo>& a_locals)
{
  std::vector<std::string> localVarNames;
  localVarNames.reserve(a_locals.size());
  for(const auto& v : a_locals)
    localVarNames.push_back(v.first);
  return localVarNames;
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

std::string kslicer::PrintGeneratedClassDecl(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo, 
                                             const std::vector<MainFuncInfo>& a_methodsToGenerate)
{
  assert(a_methodsToGenerate.size() == 1);

  std::string rawname;
  {
    size_t lastindex = a_classInfo.mainClassFileName.find_last_of("."); 
    assert(lastindex != std::string::npos);
    rawname = a_classInfo.mainClassFileName.substr(0, lastindex); 
  }

  std::string folderPath = GetFolderPath(a_classInfo.mainClassFileName);
  std::string mainInclude = a_classInfo.mainClassFileInclude;
  
  MakeAbsolutePathRelativeTo(mainInclude, folderPath);

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
  //data["MainFuncDecl"]  = a_mainFuncDecl;

  data["PlainMembersUpdateFunctions"]  = "";
  data["VectorMembersUpdateFunctions"] = "";
  data["KernelsDecl"]                  = strOut2.str();   
  data["TotalDSNumber"]                = a_classInfo.allDescriptorSetsInfo.size();

  data["MainFunctions"] = std::vector<std::string>(); 
  for(const auto& mainFunc : a_methodsToGenerate)
  {
    json local;
    local["Decl"] = mainFunc.GeneratedDecl;
    local["Name"] = mainFunc.Name;
    local["LocalVarsBuffersDecl"] = GetVarNames(mainFunc.Locals);

    local["InOutVars"] = std::vector<std::string>();
    for(const auto& v : mainFunc.InOuts)
      local["InOutVars"].push_back(v.first);

    data["MainFunctions"].push_back(local);
  }

  data["KernelNames"] = std::vector<std::string>();  
  for(const auto& k : a_classInfo.kernels)
  {
    std::string kernName = k.name;
    auto pos = kernName.find("kernel_");
    if(pos != std::string::npos)
      kernName = kernName.substr(7);
    data["KernelNames"].push_back(kernName);
  }
  
  inja::Environment env;
  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, data);
  
  std::string includeFileName = rawname + "_generated.h";
  std::ofstream fout(includeFileName);
  fout << result.c_str() << std::endl;
  fout.close();

  return includeFileName;
} 

void kslicer::PrintGeneratedClassImpl(const std::string& a_declTemplateFilePath, const std::string& a_includeName, const MainClassInfo& a_classInfo,
                                      const std::vector<MainFuncInfo>& a_methodsToGenerate)
{
  
  assert(a_methodsToGenerate.size() == 1);

  const std::string& a_mainFuncName     = a_methodsToGenerate[0].Name;

  std::string folderPath  = GetFolderPath(a_includeName);
  std::string mainInclude = a_includeName;
  MakeAbsolutePathRelativeTo(mainInclude, folderPath);

  std::string rawname;
  {
    size_t lastindex = a_classInfo.mainClassFileName.find_last_of("."); 
    assert(lastindex != std::string::npos);
    rawname = a_classInfo.mainClassFileName.substr(0, lastindex); 
  }

  std::vector<std::string> kernelsCallCmdDecl(a_classInfo.kernels.size());
  for(size_t i=0;i<kernelsCallCmdDecl.size();i++)
    kernelsCallCmdDecl[i] = a_classInfo.kernels[i].DeclCmd;
 
  auto kernelDeclByName = MakeMapForKernelsDeclByName(kernelsCallCmdDecl);

  json data;
  data["Includes"]         = "";
  data["IncludeClassDecl"] = mainInclude;
  data["MainClassName"]    = a_classInfo.mainClassName;
  data["TotalDescriptorSets"] = a_classInfo.allDescriptorSetsInfo.size(); ///// ?????????????????????????????????????????????????????

  size_t allClassVarsSizeInBytes = 0;
  for(const auto& var : a_classInfo.classVariables)
    allClassVarsSizeInBytes += var.sizeInBytes;
  
  data["AllClassVarsSize"]  = allClassVarsSizeInBytes;

  data["ClassVars"] = std::vector<std::string>();
  for(const auto& v : a_classInfo.classVariables)
  {
    json local;
    local["Name"]   = v.name;
    local["Offset"] = v.offsetInTargetBuffer;
    local["Size"]   = v.sizeInBytes;
    data["ClassVars"].push_back(local);
  }

  auto predefinedNames = kslicer::GetAllPredefinedThreadIdNames();

  data["Kernels"] = std::vector<std::string>();  
  for(const auto& k : a_classInfo.allKernels)
  {
    std::string kernName = k.first;
    auto pos = kernName.find("kernel_");
    if(pos != std::string::npos)
      kernName = kernName.substr(7);
    
    json local;
    local["Name"]         = kernName;
    local["OriginalName"] = k.first;
    local["ArgCount"]     = k.second.args.size();
    local["Decl"]         = kernelDeclByName[kernName];

    local["Args"]         = std::vector<std::string>();
    size_t actualSize     = 0;
    for(const auto& arg : k.second.args)
    {
      auto elementId = std::find(predefinedNames.begin(), predefinedNames.end(), arg.name);
      if(elementId != predefinedNames.end()) // exclude predefined names from bindings
        continue;

      json argData;
      argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
      argData["Name"]  = a_mainFuncName + "_" + arg.name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
      local["Args"].push_back(argData);
      actualSize++;
    }
    local["ArgCount"] = actualSize;

    data["Kernels"].push_back(local);
  }

  data["TotalDSNumber"] = a_classInfo.allDescriptorSetsInfo.size();

  data["MainFunctions"] = std::vector<std::string>();
  for(const auto& mainFunc : a_methodsToGenerate)
  {
    json data2;
    data2["Name"]           = mainFunc.Name;
    data2["DescriptorSets"] = std::vector<std::string>();
    for(size_t i=0;i<a_classInfo.allDescriptorSetsInfo.size();i++)
    {
      auto& dsArgs = a_classInfo.allDescriptorSetsInfo[i];
  
      json local;
      local["Id"]        = i;
      local["Layout"]    = dsArgs.kernelName + "DSLayout";
      local["ArgNumber"] = dsArgs.allDescriptorSetsInfo.size();
      local["Args"]      = std::vector<std::string>();
     
      for(size_t j=0;j<dsArgs.allDescriptorSetsInfo.size();j++)
      {
        json arg;
        arg["Id"]   = j;
        arg["Name"] = mainFunc.Name + "_local." + dsArgs.allDescriptorSetsInfo[j].varName;
        local["Args"].push_back(arg);
      }
      data2["DescriptorSets"].push_back(local);
    }

    data2["MainFuncCmd"]      = mainFunc.CodeGenerated;
    data2["LocalVarsBuffers"] = std::vector<std::string>();
    for(const auto& v : a_classInfo.mainFunc[0].Locals)
    {
      json local;
      local["Name"] = mainFunc.Name + "_local." + v.second.name;
      local["Type"] = v.second.type;
      data2["LocalVarsBuffers"].push_back(local);
    }

    data["MainFunctions"].push_back(data2);
  }

  inja::Environment env;
  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, data);
  
  std::string cppFileName = rawname + "_generated.cpp";
  std::ofstream fout(cppFileName);
  fout << result.c_str() << std::endl;
  fout.close();
}
