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

std::string kslicer::PrintGeneratedClassDecl(const std::string& a_declTemplateFilePath, const MainClassInfo& a_classInfo, 
                                             const std::string& a_mainFuncDecl, const std::vector<std::string>& kernelsCallCmdDecl)
{
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
  for(size_t i=0;i<kernelsCallCmdDecl.size();i++)
  {
    if(i != 0)
      strOut2 << "  ";
    strOut2 << "virtual " << kernelsCallCmdDecl[i].c_str() << ";\n";
  }

  json data;
  data["Includes"]      = strOut.str();
  data["MainClassName"] = a_classInfo.mainClassName;
  data["MainFuncDecl"]  = a_mainFuncDecl;

  data["PlainMembersUpdateFunctions"]  = "";
  data["VectorMembersUpdateFunctions"] = "";
  data["KernelsDecl"]                  = strOut2.str();
  data["LocalVarsBuffersDecl"]         = GetVarNames(a_classInfo.mainFuncLocals);   

  data["KernelNames"] = std::vector<std::string>();  
  for(const auto& k : a_classInfo.allKernels)
  {
    std::string kernName = k.first;
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

void kslicer::PrintGeneratedClassImpl(const std::string& a_declTemplateFilePath, 
                                      const std::string& a_includeName, 
                                      const MainClassInfo& a_classInfo,
                                      const std::string& a_mainFuncCodeGen,
                                      const std::vector<std::string>& kernelsCallCmdDecl)
{
  std::string folderPath  = GetFolderPath(a_includeName);
  std::string mainInclude = a_includeName;
  MakeAbsolutePathRelativeTo(mainInclude, folderPath);

  std::string rawname;
  {
    size_t lastindex = a_classInfo.mainClassFileName.find_last_of("."); 
    assert(lastindex != std::string::npos);
    rawname = a_classInfo.mainClassFileName.substr(0, lastindex); 
  }

  std::stringstream strOut2;
  for(size_t i=0;i<kernelsCallCmdDecl.size();i++)
  {
    std::string kernDecl = kernelsCallCmdDecl[i];
    size_t voidPos = kernDecl.find("void ");
    assert(voidPos != std::string::npos);
    std::string kernDecl2 = kernDecl.substr(0, voidPos + 4) + " " + a_classInfo.mainClassName + "_Generated::" + kernDecl.substr(voidPos + 5); 
    strOut2 << kernDecl2.c_str() << "\n";
    strOut2 << "{" << std::endl;
    strOut2 << std::endl;
    strOut2 << "}" << std::endl << std::endl;
  }

  json data;
  data["Includes"]         = "";
  data["IncludeClassDecl"] = mainInclude;
  data["MainClassName"]    = a_classInfo.mainClassName;
  data["MainFuncCmd"]      = a_mainFuncCodeGen;
  data["KernelsCmd"]       = strOut2.str();
  data["TotalDescriptorSets"]  = 16;

  size_t allClassVarsSizeInBytes = 0;
  for(const auto& var : a_classInfo.classVariables)
    allClassVarsSizeInBytes += var.sizeInBytes;
  
  data["AllClassVarsSize"]  = allClassVarsSizeInBytes;

  data["LocalVarsBuffers"] = std::vector<std::string>();
  for(const auto& v : a_classInfo.mainFuncLocals)
  {
    json local;
    local["Name"] = v.second.name;
    local["Type"] = v.second.type;
    data["LocalVarsBuffers"].push_back(local);
  }

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

    local["Args"]         = std::vector<std::string>();
    size_t actualSize     = 0;
    for(const auto& arg : k.second.args)
    {
      auto elementId = std::find(predefinedNames.begin(), predefinedNames.end(), arg.name);
      if(elementId != predefinedNames.end()) // exclude predefined names from bindings
        continue;

      json argData;
      argData["Type"]  = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
      argData["Name"]  = arg.name;
      argData["Flags"] = "VK_SHADER_STAGE_COMPUTE_BIT";
      argData["Id"]    = actualSize;
      local["Args"].push_back(argData);
      actualSize++;
    }
    local["ArgCount"] = actualSize;

    data["Kernels"].push_back(local);
  }

  inja::Environment env;
  inja::Template temp = env.parse_template(a_declTemplateFilePath.c_str());
  std::string result  = env.render(temp, data);
  
  std::string cppFileName = rawname + "_generated.cpp";
  std::ofstream fout(cppFileName);
  fout << result.c_str() << std::endl;
  fout.close();
}
