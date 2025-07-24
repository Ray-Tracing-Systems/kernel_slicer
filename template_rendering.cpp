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

bool kslicer::MainClassInfo::HasBufferReferenceBind() const
{
  bool hasBufferReferenceBind = false;
  for(const auto& member : this->dataMembers) {
    auto pFound = this->allDataMembers.find(member.name);
    if(pFound != this->allDataMembers.end()) {
      if(pFound->second.bindWithRef) {
        hasBufferReferenceBind = true;
        break;
      }
    }
  }
  return hasBufferReferenceBind;
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

  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.sizeOf > b.sizeOf; });

  return result;
}

nlohmann::json kslicer::GetOriginalKernelJson(const KernelInfo& k, const MainClassInfo& a_classInfo)
{
  auto pShaderRewriter = a_classInfo.pShaderFuncRewriter;

  json allArgs = std::vector<json>();
  for(const auto& arg : k.args)
  {
    std::string typeName = pShaderRewriter->RewriteStdVectorTypeStr(arg.type);
    json argj;
    if(a_classInfo.pShaderCC->IsCUDA() && arg.IsPointer()) // strange bug with spaces when use template text rendering for that
      typeName += "* __restrict__ ";                        // 
    argj["Type"]      = typeName;
    argj["Name"]      = arg.name;
    argj["IsPointer"] = arg.IsPointer();
    argj["IsConst"]   = arg.isConstant;
    allArgs.push_back(argj);
  }
  return allArgs;
}

static inline size_t AlignedSize(const size_t a_size)
{
  size_t currSize = 4;
  while(a_size > currSize)
    currSize = currSize*2;
  return currSize;
}

nlohmann::json kslicer::PutHierarchyToJson(const kslicer::MainClassInfo::VFHHierarchy& h, 
                                           const clang::CompilerInstance& compiler,
                                           const MainClassInfo& a_classInfo,
                                           size_t& fnGroupOffset)
{
  json hierarchy;
  hierarchy["Name"]             = h.interfaceName;
  hierarchy["ObjBufferName"]    = h.objBufferName;
  hierarchy["IndirectDispatch"] = 0;
  hierarchy["IndirectOffset"]   = 0;
  hierarchy["VFHLevel"]         = h.hasIntersection ? 3 : int(h.level);
  hierarchy["HasIntersection"]  = h.hasIntersection;
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  size_t summOfFieldsSize = 0;
  auto fieldsInterface = a_classInfo.GetFieldsFromStruct(h.interfaceDecl, &summOfFieldsSize);

  hierarchy["InterfaceFields"] = std::vector<json>();
  for(auto field : fieldsInterface) 
  {
    json local;
    local["Type"] = field.first;
    local["Name"] = field.second;
    hierarchy["InterfaceFields"].push_back(local);
  }

  if(summOfFieldsSize % 8 != 0) // manually align struct to 64 bits (8 bytes) if needed
  {
    json local;
    local["Type"] = "uint";
    local["Name"] = "dummy";
    hierarchy["InterfaceFields"].push_back(local);
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    json currImpl;
    currImpl["ClassName"]       = impl.name;
    currImpl["TagName"]         = impl.tagName;
    currImpl["TagId"]           = impl.tagId;
    currImpl["MemberFunctions"] = std::vector<json>();
    currImpl["ObjBufferName"]   = h.objBufferName;
    currImpl["IsEmpty"]         = impl.isEmpty;
    for(const auto& member : impl.memberFunctions)
    {
      json local;
      local["Name"]           = member.name;
      local["NameRewritten"]  = member.nameRewritten;
      local["IsIntersection"] = member.isIntersection;
      local["Source"]         = member.srcRewritten;
      currImpl["MemberFunctions"].push_back(local);
    }

    // 'old' fields for level 1 (?)
    //
    currImpl["Fields"] = std::vector<json>();
    for(const auto& field : impl.fields)
      currImpl["Fields"].push_back(field);
    // 'new' fields for level 2 and 3
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    {
      size_t summOfFieldsSize2 = 0;
      auto fieldsImpl  = a_classInfo.GetFieldsFromStruct(impl.decl, &summOfFieldsSize2);
      auto fieldsImpl2 = fieldsInterface;
      
      // join interface and implementation
      //
      fieldsImpl2.insert(fieldsImpl2.end(), fieldsImpl.begin(), fieldsImpl.end());
      summOfFieldsSize2 += summOfFieldsSize;
  
      if(summOfFieldsSize2 % sizeof(void*) != 0 && int(h.level) >= 2)
      {
        std::cout << "  [ALIGMENT VIOLATION]: sizeof(" << impl.name << ") = " << summOfFieldsSize2 + sizeof(void*) << " which is not a multiple of " << sizeof(void*) << std::endl;
        std::cout << "  [ALIGMENT VIOLATION]: sizeof any class in VFH hierarchy with virtual functions must be multiple of " << sizeof(void*) << std::endl;
      }
  
      json local;
      local["Name"]       = impl.name;
      local["BufferName"] = impl.objBufferName; // + "_" + impl.name;
      local["Fields"]     = std::vector<json>();
      { 
        // (1) put fields
        // 
        for(auto field : fieldsImpl2) 
        {
          json local2;
          local2["Type"] = field.first;
          local2["Name"] = field.second;
          local["Fields"].push_back(local2);
        }
      }
      currImpl["DataStructure"] = local;

      bool isTriangles = false;
      for(auto x : a_classInfo.intersectionTriangle)
        if(x.first == h.interfaceName && x.second == impl.name)
          isTriangles = true;
      currImpl["IsTriangleMesh"] = isTriangles;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(impl.isEmpty) 
    //if(impl.name.find("Empty") != std::string::npos)
    {
      hierarchy["EmptyImplementation"] = currImpl;
      emptyIsFound = true;
    }
    else
      hierarchy["Implementations"].push_back(currImpl);
  }
  hierarchy["ImplAlignedSize"] = AlignedSize(h.implementations.size()+1);
  if(h.implementations.size()!= 0 && !emptyIsFound)
    std::cout << "  VFH::ALERT! Empty implementation is not found! You need to declate a default implementation class with 'Empty' prefix in its name " << std::endl; // Don't add any functions except 'GetTag()' to EmptyImpl class

  std::unordered_map<std::string, const clang::CXXRecordDecl*> retDeclHash;
  hierarchy["VirtualFunctions"] = std::vector<json>();
  for(const auto& vf : h.virtualFunctions)
  {
    json virtualFunc;
    virtualFunc["Name"] = vf.second.name;
    virtualFunc["Decl"] = vf.second.declRewritten;
    virtualFunc["Args"] = std::vector<json>();
    virtualFunc["FuncGroupOffset"] = fnGroupOffset;
    fnGroupOffset += (h.implementations.size() - 1);
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
    
    if(vf.second.retTypeDecl != nullptr && !a_classInfo.pShaderFuncRewriter->NeedsVectorTypeRewrite(vf.second.retTypeName)) // not some of predefined types
      retDeclHash[vf.second.retTypeName] = vf.second.retTypeDecl;

    hierarchy["VirtualFunctions"].push_back(virtualFunc);
  }

  hierarchy["AuxDecls"] = std::vector<json>();
  for(auto retDecl : retDeclHash) 
  {
    json declOfRetType;
    declOfRetType["Name"]   = retDecl.first;
    declOfRetType["Fields"] = std::vector<json>();
    auto fields = a_classInfo.GetFieldsFromStruct(retDecl.second);
    for(auto field : fields) 
    {
      json local;
      local["Type"] = field.first;
      local["Name"] = field.second;
      declOfRetType["Fields"].push_back(local);
    }
    hierarchy["AuxDecls"].push_back(declOfRetType);
  }

  return hierarchy;
}

nlohmann::json kslicer::PutHierarchiesDataToJson(const std::unordered_map<std::string, kslicer::MainClassInfo::VFHHierarchy>& hierarchies,
                                                 const clang::CompilerInstance& compiler,
                                                 const MainClassInfo& a_classInfo)
{
  json data = std::vector<json>();
  size_t fnGroupOffset = 0;
  
  for(const auto& p : hierarchies)
    data.push_back(PutHierarchyToJson(p.second, compiler, a_classInfo, fnGroupOffset));
  
  return data;
}

nlohmann::json kslicer::FindIntersectionHierarchy(nlohmann::json a_hierarchies)
{
  json result;
  result["Implementations"] = std::vector<json>();
  for(auto h : a_hierarchies) {
    if(h["HasIntersection"]) {
      result = h;
      break;
    }
  }
  return result;
}

nlohmann::json kslicer::ListCallableStructures(const std::unordered_map<std::string, kslicer::MainClassInfo::VFHHierarchy>& hierarchies,
                                               const clang::CompilerInstance& compiler,
                                               const MainClassInfo& a_classInfo,
                                               uint32_t& a_totalShaders)
{
  auto pShaderRewriter = a_classInfo.pShaderFuncRewriter;
  nlohmann::json data  = std::vector<json>();
  size_t fnGroupOffset = 0;

  for(const auto& h : hierarchies) {
    
    if(h.second.hasIntersection)
      continue;

    for(const auto& f : h.second.virtualFunctions) {
      
      nlohmann::json funcData;
      funcData["Name"] = f.second.name;
      funcData["Args"] = std::vector<json>();
      funcData["FuncGroupOffset"] = fnGroupOffset;

      // self offset/ptr
      {
        nlohmann::json arg;
        arg["Name"]  = "selfId";
        arg["Type"]  = "uint";
        arg["IsRet"] = false;
        funcData["Args"].push_back(arg);
      }

      // list arguments
      // 
      for (const auto* param : f.second.astNode->parameters()) {   
        std::string paramName  = param->getNameAsString();       // Получаем имя параметра
        std::string paramType  = param->getType().getAsString(); // Получаем тип параметра
        std::string paramType2 = pShaderRewriter->RewriteStdVectorTypeStr(paramType);
        nlohmann::json arg;
        arg["Name"]  = paramName;
        arg["Type"]  = paramType2;
        arg["IsRet"] = false;
        if(a_classInfo.dataClassNames.find(paramType) == a_classInfo.dataClassNames.end())
          funcData["Args"].push_back(arg);
      }

      funcData["ArgLen"] = funcData["Args"].size()-1;
      
      // get return type
      //
      if(f.second.retTypeDecl != nullptr)
      {
        const clang::Type* type = f.second.retTypeDecl->getTypeForDecl();
        if (const auto* recordType = clang::dyn_cast<clang::RecordType>(type)) {
          std::string retTypeName = recordType->getDecl()->getQualifiedNameAsString();
          nlohmann::json arg;
          arg["Name"]  = "ret";
          arg["Type"]  =  pShaderRewriter->RewriteStdVectorTypeStr(retTypeName);
          arg["IsRet"] = true;
          funcData["Args"].push_back(arg);
        }
      }
      
      fnGroupOffset += (h.second.implementations.size() - 1); // exclude empty impl
      data.push_back(funcData);
    }
  }

  a_totalShaders = uint32_t(fnGroupOffset);

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
  varJ["SupportAtomic"] = pShaderCC->SupportAtomicGlobal(second); 
  varJ["AtomicOp"]      = pShaderCC->GetAtomicImplCode(second);
  varJ["SubgroupOp"]    = pShaderCC->GetSubgroupOpCode(second);
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
                                    const nlohmann::json& uboJson,
                                    const nlohmann::json& kernelOptions,
                                    const std::vector<std::string>& usedDefines,
                                    const TextGenSettings& a_settings)
{
  auto pShaderRewriter = a_classInfo.pShaderFuncRewriter;

  auto pDefaultOpts = kernelOptions.find("all");
  if(pDefaultOpts == kernelOptions.end())
    pDefaultOpts = kernelOptions.find("default");

  std::unordered_map<std::string, DataMemberInfo> dataMembersCached;
  dataMembersCached.reserve(a_classInfo.dataMembers.size());
  for(const auto& member : a_classInfo.dataMembers)
    dataMembersCached[member.name] = member;
  for(const auto& cont : a_classInfo.usedProbably) 
  {
    if(!cont.second.isContainer)
      continue;
    DataMemberInfo containerInfo;
    containerInfo.isArray     = false;
    containerInfo.isPointer   = false;
    containerInfo.isContainer = true;
    containerInfo.name          = cont.first;
    containerInfo.type          = cont.second.containerType + std::string("<") + cont.second.containerDataType + ">";
    containerInfo.sizeInBytes   = 0; // not used by containers
    containerInfo.usedInKernel  = true;
    containerInfo.containerType = cont.second.containerType;
    containerInfo.containerDataType = cont.second.containerDataType;
    containerInfo.usage = kslicer::DATA_USAGE::USAGE_USER;
    containerInfo.kind  = cont.second.info.kind;
    dataMembersCached[cont.first] = containerInfo;
  }
  //for (const auto& nk : a_classInfo.kernels) {
  //  for(const auto& container : nk.second.usedContainers)
  //  {
  //    if(dataMembersCached.find(container.first) != dataMembersCached.end())
  //      continue;
  //
  //    //DataMemberInfo containerInfo;
  //    //containerInfo.isArray     = false;
  //    //containerInfo.isPointer   = false;
  //    //containerInfo.isContainer = true;
  //    //containerInfo.name          = cont.first;
  //    //containerInfo.type          = cont.second.containerType + std::string("<") + cont.second.containerDataType + ">";
  //    //containerInfo.sizeInBytes   = 0; // not used by containers
  //    //containerInfo.usedInKernel  = true;
  //    //containerInfo.containerType = cont.second.containerType;
  //    //containerInfo.containerDataType = cont.second.containerDataType;
  //    //containerInfo.usage = kslicer::DATA_USAGE::USAGE_USER;
  //    //containerInfo.kind  = cont.second.info.kind;
  //    //dataMembersCached[container.first] = containerInfo;
  //  }
  //}

  std::unordered_map<std::string, kslicer::ShittyFunction> shittyFunctions; //
  if(!a_classInfo.pShaderCC->BuffersAsPointersInShaders())
  {
    for(const auto& k : a_classInfo.kernels)
    {
      for(auto f : k.second.shittyFunctions)
        shittyFunctions[f.originalName] = f;
    }
  }

  const bool hasBufferReferenceBind = a_classInfo.HasBufferReferenceBind();

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
  data["UseCallable"]        = a_settings.enableCallable;
  data["HasAllRefs"]         = bool(a_classInfo.m_allRefsFromVFH.size() != 0) || hasBufferReferenceBind;
  data["UsePersistentThreads"] = a_classInfo.persistentRTV;
  data["WGPUMode"]           = a_classInfo.pShaderCC->IsWGPU();

  data["VectorBufferRefs"] = std::vector<json>();
  for(const auto& v : a_classInfo.dataMembers)
  {
    if(v.isContainer && kslicer::IsVectorContainer(v.containerType))
    {
      kslicer::MainClassInfo::VFHHierarchy hierarchy;
      MainClassInfo::VFH_LEVEL level = MainClassInfo::VFH_LEVEL_1;
      if(a_classInfo.IsVFHBuffer(v.name, &level, &hierarchy))
        continue;
      
      json local;
      local["Name"] = v.name; 
      local["Type"] = pShaderRewriter->RewriteStdVectorTypeStr(v.containerDataType);
      auto pFound = a_classInfo.allDataMembers.find(v.name);
      if(pFound != a_classInfo.allDataMembers.end() && pFound->second.bindWithRef) 
        data["VectorBufferRefs"].push_back(local);
    }
  }

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
  data["UBO"] = uboJson;

  // (2) declarations of struct, constants and typedefs inside class
  //
  std::unordered_set<std::string> excludedNames; 
  {
    for(auto pair : a_classInfo.m_setterVars)
      excludedNames.insert(kslicer::CleanTypeName(pair.second));
  }

  std::unordered_set<std::string> excludedConstantsFromVFH;
  {
    for(const auto& p : a_classInfo.m_vhierarchy)
      for(const auto& decl : p.second.usedDecls)
        excludedConstantsFromVFH.insert(decl.name);
  }

  data["UserTypeDefs"] = std::vector<json>();
  for(auto userdef : a_classInfo.userTypedefs) {
    json local;
    local["Original"]  = userdef.first;
    local["Redefined"] = userdef.second;
    data["UserTypeDefs"].push_back(local);
  }

  data["ClassDecls"] = std::vector<json>();
  std::map<std::string, kslicer::DeclInClass> specConsts;
  for(const auto decl : usedDecl)
  {
    if(!decl.extracted)
      continue;
    if(excludedNames.find(decl.type) != excludedNames.end())
      continue;

    if(a_classInfo.pHostCC->HasSpecConstants() && a_classInfo.pShaderCC->IsGLSL() && decl.name.find("KSPEC_") != std::string::npos) { // process specialization constants, remove them from normal constants
      std::string val = kslicer::GetRangeSourceCode(decl.srcRange, compiler);
      specConsts[val] = decl;
      continue;
    }
    
    if(excludedConstantsFromVFH.find(decl.name) != excludedConstantsFromVFH.end())
      continue;

    json c_decl;
    c_decl["Text"]    = a_classInfo.pShaderCC->PrintHeaderDecl(decl, compiler, pShaderRewriter);
    c_decl["InClass"] = decl.inClass;
    c_decl["IsType"]  = (decl.kind == DECL_IN_CLASS::DECL_STRUCT); // || (decl.kind == DECL_IN_CLASS::DECL_TYPEDEF);
    c_decl["IsTdef"]  = (decl.kind == DECL_IN_CLASS::DECL_TYPEDEF); // || (decl.kind == DECL_IN_CLASS::DECL_TYPEDEF);
    c_decl["Type"]    = kslicer::CleanTypeName(decl.type);
    data["ClassDecls"].push_back(c_decl);
  }

  // (3) local functions preprocess
  //
  std::unordered_map<std::string, kslicer::FuncData> cachedFunc;
  {
    for (const auto& f : usedFunctions)
    {
      cachedFunc[f.name] = f;
      auto pShit = shittyFunctions.find(f.name);      // exclude shittyFunctions from 'LocalFunctions'
      if(pShit != shittyFunctions.end())
        continue;
    }
  }

  ShaderFeatures shaderFeatures = a_classInfo.globalShaderFeatures;
  for(auto k : a_classInfo.kernels)
    shaderFeatures = shaderFeatures || k.second.shaderFeatures;

  data["GlobalUseInt8"]         = shaderFeatures.useByteType;
  data["GlobalUseInt16"]        = shaderFeatures.useShortType;
  data["GlobalUseInt64"]        = shaderFeatures.useInt64Type;
  data["GlobalUseFloat64"]      = shaderFeatures.useFloat64Type;
  data["GlobalUseHalf"]         = shaderFeatures.useHalfType;
  data["GlobalUseFloatAtomics"] = shaderFeatures.useFloatAtomicAdd;

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

  std::map<uint64_t, json>        allUsedMemberFunctions;
  std::unordered_set<std::string> excludedMemberFunctions;
 
  if(a_classInfo.persistentRTV)
  {
    excludedMemberFunctions.insert("RTVPersistent_ThreadId");
    excludedMemberFunctions.insert("RTVPersistent_SetIter");
    excludedMemberFunctions.insert("RTVPersistent_Iters");
    excludedMemberFunctions.insert("RTVPersistent_IsFirst");
    excludedMemberFunctions.insert("RTVPersistent_ReduceAdd4f");
  }

  bool haveReduceAdd = false;

  data["Kernels"] = std::vector<json>();
  for (const auto& nk : kernels)
  {
    const auto& k = nk.second;
    std::cout << "  processing " << k.name << std::endl;

    auto commonArgs = a_classInfo.GetKernelCommonArgs(k);
    auto tidArgs    = a_classInfo.GetKernelTIDArgs(k);
    uint VArgsSize  = 0;
    uint MArgsSize  = 0;
    bool isTextureArrayUsedInThisKernel = false;

    json args = std::vector<json>();
    for(auto commonArg : commonArgs)
    {
      json argj;
      std::string buffType1 = a_classInfo.pShaderCC->ProcessBufferType(commonArg.type);
      std::string buffType2 = pShaderRewriter->RewriteStdVectorTypeStr(buffType1);
      argj["Type"]     = commonArg.isImage ? commonArg.imageType : buffType2;
      argj["Name"]     = commonArg.name;
      argj["IsUBO"]    = commonArg.isDefinedInClass;
      argj["IsImage"]  = commonArg.isImage;
      argj["IsImageArray"]  = false;
      argj["IsAccelStruct"] = false;
      argj["NeedFmt"]       = !commonArg.isSampler;
      argj["ImFormat"]      = commonArg.imageFormat;
      argj["IsPointer"]     = commonArg.isPointer;
      argj["IsMember"]      = false;
      argj["IsSingle"]      = false;
      argj["IsConst"]       = commonArg.isConstant;
      argj["IsVFHBuffer"]   = false;
      argj["VFHLevel"]      = 0;
      argj["WithBuffRef"]   = false;

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
    bool usedCombinedImageSamplers = false;
    json rtxNames = std::vector<json>();
    for(const auto& container : k.usedContainers)
    {
      if(container.second.bindWithRef) // do not pass it to shader via descriptor set because we pass it with separate buffer reference
        continue;

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
      if(a_classInfo.pShaderCC->BuffersAsPointersInShaders())
        buffType2 += "*";

      json argj;
      argj["Type"]       = buffType2;
      argj["Name"]       = pVecMember->second.name;
      argj["IsUBO"]      = false;
      argj["IsImage"]    = false;
      argj["IsImageArray"]  = false;
      argj["IsAccelStruct"] = false;
      argj["IsPointer"]     = (pVecMember->second.kind == kslicer::DATA_KIND::KIND_VECTOR);
      argj["IsMember"]      = true;
      argj["IsSingle"]      = pVecMember->second.isSingle;
      argj["IsConst"]       = pVecMember->second.isConst;
      argj["WithBuffRef"]   = container.second.bindWithRef;

      ////////////////////////////////////////////////////////////////////
      MainClassInfo::VFH_LEVEL level = MainClassInfo::VFH_LEVEL_1;
      bool isVFHBuffer       = a_classInfo.IsVFHBuffer(pVecMember->second.name, &level);
      argj["IsVFHBuffer"]    = isVFHBuffer;
      argj["VFHLevel"]       = int(level);
      if(isVFHBuffer && int(level) >= 2)
        argj["Type"] = "uvec2";
      ////////////////////////////////////////////////////////////////////

      std::string ispcConverted = argj["Name"];
      if(argj["IsPointer"])
        ispcConverted = ConvertVecTypesToISPC(argj["Type"], argj["Name"]);
      argj["NameISPC"] = ispcConverted;
      MArgsSize++;

      if(pVecMember->second.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED)
      {
        argj["IsImage"]  = true;
        argj["IsImageArray"] = false;
        argj["Type"]     = "sampler2D";
        argj["NeedFmt"]  = false;
        argj["ImFormat"] = "";
        argj["SizeOffset"] = 0;
        usedCombinedImageSamplers = true;
      }
      else if(pVecMember->second.kind == kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY)
      {
        argj["Name"]     = pVecMember->second.name + "[]";
        argj["NameSam"]  = pVecMember->second.name + "_sam[]";
        argj["IsImage"]  = true;
        argj["IsImageArray"] = true;
        argj["Type"]     = "sampler2D";
        argj["NeedFmt"]  = false;
        argj["ImFormat"] = "";
        argj["SizeOffset"] = 0;
        isTextureArrayUsedInThisKernel = true;
        usedCombinedImageSamplers      = true;
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
    }

    if(k.isIndirect && !a_classInfo.pShaderCC->IsISPC()) // add indirect buffer to shaders
    {
      json argj;
      argj["Type"]       = a_classInfo.pShaderCC->IndirectBufferDataType();
      argj["Name"]       = "m_indirectBuffer";
      argj["IsUBO"]      = false;
      argj["IsPointer"]  = false;
      argj["IsImage"]    = false;
      argj["IsAccelStruct"] = false;
      argj["IsMember"]   = false;
      argj["IsSingle"]   = false;
      argj["IsConst"]    = false;
      argj["NameISPC"] = argj["Name"];
      argj["IsVFHBuffer"]   = false;
      argj["VFHLevel"]      = 0;
      argj["WithBuffRef"]   = false;
      args.push_back(argj);
    }

    const auto userArgsArr = GetUserKernelArgs(k.args);
    json userArgs = std::vector<json>();
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
      argj["IsConst"]   = arg.isConstant;
      argj["NameISPC"]  = argj["Name"];
      argj["WithBuffRef"] = false;
      userArgs.push_back(argj);
    }

    json allArgs = GetOriginalKernelJson(k, a_classInfo);

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
    json reductionVars = std::vector<json>();
    json reductionArrs = std::vector<json>();
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
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint32_t totalCallableShaders = 0;
    auto selectedVFH = a_classInfo.SelectVFHOnlyUsedByKernel(a_classInfo.m_vhierarchy, k);
    kernelJson["Hierarchies"]           = kslicer::PutHierarchiesDataToJson(selectedVFH, compiler, a_classInfo); 
    kernelJson["CallableStructures"]    = kslicer::ListCallableStructures(selectedVFH, compiler, a_classInfo, totalCallableShaders);
    kernelJson["CallablesTotal"]        = totalCallableShaders;
    kernelJson["IntersectionHierarhcy"] = kslicer::FindIntersectionHierarchy(kernelJson["Hierarchies"]);
    kernelJson["HasIntersectionShader2"]= k.hasIntersectionShader2;
    if(k.hasIntersectionShader2)
    {
      kernelJson["IS2_AccObjName"] = k.intersectionShader2Info.accObjName;
      kernelJson["IS2_BufferName"] = k.intersectionShader2Info.bufferName;
      kernelJson["IS2_ShaderName"] = k.intersectionShader2Info.shaderName;
      kernelJson["IS2_TriTagName"] = k.intersectionShader2Info.triTagName;
    }

    // add primitive remap tables for intesection shaders
    //
    kernelJson["IntersectionShaderRemaps"] = std::vector<json>();
    for(const auto& vfh : a_classInfo.m_vhierarchy)
    {
      if(!vfh.second.hasIntersection)
        continue;
  
      json local;
      local["Name"]          = vfh.second.interfaceName;
      local["BType"]         = "RemapTable";
      local["DType"]         = "uvec2";
      local["InterfaceName"] = vfh.second.interfaceName;
      local["AccelName"]     = vfh.second.accStructName;
      kernelJson["IntersectionShaderRemaps"].push_back(local);
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    kernelJson["RedLoop1"] = std::vector<std::string>();
    kernelJson["RedLoop2"] = std::vector<std::string>();

    const uint32_t blockSize = k.wgSize[0]*k.wgSize[1]*k.wgSize[2];
    for (uint c = blockSize/2; c>k.warpSize; c/=2)
      kernelJson["RedLoop1"].push_back(c);
    for (uint c = k.warpSize; c>0; c/=2)
      kernelJson["RedLoop2"].push_back(c);

    kernelJson["UseSubGroups"] = k.enableSubGroups;
    
    //if(k.name == "kernel2D_BlurX")
    //{
    //  int a = 2;
    //}

    // explicitly override const flags for all args
    //
    if(kernelOptions != nullptr) {
      if(kernelOptions.find(k.name) != kernelOptions.end()) {
        
        auto thisKernelOptions = kernelOptions[k.name];
        if(thisKernelOptions == nullptr && pDefaultOpts != kernelOptions.end())
          thisKernelOptions = (*pDefaultOpts);

        if(thisKernelOptions["nonConstantData"] != nullptr) {
          auto nonConstData = thisKernelOptions["nonConstantData"];
          for(auto& arg : args) {
            if(arg["Name"] == nullptr)
              continue;
            std::string name = arg["Name"].get<std::string>();
            if(nonConstData[name.c_str()] != nullptr) {
              bool isConst = (nonConstData[name.c_str()].get<int>() == 0);
              arg["IsConst"] = isConst;
            }
            else 
              arg["IsConst"] = true;
          }
        }
      }
    }

    kernelJson["LastArgNF1"]   = VArgsSize + MArgsSize;
    kernelJson["LastArgNF"]    = VArgsSize; // Last Argument No Flags
    kernelJson["LastArgAll"]   = allArgs.size() - 1;
    kernelJson["Args"]         = args;
    kernelJson["UserArgs"]     = userArgs;
    kernelJson["OriginalArgs"] = allArgs;
    kernelJson["Name"]         = k.name;
    kernelJson["UBOBinding"]   = args.size(); // for circle
    kernelJson["HasEpilog"]    = k.isBoolTyped || reductionVars.size() != 0 || reductionArrs.size() != 0;
    kernelJson["IsBoolean"]    = k.isBoolTyped;
    kernelJson["SubjToRed"]    = reductionVars;
    kernelJson["ArrsToRed"]    = reductionArrs;
    kernelJson["FinishRed"]    = needFinishReductionPass;
    kernelJson["NeedTexArray"] = isTextureArrayUsedInThisKernel;
    kernelJson["UseCombinedImageSampler"] = usedCombinedImageSamplers;
    kernelJson["ContantUBO"]              = a_settings.uboIsAlwaysConst;
    kernelJson["UniformUBO"]              = a_settings.uboIsAlwaysUniform; 
    kernelJson["WarpSize"]     = k.warpSize;
    kernelJson["InitSource"]   = "";
    
    kernelJson["RTXNames"]        = rtxNames;
    kernelJson["UseAccelS"]       = (rtxNames.size() > 0);
    kernelJson["UseInt8"]         = k.shaderFeatures.useByteType;
    kernelJson["UseInt16"]        = k.shaderFeatures.useShortType;
    kernelJson["UseInt64"]        = k.shaderFeatures.useInt64Type;
    kernelJson["UseFloat64"]      = k.shaderFeatures.useFloat64Type;
    kernelJson["UseHalf"]         = k.shaderFeatures.useHalfType;
    kernelJson["UseFloatAtomics"] = k.shaderFeatures.useFloatAtomicAdd;
    kernelJson["UseBlockReduce"]  = k.useBlockOperations;

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
      std::shared_ptr<KernelRewriter> pRewriter = a_classInfo.pShaderCC->MakeKernRewriter(rewrite2, compiler, &a_classInfo, const_cast<kslicer::KernelInfo&>(k), std::string(""));

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

    kernelJson["SpecConstants"] = std::vector<json>();
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
      auto pVisitorK = a_classInfo.pShaderCC->MakeKernRewriter(rewrite2, compiler, &a_classInfo, const_cast<kslicer::KernelInfo&>(k), "");
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
    std::string threadOffsetStr = a_classInfo.pShaderCC->RTVGetFakeOffsetExpression(k, a_classInfo.GetKernelTIDArgs(k));

    kernelJson["shouldCheckExitFlag"] = k.checkThreadFlags;
    kernelJson["checkFlagsExpr"]      = "//xxx//";
    kernelJson["ThreadOffset"]        = threadOffsetStr;
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
        std::string exprContent      = a_classInfo.pShaderCC->ReplaceSizeCapacityExpr(k.loopIters[0].sizeText);
        if(a_classInfo.pShaderCC->IsCUDA() && a_classInfo.placeVectorsInUBO)
          exprContent = std::string("ubo.") + exprContent;
        kernelJson["IndirectSizeX"]  = a_classInfo.pShaderCC->UBOAccess(exprContent);
        kernelJson["IndirectStartX"] = kernelJson["ThreadIds"][0]["Start"];
      }

      if(k.loopIters.size() > 1)
      {
        std::string exprContent      = a_classInfo.pShaderCC->ReplaceSizeCapacityExpr(k.loopIters[1].sizeText);
        if(a_classInfo.pShaderCC->IsCUDA() && a_classInfo.placeVectorsInUBO)
          exprContent = std::string("ubo.") + exprContent;
        kernelJson["IndirectSizeY"]  = a_classInfo.pShaderCC->UBOAccess(exprContent);
        kernelJson["IndirectStartY"] = kernelJson["ThreadIds"][1]["Start"];
      }

      if(k.loopIters.size() > 2)
      {
        std::string exprContent      = a_classInfo.pShaderCC->ReplaceSizeCapacityExpr(k.loopIters[2].sizeText);
        if(a_classInfo.pShaderCC->IsCUDA() && a_classInfo.placeVectorsInUBO)
          exprContent = std::string("ubo.") + exprContent;
        kernelJson["IndirectSizeZ"]  = a_classInfo.pShaderCC->UBOAccess(exprContent);
        kernelJson["IndirectStartZ"] = kernelJson["ThreadIds"][2]["Start"];
      }
       
      kernelJson["IndirectOffset"] = k.indirectBlockOffset;
      kernelJson["threadSZName1"]  = "kgen_iNumElementsX"; // TODO: get this for inirect diapatch with CUDA
      kernelJson["threadSZName2"]  = "kgen_iNumElementsY"; // TODO: get this for inirect diapatch with CUDA
      kernelJson["threadSZName3"]  = "kgen_iNumElementsZ"; // TODO: get this for inirect diapatch with CUDA
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

    kernelJson["MemberFunctions"] = std::vector<json>();
    if(k.usedMemberFunctions.size() > 0)
    {
      clang::Rewriter rewrite2;
      rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
      auto pVisitorF = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);
      auto pVisitorK = a_classInfo.pShaderCC->MakeKernRewriter(rewrite2, compiler, &a_classInfo, const_cast<kslicer::KernelInfo&>(k), "");
      pVisitorK->ClearUserArgs();
      pVisitorK->processFuncMember = true; // signal that we process function member, not the kernel itself

      for(auto& f : k.usedMemberFunctions)
      { 
        if(!f.second.isMember) // may happen to meet non member here due to VFH hierarchy analysis  
          continue;
        bool fromVFH = false;
        if(f.second.astNode->isVirtualAsWritten()) {
          for(const auto& h : a_classInfo.m_vhierarchy) {
            auto p = h.second.virtualFunctions.find(f.second.name);
            if(p != h.second.virtualFunctions.end())
            {
              fromVFH = true;
              break;
            }
          }
        }

        if(fromVFH) // skip virtual functions because they are proccesed else-where
          continue;

        if(excludedMemberFunctions.find(f.second.name) != excludedMemberFunctions.end())
          continue;

        auto funcNode    = const_cast<clang::FunctionDecl*>(f.second.astNode);
        auto funcDataPtr = const_cast<kslicer::FuncData*>  (&f.second);

        pVisitorF->SetCurrFuncInfo(funcDataPtr); // pass auxilary function data inside pVisitorF
        pVisitorK->SetCurrFuncInfo(funcDataPtr);
        const std::string funcDeclText = pVisitorF->RewriteFuncDecl(funcNode);
        const std::string funcBodyText = pVisitorK->RecursiveRewrite(funcNode->getBody());
        pVisitorF->ResetCurrFuncInfo();
        pVisitorK->ResetCurrFuncInfo();
        
        json funData;
        funData["Decl"]       = funcDeclText;
        funData["Text"]       = funcDeclText + funcBodyText;
        funData["IsRayQuery"] = (funcDeclText.find("CRT_Hit") == 0 && funcDeclText.find("RayQuery_") != std::string::npos);
        funData["UseVFH"]     = false; 
        kernelJson["MemberFunctions"].push_back(funData);
        
        auto hash = kslicer::GetHashOfSourceRange(funcNode->getSourceRange());
        allUsedMemberFunctions[hash] = funData;
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
      auto rewritten = pVisitorF->RewriteFunction(funcNode);
      kernelJson["ShityFunctions"].push_back(rewritten.funText());
    }

    kernelJson["Subkernels"]  = std::vector<json>();
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

    kernelJson["IsSingleThreaded"] = false; 
    kernelJson["TemplatedFun"] = std::vector<json>();
    for(auto x : k.templatedFunctionsLM) {
      json local;
      local["Name"]  = x.second.name;
      local["NameT"] = x.second.nameOriginal;
      local["Type0"] = x.second.types[0];
      local["Type1"] = x.second.types[1];
      local["Type2"] = x.second.types[2];
      local["Type3"] = x.second.types[3];
      kernelJson["TemplatedFun"].push_back(local);
      if(x.second.nameOriginal == "ReduceAdd")
        haveReduceAdd = true;
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
        kernelJson["IsSingleThreaded"] = true;
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
      kernelJson["IsSingleThreaded"] = true;
      data["Kernels"].push_back(kernelJson);
    }

  } // for (const auto& nk : kernels)
  
  data["HaveReduceAdd"] = haveReduceAdd;

  // (5) generate local functions
  //
  data["LocalFunctions"] = std::vector<json>();
  {
    clang::Rewriter rewrite2;
    rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
    auto pVisitorF = a_classInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &a_classInfo);

    for (const auto& f : usedFunctions)
    {
      cachedFunc[f.name] = f;
      auto pShit = shittyFunctions.find(f.name);      // exclude shittyFunctions from 'LocalFunctions'
      if(pShit != shittyFunctions.end() || f.isMember)
        continue;
      
      //f.astNode->dump();
      pVisitorF->TraverseDecl(const_cast<clang::FunctionDecl*>(f.astNode));
      
      auto p = a_classInfo.m_functionsDone.find(GetHashOfSourceRange(f.astNode->getBody()->getSourceRange()));
      if(p == a_classInfo.m_functionsDone.end())
      {
        std::cout << "  [PrepareJsonForKernels]: ALERT! function " << f.name << " is not found in 'm_functionsDone'" << std::endl;
        continue;
      }
  
      data["LocalFunctions"].push_back(p->second.funText());
      shaderFeatures = shaderFeatures || pVisitorF->GetShaderFeatures();
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

  data["AllMemberFunctions"] = std::vector<json>();
  for(const auto& pair : allUsedMemberFunctions)
    data["AllMemberFunctions"].push_back(pair.second);

  return data;
}

std::string kslicer::IShaderCompiler::ReplaceSizeCapacityExpr(const std::string& a_str) const
{
  const auto posOfPoint = a_str.find(".");
  if(posOfPoint != std::string::npos)
  {
    const std::string memberNameA = a_str.substr(0, posOfPoint);
    const std::string fname       = a_str.substr(posOfPoint+1);

    //if(IsCUDA() && )

    return memberNameA + "_" + fname.substr(0, fname.find("("));
  }
  else
    return a_str;
}

