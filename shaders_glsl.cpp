#include "kslicer.h"
#include "template_rendering.h"
#include <iostream>

#ifdef _WIN32
  #include <sys/types.h>
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::GLSLCompiler::GLSLCompiler(const std::string& a_prefix) : m_suffix(a_prefix)
{

}

void kslicer::GLSLCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  const auto& mainClassFileName = a_codeInfo->mainClassFileName;
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;

  #ifdef _WIN32
  const std::string scriptName = "build.bat";
  #else
  const std::string scriptName = "build.sh";
  #endif

  std::filesystem::path folderPath = mainClassFileName.parent_path();
  std::filesystem::path shaderPath = folderPath / this->ShaderFolder();
  std::filesystem::path incUBOPath = folderPath / "include";
  std::filesystem::create_directory(shaderPath);
  std::filesystem::create_directory(incUBOPath);

  // generate header for all used functions in GLSL code
  //
  std::string headerCommon = "common" + ToLowerCase(m_suffix) + ".h";
  std::filesystem::path templatesFolder("templates_glsl");
  kslicer::ApplyJsonToTemplate(templatesFolder / "common_generated.h", shaderPath / headerCommon, a_kernelsJson);

  // now generate all glsl shaders
  //
  const std::filesystem::path templatePath       = templatesFolder / (a_codeInfo->megakernelRTV ? "generated_mega.glsl" : "generated.glsl");
  const std::filesystem::path templatePathUpdInd = templatesFolder / "update_indirect.glsl";
  const std::filesystem::path templatePathRedFin = templatesFolder / "reduction_finish.glsl";
  const std::filesystem::path templatePathIntShd = templatesFolder / "intersection_shader.glsl";
  const std::filesystem::path templatePathHitShd = templatesFolder / "closest_hit_shader.glsl";
  const std::filesystem::path templatePathCalShd = templatesFolder / "callable_shader.glsl";

  nlohmann::json copy, kernels, intersections;
  for (auto& el : a_kernelsJson.items())
  {
    //std::cout << el.key() << std::endl;
    if(std::string(el.key()) == "Kernels")
      kernels = a_kernelsJson[el.key()];
    else
      copy[el.key()] = a_kernelsJson[el.key()];
  }
  
  //std::cout << "shaderPath = " << shaderPath.c_str() << std::endl;

  bool needRTDummies = false;
  bool rcHitForIntersectionIsGenerated = false;

  std::ofstream buildSH(shaderPath / scriptName);
  #if not __WIN32__
  buildSH << "#!/bin/sh" << std::endl;
  #endif
  for(auto& kernel : kernels.items())
  {
    nlohmann::json currKerneJson = copy;
    currKerneJson["Kernel"] = kernel.value();

    std::string kernelName     = std::string(kernel.value()["Name"]);
    bool useRayTracingPipeline = kernel.value()["UseRayGen"];
    const bool vulkan11        = kernel.value()["UseSubGroups"];
    const bool vulkan12        = useRayTracingPipeline;
    needRTDummies              = needRTDummies || useRayTracingPipeline;
    if(useRayTracingPipeline) 
    {
      //std::ofstream file(a_codeInfo->mainClassFileName.parent_path() / "z_debug.json");
      //file << std::setw(2) << currKerneJson; //
      //file.close();
      
      std::unordered_set<std::string> intersectionShaders;
      for(auto impl : currKerneJson["Kernel"]["IntersectionHierarhcy"]["Implementations"]) 
      {
        nlohmann::json intersectionShader;
        for(auto f : impl["MemberFunctions"])
          if(f["IsIntersection"])
            intersectionShader = f;
        
        if(intersectionShader.empty())
          continue;
        
        //std::ofstream file(a_codeInfo->mainClassFileName.parent_path() / "z_intersection_shader.json");
        //file << std::setw(2) << intersectionShader; //
        //file.close();

        nlohmann::json ISData = copy;
        ISData["Kernel"]             = currKerneJson["Kernel"];
        ISData["Implementation"]     = impl;
        ISData["IntersectionShader"] = intersectionShader;
        std::string outFileName_RHIT = std::string(impl["ClassName"]) + "_" + std::string(intersectionShader["Name"]) + "_hit.glsl";
        std::string outFileName_RINT = std::string(impl["ClassName"]) + "_" + std::string(intersectionShader["Name"]) + "_int.glsl";

        if(intersectionShaders.find(outFileName_RHIT) != intersectionShaders.end())
          continue;
         
        intersectionShaders.insert(outFileName_RHIT);

        if(!rcHitForIntersectionIsGenerated)
        {
          kslicer::ApplyJsonToTemplate(templatePathHitShd.c_str(), shaderPath / "z_trace_custom_hit.glsl", ISData);
          buildSH << "glslangValidator -V --target-env vulkan1.2 -S rchit " << "z_trace_custom_hit.glsl" << " -o " << "z_trace_custom_hit.glsl" << ".spv" << " -DGLSL -I.. ";
          for(auto folder : ignoreFolders)
            buildSH << "-I" << folder.c_str() << " ";
          buildSH << std::endl;
          rcHitForIntersectionIsGenerated = false;
        }

        //kslicer::ApplyJsonToTemplate(templatePathHitShd.c_str(), shaderPath / outFileName_RHIT, ISData);
        kslicer::ApplyJsonToTemplate(templatePathIntShd.c_str(), shaderPath / outFileName_RINT, ISData);

        //buildSH << "glslangValidator -V --target-env vulkan1.2 -S rchit " << outFileName_RHIT.c_str() << " -o " << outFileName_RHIT.c_str() << ".spv" << " -DGLSL -I.. ";
        //for(auto folder : ignoreFolders)
        //  buildSH << "-I" << folder.c_str() << " ";
        //buildSH << std::endl;
        buildSH << "glslangValidator -V --target-env vulkan1.2 -S rint " << outFileName_RINT.c_str() << " -o " << outFileName_RINT.c_str() << ".spv" << " -DGLSL -I.. ";
        for(auto folder : ignoreFolders)
          buildSH << "-I" << folder.c_str() << " ";
        buildSH << std::endl;
      }
      
      if(a_settings.enableCallable)
      {
        for(auto hierarchy : currKerneJson["Kernel"]["Hierarchies"]) {
          for(auto impl : hierarchy["Implementations"]) {
            for(const auto& member : impl["MemberFunctions"]) {
            
              nlohmann::json CSData    = copy;
              CSData["Kernel"]         = currKerneJson["Kernel"];
              CSData["Implementation"] = impl;
              CSData["MemberName"]     = member["Name"];
  
              std::string outFileName  = a_codeInfo->RemoveKernelPrefix(kernelName) + "_" + std::string(impl["ClassName"]) + "_" + std::string(member["Name"]) + "_call.glsl";
        
              kslicer::ApplyJsonToTemplate(templatePathCalShd.c_str(), shaderPath / outFileName, CSData);
              
              buildSH << "glslangValidator -V --target-env vulkan1.2 -S rcall " << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
              for(auto folder : ignoreFolders)
                buildSH << "-I" << folder.c_str() << " ";
              buildSH << std::endl;
            }
          }
        }
      }
    }

    std::string outFileName = kernelName + (useRayTracingPipeline ? "RGEN.glsl" : ".comp");
    std::filesystem::path outFilePath = shaderPath / outFileName;
    kslicer::ApplyJsonToTemplate(templatePath.c_str(), outFilePath, currKerneJson);

    buildSH << "glslangValidator -V ";
    if(vulkan12)
      buildSH << "--target-env vulkan1.2 ";
    else if(vulkan11)
      buildSH << "--target-env vulkan1.1 ";
    if(useRayTracingPipeline)
      buildSH << "-S rgen ";
    buildSH << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
    for(auto folder : ignoreFolders)
      buildSH << "-I" << folder.c_str() << " ";
    buildSH << std::endl;

    if(kernel.value()["IsIndirect"])
    {
      outFileName = kernelName + "_UpdateIndirect.comp";
      outFilePath = shaderPath / outFileName;
      kslicer::ApplyJsonToTemplate(templatePathUpdInd.c_str(), outFilePath, currKerneJson);
      buildSH << "glslangValidator -V ";
      if(vulkan11)
        buildSH << "--target-env vulkan1.1 ";
      buildSH << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
      for(auto folder : ignoreFolders)
       buildSH << "-I" << folder.c_str() << " ";
      buildSH << std::endl;
    }

    if(kernel.value()["FinishRed"])
    {
      outFileName = kernelName + "_Reduction.comp";
      outFilePath = shaderPath / outFileName;
      kslicer::ApplyJsonToTemplate(templatePathRedFin.c_str(), outFilePath, currKerneJson);
      buildSH << "glslangValidator -V ";
      if(vulkan11)
        buildSH << "--target-env vulkan1.1 ";
      buildSH << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
      for(auto folder : ignoreFolders)
       buildSH << "-I" << folder.c_str() << " ";
      buildSH << std::endl;
    }
  }

  if(a_codeInfo->usedServiceCalls.find("memcpy") != a_codeInfo->usedServiceCalls.end())
  {
    nlohmann::json dummy;
    kslicer::ApplyJsonToTemplate(templatesFolder / "z_memcpy.glsl", shaderPath / "z_memcpy.comp", dummy); // just file copy actually
    buildSH << "glslangValidator -V z_memcpy.comp -o z_memcpy.comp.spv" << std::endl;
  }
  if(a_codeInfo->usedServiceCalls.find("MatMulTranspose") != a_codeInfo->usedServiceCalls.end())
  {
    nlohmann::json dummy;
    kslicer::ApplyJsonToTemplate(templatesFolder / "z_matMulTranspose.glsl", shaderPath / "z_matMulTranspose.comp", dummy); // just file copy actually
    buildSH << "glslangValidator -V z_matMulTranspose.comp -o z_matMulTranspose.comp.spv" << std::endl;
  }

  if(a_codeInfo->usedServiceCalls.find("exclusive_scan") != a_codeInfo->usedServiceCalls.end() ||
     a_codeInfo->usedServiceCalls.find("inclusive_scan") != a_codeInfo->usedServiceCalls.end())
  {
    for(auto scanImpl : a_codeInfo->serviceCalls)
    {
      if (scanImpl.second.opName == "scan")
      {
        nlohmann::json params;
        params["Type"] = scanImpl.second.dataTypeName;

        kslicer::ApplyJsonToTemplate(templatesFolder / "z_scan_block.glsl",     shaderPath / ("z_scan_" + scanImpl.second.dataTypeName + "_block.comp"), params);
        kslicer::ApplyJsonToTemplate(templatesFolder / "z_scan_propagate.glsl", shaderPath / ("z_scan_" + scanImpl.second.dataTypeName + "_propagate.comp"), params);
        buildSH << "glslangValidator -V z_scan_" + scanImpl.second.dataTypeName + "_block.comp     -o z_scan_" + scanImpl.second.dataTypeName + "_block.comp.spv" << std::endl;
        buildSH << "glslangValidator -V z_scan_" + scanImpl.second.dataTypeName + "_propagate.comp -o z_scan_" + scanImpl.second.dataTypeName + "_propagate.comp.spv" << std::endl;
      }
    }
  }

  if(a_codeInfo->usedServiceCalls.find("sort") != a_codeInfo->usedServiceCalls.end())
  {
    for(auto sortImpl : a_codeInfo->serviceCalls)
    {
      if (sortImpl.second.opName == "sort")
      {
        nlohmann::json params;
        params["Type"]   = sortImpl.second.dataTypeName;
        params["Lambda"] = sortImpl.second.lambdaSource;
        params["Suffix"] = ToLowerCase(a_codeInfo->mainClassSuffix);

        kslicer::ApplyJsonToTemplate(templatesFolder / "z_bitonic_pass.glsl",  shaderPath / ("z_bitonic_" + sortImpl.second.dataTypeName + "_pass.comp"), params);
        kslicer::ApplyJsonToTemplate(templatesFolder / "z_bitonic_512.glsl",   shaderPath / ("z_bitonic_" + sortImpl.second.dataTypeName + "_512.comp"), params);
        kslicer::ApplyJsonToTemplate(templatesFolder / "z_bitonic_1024.glsl",  shaderPath / ("z_bitonic_" + sortImpl.second.dataTypeName + "_1024.comp"), params);
        kslicer::ApplyJsonToTemplate(templatesFolder / "z_bitonic_2048.glsl",  shaderPath / ("z_bitonic_" + sortImpl.second.dataTypeName + "_2048.comp"), params);

        buildSH << "glslangValidator -V z_bitonic_" + sortImpl.second.dataTypeName + "_pass.comp -o z_bitonic_" + sortImpl.second.dataTypeName + "_pass.comp.spv" << std::endl;
        buildSH << "glslangValidator -V z_bitonic_" + sortImpl.second.dataTypeName + "_512.comp  -o z_bitonic_" + sortImpl.second.dataTypeName + "_512.comp.spv"  << std::endl;
        buildSH << "glslangValidator -V z_bitonic_" + sortImpl.second.dataTypeName + "_1024.comp -o z_bitonic_" + sortImpl.second.dataTypeName + "_1024.comp.spv" << std::endl;
        buildSH << "glslangValidator -V z_bitonic_" + sortImpl.second.dataTypeName + "_2048.comp -o z_bitonic_" + sortImpl.second.dataTypeName + "_2048.comp.spv" << std::endl;
      }
    }
  }

  if(needRTDummies)
  {
    nlohmann::json params;
    kslicer::ApplyJsonToTemplate(templatesFolder / "z_trace_rchit.glsl", shaderPath / "z_trace_rchit.glsl", params);
    kslicer::ApplyJsonToTemplate(templatesFolder / "z_trace_rmiss.glsl", shaderPath / "z_trace_rmiss.glsl", params);
    kslicer::ApplyJsonToTemplate(templatesFolder / "z_trace_smiss.glsl", shaderPath / "z_trace_smiss.glsl", params);

    buildSH << "glslangValidator -V --target-env vulkan1.2 -S rchit z_trace_rchit.glsl -o z_trace_rchit.glsl.spv" << std::endl;
    buildSH << "glslangValidator -V --target-env vulkan1.2 -S rmiss z_trace_rmiss.glsl -o z_trace_rmiss.glsl.spv" << std::endl;
    buildSH << "glslangValidator -V --target-env vulkan1.2 -S rmiss z_trace_smiss.glsl -o z_trace_smiss.glsl.spv" << std::endl;
  }

  buildSH.close();
}

std::string kslicer::GLSLCompiler::LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const
{
  if(a_kernelDim == 1)
    return "gl_LocalInvocationID.x";
  else if(a_kernelDim == 2)
  {
    std::stringstream strOut;
    strOut << "gl_LocalInvocationID.x + " << a_wgSize[0] << "*gl_LocalInvocationID.y";
    return strOut.str();
  }
  else if(a_kernelDim == 3)
  {
    std::stringstream strOut;
    strOut << "gl_LocalInvocationID.x + " << a_wgSize[0] << "*gl_LocalInvocationID.y + " << a_wgSize[0]*a_wgSize[1] << "*gl_LocalInvocationID.z";
    return strOut.str();
  }
  else
  {
    std::cout << "  [GLSLCompiler::LocalIdExpr]: Error, bad kernelDim = " << a_kernelDim << std::endl;
    return "gl_LocalInvocationID.x";
  }
}


void kslicer::GLSLCompiler::GetThreadSizeNames(std::string a_strs[3]) const
{
  a_strs[0] = "iNumElementsX";
  a_strs[1] = "iNumElementsY";
  a_strs[2] = "iNumElementsZ";
}

std::string kslicer::GLSLCompiler::GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const
{
  std::string res = "unknownSubgroup"; 
  switch(a_access.type)
  {
    case KernelInfo::REDUCTION_TYPE::ADD_ONE:
    case KernelInfo::REDUCTION_TYPE::ADD:
    res = "subgroupAdd";
    break;

    case KernelInfo::REDUCTION_TYPE::SUB:
    case KernelInfo::REDUCTION_TYPE::SUB_ONE:
    res = "subgroupAdd";
    break;

    case KernelInfo::REDUCTION_TYPE::FUNC:
    {
      if(a_access.funcName == "min" || a_access.funcName == "std::min") res = "subgroupMin";
      if(a_access.funcName == "max" || a_access.funcName == "std::max") res = "subgroupMax";
    }
    break;

    case KernelInfo::REDUCTION_TYPE::MUL:
    res = "subgroupMul";
    break;

    default:
    break;
  };
  return res;
}

std::string kslicer::GLSLCompiler::GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const
{
  std::string res = "unknownAtomic";
  switch(a_access.type)
  {
    case KernelInfo::REDUCTION_TYPE::ADD_ONE:
    case KernelInfo::REDUCTION_TYPE::ADD:
    res = "atomicAdd";
    break;

    case KernelInfo::REDUCTION_TYPE::SUB:
    case KernelInfo::REDUCTION_TYPE::SUB_ONE:
    res = "atomicSub";
    break;

    case KernelInfo::REDUCTION_TYPE::FUNC:
    {
      if(a_access.funcName == "min" || a_access.funcName == "std::min") res = "atomicMin";
      if(a_access.funcName == "max" || a_access.funcName == "std::max") res = "atomicMax";
    }
    break;

    default:
    break;
  };
  return res;
}


std::string kslicer::GLSLCompiler::ProcessBufferType(const std::string& a_typeName) const
{
  std::string type = kslicer::CleanTypeName(a_typeName);
  ReplaceFirst(type, "*", "");
  if(type[type.size()-1] == ' ')
    type = type.substr(0, type.size()-1);

  return type;
}

std::string kslicer::GLSLCompiler::RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const
{
  return std::string("{ uint offset = atomicAdd(") + UBOAccess(memberNameB) + ", 1); " + memberNameA + "[offset] = " + newElemValue + ";}";
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////  GLSLFunctionRewriter  ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<std::string, std::string> kslicer::ListGLSLVectorReplacements()
{
  std::unordered_map<std::string, std::string> m_vecReplacements;
  m_vecReplacements["double2"] ="dvec2";
  m_vecReplacements["double3"] ="dvec3";
  m_vecReplacements["double4"] ="dvec4";
  m_vecReplacements["float2"] = "vec2";
  m_vecReplacements["float3"] = "vec3";
  m_vecReplacements["float4"] = "vec4";
  m_vecReplacements["int2"]   = "ivec2";
  m_vecReplacements["int3"]   = "ivec3";
  m_vecReplacements["int4"]   = "ivec4";
  m_vecReplacements["uint2"]  = "uvec2";
  m_vecReplacements["uint3"]  = "uvec3";
  m_vecReplacements["uint4"]  = "uvec4";
  m_vecReplacements["float4x4"] = "mat4";
  m_vecReplacements["float3x3"] = "mat3";
  m_vecReplacements["float2x2"] = "mat2";
  m_vecReplacements["_Bool"] = "bool";
  m_vecReplacements["unsigned int"]   = "uint";
  m_vecReplacements["unsigned"]       = "uint";
  m_vecReplacements["unsigned char"]  = "uint8_t";
  m_vecReplacements["unsigned short"] = "uint16_t";
  m_vecReplacements["char"]           = "int8_t";
  m_vecReplacements["short"]          = "int16_t";
  m_vecReplacements["uchar"]          = "uint8_t";
  m_vecReplacements["ushort"]         = "uint16_t";
  m_vecReplacements["int32_t"]        = "int";
  m_vecReplacements["uint32_t"]       = "uint";
  m_vecReplacements["size_t"]         = "uint64_t";
  
  std::unordered_map<std::string, std::string> m_vecReplacementsConst;
  for(auto r : m_vecReplacements)
    m_vecReplacementsConst[std::string("const ") + r.first] = std::string("const ") + m_vecReplacements[r.first];
  
  for(auto rc : m_vecReplacementsConst)
    m_vecReplacements[rc.first] = rc.second;
    
  return m_vecReplacements;
}

std::unordered_set<std::string> kslicer::ListPredefinedMathTypes()
{
  auto types = kslicer::ListGLSLVectorReplacements();
  std::unordered_set<std::string> res;
  for(auto p : types)
  {
    res.insert(p.first);
    res.insert(p.second);
  };

  res.insert("short");
  res.insert("int");
  res.insert("float");
  res.insert("double");

  return res;
}

std::string kslicer::CleanTypeName(const std::string& a_str)
{
  if (a_str.empty())
    return a_str;
  std::string typeName = a_str;
  ReplaceFirst(typeName, "const ",     "");
  ReplaceFirst(typeName, "const",      ""); // for 'const*'
  ReplaceFirst(typeName, "struct ",    "");
  ReplaceFirst(typeName, "class ",     "");
  ReplaceFirst(typeName, "&",          "");
  ReplaceFirst(typeName, "*",          "");
  auto posOfDD = typeName.find("::");
  if(posOfDD != std::string::npos)
    typeName = typeName.substr(posOfDD+2);

  // remove spaces at the end
  while(typeName[typeName.size()-1] == ' ')
    typeName = typeName.substr(0, typeName.size()-1);

  return typeName;
}

std::vector<std::pair<std::string, std::string> > SortByKeysByLen(const std::unordered_map<std::string, std::string>& a_map)
{
  std::vector<std::pair<std::string, std::string> > res;
  res.reserve(a_map.size());
  for(auto p : a_map)
    res.push_back(p);
  std::sort(res.begin(), res.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
  return res;
}

kslicer::GLSLFunctionRewriter::GLSLFunctionRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) : FunctionRewriter(R,a_compiler,a_codeInfo)
{
  m_vecReplacements  = kslicer::ListGLSLVectorReplacements();
  m_vecReplacements2 = SortByKeysByLen(m_vecReplacements);
  m_funReplacements["fmin"]  = "min";
  m_funReplacements["fmax"]  = "max";
  m_funReplacements["fminf"] = "min";
  m_funReplacements["fmaxf"] = "max";
  m_funReplacements["fsqrt"] = "sqrt";
  m_funReplacements["sqrtf"] = "sqrt";
  m_funReplacements["fabs"]  = "abs";
  m_funReplacements["to_float4"] = "vec4";
  m_shit = a_shit;
}

std::string kslicer::GLSLFunctionRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  
  auto old = expr;
  while(clang::isa<clang::ImplicitCastExpr>(expr))
    expr = clang::dyn_cast<clang::ImplicitCastExpr>(expr)->IgnoreImpCasts();

  if(m_pKernelRewriter != nullptr) // we actually do kernel rewrite
  {
    std::string result = m_pKernelRewriter->RecursiveRewriteImpl(expr);
    sFeatures = sFeatures || m_pKernelRewriter->GetShaderFeatures();
    MarkRewritten(expr);
    //MarkRewritten(old);
    return result;
  }
  else
  {
    GLSLFunctionRewriter rvCopy = *this;
    rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
    sFeatures = sFeatures || rvCopy.sFeatures;
    MarkRewritten(expr);
    
    auto range = expr->getSourceRange();
    auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
    if(p != rvCopy.m_workAround.end())
      return p->second;
    else
    {
      std::string res = m_rewriter.getRewrittenText(range);
      return (res != "") ? res : GetRangeSourceCode(range, m_compiler);
    }
  }
  return "";
}

void kslicer::GLSLFunctionRewriter::ApplyDefferedWorkArounds()
{
  // replace all work arounds if they were not processed
  //
  std::map<uint32_t, std::string> sorted;
  {
    for(const auto& pair : m_workAround)
      sorted.insert(std::make_pair(uint32_t(pair.first & uint64_t(0xFFFFFFFF)), pair.second));
  }

  for(const auto& pair : sorted) // TODO: sort nodes by their rucursion depth or source location?
  {
    auto loc = clang::SourceLocation::getFromRawEncoding(pair.first);
    clang::SourceRange range(loc, loc); 
    m_rewriter.ReplaceText(range, pair.second);
  }

  m_workAround.clear();
}

kslicer::ShaderFeatures kslicer::GetUsedShaderFeaturesFromTypeName(const std::string& a_str)
{
  const bool isConst  = (a_str.find("const") != std::string::npos);
  const bool isUshort = (a_str.find("short") != std::string::npos)    || (a_str.find("ushort") != std::string::npos) ||
                        (a_str.find("uint16_t") != std::string::npos) || (a_str.find("int16_t") != std::string::npos);
  const bool isByte   = (a_str.find("char") != std::string::npos)    || (a_str.find("uchar") != std::string::npos) || (a_str.find("unsigned char") != std::string::npos) ||
                        (a_str.find("uint8_t") != std::string::npos) || (a_str.find("int8_t") != std::string::npos);
  const bool isInt64  = (a_str.find("long long int") != std::string::npos) ||
                        (a_str.find("uint64_t") != std::string::npos) || (a_str.find("int64_t") != std::string::npos);

  const bool isFloat64 = (a_str.find("double") != std::string::npos);

  std::string copy = a_str;
  ReplaceFirst(copy, "const ", "");
  while(ReplaceFirst(copy, " ", ""));
  const bool isHalf = (copy == "half") || (copy == "half2") || (copy == "half3") || (copy == "half4");

  kslicer::ShaderFeatures sFeatures;
  sFeatures.useByteType    = isByte;
  sFeatures.useShortType   = isUshort;
  sFeatures.useInt64Type   = isInt64;
  sFeatures.useFloat64Type = isFloat64;
  sFeatures.useHalfType    = isHalf;
  return sFeatures;
}

std::string kslicer::GLSLFunctionRewriter::RewriteStdVectorTypeStr(const std::string& a_str) const
{
  const bool isConst  = (a_str.find("const") != std::string::npos);
  std::string copy = a_str;
  ReplaceFirst(copy, "const ", "");
  while(ReplaceFirst(copy, " ", ""));

  auto sFeatures2 = kslicer::GetUsedShaderFeaturesFromTypeName(a_str);
  sFeatures = sFeatures || sFeatures2;
  
  std::string resStr;
  std::string typeStr = kslicer::CleanTypeName(a_str);
  ReplaceFirst(typeStr, "unsigned long", "uint");
  ReplaceFirst(typeStr, "unsigned char", "uint8_t");

  if(typeStr.size() > 0 && typeStr[typeStr.size()-1] == ' ')
    typeStr = typeStr.substr(0, typeStr.size()-1);

  if(typeStr.size() > 0 && typeStr[0] == ' ')
    typeStr = typeStr.substr(1, typeStr.size()-1);

  auto p = m_vecReplacements.find(typeStr);
  if(p == m_vecReplacements.end())
    resStr = typeStr;
  else
    resStr = p->second;

  if(isConst)
    resStr = std::string("const ") + resStr;

  return resStr;
}

// process arrays: 'float[3] data' --> 'float data[3]' 
std::string kslicer::GLSLFunctionRewriter::RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const
{
  auto typeNameR = a_typeName;
  auto posArrayBegin = typeNameR.find("[");
  auto posArrayEnd   = typeNameR.find("]");
  if(posArrayBegin != std::string::npos && posArrayEnd != std::string::npos)
  {
    varName   = varName + typeNameR.substr(posArrayBegin, posArrayEnd-posArrayBegin+1);
    typeNameR = typeNameR.substr(0, posArrayBegin);
  }

  return RewriteStdVectorTypeStr(typeNameR);
}

std::string kslicer::GLSLFunctionRewriter::RewriteImageType(const std::string& a_containerType, const std::string& a_containerDataType, kslicer::TEX_ACCESS a_accessType, std::string& outImageFormat) const
{
  std::string result = "";
  if(a_accessType == kslicer::TEX_ACCESS::TEX_ACCESS_READ)
    result = "readonly  ";
  else if(a_accessType == kslicer::TEX_ACCESS::TEX_ACCESS_WRITE)
    result = "writeonly ";

  if(a_accessType != kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE)
  {
    const std::string dataTypeRewritten = RewriteStdVectorTypeStr(a_containerDataType);
    if(dataTypeRewritten == "uint" || dataTypeRewritten == "uvec2" || dataTypeRewritten == "uvec4")
      result += "u";
    if(dataTypeRewritten == "int" || dataTypeRewritten == "ivec2" || dataTypeRewritten == "ivec4")
      result += "i";

    if(a_containerType == "Texture1D" || a_containerType == "Image1D")
    {
      result += "image1D";
    }
    else if(a_containerType == "Texture2D" || a_containerType == "Image2D")
    {
      result += "image2D";
    }
    else if(a_containerType == "Texture3D" || a_containerType == "Image3D")
    {
      result += "image3D";
    }

    // image Format qualifiers, post it to govnokod.ru
    //
    if(dataTypeRewritten == "float" || dataTypeRewritten == "int" || dataTypeRewritten == "uint" ||
       dataTypeRewritten == "uint8_t" || dataTypeRewritten == "uint16_t")
    {
      outImageFormat = "r";
    }
    else if(dataTypeRewritten == "vec2" || dataTypeRewritten == "uvec2" || dataTypeRewritten == "ivec2")
    {
      outImageFormat = "rg";
    }

    else if(dataTypeRewritten == "vec4" || dataTypeRewritten == "uvec4" || dataTypeRewritten == "ivec4")
    {
      outImageFormat = "rgba";
    }

    if(dataTypeRewritten == "float" || dataTypeRewritten == "vec2" || dataTypeRewritten == "vec3" || dataTypeRewritten == "vec4")
      outImageFormat += "32f"; // TODO: 16f ???
    else if(dataTypeRewritten == "int" || dataTypeRewritten == "ivec2" || dataTypeRewritten == "ivec3" || dataTypeRewritten == "ivec4")
      outImageFormat += "32i"; // TODO: 16i ???
    else if(dataTypeRewritten == "uint" || dataTypeRewritten == "uvec2" || dataTypeRewritten == "uvec3" || dataTypeRewritten == "uvec4")
      outImageFormat += "32ui"; // TODO: 16ui ???
  }
  else
  {
    if(a_containerType == "Texture1D" || a_containerType == "Image1D")
    {
      result += "sampler1D";
    }
    else if(a_containerType == "Texture2D" || a_containerType == "Image2D")
    {
      result += "sampler2D";
    }
    else if(a_containerType == "Texture3D" || a_containerType == "Image3D")
    {
      result += "sampler3D";
    }
  }

  return result;
}

std::string kslicer::GLSLFunctionRewriter::VectorTypeContructorReplace(const std::string& fname, const std::string& callText)
{
  auto p = m_vecReplacements.find(fname);
  if(p == m_vecReplacements.end())
    return std::string("make_") + fname + callText;
  else
    return p->second + callText;
}

bool kslicer::GLSLFunctionRewriter::NeedsVectorTypeRewrite(const std::string& a_str) // TODO: make this implementation more smart, bad implementation actually!
{
  if(a_str.find("glm::") != std::string::npos)
    return true;
  std::string name2 = std::string("LiteMath::") + a_str;
  bool need = false;
  for(auto p = m_vecReplacements2.begin(); p != m_vecReplacements2.end(); ++p)
  {
    if(name2.find(p->first) != std::string::npos || a_str.find(p->first) != std::string::npos)
    {
      need = true;
      break;
    }
  }
  return need;
}

bool kslicer::GLSLFunctionRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)
{
  if(m_shit.originalName == "")
    return true;

  clang::Expr* left  = arrayExpr->getLHS();
  clang::Expr* right = arrayExpr->getRHS();

  const std::string leftText  = kslicer::GetRangeSourceCode(left->getSourceRange(), m_compiler);
  //const std::string rightText = kslicer::GetRangeSourceCode(right->getSourceRange(), m_compiler);
  for(auto globalPointer : m_shit.pointers)
  {
    if(globalPointer.formal == leftText && WasNotRewrittenYet(right))
    { 
      bool isBufferReferenceAccess = false;
      auto pFound = m_codeInfo->allDataMembers.find(globalPointer.actual);
      if(pFound != m_codeInfo->allDataMembers.end())
        isBufferReferenceAccess = pFound->second.bindWithRef;

      const std::string rightText = RecursiveRewrite(right);
      std::string textRes;
      if(isBufferReferenceAccess)
        textRes = std::string("all_references.") + globalPointer.actual + "." + globalPointer.actual + "[" + rightText + " + " + leftText + "Offset]";
      else
        textRes = globalPointer.actual + "[" + rightText + " + " + leftText + "Offset]";
      ReplaceTextOrWorkAround(arrayExpr->getSourceRange(), textRes); // process shitty global pointers
      MarkRewritten(right);
      break;
    }
  }

  return true;
}

bool kslicer::GLSLFunctionRewriter::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr)
{
  if(!clang::isa<clang::UnaryExprOrTypeTraitExpr>(szOfExpr))
    return true;
  const clang::UnaryExprOrTypeTraitExpr* SizeOf = clang::dyn_cast<clang::UnaryExprOrTypeTraitExpr>(szOfExpr);
  if(SizeOf->getKind() != clang::UETT_SizeOf)
    return true;

  clang::QualType qt = SizeOf->getTypeOfArgument();
  auto typeInfo      = m_compiler.getASTContext().getTypeInfo(qt);
  auto sizeInBytes   = typeInfo.Width / 8;

  if(WasNotRewrittenYet(szOfExpr))
  {
    std::stringstream str;
    str << sizeInBytes;
    ReplaceTextOrWorkAround(szOfExpr->getSourceRange(), str.str());
    MarkRewritten(szOfExpr);
  }

  return true;
}

bool kslicer::GLSLFunctionRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f) 
{ 
  return true; 
}

bool kslicer::GLSLFunctionRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)
{
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
  std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler); 
  
  if(op == "[" || op == "]" || op == "[]")
  {
    const clang::Expr* nodes[3] = {nullptr,nullptr,nullptr};
    Get2DIndicesOfFloat4x4(expr,nodes);

    if(nodes[0] != nullptr)
    {
      std::string varName = RecursiveRewrite(nodes[0]);
      std::string yText   = RecursiveRewrite(nodes[1]);
      std::string xText   = RecursiveRewrite(nodes[2]);
      std::string resVal  = varName + "[" + yText + "][" + xText + "]";

      ReplaceTextOrWorkAround(expr->getSourceRange(), resVal);
      MarkRewritten(expr);
      return true;
    }
  }

  return FunctionRewriter::VisitCXXOperatorCallExpr_Impl(expr);
}

std::string kslicer::GLSLFunctionRewriter::RewriteFuncDecl(clang::FunctionDecl* fDecl)
{
  std::string retT   = RewriteStdVectorTypeStr(fDecl->getReturnType().getAsString());
  std::string fname  = fDecl->getNameInfo().getName().getAsString();
  
  if(fname == "NextState")
  {
    int a = 2;
  }

  if(m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->hasPrefix) // alter function name if it has any prefix
    if(fname.find(m_pCurrFuncInfo->prefixName) == std::string::npos)
      fname = m_pCurrFuncInfo->prefixName + "_" + fname;

  std::string result = retT + " " + fname + "(";

  const bool shitHappends = (fname == m_shit.originalName);
  if(shitHappends)
    result = retT + " " + m_shit.ShittyName() + "(";

  for(uint32_t i=0; i < fDecl->getNumParams(); i++)
  {
    const clang::ParmVarDecl* pParam  = fDecl->getParamDecl(i);
    const clang::QualType typeOfParam =	pParam->getType();
    std::string typeStr = typeOfParam.getAsString();
    if(typeOfParam->isPointerType())
    {
      bool pointerToGlobalMemory = false;
      if(shitHappends)
      {
        for(auto p : m_shit.pointers)
        {
          if(p.formal == pParam->getNameAsString() )
          {
            pointerToGlobalMemory = true;
            break;
          }
        }
      }

      const auto originalText = kslicer::GetRangeSourceCode(pParam->getSourceRange(), m_compiler);

      if(pointerToGlobalMemory)
        result += std::string("uint ") + pParam->getNameAsString() + "Offset";
      else if(originalText.find("[") != std::string::npos && originalText.find("]") != std::string::npos) // fixed size arrays
      {
        if(typeOfParam->getPointeeType().isConstQualified())
        {
          ReplaceFirst(typeStr, "const ", "");
          result += originalText;
        }
        else
          result += std::string("inout ") + originalText;
      }
      else
      {
        std::string paramName = pParam->getNameAsString();
        ReplaceFirst(typeStr, "*", "");
        if(typeOfParam->getPointeeType().isConstQualified())
        {
          ReplaceFirst(typeStr, "const ", "");
          result += std::string("in ") + RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();
        }
        else
          result += std::string("inout ") + RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();
      }
    }
    else if(typeOfParam->isReferenceType())
    {
      if(typeOfParam->getPointeeType().isConstQualified())
      {
        if(typeStr.find("Texture") != std::string::npos || typeStr.find("Image") != std::string::npos)
        {
          auto dataType = typeOfParam.getNonReferenceType();
          auto typeDecl = dataType->getAsRecordDecl();
          if(typeDecl != nullptr && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl))
          {
            std::string containerType, containerDataType;
            auto specDecl = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(typeDecl);
            kslicer::SplitContainerTypes(specDecl, containerType, containerDataType);
            ReplaceFirst(containerType, "Texture", "sampler");
            ReplaceFirst(containerType, "Image",   "sampler");
            result += std::string("in ") + containerType + " " + pParam->getNameAsString();
          }
          else
            result += std::string("in ") + dataType.getAsString() + " " + pParam->getNameAsString();
        }
        else
        {
          ReplaceFirst(typeStr, "const ", "");
          result += std::string("in ") + RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();
        }
      }
      else
      {
        std::string typeStr2 = typeOfParam->getPointeeType().getAsString();
        if(m_codeInfo->megakernelRTV && (typeStr.find("Texture") != std::string::npos || typeStr.find("Image") != std::string::npos))
        {
          result += std::string("uint a_dummyOf") + pParam->getNameAsString();
        }
        else
          result += std::string("inout ") + RewriteStdVectorTypeStr(typeStr2) + " " + pParam->getNameAsString();
      }
    }
    else
      result += RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();

    if(i!=fDecl->getNumParams()-1)
      result += ", ";
  }

  return result + ") ";
}

bool kslicer::GLSLFunctionRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)
{
  if(clang::isa<clang::CXXMethodDecl>(fDecl)) // ignore methods here, for a while ...
    return true;

  if(WasNotRewrittenYet(fDecl->getBody()))
  {
    RewrittenFunction done = RewriteFunction(fDecl);
    const auto hash = GetHashOfSourceRange(fDecl->getBody()->getSourceRange());
    if(m_codeInfo->m_functionsDone.find(hash) == m_codeInfo->m_functionsDone.end())
      m_codeInfo->m_functionsDone[hash] = done;
    MarkRewritten(fDecl->getBody());
  }

  return true;
}

bool kslicer::GLSLFunctionRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)
{
  if(expr->isArrow() && WasNotRewrittenYet(expr->getBase()) )
  {
    const auto exprText     = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    const std::string lText = exprText.substr(exprText.find("->")+2);
    const std::string rText = RecursiveRewrite(expr->getBase());
    ReplaceTextOrWorkAround(expr->getSourceRange(), rText + "." + lText);
    MarkRewritten(expr->getBase());
  }

  return true;
}

bool kslicer::GLSLFunctionRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{
  const auto op = expr->getOpcodeStr(expr->getOpcode());
  if((op == "*" || op == "&") && WasNotRewrittenYet(expr->getSubExpr()) )
  {
    std::string text = RecursiveRewrite(expr->getSubExpr());
    ReplaceTextOrWorkAround(expr->getSourceRange(), text);
    MarkRewritten(expr->getSubExpr());
  }

  return true;
}

std::string kslicer::GLSLFunctionRewriter::CompleteFunctionCallRewrite(clang::CallExpr* call)
{
  std::string rewrittenRes = "";
  for(unsigned i=0;i<call->getNumArgs(); i++)
  {
    rewrittenRes += RecursiveRewrite(call->getArg(i));
    if(i!=call->getNumArgs()-1)
      rewrittenRes += ", ";
  }
  rewrittenRes += ")";
  return rewrittenRes;
}

bool kslicer::GLSLFunctionRewriter::VisitCallExpr_Impl(clang::CallExpr* call)
{
  if(clang::isa<clang::CXXMemberCallExpr>(call) || clang::isa<clang::CXXConstructExpr>(call)) // process CXXMemberCallExpr else-where
    return true;

  clang::FunctionDecl* fDecl = call->getDirectCallee();
  if(fDecl == nullptr)
    return true;
  
  const std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler);
  const std::string fname = fDecl->getNameInfo().getName().getAsString();
  ///////////////////////////////////////////////////////////////////////
  std::string makeSmth = "";
  if(fname == "make_float3x3_by_columns") // mat3(a,b,c) == make_float3x3_by_columns(a,b,c)
    makeSmth = "float3x3";
  else if(fname == "make_float3x3")       // don't change it!
    ;
  else if(fname.substr(0, 5) == "make_")
    makeSmth = fname.substr(5);
  auto pVecMaker = m_vecReplacements.find(makeSmth);
  ///////////////////////////////////////////////////////////////////////

  if(fname == "atomicAdd" && call->getNumArgs() >= 2)
  {
    const auto arg1        = call->getArg(1); 
    clang::QualType aType1 = arg1->getType();
    std::string aTypeName  = aType1.getAsString();

    if(aTypeName == "float" || aTypeName == "double")
      sFeatures.useFloatAtomicAdd = true;
  }

  auto pFoundSmth = m_funReplacements.find(fname);
  if(fname == "to_float3" && call->getNumArgs() == 1 && WasNotRewrittenYet(call) )
  {
    const auto qt = call->getArg(0)->getType();
    const std::string typeName = qt.getAsString();
    if(typeName.find("float4") != std::string::npos)
    {
      const std::string exprText = RecursiveRewrite(call->getArg(0));
      std::string textRes;
      if(clang::isa<clang::CXXConstructExpr>(call->getArg(0))) // TODO: add other similar node types process here
        textRes = exprText + ".xyz"; // to_float3(f4Data) ==> f4Data.xyz
      else
        textRes = std::string("(") + exprText + ").xyz"; // to_float3(a+b) ==> (a+b).xyz

      ReplaceTextOrWorkAround(call->getSourceRange(), textRes);
      MarkRewritten(call);
    }
  }
  else if(makeSmth != "" && pVecMaker != m_vecReplacements.end() && call->getNumArgs() !=0 && WasNotRewrittenYet(call) )
  {
    const std::string rewrittenRes = pVecMaker->second + "(" + CompleteFunctionCallRewrite(call);
    
    ReplaceTextOrWorkAround(call->getSourceRange(), rewrittenRes);
    MarkRewritten(call);
  }
  else if(fname == "mul4x4x4" && call->getNumArgs() == 2 && WasNotRewrittenYet(call))
  {
    const std::string A = RecursiveRewrite(call->getArg(0));
    const std::string B = RecursiveRewrite(call->getArg(1));
    ReplaceTextOrWorkAround(call->getSourceRange(), "(" + A + "*" + B + ")");
    MarkRewritten(call);
  }
  else if(fname == "lerp" && call->getNumArgs() == 3 && WasNotRewrittenYet(call))
  {
    const std::string A = RecursiveRewrite(call->getArg(0));
    const std::string B = RecursiveRewrite(call->getArg(1));
    const std::string C = RecursiveRewrite(call->getArg(2));
    ReplaceTextOrWorkAround(call->getSourceRange(), "mix(" + A + ", " + B + ", " + C + ")");
    MarkRewritten(call);
  }
  else if(fname == "atan2" && call->getNumArgs() == 2 && WasNotRewrittenYet(call))
  {
    const std::string arg1 = RecursiveRewrite(call->getArg(0));
    const std::string arg2 = RecursiveRewrite(call->getArg(1));
    ReplaceTextOrWorkAround(call->getSourceRange(), "atan(" + arg1 + "," + arg2 + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_int32" || fname == "as_int") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceTextOrWorkAround(call->getSourceRange(), "floatBitsToInt(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_uint32" || fname == "as_uint") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceTextOrWorkAround(call->getSourceRange(), "floatBitsToUint(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_float" || fname == "as_float32")  && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text  = RecursiveRewrite(call->getArg(0));
    const auto qtOfArg      = call->getArg(0)->getType();
    const std::string tname = kslicer::CleanTypeName(qtOfArg.getAsString());
    
    std::string lastRewrittenText;
    if(tname == "uint" || tname == "const uint" || tname == "uint32_t" || tname == "const uint32_t" || tname == "unsigned" || tname == "const unsigned")
      lastRewrittenText = "uintBitsToFloat(" + text + ")";
    else
      lastRewrittenText = "intBitsToFloat(" + text + ")";
    ReplaceTextOrWorkAround(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if((fname == "inverse4x4" || fname == "inverse3x3" || fname == "inverse2x2") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceTextOrWorkAround(call->getSourceRange(), "inverse(" + text + ")");
    MarkRewritten(call);
  }
  else if(pFoundSmth != m_funReplacements.end() && WasNotRewrittenYet(call))
  {
    std::string lastRewrittenText = pFoundSmth->second + "(" + CompleteFunctionCallRewrite(call);
    ReplaceTextOrWorkAround(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if(fDecl->isInStdNamespace() && WasNotRewrittenYet(call)) // remove "std::"
  {
    std::string lastRewrittenText = fname + "(" + CompleteFunctionCallRewrite(call);
    ReplaceTextOrWorkAround(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }

  return true;
}

bool kslicer::GLSLFunctionRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl) // special case for process multiple decls in line, like 'int i,j,k=2'
{
  if(!decl->isSingleDecl())
  {
    //const std::string debugText = kslicer::GetRangeSourceCode(decl->getSourceRange(), m_compiler);
    std::string varType = "";
    std::string resExpr = "";
    for(auto it = decl->decl_begin(); it != decl->decl_end(); ++it)
    {
      clang::Decl* c_decl = (*it);
      if(!clang::isa<clang::VarDecl>(c_decl))
        continue;

      clang::VarDecl* vdecl = clang::dyn_cast<clang::VarDecl>(c_decl);
      const auto qt         = vdecl->getType();
      const auto pValue     = vdecl->getAnyInitializer();
      const std::string varName  = vdecl->getNameAsString();
      const std::string varValue = RecursiveRewrite(pValue);

      if(varType == "") // first element
      {
        varType = qt.getAsString();
        if(!NeedsVectorTypeRewrite(varType)) // immediately ignore DeclStmt like 'int i,j,k=2' if we dont need to rewrite the type
          return true;
        varType = RewriteStdVectorTypeStr(varType);

        if(varValue == "" || varValue == varName)
          resExpr = varType + " " + varName;
        else
          resExpr = varType + " " + varName + " = " + varValue;
      }
      else              // second or other
      {
        if(varValue == "" || varValue == varName)
          resExpr += (" " + varName);
        else
          resExpr += (varName + " = " + varValue);
      }

      auto next = it; ++next;
      if(next != decl->decl_end())
        resExpr += ", ";
      else
        resExpr += ";";

      MarkRewritten(pValue);
    }

    if(WasNotRewrittenYet(decl))
    {
      ReplaceTextOrWorkAround(decl->getSourceRange(), resExpr);
      MarkRewritten(decl);
    }
  }

  return true;
}

const clang::DeclStmt* getParentDeclContext(const clang::VarDecl* varDecl, clang::ASTContext& context) 
{
    clang::ParentMapContext& parentMapContext = context.getParentMapContext();

    // Получаем родительский узел
    const clang::DynTypedNodeList parents = parentMapContext.getParents(*varDecl);
    
    // Перебираем всех родителей
    for (const clang::DynTypedNode& parent : parents) {
      auto stmt = parent.get<clang::DeclStmt>();
      if(stmt != nullptr)
        return stmt;
      //parent.dump(llvm::outs(), context);
    }
    return nullptr;
}

bool kslicer::GLSLFunctionRewriter::VisitVarDecl_Impl(clang::VarDecl* decl)
{
  if(clang::isa<clang::ParmVarDecl>(decl)) // process else-where (VisitFunctionDecl_Impl)
    return true;

  const auto qt      = decl->getType();
  const auto pValue  = decl->getAnyInitializer();

  const std::string debugText    = kslicer::GetRangeSourceCode(decl->getSourceRange(), m_compiler);
  //const std::string debugTextVal = kslicer::GetRangeSourceCode(pValue->getSourceRange(), m_compiler);
  const std::string varType = qt.getAsString();
  auto sFeatures2 = kslicer::GetUsedShaderFeaturesFromTypeName(varType);
  m_codeInfo->globalShaderFeatures = m_codeInfo->globalShaderFeatures || sFeatures2;

  for (const clang::Attr* attr : decl->attrs()) 
  {
    //const std::string attrType = attr->getSpelling();
    const std::string attrText = kslicer::GetRangeSourceCode(attr->getRange(), m_compiler);

    clang::QualType varType = decl->getType();

    if (varType->isArrayType() && attrText == "threadlocal") 
    {
      std::cout << "  found: " << attrText.c_str() << " for " << debugText.c_str() << std::endl;
      std::string varName = decl->getNameAsString();
      auto p = m_codeInfo->m_threadLocalArrays.find(varName.c_str());
      if(p == m_codeInfo->m_threadLocalArrays.end())
      {
        const clang::ArrayType* arrayType = varType->getAsArrayTypeUnsafe();
        clang::QualType elementType = arrayType->getElementType();
        
        int arraySize = 0;
        if (const clang::ConstantArrayType* constantArrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType)) {
          clang::SmallVector<char, 16> tmpBuf;
          constantArrayType->getSize().toStringUnsigned(tmpBuf, 10);
          arraySize = std::atoi(tmpBuf.data());
        }

        const clang::DeclStmt* declStmt = getParentDeclContext(decl, m_compiler.getASTContext());
        if(declStmt != nullptr && WasNotRewrittenYet(declStmt))
        {
          std::stringstream strOut;
          strOut << "// " << debugText.c_str() << "; was moved to global scope in GLSL"; // 123
          ReplaceTextOrWorkAround(declStmt->getSourceRange(), strOut.str());
          MarkRewritten(declStmt);
          
          kslicer::ArrayData array;
          array.arrayName = varName;
          array.elemType  = elementType.getAsString();
          array.arraySize = arraySize;

          if(m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->isKernel) 
          {
            auto pKernel = m_codeInfo->kernels.find(m_pCurrFuncInfo->name);
            if(pKernel != m_codeInfo->kernels.end())
              pKernel->second.threadLocalArrays[array.arrayName] = array;
            else
              m_codeInfo->m_threadLocalArrays[array.arrayName] = array;
          }
          else
            m_codeInfo->m_threadLocalArrays[array.arrayName] = array;
        }
      }
    }
  }

  const clang::Type::TypeClass typeClass = qt->getTypeClass();
  const bool isAuto = (typeClass == clang::Type::Auto);
  if(pValue != nullptr && WasNotRewrittenYet(pValue) && (NeedsVectorTypeRewrite(varType) || isAuto))
  {
    std::string varName  = decl->getNameAsString();
    std::string varNameOld = varName;
    std::string varValue = RecursiveRewrite(pValue);
    std::string varType2 = RewriteStdVectorTypeStr(varType, varName);
    
    std::string lastRewrittenText;
    if(varValue == "" || varNameOld == varValue) // 'float3 deviation;' for some reason !decl->hasInit() does not works
      lastRewrittenText = varType2 + " " + varName;
    else
      lastRewrittenText = varType2 + " " + varName + " = " + varValue;

    ReplaceTextOrWorkAround(decl->getSourceRange(), lastRewrittenText);
    MarkRewritten(pValue);
  }
  return true;
}

bool kslicer::GLSLFunctionRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)
{
  clang::QualType qt   = cast->getTypeAsWritten();
  clang::Expr* 	  next = cast->getSubExpr();
  std::string typeCast = qt.getAsString();
  //const std::string debugText = kslicer::GetRangeSourceCode(cast->getSourceRange(), m_compiler);
  if(WasNotRewrittenYet(next))
  {
    typeCast = RewriteStdVectorTypeStr(typeCast);
    const std::string exprText = RecursiveRewrite(next);
    if(exprText == ")") // strange bug for casts inside 'DeclStmt'
      return true;

    ReplaceTextOrWorkAround(cast->getSourceRange(), typeCast + "(" + exprText + ")");
    MarkRewritten(next);
  }

  return true;
}

bool kslicer::GLSLFunctionRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast)
{
  if(cast->isPartOfExplicitCast())
    return true;
  auto kind = cast->getCastKind();
  std::string debugTxt = kslicer::GetRangeSourceCode(cast->getSourceRange(), m_compiler);

  clang::Expr* preNext = cast->getSubExpr();

  if(clang::isa<clang::CXXConstructExpr>(preNext))
  {
    auto call = clang::dyn_cast<clang::CXXConstructExpr>(preNext);
    clang::CXXConstructorDecl* ctorDecl = call->getConstructor();
    const std::string fname = ctorDecl->getNameInfo().getName().getAsString();
    
    if(kslicer::IsVectorContructorNeedsReplacement(fname) && WasNotRewrittenYet(call) && !ctorDecl->isCopyOrMoveConstructor() && call->getNumArgs() > 0 ) //
    {
      const std::string textRes = RewriteConstructCall(call);
      //ReplaceTextOrWorkAround(call->getSourceRange(), textRes); //
      m_rewriter.ReplaceText(call->getSourceRange(), textRes);    //
      MarkRewritten(call);
    }

    return true;
  }
  else if(clang::isa<clang::ImplicitCastExpr>(preNext))
  {
    clang::Expr* next = clang::dyn_cast<clang::ImplicitCastExpr>(preNext)->getSubExpr();
  
    //https://code.woboq.org/llvm/clang/include/clang/AST/OperationKinds.def.html
    if(kind != clang::CK_IntegralCast && kind != clang::CK_IntegralToFloating && kind != clang::CK_FloatingToIntegral) // in GLSL we don't have implicit casts
      return true;
  
    clang::QualType qt = cast->getType(); qt.removeLocalFastQualifiers();
    std::string castTo = RewriteStdVectorTypeStr(qt.getAsString());
  
    if(WasNotRewrittenYet(next) && qt.getAsString() != "size_t" && qt.getAsString() != "std::size_t")
    {
      const std::string exprText = RecursiveRewrite(next);
      //ReplaceTextOrWorkAround(next->getSourceRange(), castTo + "(" + exprText + ")");
      m_rewriter.ReplaceText(next->getSourceRange(), castTo + "(" + exprText + ")");
      MarkRewritten(next);
    }
  }

  return true;
}

void kslicer::GLSLFunctionRewriter::Get2DIndicesOfFloat4x4(const clang::CXXOperatorCallExpr* expr, const clang::Expr* out[3])
{
  out[0] = nullptr;
  out[1] = nullptr;
  out[2] = nullptr;

  const auto *arg0 = expr->getArg(0)->IgnoreImpCasts();
  const auto *arg1 = expr->getArg(1)->IgnoreImpCasts();
  
  const clang::QualType arg0T    = arg0->getType();
  const std::string arg0TypeName = arg0T.getAsString();

  if(clang::isa<clang::MaterializeTemporaryExpr>(arg0) && arg0TypeName == "RowTmp")
  {
    auto arg0_op = clang::dyn_cast<const clang::MaterializeTemporaryExpr>(arg0);
    auto nextOp  = arg0_op->getSubExpr();
    
    if(clang::isa<clang::CXXOperatorCallExpr>(nextOp)) 
    { 
      const auto nextOp2 = clang::dyn_cast<const clang::CXXOperatorCallExpr>(nextOp);
      const auto arg00   = nextOp2->getArg(0)->IgnoreImpCasts();
      const auto arg01   = nextOp2->getArg(1)->IgnoreImpCasts();
      
      out[0] = arg00;
      out[1] = arg1;
      out[2] = arg01;
    }
  }
}

void kslicer::GLSLCompiler::ProcessVectorTypesString(std::string& a_str)
{
  static auto vecReplacements = SortByKeysByLen(ListGLSLVectorReplacements());
  for(auto p : vecReplacements)
  {
    std::string strToSearch = p.first + " ";
    while(a_str.find(strToSearch) != std::string::npos) // replace all of them
      ReplaceFirst(a_str, p.first, p.second);
  }
}

std::string kslicer::GLSLCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter)
{
  std::string typeInCL = a_decl.type;
  std::string result = "";
  std::string nameWithoutStruct = typeInCL;
  ReplaceFirst(nameWithoutStruct, "struct ", "");
  nameWithoutStruct = a_pRewriter->RewriteStdVectorTypeStr(nameWithoutStruct);
  switch(a_decl.kind)
  {
    case kslicer::DECL_IN_CLASS::DECL_STRUCT:
    result = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    ProcessVectorTypesString(result);
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    {
      std::string originalText = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler);
      if(originalText == "")
        originalText = a_decl.lostValue;
      if(typeInCL.find("const ") == std::string::npos)
        typeInCL = "const " + typeInCL;
      ReplaceFirst(typeInCL,"LiteMath::", "");
      if(a_decl.isArray)
      {
        std::stringstream sizeStr;
        sizeStr << "[" << a_decl.arraySize << "]";
        result = typeInCL + " " + a_decl.name + sizeStr.str() + " = " + originalText + ";";
      }
      else
        result = typeInCL + " " + a_decl.name + " = " + originalText + ";";
    }
    ProcessVectorTypesString(result);
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    {
      if(a_decl.astNode != nullptr)
      {
        const clang::TypedefNameDecl* typedefDecl = llvm::dyn_cast<clang::TypedefNameDecl>(a_decl.astNode); // normal typedef
        if(typedefDecl != nullptr)
        {
          clang::QualType underlyingType = typedefDecl->getUnderlyingType();
          std::string originalTypeName = underlyingType.getAsString();
          result = "#define " + a_decl.name + " " + a_pRewriter->RewriteStdVectorTypeStr(originalTypeName); // a_classInfo.pShaderFuncRewriter->
        }
        else
          result = "#define " + a_decl.name + " " + nameWithoutStruct;                                      
      }
      else
        result = "#define " + a_decl.name + " " + nameWithoutStruct; // typedef struct
    }
    break;
    default:
    break;
  };
  return result;
}

std::shared_ptr<kslicer::FunctionRewriter> kslicer::GLSLCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit)
{
  return std::make_shared<GLSLFunctionRewriter>(R, a_compiler, a_codeInfo, a_shit);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class GLSLKernelRewriter : public kslicer::KernelRewriter, kslicer::IRecursiveRewriteOverride
{
public:

  GLSLKernelRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, const std::string& a_fakeOffsetExpr) :
                     kslicer::KernelRewriter(R, a_compiler, a_codeInfo, a_kernel, a_fakeOffsetExpr), m_glslRW(R, a_compiler, a_codeInfo, a_kernel.currentShit) //
  {
    m_glslRW.m_pKernelRewriter = this;
    m_glslRW.m_pRewrittenNodes = this->m_pRewrittenNodes;
    for(auto arg : a_kernel.args)
    {
      if(arg.isLoopSize || arg.IsUser())
        m_userArgs.insert(arg.name);
    }
  }

  bool VisitCallExpr_Impl(clang::CallExpr* f) override;
  bool VisitVarDecl_Impl(clang::VarDecl* decl) override;
  bool VisitDeclStmt_Impl(clang::DeclStmt* stmt) override;
  bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) override;
  bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;
  bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f) override;
  bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) override;
  bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr) override;
  bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
  bool VisitMemberExpr_Impl(clang::MemberExpr* expr) override;
  bool VisitReturnStmt_Impl(clang::ReturnStmt* ret) override;
  bool VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) override;
  bool VisitBinaryOperator_Impl(clang::BinaryOperator* expr) override;
  bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast) override;
  bool VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) override;
  bool VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) override;

  std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override { return m_glslRW.VectorTypeContructorReplace(fname, callText); }

  std::string RecursiveRewrite(const clang::Stmt* expr) override;
  void ApplyDefferedWorkArounds() override;

  void ClearUserArgs() override { m_userArgs.clear(); }

  kslicer::ShaderFeatures GetShaderFeatures()       const override { return m_glslRW.GetShaderFeatures(); }
  kslicer::ShaderFeatures GetKernelShaderFeatures() const override { return m_glslRW.GetShaderFeatures(); }

protected:

  kslicer::GLSLFunctionRewriter m_glslRW;
  std::string RecursiveRewriteImpl(const clang::Stmt* expr) override { return RecursiveRewrite(expr); }
  std::unordered_set<uint64_t> GetVisitedNodes() const override { return *m_pRewrittenNodes; }

  bool IsGLSL() const override { return true; }

  void RewriteTextureAccess(clang::CXXOperatorCallExpr* expr, clang::Expr* a_assignOp, const std::string& rhsText);
  std::unordered_set<std::string> m_userArgs;
};

std::string GLSLKernelRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  
  auto old = expr;
  while(clang::isa<clang::ImplicitCastExpr>(expr))
    expr = clang::dyn_cast<clang::ImplicitCastExpr>(expr)->IgnoreImpCasts();

  //expr->dump();
  if(clang::isa<clang::DeclRefExpr>(expr)) // bugfix for recurive rewrite of single node, function args access
  {
    std::string text = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); //
    const auto pRef  = clang::dyn_cast<clang::DeclRefExpr>(expr);

    const clang::ValueDecl* pDecl = pRef->getDecl();
    if(!clang::isa<clang::ParmVarDecl>(pDecl))
      return text;
    clang::QualType qt = pDecl->getType();
    if(qt->isPointerType() || qt->isReferenceType()) // we can't put references to push constants
      return text;
    if(m_userArgs.find(text) == m_userArgs.end())
      return text;
    return std::string("kgenArgs.") + text;
  }

  //// check CXXConstructExpr->ImplicitCastExpr->MemberExpr, CXXConstructExpr->MemberExpr and NeedToRewriteMemberExpr(MemberExpr)
  //
  if(clang::isa<clang::CXXConstructExpr>(expr)) // bugfix for recurive rewrite of single node, MemberExpr access in kernel
  {
    const clang::CXXConstructExpr* pConstruct = clang::dyn_cast<clang::CXXConstructExpr>(expr);
    const clang::CXXConstructorDecl* ctorDecl = pConstruct->getConstructor();
    //const std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler);
    //const std::string fname = ctorDecl->getNameInfo().getName().getAsString();
    if(ctorDecl->isCopyOrMoveConstructor()) // || call->getNumArgs() == 0
    {
      const clang::Expr* pExprInsideConstructor =	pConstruct->getArg(0);
      if(clang::isa<clang::ImplicitCastExpr>(pExprInsideConstructor))                          // CXXConstructExpr->ImplicitCastExpr->MemberExpr
        expr = clang::dyn_cast<clang::ImplicitCastExpr>(pExprInsideConstructor)->getSubExpr(); // CXXConstructExpr->MemberExpr
      else if(clang::isa<clang::MemberExpr>(pExprInsideConstructor))
        expr = pExprInsideConstructor;
    }
  }

  if(clang::isa<clang::MemberExpr>(expr)) // same bugfix for recurive rewrite of single node, MemberExpr access in kernel
  {
    const clang::MemberExpr* pMemberExpr = clang::dyn_cast<const clang::MemberExpr>(expr);
    std::string rewrittenText;
    if(NeedToRewriteMemberExpr(pMemberExpr, rewrittenText)) {
      //MarkRewritten(expr);
      //MarkRewritten(old);
      return rewrittenText;
    }
  }

  GLSLKernelRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  MarkRewritten(expr);

  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(kslicer::GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
  {
    //rvCopy.ApplyDefferedWorkArounds();
    return m_rewriter.getRewrittenText(range);
  }
}

void GLSLKernelRewriter::ApplyDefferedWorkArounds()
{
  // replace all work arounds if they were not processed
  //
  std::map<uint32_t, std::string> sorted;
  {
    for(const auto& pair : m_workAround)
      sorted.insert(std::make_pair(uint32_t(pair.first & uint64_t(0xFFFFFFFF)), pair.second));
    
    for(const auto& pair : m_glslRW.WorkAroundRef())
      sorted.insert(std::make_pair(uint32_t(pair.first & uint64_t(0xFFFFFFFF)), pair.second));
  }

  for(const auto& pair : sorted) // TODO: sort nodes by their rucursion depth or source location?
  {
    auto loc = clang::SourceLocation::getFromRawEncoding(pair.first);
    clang::SourceRange range(loc, loc); 
    m_rewriter.ReplaceText(range, pair.second);
  }

  m_workAround.clear();
  m_glslRW.WorkAroundRef().clear();
}


bool GLSLKernelRewriter::VisitCallExpr_Impl(clang::CallExpr* call)
{
  // (#1) check if buffer/pointer to global memory is passed to a function
  //
  std::vector<kslicer::ArgMatch> usedArgMatches = kslicer::MatchCallArgsForKernel(call, m_currKernel, m_compiler);
  std::vector<kslicer::ArgMatch> shittyPointers; shittyPointers.reserve(usedArgMatches.size());
  for(const auto& x : usedArgMatches) {
    const bool exclude = NameNeedsFakeOffset(x.actual); // #NOTE! seems that formal/actual parameters have to be swaped for the whole code
    if(x.isPointer && !exclude)
      shittyPointers.push_back(x);
  }

  // (#2) check if at leat one argument of a function call require function call rewrite due to fake offset
  //
  bool rewriteDueToFakeOffset = false;
  {
    rewriteDueToFakeOffset = false;
    for(unsigned i=0;i<call->getNumArgs(); i++)
    {
      const std::string argName = kslicer::GetRangeSourceCode(call->getArg(i)->getSourceRange(), m_compiler);
      if(NameNeedsFakeOffset(argName))
      {
        rewriteDueToFakeOffset = true;
        break;
      }
    }
  }
  rewriteDueToFakeOffset = rewriteDueToFakeOffset && !processFuncMember;      // function members don't apply fake offsets because they are not kernels
  rewriteDueToFakeOffset = rewriteDueToFakeOffset && (m_fakeOffsetExp != ""); // if fakeOffset is not set for some reason, don't use it.

  const clang::FunctionDecl* fDecl = call->getDirectCallee();
  if(shittyPointers.size() > 0 && fDecl != nullptr)
  {
    std::string fname = fDecl->getNameInfo().getName().getAsString();

    kslicer::ShittyFunction func;
    func.pointers     = shittyPointers;
    func.originalName = fname;
    m_currKernel.shittyFunctions.push_back(func);

    std::string rewrittenRes = func.ShittyName() + "(";
    for(unsigned i=0;i<call->getNumArgs(); i++)
    {
      size_t found = size_t(-1);
      for(size_t j=0;j<shittyPointers.size();j++)
      {
        if(shittyPointers[j].argId == i)
        {
          found = j;
          break;
        }
      }

      if(found != size_t(-1))
      {
        std::string offset = "0";
        const auto arg = kslicer::RemoveImplicitCast(call->getArg(i));
        //const std::string debugText = kslicer::GetRangeSourceCode(arg->getSourceRange(), m_compiler);
        //arg->dump();
        if(clang::isa<clang::BinaryOperator>(arg))
        {
          const auto bo = clang::dyn_cast<clang::BinaryOperator>(arg);
          const clang::Expr *lhs = bo->getLHS();
          const clang::Expr *rhs = bo->getRHS();
          if(bo->getOpcodeStr() == "+")
            offset = RecursiveRewrite(rhs);
        }
        rewrittenRes += offset;
      }
      else
        rewrittenRes += RecursiveRewrite(call->getArg(i));

      if(i!=call->getNumArgs()-1)
        rewrittenRes += ", ";
    }
    rewrittenRes += ")";

    ReplaceTextOrWorkAround(call->getSourceRange(), rewrittenRes);
    MarkRewritten(call);
  }
  else if (m_codeInfo->IsRTV() && rewriteDueToFakeOffset)
  {
    std::string fname        = fDecl->getNameInfo().getName().getAsString();
    std::string rewrittenRes = fname + "(";
    for(unsigned i=0;i<call->getNumArgs(); i++)
    {
      const std::string argName = kslicer::GetRangeSourceCode(call->getArg(i)->getSourceRange(), m_compiler);
      if(NameNeedsFakeOffset(argName) && !m_codeInfo->megakernelRTV)
        rewrittenRes += RecursiveRewrite(call->getArg(i)) + "[" + m_fakeOffsetExp + "]";
      else
        rewrittenRes += RecursiveRewrite(call->getArg(i));

      if(i!=call->getNumArgs()-1)
        rewrittenRes += ", ";
    }
    rewrittenRes += ")";
    ReplaceTextOrWorkAround(call->getSourceRange(), rewrittenRes);
    MarkRewritten(call);
  }
  else
    m_glslRW.VisitCallExpr_Impl(call);

  return true;
}

bool GLSLKernelRewriter::VisitVarDecl_Impl(clang::VarDecl* decl)
{
  kslicer::FuncData fdata;
  if(m_pCurrKernelInfo != nullptr) 
  {
    fdata.astNode  = m_pCurrKernelInfo->astNode;
    fdata.name     = m_pCurrKernelInfo->name;
    fdata.srcRange = fdata.astNode->getSourceRange();
    fdata.srcHash  = kslicer::GetHashOfSourceRange(fdata.srcRange);
    fdata.isMember = false;
    fdata.isKernel = true;
    fdata.depthUse = 0;    
  };
  m_glslRW.SetCurrFuncInfo(&fdata);  
  m_glslRW.VisitVarDecl_Impl(decl);
  m_glslRW.ResetCurrFuncInfo();
  return true;
}

bool GLSLKernelRewriter::VisitDeclStmt_Impl(clang::DeclStmt* stmt)
{
  m_glslRW.VisitDeclStmt_Impl(stmt);
  return true;
}

bool GLSLKernelRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)
{
  m_glslRW.VisitCStyleCastExpr_Impl(cast);
  return true;
}

bool GLSLKernelRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)
{
  m_glslRW.VisitArraySubscriptExpr_Impl(arrayExpr);
  return true;
}

bool GLSLKernelRewriter::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr)
{
  m_glslRW.VisitUnaryExprOrTypeTraitExpr_Impl(szOfExpr);
  return true;
}

bool GLSLKernelRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{
  const auto op = expr->getOpcodeStr(expr->getOpcode());
  //std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); //
  if(op == "*")
  {
    auto subExpr           = expr->getSubExpr();
    std::string exprInside = RecursiveRewrite(subExpr);
    const bool needOffset  = CheckIfExprHasArgumentThatNeedFakeOffset(exprInside);
    if(needOffset) // process reduction for ++ and  --
      return kslicer::KernelRewriter::VisitUnaryOperator_Impl(expr);
    else
      m_glslRW.VisitUnaryOperator_Impl(expr);
  }
  else if(op == "++" || op == "--") // process reduction for ++ and  --
    return kslicer::KernelRewriter::VisitUnaryOperator_Impl(expr);
  else
    m_glslRW.VisitUnaryOperator_Impl(expr);

  return true;
}


bool GLSLKernelRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call)
{
  return m_glslRW.VisitCXXConstructExpr_Impl(call);
}

void GLSLKernelRewriter::RewriteTextureAccess(clang::CXXOperatorCallExpr* expr, clang::Expr* a_assignOp, const std::string& rhsText)
{
  if(!WasNotRewrittenYet(expr))
    return;

  if(a_assignOp != nullptr && !WasNotRewrittenYet(expr))
    return;

  const clang::QualType leftType = expr->getArg(0)->getType();
  if(leftType->isPointerType()) // buffer ? --> ignore
    return;

  if(!kslicer::IsTexture(leftType))
    return;

  std::string objName = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getExprLoc()), m_compiler);

  // (1) process if member access
  //
  bool shouldRewrite = false;

  auto pMember = m_codeInfo->allDataMembers.find(objName);
  if(pMember != m_codeInfo->allDataMembers.end())
    shouldRewrite = true;

  // (2) process if kernel argument access
  //
  std::string dataType = "";
  for(const auto& arg : m_currKernel.args)
  {
    if(arg.name == objName)
    {
      shouldRewrite = true;
      dataType = arg.containerDataType;
      break;
    }
  }

  std::string convertedType = "vec4";
  if(dataType == "unsigned int" || dataType == "unsigned" || dataType == "uint" || dataType == "uint2" || dataType == "uint3" || dataType == "uint4")
    convertedType = "uvec4";
  else if(dataType == "int" || dataType == "int2" || dataType == "int3" || dataType == "int4")
    convertedType = "ivec4";

  // (3) rewrite if do needed
  //
  if(shouldRewrite)
  {
    std::string indexText = RecursiveRewrite(expr->getArg(1));
    if(a_assignOp != nullptr && WasNotRewrittenYet(a_assignOp)) // write
    {
      std::string lastRewrittenText = std::string("imageStore") + "(" + objName + ", " + indexText + ", " + convertedType + "(" + rhsText + "))";
      ReplaceTextOrWorkAround(a_assignOp->getSourceRange(), lastRewrittenText);
      MarkRewritten(a_assignOp);
    }
    else if(WasNotRewrittenYet(expr))                           // read
    {
      std::string lastRewrittenText = std::string("imageLoad") + "(" + objName + ", " + indexText + ")";
      ReplaceTextOrWorkAround(expr->getSourceRange(), lastRewrittenText);
      MarkRewritten(expr);
    }
  }
}


bool GLSLKernelRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)
{
  std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler);
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);

  //if(debugText.find("m_bodies") != std::string::npos)
  //{
  //  int a = 2;
  //}

  if(op == "+=" || op == "-=" || op == "*=") // detect reduction access
  {
    return kslicer::KernelRewriter::VisitCXXOperatorCallExpr_Impl(expr); // process reduction access in KernelRewriter
  }
  else if(expr->isAssignmentOp()) // detect 'a_brightPixels[coord] = color'
  {
    clang::Expr* left = kslicer::RemoveImplicitCast(expr->getArg(0));
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = kslicer::GetRangeSourceCode(clang::SourceRange(leftOp->getOperatorLoc()), m_compiler);
      if((op2 == "]" || op2 == "[" || op2 == "[]") && WasNotRewrittenYet(expr)) // detect 'a_brightPixels[coord]'
      {
        const clang::Expr* leftLeft = kslicer::RemoveImplicitCast(leftOp->getArg(0));
        if(kslicer::IsTexture(leftLeft->getType())) // detect that 'a_brightPixels' is texture type
        {
          std::string assignExprText = RecursiveRewrite(expr->getArg(1));
          RewriteTextureAccess(leftOp, expr, assignExprText);
        }
        else
          return kslicer::KernelRewriter::VisitCXXOperatorCallExpr_Impl(expr); // detect reduction access
      }
    }
    else
      return kslicer::KernelRewriter::VisitCXXOperatorCallExpr_Impl(expr); // detect reduction access
  }
  else if(op == "+" || op == "-" || op == "*" || op == "/")                // WARNING! Could be also unary "-" and "+"
  {
    //clang::Expr* left  = kslicer::RemoveImplicitCast(expr->getArg(0));   // ok
    //clang::Expr* tight = kslicer::RemoveImplicitCast(expr->getArg(1));   // WARNING! Could be also unary "-" and "+"
    //const std::string leftType  = left->getType().getAsString();
    //const std::string rightType = tight->getType().getAsString();
  }
  else if((op == "]" || op == "[" || op == "[]") && WasNotRewrittenYet(expr)) // swap access of coords fopr mat4: mat[row][col] --> mat[col][row]
  {
    const clang::Expr* nodes[3] = {nullptr,nullptr,nullptr};
    m_glslRW.Get2DIndicesOfFloat4x4(expr,nodes);

    if(nodes[0] != nullptr)
    {
      std::string varName = RecursiveRewrite(nodes[0]);
      std::string yText   = RecursiveRewrite(nodes[1]);
      std::string xText   = RecursiveRewrite(nodes[2]);
      std::string resVal = varName + "[" + yText + "][" + xText + "]";
      ReplaceTextOrWorkAround(expr->getSourceRange(), resVal);
      MarkRewritten(expr);
    }
    else
      RewriteTextureAccess(expr, nullptr, "");
  }
  else
    return kslicer::KernelRewriter::VisitCXXOperatorCallExpr_Impl(expr);

  return true;
}

bool GLSLKernelRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* call)
{
  clang::CXXMethodDecl* fDecl = call->getMethodDecl();
  if(fDecl != nullptr && WasNotRewrittenYet(call))
  {
    //std::string debugText = kslicer::GetRangeSourceCode(call->getSourceRange(), m_compiler);
    std::string fname     = fDecl->getNameInfo().getName().getAsString();
    clang::Expr* pTexName =	call->getImplicitObjectArgument();
    std::string objName   = kslicer::GetRangeSourceCode(pTexName->getSourceRange(), m_compiler);

    if(fname == "sample" || fname == "Sample")
    {
      bool needRewrite = true;
      const clang::QualType leftType = pTexName->getType();
      std::string typeName = leftType.getAsString();
      int texCoordId = 1;                                        ///<! texture.sample(m_sampler, texCoord)
      if(leftType->isPointerType()) // buffer ? --> ignore
      {
        const auto qt2        = leftType->getPointeeType();
        std::string typeName2 = qt2.getAsString();
        ReplaceFirst(typeName2, "const ",  ""); // remove 'const '
        ReplaceFirst(typeName2, "struct ", ""); // remove 'struct '
        ReplaceFirst(objName, "->",  "");       // remove '->'
        needRewrite = kslicer::IsCombinedImageSamplerTypeName(typeName2);
        texCoordId  = 0;                                         ///<! combinedObject->sample(texCoord)
      }
      else if(!kslicer::IsTexture(leftType))
      {
        needRewrite = false;
      }

      if(needRewrite)
      {
        //clang::Expr* samplerExpr = call->getArg(0); // TODO: process sampler? use separate sampler and image?
        //clang::Expr* txCoordExpr = call->getArg(1);
        //std::string text1 = kslicer::GetRangeSourceCode(samplerExpr->getSourceRange(), m_compiler);
        //std::string text2 = kslicer::GetRangeSourceCode(txCoordExpr->getSourceRange(), m_compiler);
        const std::string texCoord = RecursiveRewrite(call->getArg(texCoordId));
        const std::string lastRewrittenText = std::string("texture") + "(" + objName + ", " + texCoord + ")";
        ReplaceTextOrWorkAround(call->getSourceRange(), lastRewrittenText);
        MarkRewritten(call);
      }
    }
    else if(m_codeInfo->megakernelRTV && m_codeInfo->IsKernel(fname) && WasNotRewrittenYet(call)) // replace 'Texture2D<...>&' arguments to '0' if this is not sampler
    {
      bool foundAtLeastOneTexReference = false;
      for(unsigned i=0;i<call->getNumArgs();i++)
      {
        const clang::QualType argT = call->getArg(i)->getType();
        if(!argT.isConstQualified() && kslicer::IsTexture(argT))
        {
          foundAtLeastOneTexReference = true;
          break;
        }
      }

      if(foundAtLeastOneTexReference)
      {
        std::stringstream callOut;
        callOut << fname.c_str() << "(";
        for(unsigned i=0;i<call->getNumArgs();i++)
        {
          const clang::QualType argT = call->getArg(i)->getType();
          std::string argText = "";
          if(!argT.isConstQualified() && kslicer::IsTexture(argT))
            argText = "0";
          else
            argText = RecursiveRewrite(call->getArg(i));

          callOut << argText;
          if(i < call->getNumArgs()-1)
            callOut << ", ";
        }
        callOut << ")";
        ReplaceTextOrWorkAround(call->getSourceRange(), callOut.str());
        MarkRewritten(call);
      }
    }

  }

  return kslicer::KernelRewriter::VisitCXXMemberCallExpr_Impl(call);
}

bool GLSLKernelRewriter::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)
{
  const clang::ValueDecl* pDecl = expr->getDecl();
  if(!clang::isa<clang::ParmVarDecl>(pDecl))
    return true;

  clang::QualType qt = pDecl->getType();
  if(qt->isPointerType() || qt->isReferenceType()) // we can't put references to push constants
    return true;

  const std::string textOri = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); //
  //const std::string textRes = RecursiveRewrite(expr);
  if(m_userArgs.find(textOri) != m_userArgs.end() && WasNotRewrittenYet(expr))
  {
    if(!m_codeInfo->megakernelRTV || m_currKernel.isMega)
    {
      //ReplaceTextOrWorkAround(expr->getSourceRange(), std::string("kgenArgs.") + textOri);
      m_rewriter.ReplaceText(expr->getSourceRange(), std::string("kgenArgs.") + textOri);
      MarkRewritten(expr);
    }
  }

  return true;
}

bool GLSLKernelRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast)
{
  return m_glslRW.VisitImplicitCastExpr_Impl(cast);
}

bool GLSLKernelRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)
{
  return KernelRewriter::VisitMemberExpr_Impl(expr);
}

bool GLSLKernelRewriter::VisitReturnStmt_Impl(clang::ReturnStmt* ret)
{
  return KernelRewriter::VisitReturnStmt_Impl(ret);
}

bool GLSLKernelRewriter::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr)
{
  return KernelRewriter::VisitCompoundAssignOperator_Impl(expr);
}

bool GLSLKernelRewriter::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler);
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);

  if(expr->isAssignmentOp())
  {
    clang::Expr* left = expr->getLHS();
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = kslicer::GetRangeSourceCode(clang::SourceRange(leftOp->getOperatorLoc()), m_compiler);
      if((op2 == "]" || op2 == "[" || op2 == "[]") && WasNotRewrittenYet(expr))
      {
        std::string assignExprText = RecursiveRewrite(expr->getRHS());
        RewriteTextureAccess(leftOp, expr, assignExprText);
        return true;
      }
    }
  }

  return KernelRewriter::VisitBinaryOperator_Impl(expr);
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::GLSLCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                                 kslicer::KernelInfo& a_kernel, const std::string& fakeOffs)
{
  return std::make_shared<GLSLKernelRewriter>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs);
}

std::string kslicer::GLSLCompiler::RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds)
{
  std::string names[3];
  this->GetThreadSizeNames(names);

  const std::string names0 = std::string("kgenArgs.") + names[0];
  const std::string names1 = std::string("kgenArgs.") + names[1];
  const std::string names2 = std::string("kgenArgs.") + names[2];

  if(threadIds.size() == 1)
    return threadIds[0].name;
  else if(threadIds.size() == 2)
    return std::string("fakeOffset(") + threadIds[0].name + "," + threadIds[1].name + "," + names0 + ")";
  else if(threadIds.size() == 3)
    return std::string("fakeOffset2(") + threadIds[0].name + "," + threadIds[1].name + "," + threadIds[2].name + "," + names0 + "," + names1 + ")";
  else
    return "a_globalTID.x";
} 
