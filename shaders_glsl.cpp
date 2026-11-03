#include "kslicer.h"
#include "initial_pass.h"
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
  const std::filesystem::path templatePathMega   = templatesFolder / "generated_mega.glsl"; // only for megakernels of RTV control functions ("-megakernel 1"); IPV kernels always use usual template
  const std::filesystem::path templatePathUsual  = templatesFolder / "generated.glsl";
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

  std::stringstream buildSH; // script text, written to file by WriteShaderBuildScript at the end
  #ifndef _WIN32
  buildSH << "#!/bin/sh" << std::endl;
  #endif
  for(auto& kernel : kernels.items())
  {
    nlohmann::json currKerneJson = copy;
    currKerneJson["Kernel"] = kernel.value();

    std::string kernelName     = std::string(kernel.value()["Name"]);
    bool useRayTracingPipeline = kernel.value()["UseRayGen"];
    const bool vulkan11        = kernel.value()["UseSubGroups"] || a_codeInfo->persistentRTV;
    const bool vulkan12        = useRayTracingPipeline;
    needRTDummies              = needRTDummies || useRayTracingPipeline;
    if(useRayTracingPipeline) 
    {
      //std::ofstream file(a_codeInfo->mainClassFileName.parent_path() / "z_debug.json");
      //file << std::setw(2) << currKerneJson; //
      //file.close();
      
      kslicer::uset<std::string> intersectionShaders;
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
        std::string outFileName_RHIT = kernelName + std::string("_") + std::string(impl["ClassName"]) + "_" + std::string(intersectionShader["Name"]) + "_hit.glsl";
        std::string outFileName_RINT = kernelName + std::string("_") + std::string(impl["ClassName"]) + "_" + std::string(intersectionShader["Name"]) + "_int.glsl";

        if(intersectionShaders.find(outFileName_RHIT) != intersectionShaders.end())
          continue;
         
        intersectionShaders.insert(outFileName_RHIT);
        
        std::string target = "vulkan1.2";

        if(!rcHitForIntersectionIsGenerated)
        {
          kslicer::ApplyJsonToTemplate(templatePathHitShd.c_str(), shaderPath / "z_trace_custom_hit.glsl", ISData);
          buildSH << "glslangValidator -V --target-env " + target + " -S rchit " << "z_trace_custom_hit.glsl" << " -o " << "z_trace_custom_hit.glsl" << ".spv" << " -DGLSL -I.. ";
          for(auto folder : ignoreFolders)
            buildSH << "-I" << folder.c_str() << " ";
          buildSH << std::endl;
          rcHitForIntersectionIsGenerated = false;
        }

        //kslicer::ApplyJsonToTemplate(templatePathHitShd.c_str(), shaderPath / outFileName_RHIT, ISData);
        kslicer::ApplyJsonToTemplate(templatePathIntShd.c_str(), shaderPath / outFileName_RINT, ISData);

        buildSH << "glslangValidator -V --target-env " + target + " -S rint " << outFileName_RINT.c_str() << " -o " << outFileName_RINT.c_str() << ".spv" << " -DGLSL -I.. ";
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
  
              std::string outFileName  = kernelName + "_" + std::string(impl["ClassName"]) + "_" + std::string(member["Name"]) + "_call.glsl";
        
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
    const bool isMega = a_codeInfo->megakernelsByName.find(kernelName) != a_codeInfo->megakernelsByName.end(); // RTV control function joined to megakernel
    const std::filesystem::path templatePath = isMega ? templatePathMega : templatePathUsual;
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

  kslicer::WriteShaderBuildScript(shaderPath / scriptName, buildSH.str(), a_settings.mtShaderCompile);
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
////////////////////////////////////////  GLSLRewriter  ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::umap<std::string, std::string> kslicer::ListGLSLVectorReplacements()
{
  kslicer::umap<std::string, std::string> m_vecReplacements;
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
  
  kslicer::umap<std::string, std::string> m_vecReplacementsConst;
  for(auto r : m_vecReplacements)
    m_vecReplacementsConst[std::string("const ") + r.first] = std::string("const ") + m_vecReplacements[r.first];
  
  for(auto rc : m_vecReplacementsConst)
    m_vecReplacements[rc.first] = rc.second;
    
  return m_vecReplacements;
}

kslicer::uset<std::string> kslicer::ListPredefinedMathTypes()
{
  auto types = kslicer::ListGLSLVectorReplacements();
  kslicer::uset<std::string> res;
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

std::vector<std::pair<std::string, std::string> > kslicer::SortByKeysByLen(const kslicer::umap<std::string, std::string>& a_map)
{
  std::vector<std::pair<std::string, std::string> > res;
  res.reserve(a_map.size());
  for(auto p : a_map)
    res.push_back(p);
  std::sort(res.begin(), res.end(), [](auto& a, auto& b) { return (a.first.size() != b.first.size()) ? (a.first.size() > b.first.size()) : (a.first < b.first); });
  return res;
}

kslicer::GLSLRewriter::GLSLRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit) : FunctionRewriter2(R,a_compiler,a_codeInfo)
{
  m_vecReplacements  = kslicer::ListGLSLVectorReplacements();
  m_vecReplacements2 = kslicer::SortByKeysByLen(m_vecReplacements);
  m_funReplacements["fmin"]  = "min";
  m_funReplacements["fmax"]  = "max";
  m_funReplacements["fminf"] = "min";
  m_funReplacements["fmaxf"] = "max";
  m_funReplacements["fsqrt"] = "sqrt";
  m_funReplacements["sqrtf"] = "sqrt";
  m_funReplacements["fabs"]  = "abs";
  m_funReplacements["to_float4"] = "vec4";
  m_funReplacements["InterlockedAdd"] = "atomicAdd";
  m_funReplacements["AtomicAdd"]      = "atomicAdd";
  m_shit = a_shit;
}

std::string kslicer::GLSLRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  
  while(clang::isa<clang::ImplicitCastExpr>(expr))
    expr = clang::dyn_cast<clang::ImplicitCastExpr>(expr)->IgnoreImpCasts();

  GLSLRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
  sFeatures = sFeatures || rvCopy.sFeatures;
  MarkRewritten(expr);
  return kslicer::GetRewrittenText(m_rewriter, expr->getSourceRange());
}

kslicer::ShaderFeatures kslicer::GetUsedShaderFeaturesFromTypeName(const std::string& a_str)
{
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

std::string kslicer::GLSLRewriter::RewriteStdVectorTypeStr(const std::string& a_str) const
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
std::string kslicer::GLSLRewriter::RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const
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

std::string kslicer::GLSLRewriter::RewriteImageType(const std::string& a_containerType, const std::string& a_containerDataType, kslicer::TEX_ACCESS a_accessType, std::string& outImageFormat) const
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

std::string kslicer::GLSLRewriter::VectorTypeContructorReplace(const std::string& fname, const std::string& callText)
{
  auto p = m_vecReplacements.find(fname);
  if(p == m_vecReplacements.end())
    return std::string("make_") + fname + callText;
  else
    return p->second + callText;
}

bool kslicer::GLSLRewriter::NeedsVectorTypeRewrite(const std::string& a_str) // TODO: make this implementation more smart, bad implementation actually!
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

bool kslicer::GLSLRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)
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
      ReplaceText(arrayExpr->getSourceRange(), textRes); // process shitty global pointers
      MarkRewritten(right);
      break;
    }
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr)
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
    ReplaceText(szOfExpr->getSourceRange(), str.str());
    MarkRewritten(szOfExpr);
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)
{
  if(m_kernelMode)
    return VisitCXXMemberCallExprKernel(f);
  return true;
}

bool kslicer::GLSLRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)
{
  if(m_kernelMode)
    return VisitCXXOperatorCallExprKernel(expr);

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

      ReplaceText(expr->getSourceRange(), resVal);
      MarkRewritten(expr);
      return true;
    }
  }

  return FunctionRewriter::VisitCXXOperatorCallExpr_Impl(expr);
}

std::string kslicer::GLSLRewriter::RewriteFuncDecl(clang::FunctionDecl* fDecl)
{
  std::string retT   = RewriteStdVectorTypeStr(fDecl->getReturnType().getAsString());
  std::string fname  = fDecl->getNameInfo().getName().getAsString();

  if(m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->hasPrefix)          // alter function name if it has any prefix
  { 
    if(fname.find(m_pCurrFuncInfo->prefixName) == std::string::npos)
      fname = m_pCurrFuncInfo->prefixName + "_" + fname;
  }
  else if(m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->name != fname) // alter function name if was changed
    fname = m_pCurrFuncInfo->name;

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

bool kslicer::GLSLRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)
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

bool kslicer::GLSLRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)
{
  if(m_kernelMode)
    return VisitMemberExprKernel(expr);

  if(expr->isArrow() && WasNotRewrittenYet(expr->getBase()) )
  {
    const auto exprText     = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    const std::string lText = exprText.substr(exprText.find("->")+2);
    const std::string rText = RecursiveRewrite(expr->getBase());
    ReplaceText(expr->getSourceRange(), rText + "." + lText);
    MarkRewritten(expr->getBase());
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{
  if(m_kernelMode && VisitUnaryOperatorKernel(expr))
    return true;

  const auto op = expr->getOpcodeStr(expr->getOpcode());
  if((op == "*" || op == "&") && WasNotRewrittenYet(expr->getSubExpr()) )
  {
    std::string text = RecursiveRewrite(expr->getSubExpr());
    ReplaceText(expr->getSourceRange(), text);
    MarkRewritten(expr->getSubExpr());
  }

  return true;
}

std::string kslicer::GLSLRewriter::CompleteFunctionCallRewrite(clang::CallExpr* call)
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

bool kslicer::GLSLRewriter::VisitCallExpr_Impl(clang::CallExpr* call)
{
  if(m_kernelMode && VisitCallExprKernel(call))
    return true;

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

      ReplaceText(call->getSourceRange(), textRes);
      MarkRewritten(call);
    }
  }
  else if(makeSmth != "" && pVecMaker != m_vecReplacements.end() && call->getNumArgs() !=0 && WasNotRewrittenYet(call) )
  {
    const std::string rewrittenRes = pVecMaker->second + "(" + CompleteFunctionCallRewrite(call);
    
    ReplaceText(call->getSourceRange(), rewrittenRes);
    MarkRewritten(call);
  }
  else if(fname == "mul4x4x4" && call->getNumArgs() == 2 && WasNotRewrittenYet(call))
  {
    const std::string A = RecursiveRewrite(call->getArg(0));
    const std::string B = RecursiveRewrite(call->getArg(1));
    ReplaceText(call->getSourceRange(), "(" + A + "*" + B + ")");
    MarkRewritten(call);
  }
  else if(fname == "lerp" && call->getNumArgs() == 3 && WasNotRewrittenYet(call))
  {
    const std::string A = RecursiveRewrite(call->getArg(0));
    const std::string B = RecursiveRewrite(call->getArg(1));
    const std::string C = RecursiveRewrite(call->getArg(2));
    ReplaceText(call->getSourceRange(), "mix(" + A + ", " + B + ", " + C + ")");
    MarkRewritten(call);
  }
  else if(fname == "atan2" && call->getNumArgs() == 2 && WasNotRewrittenYet(call))
  {
    const std::string arg1 = RecursiveRewrite(call->getArg(0));
    const std::string arg2 = RecursiveRewrite(call->getArg(1));
    ReplaceText(call->getSourceRange(), "atan(" + arg1 + "," + arg2 + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_int32" || fname == "as_int") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceText(call->getSourceRange(), "floatBitsToInt(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_uint32" || fname == "as_uint") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceText(call->getSourceRange(), "floatBitsToUint(" + text + ")");
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
    ReplaceText(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if((fname == "inverse4x4" || fname == "inverse3x3" || fname == "inverse2x2") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceText(call->getSourceRange(), "inverse(" + text + ")");
    MarkRewritten(call);
  }
  else if(pFoundSmth != m_funReplacements.end() && WasNotRewrittenYet(call))
  {
    std::string lastRewrittenText = pFoundSmth->second + "(" + CompleteFunctionCallRewrite(call);
    ReplaceText(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if(fDecl->isInStdNamespace() && WasNotRewrittenYet(call)) // remove "std::"
  {
    std::string lastRewrittenText = fname + "(" + CompleteFunctionCallRewrite(call);
    ReplaceText(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl) // special case for process multiple decls in line, like 'int i,j,k=2'
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
      ReplaceText(decl->getSourceRange(), resExpr);
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

void kslicer::FunctionRewriter::CkeckAndProcessForThreadLocalVarDecl(clang::VarDecl* decl)
{
  for (const clang::Attr* attr : decl->attrs()) 
  {
    //const std::string attrType = attr->getSpelling();
    const std::string attrText  = kslicer::GetRangeSourceCode(attr->getRange(), m_compiler);
    const std::string debugText = kslicer::GetRangeSourceCode(decl->getSourceRange(), m_compiler);

    clang::QualType varType = decl->getType();

    if (varType->isArrayType() && attrText == "threadlocal") 
    {
      std::cout << "  found: " << attrText.c_str() << " for " << debugText.c_str() << std::endl;
      std::string varName = decl->getNameAsString();
      
      const clang::ArrayType* arrayType = varType->getAsArrayTypeUnsafe();
      clang::QualType elementType = arrayType->getElementType();
      
      int arraySize = 0;
      if (const clang::ConstantArrayType* constantArrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType))
        arraySize = int(constantArrayType->getSize().getLimitedValue());

      const clang::DeclStmt* declStmt = getParentDeclContext(decl, m_compiler.getASTContext());
      if(declStmt != nullptr && WasNotRewrittenYet(declStmt))
      {
        std::stringstream strOut;
        strOut << "// " << debugText.c_str() << "; // was moved to global scope in shader"; // 123
        ReplaceText(declStmt->getSourceRange(), strOut.str());
        MarkRewritten(declStmt);
        
        kslicer::ArrayData array;
        array.arrayName = varName;
        array.elemType  = elementType.getAsString();
        array.arraySize = arraySize;
        
        auto p = m_codeInfo->m_threadLocalArrays.find(varName.c_str());
        if(p == m_codeInfo->m_threadLocalArrays.end())
        {
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

}

bool kslicer::GLSLRewriter::VisitVarDecl_Impl(clang::VarDecl* decl)
{
  if(clang::isa<clang::ParmVarDecl>(decl)) // process else-where (VisitFunctionDecl_Impl)
    return true;

  if(m_kernelMode && m_pCurrFuncInfo == nullptr)
  {
    kslicer::FuncData fdata;
    fdata.astNode  = m_pCurrKernel->astNode;
    fdata.name     = m_pCurrKernel->name;
    fdata.isKernel = true;
    m_pCurrFuncInfo = &fdata;
    CkeckAndProcessForThreadLocalVarDecl(decl);
    m_pCurrFuncInfo = nullptr;
  }
  else
    CkeckAndProcessForThreadLocalVarDecl(decl);

  const auto qt      = decl->getType();
  const auto pValue  = decl->getAnyInitializer();

  //const std::string debugText = kslicer::GetRangeSourceCode(decl->getSourceRange(), m_compiler);
  //const std::string debugTextVal = kslicer::GetRangeSourceCode(pValue->getSourceRange(), m_compiler);
  const std::string varType = qt.getAsString();

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

    ReplaceText(decl->getSourceRange(), lastRewrittenText);
    MarkRewritten(pValue);
  }
  return true;
}

bool kslicer::GLSLRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)
{
  clang::QualType qt   = cast->getTypeAsWritten();
  clang::Expr* 	  next = cast->getSubExpr();
  std::string typeCast = qt.getAsString();
  //const std::string debugText = kslicer::GetRangeSourceCode(cast->getSourceRange(), m_compiler);
  if(WasNotRewrittenYet(next))
  {
    typeCast = RewriteStdVectorTypeStr(typeCast);
    const std::string exprText = RecursiveRewrite(next);

    ReplaceText(cast->getSourceRange(), typeCast + "(" + exprText + ")");
    MarkRewritten(next);
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast)
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
      ReplaceText(call->getSourceRange(), textRes);    //
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
      ReplaceText(next->getSourceRange(), castTo + "(" + exprText + ")");
      MarkRewritten(next);
    }
  }

  return true;
}

void kslicer::GLSLRewriter::Get2DIndicesOfFloat4x4(const clang::CXXOperatorCallExpr* expr, const clang::Expr* out[3])
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
  static auto vecReplacements = kslicer::SortByKeysByLen(ListGLSLVectorReplacements());
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
  return std::make_shared<GLSLRewriter>(R, a_compiler, a_codeInfo, a_shit);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Kernel mode of GLSLRewriter (after InitKernelData): kernels and member functions which are called from kernels
//

bool kslicer::GLSLRewriter::VisitCallExprKernel(clang::CallExpr* call)
{
  // (#1) check if buffer/pointer to global memory is passed to a function
  //
  std::vector<kslicer::ArgMatch> usedArgMatches = kslicer::MatchCallArgsForKernel(call, *m_pCurrKernel, m_compiler);
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
    m_pCurrKernel->shittyFunctions.push_back(func);

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

    ReplaceText(call->getSourceRange(), rewrittenRes);
    MarkRewritten(call);
  }
  else if (m_pCurrKernel->pattern == kslicer::PATTERN_TP::PATTERN_RTV && rewriteDueToFakeOffset)
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
    ReplaceText(call->getSourceRange(), rewrittenRes);
    MarkRewritten(call);
  }
  else
    return false;

  return true;
}

bool kslicer::GLSLRewriter::VisitUnaryOperatorKernel(clang::UnaryOperator* expr)
{
  clang::Expr* subExpr = expr->getSubExpr();
  if(subExpr == nullptr)
    return true;

  const auto op = expr->getOpcodeStr(expr->getOpcode());
  if(op == "++" || op == "--") // detect ++ and -- for reduction
  {
    auto opRange = expr->getSourceRange();
    if(opRange.getEnd()   <= m_pCurrKernel->loopInsides.getBegin() ||
       opRange.getBegin() >= m_pCurrKernel->loopInsides.getEnd() ) // not inside loop
      return true;

    std::string leftStr = kslicer::GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);
    auto p = m_pCurrKernel->usedMembers.find(leftStr);
    if(p != m_pCurrKernel->usedMembers.end() && WasNotRewrittenYet(expr))
    {
      std::string leftStr2   = RecursiveRewrite(expr->getSubExpr());
      std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_pCurrKernel->GetDim(), m_pCurrKernel->wgSize);
      ReplaceText(expr->getSourceRange(), leftStr2 + "Shared[" + localIdStr + "]++");
      MarkRewritten(expr);
    }
    return true;
  }
  else if(op != "*")
    return false;

  // check fake offset by original text: RecursiveRewrite of operand marks it as rewritten, so '*' would not be removed in function mode
  //
  const std::string exprText = kslicer::GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);
  if(!CheckIfExprHasArgumentThatNeedFakeOffset(exprText))
    return false;

  if(expr->canOverflow()) // -UnaryOperator ...'LiteMath::uint':'unsigned int' lvalue prefix '*' cannot overflow
    return true;

  const std::string exprInside = RecursiveRewrite(subExpr);
  if(CheckIfExprHasArgumentThatNeedFakeOffset(exprInside) && WasNotRewrittenYet(expr))
  {
    if(m_codeInfo->megakernelRTV || m_fakeOffsetExp == "")
      ReplaceText(expr->getSourceRange(), exprInside);
    else
      ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
    MarkRewritten(expr);
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitMemberExprKernel(clang::MemberExpr* expr)
{
  std::string rewrittenText;
  if(NeedToRewriteMemberExpr(expr, rewrittenText))
  {
    ReplaceText(expr->getSourceRange(), rewrittenText);
    MarkRewritten(expr);
  }
  return true;
}

void kslicer::GLSLRewriter::RewriteTextureAccess(clang::CXXOperatorCallExpr* expr, clang::Expr* a_assignOp, const std::string& rhsText)
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
  for(const auto& arg : m_pCurrKernel->args)
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
      ReplaceText(a_assignOp->getSourceRange(), lastRewrittenText);
      MarkRewritten(a_assignOp);
    }
    else if(WasNotRewrittenYet(expr))                           // read
    {
      std::string lastRewrittenText = std::string("imageLoad") + "(" + objName + ", " + indexText + ")";
      ReplaceText(expr->getSourceRange(), lastRewrittenText);
      MarkRewritten(expr);
    }
  }
}

bool kslicer::GLSLRewriter::VisitCXXOperatorCallExprKernel(clang::CXXOperatorCallExpr* expr)
{
  std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler);
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);

  if(op == "+=" || op == "-=" || op == "*=") // detect reduction access
  {
    return FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(expr); // process reduction access in FunctionRewriter2
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
          return FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(expr); // detect reduction access
      }
    }
    else
      return FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(expr); // detect reduction access
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
    Get2DIndicesOfFloat4x4(expr,nodes);

    if(nodes[0] != nullptr)
    {
      std::string varName = RecursiveRewrite(nodes[0]);
      std::string yText   = RecursiveRewrite(nodes[1]);
      std::string xText   = RecursiveRewrite(nodes[2]);
      std::string resVal = varName + "[" + yText + "][" + xText + "]";
      ReplaceText(expr->getSourceRange(), resVal);
      MarkRewritten(expr);
    }
    else
      RewriteTextureAccess(expr, nullptr, "");
  }
  else
    return FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(expr);

  return true;
}

bool kslicer::NeedRewriteTextureArray(clang::CXXMemberCallExpr* a_call, std::string& objName, int& texCoordId)
{
  clang::Expr* pTexName =	a_call->getImplicitObjectArgument();
  bool needRewrite = true;
  const clang::QualType leftType = pTexName->getType();
  std::string typeName = leftType.getAsString();
  texCoordId = 1;  ///<! texture.sample(m_sampler, texCoord)
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
  return needRewrite;
}

bool ProcessTexArrayAccessGLSL(std::string& s)
{
    auto open  = s.find('[');
    auto close = s.rfind(']');

    if (open == std::string::npos || close == std::string::npos || close < open)
        return false;

    std::string inner = s.substr(open + 1, close - open - 1);
    s = s.substr(0, open + 1)
      + "nonuniformEXT(" + inner + ")"
      + s.substr(close);

    return true;
}

bool kslicer::GLSLRewriter::VisitCXXMemberCallExprKernel(clang::CXXMemberCallExpr* call)
{
  clang::CXXMethodDecl* fDecl = call->getMethodDecl();
  const clang::QualType qt    = call->getObjectType();
  const auto& thisTypeName    = qt.getAsString();

  if(fDecl != nullptr && WasNotRewrittenYet(call))
  {
    //std::string debugText = kslicer::GetRangeSourceCode(call->getSourceRange(), m_compiler);
    std::string fname = fDecl->getNameInfo().getName().getAsString();
    
    if(fname == "set_row" && kslicer::IsMatrixTypeName(thisTypeName))
    {
      clang::Expr* pMatName =	call->getImplicitObjectArgument();
      std::string  objName  = kslicer::GetRangeSourceCode(pMatName->getSourceRange(), m_compiler);
      std::string indexVal  = kslicer::GetRangeSourceCode(call->getArg(0)->getSourceRange(), m_compiler);
      std::string argValue  = RecursiveRewrite(call->getArg(1));
      
      std::string baseType = thisTypeName.substr(0, 1);
      std::string number   = thisTypeName.substr(thisTypeName.size()-1, 1);
      std::string lastRewrittenText = "set_row" + number + baseType + "(" + objName + ", " + indexVal + ", " + argValue + ")";

      ReplaceText(call->getSourceRange(), lastRewrittenText);
      MarkRewritten(call);
    }
    else if(fname == "set_col" && kslicer::IsMatrixTypeName(thisTypeName))
    {
      clang::Expr* pMatName =	call->getImplicitObjectArgument();
      std::string  objName  = kslicer::GetRangeSourceCode(pMatName->getSourceRange(), m_compiler);
      std::string indexVal  = kslicer::GetRangeSourceCode(call->getArg(0)->getSourceRange(), m_compiler);
      std::string argValue  = RecursiveRewrite(call->getArg(1));
      
      std::string lastRewrittenText = objName + "[" + indexVal + "] = " + argValue;
      ReplaceText(call->getSourceRange(), lastRewrittenText);
      MarkRewritten(call);
    }
    else if(fname == "sample" || fname == "Sample")
    {
      clang::Expr* pTexName =	call->getImplicitObjectArgument();
      std::string objName   = kslicer::GetRangeSourceCode(pTexName->getSourceRange(), m_compiler);
      int texCoordId   = 0;    
      bool needRewrite = kslicer::NeedRewriteTextureArray(call, objName, texCoordId);

      if(needRewrite)
      {
        ProcessTexArrayAccessGLSL(objName); // "m_textures[...]" --> "m_textures[nonuniformEXT(...)]"
        const std::string texCoord = RecursiveRewrite(call->getArg(texCoordId));
        const std::string lastRewrittenText = std::string("textureLod") + "(" + objName + ", " + texCoord + ", 0)";
        ReplaceText(call->getSourceRange(), lastRewrittenText);
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
        ReplaceText(call->getSourceRange(), callOut.str());
        MarkRewritten(call);
      }
    }

  }

  return RewriteMemberCallInKernel(call);
}

bool kslicer::GLSLRewriter::RewriteMemberCallInKernel(clang::CXXMemberCallExpr* f)
{
  // Get name of function
  //
  const clang::DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const clang::DeclarationName dn      = dni.getName();
        std::string fname       = dn.getAsString();

  if(kslicer::IsCalledWithArrowAndVirtual(f) && WasNotRewrittenYet(f))
  {
    auto buffAndOffset = kslicer::GetVFHAccessNodes(f, m_compiler);
    if(buffAndOffset.buffName != "" && buffAndOffset.offsetName != "")
    {
      std::string buffText2  = buffAndOffset.buffName;
      std::string offsetText = buffAndOffset.offsetName; //GetRangeSourceCode(buffAndOffset.offsetNode->getSourceRange(), m_compiler); 
      
      std::string textCallNoName = "(" + offsetText; 
      if(f->getNumArgs() != 0)
        textCallNoName += ",";
        
      for(unsigned i=0;i<f->getNumArgs();i++)
      {
        const auto pParam                   = f->getArg(i);
        const clang::QualType typeOfParam   =	pParam->getType();
        const std::string typeNameRewritten = kslicer::ClearTypeName(typeOfParam.getAsString());
        if(m_codeInfo->dataClassNames.find(typeNameRewritten) != m_codeInfo->dataClassNames.end()) 
        {
          if(i==f->getNumArgs()-1)
            textCallNoName[textCallNoName.rfind(",")] = ' ';
          continue;
        }

        textCallNoName += RecursiveRewrite(f->getArg(i));
        if(i < f->getNumArgs()-1)
          textCallNoName += ",";
      }
      
      auto pBuffNameFromVFH = m_codeInfo->m_vhierarchy.find(buffAndOffset.interfaceTypeName);
      if(pBuffNameFromVFH != m_codeInfo->m_vhierarchy.end())
        buffText2 = pBuffNameFromVFH->second.objBufferName;

      std::string vcallFunc  = buffAndOffset.interfaceName + "_" + fname + "_" + buffText2 + textCallNoName + ")";
      ReplaceText(f->getSourceRange(), vcallFunc);
      MarkRewritten(f);
    }
  }

  // Get name of "this" type; we should check wherther this member is std::vector<T>  
  //
  const clang::QualType qt = f->getObjectType();
  const auto& thisTypeName = qt.getAsString();
  clang::CXXRecordDecl* typeDecl  = f->getRecordDecl(); 
  const std::string cleanTypeName = kslicer::CleanTypeName(thisTypeName);
  
  const bool isVector   = (typeDecl != nullptr && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl)) && thisTypeName.find("vector<") != std::string::npos; 
  const bool isRTX      = (thisTypeName == "struct ISceneObject") && (fname.find("RayQuery_") != std::string::npos);
  const auto pPrefix    = m_codeInfo->composPrefix.find(cleanTypeName);
  const bool isPrefixed = (pPrefix != m_codeInfo->composPrefix.end());
  
  if(isVector && WasNotRewrittenYet(f))
  {
    const std::string exprContent = kslicer::GetRangeSourceCode(f->getSourceRange(), m_compiler);
    const auto posOfPoint         = exprContent.find(".");
    std::string memberNameA       = exprContent.substr(0, posOfPoint);
    
    if(processFuncMember && m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->hasPrefix)
      memberNameA = m_pCurrFuncInfo->prefixName + "_" + memberNameA;

    if(fname == "size" || fname == "capacity")
    {
      const std::string memberNameB = memberNameA + "_" + fname;
      ReplaceText(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
      MarkRewritten(f);
    }
    else if(fname == "resize")
    {
      if(f->getSourceRange().getBegin() <= m_pCurrKernel->loopOutsidesInit.getEnd()) // TODO: SEEMS INCORECT LOGIC
      {
        assert(f->getNumArgs() == 1);
        const clang::Expr* currArgExpr  = f->getArgs()[0];
        std::string newSizeValue = RecursiveRewrite(currArgExpr); 
        std::string memberNameB  = memberNameA + "_size = " + newSizeValue;
        ReplaceText(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
        MarkRewritten(f);
      }
    }
    else if(fname == "push_back")
    {
      assert(f->getNumArgs() == 1);
      const clang::Expr* currArgExpr  = f->getArgs()[0];
      std::string newElemValue = RecursiveRewrite(currArgExpr);

      std::string memberNameB  = memberNameA + "_size";
      std::string resulingText = m_codeInfo->pShaderCC->RewritePushBack(memberNameA, memberNameB, newElemValue);
      ReplaceText(f->getSourceRange(), resulingText);
      MarkRewritten(f);
    }
    else if(fname == "data")
    {
      ReplaceText(f->getSourceRange(), memberNameA);
      MarkRewritten(f);
    }
    else 
    {
      kslicer::PrintError(std::string("Unsuppoted std::vector method") + fname, f->getSourceRange(), m_compiler.getSourceManager());
    }
  }
  else if((isRTX || isPrefixed) && WasNotRewrittenYet(f))
  {
    const auto exprContent = kslicer::GetRangeSourceCode(f->getSourceRange(), m_compiler);
    const auto posOfPoint  = exprContent.find("->"); // seek for "m_pImpl->Func()" 
    
    std::string memberNameA;
    if(isPrefixed && posOfPoint == std::string::npos)   // Func() inside composed class of m_pImpl
      memberNameA = pPrefix->second;
    else
      memberNameA = exprContent.substr(0, posOfPoint);  // m_pImpl->Func() inside main class
    
    //bool lastArgIsEmpty = false;
    //if((fname == "RayQuery_NearestHit" || fname == "RayQuery_AnyHit") && f->getNumArgs() == 3) // support for motion blur inside RTX
    //{
    //  const auto argText = GetRangeSourceCode(f->getArg(2)->getSourceRange(), m_compiler);
    //  lastArgIsEmpty = (argText == "");
    //  if(!lastArgIsEmpty)
    //    fname += "Motion";
    //}

    std::string resCallText = memberNameA + "_" + fname + "(";
    for(unsigned i=0;i<f->getNumArgs(); i++)
    {
      resCallText += RecursiveRewrite(f->getArg(i));
      //if(i == f->getNumArgs()-2 && lastArgIsEmpty)
      //  break;
      if(i!=f->getNumArgs()-1)
        resCallText += ", ";
    }
    resCallText += ")";
    ReplaceText(f->getSourceRange(), resCallText);
    MarkRewritten(f);
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)
{
  if(!m_kernelMode)
    return true;

  const clang::ValueDecl* pDecl = expr->getDecl();
  if(!clang::isa<clang::ParmVarDecl>(pDecl))
    return true;

  clang::QualType qt = pDecl->getType();
  if(qt->isPointerType() || qt->isReferenceType()) // we can't put references to push constants
    return true;

  const std::string textOri = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); //
  //const std::string textRes = RecursiveRewrite(expr);
  if(m_kernelUserArgs.find(textOri) != m_kernelUserArgs.end() && WasNotRewrittenYet(expr))
  {
    if(!m_codeInfo->megakernelRTV || m_pCurrKernel->isMega)
    {
      ReplaceText(expr->getSourceRange(), std::string("kgenArgs.") + textOri);
      MarkRewritten(expr);
    }
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitReturnStmt_Impl(clang::ReturnStmt* ret)
{
  clang::Expr* retExpr = ret->getRetValue();
  if(!m_kernelMode || retExpr == nullptr)
    return true;
  
  if(processFuncMember) // don't rewrite return statements for function members
    return true;
  
  if(WasNotRewrittenYet(ret) && m_pCurrKernel->isBoolTyped && !m_codeInfo->megakernelRTV)
  {
    // the kernel body is placed inside 'do { ... } while(false)' (see 'generated.glsl'), so 'break' leaves it; GLSL does not support goto
    //
    if(kslicer::IsInsideLoopOrSwitch(ret, m_compiler.getASTContext()))
      kslicer::PrintError("'return' inside a loop or switch of RTV kernel which returns bool is not supported, assign result to a variable and return it after the loop", ret->getSourceRange(), m_compiler.getSourceManager());
    std::string retExprText = RecursiveRewrite(retExpr);
    ReplaceText(ret->getSourceRange(), std::string("{ kgenExitCond = ") + retExprText + "; break; }");
    MarkRewritten(ret);
    return true;
  }

  clang::Expr* pRetExpr = ret->getRetValue();
  if(!clang::isa<clang::CallExpr>(pRetExpr))
    return true;
  
  clang::CallExpr* callExpr = clang::dyn_cast<clang::CallExpr>(pRetExpr);

  // assotiate this buffer with target hierarchy 
  //
  auto retQt = pRetExpr->getType();
  if(retQt->isPointerType())
    retQt = retQt->getPointeeType();

  std::string fname = callExpr->getDirectCallee()->getNameInfo().getName().getAsString();
  if(fname != "MakeObjPtr")
    return true;

  assert(callExpr->getNumArgs() == 2);
  const clang::Expr* firstArgExpr  = callExpr->getArgs()[0];
  const clang::Expr* secondArgExpr = callExpr->getArgs()[1];
 
  if(WasNotRewrittenYet(ret) && !m_codeInfo->megakernelRTV)
  { 
    // change 'return MakeObjPtr(objPtr, ObjData) to 'kgen_objPtr = objPtr'
    //
    std::string retExprText = RecursiveRewrite(firstArgExpr);
    ReplaceText(ret->getSourceRange(), std::string("{ kgen_objPtr = ") + retExprText + "; }"); // goto KGEN_EPILOG;  
    MarkRewritten(ret);
  }

  return true;
}

bool kslicer::GLSLRewriter::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  if(!m_kernelMode)
    return true;

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

  return FunctionRewriter2::VisitBinaryOperator_Impl(expr);
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::GLSLCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                                 kslicer::KernelInfo& a_kernel, const std::string& fakeOffs)
{
  auto pFunc = std::make_shared<GLSLRewriter>(R, a_compiler, a_codeInfo, a_kernel.currentShit);
  pFunc->InitKernelData(a_kernel, fakeOffs);
  return std::make_shared<KernelRewriter2>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs, pFunc);
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
