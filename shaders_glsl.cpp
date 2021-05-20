#include "kslicer.h"
#include "template_rendering.h"
#include <iostream>

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif


std::string GetFolderPath(const std::string& a_filePath);


void kslicer::GLSLCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const std::string& mainClassFileName, const std::vector<std::string>& includeToShadersFolders)
{
  std::string folderPath = GetFolderPath(mainClassFileName);
  std::string shaderPath = folderPath + "/" + this->ShaderFolder();
  #ifdef WIN32
  mkdir(shaderPath.c_str());
  #else
  mkdir(shaderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif
  
  // generate header for all used functions in GLSL code
  //
  const std::string outFileNameH  = GetFolderPath(mainClassFileName) + "/z_generated.cl";
  kslicer::ApplyJsonToTemplate("templates_glsl/common_generated.h", shaderPath + "/common_generated.h", a_kernelsJson);  
  
  // now generate all glsl shaders
  //
  const std::string templatePath = "templates_glsl/generated.glsl";
  
  nlohmann::json copy, kernels;
  for (auto& el : a_kernelsJson.items())
  {
    //std::cout << el.key() << std::endl;
    if(std::string(el.key()) == "Kernels")
      kernels = a_kernelsJson[el.key()];
    else
      copy[el.key()] = a_kernelsJson[el.key()];
  }
  
    
  std::ofstream buildSH(shaderPath + "/build.sh");
  buildSH << "#!/bin/sh" << std::endl;
  for(auto& kernel : kernels.items())
  {
    nlohmann::json currKerneJson = copy;
    currKerneJson["Kernel"] = kernel.value();
    
    std::string kernelName  = std::string(kernel.value()["Name"]);
    std::string outFileName = kernelName + ".glsl";
    std::string outFilePath = shaderPath + "/" + outFileName;
    kslicer::ApplyJsonToTemplate(templatePath.c_str(), outFilePath, currKerneJson);
    buildSH << "glslangValidator -V " << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -e " << kernelName.c_str() << " -DGLSL -I.. ";
    for(auto folder : includeToShadersFolders)
     buildSH << "-I" << folder.c_str() << " ";
    buildSH << std::endl;

    //glslangValidator -e myEntryPoint // is needed for auxilary kernels!
  }
    
  //nlohmann::json emptyJson;
  //std::string outFileServ = shaderPath + "/" + "serv_kernels.cpp";
  //kslicer::ApplyJsonToTemplate("templates/ser_circle.cxx", outFileServ, emptyJson);
  //buildSH << "../../circle -shader -c -emit-spirv " << outFileServ.c_str() << " -o " << outFileServ.c_str() << ".spv" << " -DUSE_CIRCLE_CC -I.. " << std::endl;
  
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
  a_strs[0] = "kgenArgs.iNumElementsX";
  a_strs[1] = "kgenArgs.iNumElementsY";
  a_strs[2] = "kgenArgs.iNumElementsZ";
}


std::string kslicer::GLSLCompiler::ProcessBufferType(const std::string& a_typeName) const 
{ 
  std::string type = a_typeName;
  ReplaceFirst(type, "*", "");
  ReplaceFirst(type, "const", "");
  return type; 
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////  GLSLFunctionRewriter  ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  

/**
\brief process local functions (data["LocalFunctions"]), float3 --> make_float3, std::max --> fmax and e.t.c.
*/
class GLSLFunctionRewriter : public kslicer::FunctionRewriter // 
{
public:
  
  GLSLFunctionRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo) : FunctionRewriter(R,a_compiler,a_codeInfo)
  { 
    m_vecReplacements["float2"] = "vec2";
    m_vecReplacements["float3"] = "vec3";
    m_vecReplacements["float4"] = "vec4";
    m_vecReplacements["int2"]   = "ivec2";
    m_vecReplacements["int3"]   = "ivec3";
    m_vecReplacements["int4"]   = "ivec4";
    m_vecReplacements["uint2"]  = "uvec2";
    m_vecReplacements["uint3"]  = "uvec3";
    m_vecReplacements["uint4"]  = "uvec4";
  }

  ~GLSLFunctionRewriter()
  {
  }

  bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl) override;

protected:
  
  std::unordered_map<std::string, std::string> m_vecReplacements;

  std::string RewriteVectorTypeStr(const std::string& a_str);
  std::string RewriteFuncDecl(clang::FunctionDecl* fDecl);

  std::string RecursiveRewrite(const clang::Stmt* expr) override;
};


std::string GLSLFunctionRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  GLSLFunctionRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
  return m_rewriter.getRewrittenText(expr->getSourceRange());
}

std::string GLSLFunctionRewriter::RewriteVectorTypeStr(const std::string& a_str)
{
  std::string typeStr = a_str;
  ReplaceFirst(typeStr, "LiteMath::", "");
  ReplaceFirst(typeStr, "glm::",      "");
  ReplaceFirst(typeStr, "struct ",    "");
  
  auto p = m_vecReplacements.find(typeStr);
  if(p == m_vecReplacements.end())
    return typeStr;
  
  return p->second;
}

std::string GLSLFunctionRewriter::RewriteFuncDecl(clang::FunctionDecl* fDecl)
{
  std::string retT   = RewriteVectorTypeStr(fDecl->getReturnType().getAsString()); 
  std::string fname  = fDecl->getNameInfo().getName().getAsString();
  std::string result = retT + " " + fname + "(";

  for(uint32_t i=0; i < fDecl->getNumParams(); i++)
  {
    const clang::ParmVarDecl* pParam  = fDecl->getParamDecl(i);
    const clang::QualType typeOfParam =	pParam->getType();
    result += RewriteVectorTypeStr(typeOfParam.getAsString()) + " " + pParam->getNameAsString();
   
    if(i!=fDecl->getNumParams()-1)
      result += ", ";
  }
  return result + ") ";
}

bool GLSLFunctionRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl) 
{ 
  if(clang::isa<clang::CXXMethodDecl>(fDecl)) // ignore methods here, for a while ... 
    return true;

  if(WasNotRewrittenYet(fDecl->getBody()))
  {
    const std::string funcDeclText = RewriteFuncDecl(fDecl);
    const std::string funcBodyText = RecursiveRewrite(fDecl->getBody());
 
    //auto debugMeIn = GetRangeSourceCode(call->getSourceRange(), m_compiler);     
    m_rewriter.ReplaceText(fDecl->getSourceRange(), funcDeclText + funcBodyText);
    MarkRewritten(fDecl->getBody());
  }

  return true; 
}


std::shared_ptr<kslicer::FunctionRewriter> kslicer::GLSLCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo)
{
  return std::make_shared<GLSLFunctionRewriter>(R, a_compiler, a_codeInfo);
}
