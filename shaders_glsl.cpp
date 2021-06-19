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


void kslicer::GLSLCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo)
{
  const auto& mainClassFileName       = a_codeInfo->mainClassFileName;
  const auto& includeToShadersFolders = a_codeInfo->includeToShadersFolders;

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
    std::string outFileName = kernelName + ".comp";
    std::string outFilePath = shaderPath + "/" + outFileName;
    kslicer::ApplyJsonToTemplate(templatePath.c_str(), outFilePath, currKerneJson);
    buildSH << "glslangValidator -V " << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
    for(auto folder : includeToShadersFolders)
     buildSH << "-I" << folder.c_str() << " ";
    buildSH << std::endl;

    //glslangValidator -e myEntryPoint // is needed for auxilary kernels!
  }
    
  if(a_codeInfo->usedServiceCalls.find("memcpy") != a_codeInfo->usedServiceCalls.end())
  {
    nlohmann::json dummy;
    kslicer::ApplyJsonToTemplate("templates_glsl/z_memcpy.glsl", shaderPath + "/z_memcpy.comp", dummy); // just file copy actually
    buildSH << "glslangValidator -V z_memcpy.comp -o z_memcpy.comp.spv";
    buildSH << std::endl;
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
  a_strs[0] = "kgenArgs.iNumElementsX";
  a_strs[1] = "kgenArgs.iNumElementsY";
  a_strs[2] = "kgenArgs.iNumElementsZ";
}


std::string kslicer::GLSLCompiler::ProcessBufferType(const std::string& a_typeName) const 
{ 
  std::string type = a_typeName;
  ReplaceFirst(type, "*", "");
  ReplaceFirst(type, "const", "");

  if(type[type.size()-1] == ' ')
    type = type.substr(0, type.size()-1);

  return type; 
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////  GLSLFunctionRewriter  ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  

std::unordered_map<std::string, std::string> ListGLSLVectorReplacements()
{
  std::unordered_map<std::string, std::string> m_vecReplacements;
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
  m_vecReplacements["_Bool"] = "bool";
  m_vecReplacements["unsigned int"]   = "uint";
  m_vecReplacements["unsigned char"]  = "uint8_t";
  m_vecReplacements["unsigned short"] = "uint16_t";
  m_vecReplacements["char"]           = "int8_t";
  m_vecReplacements["short"]          = "int16_t";
  m_vecReplacements["uchar"]          = "uint8_t";
  m_vecReplacements["ushort"]         = "uint16_t";
  return m_vecReplacements;
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

struct IRecursiveRewriteOverride
{
  virtual std::string RecursiveRewriteImpl(const clang::Stmt* expr) = 0;
};

/**
\brief process local functions
*/
class GLSLFunctionRewriter : public kslicer::FunctionRewriter // 
{
public:
  
  GLSLFunctionRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo) : FunctionRewriter(R,a_compiler,a_codeInfo)
  { 
    m_vecReplacements  = ListGLSLVectorReplacements();
    m_vecReplacements2 = SortByKeysByLen(m_vecReplacements);
   
    m_funReplacements["fmin"]  = "min";
    m_funReplacements["fmax"]  = "max";
    m_funReplacements["fminf"] = "min";
    m_funReplacements["fmaxf"] = "max";
    m_funReplacements["fsqrt"] = "sqrt";
    m_funReplacements["sqrtf"] = "sqrt";
    m_funReplacements["fabs"]  = "abs";
    m_funReplacements["to_float4"] = "vec4";
  }

  ~GLSLFunctionRewriter()
  {
  }

  bool VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl) override;
  bool VisitCallExpr_Impl(clang::CallExpr* f)             override;
  bool VisitVarDecl_Impl(clang::VarDecl* decl)            override;
  bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast) override;
  bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
  bool VisitMemberExpr_Impl(clang::MemberExpr* expr)         override;
  bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr)   override;
  bool VisitDeclStmt_Impl(clang::DeclStmt* decl)             override;

  std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override;
  IRecursiveRewriteOverride* m_pKernelRewriter = nullptr;

  std::string RewriteStdVectorTypeStr(const std::string& a_str) const override;
  std::string RewriteImageType(const std::string& a_containerType, const std::string& a_containerDataType, kslicer::TEX_ACCESS a_accessType, std::string& outImageFormat) const override;

  std::unordered_map<std::string, std::string> m_vecReplacements;
  std::unordered_map<std::string, std::string> m_funReplacements;
  std::vector<std::pair<std::string, std::string> > m_vecReplacements2;

  mutable kslicer::ShaderFeatures sFeatures;
  kslicer::ShaderFeatures GetShaderFeatures() const override { return sFeatures; }

protected:
  bool        NeedsVectorTypeRewrite(const std::string& a_str);
  std::string RewriteFuncDecl(clang::FunctionDecl* fDecl);
  std::string CompleteFunctionCallRewrite(clang::CallExpr* call);

  std::string RecursiveRewrite(const clang::Stmt* expr) override;
};


std::string GLSLFunctionRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  if(m_pKernelRewriter != nullptr) // we actually do kernel rewrite
    return m_pKernelRewriter->RecursiveRewriteImpl(expr);
  else
  {
    GLSLFunctionRewriter rvCopy = *this;
    rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

    std::string text = m_rewriter.getRewrittenText(expr->getSourceRange());
    if(text == "")                                                            // try to repair from the errors
      return kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); // which reason is unknown ... 
    else
      return text;
  }
}

std::string GLSLFunctionRewriter::RewriteStdVectorTypeStr(const std::string& a_str) const
{
  const bool isConst  = (a_str.find("const") != std::string::npos);
  const bool isUshort = (a_str.find("short") != std::string::npos)    || (a_str.find("ushort") != std::string::npos) || 
                        (a_str.find("uint16_t") != std::string::npos) || (a_str.find("int16_t") != std::string::npos);
  const bool isByte   = (a_str.find("char") != std::string::npos)    || (a_str.find("uchar") != std::string::npos) || 
                        (a_str.find("uint8_t") != std::string::npos) || (a_str.find("int8_t") != std::string::npos);
  const bool isInt64  = (a_str.find("long long int") != std::string::npos) ||
                        (a_str.find("uint64_t") != std::string::npos) || (a_str.find("int64_t") != std::string::npos);
 
  sFeatures.useByteType  = sFeatures.useByteType  || isByte;
  sFeatures.useShortType = sFeatures.useShortType || isUshort;
  sFeatures.useInt64Type = sFeatures.useInt64Type || isInt64;

  std::string resStr;
  std::string typeStr = a_str;
  ReplaceFirst(typeStr, "LiteMath::", "");
  ReplaceFirst(typeStr, "glm::",      "");
  ReplaceFirst(typeStr, "struct ",    "");
  ReplaceFirst(typeStr, "const ",     "");
  ReplaceFirst(typeStr, m_codeInfo->mainClassName + "::", "");
  
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

std::string GLSLFunctionRewriter::RewriteImageType(const std::string& a_containerType, const std::string& a_containerDataType, kslicer::TEX_ACCESS a_accessType, std::string& outImageFormat) const 
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

    if(dataTypeRewritten == "float" || dataTypeRewritten == "vec2" || dataTypeRewritten == "vec4")
      outImageFormat += "32f"; // TODO: 16f ???
    else if(dataTypeRewritten == "int" || dataTypeRewritten == "uint" || 
            dataTypeRewritten == "uvec2" || dataTypeRewritten == "ivec2" || 
            dataTypeRewritten == "uvec4" || dataTypeRewritten == "ivec4")
      outImageFormat += "32i"; // TODO: 16i ???
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

std::string GLSLFunctionRewriter::VectorTypeContructorReplace(const std::string& fname, const std::string& callText)
{
  return m_vecReplacements[fname] + callText;
}

bool GLSLFunctionRewriter::NeedsVectorTypeRewrite(const std::string& a_str) // TODO: make this implementation more smart, bad implementation actually!
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

std::string GLSLFunctionRewriter::RewriteFuncDecl(clang::FunctionDecl* fDecl)
{
  std::string retT   = RewriteStdVectorTypeStr(fDecl->getReturnType().getAsString()); 
  std::string fname  = fDecl->getNameInfo().getName().getAsString();
  std::string result = retT + " " + fname + "(";

  for(uint32_t i=0; i < fDecl->getNumParams(); i++)
  {
    const clang::ParmVarDecl* pParam  = fDecl->getParamDecl(i);
    const clang::QualType typeOfParam =	pParam->getType();
    std::string typeStr = typeOfParam.getAsString();
    if(typeOfParam->isPointerType())
    {
      ReplaceFirst(typeStr, "*", "");
      if(typeOfParam->getPointeeType().isConstQualified())
      {
        ReplaceFirst(typeStr, "const ", "");
        result += std::string("in ") + RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();
      }
      else
        result += std::string("inout ") + RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();
    }
    else
      result += RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();

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

bool GLSLFunctionRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)
{
  if(expr->isArrow() && WasNotRewrittenYet(expr->getBase()) )
  {
    const auto exprText     = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    const std::string lText = exprText.substr(exprText.find("->")+2);
    const std::string rText = RecursiveRewrite(expr->getBase());
    m_rewriter.ReplaceText(expr->getSourceRange(), rText + "." + lText);
    MarkRewritten(expr->getBase()); 
  }

  return true;
}

bool GLSLFunctionRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{
  const auto op = expr->getOpcodeStr(expr->getOpcode());
  if((op == "*" || op == "&") && WasNotRewrittenYet(expr->getSubExpr()) )
  {
    auto subExpr      = expr->getSubExpr();
    std::string text  = RecursiveRewrite(expr->getSubExpr());
    m_rewriter.ReplaceText(expr->getSourceRange(), text);
    MarkRewritten(expr->getSubExpr()); 
  }

  return true;
}

std::string GLSLFunctionRewriter::CompleteFunctionCallRewrite(clang::CallExpr* call)
{
  std::string rewrittenRes = "";
  for(int i=0;i<call->getNumArgs(); i++)
  {
    rewrittenRes += RecursiveRewrite(call->getArg(i));
    if(i!=call->getNumArgs()-1)
      rewrittenRes += ", ";
  }
  rewrittenRes += ")";
  return rewrittenRes;
}

//bool GLSLFunctionRewriter::VisitCXXConstructExpr_Impl(CXXConstructExpr* call)
//{
//
//}

bool GLSLFunctionRewriter::VisitCallExpr_Impl(clang::CallExpr* call)
{
  if(clang::isa<clang::CXXMemberCallExpr>(call) || clang::isa<clang::CXXConstructExpr>(call)) // process CXXMemberCallExpr else-where
    return true;

  clang::FunctionDecl* fDecl = call->getDirectCallee();
  if(fDecl == nullptr)
    return true;

  const std::string fname = fDecl->getNameInfo().getName().getAsString();
  std::string makeSmth = "";
  if(fname.substr(0, 5) == "make_")
    makeSmth = fname.substr(5);

  auto pFoundSmth = m_funReplacements.find(fname);

  if(fname == "to_float3" && call->getNumArgs() == 1 && WasNotRewrittenYet(call) )
  {
    const auto qt = call->getArg(0)->getType();
    const std::string typeName = qt.getAsString();
    if(typeName.find("float4") != std::string::npos)
    {
      const std::string exprText = RecursiveRewrite(call->getArg(0));
      
      if(clang::isa<clang::CXXConstructExpr>(call->getArg(0)))                                 // TODO: add other similar node types process here
        m_rewriter.ReplaceText(call->getSourceRange(), exprText + ".xyz");                     // to_float3(f4Data) ==> f4Data.xyz
      else
        m_rewriter.ReplaceText(call->getSourceRange(), std::string("(") + exprText + ").xyz"); // to_float3(a+b)    ==> (a+b).xyz
        
      MarkRewritten(call);
    }
  }
  else if(makeSmth != "" && call->getNumArgs() !=0 && WasNotRewrittenYet(call) )
  {
    std::string rewrittenRes = m_vecReplacements[makeSmth] + "(" + CompleteFunctionCallRewrite(call);
    m_rewriter.ReplaceText(call->getSourceRange(), rewrittenRes);
    MarkRewritten(call);
  }
  else if(fname == "mul4x4x4" && call->getNumArgs() == 2 && WasNotRewrittenYet(call))
  {
    const std::string A = RecursiveRewrite(call->getArg(0));
    const std::string B = RecursiveRewrite(call->getArg(1));
    m_rewriter.ReplaceText(call->getSourceRange(), "(" + A + "*" + B + ")");
    MarkRewritten(call);
  }
  else if(pFoundSmth != m_funReplacements.end() && WasNotRewrittenYet(call))
  {
    m_rewriter.ReplaceText(call->getSourceRange(), pFoundSmth->second + "(" + CompleteFunctionCallRewrite(call));
    MarkRewritten(call);
  }

  return true; 
}

bool GLSLFunctionRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl) // special case for process multiple decls in line, like 'int i,j,k=2'
{
  if(!decl->isSingleDecl())
  {
    const std::string debugText = kslicer::GetRangeSourceCode(decl->getSourceRange(), m_compiler); 
    std::string varType = "";
    std::string resExpr = "";
    for(auto it = decl->decl_begin(); it != decl->decl_end(); ++it)
    {
      clang::Decl* cdecl = (*it);
      if(!clang::isa<clang::VarDecl>(cdecl))
        continue;
      
      clang::VarDecl* vdecl = clang::dyn_cast<clang::VarDecl>(cdecl);
      const auto qt         = vdecl->getType();
      const auto pValue     = vdecl->getAnyInitializer();
      const std::string varName  = vdecl->getNameAsString();
      const std::string varValue = RecursiveRewrite(pValue);

      if(varType == "") // first element
      {
        varType = qt.getAsString();
        if(varType.find("unsigned char") != std::string::npos)
        {
          int a = 2;
        }
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
      m_rewriter.ReplaceText(decl->getSourceRange(), resExpr);
      MarkRewritten(decl);
    }
  }

  return true;
}

bool GLSLFunctionRewriter::VisitVarDecl_Impl(clang::VarDecl* decl) 
{
  if(clang::isa<clang::ParmVarDecl>(decl)) // process else-where (VisitFunctionDecl_Impl)
    return true;

  //bool isFirstInList = decl->isFirstDecl();

  const auto qt      = decl->getType();
  const auto pValue  = decl->getAnyInitializer();
      
  const std::string debugText = kslicer::GetRangeSourceCode(decl->getSourceRange(), m_compiler); 
  const std::string varType   = qt.getAsString();

  //if(varType.find("unsigned char") != std::string::npos)
  //{
  //  int a = 2;
  //}

  if(NeedsVectorTypeRewrite(varType) && WasNotRewrittenYet(pValue))
  {
    const std::string varName  = decl->getNameAsString();
    //if(varName == "red")
    //{
    //  int a = 2;
    //}
    const std::string varValue = RecursiveRewrite(pValue);
    const std::string varType2 = RewriteStdVectorTypeStr(varType);

    if(varValue == "" || varValue == varName) // 'float3 deviation;' for some reason !decl->hasInit() does not works 
      m_rewriter.ReplaceText(decl->getSourceRange(), varType2 + " " + varName);
    else
      m_rewriter.ReplaceText(decl->getSourceRange(), varType2 + " " + varName + " = " + varValue);
    MarkRewritten(pValue);
  }
  return true;
}

bool GLSLFunctionRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)
{
  clang::QualType qt   = cast->getTypeAsWritten();
  clang::Expr* 	  next = cast->getSubExpr();
  std::string typeCast = qt.getAsString();
  //const std::string debugText = kslicer::GetRangeSourceCode(cast->getSourceRange(), m_compiler); 
  if(WasNotRewrittenYet(next))
  {
    //if(typeCast.find("unsigned char") != std::string::npos)
    //{
    //  int a = 2;
    //}
    typeCast = RewriteStdVectorTypeStr(typeCast);
    const std::string exprText = RecursiveRewrite(next);
    if(exprText == ")") // strange bug for casts inside 'DeclStmt' 
      return true;
    m_rewriter.ReplaceText(cast->getSourceRange(), typeCast + "(" + exprText + ")");
    MarkRewritten(next);
  }

  return true;
}

bool GLSLFunctionRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast)
{
  //std::string dbgTxt = kslicer::GetRangeSourceCode(cast->getSourceRange(), m_compiler); 
  //auto kind          = cast->getCastKind();
  //
  ////https://code.woboq.org/llvm/clang/include/clang/AST/OperationKinds.def.html
  //if(kind != clang::CK_IntegralCast && kind != clang::CK_IntegralToFloating && kind != clang::CK_FloatingToIntegral) // in GLSL we don't have implicit casts
  //  return true;
  //
  //clang::Expr* next  = cast->getSubExpr(); 
  //clang::QualType qt = next->getType();
  //std::string castTo = qt.getAsString();
  //
  //if(WasNotRewrittenYet(next))
  //{
  //  const std::string exprText = RecursiveRewrite(next);
  //  m_rewriter.ReplaceText(next->getSourceRange(), castTo + "(" + exprText + ")");
  //  MarkRewritten(next);
  //}

  return true;
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


std::string kslicer::GLSLCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler)
{
  std::string typeInCL = a_decl.type;
  std::string result = "";  
  std::string nameWithoutStruct = typeInCL;
  ReplaceFirst(nameWithoutStruct, "struct ", "");
  switch(a_decl.kind)
  {
    case kslicer::DECL_IN_CLASS::DECL_STRUCT:
    result = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    ProcessVectorTypesString(result);
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    result = typeInCL + " " + a_decl.name + " = " + kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    result = "#define " + a_decl.name + " " + nameWithoutStruct;
    break;
    default:
    break;
  };
  return result;
}

std::shared_ptr<kslicer::FunctionRewriter> kslicer::GLSLCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo)
{
  return std::make_shared<GLSLFunctionRewriter>(R, a_compiler, a_codeInfo);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class GLSLKernelRewriter : public kslicer::KernelRewriter, IRecursiveRewriteOverride
{
public:
  
  GLSLKernelRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, const std::string& a_fakeOffsetExpr, const bool a_infoPass) : 
                     kslicer::KernelRewriter(R, a_compiler, a_codeInfo, a_kernel, a_fakeOffsetExpr, a_infoPass), m_glslRW(R, a_compiler, a_codeInfo)
  {
    m_glslRW.m_pKernelRewriter = this;
    auto userArgs = kslicer::GetUserKernelArgs(a_kernel.args);
    for(auto arg : userArgs)
      m_userArgs.insert(arg.name);
  }

  bool VisitCallExpr_Impl(clang::CallExpr* f) override;
  bool VisitVarDecl_Impl(clang::VarDecl* decl) override;
  bool VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) override;
  bool VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) override;
  bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f) override;
  bool VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) override;
  bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr) override;
  bool VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) override;
  
  std::string VectorTypeContructorReplace(const std::string& fname, const std::string& callText) override { return m_glslRW.VectorTypeContructorReplace(fname, callText); }

protected: 

  std::string RecursiveRewrite(const clang::Stmt* expr) override;
  std::string RecursiveRewriteImpl(const clang::Stmt* expr) override { return GLSLKernelRewriter::RecursiveRewrite(expr); }
  bool IsGLSL() const override { return true; }

  void RewriteTextureAccess(clang::CXXOperatorCallExpr* expr, clang::CXXOperatorCallExpr* a_assignOp);

  GLSLFunctionRewriter m_glslRW;
  void sync()
  {
    auto done = m_glslRW.GetProcessedNodes();
    m_rewrittenNodes.insert(done.begin(), done.end()); // make sure they contain same data
    m_glslRW.SetProcessedNodes(m_rewrittenNodes);      // make sure they contain same data
  }

  bool VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast) override;
  std::unordered_set<std::string> m_userArgs;
};

std::string GLSLKernelRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";

  if(!clang::isa<clang::DeclRefExpr>(expr))
  {
    GLSLKernelRewriter rvCopy = *this;
    rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
    return m_rewriter.getRewrittenText(expr->getSourceRange());
  }
  else
  {
    const auto pRef = clang::dyn_cast<clang::DeclRefExpr>(expr);
    const clang::ValueDecl* pDecl = pRef->getDecl();
    if(!clang::isa<clang::ParmVarDecl>(pDecl))
      return kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); 
  
    clang::QualType qt = pDecl->getType();
    if(qt->isPointerType() || qt->isReferenceType()) // we can't put references to push constants
      return  kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);

    return std::string("kgenArgs.") + kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
  }
}


bool GLSLKernelRewriter::VisitCallExpr_Impl(clang::CallExpr* f)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 

  m_glslRW.VisitCallExpr_Impl(f);
  sync();
  //kslicer::KernelRewriter::VisitCallExpr_Impl(f); // TODO: move same logic to m_glslRW
  return true;
}

bool GLSLKernelRewriter::VisitVarDecl_Impl(clang::VarDecl* decl)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 

  m_glslRW.VisitVarDecl_Impl(decl);
  sync();
  return true;
}

bool GLSLKernelRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 

  m_glslRW.VisitCStyleCastExpr_Impl(cast);
  sync();
  return true;
}

bool GLSLKernelRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{
  if(m_infoPass)
    return kslicer::KernelRewriter::VisitUnaryOperator_Impl(expr);
  
  const auto op = expr->getOpcodeStr(expr->getOpcode());
  if(op == "*")
  {
    auto subExpr           = expr->getSubExpr();
    std::string exprInside = RecursiveRewrite(subExpr);
    const bool needOffset  = CheckIfExprHasArgumentThatNeedFakeOffset(exprInside);
    if(needOffset)
      return kslicer::KernelRewriter::VisitUnaryOperator_Impl(expr);
    else
    {
      m_glslRW.VisitUnaryOperator_Impl(expr);
      sync();
    }
  } 
  else
  {
    m_glslRW.VisitUnaryOperator_Impl(expr);
    sync();
  }
  
  return true;
}


bool GLSLKernelRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 
  return kslicer::KernelRewriter::VisitCXXConstructExpr_Impl(call);
}

void GLSLKernelRewriter::RewriteTextureAccess(clang::CXXOperatorCallExpr* expr, clang::CXXOperatorCallExpr* a_assignOp)
{
  if(!WasNotRewrittenYet(expr))
    return;

  const clang::QualType leftType = expr->getArg(0)->getType(); 
  if(leftType->isPointerType()) // buffer ? --> ignore
    return;
  const std::string leftTypeName = leftType.getAsString();
  if(leftTypeName.find("Texture") == std::string::npos && leftTypeName.find("Image") == std::string::npos) // not an image ? --> ignore
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
  for(const auto& arg : m_currKernel.args)
  {
    if(arg.name == objName)
    {
      shouldRewrite = true;
      break;
    }
  }

  // (3) rewrite if do needed
  //
  if(shouldRewrite)
  {
    std::string indexText =  std::string("ivec2(") + RecursiveRewrite(expr->getArg(1)) + ")";
    if(a_assignOp != nullptr) // write 
    {
      std::string assignExprText = RecursiveRewrite(a_assignOp->getArg(1));
      m_rewriter.ReplaceText(a_assignOp->getSourceRange(), std::string("imageStore") + "(" + objName + ", " + indexText + ", " + assignExprText + ")");
      MarkRewritten(a_assignOp);
    }
    else                      // read
    {
      m_rewriter.ReplaceText(expr->getSourceRange(), std::string("imageLoad") + "(" + objName + ", " + indexText + ")");
      MarkRewritten(expr); 
    }
  }
}

bool GLSLKernelRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return kslicer::KernelRewriter::VisitCXXOperatorCallExpr_Impl(expr);
  
  std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler); 
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);   

  if(expr->isAssignmentOp()) // detect a_brightPixels[coord] = color;
  {
    clang::Expr* left = expr->getArg(0); 
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = kslicer::GetRangeSourceCode(clang::SourceRange(leftOp->getOperatorLoc()), m_compiler);  
      if(op2 == "]" || op2 == "[" || op2 == "[]")
        RewriteTextureAccess(leftOp, expr);
    }
  }
  else if(op == "]" || op == "[" || op == "[]")
     RewriteTextureAccess(expr, nullptr);

  sync();

  return kslicer::KernelRewriter::VisitCXXOperatorCallExpr_Impl(expr);
}

bool GLSLKernelRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* call)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return kslicer::KernelRewriter::VisitCXXMemberCallExpr_Impl(call);
 
  clang::CXXMethodDecl* fDecl = call->getMethodDecl();  
  if(fDecl != nullptr && WasNotRewrittenYet(call))  
  {
    //std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler); 
    std::string fname     = fDecl->getNameInfo().getName().getAsString();
    clang::Expr* pTexName =	call->getImplicitObjectArgument(); 
    std::string objName   = kslicer::GetRangeSourceCode(pTexName->getSourceRange(), m_compiler);     
  
    if(fname == "sample" || fname == "Sample")
    {
      bool needRewrite = true;
      const clang::QualType leftType = pTexName->getType(); 
      if(leftType->isPointerType()) // buffer ? --> ignore
        needRewrite = false;
      const std::string leftTypeName = leftType.getAsString();
      if(leftTypeName.find("Texture") == std::string::npos && leftTypeName.find("Image") == std::string::npos) // not an image ? --> ignore
        needRewrite = false;
      
      if(needRewrite)
      {
        //clang::Expr* samplerExpr = call->getArg(0); // TODO: process sampler? use separate sampler and image?
        //clang::Expr* txCoordExpr = call->getArg(1);
        //std::string text1 = kslicer::GetRangeSourceCode(samplerExpr->getSourceRange(), m_compiler); 
        //std::string text2 = kslicer::GetRangeSourceCode(txCoordExpr->getSourceRange(), m_compiler); 
        std::string texCoord = RecursiveRewrite(call->getArg(1));
        m_rewriter.ReplaceText(call->getSourceRange(), std::string("texture") + "(" + objName + ", " + texCoord + ")");
        MarkRewritten(call); 
        sync();
      }
    }
  }

  return kslicer::KernelRewriter::VisitCXXMemberCallExpr_Impl(call);
}

bool GLSLKernelRewriter::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 
  
  const clang::ValueDecl* pDecl = expr->getDecl();
  if(!clang::isa<clang::ParmVarDecl>(pDecl))
    return true; 

  clang::QualType qt = pDecl->getType();
  if(qt->isPointerType() || qt->isReferenceType()) // we can't put references to push constants
    return true;
  
  //m_userArgs.find(text) != m_userArgs.end() 
  if(WasNotRewrittenYet(expr))
  {
    std::string text = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); // 
    m_rewriter.ReplaceText(expr->getSourceRange(), std::string("kgenArgs.") + text);
    MarkRewritten(expr);
  }
  sync();

  return true;
} 

bool GLSLKernelRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast)
{
  if(m_infoPass)
    return true;
  
  if(cast->isPartOfExplicitCast())
    return true;

  auto kind = cast->getCastKind();

  clang::Expr* preNext = cast->getSubExpr(); 
  if(!clang::isa<clang::ImplicitCastExpr>(preNext))
    return true;
  
  clang::Expr* next = clang::dyn_cast<clang::ImplicitCastExpr>(preNext)->getSubExpr(); 
  //std::string dbgTxt = kslicer::GetRangeSourceCode(cast->getSourceRange(), m_compiler); 
  
  //https://code.woboq.org/llvm/clang/include/clang/AST/OperationKinds.def.html
  if(kind != clang::CK_IntegralCast && kind != clang::CK_IntegralToFloating && kind != clang::CK_FloatingToIntegral) // in GLSL we don't have implicit casts
    return true;
  
  clang::QualType qt = cast->getType();
  std::string castTo = m_glslRW.RewriteStdVectorTypeStr(qt.getAsString());
  
  if(WasNotRewrittenYet(next))
  {
    const std::string exprText = RecursiveRewrite(next);
    m_rewriter.ReplaceText(next->getSourceRange(), castTo + "(" + exprText + ")");
    MarkRewritten(next);
  }
  
  sync();
  return true;
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::GLSLCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, 
                                                                                 kslicer::KernelInfo& a_kernel, const std::string& fakeOffs, bool a_infoPass)
{
  return std::make_shared<GLSLKernelRewriter>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs, a_infoPass);
}

