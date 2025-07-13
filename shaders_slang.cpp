#include "kslicer.h"
#include "template_rendering.h"

#ifdef _WIN32
  #include <sys/types.h>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<std::string, std::string> kslicer::ListSlangStandartTypeReplacements(bool a_NeedConstCopy)
{
  std::unordered_map<std::string, std::string> m_vecReplacements;
  m_vecReplacements["_Bool"]              = "bool";
  m_vecReplacements["long int"]           = "int";
  m_vecReplacements["unsigned long"]      = "uint";
  m_vecReplacements["unsigned int"]       = "uint";
  m_vecReplacements["unsigned"]           = "uint";
  m_vecReplacements["unsigned char"]      = "uint8_t";
  m_vecReplacements["char"]               = "int8_t";
  m_vecReplacements["unsigned short"]     = "uint16_t";
  m_vecReplacements["short"]              = "int16_t";
  m_vecReplacements["uchar"]              = "uint8_t";
  m_vecReplacements["ushort"]             = "uint16_t";
  m_vecReplacements["int32_t"]            = "int";
  m_vecReplacements["uint32_t"]           = "uint";
  m_vecReplacements["size_t"]             = "uint64_t";
  m_vecReplacements["unsigned long long"] = "uint64_t";
  m_vecReplacements["long long int"]      = "int64_t";
  
  if(a_NeedConstCopy)
  {
    std::unordered_map<std::string, std::string> m_vecReplacementsConst;
    for(auto r : m_vecReplacements)
      m_vecReplacementsConst[std::string("const ") + r.first] = std::string("const ") + m_vecReplacements[r.first];
    
    for(auto rc : m_vecReplacementsConst)
      m_vecReplacements[rc.first] = rc.second;
  }
    
  return m_vecReplacements;
}

void kslicer::SlangRewriter::Init()
{
  m_typesReplacement = ListSlangStandartTypeReplacements(true);
  
  m_funReplacements.clear();
  m_funReplacements["atomicAdd"] = "InterlockedAdd";
  m_funReplacements["AtomicAdd"] = "InterlockedAdd";
}

std::string kslicer::SlangRewriter::RewriteStdVectorTypeStr(const std::string& a_str) const
{
  const bool isConst  = (a_str.find("const") != std::string::npos);
  std::string copy = a_str;
  ReplaceFirst(copy, "const ", "");
  while(ReplaceFirst(copy, " ", ""));

  auto sFeatures2 = kslicer::GetUsedShaderFeaturesFromTypeName(a_str);
  sFeatures = sFeatures || sFeatures2;

  std::string resStr;
  std::string typeStr = kslicer::CleanTypeName(a_str);

  if(typeStr.size() > 0 && typeStr[typeStr.size()-1] == ' ')
    typeStr = typeStr.substr(0, typeStr.size()-1);

  if(typeStr.size() > 0 && typeStr[0] == ' ')
    typeStr = typeStr.substr(1, typeStr.size()-1);

  auto p = m_typesReplacement.find(typeStr);
  if(p == m_typesReplacement.end())
    resStr = typeStr;
  else
    resStr = p->second;

  if(isConst)
    resStr = std::string("const ") + resStr;

  return resStr;
}

// process arrays: 'float[3] data' --> 'float data[3]' 
std::string kslicer::SlangRewriter::RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const
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

bool kslicer::SlangRewriter::NeedsVectorTypeRewrite(const std::string& a_str) // TODO: make this implementation more smart, bad implementation actually!
{
  std::string typeStr = kslicer::CleanTypeName(a_str);
  return (m_typesReplacement.find(typeStr) != m_typesReplacement.end());
}

std::string kslicer::SlangRewriter::RewriteFuncDecl(clang::FunctionDecl* fDecl)
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
  //if(shitHappends)
  //  result = retT + " " + m_shit.ShittyName() + "(";

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
      {
        ReplaceFirst(typeStr, "*", "");
        std::string bufferType = "RWStructuredBuffer";
        if(typeStr.find("const ") != std::string::npos)
        {
          //bufferType = "StructuredBuffer";
          ReplaceFirst(typeStr, "const ", "");
        }
        while(ReplaceFirst(typeStr, " ", ""));

        result += bufferType + "<" + typeStr + ">" + " " + pParam->getNameAsString();
        if(SLANG_SUPPORT_POINTER_ADD_IN_ARGS)
          result += std::string(",") + std::string("uint ") + pParam->getNameAsString() + "Offset"; //
      }
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::SlangRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)        
{
  if(clang::isa<clang::CXXMethodDecl>(fDecl)) // ignore methods here, process them inside VisitCXXMethodDecl_Impl
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

bool kslicer::SlangRewriter::VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)      
{ 
  return true; 
}

bool kslicer::SlangRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)             
{
  if(m_kernelMode)
  {
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    std::string rewrittenText;
    if(NeedToRewriteMemberExpr(expr, rewrittenText))
    {
      //ReplaceTextOrWorkAround(expr->getSourceRange(), rewrittenText);
      m_rewriter.ReplaceText(expr->getSourceRange(), rewrittenText);
      MarkRewritten(expr);
    }
  }
  
  // 'pStruct->member' ==> 'pStruct.member'
  //
  const auto exprText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
  if(expr->isArrow() && WasNotRewrittenYet(expr->getBase()) && exprText.find("->") != std::string::npos)
  {
    const std::string lText = exprText.substr(exprText.find("->")+2);
    const std::string rText = RecursiveRewrite(expr->getBase());
    ReplaceTextOrWorkAround(expr->getSourceRange(), rText + "." + lText);
    //m_rewriter.ReplaceText(expr->getSourceRange(), rText + "." + lText);
    MarkRewritten(expr->getBase());
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* call)  
{ 
   // Get name of function
  //
  const clang::DeclarationNameInfo dni = call->getMethodDecl()->getNameInfo();
  const clang::DeclarationName dn      = dni.getName();
        std::string fname              = dn.getAsString();

  std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler); 
  if(debugText.find("ReduceAdd") != std::string::npos)
  {
    int a = 2;
  }

  //if(kslicer::IsCalledWithArrowAndVirtual(call) && WasNotRewrittenYet(call))
  //{
  //  auto buffAndOffset = kslicer::GetVFHAccessNodes(call, m_compiler);
  //  if(buffAndOffset.buffName != "" && buffAndOffset.offsetName != "")
  //  {
  //    std::string buffText2  = buffAndOffset.buffName;
  //    std::string offsetText = buffAndOffset.offsetName; //GetRangeSourceCode(buffAndOffset.offsetNode->getSourceRange(), m_compiler); 
  //    
  //    std::string textCallNoName = "(" + offsetText; 
  //    if(call->getNumArgs() != 0)
  //      textCallNoName += ",";
  //      
  //    for(unsigned i=0;i<call->getNumArgs();i++)
  //    {
  //      const auto pParam                   = call->getArg(i);
  //      const clang::QualType typeOfParam   =	pParam->getType();
  //      const std::string typeNameRewritten = kslicer::CleanTypeName(typeOfParam.getAsString());
  //      if(m_codeInfo->dataClassNames.find(typeNameRewritten) != m_codeInfo->dataClassNames.end()) 
  //      {
  //        if(i==call->getNumArgs()-1)
  //          textCallNoName[textCallNoName.rfind(",")] = ' ';
  //        continue;
  //      }
  //
  //      textCallNoName += RecursiveRewrite(call->getArg(i));
  //      if(i < call->getNumArgs()-1)
  //        textCallNoName += ",";
  //    }
  //    
  //    auto pBuffNameFromVFH = m_codeInfo->m_vhierarchy.find(buffAndOffset.interfaceTypeName);
  //    if(pBuffNameFromVFH != m_codeInfo->m_vhierarchy.end())
  //      buffText2 = pBuffNameFromVFH->second.objBufferName;
  //
  //    std::string vcallFunc  = buffAndOffset.interfaceName + "_" + fname + "_" + buffText2 + textCallNoName + ")";
  //    ReplaceTextOrWorkAround(call->getSourceRange(), vcallFunc);
  //    MarkRewritten(call);
  //  }
  //}

  // Get name of "this" type; we should check wherther this member is std::vector<T>  
  //
  const clang::QualType qt = call->getObjectType();
  const auto& thisTypeName = qt.getAsString();
  clang::CXXRecordDecl* typeDecl  = call->getRecordDecl(); 
  const std::string cleanTypeName = kslicer::CleanTypeName(thisTypeName);
  
  const bool isVector   = (typeDecl != nullptr && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl)) && thisTypeName.find("vector<") != std::string::npos; 
  const bool isRTX      = (thisTypeName == "struct ISceneObject") && (fname.find("RayQuery_") != std::string::npos);
  const auto pPrefix    = m_codeInfo->composPrefix.find(cleanTypeName);
  const bool isPrefixed = (pPrefix != m_codeInfo->composPrefix.end());
  
  if(isVector && WasNotRewrittenYet(call))
  {
    const std::string exprContent = GetRangeSourceCode(call->getSourceRange(), m_compiler);
    const auto posOfPoint         = exprContent.find(".");
    std::string memberNameA       = exprContent.substr(0, posOfPoint);
    
    if(m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->hasPrefix)
      memberNameA = m_pCurrFuncInfo->prefixName + "_" + memberNameA;

    if(fname == "size" || fname == "capacity")
    {
      const std::string memberNameB = memberNameA + "_" + fname;
      ReplaceTextOrWorkAround(call->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
      MarkRewritten(call);
    }
    else if(fname == "resize")
    {
      if(m_pCurrKernel != nullptr && call->getSourceRange().getBegin() <= m_pCurrKernel->loopOutsidesInit.getEnd()) // TODO: SEEMS INCORECT LOGIC
      {
        assert(call->getNumArgs() == 1);
        const clang::Expr* currArgExpr  = call->getArgs()[0];
        std::string newSizeValue = RecursiveRewrite(currArgExpr); 
        std::string memberNameB  = memberNameA + "_size = " + newSizeValue;
        ReplaceTextOrWorkAround(call->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
        MarkRewritten(call);
      }
    }
    else if(fname == "push_back")
    {
      assert(call->getNumArgs() == 1);
      const clang::Expr* currArgExpr  = call->getArgs()[0];
      std::string newElemValue = RecursiveRewrite(currArgExpr);

      std::string memberNameB  = memberNameA + "_size";
      std::string resulingText = m_codeInfo->pShaderCC->RewritePushBack(memberNameA, memberNameB, newElemValue);
      ReplaceTextOrWorkAround(call->getSourceRange(), resulingText);
      MarkRewritten(call);
    }
    else if(fname == "data")
    {
      ReplaceTextOrWorkAround(call->getSourceRange(), memberNameA);
      MarkRewritten(call);
    }
    else 
    {
      kslicer::PrintError(std::string("Unsuppoted std::vector method") + fname, call->getSourceRange(), m_compiler.getSourceManager());
    }
  }
  else if((isRTX || isPrefixed) && WasNotRewrittenYet(call))
  {
    const auto exprContent = GetRangeSourceCode(call->getSourceRange(), m_compiler);
    const auto posOfPoint  = exprContent.find("->"); // seek for "m_pImpl->Func()" 
    
    std::string memberNameA;
    if(isPrefixed && posOfPoint == std::string::npos)   // Func() inside composed class of m_pImpl
      memberNameA = pPrefix->second;
    else
      memberNameA = exprContent.substr(0, posOfPoint);  // m_pImpl->Func() inside main class

    std::string resCallText = memberNameA + "_" + fname + "(";
    for(unsigned i=0;i<call->getNumArgs(); i++)
    {
      resCallText += RecursiveRewrite(call->getArg(i));
      //if(i == f->getNumArgs()-2 && lastArgIsEmpty)
      //  break;
      if(i!=call->getNumArgs()-1)
        resCallText += ", ";
    }
    resCallText += ")";
    ReplaceTextOrWorkAround(call->getSourceRange(), resCallText);
    MarkRewritten(call);
  }
  else if((fname == "sample" || fname == "Sample") && WasNotRewrittenYet(call))
  {
    //std::string debugText = kslicer::GetRangeSourceCode(f->getSourceRange(), m_compiler);
    clang::Expr* pTexName =	call->getImplicitObjectArgument();
    std::string objName   = kslicer::GetRangeSourceCode(pTexName->getSourceRange(), m_compiler);
    int texCoordId   = 0;    
    bool needRewrite = kslicer::NeedRewriteTextureArray(call, objName, texCoordId);
    if(needRewrite)
    {
      const std::string texCoord = RecursiveRewrite(call->getArg(texCoordId));
      //const std::string lastRewrittenText = std::string("textureLod") + "(" + objName + ", " + texCoord + ", 0)";
      auto posBrace = objName.find_first_of("[");
      assert(posBrace != std::string::npos);
      const std::string samplerName   = objName.substr(0, posBrace) + "_sam" + objName.substr(posBrace);
      const std::string rewrittenText = objName + ".SampleLevel(" + samplerName + ", " + texCoord + ", 0)";
      ReplaceTextOrWorkAround(call->getSourceRange(), rewrittenText);
      MarkRewritten(call);
    }
  }

  return true; 
} 

bool kslicer::SlangRewriter::VisitFieldDecl_Impl(clang::FieldDecl* decl)               { return true; }

std::string kslicer::SlangRewriter::VectorTypeContructorReplace(const std::string& fname, const std::string& callText)
{
  return fname + callText;
}

bool kslicer::SlangRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) 
{ 
  const std::string originalText = GetRangeSourceCode(call->getSourceRange(), m_compiler);
     
  clang::CXXConstructorDecl* ctorDecl = call->getConstructor();
  assert(ctorDecl != nullptr);
 
  if(WasNotRewrittenYet(call) && !ctorDecl->isCopyOrMoveConstructor() && call->getNumArgs() > 0) //
  {
    std::string varName, typeName;
    ExtractTypeAndVarNameFromConstructor(call, &m_compiler.getASTContext(), varName, typeName);
    const bool hasVarName = (varName != "") && (originalText.find(varName) == 0); // found in the start of the string

    std::string textRes = RewriteConstructCall(call);
    if(hasVarName)
      textRes = varName + " = " + textRes;
    ReplaceTextOrWorkAround(call->getSourceRange(), textRes); //
    //m_rewriter.ReplaceText(call->getSourceRange(), textRes);
    MarkRewritten(call);
  }

  return true; 
} 

bool kslicer::SlangRewriter::VisitCallExpr_Impl(clang::CallExpr* call)                    
{ 
  clang::FunctionDecl* fDecl = call->getDirectCallee();
  if(fDecl == nullptr)
    return true;
  
  const std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler);
  const std::string fname = fDecl->getNameInfo().getName().getAsString();

  if((fname == "as_int32" || fname == "as_int") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceTextOrWorkAround(call->getSourceRange(), "asint(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_uint32" || fname == "as_uint") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceTextOrWorkAround(call->getSourceRange(), "asuint(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_float" || fname == "as_float32")  && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text  = RecursiveRewrite(call->getArg(0));
    const auto qtOfArg      = call->getArg(0)->getType();
    const std::string tname = kslicer::CleanTypeName(qtOfArg.getAsString());
    std::string lastRewrittenText = "asfloat(" + text + ")";
    ReplaceTextOrWorkAround(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if(fname == "ReduceAdd" && call->getNumArgs() == 3 && WasNotRewrittenYet(call))
  {
    const std::string argText0 = RecursiveRewrite(call->getArg(0));
    const std::string argText1 = RecursiveRewrite(call->getArg(1));
    const std::string argText2 = RecursiveRewrite(call->getArg(2));
    auto posOfTypeBeg = debugText.find("<");
    auto posOfTypeEnd = debugText.find(",");
    std::string typeName = debugText.substr(posOfTypeBeg+1, posOfTypeEnd-posOfTypeBeg-1);
    ReplaceFirst(typeName, " ", "");
    std::string suffix = "F";
    if(typeName == "double")
      suffix = "D";
    else if(typeName == "uint" || typeName == "uint32_t" || typeName == "unsignedint")
      suffix = "U";
    else if(typeName == "int" || typeName == "int32_t")
      suffix = "I";
    const std::string rewrittenText = "ReduceAdd" + suffix + "(" + argText0 + ", uint(" + argText1 + "), " + argText2 + ", a_localTID.x)";

    if(m_pCurrKernel != nullptr)
    {
      auto found = m_pCurrKernel->templatedFunctionsLM.find("ReduceAdd" + suffix);
      if(found == m_pCurrKernel->templatedFunctionsLM.end())
      {
        TemplatedFunctionLM funInfo;
        funInfo.name         = "ReduceAdd" + suffix;
        funInfo.nameOriginal = "ReduceAdd";
        funInfo.types[0]     = typeName;
        m_pCurrKernel->templatedFunctionsLM[funInfo.name] = funInfo;
        if(typeName == "float")
          m_codeInfo->globalShaderFeatures.useFloatAtomicAdd = true;
        else if(typeName == "double")
          m_codeInfo->globalShaderFeatures.useDoubleAtomicAdd = true;
      }
    }

    ReplaceTextOrWorkAround(call->getSourceRange(), rewrittenText);
    MarkRewritten(call);
  }
  else if(m_kernelMode && WasNotRewrittenYet(call))
  {
    // (#1) check if buffer/pointer to global memory is passed to a function
    //
    std::vector<kslicer::ArgMatch> usedArgMatches = kslicer::MatchCallArgsForKernel(call, (*m_pCurrKernel), m_compiler);
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
  
      std::string rewrittenRes = func.originalName + "(";
      
      //if(func.originalName == "kernel_GetMaterialColor")
      //{
      //  rewrittenRes = func.originalName + "(" + "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-=)(" + ")";
      //}
      //else {
      for(unsigned i=0;i<call->getNumArgs(); i++)
      {
        const auto arg = kslicer::RemoveImplicitCast(call->getArg(i));
        rewrittenRes += RecursiveRewrite(arg);
        if(SLANG_SUPPORT_POINTER_ADD_IN_ARGS)
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
            //const std::string debugText = kslicer::GetRangeSourceCode(arg->getSourceRange(), m_compiler);
            if(clang::isa<clang::BinaryOperator>(arg))
            {
              const auto bo = clang::dyn_cast<clang::BinaryOperator>(arg);
              const clang::Expr *lhs = bo->getLHS();
              const clang::Expr *rhs = bo->getRHS();
              if(bo->getOpcodeStr() == "+")
                offset = RecursiveRewrite(rhs);
            }
            rewrittenRes += ", " + offset;
          }
        }

        if(i!=call->getNumArgs()-1)
          rewrittenRes += ", ";
      }
      rewrittenRes += ")";
      //}
      
      ReplaceTextOrWorkAround(call->getSourceRange(), rewrittenRes);
      //m_rewriter.ReplaceText(call->getSourceRange(), rewrittenRes);
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
      //m_rewriter.ReplaceText(call->getSourceRange(), rewrittenRes);
      MarkRewritten(call);
    }
  }
  
  if(!clang::isa<clang::CXXMemberCallExpr>(call) && !clang::isa<clang::CXXConstructExpr>(call)) // process CXXMemberCallExpr/CXXConstructExpr else-where
  {
    clang::FunctionDecl* fDecl = call->getDirectCallee();
    if(fDecl == nullptr)
      return true;
    
    const std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler);
    const std::string fname     = fDecl->getNameInfo().getName().getAsString();
  
    /////////////////////////////////////////////////////////////////////////
    //std::string makeSmth = "";
    //if(fname == "make_float3x3_by_columns") // mat3(a,b,c) == make_float3x3_by_columns(a,b,c)
    //  makeSmth = "float3x3";
    //else if(fname == "make_float3x3")       // don't change it!
    //  ;
    //else if(fname.substr(0, 5) == "make_")
    //  makeSmth = fname.substr(5);
    /////////////////////////////////////////////////////////////////////////

    auto pFoundSmth = m_funReplacements.find(fname);
    if(pFoundSmth != m_funReplacements.end() && WasNotRewrittenYet(call))
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

  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions

bool kslicer::SlangRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{ 
  if(m_kernelMode)
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
      
      const auto op = expr->getOpcodeStr(expr->getOpcode());
      std::string leftStr = kslicer::GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);
  
      auto p = m_pCurrKernel->usedMembers.find(leftStr);
      if(p != m_pCurrKernel->usedMembers.end() && WasNotRewrittenYet(expr))
      {
        KernelInfo::ReductionAccess access;
        access.type      = KernelInfo::REDUCTION_TYPE::UNKNOWN;
        access.rightExpr = "";
        access.leftExpr  = leftStr;
        access.dataType  = subExpr->getType().getAsString();
  
        if(op == "++")
          access.type    = KernelInfo::REDUCTION_TYPE::ADD_ONE;
        else if(op == "--")
          access.type    = KernelInfo::REDUCTION_TYPE::SUB_ONE;
        
        std::string leftStr2   = RecursiveRewrite(expr->getSubExpr()); 
        std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_pCurrKernel->GetDim(), m_pCurrKernel->wgSize);
        ReplaceTextOrWorkAround(expr->getSourceRange(), leftStr2 + "Shared[" + localIdStr + "]++");
        MarkRewritten(expr);
      }
    }

    // detect " *something and &something"
    //
    const std::string exprInside = RecursiveRewrite(subExpr);

    if(op == "*" && !expr->canOverflow() && CheckIfExprHasArgumentThatNeedFakeOffset(exprInside) && WasNotRewrittenYet(expr)) 
    {
      if(m_codeInfo->megakernelRTV || m_fakeOffsetExp == "")
        ReplaceTextOrWorkAround(expr->getSourceRange(), exprInside);
      else
        ReplaceTextOrWorkAround(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
      MarkRewritten(expr);
    }
  }
  
  // remove "&" and "*" from all arguments and expressions
  //
  const auto op = expr->getOpcodeStr(expr->getOpcode());
  if(SLANG_ELIMINATE_LOCAL_POINTERS && (op == "*" || op == "&") && WasNotRewrittenYet(expr->getSubExpr()) )
  {
    std::string text = RecursiveRewrite(expr->getSubExpr());
    ReplaceTextOrWorkAround(expr->getSourceRange(), text);
    //m_rewriter.ReplaceText(expr->getSourceRange(), text);
    MarkRewritten(expr->getSubExpr());
  }

  return true; 
}

static bool IsMatrixType(const std::string& a_typeName) // TODO: make it more 'soft'
{
  std::string typeName = a_typeName;
  ReplaceFirst(typeName, "LiteMath::", "");
  ReplaceFirst(typeName, "glm::", "");
  ReplaceFirst(typeName, "std::", "");
  if(typeName == "float2x2" || typeName == "double2x2")
    return true;
  else if(typeName == "float3x3" || typeName == "double3x3")
    return true;
  else if(typeName == "float4x4" || typeName == "double4x4")
    return true;
  else
    return false;
}

bool kslicer::SlangRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) 
{ 
  if(m_kernelMode)
  {
    FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(expr);
  }
  
  std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler); 
  if(op == "*" && expr->getNumArgs() == 2)
  {
    const clang::Expr* left  = expr->getArg(0);
    const clang::Expr* right = expr->getArg(1);

    const clang::QualType lhsType = left->getType();
    const clang::QualType rhsType = right->getType();
    
    const std::string lhsTypeStr = lhsType.getAsString();
    const std::string rhsTypeStr = rhsType.getAsString();
    
    if(WasNotRewrittenYet(expr) && IsMatrixType(lhsTypeStr) && IsMatrixType(rhsTypeStr))
    {
      const std::string leftText  = RecursiveRewrite(left);
      const std::string rightText = RecursiveRewrite(right);
      const std::string text      = "mul(" + leftText + "," + rightText + ")";
      ReplaceTextOrWorkAround(expr->getSourceRange(), text);
      //m_rewriter.ReplaceText(expr->getSourceRange(), text);
      MarkRewritten(expr);
    }
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitVarDecl_Impl(clang::VarDecl* decl) 
{ 
  if(clang::isa<clang::ParmVarDecl>(decl)) // process else-where (VisitFunctionDecl_Impl)
    return true;
  
  CkeckAndProcessForThreadLocalVarDecl(decl);

  const auto qt      = decl->getType();
  const auto pValue  = decl->getAnyInitializer();

  const std::string originalText = kslicer::GetRangeSourceCode(decl->getSourceRange(), m_compiler);
  const std::string varType      = qt.getAsString();

  if(m_kernelMode)
  {
    // ...
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
    //m_rewriter.ReplaceText(decl->getSourceRange(), lastRewrittenText);
    MarkRewritten(pValue);
  }
  
  //if(wasSet) 
  //  this->ResetCurrFuncInfo();  

  return true; 
}

bool kslicer::SlangRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)           
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true;   
}

bool kslicer::SlangRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl)             
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr) 
{
  clang::QualType type = expr->getType();

  const bool isDoubleLiteral = type->isRealFloatingType() && type->isSpecificBuiltinType(clang::BuiltinType::Double);

  if(isDoubleLiteral && WasNotRewrittenYet(expr))
  {
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    //ReplaceTextOrWorkAround(expr->getSourceRange(), originalText + "l");
    m_rewriter.ReplaceText(expr->getSourceRange(), originalText + "l");
    MarkRewritten(expr);
  }

  return true;
}

bool kslicer::SlangRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel

bool kslicer::SlangRewriter::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) 
{ 
  if(m_kernelMode)
  {
    return FunctionRewriter2::VisitCompoundAssignOperator_Impl(expr);
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  if(m_kernelMode)
  {
    return FunctionRewriter2::VisitBinaryOperator_Impl(expr);
  }

  return true;
}

bool  kslicer::SlangRewriter::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) 
{
  if(m_kernelMode)
  { 
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    if(originalText == "scale")
    {
      int a = 2;
    }
    
    std::string rewrittenText;
    if(NeedToRewriteDeclRefExpr(expr,rewrittenText) && WasNotRewrittenYet(expr))
    {
      //ReplaceTextOrWorkAround(expr->getSourceRange(), rewrittenText);
      m_rewriter.ReplaceText(expr->getSourceRange(), rewrittenText);
      MarkRewritten(expr);
    }
  }

  return true;
}

std::string kslicer::SlangRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  
  std::string shallow;
  if(DetectAndRewriteShallowPattern(expr, shallow)) 
    return shallow;
    
  SlangRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
    return m_rewriter.getRewrittenText(range);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::SlangCompiler::SlangCompiler(const std::string& a_prefix, bool a_wgpuEnabled) : m_suffix(a_prefix), m_wgpuEnabled(a_wgpuEnabled)
{
  m_typesReplacement = ListSlangStandartTypeReplacements(false);
}

void kslicer::SlangCompiler::ProcessVectorTypesString(std::string& a_str)
{
  static auto vecReplacements = kslicer::SortByKeysByLen(ListSlangStandartTypeReplacements(false));
  for(auto p : vecReplacements)
  {
    std::string strToSearch = p.first + " ";
    while(a_str.find(strToSearch) != std::string::npos) // replace all of them
      ReplaceFirst(a_str, p.first, p.second);
  }
}

std::string kslicer::SlangCompiler::LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const
{
  // uint3 a_localTID  : SV_GroupThreadID
  // uint3 a_globalTID : SV_DispatchThreadID
  if(a_kernelDim == 1)
    return "a_localTID.x";
  else if(a_kernelDim == 2)
  {
    std::stringstream strOut;
    strOut << "a_localTID.x + " << a_wgSize[0] << "*a_localTID.y";
    return strOut.str();
  }
  else if(a_kernelDim == 3)
  {
    std::stringstream strOut;
    strOut << "a_localTID.x + " << a_wgSize[0] << "*a_localTID.y + " << a_wgSize[0]*a_wgSize[1] << "*a_localTID.z";
    return strOut.str();
  }
  else
  {
    std::cout << "  [SlangCompiler::LocalIdExpr]: Error, bad kernelDim = " << a_kernelDim << std::endl;
    return "a_localTID.x";
  }
}

void kslicer::SlangCompiler::GetThreadSizeNames(std::string a_strs[3]) const
{
  a_strs[0] = "iNumElementsX";
  a_strs[1] = "iNumElementsY";
  a_strs[2] = "iNumElementsZ";
}

std::string kslicer::SlangCompiler::GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const
{
  std::string res = "WaveActiveSumUnknown"; 
  switch(a_access.type)
  {
    case KernelInfo::REDUCTION_TYPE::ADD_ONE:
    case KernelInfo::REDUCTION_TYPE::ADD:
    res = "WaveActiveSum";
    break;

    case KernelInfo::REDUCTION_TYPE::SUB:
    case KernelInfo::REDUCTION_TYPE::SUB_ONE:
    res = "WaveActiveSum";
    break;

    case KernelInfo::REDUCTION_TYPE::FUNC:
    {
      if(a_access.funcName == "min" || a_access.funcName == "std::min") res = "WaveActiveMin";
      if(a_access.funcName == "max" || a_access.funcName == "std::max") res = "WaveActiveMax";
    }
    break;

    case KernelInfo::REDUCTION_TYPE::MUL:
    res = "WaveActiveMul";
    break;

    default:
    break;
  };
  return res;
}

std::string kslicer::SlangCompiler::GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const
{
  std::string res = "InterlockedUnknown";
  switch(a_access.type)
  {
    case KernelInfo::REDUCTION_TYPE::ADD_ONE:
    case KernelInfo::REDUCTION_TYPE::ADD:
    res = "InterlockedAdd";
    break;

    case KernelInfo::REDUCTION_TYPE::SUB:
    case KernelInfo::REDUCTION_TYPE::SUB_ONE:
    res = "InterlockedAdd";
    break;

    case KernelInfo::REDUCTION_TYPE::FUNC:
    {
      if(a_access.funcName == "min" || a_access.funcName == "std::min") res = "InterlockedMin";
      if(a_access.funcName == "max" || a_access.funcName == "std::max") res = "InterlockedMax";
    }
    break;

    default:
    break;
  };
  return res;
}

std::string kslicer::SlangCompiler::ProcessBufferType(const std::string& a_typeName) const
{
  std::string type = kslicer::CleanTypeName(a_typeName);
  ReplaceFirst(type, "*", "");
  
  if(type[type.size()-1] == ' ')
    type = type.substr(0, type.size()-1);

  return type;
}

std::string kslicer::SlangCompiler::RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const
{
  return std::string("{ uint offset = 0; InterlockedAdd(") + UBOAccess(memberNameB) + ", 1, offset); " + memberNameA + "[offset] = " + newElemValue + ";}";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<kslicer::FunctionRewriter> kslicer::SlangCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                                                                    MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit)
{
  auto pFunc = std::make_shared<SlangRewriter>(R, a_compiler, a_codeInfo);
  pFunc->m_shit = a_shit;
  return pFunc;
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::SlangCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                                                                  MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, const std::string& fakeOffs)
{
  auto pFunc = std::make_shared<SlangRewriter>(R, a_compiler, a_codeInfo);
  pFunc->InitKernelData(a_kernel, fakeOffs);
  return std::make_shared<KernelRewriter2>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs, pFunc);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void kslicer::SlangCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  const auto& mainClassFileName = a_codeInfo->mainClassFileName;
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;

  #ifdef _WIN32
  const std::string scriptName = "build_slang.bat";
  #else
  const std::string scriptName = "build_slang.sh";
  #endif

  std::filesystem::path folderPath = mainClassFileName.parent_path();
  std::filesystem::path shaderPath = folderPath / this->ShaderFolder();
  std::filesystem::path incUBOPath = folderPath / "include";
  std::filesystem::create_directory(shaderPath);
  std::filesystem::create_directory(incUBOPath);

  // generate header for all used functions in GLSL code
  //
  std::string headerCommon = "common" + ToLowerCase(m_suffix) + "_slang.h";
  std::filesystem::path templatesFolder("templates_slang");
  kslicer::ApplyJsonToTemplate(templatesFolder / "common_generated_slang.h", shaderPath / headerCommon, a_kernelsJson);

  const std::filesystem::path templatePath       = templatesFolder / (a_codeInfo->megakernelRTV ? "generated_mega.slang" : "generated.slang");
  const std::filesystem::path templatePathUpdInd = templatesFolder / "update_indirect.slang";
  const std::filesystem::path templatePathRedFin = templatesFolder / "reduction_finish.slang";
  
  nlohmann::json copy, kernels, intersections;
  for (auto& el : a_kernelsJson.items())
  {
    //std::cout << el.key() << std::endl;
    if(std::string(el.key()) == "Kernels")
      kernels = a_kernelsJson[el.key()];
    else
      copy[el.key()] = a_kernelsJson[el.key()];
  }

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

    std::string outFileName           = kernelName + ".slang";
    std::filesystem::path outFilePath = shaderPath / outFileName;
    kslicer::ApplyJsonToTemplate(templatePath.c_str(), outFilePath, currKerneJson);
    
    std::string targetString;
    std::string targetSuffix;
    if(m_wgpuEnabled)
    {
      targetString = " -target wgsl -stage compute -entry main -o ";
      targetSuffix = ".wgsl";
    }
    else
    {
      targetString = " -o ";
      targetSuffix = ".comp.spv";
    }

    buildSH << "slangc " << outFileName.c_str() << targetString.c_str() << kernelName.c_str() << targetSuffix.c_str() << " -I.. ";
    for(auto folder : ignoreFolders)
      buildSH << "-I" << folder.c_str() << " ";
    buildSH << std::endl;
  
    if(kernel.value()["IsIndirect"])
    {
      outFileName = kernelName + "_UpdateIndirect.slang";
      outFilePath = shaderPath / outFileName;
      kslicer::ApplyJsonToTemplate(templatePathUpdInd.c_str(), outFilePath, currKerneJson);
      buildSH << "slangc " << outFileName.c_str() << targetString.c_str() << kernelName.c_str() << "_UpdateIndirect" << targetSuffix.c_str() << " -I.. ";
      for(auto folder : ignoreFolders)
       buildSH << "-I" << folder.c_str() << " ";
      buildSH << std::endl;
    }

    if(kernel.value()["FinishRed"])
    {
      outFileName = kernelName + "_Reduction.slang";
      outFilePath = shaderPath / outFileName;
      kslicer::ApplyJsonToTemplate(templatePathRedFin.c_str(), outFilePath, currKerneJson);
      buildSH << "slangc " << outFileName.c_str() << targetString.c_str() << kernelName.c_str() << "_Reduction" << targetSuffix.c_str() << " -I.. ";
      for(auto folder : ignoreFolders)
       buildSH << "-I" << folder.c_str() << " ";
      buildSH << std::endl;
    }
  }

  if(a_codeInfo->usedServiceCalls.find("memcpy") != a_codeInfo->usedServiceCalls.end())
  {
    nlohmann::json dummy;
    kslicer::ApplyJsonToTemplate(templatesFolder / "z_memcpy.slang", shaderPath / "z_memcpy.slang", dummy); // just file copy actually
    buildSH << "slangc z_memcpy.slang -o z_memcpy.comp.spv" << std::endl;
  }

}


std::string kslicer::SlangCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter)
{
  std::string typeInCL = a_decl.type;
  ReplaceFirst(typeInCL, "struct ", "");
  ReplaceFirst(typeInCL, "LiteMath::", "");

  std::string originalText = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler);

  std::string result = "";
  switch(a_decl.kind)
  {
    case kslicer::DECL_IN_CLASS::DECL_STRUCT:
    result = originalText + ";";
    ProcessVectorTypesString(result);
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    //ReplaceFirst(typeInCL, "unsigned int", "uint");
    //ReplaceFirst(typeInCL, "unsigned", "uint");
    //ReplaceFirst(typeInCL, "_Bool", "bool");
    for(const auto& pair : m_typesReplacement)
      ReplaceFirst(typeInCL, pair.first, pair.second);
    if(originalText == "")
      originalText = a_decl.lostValue;

    if(a_decl.isArray)
    {
      std::stringstream sizeStr;
      sizeStr << "[" << a_decl.arraySize << "]";
      result = "static const " + typeInCL + " " + a_decl.name + sizeStr.str() + " = " + originalText + ";";
    }
    else
      result = "static " + typeInCL + " " + a_decl.name + " = " + originalText + ";";
   
    ProcessVectorTypesString(result);
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    for(const auto& pair : m_typesReplacement)
      ReplaceFirst(typeInCL, pair.first, pair.second);
    result = "typealias " + a_decl.name + " = " + typeInCL + ";";
    break;
    default:
    break;
  };
  return result;
}

std::string kslicer::SlangCompiler::RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds)
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
