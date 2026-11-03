#include "kslicer.h"
#include "template_rendering.h"
#include "clang/AST/RecordLayout.h"

#ifdef _WIN32
  #include <sys/types.h>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::umap<std::string, std::string> kslicer::ListSlangStandartTypeReplacements(bool a_NeedConstCopy)
{
  kslicer::umap<std::string, std::string> m_vecReplacements;
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
    kslicer::umap<std::string, std::string> m_vecReplacementsConst;
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
  m_funReplacements["f32tof16"]  = "f32tof16_kgen"; // see 'common_generated_slang.h': built-in functions use 'half' (Float16 and Int16 capabilities)
  m_funReplacements["f16tof32"]  = "f16tof32_kgen";
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
          if(m_codeInfo->shitIsAlwaysConst)
            bufferType = "StructuredBuffer";
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
  if(RewriteHashMapIteratorAccess(expr)) // 'it->second' ==> 'm_items[it].val'
    return true;

  if(expr->isArrow() && RewriteGlobalPtrAccess(expr)) // 'p->x' for pointer to buffer
    return true;

  if(m_kernelMode)
  {
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    std::string rewrittenText;
    if(NeedToRewriteMemberExpr(expr, rewrittenText))
    {
      ReplaceText(expr->getSourceRange(), rewrittenText);
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
    ReplaceText(expr->getSourceRange(), rText + "." + lText);
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

  if(RewriteHashMapMemberCall(call)) // 'm_items.find(key)' ==> 'hmap_m_items_find(key)'
    return true;

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
      ReplaceText(call->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
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
        ReplaceText(call->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
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
      ReplaceText(call->getSourceRange(), resulingText);
      MarkRewritten(call);
    }
    else if(fname == "data")
    {
      ReplaceText(call->getSourceRange(), memberNameA);
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
    ReplaceText(call->getSourceRange(), resCallText);
    MarkRewritten(call);
  }
  else if(fname == "set_row" && kslicer::IsMatrixTypeName(thisTypeName) && WasNotRewrittenYet(call))
  {
    clang::Expr* pMatName =	call->getImplicitObjectArgument();
    std::string  objName  = kslicer::GetRangeSourceCode(pMatName->getSourceRange(), m_compiler);
    std::string indexVal  = kslicer::GetRangeSourceCode(call->getArg(0)->getSourceRange(), m_compiler);
    std::string argValue  = RecursiveRewrite(call->getArg(1));
      
    std::string lastRewrittenText = objName + "[" + indexVal + "] = " + argValue;
    ReplaceText(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if(fname == "set_col" && kslicer::IsMatrixTypeName(thisTypeName) && WasNotRewrittenYet(call))
  {
    clang::Expr* pMatName =	call->getImplicitObjectArgument();
    std::string  objName  = kslicer::GetRangeSourceCode(pMatName->getSourceRange(), m_compiler);
    std::string indexVal  = kslicer::GetRangeSourceCode(call->getArg(0)->getSourceRange(), m_compiler);
    std::string argValue  = RecursiveRewrite(call->getArg(1));
      
    std::string baseType = thisTypeName.substr(0, 1);
    std::string number   = thisTypeName.substr(thisTypeName.size()-1, 1);
    std::string lastRewrittenText = "set_col" + number + baseType + "(" + objName + ", " + indexVal + ", " + argValue + ")";

    ReplaceText(call->getSourceRange(), lastRewrittenText);
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
      std::string baseName = "", indexName = "";
      pTexName = kslicer::RemoveImplicitCast(pTexName);
      if(clang::isa<clang::CXXOperatorCallExpr>(pTexName))
      {
        auto pArrorOp  = clang::dyn_cast<clang::CXXOperatorCallExpr>(pTexName);
        std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(pArrorOp->getOperatorLoc()), m_compiler); 
        clang::Expr* left  = kslicer::RemoveImplicitCast(pArrorOp->getArg(0));

        if(clang::isa<clang::CXXOperatorCallExpr>(left) && op == "->")
        {
          auto pArrayAccess = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
          std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(pArrayAccess->getOperatorLoc()), m_compiler); 
          if(op == "[]" || op == "[" || op == "]")
          {
            clang::Expr* left     = kslicer::RemoveImplicitCast(pArrayAccess->getArg(0));
            clang::Expr* right    = kslicer::RemoveImplicitCast(pArrayAccess->getArg(1));
            std::string textLeft  = kslicer::GetRangeSourceCode(left->getSourceRange(), m_compiler);
            std::string textRight = kslicer::GetRangeSourceCode(right->getSourceRange(), m_compiler);
            baseName  = textLeft;
            indexName = std::string("NonUniformResourceIndex(") + textRight + std::string(")");
          }
        }
      }
      
      if(baseName == "")
      {
        auto posBraceOpen  = objName.find_first_of("[");
        auto posBraceClose = objName.find_first_of("]");
        assert(posBraceOpen != std::string::npos);
        assert(posBraceClose != std::string::npos);
  
        baseName  = objName.substr(0, posBraceOpen);
        indexName = std::string("NonUniformResourceIndex(") + objName.substr(posBraceOpen+1, posBraceClose-posBraceOpen-1) + std::string(")");
      }

      const std::string samplerName   = baseName + "_sam" + "[" + indexName + "]";
      const std::string rewrittenText = baseName + "[" + indexName + "]" + ".SampleLevel(" + samplerName + ", " + texCoord + ", 0)";

      ReplaceText(call->getSourceRange(), rewrittenText);
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
    ReplaceText(call->getSourceRange(), textRes);
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

  if(fname == "InterlockedAdd3f")
    m_codeInfo->globalShaderFeatures.useInterlockedAdd3f = true; // may be called from non-member function which is not processed by KernelInfoVisitor

  // Inside a copy of function for global memory ('m_shit') its pointer argument may be passed to other function:
  // 'f(..., a_data, ...)' or 'f(..., a_data + k, ...)' ==> 'f(..., a_data, a_dataOffset, ...)' or 'f(..., a_data, a_dataOffset + k, ...)';
  // copy of 'f' is created by ExpandShittyFunctions.
  //
  if(!m_kernelMode && !m_shit.pointers.empty() && SLANG_SUPPORT_POINTER_ADD_IN_ARGS && WasNotRewrittenYet(call) && IsUserFunction(fDecl) &&
     !clang::isa<clang::CXXMemberCallExpr>(call) && !clang::isa<clang::CXXOperatorCallExpr>(call))
  {
    auto globalPointerOfArg = [&](unsigned i, std::vector<const clang::Expr*>* a_pAdds) -> const kslicer::ArgMatch* {
      if(i >= fDecl->getNumParams() || !fDecl->getParamDecl(i)->getType()->isPointerType())
        return nullptr;
      return FindShitPointer(call->getArg(i), a_pAdds);
    };

    bool hasGlobalPointer = false;
    for(unsigned i=0; i<call->getNumArgs() && !hasGlobalPointer; i++)
      hasGlobalPointer = (globalPointerOfArg(i, nullptr) != nullptr);

    if(hasGlobalPointer)
    {
      std::string rewrittenRes = fname + "(";
      for(unsigned i=0; i<call->getNumArgs(); i++)
      {
        std::vector<const clang::Expr*> adds;
        const kslicer::ArgMatch* pPointer = globalPointerOfArg(i, &adds);
        if(pPointer != nullptr)
          rewrittenRes += pPointer->formal + ", " + ShitPointerOffsetText(*pPointer, adds);
        else
          rewrittenRes += RecursiveRewrite(call->getArg(i));
        if(i != call->getNumArgs()-1)
          rewrittenRes += ", ";
      }
      rewrittenRes += ")";
      ReplaceText(call->getSourceRange(), rewrittenRes);
      MarkRewritten(call);
      return true;
    }
  }

  if(fname == "InterlockedAdd3f" && m_codeInfo->atomicFloatEmul && call->getNumArgs() == 3 && WasNotRewrittenYet(call))
  {
    // CAS is done on 'uint' view of the same buffer ('out_color_asUint'), declared at the same binding:
    // no pointer casts and no atomics on 'float' in SPIR-V
    //
    const std::string bufferName = RecursiveRewrite(call->getArg(0));
    m_codeInfo->uintAliasBuffers.insert(bufferName);
    const std::string text = "InterlockedAdd3f(" + bufferName + "_asUint, " + RecursiveRewrite(call->getArg(1)) + ", " + RecursiveRewrite(call->getArg(2)) + ")";
    ReplaceText(call->getSourceRange(), text);
    MarkRewritten(call);
    return true;
  }

  // LiteNN (apps/LiteMathAux/neural.h): 'LinearForward(w, matOffs, biasOffs, in, inDim, out, outDim)' ==> 'LiteNN_LinearForward<inDim, outDim>(w, matOffs, biasOffs, in, out)',
  // 'LinearForward(w, m_layers[i], in, inDim, out, outDim)' ==> 'LiteNN_LinearForward<inDim, outDim>(w, m_layers[i], in, out)';
  // implementation with cooperative vectors is in 'common_generated_slang.h', sizes of cooperative vectors must be compile time constants
  //
  if(fDecl->getQualifiedNameAsString().rfind("LiteNN::", 0) == 0 && WasNotRewrittenYet(call))
  {
    std::string text;
    const unsigned numArgs = call->getNumArgs();
    if((fname == "LinearForward" || fname == "LinearForward_ReLU") && (numArgs == 7 || numArgs == 6))
    {
      m_codeInfo->globalShaderFeatures.useCoopVec      = true; // may be called from non-member function which is not processed by KernelInfoVisitor
      m_codeInfo->globalShaderFeatures.useHalfType     = true;
      m_codeInfo->globalShaderFeatures.use16BitStorage = true;
      const unsigned inArg = numArgs - 4; // (..., in, inDim, out, outDim)
      std::string dims[2];
      for(int i = 0; i < 2; i++)
      {
        const clang::Expr* dimExpr = call->getArg(inArg + 1 + 2*i);
        auto dimValue = dimExpr->getIntegerConstantExpr(m_compiler.getASTContext());
        if(dimValue)
          dims[i] = std::to_string(dimValue->getExtValue());
        else
        {
          kslicer::PrintError(std::string("LiteNN::") + fname + ": layer size '" + GetRangeSourceCode(dimExpr->getSourceRange(), m_compiler) + "' must be a compile time constant",
                              dimExpr->getSourceRange(), m_compiler.getSourceManager());
          dims[i] = RecursiveRewrite(dimExpr);
        }
      }
      text = "LiteNN_" + fname + "<" + dims[0] + ", " + dims[1] + ">(";
      for(unsigned i = 0; i < inArg; i++)
        text += RecursiveRewrite(call->getArg(i)) + ", ";
      text += RecursiveRewrite(call->getArg(inArg)) + ", " + RecursiveRewrite(call->getArg(inArg + 2)) + ")";
    }
    else
      text = "LiteNN_" + fname + "(" + CompleteFunctionCallRewrite(call);
    ReplaceText(call->getSourceRange(), text);
    MarkRewritten(call);
    return true;
  }

  if((fname == "as_int32" || fname == "as_int") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceText(call->getSourceRange(), "asint(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_uint32" || fname == "as_uint") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceText(call->getSourceRange(), "asuint(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_ushort" || fname == "as_uint16") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceText(call->getSourceRange(), "bit_cast<uint16_t>(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_half" || fname == "as_float16") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceText(call->getSourceRange(), "bit_cast<half>(" + text + ")");
    MarkRewritten(call);
  }
  else if((fname == "as_float" || fname == "as_float32")  && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text  = RecursiveRewrite(call->getArg(0));
    const auto qtOfArg      = call->getArg(0)->getType();
    const std::string tname = kslicer::CleanTypeName(qtOfArg.getAsString());
    std::string lastRewrittenText = "asfloat(" + text + ")";
    ReplaceText(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if((fname == "all_of")  && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text  = RecursiveRewrite(call->getArg(0));
    const auto qtOfArg      = call->getArg(0)->getType();
    const std::string tname = kslicer::CleanTypeName(qtOfArg.getAsString());
    std::string lastRewrittenText = "all(" + text + ")";
    ReplaceText(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if((fname == "any_of")  && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text  = RecursiveRewrite(call->getArg(0));
    const auto qtOfArg      = call->getArg(0)->getType();
    const std::string tname = kslicer::CleanTypeName(qtOfArg.getAsString());
    std::string lastRewrittenText = "any(" + text + ")";
    ReplaceText(call->getSourceRange(), lastRewrittenText);
    MarkRewritten(call);
  }
  else if((fname == "inverse4x4") && call->getNumArgs() == 1 && WasNotRewrittenYet(call))
  {
    const std::string text = RecursiveRewrite(call->getArg(0));
    ReplaceText(call->getSourceRange(), "inverse(" + text + ")");
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
    std::string bufferText = argText0;
    if(suffix == "F" && m_codeInfo->atomicFloatEmul) // CAS on 'uint' view of the same buffer
    {
      m_codeInfo->uintAliasBuffers.insert(argText0);
      bufferText = argText0 + "_asUint";
    }
    const std::string rewrittenText = "ReduceAdd" + suffix + "(" + bufferText + ", uint(" + argText1 + "), " + argText2 + ", a_localTID.x)";

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
          m_codeInfo->globalShaderFeatures.useFloatAtomicAdd = !m_codeInfo->atomicFloatEmul;
        else if(typeName == "double")
          m_codeInfo->globalShaderFeatures.useDoubleAtomicAdd = true;
      }
    }

    ReplaceText(call->getSourceRange(), rewrittenText);
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
        size_t found = size_t(-1);
        for(size_t j=0;j<shittyPointers.size();j++)
        {
          if(shittyPointers[j].argId == i)
          {
            found = j;
            break;
          }
        }
        
        if(found != size_t(-1) && SLANG_SUPPORT_POINTER_ADD_IN_ARGS && IsUserFunction(fDecl)) // pointer to global memory ==> (buffer, offset); library functions get only buffer
        {
          auto match = kslicer::MatchGlobalPointer(call->getArg(i), *m_pCurrKernel, m_compiler);
          std::string bufferText = match.container;
          if(!match.valid || (match.localVar == nullptr && match.offsetExprs.empty())) // 'm_vec.data()' or kernel argument
            bufferText = RecursiveRewrite(arg);
          rewrittenRes += bufferText + ", " + (match.valid ? GlobalPtrOffsetText(match) : std::string("0"));
        }
        else
          rewrittenRes += RecursiveRewrite(arg);

        if(i!=call->getNumArgs()-1)
          rewrittenRes += ", ";
      }
      rewrittenRes += ")";
      //}
      
      ReplaceText(call->getSourceRange(), rewrittenRes);
      MarkRewritten(call);
    }
    else if ((m_pCurrKernel != nullptr) && (m_pCurrKernel->pattern == kslicer::PATTERN_TP::PATTERN_RTV) && rewriteDueToFakeOffset)
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

    const bool isAtomicAdd = (fname == "InterlockedAdd" || fname == "atomicAdd" || fname == "AtomicAdd");
    auto pFoundSmth = m_funReplacements.find(fname);
    if(isAtomicAdd && m_codeInfo->atomicFloatEmul && call->getNumArgs() >= 2 && call->getArg(1)->getType().getAsString() == "float" && WasNotRewrittenYet(call))
    {
      std::string bufferName, wordIndex;
      std::string lastRewrittenText;
      if(GetUintAliasWord(call->getArg(0), bufferName, wordIndex)) // CAS on 'uint' view of the same buffer
      {
        m_codeInfo->uintAliasBuffers.insert(bufferName);
        lastRewrittenText = "InterlockedAddEmul1fU(" + bufferName + "_asUint, " + wordIndex + ", " + RecursiveRewrite(call->getArg(1));
        if(call->getNumArgs() >= 3)
          lastRewrittenText += ", " + RecursiveRewrite(call->getArg(2));
        lastRewrittenText += ")";
      }
      else
      {
        kslicer::PrintWarning("'-atomicf_emul' supports only elements of buffers ('a_out[i]', 'm_vec[i].field'); hardware float atomics are used here (shaderBufferFloat32AtomicAdd is required)", call->getSourceRange(), m_compiler.getSourceManager());
        m_codeInfo->globalShaderFeatures.useFloatAtomicAdd = true;
        lastRewrittenText = "InterlockedAdd(" + CompleteFunctionCallRewrite(call);
      }
      ReplaceText(call->getSourceRange(), lastRewrittenText);
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

  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions

bool kslicer::SlangRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{ 
  if(expr->getOpcode() == clang::UO_Deref && RewriteGlobalPtrAccess(expr)) // '*p' for pointer to buffer
    return true;

  if(expr->isIncrementDecrementOp() && GetHashMapSubscript(expr->getSubExpr()) != nullptr) // 'm_hist[key]++' ==> 'InterlockedAdd(m_hist[...].val, uint(1))'
  {
    RewriteHashMapAtomicAdd(expr, expr->getSubExpr(), expr->isIncrementOp() ? "1" : "-1");
    return true;
  }

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
        ReplaceText(expr->getSourceRange(), leftStr2 + "Shared[" + localIdStr + "]++");
        MarkRewritten(expr);
      }
    }

    // detect " *something and &something"
    //
    const std::string exprInside = RecursiveRewrite(subExpr);

    if(op == "*" && !expr->canOverflow() && CheckIfExprHasArgumentThatNeedFakeOffset(exprInside) && WasNotRewrittenYet(expr)) 
    {
      if(m_codeInfo->megakernelRTV || m_fakeOffsetExp == "")
        ReplaceText(expr->getSourceRange(), exprInside);
      else
        ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
      MarkRewritten(expr);
    }
  }
  
  // remove "&" and "*" from all arguments and expressions
  //
  const auto op = expr->getOpcodeStr(expr->getOpcode());
  if(SLANG_ELIMINATE_LOCAL_POINTERS && (op == "*" || op == "&") && WasNotRewrittenYet(expr->getSubExpr()) )
  {
    std::string text = RecursiveRewrite(expr->getSubExpr());
    ReplaceText(expr->getSourceRange(), text);
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
  if(RewriteHashMapSubscript(expr)) // 'm_items[key]' ==> 'm_items[hmap_m_items_insert(key)].val'
    return true;
  if(expr->getNumArgs() == 2 && GetHashMapSubscript(expr->getArg(0)) != nullptr && WasNotRewrittenYet(expr))
  {
    const std::string op = clang::getOperatorSpelling(expr->getOperator());
    if(op != "=")
      kslicer::PrintError("operator '" + op + "' for std::unordered_map value of class type is not supported in GPU code", expr->getSourceRange(), m_compiler.getSourceManager());
  }

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
      ReplaceText(expr->getSourceRange(), text);
      MarkRewritten(expr);
    }
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitVarDecl_Impl(clang::VarDecl* decl) 
{ 
  if(clang::isa<clang::ParmVarDecl>(decl)) // process else-where (VisitFunctionDecl_Impl)
    return true;

  if(RewriteLocalGlobalPtrDecl(decl)) // 'const float* p = &m_vec[e];' ==> 'uint pOffset = uint(e);'
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
  const bool isAuto       = (typeClass == clang::Type::Auto);
  const bool isMapIterator = kslicer::IsHashMapIteratorType(qt); // std::unordered_map<...>::iterator ==> uint (slot index)
  if(pValue != nullptr && WasNotRewrittenYet(pValue) && (NeedsVectorTypeRewrite(varType) || isAuto || isMapIterator))
  {
    std::string varName  = decl->getNameAsString();
    std::string varNameOld = varName;
    std::string varValue = RecursiveRewrite(pValue);
    std::string varType2 = isMapIterator ? (qt.isConstQualified() ? "const uint" : "uint") : RewriteStdVectorTypeStr(varType, varName);
    
    std::string lastRewrittenText;
    if(varValue == "" || varNameOld == varValue) // 'float3 deviation;' for some reason !decl->hasInit() does not works
      lastRewrittenText = varType2 + " " + varName;
    else
      lastRewrittenText = varType2 + " " + varName + " = " + varValue;
  
    ReplaceText(decl->getSourceRange(), lastRewrittenText);
    MarkRewritten(pValue);
  }
  
  //if(wasSet) 
  //  this->ResetCurrFuncInfo();  

  return true; 
}

bool kslicer::SlangRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)
{
  // '(unsigned int)x' ==> '(uint)(x)', rewrite only type names which are absent in Slang
  //
  const std::string typeOrig = cast->getTypeAsWritten().getAsString();
  const std::string typeCast = RewriteStdVectorTypeStr(typeOrig);
  if(typeCast != typeOrig && WasNotRewrittenYet(cast))
  {
    const std::string exprText = RecursiveRewrite(cast->getSubExpr());
    ReplaceText(cast->getSourceRange(), "(" + typeCast + ")(" + exprText + ")");
    MarkRewritten(cast);
  }

  return true;
}

bool kslicer::SlangRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  clang::Expr* preNext = cast->getSubExpr();

  if(clang::isa<clang::CXXConstructExpr>(preNext))
  {
    auto call = clang::dyn_cast<clang::CXXConstructExpr>(preNext);
    clang::CXXConstructorDecl* ctorDecl = call->getConstructor();
    const std::string fname = ctorDecl->getNameInfo().getName().getAsString();
    
    if(kslicer::IsVectorContructorNeedsReplacement(fname) && WasNotRewrittenYet(call) && !ctorDecl->isCopyOrMoveConstructor() && call->getNumArgs() > 0 ) //
    {
      const std::string textRes = RewriteConstructCall(call);
      ReplaceText(call->getSourceRange(), textRes);
      MarkRewritten(call);
    }

    return true;
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl)             
{ 
  // variables declared before the loop of IPV kernel are placed to UBO: 'float x = val;' ==> 'ubo[0].Kernel_x = val;' 
  //
  if(m_kernelMode && m_pCurrKernel != nullptr && m_pCurrKernel->localVarsInUBO.size() != 0 && WasNotRewrittenYet(decl))
  {
    std::stringstream strOut;
    bool found = false;
    for(auto it = decl->decl_begin(); it != decl->decl_end(); ++it)
    {
      const clang::VarDecl* vdecl = clang::dyn_cast<clang::VarDecl>(*it);
      for(const auto& var : m_pCurrKernel->localVarsInUBO)
      {
        if(vdecl == nullptr || var.declLoc != vdecl->getLocation())
          continue;
        found = true;
        const clang::Expr* pInit = vdecl->getInit();
        if(pInit != nullptr)
          strOut << m_codeInfo->pShaderCC->UBOAccess(var.uboName) << " = " << RecursiveRewrite(pInit) << "; ";
      }
    }
    if(found)
    {
      std::string text = strOut.str();
      ReplaceText(decl->getSourceRange(), (text == "") ? std::string(";") : text);
      MarkRewritten(decl);
      return true;
    }
  }

  // multiple declarations in one statement, like 'uint32_t a = 0, b = 1;'. Source ranges of such VarDecls overlap 
  // (each of them begins from the type), so we can't rewrite them separately in VisitVarDecl and rewrite the whole statement here
  //
  if(decl->isSingleDecl() || !WasNotRewrittenYet(decl))
    return true;

  struct VarText { std::string type; std::string name; std::string value; };
  std::vector<VarText> vars;
  bool needRewrite = false;
  for(auto it = decl->decl_begin(); it != decl->decl_end(); ++it)
  {
    clang::VarDecl* vdecl = clang::dyn_cast<clang::VarDecl>(*it);
    if(vdecl == nullptr) // struct declaration or something else, don't touch it
      return true;
    
    const auto qt         = vdecl->getType();
    const auto pValue     = vdecl->getAnyInitializer();
    const auto varType    = qt.getAsString();
    const bool isAuto     = (qt->getTypeClass() == clang::Type::Auto);
    const bool isMapIter  = kslicer::IsHashMapIteratorType(qt);
    needRewrite = needRewrite || NeedsVectorTypeRewrite(varType) || isAuto || isMapIter;

    VarText var;
    var.name  = vdecl->getNameAsString();
    var.type  = isMapIter ? (qt.isConstQualified() ? "const uint" : "uint") : RewriteStdVectorTypeStr(varType, var.name);
    var.value = (pValue != nullptr) ? RecursiveRewrite(pValue) : "";
    if(var.value == var.name) // 'float3 deviation;' for some reason !decl->hasInit() does not works
      var.value = "";
    vars.push_back(var);
  }

  if(!needRewrite || vars.size() == 0)
    return true;

  bool sameTypes = true;
  for(const auto& var : vars)
    sameTypes = sameTypes && (var.type == vars[0].type);

  std::stringstream strOut;
  for(size_t i=0; i<vars.size(); i++)
  {
    if(i == 0 || !sameTypes)
      strOut << vars[i].type << " ";
    strOut << vars[i].name;
    if(vars[i].value != "")
      strOut << " = " << vars[i].value;
    if(i+1 != vars.size())
      strOut << (sameTypes ? ", " : "; ");
  }
  strOut << ";";

  ReplaceText(decl->getSourceRange(), strOut.str());
  MarkRewritten(decl);
  return true; 
}

bool kslicer::SlangRewriter::VisitReturnStmt_Impl(clang::ReturnStmt* ret)
{
  // RTV kernel which returns bool: 'return expr;' ==> '{ kgenExitCond = expr; break; }', thread flags are set in kernel epilog;
  // the kernel body is placed inside 'do { ... } while(false)' (see 'generated.slang'), so 'break' leaves it
  //
  const bool inKernelBody = m_kernelMode && m_pCurrKernel != nullptr && m_pCurrFuncInfo == nullptr && !processFuncMember;
  if(!inKernelBody || !m_pCurrKernel->isBoolTyped || m_codeInfo->megakernelRTV || ret->getRetValue() == nullptr || !WasNotRewrittenYet(ret))
    return true;

  if(kslicer::IsInsideLoopOrSwitch(ret, m_compiler.getASTContext()))
    kslicer::PrintError("'return' inside a loop or switch of RTV kernel which returns bool is not supported, assign result to a variable and return it after the loop", ret->getSourceRange(), m_compiler.getSourceManager());
  const std::string retExprText = RecursiveRewrite(ret->getRetValue());
  ReplaceText(ret->getSourceRange(), "{ kgenExitCond = " + retExprText + "; break; }");
  MarkRewritten(ret);
  return true;
}

bool kslicer::SlangRewriter::VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr) 
{
  clang::QualType type = expr->getType();

  const bool isDoubleLiteral = type->isRealFloatingType() && type->isSpecificBuiltinType(clang::BuiltinType::Double);

  if(isDoubleLiteral && WasNotRewrittenYet(expr))
  {
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    ReplaceText(expr->getSourceRange(), originalText + "l");
    MarkRewritten(expr);
  }

  return true;
}

bool kslicer::SlangRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) 
{ 
  if(RewriteGlobalPtrAccess(arrayExpr)) // local pointer to buffer in kernel ('p[i]') or pointer argument of function copy for global memory
    return true;

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
  if(GetHashMapSubscript(expr->getLHS()) != nullptr && WasNotRewrittenYet(expr)) // 'm_hist[key] += 1' ==> 'InterlockedAdd(m_hist[...].val, uint(1))'
  {
    const std::string op = std::string(expr->getOpcodeStr());
    if(op == "+=" || op == "-=")
    {
      const std::string rhsText = RecursiveRewrite(expr->getRHS());
      RewriteHashMapAtomicAdd(expr, expr->getLHS(), (op == "+=") ? rhsText : "-(" + rhsText + ")");
    }
    else
      kslicer::PrintError("operator '" + op + "' for std::unordered_map value is not supported in GPU code, only '+=' and '-=' (they become atomic)", expr->getSourceRange(), m_compiler.getSourceManager());
    return true;
  }

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
  if(CheckLocalGlobalPtrUse(expr))
    return true;
  if(m_kernelMode)
  { 
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    
    std::string rewrittenText;
    if(NeedToRewriteDeclRefExpr(expr,rewrittenText) && WasNotRewrittenYet(expr))
    {
      ReplaceText(expr->getSourceRange(), rewrittenText);
      MarkRewritten(expr);
    }
  }

  return true;
}

std::string kslicer::SlangRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  if(clang::isa<clang::CXXDefaultArgExpr>(expr)) // default argument of function call: its value
    return RecursiveRewrite(clang::dyn_cast<clang::CXXDefaultArgExpr>(expr)->getExpr());
  
  std::string shallow;
  if(DetectAndRewriteShallowPattern(expr, shallow)) 
    return shallow;
    
  SlangRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  return kslicer::GetRewrittenText(m_rewriter, expr->getSourceRange());
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
  if(a_typeName.find("LiteNN::Layer") != std::string::npos) // declared in 'common_generated_slang.h'
    return "LiteNN_Layer";
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

std::string kslicer::SlangCompiler::UBOAccess(const std::string& a_name) const 
{ 
  return std::string("ubo[0].") + a_name; 
};

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
std::string toSpirvString(int n);

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

  const std::filesystem::path templatePathMega   = templatesFolder / "generated_mega.slang"; // only for megakernels of RTV control functions ("-megakernel 1"); IPV kernels always use usual template
  const std::filesystem::path templatePathUsual  = templatesFolder / "generated.slang";
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

  // include folders are needed only if generated shaders include files from them ('Includes' in common header);
  // '-skip_shader_includes 0' always puts them to build script
  //
  const bool hasExternalIncludes = copy.contains("Includes") && copy["Includes"].is_array() && !copy["Includes"].empty();
  std::string includePaths = " ";
  if(!a_settings.skipShaderIncludes || hasExternalIncludes)
  {
    includePaths = " -I.. ";
    for(auto folder : ignoreFolders)
      includePaths += "-I" + folder.u8string() + " ";
    if(a_settings.skipShaderIncludes)
      std::cout << "  [kslicer]: shaders include files from external folders, so include paths are put to " << scriptName.c_str() << std::endl;
  }

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
    const bool vulkan11        = kernel.value()["UseSubGroups"];
    const bool vulkan12        = useRayTracingPipeline;

    std::string outFileName           = kernelName + ".slang";
    std::filesystem::path outFilePath = shaderPath / outFileName;
    const bool isMega = a_codeInfo->megakernelsByName.find(kernelName) != a_codeInfo->megakernelsByName.end(); // RTV control function joined to megakernel
    const std::filesystem::path templatePath = isMega ? templatePathMega : templatePathUsual;
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

    buildSH << "slangc " << outFileName.c_str() << targetString.c_str() << kernelName.c_str() << targetSuffix.c_str() << includePaths.c_str();
    buildSH << "-warnings-disable 39001 "; // several variables at the same binding on purpose: combined image samplers, '-atomicf_emul 2'
    if(a_settings.auxShaderCCOptions != "")
      buildSH << " " << a_settings.auxShaderCCOptions.c_str();
    
    
    auto spirv_ver = a_settings.spirv_ver;
    if(a_settings.vulkan_ver == 11)
      spirv_ver = 13;

    if(spirv_ver != 0)
    {
      const std::string ver = toSpirvString(spirv_ver);
      buildSH << " -profile " << ver.c_str() << " ";
    }
    buildSH << std::endl;
  
    if(kernel.value()["IsIndirect"])
    {
      outFileName = kernelName + "_UpdateIndirect.slang";
      outFilePath = shaderPath / outFileName;
      kslicer::ApplyJsonToTemplate(templatePathUpdInd.c_str(), outFilePath, currKerneJson);
      buildSH << "slangc " << outFileName.c_str() << targetString.c_str() << kernelName.c_str() << "_UpdateIndirect" << targetSuffix.c_str() << includePaths.c_str() << "-warnings-disable 39001 ";
      buildSH << std::endl;
    }

    if(kernel.value()["FinishRed"])
    {
      outFileName = kernelName + "_Reduction.slang";
      outFilePath = shaderPath / outFileName;
      kslicer::ApplyJsonToTemplate(templatePathRedFin.c_str(), outFilePath, currKerneJson);
      buildSH << "slangc " << outFileName.c_str() << targetString.c_str() << kernelName.c_str() << "_Reduction" << targetSuffix.c_str() << includePaths.c_str() << "-warnings-disable 39001 ";
      buildSH << std::endl;
    }
  }

  if(a_codeInfo->usedServiceCalls.find("memcpy") != a_codeInfo->usedServiceCalls.end())
  {
    nlohmann::json dummy;
    kslicer::ApplyJsonToTemplate(templatesFolder / "z_memcpy.slang", shaderPath / "z_memcpy.slang", dummy); // just file copy actually
    buildSH << "slangc z_memcpy.slang -o z_memcpy.comp.spv -warnings-disable 39001" << std::endl;
  }

  // scan and sort are implemented for Vulkan host only (not for WebGPU)
  //
  if(!m_wgpuEnabled && (a_codeInfo->usedServiceCalls.find("exclusive_scan") != a_codeInfo->usedServiceCalls.end() ||
                        a_codeInfo->usedServiceCalls.find("inclusive_scan") != a_codeInfo->usedServiceCalls.end()))
  {
    for(auto scanImpl : a_codeInfo->serviceCalls)
    {
      if (scanImpl.second.opName == "scan")
      {
        nlohmann::json params;
        params["Type"] = scanImpl.second.dataTypeName;

        const std::string name = "z_scan_" + scanImpl.second.dataTypeName;
        kslicer::ApplyJsonToTemplate(templatesFolder / "z_scan_block.slang",     shaderPath / (name + "_block.slang"),     params);
        kslicer::ApplyJsonToTemplate(templatesFolder / "z_scan_propagate.slang", shaderPath / (name + "_propagate.slang"), params);
        buildSH << "slangc " << name << "_block.slang     -o " << name << "_block.comp.spv -warnings-disable 39001"     << std::endl;
        buildSH << "slangc " << name << "_propagate.slang -o " << name << "_propagate.comp.spv -warnings-disable 39001" << std::endl;
      }
    }
  }

  if(!m_wgpuEnabled && a_codeInfo->usedServiceCalls.find("sort") != a_codeInfo->usedServiceCalls.end())
  {
    for(auto sortImpl : a_codeInfo->serviceCalls)
    {
      if (sortImpl.second.opName == "sort")
      {
        nlohmann::json params;
        params["Type"]   = sortImpl.second.dataTypeName;
        params["Lambda"] = sortImpl.second.lambdaSource;
        params["Suffix"] = ToLowerCase(m_suffix);

        const std::string name = "z_bitonic_" + sortImpl.second.dataTypeName;
        kslicer::ApplyJsonToTemplate(templatesFolder / "z_bitonic_pass.slang", shaderPath / (name + "_pass.slang"), params);
        buildSH << "slangc " << name << "_pass.slang -o " << name << "_pass.comp.spv -warnings-disable 39001" << std::endl;
        for(int blockSize : {512, 1024, 2048})
        {
          params["BlockSize"] = blockSize;
          const std::string blockName = name + "_" + std::to_string(blockSize);
          kslicer::ApplyJsonToTemplate(templatesFolder / "z_bitonic_block.slang", shaderPath / (blockName + ".slang"), params);
          buildSH << "slangc " << blockName << ".slang -o " << blockName << ".comp.spv -warnings-disable 39001" << std::endl;
        }
      }
    }
  }

  kslicer::WriteShaderBuildScript(shaderPath / scriptName, buildSH.str(), a_settings.mtShaderCompile);
}


bool kslicer::HasUserMethods(const clang::CXXRecordDecl* pRecord)
{
  if(pRecord == nullptr || !pRecord->hasDefinition())
    return false;
  for(const clang::CXXMethodDecl* method : pRecord->methods())
    if(!method->isImplicit())
      return true;
  return false;
}

// checks if member function modifies fields of 'this' (directly or via call of other such member function); Slang needs '[mutating]' only for them
//
class MutatingMethodChecker : public clang::RecursiveASTVisitor<MutatingMethodChecker>
{
public:
  MutatingMethodChecker(int a_depth) : m_depth(a_depth) {}
  bool modifiesThis = false;

  bool VisitBinaryOperator(clang::BinaryOperator* op)
  {
    if(op->isAssignmentOp() && RefersToThisField(op->getLHS())) // also catches CompoundAssignOperator
      modifiesThis = true;
    return !modifiesThis;
  }

  bool VisitUnaryOperator(clang::UnaryOperator* op)
  {
    if(op->isIncrementDecrementOp() && RefersToThisField(op->getSubExpr()))
      modifiesThis = true;
    return !modifiesThis;
  }

  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* call) // 'color += x' for LiteMath vector types
  {
    if(call->isAssignmentOp() || call->getOperator() == clang::OO_PlusPlus || call->getOperator() == clang::OO_MinusMinus)
      if(call->getNumArgs() > 0 && RefersToThisField(call->getArg(0)))
        modifiesThis = true;
    return !modifiesThis;
  }

  bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call)
  {
    const clang::CXXMethodDecl* method = call->getMethodDecl();
    if(method == nullptr || method->isConst() || method->isStatic() || !RefersToThisField(call->getImplicitObjectArgument()))
      return true;
    const clang::FunctionDecl* pDef = nullptr;
    if(method->hasBody(pDef) && m_depth < 8)
    {
      MutatingMethodChecker checker(m_depth+1);
      checker.TraverseStmt(pDef->getBody());
      modifiesThis = modifiesThis || checker.modifiesThis;
    }
    return !modifiesThis;
  }

private:
  int m_depth = 0;

  static bool RefersToThisField(const clang::Expr* a_expr)
  {
    const clang::Expr* expr = a_expr;
    while(expr != nullptr)
    {
      expr = expr->IgnoreParenImpCasts();
      if(llvm::isa<clang::CXXThisExpr>(expr))
        return true;
      else if(auto pMember = llvm::dyn_cast<clang::MemberExpr>(expr))
        expr = pMember->getBase();
      else if(auto pSubscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr))
        expr = pSubscript->getBase();
      else if(auto pOpCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr); pOpCall != nullptr && pOpCall->getOperator() == clang::OO_Subscript)
        expr = pOpCall->getArg(0);
      else if(auto pMemberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) // 'm_data.x()' like accessors returning reference
        expr = pMemberCall->getImplicitObjectArgument();
      else
        return false;
    }
    return false;
  }
};

// if declaration is followed by ';' (in-class declaration of out-of-line member function), extend the range to it
//
static clang::SourceRange ExtendRangeToSemicolon(const clang::SourceRange a_range, const clang::CompilerInstance& a_compiler)
{
  const clang::SourceManager& sm = a_compiler.getSourceManager();
  std::optional<clang::Token> next = clang::Lexer::findNextToken(a_range.getEnd(), sm, a_compiler.getLangOpts());
  if(next.has_value() && next->is(clang::tok::semi))
    return clang::SourceRange(a_range.getBegin(), next->getLocation());
  return a_range;
}

static bool IsMutatingMethod(const clang::FunctionDecl* a_func)
{
  MutatingMethodChecker checker(0);
  checker.TraverseStmt(a_func->getBody());
  return checker.modifiesThis;
}

std::string kslicer::SlangRewriter::RewriteStructWithMethods(const clang::CXXRecordDecl* pRecord)
{
  // Slang supports methods inside structures, but their code must be rewritten as for any other function:
  // (1) 'const' after parameter list is removed (Slang methods are immutable by default),
  // (2) non-const methods are marked with '[mutating]',
  // (3) constructors are converted to '__init(...)' with member initializers moved to the body,
  // (4) bodies are processed with the same rewriter as for ordinary functions.
  //
  clang::Rewriter rewrite2;
  rewrite2.setSourceMgr(m_compiler.getSourceManager(), m_compiler.getLangOpts());
  auto pVisitor = m_codeInfo->pShaderCC->MakeFuncRewriter(rewrite2, m_compiler, m_codeInfo);

  for(clang::CXXMethodDecl* method : pRecord->methods())
  {
    if(method->isImplicit())
      continue;

    const clang::FunctionDecl* pDefinition = nullptr;
    const bool hasBody = method->hasBody(pDefinition);
    clang::SourceRange declRange = method->getSourceRange();
    if(!method->doesThisDeclarationHaveABody())
      declRange = ExtendRangeToSemicolon(declRange, m_compiler); // in-class declaration is replaced with full definition

    if(llvm::isa<clang::CXXDestructorDecl>(method) || !hasBody)  // no destructors in Slang; declarations without definition are useless
    {
      kslicer::ReplaceTextSafe(rewrite2, declRange, "");
      continue;
    }

    clang::FunctionDecl* pFunc = const_cast<clang::FunctionDecl*>(pDefinition);
    auto pMethod = llvm::dyn_cast<clang::CXXMethodDecl>(pFunc);
    std::string declText = pVisitor->RewriteFuncDecl(pFunc);
    std::string bodyText = pVisitor->RecursiveRewrite(pFunc->getBody());

    std::string result;
    if(auto pCtor = llvm::dyn_cast<clang::CXXConstructorDecl>(pMethod))
    {
      if(pCtor->isCopyOrMoveConstructor()) // Slang copies structures by value
      {
        kslicer::ReplaceTextSafe(rewrite2, declRange, "");
        continue;
      }

      std::stringstream inits;
      for(const clang::CXXCtorInitializer* init : pCtor->inits())
      {
        if(!init->isWritten() || !init->isMemberInitializer())
          continue;
        const clang::Expr* pInit = init->getInit();
        if(auto pConstruct = llvm::dyn_cast<clang::CXXConstructExpr>(pInit))  // 'member(a_value)' is a copy constructor call, its range includes 'member'
        {
          if(pConstruct->getConstructor()->isCopyOrMoveConstructor() && pConstruct->getNumArgs() == 1)
            pInit = pConstruct->getArg(0);
        }
        inits << " " << init->getMember()->getNameAsString() << " = " << pVisitor->RecursiveRewrite(pInit) << ";";
      }

      const auto posArgs  = declText.find("(");
      const auto posBrace = bodyText.find("{");
      const std::string args = (posArgs != std::string::npos) ? declText.substr(posArgs) : std::string("()");
      if(posBrace != std::string::npos)
        bodyText = bodyText.substr(0, posBrace+1) + inits.str() + bodyText.substr(posBrace+1);
      result = "__init" + args + " " + bodyText;
    }
    else if(pMethod->isStatic())
      result = "static " + declText + " " + bodyText;
    else if(!pMethod->isConst() && IsMutatingMethod(pFunc))
      result = "[mutating] " + declText + " " + bodyText;
    else
      result = declText + " " + bodyText;

    kslicer::ReplaceTextSafe(rewrite2, declRange, result);
  }

  return kslicer::GetRewrittenText(rewrite2, pRecord->getSourceRange());
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
    {
      const clang::CXXRecordDecl* pRecord = (a_decl.astNode != nullptr) ? llvm::dyn_cast<clang::CXXRecordDecl>(a_decl.astNode) : nullptr;
      auto pSlangRewriter = std::dynamic_pointer_cast<kslicer::SlangRewriter>(a_pRewriter);
      if(pRecord != nullptr && pSlangRewriter != nullptr && kslicer::HasUserMethods(pRecord))
        result = pSlangRewriter->RewriteStructWithMethods(pRecord) + ";";
      else
        result = originalText + ";";
    }
    ProcessVectorTypesString(result);
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    //ReplaceFirst(typeInCL, "unsigned int", "uint");
    //ReplaceFirst(typeInCL, "unsigned", "uint");
    //ReplaceFirst(typeInCL, "_Bool", "bool");
    for(const auto& pair : kslicer::SortByKeysByLen(m_typesReplacement)) // longest first: "unsigned int" before "unsigned"
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
    for(const auto& pair : kslicer::SortByKeysByLen(m_typesReplacement)) // longest first: "unsigned int" before "unsigned"
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const clang::Expr* kslicer::GetMemberPathOffset(const clang::Expr* a_expr, const clang::ASTContext& a_astContext, uint64_t& a_offsetInBytes, bool& a_ok)
{
  a_offsetInBytes = 0;
  a_ok            = true;
  const clang::Expr* expr = a_expr->IgnoreParenImpCasts();
  while(auto memberExpr = clang::dyn_cast<clang::MemberExpr>(expr))  // 'x[i].vel.x', including anonymous structs and unions inside LiteMath types
  {
    auto fieldDecl = clang::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
    if(fieldDecl == nullptr || memberExpr->isArrow())
    {
      a_ok = false;
      return expr;
    }
    const clang::ASTRecordLayout& layout = a_astContext.getASTRecordLayout(fieldDecl->getParent());
    a_offsetInBytes += layout.getFieldOffset(fieldDecl->getFieldIndex()) / 8;
    expr = memberExpr->getBase()->IgnoreParenImpCasts();
  }
  return expr;
}

