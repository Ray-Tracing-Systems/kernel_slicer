#include "extractor.h"
#include "ast_matchers.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclTemplate.h"

#include <queue>
#include <stack>

class FuncExtractor : public clang::RecursiveASTVisitor<FuncExtractor>
{
public:
  
  FuncExtractor(const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo& a_codeInfo) : m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_patternImpl(a_codeInfo)
  { 
    
  }

  kslicer::FuncData* pCurrProcessedFunc = nullptr; 

  bool VisitCallExpr(clang::CallExpr* call)
  {
    std::string debugText = kslicer::GetRangeSourceCode(call->getSourceRange(), m_compiler); 
    const clang::FunctionDecl* f = call->getDirectCallee();
    if(f == nullptr)
      return true;
    
    std::string debugName = f->getNameAsString();
    if(f->isOverloadedOperator())
      return true;
    
    //if(debugName.find("Shade") != std::string::npos)
    //{
    //  int a = 2;
    //}

    std::string fileName = std::string(m_sm.getFilename(f->getSourceRange().getBegin())); // check that we are in test_class.cpp or test_class.h or sms like that;                                                                             
    for(auto f : m_patternImpl.ignoreFolders)                                   // exclude everything from "shader" folders
    {
      if(fileName.find(f) != std::string::npos)
       return true;
    }

    if(fileName.find(".h") == std::string::npos && fileName.find(".cpp") == std::string::npos && fileName.find(".cxx") == std::string::npos)
      return true;
    
    kslicer::FuncData func;
    func.name = f->getNameAsString();

    const std::string fsrc = kslicer::GetRangeSourceCode(f->getSourceRange(), m_compiler);
    if(fsrc.find("{") == std::string::npos) // if don't have full source code in this node, just decl, need to obtain correct node
    {
      auto pNodeByDecl = m_patternImpl.allMemberFunctions.find(func.name);
      if(pNodeByDecl != m_patternImpl.allMemberFunctions.end())
        f = pNodeByDecl->second;
    }

    func.astNode  = f;
    func.srcRange = f->getSourceRange(); 
    func.srcHash  = kslicer::GetHashOfSourceRange(func.srcRange);
    func.isMember = clang::isa<clang::CXXMethodDecl>(func.astNode);
    func.isKernel = m_patternImpl.IsKernel(func.name);
    func.depthUse = 0;

     //pCurrProcessedFunc->calledMembers.insert(func.name);

    if(func.isKernel)
    {
      assert(pCurrProcessedFunc != nullptr);
      kslicer::PrintError(std::string("Calling kernel + '" + func.name + "' from a kernel is not allowed currently, ") + pCurrProcessedFunc->name, func.srcRange, m_sm);
      return true;
    }
    
    if(func.isMember && clang::isa<clang::CXXMemberCallExpr>(call)) // currently we support export for members of current class only
    {
      clang::CXXRecordDecl* recordDecl = clang::dyn_cast<clang::CXXMemberCallExpr>(call)->getRecordDecl();
      const auto pType    = recordDecl->getTypeForDecl();     
      const auto qt       = pType->getLocallyUnqualifiedSingleStepDesugaredType();
      const auto typeName = qt.getAsString();
      if(typeName != std::string("class ") + m_patternImpl.mainClassName && 
         typeName != std::string("struct ") + m_patternImpl.mainClassName && 
         typeName != m_patternImpl.mainClassName)
        return true;
    }

    usedFunctions[func.srcHash] = func;

    return true;
  }

  std::unordered_map<uint64_t, kslicer::FuncData> usedFunctions;

private:
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
  kslicer::MainClassInfo&        m_patternImpl;

};


std::vector<kslicer::FuncData> kslicer::ExtractUsedFunctions(MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  std::queue<FuncData> functionsToProcess; 

  for(const auto& k : a_codeInfo.kernels)
  {
    kslicer::FuncData func;
    func.name     = k.second.name;
    func.astNode  = k.second.astNode;
    func.srcRange = k.second.astNode->getSourceRange();
    func.srcHash  = GetHashOfSourceRange(func.srcRange);
    func.isMember = true; //isa<clang::CXXMethodDecl>(kernel.astNode);
    func.isKernel = true;
    func.depthUse = 0;
    functionsToProcess.push(func);
  }

  std::unordered_map<uint64_t, FuncData> usedFunctions;
  usedFunctions.reserve(functionsToProcess.size()*10);
  
  FuncExtractor visitor(a_compiler, a_codeInfo); // first traverse kernels to used functions, then repeat this recursivelly in a breadth first manner ... 
  while(!functionsToProcess.empty())
  {
    auto currFunc = functionsToProcess.front(); functionsToProcess.pop();
    
    //if(currFunc.name == "CookTorrance")
    //{
    //  std::string text = kslicer::GetRangeSourceCode(currFunc.astNode->getSourceRange(), a_compiler);
    //  std::cout << text.c_str() << std::endl;
    //  currFunc.astNode->dump();
    //}

    visitor.pCurrProcessedFunc = &currFunc;
    visitor.TraverseDecl(const_cast<clang::FunctionDecl*>(currFunc.astNode));

    for(auto& f : visitor.usedFunctions)
    {
      auto nextFunc     = f.second;
      nextFunc.depthUse = currFunc.depthUse + 1;
      functionsToProcess.push(nextFunc);
      if(f.second.isMember)
        currFunc.calledMembers.insert(f.second.name);      
    }

    if(!currFunc.isKernel)
      usedFunctions[currFunc.srcHash] = currFunc;
    
    for(auto foundCall : visitor.usedFunctions)
    {
      auto p = usedFunctions.find(foundCall.first);
      if(p == usedFunctions.end())
        usedFunctions[foundCall.first] = foundCall.second;
      else
        p->second.depthUse = std::max(p->second.depthUse, currFunc.depthUse + 1); // if we found func call at 2-nd and 3-td levels, take 3-rd one.
    }

    visitor.usedFunctions.clear();
  }

  std::vector<kslicer::FuncData> result; result.reserve(usedFunctions.size());
  for(const auto& f : usedFunctions)
    result.push_back(f.second);
  
  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.depthUse > b.depthUse; });

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DataExtractor : public clang::RecursiveASTVisitor<DataExtractor>
{
public:
  
  DataExtractor(const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo& a_codeInfo, 
                std::unordered_map<std::string, kslicer::DataMemberInfo>& a_members, std::unordered_map<std::string, kslicer::UsedContainerInfo>& a_auxContainers) : 
                m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_patternImpl(a_codeInfo), m_usedMembers(a_members), m_auxContainers(a_auxContainers)
  { 
    
  }
  
  bool VisitMemberExpr(clang::MemberExpr* expr) 
  {
    //const std::string debugText1 = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    if(false) // TODO: check if this is texture.sample, save texture name 
    {
      // clang::CXXMethodDecl* fDecl = call->getMethodDecl();  
      // if(fDecl == nullptr)  
      //   return;
      // 
      // //std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler); 
      // std::string fname     = fDecl->getNameInfo().getName().getAsString();
      // clang::Expr* pTexName =	call->getImplicitObjectArgument(); 
      // std::string objName   = GetRangeSourceCode(pTexName->getSourceRange(), m_compiler);     
      // 
      // if(fname == "sample" || fname == "Sample")
    }
    
    std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);

    std::string setter, containerName;
    if(kslicer::CheckSettersAccess(expr, &m_patternImpl, m_compiler, &setter, &containerName))
    {
      clang::QualType qt = expr->getType(); // 
      kslicer::UsedContainerInfo container;
      container.type     = qt.getAsString();
      container.name     = setter + "_" + containerName;            
      container.kind     = kslicer::GetKindOfType(qt);
      container.isConst  = qt.isConstQualified();
      container.isSetter = true;
      container.setterPrefix = setter;
      container.setterSuffix = containerName;
      m_auxContainers[container.name] = container;
      return true;
    }

    clang::ValueDecl* pValueDecl = expr->getMemberDecl();
    if(!clang::isa<clang::FieldDecl>(pValueDecl))
      return true;

    clang::FieldDecl* pFieldDecl  = clang::dyn_cast<clang::FieldDecl>(pValueDecl);
    clang::RecordDecl* pRecodDecl = pFieldDecl->getParent();

    const std::string thisTypeName = pRecodDecl->getNameAsString();
    if(thisTypeName != m_patternImpl.mainClassName)
      return true;

    // process access to arguments payload->xxx
    //
    clang::Expr* baseExpr = expr->getBase(); 
    assert(baseExpr != nullptr);

    auto baseName = kslicer::GetRangeSourceCode(baseExpr->getSourceRange(), m_compiler);
    auto member   = kslicer::ExtractMemberInfo(pFieldDecl, m_compiler.getASTContext());
    m_usedMembers[member.name] = member;

    return true;
  }

private:
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
  kslicer::MainClassInfo&        m_patternImpl;
  std::unordered_map<std::string, kslicer::DataMemberInfo>& m_usedMembers; 
  std::unordered_map<std::string, kslicer::UsedContainerInfo>& m_auxContainers;

};

std::unordered_map<std::string, kslicer::DataMemberInfo> kslicer::ExtractUsedMemberData(kslicer::KernelInfo* pKernel, const kslicer::FuncData& a_funcData, const std::vector<kslicer::FuncData>& a_otherMembers,
                                                                                        std::unordered_map<std::string, kslicer::UsedContainerInfo>& a_auxContainers, MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  std::unordered_map<std::string, kslicer::DataMemberInfo> result;
  std::unordered_map<std::string, kslicer::FuncData>       allMembers;

  for(auto f : a_otherMembers)
    allMembers[f.name] = f;
  
  //process a_funcData.astNode to get all accesed data, then recursivelly process calledMembers
  //
  std::queue<FuncData> functionsToProcess; 
  functionsToProcess.push(a_funcData);

  DataExtractor visitor(a_compiler, a_codeInfo, result, a_auxContainers);

  while(!functionsToProcess.empty())
  {
    auto currFunc = functionsToProcess.front(); functionsToProcess.pop();
    
    // (1) process curr function to get all accesed data
    //
    visitor.TraverseDecl(const_cast<clang::FunctionDecl*>(currFunc.astNode));

    // (2) then recursivelly process calledMembers
    //
    for(auto nextFuncName : currFunc.calledMembers)
    {
      auto pNext = allMembers.find(nextFuncName);
      if(pNext != allMembers.end())
        functionsToProcess.push(pNext->second);
    }
  }
  
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<kslicer::ArgMatch> kslicer::MatchCallArgsForKernel(clang::CallExpr* call, const KernelInfo& k, const clang::CompilerInstance& a_compiler)
{
  std::vector<kslicer::ArgMatch> result;
  const clang::FunctionDecl* fDecl = call->getDirectCallee();  
  if(fDecl == nullptr || clang::isa<clang::CXXOperatorCallExpr>(call)) 
    return result;
  
  std::string debugText = kslicer::GetRangeSourceCode(call->getSourceRange(), a_compiler);

  for(size_t i=0;i<call->getNumArgs();i++)
  {
    const clang::ParmVarDecl* formalArg = fDecl->getParamDecl(i);
    const clang::Expr*        actualArg = call->getArg(i);
    const clang::QualType     qtFormal  = formalArg->getType();
    //const clang::QualType     qtActual  = actualArg->getType();
    std::string formalTypeName          = qtFormal.getAsString();

    std::string formalName = formalArg->getNameAsString();
    std::string actualText = kslicer::GetRangeSourceCode(actualArg->getSourceRange(), a_compiler);
    if(actualText.find(".data()") != std::string::npos) 
      actualText = actualText.substr(0, actualText.find(".data()"));
    
    for(const auto& argOfCurrKernel : k.args)
    {
      if(argOfCurrKernel.name == actualText)
      {
        ArgMatch arg;
        arg.formal    = formalName;
        arg.actual    = actualText;
        arg.type      = formalTypeName;
        arg.argId     = i;
        arg.isPointer = (qtFormal->isPointerType());
        result.push_back(arg);
      }
    }
    
    for(const auto& container : k.usedContainers)
    {
      if(container.second.name == actualText)
      {
        ArgMatch arg;
        arg.formal    = formalName;
        arg.actual    = actualText;
        arg.type      = formalTypeName;
        arg.argId     = i;
        arg.isPointer = (qtFormal->isPointerType());
        result.push_back(arg);
      }
    }
  }
  
  return result;
}


class ArgMatcher : public clang::RecursiveASTVisitor<ArgMatcher>
{
public:
  
  ArgMatcher(const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo& a_codeInfo, std::vector< std::unordered_map<std::string, std::string> >& a_match, std::string a_funcName) : 
             m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_patternImpl(a_codeInfo), m_argMatch(a_match), m_currFuncName(a_funcName)
  { 
        
  }

  bool VisitCallExpr(clang::CallExpr* call)
  {
    const clang::FunctionDecl* fDecl = call->getDirectCallee();  
    if(fDecl == nullptr)             // definitely can't process nullpointer 
      return true;
    
    std::string fname = fDecl->getNameInfo().getName().getAsString();
    if(fname != m_currFuncName)
      return true;
    
    std::unordered_map<std::string, std::string> agrMap;
    for(size_t i=0;i<call->getNumArgs();i++)
    {
      const clang::ParmVarDecl* formalArg = fDecl->getParamDecl(i);
      const clang::Expr*        actualArg = call->getArg(i);
      const clang::QualType     qtFormal  = formalArg->getType();
      //const clang::QualType     qtActual  = actualArg->getType();
      std::string formalTypeName          = qtFormal.getAsString();

      bool supportedContainerType = (formalTypeName.find("Texture") != std::string::npos) || 
                                    (formalTypeName.find("Image") != std::string::npos)   || 
                                    (formalTypeName.find("vector") != std::string::npos)  || 
                                    (formalTypeName.find("Sampler") != std::string::npos) || 
                                    (formalTypeName.find("sampler") != std::string::npos);

      if(qtFormal->isPointerType() || (qtFormal->isReferenceType() && supportedContainerType))
      {
        std::string formalName = formalArg->getNameAsString();
        std::string actualText = kslicer::GetRangeSourceCode(actualArg->getSourceRange(), m_compiler);
        if(actualText.find(".data()") != std::string::npos) 
          actualText = actualText.substr(0, actualText.find(".data()"));
        //TODO: for pointers we should support pointrer arithmatic here ... or not?
        agrMap[formalName] = actualText;
      }
    }
    
    if(!agrMap.empty())
      m_argMatch.push_back(agrMap);

    return true;
  }

private:
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
  kslicer::MainClassInfo&        m_patternImpl;
  std::vector< std::unordered_map<std::string, std::string> >& m_argMatch; 
  std::string                    m_currFuncName;
};


std::vector< std::unordered_map<std::string, std::string> > kslicer::ArgMatchTraversal(kslicer::KernelInfo* pKernel, const kslicer::FuncData& a_funcData, const std::vector<kslicer::FuncData>& a_otherMambers,
                                                                                       MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  std::vector< std::unordered_map<std::string, std::string> > result;
  ArgMatcher visitor(a_compiler, a_codeInfo, result, a_funcData.name);
  visitor.TraverseDecl(const_cast<clang::CXXMethodDecl*>(pKernel->astNode));
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DeclExtractor : public clang::RecursiveASTVisitor<DeclExtractor>
{
public:
  
  DeclExtractor(const clang::CompilerInstance& a_compiler, const std::vector<kslicer::DeclInClass>& a_listedNames) : m_compiler(a_compiler), m_sm(a_compiler.getSourceManager())
  { 
    for(const auto& decl : a_listedNames)
      usedDecls[decl.name] = decl;
  }

  std::unordered_map<std::string, kslicer::DeclInClass> usedDecls;

  bool VisitCXXRecordDecl(clang::CXXRecordDecl* record)
  {
    const auto pType = record->getTypeForDecl(); 
    const auto qt    = pType->getLocallyUnqualifiedSingleStepDesugaredType();
    const std::string typeName = qt.getAsString();

    const auto ddPos = typeName.find("::");
    if(ddPos == std::string::npos)
      return true;

    const std::string key = typeName.substr(ddPos+2);
    auto p = usedDecls.find(key);
    if(p != usedDecls.end())
    {
      p->second.srcRange  = record->getSourceRange();
      p->second.srcHash   = kslicer::GetHashOfSourceRange( p->second.srcRange);
      p->second.extracted = true;
    }

    return true;
  } 

  bool VisitVarDecl(clang::VarDecl* var)
  {
    if(var->isImplicit() || !var->isConstexpr())
      return true;

    const clang::QualType qt = var->getType();
    const auto typePtr = qt.getTypePtr(); 
    if(typePtr->isPointerType())
      return true;
    
    auto p = usedDecls.find(var->getNameAsString());
    if(p != usedDecls.end())
    {
      const clang::Expr* initExpr = var->getAnyInitializer();
      if(initExpr != nullptr)
      {
        p->second.srcRange  = initExpr->getSourceRange();
        p->second.srcHash   = kslicer::GetHashOfSourceRange(p->second.srcRange);
        p->second.extracted = true;
      }
    }

    return true;
  }

  bool VisitTypedefDecl(clang::TypedefDecl* tDecl)
  { 
    auto p = usedDecls.find(tDecl->getNameAsString());
    if(p != usedDecls.end())
    {
      //const std::string typeName = qt2.getAsString(); 
      //const auto qt2 = tDecl->getUnderlyingType();
      //const std::string typeName1 = tDecl->getNameAsString();
      p->second.srcRange  = tDecl->getSourceRange();
      p->second.srcHash   = kslicer::GetHashOfSourceRange(p->second.srcRange);
      p->second.extracted = true;
    }
    return true;
  }

private:
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
};


std::vector<kslicer::DeclInClass> kslicer::ExtractUsedTC(const std::vector<kslicer::DeclInClass>& a_listedNames, const clang::CXXRecordDecl* classAstNode, const clang::CompilerInstance& a_compiler)
{
  DeclExtractor visitor(a_compiler, a_listedNames);
  visitor.TraverseDecl(const_cast<clang::CXXRecordDecl*>(classAstNode));

  std::vector<kslicer::DeclInClass> result(a_listedNames.size());
  for(const auto& decl : visitor.usedDecls)
    result[decl.second.order] = decl.second;

  return result;
}

const char* GetClangToolingErrorCodeMessage(int code);

std::vector<kslicer::DeclInClass> kslicer::ExtractTCFromClass(const std::string& a_className, const clang::CXXRecordDecl* classAstNode,
                                                              const clang::CompilerInstance& compiler, clang::tooling::ClangTool& Tool)
{
  auto structMatcher = kslicer::MakeMatch_StructDeclInsideClass(a_className);
  auto varMatcher    = kslicer::MakeMatch_VarDeclInsideClass(a_className);
  auto tpdefMatcher  = kslicer::MakeMatch_TypedefInsideClass(a_className);
    
  clang::ast_matchers::MatchFinder finder;
  kslicer::TC_Extractor typeAndConstantsHandler(compiler);
  finder.addMatcher(clang::ast_matchers::traverse(clang::TK_IgnoreUnlessSpelledInSource, structMatcher), &typeAndConstantsHandler);
  finder.addMatcher(clang::ast_matchers::traverse(clang::TK_IgnoreUnlessSpelledInSource, varMatcher),    &typeAndConstantsHandler);
  finder.addMatcher(clang::ast_matchers::traverse(clang::TK_IgnoreUnlessSpelledInSource, tpdefMatcher),  &typeAndConstantsHandler);

  auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
  std::cout << "  [TC_Extractor]: end process constants and structs:\t" << GetClangToolingErrorCodeMessage(res) << std::endl;
  
  std::vector<kslicer::DeclInClass> usedDecls;
  usedDecls.reserve(typeAndConstantsHandler.foundDecl.size());
  for(const auto decl : typeAndConstantsHandler.foundDecl)
    usedDecls.push_back(decl.second);
  
  std::sort(usedDecls.begin(), usedDecls.end(), [](const auto& a, const auto& b) { return a.order < b.order; } );
  return kslicer::ExtractUsedTC(usedDecls, classAstNode, compiler);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<const kslicer::KernelInfo*> kslicer::extractUsedKernelsByName(const std::unordered_set<std::string>& a_usedNames, const std::unordered_map<std::string, KernelInfo>& a_kernels)
{
  std::vector<const kslicer::KernelInfo*> result;
  result.reserve(16);
  for(const auto& name : a_usedNames)
  {
    auto p = a_kernels.find(name);
    if(p != a_kernels.end())
      result.push_back(&p->second);
  }
  return result;
}

kslicer::DATA_KIND kslicer::GetKindOfType(const clang::QualType qt)
{
  std::string typeName = qt.getAsString();

  bool isContainer = false;
  std::string containerType, containerDataType;
 
  const clang::Type* fieldTypePtr = qt.getTypePtr(); 
  if(fieldTypePtr != nullptr)
  {
    auto typeDecl = fieldTypePtr->getAsRecordDecl();  
    isContainer = (typeDecl != nullptr) && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl);
    if(isContainer)
    {
      auto specDecl = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(typeDecl); 
      kslicer::SplitContainerTypes(specDecl, containerType, containerDataType);
    }
  }

  DATA_KIND kind = DATA_KIND::KIND_UNKNOWN;
  if(qt->isPointerType())
  {
    auto dataType     = qt->getPointeeType();
    containerDataType = kslicer::CleanTypeName(dataType.getAsString());
    
    if(containerDataType == "ISceneObject")
      kind = kslicer::DATA_KIND::KIND_ACCEL_STRUCT;
    else if(containerDataType == "ITexture2DCombined" || containerDataType == "ITexture3DCombined" || containerDataType == "ITextureCubeCombined")
      kind = kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED;
    else
      kind = kslicer::DATA_KIND::KIND_POINTER;
  }
  else if(isContainer)
  {
    containerType     = kslicer::CleanTypeName(containerType);
    containerDataType = kslicer::CleanTypeName(containerDataType);

    if(kslicer::IsTextureContainer(containerType))
    {
      kind = kslicer::DATA_KIND::KIND_TEXTURE;
    }
    else if(containerType == "shared_ptr" || containerType == "unique_ptr")
    {
      if(containerDataType == "ISceneObject")
        kind = kslicer::DATA_KIND::KIND_ACCEL_STRUCT;
      else if(containerDataType == "ITexture2DCombined" || containerDataType == "ITexture3DCombined" || containerDataType == "ITextureCubeCombined")
        kind = kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED;
    }
    else if(containerType.find("vector") != std::string::npos)
    {
      auto typeDecl     = fieldTypePtr->getAsRecordDecl(); 
      auto specDecl     = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(typeDecl); 
      auto typeOfData   = specDecl->getTemplateArgs()[0].getAsType();
      auto typePtr2     = typeOfData.getTypePtr(); 
      
      if(typePtr2 != nullptr)
      {
        auto typeDecl2    = typePtr2->getAsRecordDecl();  
        bool isContainer2 = (typeDecl2 != nullptr) && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl2);
        if(isContainer2)
        {
          auto specDecl2 = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(typeDecl2); 
          kslicer::SplitContainerTypes(specDecl2, containerType, containerDataType);
          containerDataType = kslicer::CleanTypeName(containerDataType);
          if(containerDataType == "ITexture2DCombined" || containerDataType == "ITexture3DCombined" || containerDataType == "ITextureCubeCombined")
            kind = kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY;
          else
            kind = kslicer::DATA_KIND::KIND_VECTOR; 
        }
        else if(typeOfData->isPointerType())
        {
          auto dataType2 = typeOfData->getPointeeType();
          containerDataType = kslicer::CleanTypeName(dataType2.getAsString());
          if(containerDataType == "ITexture2DCombined" || containerDataType == "ITexture3DCombined" || containerDataType == "ITextureCubeCombined")
            kind = kslicer::DATA_KIND::KIND_TEXTURE_SAMPLER_COMBINED_ARRAY;
          else
            kind = kslicer::DATA_KIND::KIND_VECTOR; 
        }
      }
      else
        kind = kslicer::DATA_KIND::KIND_VECTOR; 
    }
    else
      kind = kslicer::DATA_KIND::KIND_VECTOR;  
  }
  else 
    kind = kslicer::DATA_KIND::KIND_POD; 
  return kind;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::IsInExcludedFolder(const std::string& fileName, const std::vector<std::string>& a_excludeFolderList)
{
  bool exclude = false;
  for(auto folder : a_excludeFolderList)       //
  {
    if(fileName.find(folder) != std::string::npos)
    {
      exclude = true;
      break;
    }
  }
  return exclude;
}

struct TypePair
{
  TypePair(){}
  TypePair(const std::string& a_name, const clang::TypeDecl* a_node) : typeName(a_name), node(a_node) {}
  std::string typeName;
  const clang::TypeDecl* node;
  size_t aligment = sizeof(int);  
};

void kslicer::MainClassInfo::ProcessMemberTypes(const std::unordered_map<std::string, kslicer::DeclInClass>& a_otherDecls, clang::SourceManager& a_srcMgr, 
                                                std::vector<kslicer::DeclInClass>& generalDecls)
{
  const auto& a_members           = this->dataMembers;
  const auto  a_additionalTypes   = this->ExtractTypesFromUsedContainers(a_otherDecls);
  const auto& a_excludeFolderList = this->ignoreFolders;
  const auto& a_allDataMembers    = this->allDataMembers;     

  std::unordered_map<std::string, kslicer::DeclInClass> declsByName;
  for(const auto& decl : generalDecls)
    declsByName[decl.name] = decl;
  
  auto internalTypes = kslicer::ListPredefinedMathTypes();
  std::queue<TypePair> typesToProcess;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  for(const auto& member : a_members)
  {
    std::string typeName = kslicer::CleanTypeName(member.type);       // TODO: make type clear function 
    if(member.pTypeDeclIfRecord != nullptr && 
       declsByName.find(typeName) == declsByName.end() && 
       internalTypes.find(typeName) == internalTypes.end())
    {
      auto pFound = a_otherDecls.find(typeName);
      if(pFound != a_otherDecls.end())
        typesToProcess.push(TypePair(typeName, member.pTypeDeclIfRecord));
    }
  }

  for(auto tn : a_additionalTypes)
  {
    std::string typeName = kslicer::CleanTypeName(tn);
    if(declsByName.find(typeName) == declsByName.end() && 
       internalTypes.find(typeName) == internalTypes.end())
    {
      const clang::TypeDecl* node = nullptr;
      for(auto memb : a_allDataMembers)
      {
        if(!memb.second.isContainer)
          continue;
        auto memberTypeName = kslicer::CleanTypeName(memb.second.containerDataType);
        if(memberTypeName == typeName)
        {
          node = memb.second.pContainerDataTypeDeclIfRecord;
          break;
        }
      }
      if(node != nullptr)
        typesToProcess.push(TypePair(typeName, node)); 
    }
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////

  size_t lastDeclOrder = generalDecls.size() == 0 ? 0 : generalDecls.back().order+1;
  std::vector<kslicer::DeclInClass> auxDecls;

  while(!typesToProcess.empty())
  {
    TypePair elem = typesToProcess.front(); typesToProcess.pop();
    if(declsByName.find(elem.typeName) == declsByName.end())
    {
      kslicer::DeclInClass tdecl;
      tdecl.type     = elem.typeName;
      tdecl.name     = elem.typeName;
      tdecl.srcRange = elem.node->getSourceRange();    
      tdecl.srcHash  = kslicer::GetHashOfSourceRange(tdecl.srcRange); 
      tdecl.order    = lastDeclOrder; lastDeclOrder++;
      tdecl.kind     = kslicer::DECL_IN_CLASS::DECL_STRUCT;
      tdecl.extracted= true;
      declsByName[elem.typeName] = tdecl;
      const clang::FileEntry* Entry = a_srcMgr.getFileEntryForID(a_srcMgr.getFileID(elem.node->getLocation()));
      const std::string fileName    = std::string(Entry->getName());        
      const bool        exclude     = IsInExcludedFolder(fileName, a_excludeFolderList); 
      if(!exclude)
        auxDecls.push_back(tdecl);
      
      // process all field types 
      //
      if(elem.node != nullptr && clang::isa<clang::RecordDecl>(elem.node))
      {
        auto pRecordDecl = clang::dyn_cast<clang::RecordDecl>(elem.node);
        for(auto field : pRecordDecl->fields())
        {
          clang::QualType qt = field->getType();
          const std::string typeName2 = kslicer::CleanTypeName(qt.getAsString());
          auto pRecordType = qt->getAsStructureType();
          if(pRecordType != nullptr && declsByName.find(typeName2) == declsByName.end() && internalTypes.find(typeName2) == internalTypes.end())
            typesToProcess.push(TypePair(typeName2, pRecordType->getDecl()));
        }
      }
    }
  }

  std::reverse_copy(auxDecls.begin(), auxDecls.end(), std::back_inserter(generalDecls)); 
}

std::unordered_map<std::string, size_t> ListPredefinedAligmentTypes()
{
  std::unordered_map<std::string, size_t> res;
  res["float2"]   = sizeof(float)*2;
  res["float3"]   = sizeof(float)*4;
  res["float4"]   = sizeof(float)*4;
  res["int2"]     = sizeof(int)*2;
  res["int3"]     = sizeof(int)*4;
  res["int4"]     = sizeof(int)*4;
  res["uint2"]    = sizeof(unsigned)*2;
  res["uint3"]    = sizeof(unsigned)*4;
  res["uint4"]    = sizeof(unsigned)*4;
  res["float4x4"] = sizeof(float)*4;
  res["float3x3"] = sizeof(float)*4; // UNTESTED !!!
  res["float2x2"] = sizeof(float)*2; // UNTESTED !!!

  res["vec2"]  = sizeof(float)*2;
  res["vec3"]  = sizeof(float)*4;
  res["vec4"]  = sizeof(float)*4;
  res["ivec2"] = sizeof(int)*2;
  res["ivec3"] = sizeof(int)*4;
  res["ivec4"] = sizeof(int)*4;
  res["uvec2"] = sizeof(unsigned)*2;
  res["uvec3"] = sizeof(unsigned)*4;
  res["uvec4"] = sizeof(unsigned)*4;
  res["mat4"]  = sizeof(float)*4;
  res["mat3"]  = sizeof(float)*4; // UNTESTED !!!
  res["mat2"]  = sizeof(float)*2; // UNTESTED !!!

  return res;
}

std::unordered_map<std::string, size_t> ListForbiddenTypes()
{
  std::unordered_map<std::string, size_t> res;
  res["float3"]   = sizeof(float)*3;
  res["int3"]     = sizeof(int)*3;
  res["uint3"]    = sizeof(unsigned)*4;
  res["float3x3"] = sizeof(float)*3*3; // UNTESTED !!!

  res["vec3"]  = sizeof(float)*4;
  res["ivec3"] = sizeof(int)*4;
  res["uvec3"] = sizeof(unsigned)*4;
  res["mat3"]  = sizeof(float)*3*3; // UNTESTED !!!

  res["char"]    = sizeof(char);
  res["uchar"]   = sizeof(char);
  res["int8_t"]  = sizeof(char);
  res["uint8_t"] = sizeof(char);

  res["short"]    = sizeof(short);
  res["ushort"]   = sizeof(short);
  res["int16_t"]  = sizeof(short);
  res["uint16_t"] = sizeof(short);

  return res;
}


size_t GetBaseAligmentForGLSL(TypePair* pCurrType, 
                              const std::unordered_map<std::string, TypePair>& a_typeToProcess, 
                              const std::unordered_map<std::string, size_t>& a_endTypes, int a_level)
{
  auto pEndType = a_endTypes.find(pCurrType->typeName);
  if(pEndType != a_endTypes.end())
    return pEndType->second;
  else if(pCurrType->node != nullptr && clang::isa<clang::RecordDecl>(pCurrType->node))
  {
    auto badTypes    = ListForbiddenTypes();   
    auto pRecordDecl = clang::dyn_cast<clang::RecordDecl>(pCurrType->node);
    size_t maxAligment = 0;
    for(auto field : pRecordDecl->fields())
    {
      clang::QualType qt = field->getType();
      const std::string typeName2 = kslicer::CleanTypeName(qt.getAsString());
      
      auto pForbidden = badTypes.find(typeName2);
      if(pForbidden != badTypes.end())
      {
        const std::string varName = field->getNameAsString();
        std::cout << "  [PADDING ALERT]: structure '" << pCurrType->typeName << "' has field '" <<  varName << "' of type '" << typeName2 << "' at level " << a_level+1 << std::endl;
        std::cout << "  [PADDING ALERT]: type '" << typeName2 << "' has different 'size' and 'aligned size' in GLSL which is not possible on the host side in C++" << std::endl;
        std::cout << "  [PADDING ALERT]: we don't allow such types inside structures; please use aligned types inside structures." << std::endl;
      }

      auto pFoundType = a_typeToProcess.find(typeName2);
      TypePair fieldPair(typeName2, pFoundType == a_typeToProcess.end() ? nullptr : pFoundType->second.node);
      size_t fieldAligment = GetBaseAligmentForGLSL(&fieldPair, a_typeToProcess, a_endTypes, a_level+1);
      maxAligment = std::max(maxAligment, fieldAligment);
    }
    return maxAligment;
  }

  return sizeof(int);
}

static inline size_t Padding(size_t a_size, size_t a_alignment)
{
  if (a_size % a_alignment == 0)
    return a_size;
  else
  {
    size_t sizeCut = a_size - (a_size % a_alignment);
    return sizeCut + a_alignment;
  }
}

void kslicer::MainClassInfo::ProcessMemberTypesAligment(std::vector<DataMemberInfo>& a_members, 
                                                        const std::unordered_map<std::string, kslicer::DeclInClass>& a_otherDecls, 
                                                        const clang::ASTContext& a_astContext)
{
  const auto  a_additionalTypes = this->ExtractTypesFromUsedContainers(a_otherDecls);
  const auto& a_allDataMembers  = this->allDataMembers;

  auto internalTypes = kslicer::ListPredefinedMathTypes();
  auto endTypes      = ListPredefinedAligmentTypes();
  for(auto type : endTypes)
  {
    auto p = internalTypes.find(type.first);
    if(p != internalTypes.end())
      internalTypes.erase(p);
  }

  std::unordered_map<std::string, TypePair> typesToProcess;
  for(const auto& member : a_members)
  {
    std::string typeName = kslicer::CleanTypeName(member.type);   
    if(member.pTypeDeclIfRecord != nullptr && internalTypes.find(typeName) == internalTypes.end())
      typesToProcess[typeName] = TypePair(typeName, member.pTypeDeclIfRecord);
  }

  for(auto tn : a_additionalTypes)
  {
    std::string typeName = kslicer::CleanTypeName(tn);
    if(typesToProcess.find(typeName) == typesToProcess.end() && internalTypes.find(typeName) == internalTypes.end())
    {
      const clang::TypeDecl* node = nullptr;
      for(auto memb : a_allDataMembers)
      {
        if(!memb.second.isContainer)
          continue;
        auto memberTypeName = kslicer::CleanTypeName(memb.second.containerDataType);
        if(memberTypeName == typeName)
        {
          node = memb.second.pContainerDataTypeDeclIfRecord;
          break;
        }
      } 
      typesToProcess[typeName] = TypePair(typeName, node);  // #TODO: extract pTypeDeclIfRecord for containers data also  
    }
  }

  for(auto& type : typesToProcess)
  {
    type.second.aligment = GetBaseAligmentForGLSL(&type.second, typesToProcess, endTypes, 0);
    if(type.second.node == nullptr)
      continue;
    auto clangType = type.second.node->getTypeForDecl();
    if(clangType == nullptr)
      continue;

    auto typeInfo    = a_astContext.getTypeInfo(clangType);
    auto sizeInBytes = typeInfo.Width / 8;
    if(sizeInBytes > 0 && sizeInBytes % type.second.aligment != 0)
    {
      std::cout << "  [PADDING ALERT]: structure '" << type.second.typeName << "' has align of " << type.second.aligment << " and size of " << sizeInBytes << " bytes!" << std::endl;
      std::cout << "  [PADDING ALERT]: structure '" << type.second.typeName << "' has different 'size' and 'aligned size' in GLSL which is not possible on the host side in C++" << std::endl;
      std::cout << "  [PADDING ALERT]: we don't allow such types inside buffers or any other containers; please use aligned types." << std::endl;
    }
  }

  for(auto& member : a_members)
  {
    std::string typeName = kslicer::CleanTypeName(member.type);
    auto p = typesToProcess.find(typeName);
    if(p != typesToProcess.end())
    {
      member.aligmentGLSL       = std::max(p->second.aligment, sizeof(int));
      member.alignedSizeInBytes = Padding(member.sizeInBytes, member.aligmentGLSL);
    }
  }

}

std::unordered_set<std::string> kslicer::MainClassInfo::ExtractTypesFromUsedContainers(const std::unordered_map<std::string, kslicer::DeclInClass>& a_otherDecls)
{
  std::unordered_set<std::string> res;
  for(const auto& k : this->kernels) // fix this flag for members that were used in member functions but not in kernels directly
  {
    for(const auto& c : k.second.usedContainers)
    {
      auto pFound = this->allDataMembers.find(c.second.name);
      if(pFound != this->allDataMembers.end())
      {
        std::string typeForSeek = kslicer::CleanTypeName(pFound->second.containerDataType);
        auto pFoundTypeDecl = a_otherDecls.find(typeForSeek);
        if(pFoundTypeDecl != a_otherDecls.end() && pFound->second.containerType != "shared_ptr" &&  pFound->second.containerType != "unique_ptr")
          res.insert(pFound->second.containerDataType);
      }
    }
  }
  return res;
}