#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"
#include "extractor.h"
#include "initial_pass.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"

#include <sstream>
#include <algorithm>


/**
\brief processing of C++ member function for virtual functions
*/
class MemberRewriter : public kslicer::GLSLFunctionRewriter
{
public:
  
  MemberRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::MainClassInfo::DImplClass& dImpl, int a_level) : 
                kslicer::GLSLFunctionRewriter(R, a_compiler, a_codeInfo, kslicer::ShittyFunction()), 
                m_processed(dImpl.memberFunctions), m_fields(dImpl.fields), m_className(dImpl.name), m_objBufferName(dImpl.objBufferName), m_interfaceName(dImpl.interfaceName), 
                m_mainClassName(a_codeInfo->mainClassName), dataClassNames(a_codeInfo->dataClassNames), m_vfhLevel(a_level)
  { 

  }

  bool VisitMemberExpr_Impl(clang::MemberExpr* expr) override
  {
    clang::ValueDecl* pValueDecl = expr->getMemberDecl();
    if(!clang::isa<clang::FieldDecl>(pValueDecl))
      return true;

    clang::FieldDecl* pFieldDecl   = clang::dyn_cast<clang::FieldDecl>(pValueDecl);
    clang::RecordDecl* pRecodDecl  = pFieldDecl->getParent();
    const std::string thisTypeName = pRecodDecl->getNameAsString();
    const std::string debugText    = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); 

    const std::string fieldName     = pFieldDecl->getNameAsString();
    const clang::QualType qt        = pFieldDecl->getType();
    const std::string fieldTypeName = qt.getAsString(); 

    const auto typeDecl    = qt->getAsRecordDecl();
    const bool isContainer = (typeDecl != nullptr) && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl);

    if((thisTypeName == m_className || thisTypeName == m_interfaceName) && WasNotRewrittenYet(expr))
    {
      const clang::Expr* baseExpr = expr->getBase(); 
      std::string exprContent     = RecursiveRewrite(baseExpr);
      std::string exprReplaced    = m_objBufferName + "[selfId]." + exprContent;
      if(m_vfhLevel >= 2)
        exprReplaced = "all_references." + m_className + "_buffer." + exprReplaced;
      m_rewriter.ReplaceText(expr->getSourceRange(), exprReplaced);
      MarkRewritten(expr);
    }
    else if(dataClassNames.find(thisTypeName) != dataClassNames.end() && WasNotRewrittenYet(expr))
    {
      std::string prefix = "ubo.";
      if(isContainer)
      {
        prefix = "";
        auto pComposPrefix = m_codeInfo->composPrefix.find(thisTypeName);
        if(pComposPrefix != m_codeInfo->composPrefix.end())
          prefix = pComposPrefix->second + "_";
      }
      
      //std::cout << "  [MemberRewriter]: fieldName = " << fieldName.c_str() << std::endl;

      const std::string filedNameToFind = (isContainer && prefix != "") ? prefix + fieldName : fieldName;
      auto p = m_codeInfo->allDataMembers.find(filedNameToFind);    
      if(p != m_codeInfo->allDataMembers.end()) 
      {
        kslicer::UsedContainerInfo container;
        container.name    = p->first;
        container.type    = p->second.type;
        container.kind    = p->second.kind;
        container.isConst = qt.isConstQualified();

        if(isContainer)
        {
          kslicer::ProbablyUsed pcontainer;
          pcontainer.astNode       = pFieldDecl;
          pcontainer.isContainer   = true;
          pcontainer.info          = container;
          pcontainer.interfaceName = m_interfaceName;
          pcontainer.className     = m_className;
          pcontainer.objBufferName = m_objBufferName;
          auto specDecl = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(typeDecl);
          kslicer::SplitContainerTypes(specDecl, pcontainer.containerType, pcontainer.containerDataType);
          m_codeInfo->usedProbably[container.name] = pcontainer;
        }
        else
        {
          kslicer::ProbablyUsed pvar;
          pvar.astNode       = pFieldDecl;
          pvar.isContainer   = false;
          pvar.info          = container;
          pvar.interfaceName = m_interfaceName;
          pvar.className     = m_className;
          pvar.objBufferName = m_objBufferName;
          m_codeInfo->usedProbably[container.name] = pvar;
          p->second.usedInKernel = true;
        }
      }

      m_rewriter.ReplaceText(expr->getSourceRange(), prefix + fieldName);
      MarkRewritten(expr);
    }  
    else if(expr->isArrow() && WasNotRewrittenYet(expr))
    {
      // a->b ==> a.b
      clang::Expr* base    = expr->getBase();       // Получаем указатель на ноду, соответствующую a
      auto  memberNameInfo = expr->getMemberNameInfo();
      auto  memberName     = memberNameInfo.getName().getAsString();
      
      //std::cout << "  [MemberRewriter]: process with '.' for " << thisTypeName.c_str() << "::" << fieldName.c_str() << std::endl;

      m_rewriter.ReplaceText(expr->getSourceRange(), kslicer::GetRangeSourceCode(base->getSourceRange(), m_compiler) + "." + memberName);
      MarkRewritten(expr);
    }

    return true;
  }

  bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr) override 
  { 
    const auto op = expr->getOpcodeStr(expr->getOpcode());
  
    clang::Expr* subExpr = expr->getSubExpr();
    if(subExpr == nullptr)
      return true;

    if((op == "*" || op == "&") && WasNotRewrittenYet(expr->getSubExpr()) )
    {
      std::string text = RecursiveRewrite(expr->getSubExpr());
      m_lastRewrittenText = text;
      m_rewriter.ReplaceText(expr->getSourceRange(), text);
      MarkRewritten(expr->getSubExpr());
    }

    return true; 
  }
  
  /// \return whether \p Ty points to a const type, or is a const reference.
  //
  static bool isPointerToConst(clang::QualType Ty) 
  {
    return !Ty->getPointeeType().isNull() && Ty->getPointeeType().getCanonicalType().isConstQualified();
  }

  using ArgList = std::vector< std::pair<std::string, std::string> >; 

  std::string RewriteMemberDecl(clang::CXXMethodDecl* fDecl, const std::string& classTypeName, ArgList* pList = nullptr)
  {
    std::string fname  = fDecl->getNameInfo().getName().getAsString();
    std::string result = m_codeInfo->pShaderFuncRewriter->RewriteStdVectorTypeStr(fDecl->getReturnType().getAsString()) + " " + classTypeName + "_" + fname + "_" + m_objBufferName + "(uint selfId" ;
    if(fDecl->getNumParams() != 0)
      result += ", ";

    bool isKernel = m_codeInfo->IsKernel(fname); 

    for(uint32_t i=0; i < fDecl->getNumParams(); i++)
    {
      const clang::ParmVarDecl* pParam        = fDecl->getParamDecl(i);
      const clang::QualType typeOfParam       =	pParam->getType();
      const clang::IdentifierInfo* identifier = pParam->getIdentifier();
      if(identifier == nullptr)
        continue;
      
      std::string typeNameRewritten = m_codeInfo->pShaderFuncRewriter->RewriteStdVectorTypeStr(typeOfParam.getAsString());
      if(dataClassNames.find(typeNameRewritten) != dataClassNames.end()) 
      {
        if(i==fDecl->getNumParams()-1)
          result[result.rfind(",")] = ' ';
        continue;
      }

      if(typeOfParam->isPointerType() && !typeOfParam->getPointeeType().isConstQualified())
        typeNameRewritten = std::string("inout ") + typeNameRewritten;

      result += typeNameRewritten + " " + std::string(identifier->getName());
      if(pList != nullptr)
        pList->push_back(std::make_pair(typeNameRewritten, std::string(identifier->getName())));

      if(i!=fDecl->getNumParams()-1)
        result += ", ";
    }

    return result + ") ";
  }

  bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl) override
  {
    if(isCopy)
      return true;

    std::string fname = fDecl->getNameInfo().getName().getAsString();
    auto thisType     = fDecl->getThisType();
    auto qtOfClass    = thisType->getPointeeType(); 
    std::string classTypeName = kslicer::CutOffStructClass(qtOfClass.getAsString());

    if(classTypeName.find(fname) != std::string::npos || classTypeName.find(fname.substr(1)) != std::string::npos || fname == "GetTag" || fname == "GetSizeOf")
      return true; // exclude constructor, destructor and special functions
    
    if(WasNotRewrittenYet(fDecl->getBody()))
    { 
      std::string declSource = RewriteMemberDecl(fDecl, classTypeName);
      std::string bodySource = RecursiveRewrite(fDecl->getBody());

      auto p = declByName.find(fname);
      if(p == declByName.end()) {
        ArgList args;
        declByName[fname] = RewriteMemberDecl(fDecl, m_interfaceName, &args); // make text decls for IMaterial_sample_materials(...)
        argsByName[fname] = args;
      }

      auto beginPos = declSource.find(" ");
      auto endPos   = declSource.find("(");

      kslicer::MainClassInfo::DImplFunc funcData;
      funcData.decl          = fDecl;
      funcData.name          = fname;
      funcData.nameRewritten = declSource.substr(beginPos+1, endPos - beginPos - 1);
      funcData.srcRewritten  = declSource + bodySource;
      funcData.isEmpty       = false;
      funcData.isConstMember = fDecl->isConst();
      //funcData.mainClassPass = mainClassDataPass;

      if(clang::isa<clang::CompoundStmt>(fDecl->getBody()))
      {
        clang::CompoundStmt* pBody = clang::dyn_cast<clang::CompoundStmt>(fDecl->getBody());
        funcData.isEmpty = pBody->body_empty();
      }
     
      m_processed.push_back(funcData);
      MarkRewritten(fDecl->getBody());
    }

    return true;
  }

  bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* call) override
  {
    const clang::FunctionDecl* fDecl = call->getDirectCallee();  
    const std::string fname          = fDecl->getNameInfo().getName().getAsString();
    const clang::QualType qt         = call->getObjectType();
    std::string classTypeName        = kslicer::CutOffStructClass(qt.getAsString());

    if(WasNotRewrittenYet(call))
    { 
      std::string textRes = classTypeName + "_" + fname + "_" + m_objBufferName + "(selfId";
      {
        const auto pPrefix  = m_codeInfo->composPrefix.find(classTypeName);
        if(pPrefix != m_codeInfo->composPrefix.end())
          textRes = pPrefix->second + "_" + fname + "(";
        else if(call->getNumArgs() > 0)
          textRes += ",";
      }

      for(unsigned i=0;i<call->getNumArgs();i++)
      {
        const auto pParam                   = call->getArg(i);
        const clang::QualType typeOfParam   =	pParam->getType();
        const std::string typeNameRewritten = typeOfParam.getAsString();
        if(dataClassNames.find(typeNameRewritten) != dataClassNames.end()) 
        {
          if(i==call->getNumArgs()-1)
            textRes[textRes.rfind(",")] = ' ';
          continue;
        }

        textRes += RecursiveRewrite(call->getArg(i));
        if(i < call->getNumArgs()-1)
          textRes += ",";
      }
      textRes += ")";
      
      m_rewriter.ReplaceText(call->getSourceRange(), textRes);
      MarkRewritten(call);
    }

    return true;
  }

  bool VisitFieldDecl_Impl(clang::FieldDecl* pFieldDecl) override 
  { 
    clang::RecordDecl* pRecodDecl  = pFieldDecl->getParent();
    const std::string thisTypeName = pRecodDecl->getNameAsString();
    if(thisTypeName == m_className)
      m_fields.push_back(kslicer::GetRangeSourceCode(pFieldDecl->getSourceRange(), m_compiler));
    return true; 
  } 

  std::unordered_map<std::string, ArgList>     argsByName;
  std::unordered_map<std::string, std::string> declByName; 
  const std::unordered_set<std::string>&       dataClassNames; 

private:
    
  std::vector<kslicer::MainClassInfo::DImplFunc>& m_processed;
  std::vector<std::string>&                       m_fields;
  const std::string&                              m_className;
  const std::string&                              m_mainClassName;
  const std::string&                              m_objBufferName;
  const std::string&                              m_interfaceName;

  bool isCopy = false;
  int m_vfhLevel = 0;
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  
  //std::unordered_set<uint64_t>  m_rewrittenNodes;
  inline void MarkRewritten(const clang::Stmt* expr) { FunctionRewriter::MarkRewritten(expr); }

  inline bool WasNotRewrittenYet(const clang::Stmt* expr) { return FunctionRewriter::WasNotRewrittenYet(expr); }

  std::string RecursiveRewrite(const clang::Stmt* expr) override
  {
    if(expr == nullptr)
      return "";
    MemberRewriter rvCopy = *this;
    rvCopy.isCopy = true;
    rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
    std::string text = m_rewriter.getRewrittenText(expr->getSourceRange());
    return (text != "") ? text : kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
  }
  
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void kslicer::MainClassInfo::ProcessVFH(const std::vector<const clang::CXXRecordDecl*>& a_decls, const clang::CompilerInstance& a_compiler)
{
  //
  //
  for(auto& p : m_vhierarchy)
  {
    const clang::CXXRecordDecl* pBaseClass = nullptr;
    std::string className = kslicer::CutOffStructClass(p.first);
    
    // find target base class
    //
    for(const auto& decl : a_decls)
    {
      const std::string testName = decl->getNameAsString();
      if(testName == className)
      {
        pBaseClass = decl;
        break;
      }
      //std::cout << "  found class: " << testName.c_str() << std::endl;
    }
    if(pBaseClass == nullptr)
      return;

    p.second.interfaceDecl = pBaseClass;

    // find all derived classes for target base class
    //
    std::unordered_map<std::string, std::string>             declByName;
    std::unordered_map<std::string, MemberRewriter::ArgList> argsByName;

    clang::Rewriter rewrite2;
    rewrite2.setSourceMgr(a_compiler.getSourceManager(), a_compiler.getLangOpts());
    
    for(const auto& decl : a_decls)
    {
      if(decl->isDerivedFrom(pBaseClass))
      {
        DImplClass dImpl;
        dImpl.decl = decl;
        dImpl.name = decl->getNameAsString();
        dImpl.objBufferName = p.second.objBufferName;
        dImpl.interfaceName = p.second.interfaceName;
        
        MemberRewriter rv(rewrite2, a_compiler, this, dImpl, int(p.second.level)); 
        rv.TraverseDecl(const_cast<clang::CXXRecordDecl*>(dImpl.decl));                                  
        
        for (const auto& kv : rv.declByName) 
          declByName[kv.first] = kv.second;
        for (const auto& kv : rv.argsByName)
          argsByName[kv.first] = kv.second; 

        dImpl.isEmpty = (dImpl.name.find("Empty") != std::string::npos);
        //for(auto member : dImpl.memberFunctions)
        //{
        //  if(!member.isEmpty)
        //  {
        //    dImpl.isEmpty = false;
        //    break;
        //  }
        //}

        for(auto& k : kernels)
        {
          if(k.second.className == dImpl.name)
            k.second.interfaceName = className;
        }
        
        bool alreadyHasSuchImpl = false;
        for(const auto& impl : p.second.implementations) {
          if(impl.name == dImpl.name) {
            alreadyHasSuchImpl = true;
            break;
          }
        }
        
        if(!alreadyHasSuchImpl)
          p.second.implementations.push_back(dImpl);
      }
    }

    for(auto& f : p.second.virtualFunctions) {
      auto pFound = declByName.find(f.second.name);
      if(pFound != declByName.end())
        f.second.declRewritten = pFound->second;

      auto pFound2 = argsByName.find(f.second.name);
      if(pFound2 != argsByName.end())
        f.second.args = pFound2->second;
    }
  }
  
  // debug output
  //
  for(const auto& p : m_vhierarchy)
  {
    for(const auto& impl : p.second.implementations)
      std::cout << "  found " << p.first.c_str() << " --> " << impl.name.c_str() << std::endl;
  }

}


class TagSeeker : public clang::RecursiveASTVisitor<TagSeeker>
{
public:
  
  TagSeeker(const clang::CompilerInstance& a_compiler, std::vector<kslicer::DeclInClass>& a_constants, std::unordered_map<std::string, std::string>& a_tagByName) : 
            m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_knownConstants(a_constants), m_tagByClassName(a_tagByName) { m_tagByClassName.clear(); }

  bool VisitCXXMethodDecl(const clang::CXXMethodDecl* f)
  { 
    if(!f->hasBody())
      return true;

    // Get name of function
    const std::string fname = f->getNameInfo().getName().getAsString();
    if(fname == "GetTag" || fname == "GetTypeId")
    {
      const clang::QualType qThisType = f->getThisType();   
      const clang::QualType classType = qThisType->getPointeeType();
      const std::string thisTypeName  = kslicer::CutOffStructClass(classType.getAsString());
      
      auto funcBody = f->getBody();
      if(clang::isa<clang::CompoundStmt>(funcBody))
      {
        clang::CompoundStmt* s2 = clang::dyn_cast<clang::CompoundStmt>(funcBody);
        for(auto iter = s2->body_begin(); iter != s2->body_end(); ++iter)
        {
          if(clang::isa<clang::ReturnStmt>(*iter))
          {
            funcBody = *iter;
            break;
          }
        }
      }
      
      if(!clang::isa<clang::ReturnStmt>(funcBody))
      {
        std::cout << "  [TagSeeker::Error]: " << "Can't find returt statement in 'GetTag/GetTypeId' fuction body for '" <<  thisTypeName.c_str() <<  "' class." << std::endl;
        return true;
      }

      clang::ReturnStmt* retStmt = clang::dyn_cast< clang::ReturnStmt>(funcBody);
      clang::Expr* retVal        = retStmt->getRetValue(); 
      const std::string tagName  = kslicer::GetRangeSourceCode(retVal->getSourceRange(), m_compiler);
      
      for(const auto& decl : m_knownConstants)
      {
        //auto pos = funcBody.find(decl.name);
        if(decl.name == tagName)
        {
          m_tagByClassName[thisTypeName] = decl.name;
          break;
        }
      }
    }
    
    return true;
  }

private:
  
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
  std::vector<kslicer::DeclInClass>& m_knownConstants;
  std::unordered_map<std::string, std::string>& m_tagByClassName; 
};


void kslicer::MainClassInfo::ExtractVFHConstants(const clang::CompilerInstance& compiler, clang::tooling::ClangTool& Tool)
{
  for(auto& p : m_vhierarchy)
  {
    //// (1) get all constants inside interface
    //
    std::cout << "  process " << p.second.interfaceName.c_str() << std::endl;
    p.second.usedDecls = kslicer::ExtractTCFromClass(p.second.interfaceName, p.second.interfaceDecl, compiler, Tool);

    //// (2) juxtapose constant TAG and class implementation by analyzing GetTag() function
    //
    TagSeeker visitor(compiler, p.second.usedDecls, p.second.tagByClassName);
    for(auto impl : p.second.implementations)
      visitor.TraverseDecl(const_cast<clang::CXXRecordDecl*>(impl.decl));
  }

}

std::vector< std::pair<std::string, std::string> > kslicer::MainClassInfo::GetFieldsFromStruct(const clang::CXXRecordDecl* recordDecl,  size_t* pSummOfFiledsSize) const
{
  std::vector< std::pair<std::string, std::string> > fieldInfo;

  size_t summOfSize = 0;
  const auto& context = recordDecl->getASTContext();

  // Проходим по всем полям структуры
  for (auto it = recordDecl->field_begin(); it != recordDecl->field_end(); ++it) 
  {
    const clang::FieldDecl* field = *it;
    const clang::QualType fieldType = field->getType();
    const clang::Type* baseType = fieldType.getTypePtrOrNull();
    // Получаем имена типов и полей и добавляем их в массив пар
    if (baseType) 
    {
      std::string typeName  = baseType->getCanonicalTypeInternal().getAsString();
      std::string fieldName = field->getNameAsString();
      std::string typeNameR = pShaderFuncRewriter->RewriteStdVectorTypeStr(typeName, fieldName);
  
      fieldInfo.push_back(std::make_pair(typeNameR, fieldName));

      const clang::Type*     fieldType = field->getType().getTypePtr();
      const clang::TypeInfo& typeInfo  = context.getTypeInfo(fieldType);

      const uint64_t sizeInBits  = typeInfo.Width;
      const uint64_t sizeInBytes = llvm::divideCeil(sizeInBits, 8);
      summOfSize += size_t(sizeInBytes);
    }
  }

  if(pSummOfFiledsSize != nullptr)
    (*pSummOfFiledsSize) = summOfSize;

  return fieldInfo;
}

bool kslicer::IsCalledWithArrowAndVirtual(const clang::CXXMemberCallExpr* f) 
{
    if (!f) return false;

    // Получаем выражение, на котором вызывается метод
    const auto* calleeExpr = f->getCallee();
    if (!calleeExpr) return false;

    // Проверяем, является ли выражение MemberExpr
    const clang::MemberExpr* memberExpr = clang::dyn_cast<clang::MemberExpr>(calleeExpr);
    if (!memberExpr) return false;

    // Получаем выражение объекта
    const clang::Expr* baseExpr = memberExpr->getBase();
    if (!baseExpr) return false;

    // Проверяем тип базового выражения
    if (const auto* baseType = baseExpr->getType()->getPointeeType().getTypePtrOrNull()) {
        if (baseType->isRecordType()) {
            // Использован оператор "->"
            const clang::CXXMethodDecl* methodDecl = clang::dyn_cast<clang::CXXMethodDecl>(memberExpr->getMemberDecl());
            if (methodDecl && methodDecl->isVirtual()) {
                // Метод является виртуальным
                return true;
            }
        }
    }
    
    // Если условие не выполнено, возвращаем false
    return false;
}

kslicer::VFHAccessNodes kslicer::GetVFHAccessNodes(const clang::CXXMemberCallExpr* f, const clang::CompilerInstance& a_compiler) 
{
  VFHAccessNodes result = {};

  if (!f) return result;
  // Get the MemberExpr for the method call
  const clang::MemberExpr* memberExpr = clang::dyn_cast<clang::MemberExpr>(f->getCallee());
  if (!memberExpr) return result;
  // Get the base of the MemberExpr
  const clang::Expr* baseExpr = memberExpr->getBase();
  if (!baseExpr) return result;
  // Expecting baseExpr to be an ImplicitCastExpr
  const clang::ImplicitCastExpr* castExpr = clang::dyn_cast<clang::ImplicitCastExpr>(baseExpr);
  if (!castExpr) return result;
  // Get the subexpression of the cast
  const clang::Expr* subExpr = kslicer::RemoveImplicitCast(castExpr->getSubExpr());
  if (!subExpr) 
    return result;

  // Проверка на первый тип/уровень вызова, соотвествующий VFH_LEVEL_1: "(m_materials.data() + matId)->GetColor()"
  //
  if (const clang::ParenExpr* parenExpr = clang::dyn_cast<clang::ParenExpr>(subExpr))
  {
    // Get the subexpression of the paren
    const clang::Expr* innerExpr = parenExpr->getSubExpr();
    if (!innerExpr) return result;

    // Expecting innerExpr to be a BinaryOperator
    const clang::BinaryOperator* binOp = clang::dyn_cast<clang::BinaryOperator>(innerExpr);
    if (!binOp || binOp->getOpcode() != clang::BO_Add) return result;

    // Get the left and right hand sides of the binary operator
    const clang::Expr* lhs = binOp->getLHS();
    const clang::Expr* rhs = binOp->getRHS();

    if (!lhs || !rhs) 
      return result;

    std::string buffText = GetRangeSourceCode(lhs->getSourceRange(), a_compiler); 

    result.buffName   = buffText.substr(0, buffText.find(".data()"));          // The left hand side should be a CXXMemberCallExpr (m_materials.data())
    result.offsetName = GetRangeSourceCode(rhs->getSourceRange(), a_compiler); // The right hand side should be a DeclRefExpr (mid)
  }
  // Проверка на второй тип/уровень вызова, соотвествующий VFH_LEVEL_2: "m_materials[matId]->GetColor()"
  else if(const clang::CXXOperatorCallExpr* arrayExpr = clang::dyn_cast<clang::CXXOperatorCallExpr>(subExpr))
  {
    if(arrayExpr->getOperator() != clang::OO_Subscript) // operator[]
      return result;

    const auto arg0 = kslicer::RemoveImplicitCast(arrayExpr->getArg(0));
    const auto arg1 = kslicer::RemoveImplicitCast(arrayExpr->getArg(1));
     
    result.buffName   = GetRangeSourceCode(arg0->getSourceRange(), a_compiler); // The left hand side should be a 'm_materials'
    result.offsetName = GetRangeSourceCode(arg1->getSourceRange(), a_compiler); // The right hand side should be a 'mid'
  }

  const clang::Expr* baseCallExpr = f->getImplicitObjectArgument();
  if (baseCallExpr) {
    clang::QualType baseType = baseCallExpr->getType();
    if (const clang::CXXRecordDecl* recordDecl = baseType->getPointeeCXXRecordDecl())
      result.interfaceName     = recordDecl->getNameAsString();
      result.interfaceTypeName = kslicer::ClearTypeName(baseType.getAsString());
      ReplaceFirst(result.interfaceTypeName, "*", "");
      while(ReplaceFirst(result.interfaceTypeName, " ", ""));
  }

  return result;
}

bool kslicer::MainClassInfo::IsVFHBuffer(const std::string& a_name, VFH_LEVEL* pOutLevel, VFHHierarchy* pHierarchy) const
{
  bool isVFHBuffer = false;
  for(const auto& vfh : this->m_vhierarchy) {
    if(vfh.second.objBufferName == a_name) {
      isVFHBuffer = true;
      if(pOutLevel != nullptr)
        (*pOutLevel) = vfh.second.level;
      if(pHierarchy != nullptr)
        (*pHierarchy) = vfh.second;
      break;
    }
  }
  return isVFHBuffer;
}

void kslicer::MainClassInfo::AppendAllRefsBufferIfNeeded(std::vector<DataMemberInfo>& a_vector)
{
  bool exitFromThisFunction = true;
  for(const auto& h : m_vhierarchy) 
    if(int(h.second.level) >= 2)
      exitFromThisFunction = false;

  if(exitFromThisFunction)
    return;
 
  const std::string nameOfBuffer = "all_references";

  auto pMember = std::find_if(a_vector.begin(), a_vector.end(), [&nameOfBuffer](const DataMemberInfo & m) { return m.name == nameOfBuffer; });
  if(pMember == a_vector.end())
  {
    // add vector itself
    //
    DataMemberInfo memberVFHTable;
    memberVFHTable.name              = nameOfBuffer;
    memberVFHTable.type              = "std::vector<AllBufferReferences>"; 
    memberVFHTable.containerDataType = "AllBufferReferences"; 
    memberVFHTable.containerType     = "std::vector";
    memberVFHTable.isContainer       = true;
    memberVFHTable.isSingle          = true;
    memberVFHTable.kind              = DATA_KIND::KIND_VECTOR;
    a_vector.push_back(memberVFHTable);
    
    // add vector size and capacity for this vector
    //
    kslicer::DataMemberInfo size;
    size.type         = "uint";
    size.sizeInBytes  = sizeof(unsigned int);
    size.name         = memberVFHTable.name + "_size";
    size.usedInKernel    = true;
    size.isContainerInfo = true;
    size.kind            = kslicer::DATA_KIND::KIND_POD;
    kslicer::DataMemberInfo capacity = size;
    capacity.name     = memberVFHTable.name + "_capacity";
    a_vector.push_back(size);
    a_vector.push_back(capacity);

    pMember = std::find_if(a_vector.begin(), a_vector.end(), [&nameOfBuffer](const DataMemberInfo & m) { return m.name == nameOfBuffer; });
  }
    
  if(pMember == a_vector.end())
  {
    std::cout << "[AppendAllRefsBufferIfNeeded]: ERROR, can't find data member '" << nameOfBuffer.c_str() << "'" << std::endl;
    return;
  }

  // add to kernel.usedContainers to bind it to shaders further (because we must readreferences from this buffer)
  //
  for(auto& k : this->kernels) // TODO: check if kernel actually needs this buffer in some way
  {
    bool usedWithAtLeastOneVFH = false;
    for(const auto& h : m_vhierarchy) 
    {
      auto p = k.second.usedContainers.find(h.second.objBufferName);
      if(p != k.second.usedContainers.end() && int(h.second.level) >= 2) {
        usedWithAtLeastOneVFH = true;
        break;
      }
    }

    if(usedWithAtLeastOneVFH)
    {
      kslicer::UsedContainerInfo info;
      info.type    = pMember->type;
      info.name    = pMember->name;
      info.kind    = pMember->kind;
      info.isConst = true;
      k.second.usedContainers[info.name] = info; 
      if(info.name == "")
      {
        int a = 2;
      }
    }
  }
  
  for(const auto& h : m_vhierarchy) {
    if(int(h.second.level) >= 2) {
      for(auto impl : h.second.implementations) {
        if(impl.isEmpty)
          continue;
        BufferReference ref;
        ref.name       = impl.name;
        ref.typeOfElem = impl.name;
        this->m_allRefsFromVFH.push_back(ref);
      }
    }
  }
}

void kslicer::MainClassInfo::AppendAccelStructForIntersectionShadersIfNeeded(std::vector<DataMemberInfo>& a_vector, std::string composImplName)
{
  struct IntersectionShader
  {
    std::string   interfaceName; 
    std::string   functionName;
    VFHHierarchy* hierarchy = nullptr;
    DataMemberInfo memberInfo;
  };

  std::vector<IntersectionShader> shaders;

  for(auto& h : m_vhierarchy) {
    for(auto is : intersectionShaders) {
      if(is.first == h.second.interfaceName) {
        IntersectionShader entity;
        entity.interfaceName = is.first;
        entity.functionName  = is.second;
        entity.hierarchy     = &h.second;
        for(auto& impl : h.second.implementations) { // mark all intersection shaders
          for(auto& func : impl.memberFunctions) {
            if(func.name == entity.functionName) {
              func.isIntersection      = true;
              h.second.hasIntersection = true;
              break;                                 // allow single intersection shader per implementation, i.e. in 'impl.memberFunctions'
            }
          }
        }
        shaders.push_back(entity);
      }
    }
  }
  
  if(shaders.size() == 0) // exit if we dont have intersection shaders indication and didn't found appropriate VFH
    return;

  for(auto& group : shaders)
  {
    std::string memberName = group.hierarchy->objBufferName;
    for(auto prefix : composPrefix) {
      auto pos = memberName.find(prefix.second);
      if(pos != std::string::npos) {
        memberName                     = prefix.second;
        group.hierarchy->accStructName = prefix.second;
        break;
      }
    }

    for(auto& member : a_vector) {
      if(member.kind == DATA_KIND::KIND_ACCEL_STRUCT && member.name == memberName) {
        member.hasIntersectionShader = true;
        member.intersectionClassName = composImplName;
        group.memberInfo = member;
      }
    }
  }

  // add to kernel.usedContainers to bind it to shaders further (because we must readreferences from this buffer)
  //
  for(auto& k : this->kernels) 
  {
    for(const auto group : shaders)
    {
      auto& h = *group.hierarchy;
      auto p = k.second.usedContainers.find(h.objBufferName);
      if(p != k.second.usedContainers.end() && int(h.level) >= 2) 
      {
        kslicer::UsedContainerInfo info;
        info.type    = "std::shared_ptr<struct ISceneObject>";
        info.name    = group.memberInfo.name;
        info.kind    = DATA_KIND::KIND_ACCEL_STRUCT;
        info.isConst = true;
        k.second.usedContainers[info.name] = info; 
      }
    }
  }
  
}

std::unordered_map<std::string, kslicer::MainClassInfo::VFHHierarchy> kslicer::MainClassInfo::SelectVFHOnlyUsedByKernel(const std::unordered_map<std::string, VFHHierarchy>& a_hierarhices, const KernelInfo& k) const
{
  auto copy = a_hierarhices; 
  {
    copy.clear();
    for(const auto& h : a_hierarhices) {
      auto p = k.usedContainers.find(h.second.objBufferName);
      if(p != k.usedContainers.end())
        copy.insert(h);
    }
  }

  return copy;
}
