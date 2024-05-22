#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"
#include "extractor.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"

#include <sstream>
#include <algorithm>


/**
\brief processing of C++ member function for virtual kernels

  1) C++ class --> C style struct; this --> self; 
  2) *payload => payload[tid]; payload->member => payload[tid].member (TBD)
  3) this->vector[...]                                                (TBD)

*/
class MemberRewriter : public kslicer::FunctionRewriter
{
public:
  
  MemberRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::MainClassInfo::DImplClass& dImpl) : 
                FunctionRewriter(R, a_compiler, a_codeInfo), 
                m_processed(dImpl.memberFunctions), m_fields(dImpl.fields), m_className(dImpl.name), m_mainClassName(a_codeInfo->mainClassName)
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

    if(thisTypeName != m_className) // ignore other than this-> expr
      return true;
    
    if(WasNotRewrittenYet(expr))
    {
      const clang::Expr* baseExpr = expr->getBase(); 
      std::string exprContent     = RecursiveRewrite(baseExpr);
      m_rewriter.ReplaceText(expr->getSourceRange(), "self->" + exprContent);
      MarkRewritten(expr);
    }

    return true;
  }

  bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr) override 
  { 
    const auto op = expr->getOpcodeStr(expr->getOpcode());
    if(expr->canOverflow() || op != "*") // -UnaryOperator ... lvalue prefix '*' cannot overflow
      return true;

    clang::Expr* subExpr = expr->getSubExpr();
    if(subExpr == nullptr)
      return true;
    
    std::string exprInside = kslicer::GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);  ; //RecursiveRewrite(subExpr);
    if(m_fakeOffsArgs.find(exprInside) == m_fakeOffsArgs.end())
      return true;

    if(WasNotRewrittenYet(subExpr))
    {
      if(m_codeInfo->megakernelRTV || m_fakeOffsetExp == "") // m_fakeOffsetExp == "" may happen for merged functions member during class composition
        m_rewriter.ReplaceText(expr->getSourceRange(), exprInside);
      else
        m_rewriter.ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
    }

    return true; 
  }
  
  /// \return whether \p Ty points to a const type, or is a const reference.
  //
  static bool isPointerToConst(clang::QualType Ty) 
  {
    return !Ty->getPointeeType().isNull() && Ty->getPointeeType().getCanonicalType().isConstQualified();
  }

  std::string RewriteMemberDecl(clang::CXXMethodDecl* fDecl, const std::string& classTypeName)
  {
    std::string fname  = fDecl->getNameInfo().getName().getAsString();
    std::string result = m_codeInfo->pShaderFuncRewriter->RewriteStdVectorTypeStr(fDecl->getReturnType().getAsString()) + " " + classTypeName + "_" + fname + "(\n  __global const " + classTypeName + "* self";
    if(fDecl->getNumParams() != 0)
      result += ", \n  ";

    bool isKernel = m_codeInfo->IsKernel(fname); 

    for(uint32_t i=0; i < fDecl->getNumParams(); i++)
    {
      const clang::ParmVarDecl* pParam  = fDecl->getParamDecl(i);
      const clang::QualType typeOfParam =	pParam->getType();

      if(typeOfParam.getAsString().find(m_mainClassName) != std::string::npos)
      {
        if(isPointerToConst(typeOfParam))
          result += "__global const struct ";
        else
          result += "__global struct ";
        
        result += m_codeInfo->mainClassName + m_codeInfo->mainClassSuffix + "_UBO_Data* " + pParam->getNameAsString();
      }
      else
      {
        if(typeOfParam->isPointerType() && isKernel)
          result += "__global ";
        result += kslicer::GetRangeSourceCode(pParam->getSourceRange(), m_compiler); 
      }

      if(i!=fDecl->getNumParams()-1)
        result += ", \n  ";
    }

    return result + ")\n  ";
  }

  std::unordered_set<std::string> ListFakeOffArgsForKernelNamed(const std::string& fname)
  {
    std::unordered_set<std::string> fakeOffsArgs;
    if(m_codeInfo->IsKernel(fname))     
    {
      auto p = m_codeInfo->kernels.find(fname);
      if(p != m_codeInfo->kernels.end())
      {
         for(const auto& arg : p->second.args)
         {
           if(arg.needFakeOffset)
             fakeOffsArgs.insert(arg.name);

           if(arg.isThreadID)
             m_fakeOffsetExp = arg.name; // TODO: if we have 2D thread id this is more complex a bit ... 
         }
      }
    }
    return fakeOffsArgs;
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
      if(m_codeInfo->IsKernel(fname))                          // enable fakeOffset rewrite
        m_fakeOffsArgs = ListFakeOffArgsForKernelNamed(fname); //
      std::string declSource = RewriteMemberDecl(fDecl, classTypeName);
      std::string bodySource = RecursiveRewrite(fDecl->getBody());
      m_fakeOffsArgs.clear();                                  // disable fakeOffset rewrite

      kslicer::MainClassInfo::DImplFunc funcData;
      funcData.decl          = fDecl;
      funcData.name          = fname;
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
      std::string textRes = classTypeName + "_" + fname;
      textRes += "(self";
      if(call->getNumArgs() > 0)
        textRes += ",";
      for(unsigned i=0;i<call->getNumArgs();i++)
      {
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

private:
    
  std::vector<kslicer::MainClassInfo::DImplFunc>& m_processed;
  std::vector<std::string>&                       m_fields;
  const std::string&                              m_className;
  const std::string&                              m_mainClassName;

  bool isCopy = false;
  std::unordered_set<std::string> m_fakeOffsArgs;
  std::string                     m_fakeOffsetExp;
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

void kslicer::MainClassInfo::AddVFH(const std::string& a_className)
{
  std::cout << " found class hierarchy: " << a_className.c_str() << std::endl;

  DHierarchy hdata;
  hdata.interfaceName = kslicer::CutOffStructClass(a_className);
  hdata.implementations.clear();
  m_vhierarchy[hdata.interfaceName] = hdata;
} 

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
    
    clang::Rewriter rewrite2;
    rewrite2.setSourceMgr(a_compiler.getSourceManager(), a_compiler.getLangOpts());

    for(const auto& decl : a_decls)
    {
      if(decl->isDerivedFrom(pBaseClass))
      {
        DImplClass dImpl;
        dImpl.decl = decl;
        dImpl.name = decl->getNameAsString();
        // extract all member functions from class that should be rewritten
        //
        MemberRewriter rv(rewrite2, a_compiler, this, dImpl); 
        rv.TraverseDecl(const_cast<clang::CXXRecordDecl*>(dImpl.decl));                                  
        
        dImpl.isEmpty = true;
        for(auto member : dImpl.memberFunctions)
        {
          if(!member.isEmpty)
          {
            dImpl.isEmpty = false;
            break;
          }
        }

        for(auto& k : kernels)
        {
          if(k.second.className == dImpl.name)
            k.second.interfaceName = className;
        }

        p.second.implementations.push_back(dImpl);
      }
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
