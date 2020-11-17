#include "class_gen.h"
#include "kslicer.h"

bool kslicer::MainFuncASTVisitor::VisitCXXMethodDecl(CXXMethodDecl* f) 
{
  if (f->hasBody())
  {
    // Get name of function
    const DeclarationNameInfo dni = f->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();

    m_rewriter.ReplaceText(dni.getSourceRange(), fname + "Cmd" );
  }

  return true; // returning false aborts the traversal
}

bool kslicer::MainFuncASTVisitor::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
{
  // Get name of function
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  auto p = fname.find("kernel_");
  if(p != std::string::npos)
  {
    std::string kernName = fname.substr(p + 7);
    m_rewriter.ReplaceText(f->getExprLoc(), kernName + "Cmd");
  }

  return true; 
}

std::string kslicer::ProcessMainFunc(const CXXMethodDecl* a_node, clang::CompilerInstance& compiler)
{
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  kslicer::MainFuncASTVisitor rv(rewrite2);
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  
  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  return rewrite2.getRewrittenText(clang::SourceRange(b,e));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::KernelReplacerASTVisitor::VisitMemberExpr(MemberExpr* expr)
{
  ValueDecl* pValueDecl =	expr->getMemberDecl();
  
  if(!isa<FieldDecl>(pValueDecl))
    return true;

  FieldDecl* pFieldDecl  = dyn_cast<FieldDecl>(pValueDecl);
  assert(pFieldDecl != nullptr);
  RecordDecl* pRecodDecl = pFieldDecl->getParent();
  assert(pRecodDecl != nullptr);

  const std::string thisTypeName = pRecodDecl->getNameAsString();
  if(thisTypeName != m_mainClassName)
    return true;
 
  // (1) get type of variable itself because we need to cast pointer to this type
  //
  QualType qt = pFieldDecl->getTypeSourceInfo()->getType();
  std::string fieldType = qt.getAsString();
  kslicer::ReplaceOpenCLBuiltInTypes(fieldType);

  // (2) get variable offset in buffer by its name 
  //
  const std::string fieldName = pFieldDecl->getNameAsString(); 
  const auto p = m_variables.find(fieldName);
  if(p == m_variables.end())
    return true;

  // (3) put *(pointer+offset) instead of variable name
  //
  const std::string buffName = kslicer::GetProjPrefix() + "data"; 
  std::stringstream strOut;
  strOut << "*(  "; 
  strOut << "(const " << fieldType.c_str() << "*)" << "(" << buffName.c_str() << "+" << (p->second.offsetInTargetBuffer/sizeof(uint32_t)) << ")";
  strOut << "  )";
  
  m_rewriter.ReplaceText(expr->getExprLoc(), strOut.str());
  
  return true;
}

std::string kslicer::ProcessKernel(const CXXMethodDecl* a_node, clang::CompilerInstance& compiler, const kslicer::MainClassInfo& a_codeInfo)
{
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  kslicer::KernelReplacerASTVisitor rv(rewrite2, a_codeInfo.mainClassName, a_codeInfo.classVariables);
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  
  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  return rewrite2.getRewrittenText(clang::SourceRange(b,e));
}

