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

    mainFuncCmdName = fname + "Cmd";
    m_rewriter.ReplaceText(dni.getSourceRange(), mainFuncCmdName);
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

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to) 
{
  size_t start_pos = str.find(from);
  if(start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

std::string kslicer::ProcessMainFunc(const CXXMethodDecl* a_node, clang::CompilerInstance& compiler, const std::string& a_mainClassName,
                                     std::string& a_outFuncDecl)
{
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  kslicer::MainFuncASTVisitor rv(rewrite2);
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  
  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  std::string sourceCode = rewrite2.getRewrittenText(clang::SourceRange(b,e));
  
  // (1) TestClass::MainFuncCmd --> TestClass_Generated::MainFuncCmd
  // 
  const std::string replaceFrom = a_mainClassName + "::" + rv.mainFuncCmdName;
  const std::string replaceTo   = a_mainClassName + "_Generated" + "::" + rv.mainFuncCmdName;

  assert(ReplaceFirst(sourceCode, replaceFrom, replaceTo));

  // (2) add input command Buffer as first argument
  //
  {
    size_t roundBracketPos = sourceCode.find("(");
    sourceCode = (sourceCode.substr(0, roundBracketPos) + "(VkCommandBuffer a_commandBuffer, " + sourceCode.substr(roundBracketPos+2)); 
  }

  // (3) set m_currCmdBuffer with input command bufer
  //
  {
    size_t bracePos = sourceCode.find("{");
    sourceCode = (sourceCode.substr(0, bracePos) + "{\n  m_currCmdBuffer = a_commandBuffer; \n\n" + sourceCode.substr(bracePos+2)); 
  }

  // (4) get function decl from full function code
  //
  std::string mainFuncDecl = sourceCode.substr(0, sourceCode.find(")")+1) + ";";
  assert(ReplaceFirst(mainFuncDecl, a_mainClassName + "_Generated" + "::", ""));

  a_outFuncDecl = "virtual " + mainFuncDecl;
  return sourceCode;
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
  strOut << "(__global const " << fieldType.c_str() << "*)" << "(" << buffName.c_str() << "+" << (p->second.offsetInTargetBuffer/sizeof(uint32_t)) << ")";
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


std::vector<std::string> kslicer::ObtainKernelsDecl(const std::vector<kslicer::KernelInfo>& a_kernelsData, clang::SourceManager& sm, const std::string& a_mainClassName)
{
  std::vector<std::string> kernelsCallCmdDecl;
  for (const auto& k : a_kernelsData)  
  {
    assert(k.astNode != nullptr);
    auto sourceRange = k.astNode->getSourceRange();
    std::string kernelSourceCode = GetRangeSourceCode(sourceRange, sm);
    
    std::string kernelCmdDecl = kernelSourceCode.substr(0, kernelSourceCode.find(")")+1);
    assert(ReplaceFirst(kernelCmdDecl, a_mainClassName + "::", ""));
    assert(ReplaceFirst(kernelCmdDecl,"kernel_", ""));
    assert(ReplaceFirst(kernelCmdDecl,"(", "Cmd("));
    kernelsCallCmdDecl.push_back(kernelCmdDecl);
  }
  return kernelsCallCmdDecl;
}
