#include "kslicer.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"

#include <sstream>
#include <algorithm>

kslicer::KernelInfo::BlockExpansionInfo ExtractBlocks(const clang::CompoundStmt* kernelBody2, const clang::CompilerInstance& a_compiler)
{
  kslicer::KernelInfo::BlockExpansionInfo be;
  be.enabled = true;

  // iterate over whole code
  for(const clang::Stmt* child : kernelBody2->children())
  {
    if(clang::isa<clang::DeclStmt>(child)) 
    {
      be.sharedDecls.push_back(clang::dyn_cast<const clang::DeclStmt>(child));
    }
    else if(clang::isa<clang::ForStmt>(child))
    {
      const clang::ForStmt* forExpr = clang::dyn_cast<const clang::ForStmt>(child);
      std::string text = kslicer::GetRangeSourceCode(forExpr->getSourceRange(), a_compiler);
      kslicer::KernelInfo::BEBlock block;
      block.isParallel = (text.find("[[parallel]]") != std::string::npos);
      block.astNode    = child;
      block.forLoop    = forExpr;
      be.statements.push_back(block); 
    }
    else
    {
      kslicer::KernelInfo::BEBlock block;
      block.isParallel = false;
      block.astNode    = child;
      block.forLoop    = nullptr;
      be.statements.push_back(block); 
    }
  }

  return be;
}

void kslicer::MainClassInfo::ProcessBlockExpansionKernel(KernelInfo& a_kernel, const clang::CompilerInstance& a_compiler)
{
  auto kernelBody = a_kernel.loopIters[a_kernel.loopIters.size()-1].bodyNode;
  if(!clang::isa<clang::CompoundStmt>(kernelBody))
    return;
  
  const clang::CompoundStmt* kernelBody2 = clang::dyn_cast<clang::CompoundStmt>(kernelBody);
  std::cout << "  BlockExpansion: " << a_kernel.name.c_str() << std::endl;
  //kernelBody->dump();
  
  a_kernel.be = ExtractBlocks(kernelBody2, a_compiler);
}

std::string kslicer::IShaderCompiler::RewriteBESharedDecl(const clang::DeclStmt* decl, std::shared_ptr<KernelRewriter> pRewriter)
{
  return pRewriter->RecursiveRewrite(decl);
}

std::string kslicer::IShaderCompiler::RewriteBEParallelFor(const clang::ForStmt* forExpr, std::shared_ptr<KernelRewriter> pRewriter)
{
  const clang::Expr* cond = forExpr->getCond();
  const clang::Stmt* body = forExpr->getBody();

  if (cond == nullptr)
    return pRewriter->RecursiveRewrite(forExpr);
  
  // Получаем итератор цикла
  std::string loopInit = "";
  const clang::Stmt *Init = forExpr->getInit();
  if (const clang::DeclStmt *DeclStatement = clang::dyn_cast<clang::DeclStmt>(Init)) {
    for (const auto *Decl : DeclStatement->decls()) {
      if (const auto *VarDecl = clang::dyn_cast<clang::VarDecl>(Decl)) {
        clang::QualType Type = VarDecl->getType();
        std::string typeName = Type.getAsString();
        loopInit = typeName + " " + VarDecl->getNameAsString() + " = " + typeName + "(gl_LocalInvocationID[0]);\n";
      }
    }
  }

  std::string condText = loopInit + "  " + std::string("if(") + kslicer::GetRangeSourceCode(cond->getSourceRange(), pRewriter->GetCompiler()) + ")"; // pRewriter->RecursiveRewrite(cond)
    
  return condText + pRewriter->RecursiveRewrite(body);
  
  //if (!clang::isa<clang::BinaryOperator>(cond))
  //  return pRewriter->RecursiveRewrite(forExpr);
  //
  //const clang::BinaryOperator* binOp = clang::dyn_cast<clang::BinaryOperator>(cond);
  //const clang::Expr *LHS = binOp->getLHS();
  //const clang::Expr *RHS = binOp->getRHS();
  
  //auto astContext = pRewriter->GetCompiler().getASTContext();
  //
  //if(LHS->isIntegerConstantExpr(astContext)) 
  //{
  //  llvm::APInt LHSVal = LHS->EvaluateKnownConstInt(astContext);
  //} 
  //else
  //{
  //
  //}

  //return pRewriter->RecursiveRewrite(forExpr);
}

std::string kslicer::IShaderCompiler::RewriteBEStmt(const clang::Stmt* stmt, std::shared_ptr<KernelRewriter> pRewriter)
{
  return pRewriter->RecursiveRewrite(stmt) + ";";
}
