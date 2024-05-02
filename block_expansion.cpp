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
}

std::string kslicer::IShaderCompiler::RewriteBEStmt(const clang::Stmt* stmt, std::shared_ptr<KernelRewriter> pRewriter)
{
  return pRewriter->RecursiveRewrite(stmt) + ";";
}

void ExtractTemplateParamsFromString(const std::string& input, int data[3]) 
{
  // Ищем позиции угловых скобок
  size_t start = input.find('<');
  size_t end = input.find('>');

  // Если скобки не найдены, выводим ошибку и выходим
  if (start == std::string::npos || end == std::string::npos) {
      std::cout << "Error: No angle brackets found!\n";
      return;
  }

  // Получаем строку между угловыми скобками
  std::string valuesStr = input.substr(start + 1, end - start - 1);

  // Используем строковый поток для разбиения строки на части
  std::istringstream iss(valuesStr);
  std::string token;
  std::vector<int> values;

  // Разбиваем строку по запятым и извлекаем числа
  while (std::getline(iss, token, ',')) {
      values.push_back(std::stoi(token));
  }

  // Проверяем, что количество чисел соответствует ожидаемому
  if (values.size() < 1 || values.size() > 3) {
      std::cerr << "Error: Invalid number of values between angle brackets!\n";
      return;
  }

  // Заполняем массив данными из вектора
  for (size_t i = 0; i < values.size(); ++i) {
      data[i] = values[i];
  }
}


void kslicer::ExtractBlockSizeFromCall(clang::CXXMemberCallExpr* f, 
                                       kslicer::KernelInfo& kernel, 
                                       const clang::CompilerInstance& compiler)
{
  const auto* methodDecl = f->getMethodDecl();
  if(methodDecl == nullptr)
    return;
  
  // (1) extract template argument names
  //
  int top = 0;
  auto templateDecl = methodDecl->getPrimaryTemplate();
  const clang::TemplateParameterList* TPL = templateDecl->getTemplateParameters();
  for (const clang::NamedDecl* arg : *TPL)  
  {
    //auto qt = arg->getDeclName().getCXXNameType(); 
    kernel.be.wgNames[top] = arg->getNameAsString();
    //kernel.be.wgTypes[top] = qt.getAsString(); 
    top++;
    if(top >= 2)
      break;
  }
  
  // (2) extract actual template argument names
  //
  std::string callText = kslicer::GetRangeSourceCode(f->getSourceRange(), compiler);
  int wgSize[3] = {1,1,1};
  ExtractTemplateParamsFromString(callText, wgSize);

  uint32_t kernelDim = kernel.GetDim();
  assert(kernelDim == top);

  for(int i=0;i<top;i++)
    kernel.wgSize[i] = wgSize[i];
}
