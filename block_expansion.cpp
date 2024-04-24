#include "kslicer.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"

#include <sstream>
#include <algorithm>

void kslicer::MainClassInfo::ProcessBlockExpansionKernel(KernelInfo& a_kernel)
{
  auto kernelBody = a_kernel.loopIters[a_kernel.loopIters.size()-1].bodyNode;
  if(!clang::isa<clang::CompoundStmt>(kernelBody))
    return;
  
  const clang::CompoundStmt* kernelBody2 = clang::dyn_cast<clang::CompoundStmt>(kernelBody);
  
  std::cout << "  BlockExpansion: " << a_kernel.name.c_str() << std::endl;
  kernelBody->dump();
  
  enum BEType{BE_SHARED_VARIABLE = 0, BE_PARALLEL_FOR = 1, BE_SINGLE = 2};
  struct BECode
  {
    BECode(){}
    BECode(const clang::Stmt* a_node, BEType a_type) : node(a_node), type(a_type) {}
    const clang::Stmt* node = nullptr;
    BEType             type = BE_SINGLE;
  };

  std::vector<BECode> sharedDecls;
  std::vector<BECode> blockOperators;

  // iterate over whole code
  for(const clang::Stmt* child : kernelBody2->children())
  {
    if(clang::isa<clang::DeclStmt>(child)) 
    {
      sharedDecls.push_back(BECode(child, BE_SHARED_VARIABLE));
    }
    else if(clang::isa<clang::ForStmt>(child))
    {
      const clang::ForStmt* forExpr = clang::dyn_cast<const clang::ForStmt>(child);
      //if(forExpr->hasAttrs())
      //{
      //  
      //}
      //const auto *AttrList = forExpr->getAttrs();
      //for (const auto *attr : AttrList) {
      //  std::cout << "  forAttr: " << attr->getName() << std::endl;
      //}
    }
  }
  
  int a = 2;

}
