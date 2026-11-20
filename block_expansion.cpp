#include "kslicer.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"

#include <sstream>
#include <algorithm>

namespace
{
  // '[[parallel]]' is written between ')' of the loop header and its body, so only this part of the text is checked
  //
  bool IsParallelFor(const clang::Stmt* a_stmt, const clang::CompilerInstance& a_compiler)
  {
    auto pFor = clang::dyn_cast_or_null<clang::ForStmt>(a_stmt);
    if(pFor == nullptr)
      return false;
    const clang::SourceRange header(pFor->getBeginLoc(), pFor->getBody()->getBeginLoc());
    const std::string text = kslicer::GetRangeSourceCode(header, a_compiler);
    return (text.find("[[parallel]]") != std::string::npos);
  }

  // 2D parallel loop: the body of a parallel loop is another parallel loop, 'for(y) [[parallel]] for(x) [[parallel]]' or 'for(y) [[parallel]] { for(x) [[parallel]] }'
  //
  const clang::ForStmt* InnerParallelFor(const clang::ForStmt* a_for, const clang::CompilerInstance& a_compiler)
  {
    const clang::Stmt* body = a_for->getBody();
    if(auto pBlock = clang::dyn_cast<clang::CompoundStmt>(body); pBlock != nullptr && pBlock->size() == 1)
      body = pBlock->body_front();
    return IsParallelFor(body, a_compiler) ? clang::dyn_cast<clang::ForStmt>(body) : nullptr;
  }

  bool IsStaticAssert(const clang::Stmt* a_stmt)
  {
    auto pDecl = clang::dyn_cast<clang::DeclStmt>(a_stmt);
    if(pDecl == nullptr)
      return false;
    for(const clang::Decl* decl : pDecl->decls())
      if(!clang::isa<clang::StaticAssertDecl>(decl))
        return false;
    return true;
  }

  class ParallelForFinder : public clang::RecursiveASTVisitor<ParallelForFinder>
  {
  public:
    ParallelForFinder(const clang::CompilerInstance& a_compiler) : m_compiler(a_compiler) {}
    bool VisitForStmt(clang::ForStmt* a_for)
    {
      found = found || IsParallelFor(a_for, m_compiler);
      return !found;
    }
    bool found = false;
  private:
    const clang::CompilerInstance& m_compiler;
  };

  bool ContainsParallelFor(const clang::Stmt* a_stmt, const clang::CompilerInstance& a_compiler)
  {
    ParallelForFinder finder(a_compiler);
    finder.TraverseStmt(const_cast<clang::Stmt*>(a_stmt));
    return finder.found;
  }

  // 'LiteNN::LinearForwardBlock(...)' and other block level functions: executed by all threads of the group together
  //
  const clang::CallExpr* BlockFunctionCall(const clang::Stmt* a_stmt)
  {
    auto pCall = clang::dyn_cast_or_null<clang::CallExpr>(a_stmt);
    const clang::FunctionDecl* pFunc = (pCall == nullptr) ? nullptr : pCall->getDirectCallee();
    if(pFunc == nullptr)
      return nullptr;
    const std::string name = pFunc->getQualifiedNameAsString();
    const bool isBlock = name.rfind("LiteNN::", 0) == 0 && name.size() > 5 && name.compare(name.size() - 5, 5, "Block") == 0;
    return isBlock ? pCall : nullptr;
  }

  class BlockFunctionFinder : public clang::RecursiveASTVisitor<BlockFunctionFinder>
  {
  public:
    bool VisitCallExpr(clang::CallExpr* a_call)
    {
      if(found == nullptr && BlockFunctionCall(a_call) != nullptr)
        found = a_call;
      return true;
    }
    const clang::CallExpr* found = nullptr;
  };

  const clang::CallExpr* FindBlockFunction(const clang::Stmt* a_stmt)
  {
    BlockFunctionFinder finder;
    finder.TraverseStmt(const_cast<clang::Stmt*>(a_stmt));
    return finder.found;
  }

  // functions of the program called directly: not methods and not functions of libraries
  //
  const clang::FunctionDecl* UserCallee(const clang::CallExpr* a_call)
  {
    const clang::FunctionDecl* pFunc = a_call->getDirectCallee();
    if(pFunc == nullptr || clang::isa<clang::CXXMethodDecl>(pFunc) || !pFunc->hasBody())
      return nullptr;
    const std::string name = pFunc->getQualifiedNameAsString();
    for(const char* ns : {"std::", "LiteMath::", "LiteImage::", "LiteNN::", "LiteRS::"})
      if(name.rfind(ns, 0) == 0)
        return nullptr;
    return pFunc;
  }

  class CalleesVisitor : public clang::RecursiveASTVisitor<CalleesVisitor>
  {
  public:
    bool VisitCallExpr(clang::CallExpr* a_call)
    {
      if(auto pFunc = UserCallee(a_call))
        callees.push_back(pFunc);
      return true;
    }
    std::vector<const clang::FunctionDecl*> callees;
  };

  // variables of the block (all local variables except of those declared inside parallel loops) and functions called from the same code
  //
  class BlockVarsVisitor : public clang::RecursiveASTVisitor<BlockVarsVisitor>
  {
  public:
    BlockVarsVisitor(kslicer::uset<const clang::VarDecl*>& a_vars, const clang::CompilerInstance& a_compiler) : m_vars(a_vars), m_compiler(a_compiler) {}
    bool TraverseForStmt(clang::ForStmt* a_for)
    {
      if(IsParallelFor(a_for, m_compiler))
        return true;
      return clang::RecursiveASTVisitor<BlockVarsVisitor>::TraverseForStmt(a_for);
    }
    bool VisitVarDecl(clang::VarDecl* a_decl) { m_vars.insert(a_decl); return true; }
    bool VisitCallExpr(clang::CallExpr* a_call)
    {
      if(auto pFunc = UserCallee(a_call))
        funcs.push_back(pFunc);
      return true;
    }
    std::vector<const clang::FunctionDecl*> funcs;
  private:
    kslicer::uset<const clang::VarDecl*>& m_vars;
    const clang::CompilerInstance&        m_compiler;
  };

  // what a statement of the block (without parallel loops inside) writes
  //
  class EffectsVisitor : public clang::RecursiveASTVisitor<EffectsVisitor>
  {
  public:
    EffectsVisitor(const kslicer::KernelInfo::BlockExpansionInfo& a_be) : m_be(a_be) {}

    bool writesShared = false;              ///<! writes shared memory of the block
    bool writesMemory = false;              ///<! writes any memory: shared, buffers, class members
    const clang::VarDecl* blockVar = nullptr; ///<! changed variable of the block declared outside of the statement

    bool VisitVarDecl(clang::VarDecl* a_decl) { m_inner.insert(a_decl); return true; }

    bool VisitBinaryOperator(clang::BinaryOperator* a_op)
    {
      if(a_op->isAssignmentOp())
        Write(a_op->getLHS());
      return true;
    }

    bool VisitUnaryOperator(clang::UnaryOperator* a_op)
    {
      if(a_op->isIncrementDecrementOp())
        Write(a_op->getSubExpr());
      return true;
    }

    bool VisitCallExpr(clang::CallExpr* a_call)
    {
      if(auto pOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(a_call))
      {
        const auto op = pOp->getOperator();
        if(pOp->isAssignmentOp() || op == clang::OO_PlusPlus || op == clang::OO_MinusMinus)
          Write(pOp->getArg(0));
        return true;
      }

      if(auto pMember = clang::dyn_cast<clang::CXXMemberCallExpr>(a_call))     // 'm_vec.push_back(x)', 'this->Func()'
      {
        const clang::CXXMethodDecl* pMethod = pMember->getMethodDecl();
        if(pMethod != nullptr && !pMethod->isConst() && !pMethod->isStatic())
          Write(pMember->getImplicitObjectArgument());
      }

      const clang::FunctionDecl* pFunc = a_call->getDirectCallee();            // arguments passed by pointer or by reference
      for(unsigned i = 0; i < a_call->getNumArgs(); i++)
      {
        const clang::QualType type = (pFunc != nullptr && i < pFunc->getNumParams()) ? pFunc->getParamDecl(i)->getType() : a_call->getArg(i)->getType();
        if(type->isPointerType() && !type->getPointeeType().isConstQualified())
          Write(a_call->getArg(i));
        else if(type->isReferenceType() && !type.getNonReferenceType().isConstQualified())
          Write(a_call->getArg(i));
      }
      return true;
    }

  private:

    void Write(const clang::Expr* a_expr)
    {
      const clang::Expr* expr = a_expr->IgnoreParenImpCasts();
      bool indirect = false;                                     // written via subscript or pointer
      while(true)
      {
        if(auto p = clang::dyn_cast<clang::ArraySubscriptExpr>(expr))
        {
          expr     = p->getBase()->IgnoreParenImpCasts();
          indirect = true;
        }
        else if(auto p = clang::dyn_cast<clang::MemberExpr>(expr))
        {
          indirect = indirect || p->isArrow();
          expr     = p->getBase()->IgnoreParenImpCasts();
        }
        else if(auto p = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr); p != nullptr && p->getOperator() == clang::OO_Subscript)
        {
          expr     = p->getArg(0)->IgnoreParenImpCasts();
          indirect = true;
        }
        else if(auto p = clang::dyn_cast<clang::UnaryOperator>(expr); p != nullptr && (p->getOpcode() == clang::UO_Deref || p->getOpcode() == clang::UO_AddrOf))
        {
          indirect = indirect || (p->getOpcode() == clang::UO_Deref);
          expr     = p->getSubExpr()->IgnoreParenImpCasts();
        }
        else
          break;
      }

      auto pRef = clang::dyn_cast<clang::DeclRefExpr>(expr);
      auto pVar = (pRef == nullptr) ? nullptr : clang::dyn_cast<clang::VarDecl>(pRef->getDecl());
      if(pVar == nullptr || pVar->getType()->isReferenceType() || (indirect && pVar->getType()->isPointerType())) // class members, buffers, references
      {
        writesMemory = true;
        return;
      }
      if(m_be.sharedVars.contains(pVar))
      {
        writesShared = true;
        writesMemory = true;
      }
      else if(m_inner.contains(pVar))                            // local variable of the statement itself
        return;
      else if(m_be.uniformVars.contains(pVar) || clang::isa<clang::ParmVarDecl>(pVar))
      {
        if(blockVar == nullptr)
          blockVar = pVar;
      }
      else
        writesMemory = true;
    }

    const kslicer::KernelInfo::BlockExpansionInfo& m_be;
    kslicer::uset<const clang::VarDecl*>           m_inner;
  };

  // lines of 'a_text' after the first one are shifted so that 'a_column' (column of the first line in the source) becomes 'a_indent'
  //
  std::vector<std::string> Reindent(const std::string& a_text, unsigned a_column, const std::string& a_indent)
  {
    std::vector<std::string> lines;
    std::istringstream input(a_text);
    std::string line;
    while(std::getline(input, line))
    {
      if(!lines.empty())
      {
        size_t spaces = 0;
        while(spaces < line.size() && spaces < a_column && (line[spaces] == ' ' || line[spaces] == '\t'))
          spaces++;
        line = line.substr(spaces);
      }
      while(!line.empty() && (line.back() == ' ' || line.back() == '\r'))
        line.pop_back();
      lines.push_back(line.empty() ? line : a_indent + line);
    }
    return lines;
  }

  // Code of a block. Barriers and the condition of a single thread are given by the back end:
  // (1) parallel loops are executed by all threads of the group, with barriers before and after them;
  // (2) ordinary loops and 'if' with parallel loops inside are executed by all threads (their conditions must be uniform);
  // (3) other statements that write memory are executed by thread 0, barriers are added if they write shared memory;
  // (4) other statements are executed by all threads, every thread has its own copy of the variables of the block.
  //
  class BEBodyGenerator
  {
  public:
    BEBodyGenerator(kslicer::IShaderCompiler* a_pCC, const kslicer::KernelInfo& a_kernel, std::shared_ptr<kslicer::KernelRewriter> a_pRewriter, const std::string& a_indent,
                    const uint32_t a_threadsOrder[3]) :
                    m_pCC(a_pCC), m_kernel(a_kernel), m_pRewriter(a_pRewriter), m_compiler(a_pRewriter->GetCompiler()), m_indent(a_indent)
    {
      m_barrier = a_pCC->BEBarrier();
      m_single  = a_pCC->BESingleThreadCond(a_kernel);
      for(uint32_t dim = 0; dim < 2; dim++)                  // as for 'ThreadIds' of IPV kernels: dimension 'dim' gets the loop of level 'a_threadsOrder[dim]'
        m_dimOfLevel[std::min<uint32_t>(a_threadsOrder[dim], 1)] = dim;
      if(m_dimOfLevel[0] == m_dimOfLevel[1])
      {
        m_dimOfLevel[0] = 0;
        m_dimOfLevel[1] = 1;
      }
    }

    void Generate(const clang::CompoundStmt* a_body)
    {
      for(const clang::Stmt* child : a_body->children())
      {
        auto pDecl = clang::dyn_cast<clang::DeclStmt>(child);
        if(pDecl != nullptr && std::find(m_kernel.be.sharedDecls.begin(), m_kernel.be.sharedDecls.end(), pDecl) != m_kernel.be.sharedDecls.end())
          continue;                                              // shared memory is declared by the template
        Statement(child);
      }
    }

    std::string Text() const
    {
      std::string res;
      for(const auto& line : m_lines)
        res += line + "\n";
      return res;
    }

  private:

    void Line(const std::string& a_line) { m_lines.push_back(m_indent + a_line); }

    void Lines(const std::string& a_text, const clang::Stmt* a_stmt)
    {
      const unsigned column = m_compiler.getSourceManager().getSpellingColumnNumber(a_stmt->getBeginLoc()) - 1;
      for(const auto& line : Reindent(a_text, column, m_indent))
        m_lines.push_back(line);
    }

    void LinesRaw(const std::string& a_text)                  // generated text, its own indentation from column 0
    {
      std::istringstream input(a_text);
      std::string line;
      while(std::getline(input, line))
        m_lines.push_back(line.empty() ? line : m_indent + line);
    }

    void Barrier()
    {
      if(m_barrier == "" || (!m_lines.empty() && m_lines.back() == m_indent + m_barrier))
        return;
      Line(m_barrier);
    }

    std::string Rewrite(const clang::Stmt* a_stmt) { return (a_stmt == nullptr) ? std::string("") : m_pRewriter->RecursiveRewrite(a_stmt); }

    void Error(const std::string& a_msg, const clang::Stmt* a_stmt) { kslicer::PrintError("block expansion: " + a_msg, a_stmt->getSourceRange(), m_compiler.getSourceManager()); }

    void Body(const clang::Stmt* a_stmt)
    {
      if(clang::isa<clang::CompoundStmt>(a_stmt))
      {
        Statement(a_stmt);
        return;
      }
      Line("{");
      Statement(a_stmt);
      Line("}");
    }

    void Statement(const clang::Stmt* a_stmt)
    {
      if(auto pBlock = clang::dyn_cast<clang::CompoundStmt>(a_stmt))
      {
        if(ContainsParallelFor(a_stmt, m_compiler))
        {
          Line("{");
          const std::string oldIndent = m_indent;
          m_indent += "  ";
          for(const clang::Stmt* child : pBlock->children())
            Statement(child);
          m_indent = oldIndent;
          Line("}");
          return;
        }
      }
      else if(IsParallelFor(a_stmt, m_compiler))
      {
        auto pFor   = clang::dyn_cast<clang::ForStmt>(a_stmt);
        auto pInner = InnerParallelFor(pFor, m_compiler);
        if(pInner != nullptr && m_kernel.GetDim() < 2)
          Error("2D parallel loop in 1D block expansion kernel, use 'kernelBE2D_'", a_stmt);
        const clang::Stmt* body = (pInner != nullptr) ? pInner->getBody() : pFor->getBody();
        if(ContainsParallelFor(body, m_compiler))
          Error("'[[parallel]]' loop inside a parallel loop is supported only as 2D parallel loop: two perfectly nested '[[parallel]]' loops", body);
        if(auto pCall = FindBlockFunction(body))
          Error("block level function must be called from the code of the block, not from a parallel loop", pCall);
        Barrier();
        if(pInner != nullptr)
          Lines(m_pCC->RewriteBEParallelFor2D(pFor, pInner, m_pRewriter, m_kernel, m_dimOfLevel), a_stmt);
        else
          Lines(m_pCC->RewriteBEParallelFor(pFor, m_pRewriter, m_kernel), a_stmt);
        Barrier();
        return;
      }
      else if(ContainsParallelFor(a_stmt, m_compiler))         // (2) ordinary loops and 'if', executed by all threads
      {
        if(auto pFor = clang::dyn_cast<clang::ForStmt>(a_stmt))
        {
          std::string init = Rewrite(pFor->getInit());
          if(init.empty() || init.back() != ';')
            init += ";";
          Line("for(" + init + " " + Rewrite(pFor->getCond()) + "; " + Rewrite(pFor->getInc()) + ")");
          Body(pFor->getBody());
        }
        else if(auto pWhile = clang::dyn_cast<clang::WhileStmt>(a_stmt))
        {
          Line("while(" + Rewrite(pWhile->getCond()) + ")");
          Body(pWhile->getBody());
        }
        else if(auto pDo = clang::dyn_cast<clang::DoStmt>(a_stmt))
        {
          Line("do");
          Body(pDo->getBody());
          Line("while(" + Rewrite(pDo->getCond()) + ");");
        }
        else if(auto pIf = clang::dyn_cast<clang::IfStmt>(a_stmt))
        {
          if(pIf->getInit() != nullptr || pIf->getConditionVariable() != nullptr)
            Error("'if' with a parallel loop inside must not declare variables in its condition", a_stmt);
          Line("if(" + Rewrite(pIf->getCond()) + ")");
          Body(pIf->getThen());
          if(pIf->getElse() != nullptr)
          {
            Line("else");
            Body(pIf->getElse());
          }
        }
        else
          Error("a parallel loop may be placed only inside 'for', 'while', 'do', 'if' and '{}'", a_stmt);
        return;
      }

      if(IsStaticAssert(a_stmt))                               // checked by C++ compiler
        return;
      if(auto pCall = BlockFunctionCall(a_stmt))               // block level function: all threads, reads and writes shared memory
      {
        const std::string text = m_pCC->RewriteBEBlockFunction(pCall, m_pRewriter, m_kernel);
        if(text == "")
        {
          Error("'" + pCall->getDirectCallee()->getQualifiedNameAsString() + "' is not supported by this back end", a_stmt);
          return;
        }
        Barrier();
        LinesRaw(text);
        Barrier();
        return;
      }
      if(auto pCall = FindBlockFunction(a_stmt))
        Error("block level function must be called as a separate statement", pCall);

      EffectsVisitor effects(m_kernel.be);
      effects.TraverseStmt(const_cast<clang::Stmt*>(a_stmt));
      const std::string text = m_pCC->RewriteBEStmt(a_stmt, m_pRewriter);
      if(!effects.writesMemory || m_single == "")             // (4) all threads, or ISPC uniform code
      {
        Lines(text, a_stmt);
        return;
      }

      if(effects.blockVar != nullptr)
        Error("statement writes memory and changes variable '" + effects.blockVar->getNameAsString() + "' of the block. It is executed by one thread of the group, so split it into two statements", a_stmt);

      if(effects.writesShared)                                  // (3) thread 0
        Barrier();
      Line("if(" + m_single + ")");
      Line("{");
      const std::string oldIndent = m_indent;
      m_indent += "  ";
      Lines(text, a_stmt);
      m_indent = oldIndent;
      Line("}");
      if(effects.writesShared)
        Barrier();
    }

    kslicer::IShaderCompiler*               m_pCC;
    const kslicer::KernelInfo&              m_kernel;
    std::shared_ptr<kslicer::KernelRewriter> m_pRewriter;
    const clang::CompilerInstance&          m_compiler;
    std::string                             m_indent;
    std::string                             m_barrier;
    std::string                             m_single;
    std::vector<std::string>                m_lines;
    uint32_t                                m_dimOfLevel[2] = {0, 1}; ///<! 2D parallel loop: dimension of the group for the outer and the inner loop
  };
}

kslicer::KernelInfo::BlockExpansionInfo ExtractBlocks(const clang::CompoundStmt* a_body, const clang::CompilerInstance& a_compiler)
{
  kslicer::KernelInfo::BlockExpansionInfo be;
  be.enabled = true;
  be.body    = a_body;

  BlockVarsVisitor blockVars(be.uniformVars, a_compiler);
  blockVars.TraverseStmt(const_cast<clang::CompoundStmt*>(a_body));

  // functions called from the code of the block and functions called from them
  //
  std::vector<const clang::FunctionDecl*> stack = blockVars.funcs;
  while(!stack.empty())
  {
    const clang::FunctionDecl* pFunc = stack.back();
    stack.pop_back();
    if(!be.uniformFuncs.insert(pFunc->getNameAsString()).second)
      continue;
    CalleesVisitor callees;
    callees.TraverseStmt(pFunc->getBody());
    stack.insert(stack.end(), callees.callees.begin(), callees.callees.end());
  }

  // arrays at the top level of the block are placed in shared memory
  //
  for(const clang::Stmt* child : a_body->children())
  {
    auto pDecl = clang::dyn_cast<clang::DeclStmt>(child);
    if(pDecl == nullptr)
      continue;
    bool allArrays = true;
    for(const clang::Decl* decl : pDecl->decls())
    {
      auto pVar = clang::dyn_cast<clang::VarDecl>(decl);
      allArrays = allArrays && (pVar != nullptr) && pVar->getType()->isArrayType();
    }
    if(!allArrays)
      continue;
    be.sharedDecls.push_back(pDecl);
    for(const clang::Decl* decl : pDecl->decls())
    {
      auto pVar = clang::dyn_cast<clang::VarDecl>(decl);
      be.sharedVars.insert(pVar);
      auto pCtor = clang::dyn_cast_or_null<clang::CXXConstructExpr>(pVar->getInit()); // 'half a[N]': implicit default constructor, not an initializer
      if(pVar->hasInit() && !(pCtor != nullptr && pCtor->getNumArgs() == 0))
        kslicer::PrintError("block expansion: arrays of the block are placed in shared memory and can't be initialized", pDecl->getSourceRange(), a_compiler.getSourceManager());
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
  
  a_kernel.be = ExtractBlocks(kernelBody2, a_compiler);
  a_kernel.be.coopMatrix = this->coopMatrix;

  if(FindBlockFunction(kernelBody2) != nullptr)            // LiteNN block functions: fp16 in shared memory and buffers, cooperative matrices on request
  {
    globalShaderFeatures.useCoopMat      = globalShaderFeatures.useCoopMat || this->coopMatrix;
    globalShaderFeatures.useHalfType     = true;
    globalShaderFeatures.use16BitStorage = true;
  }
}

bool kslicer::ParseLinearForwardBlock(const clang::CallExpr* a_call, std::shared_ptr<KernelRewriter> pRewriter, const KernelInfo& a_kernel, BELinearForwardArgs* a_args)
{
  const clang::CompilerInstance& compiler = pRewriter->GetCompiler();
  const clang::SourceManager&    sm       = compiler.getSourceManager();
  if(a_call->getDirectCallee()->getNameAsString() != "LinearForwardBlock" || a_call->getNumArgs() != 8)
  {
    kslicer::PrintError("unknown block level function", a_call->getSourceRange(), sm);
    return false;
  }

  bool ok = true;
  auto ArrayOfBlock = [&](const clang::Expr* a_expr)
  {
    auto pRef = clang::dyn_cast<clang::DeclRefExpr>(a_expr->IgnoreParenImpCasts());
    auto pVar = (pRef == nullptr) ? nullptr : clang::dyn_cast<clang::VarDecl>(pRef->getDecl());
    if(pVar == nullptr || !a_kernel.be.sharedVars.contains(pVar))
    {
      kslicer::PrintError("LiteNN::LinearForwardBlock: input and output must be arrays of the block (declared at the top level of the block)", a_expr->getSourceRange(), sm);
      ok = false;
      return std::string("");
    }
    return pVar->getNameAsString();
  };
  auto Dimension = [&](const clang::Expr* a_expr)
  {
    auto value = a_expr->getIntegerConstantExpr(compiler.getASTContext());
    if(!value || value->getExtValue() <= 0 || value->getExtValue() % 16 != 0)
    {
      kslicer::PrintError("LiteNN::LinearForwardBlock: sizes must be compile time constants, multiples of 16 (LiteNN::BLOCK_TILE)", a_expr->getSourceRange(), sm);
      ok = false;
      return uint32_t(16);
    }
    return uint32_t(value->getExtValue());
  };

  a_args->weights  = pRewriter->RecursiveRewrite(a_call->getArg(0));
  a_args->matOffs  = pRewriter->RecursiveRewrite(a_call->getArg(1));
  a_args->biasOffs = pRewriter->RecursiveRewrite(a_call->getArg(2));
  a_args->in       = ArrayOfBlock(a_call->getArg(3));
  a_args->inDim    = Dimension(a_call->getArg(4));
  a_args->out      = ArrayOfBlock(a_call->getArg(5));
  a_args->outDim   = Dimension(a_call->getArg(6));
  a_args->batch    = Dimension(a_call->getArg(7));
  a_args->stride   = (a_args->inDim + 7) / 8 * 8;          // LiteNN::RowStride
  if(ok && a_args->in == a_args->out)
  {
    kslicer::PrintError("LiteNN::LinearForwardBlock: input and output must be different arrays", a_call->getSourceRange(), sm);
    ok = false;
  }
  return ok;
}

std::string kslicer::IShaderCompiler::RewriteBESharedDecl(const clang::DeclStmt* decl, std::shared_ptr<KernelRewriter> pRewriter)
{
  return pRewriter->RecursiveRewrite(decl);
}

namespace
{
  struct BEParallelLoop
  {
    const clang::VarDecl* var = nullptr;
    std::string type, name, start, cond;
  };

  // 'for(T i = start; cond; ...)': loop variable, its type and start, condition
  //
  bool ParseParallelFor(const clang::ForStmt* a_for, std::shared_ptr<kslicer::KernelRewriter> pRewriter, BEParallelLoop* a_loop)
  {
    if(auto pDeclStmt = clang::dyn_cast_or_null<clang::DeclStmt>(a_for->getInit()))
      if(pDeclStmt->isSingleDecl())
        a_loop->var = clang::dyn_cast<clang::VarDecl>(pDeclStmt->getSingleDecl());

    const clang::CompilerInstance& compiler = pRewriter->GetCompiler();
    if(a_loop->var == nullptr || a_loop->var->getInit() == nullptr || a_for->getCond() == nullptr)
    {
      kslicer::PrintError("parallel loop of block expansion kernel must have the form 'for(T i = start; cond; i++)'", a_for->getSourceRange(), compiler.getSourceManager());
      return false;
    }
    a_loop->type = a_loop->var->getType().getUnqualifiedType().getAsString();
    ReplaceFirst(a_loop->type, "LiteMath::", "");
    a_loop->name  = a_loop->var->getNameAsString();
    a_loop->start = pRewriter->RecursiveRewrite(a_loop->var->getInit());
    a_loop->cond  = pRewriter->RecursiveRewrite(a_for->getCond());
    return true;
  }

  // body of a loop for the text of a generated loop: lines after the first one keep their indentation relative to column of the loop
  //
  std::string LoopBody(const clang::Stmt* a_body, std::shared_ptr<kslicer::KernelRewriter> pRewriter)
  {
    std::string bodyText = pRewriter->RecursiveRewrite(a_body);
    const unsigned column = pRewriter->GetCompiler().getSourceManager().getSpellingColumnNumber(a_body->getBeginLoc()) - 1;
    if(!clang::isa<clang::CompoundStmt>(a_body))       // range of a single statement does not include ';'
      bodyText += ";";
    return std::string(column, ' ') + bodyText;
  }
}

std::string kslicer::IShaderCompiler::BELocalId(uint32_t a_dim, const KernelInfo& a_kernel) const
{
  uint32_t wgSize[3] = {a_kernel.wgSize[0], a_kernel.wgSize[1], a_kernel.wgSize[2]};
  std::string res = LocalIdExpr(1, wgSize);              // 'a_localTID.x', 'gl_LocalInvocationID.x'
  if(res.size() > 2 && res.compare(res.size() - 2, 2, ".x") == 0)
    res[res.size() - 1] = char('x' + a_dim);
  return res;
}

std::string kslicer::IShaderCompiler::BELinearLocalId(const KernelInfo& a_kernel) const
{
  uint32_t wgSize[3] = {a_kernel.wgSize[0], a_kernel.wgSize[1], a_kernel.wgSize[2]};
  return LocalIdExpr(a_kernel.GetDim(), wgSize);
}

std::string kslicer::IShaderCompiler::BEGroupSize(const KernelInfo& a_kernel) const
{
  std::string res = a_kernel.be.wgNames[0];
  for(uint32_t dim = 1; dim < a_kernel.GetDim() && dim < 3; dim++)
    res += "*" + a_kernel.be.wgNames[dim];
  return res;
}

// 'for(T i = start; cond; inc) [[parallel]] body' ==> 'for(T i = T(localId) + start; cond; i += groupSize) body':
// each thread of the group executes iterations 'localId', 'localId + groupSize', ... ('localId' is linear for 2D group)
//
std::string kslicer::IShaderCompiler::RewriteBEParallelFor(const clang::ForStmt* forExpr, std::shared_ptr<KernelRewriter> pRewriter, const KernelInfo& a_kernel)
{
  BEParallelLoop loop;
  if(!ParseParallelFor(forExpr, pRewriter, &loop))
    return pRewriter->RecursiveRewrite(forExpr);

  const std::string localId = loop.type + "(" + BELinearLocalId(a_kernel) + ")";
  const std::string init    = (loop.start == "0") ? localId : localId + " + " + loop.start;
  const std::string size    = BEGroupSize(a_kernel);
  const std::string step    = (loop.type == a_kernel.be.wgTypes[0]) ? size : loop.type + "(" + size + ")"; // GLSL: no implicit 'uint' ==> 'int'
  return "for(" + loop.type + " " + loop.name + " = " + init + "; " + loop.cond + "; " + loop.name + " += " + step + ")\n" + LoopBody(forExpr->getBody(), pRewriter);
}

// 2D parallel loop: every loop is distributed over its dimension of the group, as loops of 'kernel2D_' over dimensions of the grid
//
std::string kslicer::IShaderCompiler::RewriteBEParallelFor2D(const clang::ForStmt* a_outer, const clang::ForStmt* a_inner, std::shared_ptr<KernelRewriter> pRewriter,
                                                             const KernelInfo& a_kernel, const uint32_t a_dimOfLevel[2])
{
  BEParallelLoop loops[2];
  if(!ParseParallelFor(a_outer, pRewriter, &loops[0]) || !ParseParallelFor(a_inner, pRewriter, &loops[1]))
    return pRewriter->RecursiveRewrite(a_outer);

  const unsigned column = pRewriter->GetCompiler().getSourceManager().getSpellingColumnNumber(a_outer->getBeginLoc()) - 1;
  std::string text;
  for(int level = 0; level < 2; level++)
  {
    const auto&       loop    = loops[level];
    const uint32_t    dim     = a_dimOfLevel[level];
    const std::string localId = loop.type + "(" + BELocalId(dim, a_kernel) + ")";
    const std::string init    = (loop.start == "0") ? localId : localId + " + " + loop.start;
    const std::string size    = a_kernel.be.wgNames[dim];
    const std::string step    = (loop.type == a_kernel.be.wgTypes[dim]) ? size : loop.type + "(" + size + ")";
    if(level != 0)
      text += std::string(column + 2, ' ');
    text += "for(" + loop.type + " " + loop.name + " = " + init + "; " + loop.cond + "; " + loop.name + " += " + step + ")\n";
  }
  return text + LoopBody(a_inner->getBody(), pRewriter);
}

std::string kslicer::IShaderCompiler::RewriteBEStmt(const clang::Stmt* stmt, std::shared_ptr<KernelRewriter> pRewriter)
{
  std::string text = pRewriter->RecursiveRewrite(stmt);
  if(!text.empty() && text.back() != ';' && text.back() != '}')
    text += ";";
  return text;
}

std::string kslicer::IShaderCompiler::BESingleThreadCond(const KernelInfo& a_kernel) const
{
  const std::string localId = BELinearLocalId(a_kernel);
  return (localId.find(' ') == std::string::npos) ? localId + " == 0" : "(" + localId + ") == 0";
}

std::string kslicer::IShaderCompiler::RewriteBEBody(const KernelInfo& a_kernel, std::shared_ptr<KernelRewriter> pRewriter, const uint32_t a_threadsOrder[3])
{
  if(a_kernel.be.body == nullptr)
    return "";
  BEBodyGenerator generator(this, a_kernel, pRewriter, IsISPC() ? "    " : "  ", a_threadsOrder);
  generator.Generate(a_kernel.be.body);
  return generator.Text();
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
  if(templateDecl == nullptr)
  {
    kslicer::PrintError(std::string("block expansion kernel '") + kernel.name + "' must be a template on the block size", f->getSourceRange(), compiler.getSourceManager());
    return;
  }
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
  
  // (2) actual template arguments: values are taken from the instantiated declaration,
  //     so named constants ('kernelBE1D_Test<BLOCK_SIZE>(...)') work as well as literals
  //
  int wgSize[3] = {1,1,1};
  const clang::TemplateArgumentList* pArgs = methodDecl->getTemplateSpecializationArgs();
  if(pArgs != nullptr)
  {
    for(int i = 0; i < top && i < int(pArgs->size()); i++)
    {
      const clang::TemplateArgument& arg = pArgs->get(i);
      if(arg.getKind() == clang::TemplateArgument::Integral)
        wgSize[i] = int(arg.getAsIntegral().getExtValue());
    }
  }
  else
  {
    std::string callText = kslicer::GetRangeSourceCode(f->getSourceRange(), compiler);
    ExtractTemplateParamsFromString(callText, wgSize);
  }

  uint32_t kernelDim = kernel.GetDim();
  assert(kernelDim == top);

  for(int i=0;i<top;i++)
    kernel.wgSize[i] = wgSize[i];
}

namespace
{
  class BlockExpansionCallsISPC : public clang::RecursiveASTVisitor<BlockExpansionCallsISPC>
  {
  public:
    BlockExpansionCallsISPC(kslicer::umap<std::string, kslicer::KernelInfo>& a_kernels, clang::Rewriter& a_rewriter, const clang::CompilerInstance& a_compiler) :
                            m_kernels(a_kernels), m_rewriter(a_rewriter), m_compiler(a_compiler) {}

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* f)
    {
      const clang::CXXMethodDecl* pMethod = f->getMethodDecl();
      if(pMethod == nullptr)
        return true;
      auto pKernel = m_kernels.find(pMethod->getNameAsString());
      if(pKernel == m_kernels.end() || !pKernel->second.be.enabled)
        return true;

      kslicer::ExtractBlockSizeFromCall(f, pKernel->second, m_compiler);

      // 'kernelBE1D_Test<64>(...)' ==> 'kernelBE1D_Test(...)', a member function of the generated class
      //
      auto pMember = clang::dyn_cast<clang::MemberExpr>(f->getCallee()->IgnoreParenImpCasts());
      if(pMember != nullptr && pMember->hasExplicitTemplateArgs())
        m_rewriter.ReplaceText(clang::SourceRange(pMember->getMemberLoc(), pMember->getRAngleLoc()), pKernel->first);
      found = true;
      return true;
    }

    bool found = false;

  private:
    kslicer::umap<std::string, kslicer::KernelInfo>& m_kernels;
    clang::Rewriter&                                 m_rewriter;
    const clang::CompilerInstance&                   m_compiler;
  };
}

bool kslicer::RewriteBlockExpansionCallsISPC(const clang::CXXMethodDecl* a_controlFunc, kslicer::umap<std::string, KernelInfo>& a_kernels, clang::Rewriter& a_rewriter, const clang::CompilerInstance& a_compiler)
{
  BlockExpansionCallsISPC visitor(a_kernels, a_rewriter, a_compiler);
  visitor.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_controlFunc));
  return visitor.found;
}
