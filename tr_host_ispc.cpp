#include "kslicer.h"
#include "template_rendering.h"

#include "clang/AST/RecursiveASTVisitor.h"

void kslicer::ISPCCodeGen::GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
  /**
  \brief RTV control function for wavefront ('-megakernel 0'): the control function is executed for chunks of threads [kgen_start, kgen_end),
         every kernel call processes all threads of the chunk; local variables of control function (payload) are arrays of the chunk ('kgen_payload')
  */
  class RTVCallsRewriterISPC : public clang::RecursiveASTVisitor<RTVCallsRewriterISPC>
  {
  public:
    RTVCallsRewriterISPC(clang::Rewriter& a_rewriter, const clang::CompilerInstance& a_compiler, kslicer::MainFuncInfo& a_mainFunc, kslicer::MainClassInfo& a_codeInfo) :
                         m_rewriter(a_rewriter), m_compiler(a_compiler), m_mainFunc(a_mainFunc), m_codeInfo(a_codeInfo) {}

    // 'if(!kernel_Step(tid, &x)) break;' ==> 'kernel_Step_Wave(...);': the kernel sets thread flags, other kernels skip finished threads;
    // 'if(isDeadRay(rayFlags)) break;' with payload in condition is removed: payload is stored in arrays, kernels check it themselves
    //
    bool VisitIfStmt(clang::IfStmt* a_if)
    {
      const clang::Expr* cond = a_if->getCond()->IgnoreParenImpCasts();
      if(auto pNot = clang::dyn_cast<clang::UnaryOperator>(cond); pNot != nullptr && pNot->getOpcode() == clang::UO_LNot)
        cond = pNot->getSubExpr()->IgnoreParenImpCasts();

      if(auto pCall = clang::dyn_cast<clang::CXXMemberCallExpr>(cond); pCall != nullptr && IsKernelRTV(pCall))
      {
        kslicer::ReplaceTextSafe(m_rewriter, a_if->getSourceRange(), MakeKernelCall(pCall) + ";");
        m_done.insert(kslicer::GetHashOfSourceRange(pCall->getSourceRange()));
        return true;
      }

      const clang::Stmt* pThen = a_if->getThen();
      if(auto pBlock = clang::dyn_cast<clang::CompoundStmt>(pThen); pBlock != nullptr && pBlock->size() == 1)
        pThen = pBlock->body_front();
      const bool exitStmt = clang::isa<clang::BreakStmt>(pThen) || clang::isa<clang::ReturnStmt>(pThen) || clang::isa<clang::ContinueStmt>(pThen);
      if(exitStmt && a_if->getElse() == nullptr && UsesPayload(a_if->getCond()))
      {
        const std::string text = kslicer::GetRangeSourceCode(a_if->getCond()->getSourceRange(), m_compiler);
        kslicer::ReplaceTextSafe(m_rewriter, a_if->getSourceRange(), "{} // 'if(" + text + ") ...' is checked by kernels for every thread");
      }
      return true;
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* a_call)
    {
      if(!IsKernelRTV(a_call) || m_done.find(kslicer::GetHashOfSourceRange(a_call->getSourceRange())) != m_done.end())
        return true;
      kslicer::ReplaceTextSafe(m_rewriter, a_call->getSourceRange(), MakeKernelCall(a_call));
      return true;
    }

  private:

    bool IsKernelRTV(const clang::CXXMemberCallExpr* a_call) const
    {
      if(a_call->getMethodDecl() == nullptr)
        return false;
      auto pKernel = m_codeInfo.kernels.find(a_call->getMethodDecl()->getNameAsString());
      return pKernel != m_codeInfo.kernels.end() && pKernel->second.pattern == kslicer::PATTERN_TP::PATTERN_RTV;
    }

    const clang::VarDecl* PayloadVar(const clang::Expr* a_expr) const
    {
      auto pRef = clang::dyn_cast<clang::DeclRefExpr>(a_expr->IgnoreParenImpCasts());
      if(pRef == nullptr)
        return nullptr;
      auto pVar = clang::dyn_cast<clang::VarDecl>(pRef->getDecl());
      if(pVar == nullptr || !pVar->hasLocalStorage() || clang::isa<clang::ParmVarDecl>(pVar))
        return nullptr;
      return (m_mainFunc.Locals.find(pVar->getNameAsString()) != m_mainFunc.Locals.end()) ? pVar : nullptr;
    }

    bool UsesPayload(const clang::Stmt* a_stmt) const
    {
      if(a_stmt == nullptr)
        return false;
      if(auto pExpr = clang::dyn_cast<clang::Expr>(a_stmt); pExpr != nullptr && PayloadVar(pExpr) != nullptr)
        return true;
      for(const clang::Stmt* child : a_stmt->children())
        if(UsesPayload(child))
          return true;
      return false;
    }

    std::string MakeKernelCall(const clang::CXXMemberCallExpr* a_call)
    {
      const std::string fname = a_call->getMethodDecl()->getNameAsString();
      auto& kernel            = m_codeInfo.kernels.find(fname)->second;

      // thread flags: the same rules as for Vulkan, see MainFunctionRewriterVulkan::MakeKernelCallCmdString
      //
      const auto callHash = kslicer::GetHashOfSourceRange(a_call->getSourceRange());
      const auto p3       = m_mainFunc.CallsInsideFor.find(callHash);
      const auto p4       = m_mainFunc.ExitExprIfCall.find(callHash);
      std::string flags   = (p3 != m_mainFunc.CallsInsideFor.end()) ? "inForFlags" : "outOfForFlags";
      if(kernel.isBoolTyped && p4 == m_mainFunc.ExitExprIfCall.end())
        flags += "D";
      else if(p3 != m_mainFunc.CallsInsideFor.end() && p3->second.isNegative)
        flags += "N";
      else if(p3 == m_mainFunc.CallsInsideFor.end() && p4 != m_mainFunc.ExitExprIfCall.end() && p4->second.isNegative)
        flags += "N";

      std::string res = fname + "_Wave(kgen_start, kgen_end, kgen_sizeX, kgen_payload.threadFlags.data(), " + flags;
      for(unsigned i = 0; i < a_call->getNumArgs(); i++)
      {
        if(i < kernel.args.size() && kernel.args[i].isThreadID) // thread id is computed by kernel for its chunk
          continue;
        const clang::Expr* pArg = a_call->getArg(i);
        if(auto pDefault = clang::dyn_cast<clang::CXXDefaultArgExpr>(pArg))
          pArg = pDefault->getExpr();
        std::string text = kslicer::GetRangeSourceCode(pArg->getSourceRange(), m_compiler);
        if(auto pAddr = clang::dyn_cast<clang::UnaryOperator>(pArg->IgnoreParenImpCasts()); pAddr != nullptr && pAddr->getOpcode() == clang::UO_AddrOf)
        {
          if(auto pVar = PayloadVar(pAddr->getSubExpr()))
          {
            text = "kgen_payload." + pVar->getNameAsString() + ".data()";
            if(i < kernel.args.size())
              kernel.args[i].needFakeOffset = true; // '*a_x' ==> 'a_x[kgen_i]' in kernel
          }
        }
        res += ", " + text;
      }
      return res + ")";
    }

    clang::Rewriter&               m_rewriter;
    const clang::CompilerInstance& m_compiler;
    kslicer::MainFuncInfo&         m_mainFunc;
    kslicer::MainClassInfo&        m_codeInfo;
    std::set<uint64_t>             m_done;
  };
}

bool kslicer::RewriteRTVCallsISPC(kslicer::MainFuncInfo& a_mainFunc, kslicer::MainClassInfo& a_codeInfo, clang::Rewriter& a_rewriter, const clang::CompilerInstance& a_compiler)
{
  if(a_mainFunc.pattern != kslicer::PATTERN_TP::PATTERN_RTV || a_codeInfo.megakernelRTV)
    return false;
  RTVCallsRewriterISPC rv(a_rewriter, a_compiler, a_mainFunc, a_codeInfo);
  rv.TraverseStmt(const_cast<clang::Stmt*>(a_mainFunc.Node->getBody()));
  return true;
}
