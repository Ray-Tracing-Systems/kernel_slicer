#ifndef KSLICER_CLASS_GEN
#define KSLICER_CLASS_GEN

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"

#include "kslicer.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iostream>

namespace kslicer
{
  using namespace llvm;
  using namespace clang;
  
  /**\brief put all args together with comma or ',' to gave unique key for any concrete argument sequence.
      \return unique strig key which you can pass in std::unordered_map for example 
  */
  std::string MakeKernellCallSignature(const std::string& a_mainFuncName, const std::vector<ArgReferenceOnCall>& a_args, const std::unordered_set<std::string>& a_usedVectors);

  class MainFunctionRewriter : public RecursiveASTVisitor<MainFunctionRewriter>
  {
  public:
    
    MainFunctionRewriter(Rewriter &R, const clang::CompilerInstance& a_compiler, MainFuncInfo& a_mainFunc, 
                         const std::vector<InOutVarInfo>& a_args, MainClassInfo* a_pCodeInfo) : 
                         m_rewriter(R), m_compiler(a_compiler), m_sm(R.getSourceMgr()), 
                         m_dsTagId(0), m_mainFuncName(a_mainFunc.Name), m_mainFuncLocals(a_mainFunc.Locals),
                         m_pCodeInfo(a_pCodeInfo), m_allClassMembers(a_pCodeInfo->allDataMembers), m_mainFunc(a_mainFunc), allDescriptorSetsInfo(a_pCodeInfo->allDescriptorSetsInfo),
                         m_kernels(a_pCodeInfo->kernels) 
    { 
      for(const auto& arg : a_args) 
        m_argsOfMainFunc[arg.name] = arg;
    }
    
    bool VisitCXXMethodDecl(CXXMethodDecl* f);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
    bool VisitCallExpr(CallExpr* f);
    bool VisitIfStmt(IfStmt* ifExpr);
  
    std::string                                              mainFuncCmdName;
    std::unordered_map<std::string, uint32_t>                dsIdBySignature;
    std::vector< KernelCallInfo >&                           allDescriptorSetsInfo;
    std::unordered_map<std::string, DataMemberInfo>&         m_allClassMembers;
    const std::unordered_map<std::string, DataLocalVarInfo>& m_mainFuncLocals;

  private:

    std::vector<ArgReferenceOnCall> ExtractArgumentsOfAKernelCall(CallExpr* f);
    std::string MakeKernelCallCmdString(CXXMemberCallExpr* f);
    std::string MakeServiceKernelCallCmdString(CallExpr* call);

    Rewriter&                      m_rewriter;
    const clang::SourceManager&    m_sm;
    const clang::CompilerInstance& m_compiler;
    uint32_t m_dsTagId;
    
    std::string m_mainFuncName;
    std::unordered_map<std::string, InOutVarInfo>      m_argsOfMainFunc;
    const std::unordered_map<std::string, KernelInfo>& m_kernels;
    MainFuncInfo& m_mainFunc;

    std::unordered_set<uint64_t> m_alreadyProcessedCalls;
    MainClassInfo* m_pCodeInfo = nullptr;
  };

  std::vector<InOutVarInfo> ListPointerParamsOfMainFunc(const CXXMethodDecl* a_node);
  void MarkRewrittenRecursive(const clang::Stmt* currNode, std::unordered_set<uint64_t>& a_rewrittenNodes);
  void MarkRewrittenRecursive(const clang::Decl* currNode, std::unordered_set<uint64_t>& a_rewrittenNodes);

  class KernelRewriter : public RecursiveASTVisitor<KernelRewriter> // replace all expressions with class variables to kgen_data buffer access
  {
  public:
    
    KernelRewriter(Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, 
                  const std::string& a_fakeOffsetExpr, const bool a_infoPass) : 
                   m_rewriter(R), m_compiler(a_compiler), m_codeInfo(a_codeInfo), m_mainClassName(a_codeInfo->mainClassName), 
                   m_args(a_kernel.args), m_fakeOffsetExp(a_fakeOffsetExpr), m_kernelIsBoolTyped(a_kernel.isBoolTyped), m_kernelIsMaker(a_kernel.isMaker), m_currKernel(a_kernel), m_infoPass(a_infoPass)
    { 
      const auto& a_variables = a_codeInfo->dataMembers;
      m_variables.reserve(a_variables.size());
      for(const auto& var : a_variables) 
        m_variables[var.name] = var;
    }

    bool VisitMemberExpr(MemberExpr* expr);
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f);
    bool VisitCallExpr(CallExpr* f);
    bool VisitCXXConstructExpr(CXXConstructExpr* call);
    bool VisitReturnStmt(ReturnStmt* ret);
                                                                    // to detect reduction inside IPV programming template
    bool VisitUnaryOperator(UnaryOperator* expr);                   // ++, --, (*var) =  ...
    bool VisitCompoundAssignOperator(CompoundAssignOperator* expr); // +=, *=, -=; to detect reduction
    bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr* expr);       // +=, *=, -=; to detect reduction for custom data types (float3/float4 for example)
    bool VisitBinaryOperator(BinaryOperator* expr);                 // m_var = f(m_var, expr)

  private:

    bool CheckIfExprHasArgumentThatNeedFakeOffset(const std::string& exprStr);
    void ProcessReductionOp(const std::string& op, const Expr* lhs, const Expr* rhs, const Expr* expr);

    Rewriter&                                                m_rewriter;
    const clang::CompilerInstance&                           m_compiler;
    MainClassInfo*                                           m_codeInfo;
    std::string                                              m_mainClassName;
    std::unordered_map<std::string, kslicer::DataMemberInfo> m_variables;
    const std::vector<kslicer::KernelInfo::Arg>&             m_args;
    const std::string&                                       m_fakeOffsetExp;
    bool                                                     m_kernelIsBoolTyped;
    bool                                                     m_kernelIsMaker;
    kslicer::KernelInfo&                                     m_currKernel;
    bool                                                     m_infoPass;
    std::unordered_set<uint64_t>                             m_rewrittenNodes;

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    std::string FunctionCallRewrite(const CallExpr* call);
    std::string FunctionCallRewrite(const CXXConstructExpr* call);
    std::string RecursiveRewrite   (const Stmt* expr); // double/multiple pass rewrite purpose

    inline bool WasNotRewrittenYet(const clang::Stmt* expr)
    {
      auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
      return (m_rewrittenNodes.find(exprHash) == m_rewrittenNodes.end());
    }

    inline void MarkRewritten(const clang::Stmt* expr) { kslicer::MarkRewrittenRecursive(expr, m_rewrittenNodes); }
  };
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////  FunctionRewriter  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
  \brief process local functions (data["LocalFunctions"]), float3 --> make_float3, std::max --> fmax and e.t.c.
  */
  class FunctionRewriter : public RecursiveASTVisitor<FunctionRewriter> // 
  {
  public:
    
    FunctionRewriter(Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo) : 
                     m_rewriter(R), m_compiler(a_compiler), m_codeInfo(a_codeInfo)
    { 
      
    }

    virtual ~FunctionRewriter(){}

    bool VisitCallExpr(CallExpr* f);
    bool VisitCXXConstructExpr(CXXConstructExpr* call);
    
    bool VisitMemberExpr(clang::MemberExpr* expr)        { return VisitMemberExpr_Impl(expr);     }
    bool VisitCXXMethodDecl(clang::CXXMethodDecl* fDecl) { return VisitCXXMethodDecl_Impl(fDecl); }
    bool VisitCXXMemberCallExpr(CXXMemberCallExpr* f)    { return VisitCXXMemberCallExpr_Impl(f); }
    bool VisitFieldDecl(FieldDecl* decl)                 { return VisitFieldDecl_Impl(decl);      }
    //bool VisitParmValDecl(clang::ParmVarDecl* decl)      { return VisitParmValDecl_Impl(decl);    } // dones not works for some reason ... 

  protected:

    Rewriter&                                                m_rewriter;
    const clang::CompilerInstance&                           m_compiler;
    MainClassInfo*                                           m_codeInfo;
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    
    std::unordered_set<uint64_t>                             m_rewrittenNodes;
    inline void MarkRewritten(const clang::Stmt* expr) { kslicer::MarkRewrittenRecursive(expr, m_rewrittenNodes); }
    virtual std::string RecursiveRewrite(const Stmt* expr); // double/multiple pass rewrite purpose

    inline bool WasNotRewrittenYet(const clang::Stmt* expr)
    {
      auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
      return (m_rewrittenNodes.find(exprHash) == m_rewrittenNodes.end());
    }

    std::string FunctionCallRewrite(const CallExpr* call);
    std::string FunctionCallRewrite(const CXXConstructExpr* call);
  
    virtual bool VisitMemberExpr_Impl(clang::MemberExpr* expr)        { return true; } // override this in Derived class
    virtual bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl) { return true; } // override this in Derived class
    virtual bool VisitCXXMemberCallExpr_Impl(CXXMemberCallExpr* f)    { return true; } // override this in Derived class
    virtual bool VisitFieldDecl_Impl(FieldDecl* decl)                 { return true; } // override this in Derived class
    //virtual bool VisitParmValDecl_Impl(clang::ParmVarDecl* decl)      { return true; } // override this in Derived class  // dones not works for some reason ... 
  };
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////  NodesMarker  //////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  class NodesMarker : public RecursiveASTVisitor<NodesMarker> // mark all subsequent nodes to be rewritten, put their ash codes in 'rewrittenNodes'
  {
  public:
    NodesMarker(std::unordered_set<uint64_t>& a_rewrittenNodes) : m_rewrittenNodes(a_rewrittenNodes){}
    bool VisitStmt(Stmt* expr);

  private:
    std::unordered_set<uint64_t>& m_rewrittenNodes;
  };

  void ObtainKernelsDecl(std::unordered_map<std::string, KernelInfo>& a_kernelsData, const clang::CompilerInstance& compiler, const std::string& a_mainClassName, const MainClassInfo& a_codeInfo);
  std::string GetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair>& threadIds);
}


#endif
