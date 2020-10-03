#ifndef KSLICER_AST_MATCHERS
#define KSLICER_AST_MATCHERS

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include <string>
#include <sstream>
#include <iostream>

namespace kslicer
{
  using namespace clang::tooling;
  using namespace llvm;
  using namespace clang::ast_matchers;

  clang::ast_matchers::StatementMatcher all_global_var_matcher();
  clang::ast_matchers::StatementMatcher mk_global_var_matcher(std::string const & g_var_name = "");

  clang::ast_matchers::StatementMatcher mk_local_var_matcher_of_function(std::string const& funcName);
  clang::ast_matchers::StatementMatcher mk_krenel_call_matcher_from_function(std::string const& funcName);

  std::string locationAsString(clang::SourceLocation loc, clang::SourceManager const * const sm);
  std::string sourceRangeAsString(clang::SourceRange r, clang::SourceManager const * sm);

  /**\brief Complain if pointer is invalid.
  \param p: pointer
  \param name: name of thing checked
  \param tabs: indentation
  \return none */
  template <typename T>
  inline void
  check_ptr(T * p,
            const std::string& name,
            const std::string& tabs = "",
            std::ostream&         s = std::cout)
  {
    if(!p) { s << tabs << "Invalid pointer " << name << "\n"; }
  }

  class Global_Printer : public clang::ast_matchers::MatchFinder::MatchCallback 
  {
  public:

    explicit Global_Printer(std::ostream & s) : s_(s), n_matches_(0) {}
   
    //void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
    //{
    //  using namespace clang;
    //  n_matches_++;
    // 
    //  FunctionDecl const * func_decl = result.Nodes.getNodeAs<FunctionDecl>("targetFunction");
    //  Expr const * l_var             = result.Nodes.getNodeAs<Expr>("localReference");
    //  VarDecl const * var            = result.Nodes.getNodeAs<VarDecl>("locVarName");
    //
    //  clang::SourceManager & src_manager(const_cast<clang::SourceManager &>(result.Context->getSourceManager()));
    //
    //  if(func_decl && l_var && var) 
    //  {
    //    s_ << "In function '" << func_decl->getNameAsString() << "' ";
    //    s_ << "'" << var->getNameAsString() << "' referred to at ";
    //    std::string sr(sourceRangeAsString(l_var->getSourceRange(), &src_manager));
    //    s_ << sr;
    //    s_ << "\n";
    //  }
    //  else 
    //  {
    //    check_ptr(func_decl, "func_decl", "", s_);
    //    check_ptr(l_var, "l_var", "", s_);
    //    check_ptr(var, "var", "", s_);
    //  }
    //
    //  return;
    //} 

    void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
    {
      using namespace clang;
      n_matches_++;
     
      FunctionDecl      const * func_decl = result.Nodes.getNodeAs<FunctionDecl>     ("targetFunction");
      CXXMemberCallExpr const * kern_call = result.Nodes.getNodeAs<CXXMemberCallExpr>("functionCall");
      CXXMethodDecl     const * kern      = result.Nodes.getNodeAs<CXXMethodDecl>    ("fdecl");

      clang::SourceManager& src_manager(const_cast<clang::SourceManager &>(result.Context->getSourceManager()));

      if(func_decl && kern_call && kern) 
      {
        s_ << "In function '" << func_decl->getNameAsString() << "' ";
        s_ << "'" << kern->getNameAsString() << "' referred to at ";
        std::string sr(sourceRangeAsString(kern_call->getSourceRange(), &src_manager));
        s_ << sr;
        s_ << "\n";
      }
      else 
      {
        check_ptr(func_decl, "func_decl", "", s_);
        check_ptr(kern_call, "kern_call", "", s_);
        check_ptr(kern,      "kern", "", s_);
      }

      return;
    }  // run
    
  
    std::ostream & s_;
    uint32_t n_matches_;
  };  // class Global_Printer

}

#endif
