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

    explicit Global_Printer(std::ostream & s) : m_out(s){}

    void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
    {
      using namespace clang;
     
      FunctionDecl      const * func_decl = result.Nodes.getNodeAs<FunctionDecl>     ("targetFunction");
      CXXMemberCallExpr const * kern_call = result.Nodes.getNodeAs<CXXMemberCallExpr>("functionCall");
      CXXMethodDecl     const * kern      = result.Nodes.getNodeAs<CXXMethodDecl>    ("fdecl");

      Expr              const * l_var     = result.Nodes.getNodeAs<Expr>   ("localReference");
      VarDecl           const * var       = result.Nodes.getNodeAs<VarDecl>("locVarName");

      clang::SourceManager& src_manager(const_cast<clang::SourceManager &>(result.Context->getSourceManager()));

      if(func_decl && kern_call && kern) 
      {
        m_out << "In function '" << func_decl->getNameAsString() << "' ";
        m_out << "method '" << kern->getNameAsString() << "' referred to at ";
        std::string sr(sourceRangeAsString(kern_call->getSourceRange(), &src_manager));
        m_out << sr;
        m_out << "\n";
      }
      else if(func_decl && l_var && var)
      {
        m_out << "In function '" << func_decl->getNameAsString() << "' ";
        m_out << "variable '" << var->getNameAsString() << "' referred to at ";
        std::string sr(sourceRangeAsString(l_var->getSourceRange(), &src_manager));
        m_out << sr;
        m_out << "\n";
      }
      else 
      {
        check_ptr(l_var,     "l_var", "", m_out);
        check_ptr(var,       "var",   "", m_out);

        check_ptr(func_decl, "func_decl", "", m_out);
        check_ptr(kern_call, "kern_call", "", m_out);
        check_ptr(kern,      "kern",      "", m_out);
      }

      return;
    }  // run
    
    std::ostream& m_out;

  };  // class Global_Printer

}

#endif
