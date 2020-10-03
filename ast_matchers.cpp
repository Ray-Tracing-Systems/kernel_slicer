#include "ast_matchers.h"

namespace kslicer
{
  std::string last_fname = "";
  uint32_t last_lineno = 0xFFFFFFF;
}

std::string kslicer::locationAsString(clang::SourceLocation loc, clang::SourceManager const * const sm)
{
  std::stringstream s;
  if(!sm) {
    s << "Invalid SourceManager, cannot dump Location\n";
    return s.str();
  }
  clang::SourceLocation SpellingLoc = sm->getSpellingLoc(loc);
  clang::PresumedLoc ploc = sm->getPresumedLoc(SpellingLoc);
  if(ploc.isInvalid()) {
    s << "<invalid sloc>";
    return s.str();
  }

  std::string fname = ploc.getFilename();
  uint32_t const lineno = ploc.getLine();
  uint32_t const colno = ploc.getColumn();
  if(fname != last_fname) {
    s << fname << ':' << lineno << ':' << colno;
    last_fname = fname;
    last_lineno = lineno;
  }
  else if(lineno != last_lineno) {
    s << "line" << ':' << lineno << ':' << colno;
    last_lineno = lineno;
  }
  else {
    s << "col" << ':' << colno;
  }
  return s.str();
}  // locationAsString

std::string kslicer::sourceRangeAsString(clang::SourceRange r, clang::SourceManager const * sm)
{
  if(!sm) 
    return "";
  std::stringstream s;
  s << "<" << locationAsString(r.getBegin(), sm);
  if(r.getBegin() != r.getEnd())
    s << ", " << locationAsString(r.getEnd(), sm);
  
  s << ">";
  return s.str();
}  // sourceRangeAsString


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

clang::ast_matchers::StatementMatcher kslicer::mk_local_var_matcher_of_function(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return
  declRefExpr(
    to(varDecl(hasLocalStorage()).bind("locVarName")),
       hasAncestor(functionDecl(hasName(a_funcName)).bind("targetFunction")
    )
  ).bind("localReference");
}

clang::ast_matchers::StatementMatcher kslicer::mk_krenel_call_matcher_from_function(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return 
  cxxMemberCallExpr(
    allOf(hasAncestor( functionDecl(hasName(a_funcName)).bind("targetFunction") ),
          callee(functionDecl().bind("fdecl"))
         )
  ).bind("functionCall");
}
