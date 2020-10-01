#include "ast_matchers.h"

namespace kslicer
{
  std::string last_fname = "";
  uint32_t last_lineno = 0xFFFFFFF;
};

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

clang::ast_matchers::StatementMatcher kslicer::all_global_var_matcher()
{
  using namespace clang::ast_matchers;
  return
  declRefExpr(
    to(
      varDecl(
        hasGlobalStorage()
      ).bind("gvarName")
    ) // to
   ,hasAncestor(
      functionDecl().bind("function")
    )
  ).bind("globalReference");
} 

clang::ast_matchers::StatementMatcher kslicer::mk_global_var_matcher(std::string const & g_var_name)
{
  using namespace clang::ast_matchers;
  return
  declRefExpr(
    to(
      varDecl(
        hasGlobalStorage()
       ,hasName(g_var_name)
      ).bind("gvarName")
    )
   ,hasAncestor(
      functionDecl().bind("function")
    )
  ).bind("globalReference");
} 

