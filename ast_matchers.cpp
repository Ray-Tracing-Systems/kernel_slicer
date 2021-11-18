#include "ast_matchers.h"

namespace kslicer
{
  std::string last_fname = "";
  uint32_t last_lineno = 0xFFFFFFF;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_LocalVarOfMethod(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return
  declRefExpr(
    to(varDecl(hasLocalStorage()).bind("locVarName")),
       hasAncestor(cxxMethodDecl(hasName(a_funcName)).bind("targetFunction")
    )
  ).bind("localReference");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_MethodCallFromMethod(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return 
  cxxMemberCallExpr(
    allOf(hasAncestor( cxxMethodDecl(hasName(a_funcName)).bind("targetFunction") ),
          callee(cxxMethodDecl().bind("fdecl"))
         )
  ).bind("functionCall");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_MethodCallFromMethod()
{
  using namespace clang::ast_matchers;
  return 
  cxxMemberCallExpr(
    allOf(hasAncestor( cxxMethodDecl().bind("targetFunction") ),
          callee(cxxMethodDecl().bind("fdecl"))
         )
  ).bind("functionCall");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_FunctionCallFromFunction(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return 
  callExpr(
    allOf(hasAncestor( cxxMethodDecl(hasName(a_funcName)).bind("targetFunction") ),
          callee(functionDecl().bind("fdecl"))
         )
  ).bind("functionCall");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_MemberVarOfMethod(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return
  memberExpr(
    hasDeclaration(fieldDecl().bind("memberName")),
    hasAncestor(functionDecl(hasName(a_funcName)).bind("targetFunction"))
  ).bind("memberReference");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_SingleForLoopInsideFunction(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return 
  forStmt(hasLoopInit(declStmt(hasSingleDecl(varDecl().bind("loopIter")))),
          hasAncestor(functionDecl(hasName(a_funcName)).bind("targetFunction"))
  ).bind("forLoop");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_IfInsideForLoopInsideFunction(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return 
  forStmt(hasAncestor(functionDecl(hasName(a_funcName)).bind("targetFunction")),
          hasDescendant(ifStmt(
                         hasDescendant(cxxMemberCallExpr().bind("functionCall")),            // #TODO: check this call is inside if condition !!!
                         anyOf(hasDescendant(breakStmt().bind("breakLoop")), 
                               hasDescendant(returnStmt().bind("exitFunction"))) 
                        ).bind("ifCond"))
  ).bind("forLoop");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_FunctionCallInsideForLoopInsideFunction(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return 
  forStmt(hasAncestor(functionDecl(hasName(a_funcName)).bind("targetFunction")),
          hasDescendant(cxxMemberCallExpr().bind("functionCall"))
  ).bind("forLoop");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_IfReturnFromFunction(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return 
  ifStmt(
    hasAncestor(functionDecl(hasName(a_funcName)).bind("targetFunction")),
    hasDescendant(cxxMemberCallExpr().bind("functionCall")),                                 // #TODO: check this call is inside if condition !!!
    hasDescendant(returnStmt().bind("exitFunction"))
  ).bind("ifCond");
}

// https://stackoverflow.com/questions/36880574/finding-nested-loops-with-clang-ast-statementmatcher
//
clang::ast_matchers::StatementMatcher kslicer::MakeMatch_ForLoopInsideFunction(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  auto ArrayBoundMatcher = expr(hasType(isInteger())).bind("loopSize");
  return 
  forStmt(hasAncestor(functionDecl(hasName(a_funcName)).bind("targetFunction")),
          hasLoopInit(declStmt(hasSingleDecl(varDecl().bind("initVar")))),
          hasCondition(binaryOperator(hasLHS(ignoringParenImpCasts(declRefExpr(to(varDecl().bind("condVar"))))),
                                      hasRHS(ArrayBoundMatcher))),
          hasIncrement(unaryOperator(hasUnaryOperand(declRefExpr(to(varDecl().bind("incVar")))))) // 
  ).bind("loop");
}

clang::ast_matchers::StatementMatcher kslicer::MakeMatch_BeforeForLoopInsideFunction(std::string const& a_funcName)
{
  using namespace clang::ast_matchers;
  return
  compoundStmt( hasAncestor(functionDecl(hasName(a_funcName)).bind("targetFunction")),
                unless(hasAncestor(forStmt()))
  ).bind("loopInitCode");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

clang::ast_matchers::DeclarationMatcher kslicer::MakeMatch_StructDeclInsideClass(std::string const& className)
{
  using namespace clang::ast_matchers;
  return
  cxxRecordDecl(
    hasAncestor(cxxRecordDecl(hasName(className)).bind("mainClass"))
  ).bind("targetStruct");
}

clang::ast_matchers::DeclarationMatcher kslicer::MakeMatch_VarDeclInsideClass(std::string const& className)
{
  using namespace clang::ast_matchers;
  return
  varDecl(
    hasAncestor(cxxRecordDecl(hasName(className)).bind("mainClass"))
  ).bind("targetVar");
}

clang::ast_matchers::DeclarationMatcher kslicer::MakeMatch_TypedefInsideClass(std::string const& className)
{
  using namespace clang::ast_matchers;
  return
  typedefDecl(
    hasAncestor(cxxRecordDecl(hasName(className)).bind("mainClass"))
  ).bind("targetTypedef");
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MainFuncSeeker : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
  explicit MainFuncSeeker(std::ostream& s, const std::string& a_mainClassName, const clang::ASTContext& a_astContext, const kslicer::MainClassInfo& a_codeInfo) : 
                          m_out(s), m_mainClassName(a_mainClassName), m_astContext(a_astContext), m_codeInfo(a_codeInfo) 
  {
  }

  void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
  {
    using namespace clang;
   
    CXXMethodDecl     const * func_decl = result.Nodes.getNodeAs<CXXMethodDecl>    ("targetFunction");
    CXXMemberCallExpr const * kern_call = result.Nodes.getNodeAs<CXXMemberCallExpr>("functionCall");
    CXXMethodDecl     const * kern      = result.Nodes.getNodeAs<CXXMethodDecl>    ("fdecl");

    if(func_decl && kern_call && kern) 
    {
      const auto pClass = func_decl->getParent();
      std::string kName = kern->getNameAsString();
      assert(pClass != nullptr);
      //std::cout << "MainFuncSeeker: " << kName.c_str() << "\t class name = " << m_mainClassName.c_str() << " | " << pClass->getName().str() << std::endl;
      if(pClass->getName().str() == m_mainClassName && m_codeInfo.IsKernel(kName))
      {
        //std::cout << func_decl->getNameAsString() << " --> " << kern->getNameAsString() << std::endl;
        auto p = m_mainFunctions.find(func_decl->getNameAsString());
        if(p == m_mainFunctions.end())
        {
          kslicer::CFNameInfo info;
          info.name = func_decl->getNameAsString();
          info.kernelNames.push_back(kern->getNameAsString());
          m_mainFunctions[func_decl->getNameAsString()] = info;
        }
        else
        {
          auto& kernNames = p->second.kernelNames;
          auto elementId = std::find(kernNames.begin(), kernNames.end(), kern->getNameAsString());
          if(elementId == kernNames.end())
            kernNames.push_back(kern->getNameAsString());
        }
        
      }
    }
    else 
    {
      kslicer::check_ptr(func_decl, "func_decl", "", m_out);
      kslicer::check_ptr(kern_call, "kern_call", "", m_out);
      kslicer::check_ptr(kern,      "kern",      "", m_out);
    }
    return;
  }  // run
  
  std::ostream&                 m_out;
  const std::string&            m_mainClassName;
  const clang::ASTContext&      m_astContext;
  const kslicer::MainClassInfo& m_codeInfo;
  std::unordered_map<std::string, kslicer::CFNameInfo> m_mainFunctions;
}; 

std::unordered_map<std::string, kslicer::CFNameInfo> kslicer::ListAllMainRTFunctions(clang::tooling::ClangTool& Tool, 
                                                                                           const std::string& a_mainClassName, 
                                                                                           const clang::ASTContext& a_astContext,
                                                                                           const MainClassInfo& a_codeInfo)
{
  auto kernelCallMatcher = kslicer::MakeMatch_MethodCallFromMethod();
  
  MainFuncSeeker printer(std::cout, a_mainClassName, a_astContext, a_codeInfo);
  clang::ast_matchers::MatchFinder finder;
  finder.addMatcher(kernelCallMatcher,  &printer);

  auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
  if(res != 0) 
    std::cout << "[Seeking for MainFunc]: tool run res = " << res << std::endl;

  return printer.m_mainFunctions;
}