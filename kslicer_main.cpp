// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/RecursiveASTVisitor.h"


using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

#include <string>
#include <iostream>
#include <fstream>
#include <streambuf>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FindNamedClassVisitor : public RecursiveASTVisitor<FindNamedClassVisitor> 
{
public:
  explicit FindNamedClassVisitor(ASTContext *Context): Context(Context) {}

  bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) 
  {
    auto temp = Declaration->getQualifiedNameAsString();
    if (Declaration->getQualifiedNameAsString() == m_mainClassName) 
    {
      FullSourceLoc FullLocation = Context->getFullLoc(Declaration->getBeginLoc());
      if (FullLocation.isValid())
        llvm::outs() << "Found declaration at "
                     << FullLocation.getSpellingLineNumber() << ":"
                     << FullLocation.getSpellingColumnNumber() << "\n";

      Declaration->dump();
    }
    return true;
  }

  std::string m_mainClassName;

private:
  ASTContext *Context;
};

class FindNamedClassConsumer : public clang::ASTConsumer 
{
public:
  explicit FindNamedClassConsumer(ASTContext *Context, const std::string& a_className) : Visitor(Context) 
  {
    Visitor.m_mainClassName = a_className;
  }

  virtual void HandleTranslationUnit(clang::ASTContext &Context) 
  {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  FindNamedClassVisitor Visitor;
};

class FindNamedClassAction : public clang::ASTFrontendAction 
{
public:
  
  FindNamedClassAction(const std::string& a_className) : m_mainClassName(a_className) {}

  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &Compiler, llvm::StringRef InFile) 
  {
    return std::unique_ptr<clang::ASTConsumer>(new FindNamedClassConsumer(&Compiler.getASTContext(), m_mainClassName));
  }

  std::string m_mainClassName;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//StatementMatcher LoopMatcher =
//  forStmt(hasLoopInit(declStmt(hasSingleDecl(varDecl(
//    hasInitializer(integerLiteral(equals(0)))))))).bind("forLoop");

/*

StatementMatcher LoopMatcher =
    forStmt(hasLoopInit(declStmt(
                hasSingleDecl(varDecl(hasInitializer(integerLiteral(equals(0))))
                                  .bind("initVarName")))),
            hasIncrement(unaryOperator(
                hasOperatorName("++"),
                hasUnaryOperand(declRefExpr(
                    to(varDecl(hasType(isInteger())).bind("incVarName")))))),
            hasCondition(binaryOperator(
                hasOperatorName("<"),
                hasLHS(ignoringParenImpCasts(declRefExpr(
                    to(varDecl(hasType(isInteger())).bind("condVarName"))))),
                hasRHS(expr(hasType(isInteger())))))).bind("forLoop");


static bool areSameVariable(const ValueDecl *First, const ValueDecl *Second) {
  return First && Second &&
         First->getCanonicalDecl() == Second->getCanonicalDecl();
}

static bool areSameExpr(ASTContext *Context, const Expr *First,
                        const Expr *Second) {
  if (!First || !Second)
    return false;
  llvm::FoldingSetNodeID FirstID, SecondID;
  First->Profile(FirstID, *Context, true);
  Second->Profile(SecondID, *Context, true);
  return FirstID == SecondID;
}


class LoopPrinter : public MatchFinder::MatchCallback 
{
public :
  virtual void run(const MatchFinder::MatchResult &Result) override;
};


void LoopPrinter::run(const MatchFinder::MatchResult &Result) 
{
  ASTContext *Context = Result.Context;
  const ForStmt *FS = Result.Nodes.getNodeAs<ForStmt>("forLoop");
  // We do not want to convert header files!
  if (!FS || !Context->getSourceManager().isWrittenInMainFile(FS->getForLoc()))
    return;
  const VarDecl *IncVar = Result.Nodes.getNodeAs<VarDecl>("incVarName");
  const VarDecl *CondVar = Result.Nodes.getNodeAs<VarDecl>("condVarName");
  const VarDecl *InitVar = Result.Nodes.getNodeAs<VarDecl>("initVarName");

  if (!areSameVariable(IncVar, CondVar) || !areSameVariable(IncVar, InitVar))
    return;
  llvm::outs() << "Potential array-based loop discovered.\n";
  
  FS->dump();
}

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...\n");
*/

std::string ReadFile(const char* a_fileName)
{
  std::ifstream fin(a_fileName);
  std::string str;

  fin.seekg(0, std::ios::end);   
  str.reserve(fin.tellg());
  fin.seekg(0, std::ios::beg);
  
  str.assign((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
  return str;
}


int main(int argc, const char **argv) 
{
  std::cout << "tool start" << std::endl;
  if (argc <= 1)
    return 0;

  //CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  //ClangTool Tool(OptionsParser.getCompilations(),
  //               OptionsParser.getSourcePathList());
  //
  //LoopPrinter Printer;
  //MatchFinder Finder;
  //Finder.addMatcher(LoopMatcher, &Printer);
  //
  //return Tool.run(newFrontendActionFactory(&Finder).get());

  std::string sourceCode = ReadFile(argv[1]);
  clang::tooling::runToolOnCode(std::make_unique<FindNamedClassAction>("TestClass"), sourceCode.c_str());

  return 0;
}