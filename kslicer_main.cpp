#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <vector>
#include <system_error>
#include <iostream>

#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;

struct FunctionInfo 
{
  struct Arg 
  {
    std::string type;
    std::string name;
    int size;
  };
  std::string return_type;
  std::string name;
  std::vector<Arg> args;
};

// RecursiveASTVisitor is is the big-kahuna visitor that traverses
// everything in the AST.
class MyRecursiveASTVisitor : public RecursiveASTVisitor<MyRecursiveASTVisitor>
{

 public:
  std::string ADDED_PREFFIX;// = "PREF_";
  std::string MAIN_NAME;
  MyRecursiveASTVisitor(Rewriter &R, std::string pref, std::string main_name) : Rewrite(R), ADDED_PREFFIX(pref), MAIN_NAME(main_name) { }
  void ProcessFunction(FunctionDecl *f);
  bool VisitFunctionDecl(FunctionDecl *f);
  bool VisitCallExpr(CallExpr *CE);
  std::map<std::string, FunctionInfo> functions;
  std::string GetNewNameFor(std::string s);
  Rewriter &Rewrite;
};

std::string MyRecursiveASTVisitor::GetNewNameFor(std::string s) {
  return ADDED_PREFFIX + s;
}

bool MyRecursiveASTVisitor::VisitCallExpr(CallExpr *CE) 
{
  FunctionDecl *FD = CE->getDirectCallee();
  if (FD) {
    DeclarationNameInfo dni = FD->getNameInfo();
    DeclarationName dn = dni.getName();
    std::string fname = dn.getAsString();
    if (fname == "readAttr") {
      auto expr = cast<clang::ImplicitCastExpr>(CE->getArg(1));
      std::string param = cast<clang::StringLiteral>(expr->getSubExpr())->getString().str();
      Rewrite.ReplaceText(CE->getSourceRange(), "readAttr_" + param + "(sHit)");
    }
    
    if (FD->hasBody()) {
      Rewrite.ReplaceText(CE->getBeginLoc(), fname.size(), GetNewNameFor(fname));
    }
  }
  return true;
}

int GetSizeByType(std::string t) {
  return 1;
}

FunctionInfo::Arg ProcessParameter(ParmVarDecl *p) {
  QualType q = p->getType();
  const Type *typ = q.getTypePtr();
  FunctionInfo::Arg arg;
  arg.name = p->getNameAsString();
  arg.type = QualType::getAsString(q.split(), PrintingPolicy{ {} });
  arg.size = 1;
  if (typ->isPointerType()) {
    arg.size = 1; // Because C always pass reference
  }
  
  return arg;
}

void MyRecursiveASTVisitor::ProcessFunction(FunctionDecl *f) {
  if (!f || !f->hasBody()) {
    return;
  }
  FunctionInfo info;
  DeclarationNameInfo dni = f->getNameInfo();
  DeclarationName dn = dni.getName();
  info.name = dn.getAsString();
  QualType q = f->getReturnType();
  const Type *typ = q.getTypePtr();
  info.return_type = QualType::getAsString(q.split(), PrintingPolicy{ {} });
  for (int i = 0; i < f->getNumParams(); ++i) {
    info.args.push_back(ProcessParameter(f->parameters()[i]));
  }
  functions[info.name] = info;
}

bool MyRecursiveASTVisitor::VisitFunctionDecl(FunctionDecl *f)
{
  ProcessFunction(f);
  if (f->hasBody())
  {
    SourceRange sr = f->getSourceRange();
    Stmt *s = f->getBody();

    QualType q = f->getReturnType();
    std::string ret;
    ret = QualType::getAsString(q.split(), PrintingPolicy{ {} });
    // Get name of function
    DeclarationNameInfo dni = f->getNameInfo();
    DeclarationName dn = dni.getName();
    std::string fname = dn.getAsString();
    
    // Point to start of function declaration
    SourceLocation ST = sr.getBegin();
    // Add comment
    if (fname == MAIN_NAME) {
      llvm::errs() << "Found main()\n";
      int num_params = f->getNumParams();
      auto param = f->getParamDecl(num_params - 1);
      SourceLocation decl_end = param->getEndLoc().getLocWithOffset(param->getName().size());
      //decl_end = f->getNameInfo().getEndLoc().getLocWithOffset(1);
      Rewrite.InsertText(decl_end, ", _PROCTEXTAILTAG_", true, true);
    }
    char fc[256];
    Rewrite.ReplaceText(dni.getSourceRange(), GetNewNameFor(fname));
    
  }

  return true; // returning false aborts the traversal
}

class MyASTConsumer : public ASTConsumer
{
 public:

  MyASTConsumer(Rewriter &Rewrite, std::string pref, std::string main_name) : rv(Rewrite, pref, main_name) { }
  virtual bool HandleTopLevelDecl(DeclGroupRef d);
  MyRecursiveASTVisitor rv;
};

bool MyASTConsumer::HandleTopLevelDecl(DeclGroupRef d)
{
  typedef DeclGroupRef::iterator iter;

  for (iter b = d.begin(), e = d.end(); b != e; ++b)
  {
    rv.TraverseDecl(*b);
  }

  return true; // keep going
}


int main(int argc, char **argv)
{
  struct stat sb;

  if (argc < 2)
  {
    llvm::errs() << "Usage: <filename>\n";
    return 1;
  }

  llvm::ArrayRef<const char*> args(argv+1, argv+argc);

  // Get filename
  std::string fileName(argv[argc - 1]);

  // Make sure it exists
  if (stat(fileName.c_str(), &sb) == -1)
  {
    perror(fileName.c_str());
    exit(EXIT_FAILURE);
  }

  CompilerInstance compiler;
  DiagnosticOptions diagnosticOptions;
  compiler.createDiagnostics();
  //compiler.createDiagnostics(argc, argv);

  // Create an invocation that passes any flags to preprocessor
  std::shared_ptr<CompilerInvocation> Invocation = std::make_shared<CompilerInvocation>();
  CompilerInvocation::CreateFromArgs(*Invocation, args, compiler.getDiagnostics());
  compiler.setInvocation(Invocation);

  // Set default target triple
    std::shared_ptr<clang::TargetOptions> pto = std::make_shared<clang::TargetOptions>();
  pto->Triple = llvm::sys::getDefaultTargetTriple();
    TargetInfo *pti = TargetInfo::CreateTargetInfo(compiler.getDiagnostics(), pto);
  compiler.setTarget(pti);

  compiler.createFileManager();
  compiler.createSourceManager(compiler.getFileManager());

  HeaderSearchOptions &headerSearchOptions = compiler.getHeaderSearchOpts();

  headerSearchOptions.AddPath("/usr/include/c++/4.6",
          clang::frontend::Angled,
          false,
          false);
  headerSearchOptions.AddPath("/usr/include/c++/4.6/i686-linux-gnu",
          clang::frontend::Angled,
          false,
          false);
  headerSearchOptions.AddPath("/usr/include/c++/4.6/backward",
          clang::frontend::Angled,
          false,
          false);
  headerSearchOptions.AddPath("/usr/local/include",
          clang::frontend::Angled,
          false,
          false);
  headerSearchOptions.AddPath("/usr/local/lib/clang/3.3/include",
          clang::frontend::Angled,
          false,
          false);
  headerSearchOptions.AddPath("/usr/include/i386-linux-gnu",
          clang::frontend::Angled,
          false,
          false);
  headerSearchOptions.AddPath("/usr/include",
          clang::frontend::Angled,
          false,
          false);
  // </Warning!!> -- End of Platform Specific Code


  // Allow C++ code to get rewritten
  LangOptions langOpts;
  langOpts.GNUMode = 1; 
  //langOpts.CXXExceptions = 1; 
  langOpts.RTTI = 1; 
  langOpts.Bool = 1; 
  langOpts.CPlusPlus = 1; 
  langOpts.CPlusPlus14 = 1;
  langOpts.CPlusPlus17 = 1;
  compiler.createPreprocessor(clang::TU_Complete);
  compiler.getPreprocessorOpts().UsePredefines = false;
  compiler.getLangOpts() = langOpts;
  compiler.createASTContext();
  // Initialize rewriter
  Rewriter Rewrite;
  Rewrite.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  const FileEntry *pFile = compiler.getFileManager().getFile(fileName).get();
  
  compiler.getSourceManager().setMainFileID( compiler.getSourceManager().createFileID( pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(),
                                                &compiler.getPreprocessor());

  std::string pref = "prtex4_";
  std::string main_name = "main";
  MyASTConsumer astConsumer(Rewrite, pref, main_name);

  // Convert <file>.c to <file_out>.c
  std::string outName (fileName);
  size_t ext = outName.rfind(".");
  if (ext == std::string::npos)
     ext = outName.length();
  outName.insert(ext, "_out");

  llvm::errs() << "Output to: " << outName << "\n";
  std::error_code OutErrorInfo;
  std::error_code ok;
  llvm::raw_fd_ostream outFile(llvm::StringRef(outName), OutErrorInfo, llvm::sys::fs::F_None);

  if (OutErrorInfo == ok)
  {
    // Parse the AST
    ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
    compiler.getDiagnosticClient().EndSourceFile();

    // Now output rewritten source code
    const RewriteBuffer *RewriteBuf =
      Rewrite.getRewriteBufferFor(compiler.getSourceManager().getMainFileID());
    outFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
  }
  else
  {
    llvm::errs() << "Cannot open " << outName << " for writing\n";
  }

  outFile.close();
  for (auto &a : astConsumer.rv.functions) {
    std::cout << a.first << " " << a.second.return_type << std::endl;
    for (int i = 0; i < a.second.args.size(); ++i) {
      std::cout << a.second.args[i].name << ":" << a.second.args[i].type << ":" << a.second.args[i].size << std::endl;
    }
    std::cout << std::endl;
  }
  return 0;
}