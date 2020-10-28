#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <vector>
#include <system_error>
#include <iostream>
#include <fstream>

#include <unordered_map>

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
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"

#include "ast_matchers.h"
#include "class_gen.h"

using namespace clang;

/**
\brief for each method MainClass::kernel_XXX
*/
struct KernelInfo 
{
  struct Arg 
  {
    std::string type;
    std::string name;
    int         size;
  };
  std::string      return_type;
  std::string      name;
  std::vector<Arg> args;

  const CXXMethodDecl* astNode = nullptr;
};

/**
\brief for data member of MainClass
*/
struct DataMemberInfo 
{
  std::string name;
  std::string type;
  size_t      sizeInBytes;              // may be not needed due to using sizeof in generated code, but it is useful for sorting members by size and making apropriate aligment
  size_t      offsetInTargetBuffer = 0; // offset in bytes in terget buffer that stores all data members
  
  bool isContainer = false;
  std::string containerType;
  std::string containerDataType;
};


// RecursiveASTVisitor is the big-kahuna visitor that traverses everything in the AST.
//
class MyRecursiveASTVisitor : public RecursiveASTVisitor<MyRecursiveASTVisitor>
{
public:
  
  std::string MAIN_NAME;
  std::string MAIN_CLASS_NAME;

  MyRecursiveASTVisitor(std::string main_name, std::string main_class, const ASTContext& a_astContext) : MAIN_NAME(main_name), MAIN_CLASS_NAME(main_class), m_mainFuncNode(nullptr), m_astContext(a_astContext)  { }
  
  bool VisitCXXMethodDecl(CXXMethodDecl* f);
  bool VisitFieldDecl    (FieldDecl* var);

  std::unordered_map<std::string, KernelInfo>     functions;
  std::unordered_map<std::string, DataMemberInfo> dataMembers;
  const CXXMethodDecl* m_mainFuncNode;

private:
  void ProcessKernelDef(const CXXMethodDecl *f);
  void ProcessMainFunc(const CXXMethodDecl *f);

  const ASTContext& m_astContext;
};

KernelInfo::Arg ProcessParameter(ParmVarDecl *p) {
  QualType q = p->getType();
  const Type *typ = q.getTypePtr();
  KernelInfo::Arg arg;
  arg.name = p->getNameAsString();
  arg.type = QualType::getAsString(q.split(), PrintingPolicy{ {} });
  arg.size = 1;
  if (typ->isPointerType()) {
    arg.size = 1; // Because C always pass reference
  }
  
  return arg;
}

void MyRecursiveASTVisitor::ProcessKernelDef(const CXXMethodDecl *f) 
{
  if (!f || !f->hasBody()) 
    return;
  
  KernelInfo info;
  DeclarationNameInfo dni = f->getNameInfo();
  DeclarationName dn = dni.getName();
  info.name = dn.getAsString();
  QualType q = f->getReturnType();
  //const Type *typ = q.getTypePtr();
  info.return_type = QualType::getAsString(q.split(), PrintingPolicy{ {} });
  info.astNode     = f;

  for (unsigned int i = 0; i < f->getNumParams(); ++i) {
    info.args.push_back(ProcessParameter(f->parameters()[i]));
  }
  functions[info.name] = info;
}

void MyRecursiveASTVisitor::ProcessMainFunc(const CXXMethodDecl *f)
{
  m_mainFuncNode = f;
}

bool MyRecursiveASTVisitor::VisitCXXMethodDecl(CXXMethodDecl* f) 
{
  if (f->hasBody())
  {
    // Get name of function
    const DeclarationNameInfo dni = f->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();

    if(fname.find("kernel_") != std::string::npos)
    {
      const QualType qThisType       = f->getThisType();   
      const QualType classType       = qThisType.getTypePtr()->getPointeeType();
      const std::string thisTypeName = classType.getAsString();
      
      if(thisTypeName == std::string("class ") + MAIN_CLASS_NAME || thisTypeName == std::string("struct ") + MAIN_CLASS_NAME)
      {
        ProcessKernelDef(f);
        std::cout << "found kernel:\t" << fname.c_str() << " of type:\t" << thisTypeName.c_str() << std::endl;
      }
    }
    else if(fname == MAIN_NAME)
    {
      ProcessMainFunc(f);
      std::cout << "main function has found:\t" << fname.c_str() << std::endl;
    }
  }

  return true; // returning false aborts the traversal
}


bool MyRecursiveASTVisitor::VisitFieldDecl(FieldDecl* fd)
{
  const clang::RecordDecl* rd = fd->getParent();
  const clang::QualType    qt = fd->getType();
 
  const std::string& thisTypeName = rd->getName().str();

  if(thisTypeName == MAIN_CLASS_NAME)
  {
    std::cout << "found class data member: " << fd->getName().str().c_str() << " of type:\t" << qt.getAsString().c_str() << ", isPOD = " << qt.isCXX11PODType(m_astContext) << std::endl;

    DataMemberInfo member;
    member.name        = fd->getName().str();
    member.type        = qt.getAsString();
    member.sizeInBytes = 0; 
    member.offsetInTargetBuffer = 0;
    
    if(fd->getName().str().find("m_someBufferData") != std::string::npos)
    {
      int a = 2;
    }

    // now we should check werther this field is std::vector<XXX> or just XXX; 
    //
    const Type* fieldTypePtr = qt.getTypePtr(); //qt.getTypePtr();
    assert(fieldTypePtr != nullptr);

    //const char* typeClassName = fieldTypePtr->getTypeClassName();
    //std::cout << "typeClassName = " << typeClassName << std::endl;

    auto typeDecl = fieldTypePtr->getAsRecordDecl();
    
    if (typeDecl != nullptr && isa<ClassTemplateSpecializationDecl>(typeDecl)) 
    {
      auto specDecl = dyn_cast<ClassTemplateSpecializationDecl>(typeDecl); 
      assert(specDecl != nullptr);
      
      member.isContainer   = true;
      member.containerType = specDecl->getNameAsString();
      //member.containerDataType;
      
      const auto& templateArgs = specDecl->getTemplateArgs();
      
      if(templateArgs.size() > 0)
        member.containerDataType = templateArgs[0].getAsType().getAsString();
      else
        member.containerDataType = "unknown";

      std::cout << "container " << member.containerType.c_str() << " of " <<  member.containerDataType.c_str() << std::endl;
    }
    else
    {
      auto typeInfo      = m_astContext.getTypeInfo(qt);
      member.sizeInBytes = typeInfo.Width / 8; 
      member.offsetInTargetBuffer = 0;
    }
    

    dataMembers[member.name] = member;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MyASTConsumer : public ASTConsumer
{
 public:

  MyASTConsumer(std::string main_name, std::string main_class, const ASTContext& a_astContext) : rv(main_name, main_class, a_astContext) { }
  bool HandleTopLevelDecl(DeclGroupRef d) override;
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static llvm::cl::OptionCategory GDOpts("global-detect options");

const char * addl_help = "Report all functions that use global variable, or all sites at which "
                         "global variables are used";

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

clang::LangOptions lopt;

std::string GetKernelSourceCode(const clang::CXXMethodDecl* node, clang::SourceManager& sm, const std::vector<std::string>& threadIdNames) 
{
  clang::SourceLocation b(node->getBeginLoc()), _e(node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, sm, lopt));
  std::string methodSource = std::string(sm.getCharacterData(b), sm.getCharacterData(e));

  std::stringstream strOut;
  strOut << "{" << std::endl;
  strOut << "  /////////////////////////////////////////////////" << std::endl;
  for(size_t i=0;i<threadIdNames.size();i++)
    strOut << "  const uint " << threadIdNames[i].c_str() << " = get_global_id(" << i << ");"<< std::endl;
  strOut << "  if (tid >= iNumElements)" << std::endl;
  strOut << "    return;" << std::endl;
  strOut << "  /////////////////////////////////////////////////" << std::endl;
  return strOut.str() + methodSource.substr(methodSource.find_first_of('{')+1);
}

void ReplaceOpenCLBuiltInTypes(std::string& a_typeName)
{
  std::string lmStucts("struct LiteMath::");
  auto found1 = a_typeName.find(lmStucts);
  if(found1 != std::string::npos)
    a_typeName.replace(found1, lmStucts.length(), "");

  //std::string stucts("struct");
  //auto found2 = a_typeName.find(stucts);
  //if(found2 != std::string::npos)
  //  a_typeName.replace(found2, stucts.length(), "");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<std::string, std::string> ReadCommandLineParams(int argc, const char** argv, std::string& fileName)
{
  std::unordered_map<std::string, std::string> cmdLineParams;
  for(int i=0; i<argc; i++)
  {
    std::string key(argv[i]);
    if(key.size() > 0 && key[0]=='-')
    {
      if(i != argc-1) // not last argument
      {
        cmdLineParams[key] = argv[i+1];
        i++;
      }
      else
        cmdLineParams[key] = "";
    }
    else if(key.find(".cpp") != std::string::npos)
      fileName = key;
  }
  return cmdLineParams;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PrintKernelToCL(std::ostream& outFileCL, const KernelInfo& funcInfo, const std::string& kernName, clang::SourceManager& sm)
{
  assert(funcInfo.astNode != nullptr);

  bool foundThreadIdX = false; std::string tidXName = "tid";
  bool foundThreadIdY = false; std::string tidYName = "tid2";
  bool foundThreadIdZ = false; std::string tidZName = "tid3";

  outFileCL << std::endl;
  outFileCL << "__kernel void " << kernName.c_str() << "(" << std::endl;
  for (const auto& arg : funcInfo.args) 
  {
    std::string typeStr = arg.type.c_str();
    ReplaceOpenCLBuiltInTypes(typeStr);

    bool skip = false;

    if(arg.name == "tid" || arg.name == "tidX") // todo: check several names ... 
    {
      skip           = true;
      foundThreadIdX = true;
      tidXName       = arg.name;
    }

    if(arg.name == "tidY") // todo: check several names ... 
    {
      skip           = true;
      foundThreadIdY = true;
      tidYName       = arg.name;
    }

    if(arg.name == "tidZ") // todo: check several names ... 
    {
      skip           = true;
      foundThreadIdZ = true;
      tidZName       = arg.name;
    }
    
    if(!skip)
      outFileCL << "  __global " << typeStr.c_str() << " restrict " << arg.name.c_str() << "," << std::endl;
  }
  
  outFileCL << "  const uint iNumElements)" << std::endl;

  std::vector<std::string> threadIdNames;

  if(foundThreadIdX)
    threadIdNames.push_back(tidXName);

  if(foundThreadIdY)
    threadIdNames.push_back(tidYName);

  if(foundThreadIdZ)
    threadIdNames.push_back(tidZName);

  std::string sourceCode = GetKernelSourceCode(funcInfo.astNode, sm, threadIdNames);
  outFileCL << sourceCode.c_str() << std::endl << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, const char **argv)
{
  struct stat sb;

  if (argc < 2)
  {
    llvm::errs() << "Usage: <filename>\n";
    return 1;
  }
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::string fileName;
  auto params = ReadCommandLineParams(argc, argv, fileName);

  std::string mainClassName = "TestClass";
  std::string mainFuncName  = "PathTrace";
  std::string outGenerated  = "data/generated.cl";
  std::string stdlibFolder  = "";

  
  if(params.find("-mainClass") != params.end())
    mainClassName = params["-mainClass"];

  if(params.find("-mainFunc") != params.end())
    mainFuncName = params["-mainFunc"];

  if(params.find("-out") != params.end())
    outGenerated = params["-out"];

  if(params.find("-stdlibfolder") != params.end())
    stdlibFolder = params["-stdlibfolder"];

  llvm::ArrayRef<const char*> args(argv+1, argv+argc);

  // Make sure it exists
  if (stat(fileName.c_str(), &sb) == -1)
  {
    perror(fileName.c_str());
    exit(EXIT_FAILURE);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
  pto->Triple     = llvm::sys::getDefaultTargetTriple();
  TargetInfo *pti = TargetInfo::CreateTargetInfo(compiler.getDiagnostics(), pto);
  compiler.setTarget(pti);

  compiler.createFileManager();
  compiler.createSourceManager(compiler.getFileManager());

  // <Warning!!> -- Begin of Platform Specific Code
  HeaderSearchOptions &headerSearchOptions = compiler.getHeaderSearchOpts();  
  headerSearchOptions.AddPath(stdlibFolder.c_str(), clang::frontend::Angled, false, false);

  // headerSearchOptions.AddPath("/usr/local/include", clang::frontend::Angled, false, false);
  // headerSearchOptions.AddPath("/usr/include",       clang::frontend::Angled, false, false);
  // //headerSearchOptions.AddPath("/usr/include/c++/7.5.0", clang::frontend::Angled, false, false);
  // //headerSearchOptions.AddPath("/usr/include/clang/10/include", clang::frontend::Angled, false, false);

  // </Warning!!> -- End of Platform Specific Code


  // Allow C++ code to get rewritten
  LangOptions langOpts;
  langOpts.GNUMode = 1; 
  //langOpts.CXXExceptions = 1; 
  langOpts.RTTI        = 1; 
  langOpts.Bool        = 1; 
  langOpts.CPlusPlus   = 1; 
  langOpts.CPlusPlus14 = 1;
  langOpts.CPlusPlus17 = 1;
  compiler.createPreprocessor(clang::TU_Complete);
  compiler.getPreprocessorOpts().UsePredefines = false;
  compiler.getLangOpts() = langOpts;
  compiler.createASTContext();

  const FileEntry *pFile = compiler.getFileManager().getFile(fileName).get();
  compiler.getSourceManager().setMainFileID( compiler.getSourceManager().createFileID( pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(), &compiler.getPreprocessor());

  // (1) traverse source code of main file first
  //
  MyASTConsumer astConsumer(mainFuncName.c_str(), mainClassName.c_str(), compiler.getASTContext());

  // Convert <file>.c to <file_out>.c
  std::string outName (fileName);
  {
    size_t ext = outName.rfind(".");
    if (ext == std::string::npos)
       ext = outName.length();
    outName.insert(ext, "_out");
  }

  llvm::errs() << "Output to: " << outName << "\n";
  std::error_code OutErrorInfo;
  std::error_code ok;

  if (OutErrorInfo == ok)
  {
    // Parse the AST
    ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
    compiler.getDiagnosticClient().EndSourceFile();
  }
  else
  {
    llvm::errs() << "Cannot open " << outName << " for writing\n";
  }

  // (2) traverse only main function and rename kernel_ to cmd_
  {
    std::string mainFuncCode = kslicer::ProcessMainFunc(astConsumer.rv.m_mainFuncNode, compiler);
    std::ofstream fout(outName);
    fout << mainFuncCode.c_str() << std::endl;
  }

  // (3) write kernels to .cl file
  //
  {
    std::ofstream outFileCL(outGenerated.c_str());
    if(!outFileCL.is_open())
      llvm::errs() << "Cannot open " << outGenerated.c_str() << " for writing\n";

    for (const auto& a : astConsumer.rv.functions)  
    {
      std::cout << a.first << " " << a.second.return_type << std::endl;
      PrintKernelToCL(outFileCL, a.second, a.first, compiler.getSourceManager());
    }

    outFileCL.close();
  }

  // now process variables and kernel calls
  //
  {
    const char* argv2[] = {argv[0], argv[1], "--"};
    int argc2 = sizeof(argv2)/sizeof(argv2[0]);

    clang::tooling::CommonOptionsParser OptionsParser(argc2, argv2, GDOpts, addl_help);
    clang::tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    clang::ast_matchers::StatementMatcher local_var_matcher = kslicer::mk_local_var_matcher_of_function(mainFuncName.c_str());
    clang::ast_matchers::StatementMatcher kernel_matcher    = kslicer::mk_krenel_call_matcher_from_function(mainFuncName.c_str());
    
    kslicer::Global_Printer printer(std::cout);
    clang::ast_matchers::MatchFinder finder;
    
    finder.addMatcher(local_var_matcher, &printer);
    finder.addMatcher(kernel_matcher,    &printer);
  
    auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
  
    std::cout << "tool run res = " << res << std::endl;
  }

  return 0;
}