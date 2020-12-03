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

#include "kslicer.h"
#include "initial_pass.h"
#include "ast_matchers.h"
#include "class_gen.h"

using namespace clang;

#include "template_rendering.h"


const std::string kslicer::GetProjPrefix() { return std::string("kgen_"); };

using kslicer::KernelInfo;
using kslicer::DataMemberInfo;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static llvm::cl::OptionCategory GDOpts("global-detect options");
clang::LangOptions lopt;

std::string GetRangeSourceCode(const clang::SourceRange a_range, clang::SourceManager& sm) 
{
  clang::SourceLocation b(a_range.getBegin()), _e(a_range.getEnd());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, sm, lopt));
  return std::string(sm.getCharacterData(b), sm.getCharacterData(e));
}

void kslicer::ReplaceOpenCLBuiltInTypes(std::string& a_typeName)
{
  std::string lmStucts("struct LiteMath::");
  auto found1 = a_typeName.find(lmStucts);
  if(found1 != std::string::npos)
    a_typeName.replace(found1, lmStucts.length(), "");

  std::string lm("LiteMath::");
  auto found2 = a_typeName.find(lm);
  if(found2 != std::string::npos)
    a_typeName.replace(found2, lm.length(), "");
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

std::vector<std::string> kslicer::GetAllPredefinedThreadIdNames()
{
  return {"tid", "tidX", "tidY", "tidZ"};
}

void PrintKernelToCL(std::ostream& outFileCL, const KernelInfo& funcInfo, const std::string& kernName, clang::CompilerInstance& compiler, const kslicer::MainClassInfo& a_inputCodeInfo)
{
  assert(funcInfo.astNode != nullptr);

  clang::SourceManager& sm = compiler.getSourceManager();

  bool foundThreadIdX = false; std::string tidXName = "tid";
  bool foundThreadIdY = false; std::string tidYName = "tid2";
  bool foundThreadIdZ = false; std::string tidZName = "tid3";

  outFileCL << std::endl;
  outFileCL << "__kernel void " << kernName.c_str() << "(" << std::endl;
  for (const auto& arg : funcInfo.args) 
  {
    std::string typeStr = arg.type.c_str();
    kslicer::ReplaceOpenCLBuiltInTypes(typeStr);

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
  
  std::vector<std::string> threadIdNames;
  {
    if(foundThreadIdX)
      threadIdNames.push_back(tidXName);
    if(foundThreadIdY)
      threadIdNames.push_back(tidYName);
    if(foundThreadIdZ)
      threadIdNames.push_back(tidZName);
  }

  const std::string numThreadsName = kslicer::GetProjPrefix() + "iNumElements";
  const std::string m_dataName     = kslicer::GetProjPrefix() + "data";

  outFileCL << "  __global const uint* restrict " << m_dataName.c_str() << "," << std::endl;

  if(threadIdNames.size() == 0)
    outFileCL << "  const uint " << numThreadsName.c_str() << ")" << std::endl;
  else
  {
    const char* XYZ[] = {"X","Y","Z"};
    for(size_t i=0;i<threadIdNames.size();i++)
    {
      outFileCL << "  const uint " << numThreadsName.c_str() << XYZ[i];
      if(i != threadIdNames.size()-1)
        outFileCL << "," << std::endl;
      else
        outFileCL << ")" << std::endl;
    }
  }


  std::string sourceCodeFull = kslicer::ProcessKernel(funcInfo.astNode, compiler, a_inputCodeInfo);
  std::string sourceCodeCut  = sourceCodeFull.substr(sourceCodeFull.find_first_of('{')+1);

  std::stringstream strOut;
  {
    strOut << "{" << std::endl;
    strOut << "  /////////////////////////////////////////////////" << std::endl;
    for(size_t i=0;i<threadIdNames.size();i++)
      strOut << "  const uint " << threadIdNames[i].c_str() << " = get_global_id(" << i << ");"<< std::endl; 
    
    if(threadIdNames.size() == 1)
    {
      strOut << "  if (" << threadIdNames[0].c_str() << " >= " << numThreadsName.c_str() << ")" << std::endl;                          
      strOut << "    return;" << std::endl;
    }
    else if(threadIdNames.size() == 2)
    {
      strOut << "  if (" << threadIdNames[0].c_str() << " >= " << numThreadsName.c_str() << "X";
      strOut << " || "   << threadIdNames[1].c_str() << " >= " << numThreadsName.c_str() << "Y" <<  ")" << std::endl;                          
      strOut << "    return;" << std::endl;
    }
    else if(threadIdNames.size() == 3)
    {
      strOut << "  if (" << threadIdNames[0].c_str() << " >= " << numThreadsName.c_str() << "X";
      strOut << " || "   << threadIdNames[1].c_str() << " >= " << numThreadsName.c_str() << "Y";  
      strOut << " || "   << threadIdNames[2].c_str() << " >= " << numThreadsName.c_str() << "Z" <<  ")" << std::endl;                        
      strOut << "    return;" << std::endl;
    }
    else
    {
      assert(false);
    }
    
    
    strOut << "  /////////////////////////////////////////////////" << std::endl;
  }

  outFileCL << strOut.str() << sourceCodeCut.c_str() << std::endl << std::endl;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class HeaderLister : public clang::PPCallbacks 
{
public:

  HeaderLister(kslicer::MainClassInfo* a_pInfo) : m_pGlobInfo(a_pInfo) {}

  void InclusionDirective(clang::SourceLocation HashLoc,
                          const clang::Token &IncludeTok,
                          llvm::StringRef FileName, bool IsAngled,
                          clang::CharSourceRange FilenameRange,
                          const clang::FileEntry *File,
                          llvm::StringRef SearchPath,
                          llvm::StringRef RelativePath,
                          const clang::Module *Imported,
                          clang::SrcMgr::CharacteristicKind FileType) override
  {
    if(!IsAngled)
    {
      assert(File != nullptr);
      std::string filename = std::string(RelativePath.begin(), RelativePath.end()); 
      m_pGlobInfo->allIncludeFiles[filename] = false;   
    }
  }

private:

  kslicer::MainClassInfo* m_pGlobInfo;

};

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
  std::string outGenerated  = "data/generated.cl";
  std::string stdlibFolder  = "";
  
  if(params.find("-mainClass") != params.end())
    mainClassName = params["-mainClass"];

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

  kslicer::MainClassInfo inputCodeInfo;

  CompilerInstance compiler;
  DiagnosticOptions diagnosticOptions;
  compiler.createDiagnostics();  //compiler.createDiagnostics(argc, argv);

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
  
  // (0) add path dummy include files for STL and e.t.c. (we don't want to parse actually std library)
  //
  HeaderSearchOptions &headerSearchOptions = compiler.getHeaderSearchOpts();  
  headerSearchOptions.AddPath(stdlibFolder.c_str(), clang::frontend::Angled, false, false);

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
  
  // register our header lister
  {
    auto pHeaderLister = std::make_unique<HeaderLister>(&inputCodeInfo);
    compiler.getPreprocessor().addPPCallbacks(std::move(pHeaderLister));
  }

  ////////////////////////////////////////////////////////////////////////
  std::string outName(fileName);
  {
    size_t ext = outName.rfind(".");
    if (ext == std::string::npos)
       ext = outName.length();
    outName.insert(ext, "_out");
  }
  llvm::errs() << "Output to: " << outName << "\n";
  ////////////////////////////////////////////////////////////////////////
    
  // init clang tooling
  //
  const char* argv2[] = {argv[0], argv[1], "--"};
  int argc2 = sizeof(argv2)/sizeof(argv2[0]);

  clang::tooling::CommonOptionsParser OptionsParser(argc2, argv2, GDOpts);
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

  // (0) find all "Main" functions, a functions which call kernels. Kerels are also listed for each mainFunc;
  //
  std::cout << "Seeking for MainFunc of " << mainClassName.c_str() << std::endl;
  auto MainFuncList = kslicer::ListAllMainRTFunctions(Tool, mainClassName, compiler.getASTContext());

  inputCodeInfo.mainFunc.resize(MainFuncList.size());
  inputCodeInfo.mainClassName = mainClassName;

  size_t mainFuncId = 0;
  for(const auto f : MainFuncList)
  {
    const std::string& mainFuncName = f.first;
    
    // (1) traverse source code of main file first
    //
    std::cout << "Traversing " << mainClassName.c_str() << std::endl; 
    {
      kslicer::InitialPassASTConsumer astConsumer(mainFuncName.c_str(), mainClassName.c_str(), compiler.getASTContext(), compiler.getSourceManager());  
      ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
      compiler.getDiagnosticClient().EndSourceFile();
    
      inputCodeInfo.allKernels     = astConsumer.rv.functions;    // TODO: exclude repeating!!!
      inputCodeInfo.allDataMembers = astConsumer.rv.dataMembers;  // TODO: exclude repeating!!!     
      inputCodeInfo.mainClassFileName    = fileName;
      inputCodeInfo.mainClassFileInclude = astConsumer.rv.MAIN_FILE_INCLUDE;

      inputCodeInfo.mainFunc[mainFuncId].Name = mainFuncName;
      inputCodeInfo.mainFunc[mainFuncId].Node = astConsumer.rv.m_mainFuncNode;
    }
  
    // (2) now process variables and kernel calls of main function
    //
    {
      clang::ast_matchers::StatementMatcher local_var_matcher = kslicer::MakeMatch_LocalVarOfMethod(mainFuncName.c_str());
      clang::ast_matchers::StatementMatcher kernel_matcher    = kslicer::MakeMatch_MethodCallFromMethod(mainFuncName.c_str());
      
      kslicer::MainFuncAnalyzer printer(std::cout, inputCodeInfo, compiler.getASTContext(), mainFuncId);
      clang::ast_matchers::MatchFinder finder;
      
      finder.addMatcher(local_var_matcher, &printer);
      finder.addMatcher(kernel_matcher,    &printer);
     
      auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
      std::cout << "tool run res = " << res << std::endl;
      
      // filter out unused kernels
      //
      inputCodeInfo.kernels.reserve(inputCodeInfo.allKernels.size());
      for (const auto& k : inputCodeInfo.allKernels)
        if(k.second.usedInMainFunc)
          inputCodeInfo.kernels.push_back(k.second);
    }

    mainFuncId++;
  }

  // (3) now mark all data members, methods and functions which are actually used in kernels; we will ignore others. 
  //
  kslicer::VariableAndFunctionFilter filter(std::cout, inputCodeInfo, compiler.getSourceManager());
  { 
    for(const auto& kernel : inputCodeInfo.kernels)
    {
      clang::ast_matchers::StatementMatcher dataMemberMatcher = kslicer::MakeMatch_MemberVarOfMethod(kernel.name);
      clang::ast_matchers::StatementMatcher funcMatcher       = kslicer::MakeMatch_FunctionCallFromFunction(kernel.name);
  
      clang::ast_matchers::MatchFinder finder;
      finder.addMatcher(dataMemberMatcher, &filter);
      finder.addMatcher(funcMatcher, &filter);
    
      auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
      std::cout << "[filter] for " << kernel.name.c_str() << ";\ttool run res = " << res << std::endl;
    }

    //inputCodeInfo.localFunctions = filter.GetUsedFunctions();
  }

  // (4) calc offsets for all class variables; ingore unused members that were not marked on previous step
  //
  {
    inputCodeInfo.dataMembers = kslicer::MakeClassDataListAndCalcOffsets(inputCodeInfo.allDataMembers);
    std::cout << "placed classVariables num = " << inputCodeInfo.dataMembers.size() << std::endl;
  }

  // (5) ...
  // 
  std::ofstream outFileCL(outGenerated.c_str());
  if(!outFileCL.is_open())
    llvm::errs() << "Cannot open " << outGenerated.c_str() << " for writing\n";

  // list include files
  //
  outFileCL << "/////////////////////////////////////////////////////////////////////" << std::endl;
  outFileCL << "/////////////////// include files ///////////////////////////////////" << std::endl;
  outFileCL << "/////////////////////////////////////////////////////////////////////" << std::endl;
  outFileCL << std::endl;

  for(auto keyVal : inputCodeInfo.allIncludeFiles) // we will search for only used include files among all of them (quoted, angled were excluded earlier)
  {
    for(auto keyVal2 : filter.usedFiles)
    {
      if(keyVal2.first.find(keyVal.first) != std::string::npos)
      {
        //std::cout << "[include]: " << keyVal.first.c_str() << " = " << keyVal.second << std::endl;
        outFileCL << "#include \"" << keyVal.first.c_str() << "\"" << std::endl;
      }
    }
  }
  outFileCL << std::endl;

  outFileCL << "/////////////////////////////////////////////////////////////////////" << std::endl;
  outFileCL << "/////////////////// local functions /////////////////////////////////" << std::endl;
  outFileCL << "/////////////////////////////////////////////////////////////////////" << std::endl;
  outFileCL << std::endl;

  // (6) write local functions to .cl file
  //
  for (const auto& f : filter.usedFunctions)  
  {
    std::string funcSourceCode = GetRangeSourceCode(f.second, compiler.getSourceManager());
    outFileCL << funcSourceCode.c_str() << std::endl;
  }

  outFileCL << std::endl;
  outFileCL << "/////////////////////////////////////////////////////////////////////" << std::endl;
  outFileCL << "/////////////////// kernels /////////////////////////////////////////" << std::endl;
  outFileCL << "/////////////////////////////////////////////////////////////////////" << std::endl;
  outFileCL << std::endl;

  // (7) write kernels to .cl file
  //
  {
    for (const auto& k : inputCodeInfo.kernels)  
    {
      std::cout << k.name << " " << k.return_type << std::endl;
      PrintKernelToCL(outFileCL, k, k.name, compiler, inputCodeInfo);
    }

    outFileCL.close();
  }


  // ??? // at this step we must filter data variables to store only those which are referenced inside kernels calls
  // ??? //
  
  // (8) genarate cpp code with Vulkan calls
  //
  ObtainKernelsDecl(inputCodeInfo.kernels, compiler.getSourceManager(), inputCodeInfo.mainClassName);
  inputCodeInfo.allDescriptorSetsInfo.clear();
  for(auto& mainFunc : inputCodeInfo.mainFunc)
  {
    mainFunc.CodeGenerated = inputCodeInfo.ProcessMainFunc_RTCase(mainFunc, compiler,
                                                           inputCodeInfo.allDescriptorSetsInfo);
  
    mainFunc.InOuts = kslicer::ListPointerParamsOfMainFunc(mainFunc.Node);
  }
  
  // (9) genarate cpp code with Vulkan calls using template text rendering and appropriate template FOR ALL 'mainFunc'-tions
  //
  {
    kslicer::PrintVulkanBasicsFile  ("templates/vulkan_basics.h", inputCodeInfo);
    const std::string fileName = \
    kslicer::PrintGeneratedClassDecl("templates/rt_class.h", inputCodeInfo, inputCodeInfo.mainFunc);
    kslicer::PrintGeneratedClassImpl("templates/rt_class.cpp", fileName, inputCodeInfo, inputCodeInfo.mainFunc); 
  }

  return 0;
}