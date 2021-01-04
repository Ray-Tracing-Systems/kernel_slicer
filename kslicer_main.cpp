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

std::string kslicer::GetRangeSourceCode(const clang::SourceRange a_range, const clang::CompilerInstance& compiler) 
{
  const clang::SourceManager& sm = compiler.getSourceManager();
  const clang::LangOptions& lopt = compiler.getLangOpts();

  clang::SourceLocation b(a_range.getBegin()), _e(a_range.getEnd());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, sm, lopt));

  return std::string(sm.getCharacterData(b), sm.getCharacterData(e));
}

uint64_t kslicer::GetHashOfSourceRange(const clang::SourceRange& a_range)
{
  //const uint32_t hash1 = a_range.getBegin().getHashValue(); // getHashValue presents in clang 12, but not in clang 10!
  //const uint32_t hash2 = a_range.getEnd().getHashValue();   // getHashValue presents in clang 12, but not in clang 10!
  const uint32_t hash1 = a_range.getBegin().getRawEncoding(); // getRawEncoding presents in clang 10, what about clang 12?
  const uint32_t hash2 = a_range.getEnd().getRawEncoding();   // getRawEncoding presents in clang 10, what about clang 12?
  return (uint64_t(hash1) << 32) | uint64_t(hash2);
}

std::string kslicer::CutOffFileExt(const std::string& a_filePath)
{
  const size_t lastindex = a_filePath.find_last_of("."); 
  if(lastindex != std::string::npos)
    return a_filePath.substr(0, lastindex); 
  else
    return a_filePath;
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

std::vector<std::string> kslicer::GetAllPredefinedThreadIdNamesRTV()
{
  return {"tid", "tidX", "tidY", "tidZ"};
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

std::string GetFolderPath(const std::string& a_filePath);

const char* GetClangToolingErrorCodeMessage(int code)
{
  if(code == 0)
    return "OK";
  else if (code == 1)
    return "ERROR";
  else
    return "SKIPPED_FILES";  
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
  std::string outGenerated  = "data/generated.cl";
  std::string stdlibFolder  = "";
  std::string patternName   = "rtv";
  uint32_t    threadsOrder[3] = {0,1,2};
  
  if(params.find("-mainClass") != params.end())
    mainClassName = params["-mainClass"];

  if(params.find("-out") != params.end())
    outGenerated = params["-out"];

  if(params.find("-stdlibfolder") != params.end())
    stdlibFolder = params["-stdlibfolder"];

  if(params.find("-pattern") != params.end())
    patternName = params["-pattern"];

  std::vector<const char*> argsForClang; // exclude our input from cmdline parameters and pass the rest to clang
  argsForClang.reserve(argc);
  for(int i=1;i<argc;i++)
  {
    auto p = params.find(argv[i]);
    if(p == params.end()) 
      argsForClang.push_back(argv[i]);
  }
  llvm::ArrayRef<const char*> args(argsForClang.data(), argsForClang.data() + argsForClang.size());

  // Make sure it exists
  if (stat(fileName.c_str(), &sb) == -1)
  {
    std::cout << "[main]: error, input file " << fileName.c_str() << " not found!" << std::endl;
    return 0;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::shared_ptr<kslicer::MainClassInfo> pImplPattern = nullptr;
  if(patternName == "rtv")
    pImplPattern = std::make_shared<kslicer::RTV_Pattern>();
  else if(patternName == "ipv")
    pImplPattern = std::make_shared<kslicer::IPV_Pattern>();
  else
  { 
    std::cout << "wrong pattern name '" << patternName.c_str() << "' " << std::endl; 
    exit(0);
  }
  kslicer::MainClassInfo& inputCodeInfo = *pImplPattern;
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::unique_ptr<kslicer::MainClassInfo> pInputCodeInfoImpl = nullptr;

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

  {
    compiler.getLangOpts().GNUMode = 1; 
    //compiler.getLangOpts().CXXExceptions = 1; 
    compiler.getLangOpts().RTTI        = 1; 
    compiler.getLangOpts().Bool        = 1; 
    compiler.getLangOpts().CPlusPlus   = 1; 
    compiler.getLangOpts().CPlusPlus14 = 1;
    compiler.getLangOpts().CPlusPlus17 = 1;
  }

  compiler.createFileManager();
  compiler.createSourceManager(compiler.getFileManager());
  
  // (0) add path dummy include files for STL and e.t.c. (we don't want to parse actually std library)
  //
  HeaderSearchOptions &headerSearchOptions = compiler.getHeaderSearchOpts();  
  headerSearchOptions.AddPath(stdlibFolder.c_str(), clang::frontend::Angled, false, false);

  compiler.createPreprocessor(clang::TU_Complete);
  compiler.getPreprocessorOpts().UsePredefines = false;
  compiler.createASTContext();

  const FileEntry *pFile = compiler.getFileManager().getFile(fileName).get();
  compiler.getSourceManager().setMainFileID( compiler.getSourceManager().createFileID( pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(), &compiler.getPreprocessor());
  
  // register our header lister
  {
    auto pHeaderLister = std::make_unique<HeaderLister>(&inputCodeInfo);
    compiler.getPreprocessor().addPPCallbacks(std::move(pHeaderLister));
  }
    
  // init clang tooling
  //
  const char* argv2[] = {argv[0], argv[1], "--"};
  int argc2 = sizeof(argv2)/sizeof(argv2[0]);
  
  llvm::cl::OptionCategory GDOpts("global-detect options");
  clang::tooling::CommonOptionsParser OptionsParser(argc2, argv2, GDOpts);
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

  // (0) find all "Main" functions, a functions which call kernels. Kernels are also listed for each mainFunc;
  //
  std::vector<std::string> mainFunctNames; 
  mainFunctNames.reserve(20);
  
  std::cout << "(0) Listing main functions of " << mainClassName.c_str()  << std::endl; 
  auto mainFuncList = kslicer::ListAllMainRTFunctions(Tool, mainClassName, compiler.getASTContext(), inputCodeInfo);
  std::cout << "{" << std::endl;
  for(const auto& f : mainFuncList)
  {
    std::cout << "  found " << f.first.c_str() << std::endl;
    mainFunctNames.push_back(f.first);
  }
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  inputCodeInfo.mainFunc.resize(mainFuncList.size());
  inputCodeInfo.mainClassName     = mainClassName;
  inputCodeInfo.mainClassFileName = fileName;

  std::cout << "(1) Processing class " << mainClassName.c_str() << std::endl; 
  std::cout << "{" << std::endl;

  // Parse code
  //
  kslicer::InitialPassASTConsumer astConsumer(mainFunctNames, mainClassName, compiler.getASTContext(), compiler.getSourceManager(), inputCodeInfo); 
  ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
  compiler.getDiagnosticClient().EndSourceFile(); // ??? What Is This Line For ???

  inputCodeInfo.allKernels           = astConsumer.rv.functions;    
  inputCodeInfo.allDataMembers       = astConsumer.rv.dataMembers;   
  inputCodeInfo.mainClassFileInclude = astConsumer.rv.MAIN_FILE_INCLUDE;

  std::unordered_set<std::string> processedKernels;
  size_t mainFuncId = 0;
  for(const auto f : mainFuncList)
  {
    const std::string& mainFuncName = f.first;
    auto& mainFuncRef = inputCodeInfo.mainFunc[mainFuncId];
    mainFuncRef.Name  = mainFuncName;
    mainFuncRef.Node  = astConsumer.rv.m_mainFuncNodes[mainFuncName];

    // Now process each main function: variables and kernel calls, if()->break and if()->return statements.
    //
    {
      auto allMatchers = inputCodeInfo.ListMatchers_CF(mainFuncName);
      auto pMatcherPrc = inputCodeInfo.MatcherHandler_CF(mainFuncRef, compiler);

      clang::ast_matchers::MatchFinder finder;
      for(auto& matcher : allMatchers)
        finder.addMatcher(matcher, pMatcherPrc.get());
      
      std::cout << "  process main function: " << mainFuncName.c_str() << "(...): ";
      auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
      std::cout << GetClangToolingErrorCodeMessage(res) << std::endl;
      
      // filter out unused kernels
      //
      inputCodeInfo.kernels.reserve(inputCodeInfo.allKernels.size());
      for (const auto& k : inputCodeInfo.allKernels)
      {
        if(k.second.usedInMainFunc && processedKernels.find(k.first) == processedKernels.end())
        {
          inputCodeInfo.kernels.push_back(k.second);
          processedKernels.insert(k.first);
        }
      }

      // filter out excluded local variables
      //
      for(const auto& var : mainFuncRef.ExcludeList)
      {
        auto ex = mainFuncRef.Locals.find(var);
        if(ex != mainFuncRef.Locals.end())
          mainFuncRef.Locals.erase(ex);
      }
    }

    mainFuncId++;
  }
  
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(2) Mark data members, methods and functions which are actually used in kernels." << std::endl; 
  std::cout << "{" << std::endl;
  
  std::unordered_map<std::string, clang::SourceRange> usedFunctions_KF; // ==> accumulate information about used by kernel (KF) functions here
  std::unordered_map<std::string, bool>               usedFiles_KF;     // ==> accumulate information about used by kernel (KF) files     here

  for(auto& kernel : inputCodeInfo.kernels)
  {
    auto kernelMatchers = inputCodeInfo.ListMatchers_KF(kernel.name);
    auto pFilter        = inputCodeInfo.MatcherHandler_KF(kernel, compiler);

    clang::ast_matchers::MatchFinder finder;
    for(auto& matcher : kernelMatchers)
       finder.addMatcher(matcher, pFilter.get());

    auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    std::cout << "  process " << kernel.name.c_str() << ":\t" << GetClangToolingErrorCodeMessage(res) << std::endl;

    for(auto& arg : kernel.args) // it is important to run this loop after second stage at which kernel matchers are applied!
      inputCodeInfo.ProcessKernelArg(arg, kernel);

    usedFunctions_KF.merge(pFilter->usedFunctions);
    usedFiles_KF.merge(pFilter->usedFiles);
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(3) Process control functions to generate all 'MainCmd' functions" << std::endl; 
  std::cout << "{" << std::endl;

  inputCodeInfo.AddSpecVars_CF(inputCodeInfo.mainFunc, inputCodeInfo.kernels);

  // (5) genarate cpp code with Vulkan calls
  //
  ObtainKernelsDecl(inputCodeInfo.kernels, compiler, inputCodeInfo.mainClassName, inputCodeInfo);
  inputCodeInfo.allDescriptorSetsInfo.clear();
  for(auto& mainFunc : inputCodeInfo.mainFunc)
  {
    std::cout << "  process " << mainFunc.Name.c_str() << std::endl;

    mainFunc.CodeGenerated = inputCodeInfo.VisitAndRewrite_CF(mainFunc, compiler);
    mainFunc.InOuts        = kslicer::ListPointerParamsOfMainFunc(mainFunc.Node);
  }

  inputCodeInfo.PlugSpecVarsInCalls_CF(inputCodeInfo.mainFunc, inputCodeInfo.kernels, // ==>
                                       inputCodeInfo.allDescriptorSetsInfo);          // <==

  std::cout << "}" << std::endl;
  std::cout << std::endl;
  

  
  std::cout << "(4) Calc offsets for all class members; ingore unused members that were not marked on previous step" << std::endl; 
  std::cout << "{" << std::endl;

  // (4) calc offsets for all class variables; ingore unused members that were not marked on previous step
  //
  {
    inputCodeInfo.dataMembers = kslicer::MakeClassDataListAndCalcOffsets(inputCodeInfo.allDataMembers);
    std::cout << "  placed classVariables num = " << inputCodeInfo.dataMembers.size() << std::endl;
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;
  

  std::cout << "(5) Perform final templated text rendering to generate Vulkan calls" << std::endl; 
  std::cout << "{" << std::endl;
  {
    kslicer::PrintVulkanBasicsFile("templates/vulkan_basics.h", inputCodeInfo);

    std::string rawname = kslicer::CutOffFileExt(inputCodeInfo.mainClassFileName);
    auto json = PrepareJsonForAllCPP(inputCodeInfo, inputCodeInfo.mainFunc, rawname + "_generated.h", threadsOrder); 
    
    kslicer::ApplyJsonToTemplate("templates/rt_class.h",      rawname + "_generated.h", json); 
    kslicer::ApplyJsonToTemplate("templates/rt_class.cpp",    rawname + "_generated.cpp", json);
    kslicer::ApplyJsonToTemplate("templates/rt_class_ds.cpp", rawname + "_generated_ds.cpp", json);
  }
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(6) Generate OpenCL kernels" << std::endl; 
  std::cout << "{" << std::endl;

  // analize inputCodeInfo.allDescriptorSetsInfo to mark all args of each kernel that we need to apply fakeOffset(tid) inside kernel to this arg
  //
  for(const auto& call : inputCodeInfo.allDescriptorSetsInfo)
    inputCodeInfo.ProcessCallArs_KF(call);

  for(auto& k : inputCodeInfo.kernels)
    k.rewrittenText = inputCodeInfo.VisitAndRewrite_KF(k, compiler);

  // finally generate kernels
  //
  kslicer::PrintGeneratedCLFile("templates/generated.cl", GetFolderPath(inputCodeInfo.mainClassFileName), inputCodeInfo, usedFiles_KF, usedFunctions_KF, compiler, threadsOrder);

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  return 0;
}