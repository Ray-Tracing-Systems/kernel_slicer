#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <vector>
#include <system_error>
#include <iostream>
#include <fstream>

#include <unordered_map>
#include <iomanip>

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
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclTemplate.h"

#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"

#include "kslicer.h"
#include "initial_pass.h"
#include "ast_matchers.h"
#include "class_gen.h"
#include "extractor.h"

using namespace clang;
#include "template_rendering.h"

#ifdef _WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
#endif

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

std::string kslicer::MainClassInfo::RemoveTypeNamespaces(const std::string& a_str) const
{
  std::string typeName = a_str;
  ReplaceOpenCLBuiltInTypes(typeName);
  ReplaceFirst(typeName, mainClassName + "::", "");
  return typeName;
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

void ReadThreadsOrderFromStr(const std::string& threadsOrderStr, uint32_t  threadsOrder[3])
{
  auto size = std::min<size_t>(threadsOrderStr.size(), 3);
  for(size_t symbId = 0; symbId < size; symbId++)
  {
    switch(threadsOrderStr[symbId])
    {
      case 'x':
      case 'X':
      case '0':
      threadsOrder[symbId] = 0;
      break;
      case 'y':
      case 'Y':
      case '1':
      threadsOrder[symbId] = 1;
      break;
      case 'z':
      case 'Z':
      case '2':
      threadsOrder[symbId] = 2;
      break;
      default:
      break;
    };
  }
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
  std::string shaderCCName  = "clspv";
  std::string hintFile      = "";
  uint32_t    threadsOrder[3] = {0,1,2};
  
  if(params.find("-mainClass") != params.end())
    mainClassName = params["-mainClass"];

  if(params.find("-out") != params.end())
    outGenerated = params["-out"];

  if(params.find("-stdlibfolder") != params.end())
    stdlibFolder = params["-stdlibfolder"];

  if(params.find("-pattern") != params.end())
    patternName = params["-pattern"];

  if(params.find("-reorderLoops") != params.end())
    ReadThreadsOrderFromStr(params["-reorderLoops"], threadsOrder);

  if(params.find("-shaderCC") != params.end())
    shaderCCName = params["-shaderCC"];

  if(params.find("-hint") != params.end())
    hintFile = params["-hint"];

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
  if(shaderCCName == "circle" || shaderCCName == "Circle")
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::CircleCompiler>();
  else
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::ClspvCompiler>();
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

  std::cout << "(1) Processing class " << mainClassName.c_str() << " with initial pass" << std::endl; 
  std::cout << "{" << std::endl;

  // Parse code
  //
  kslicer::InitialPassASTConsumer astConsumer(mainFunctNames, mainClassName, compiler.getASTContext(), compiler.getSourceManager(), inputCodeInfo); 
  ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
  compiler.getDiagnosticClient().EndSourceFile(); // ??? What Is This Line For ???

  inputCodeInfo.allKernels           = astConsumer.rv.functions;    
  inputCodeInfo.allDataMembers       = astConsumer.rv.dataMembers;   
  inputCodeInfo.mainClassFileInclude = astConsumer.rv.MAIN_FILE_INCLUDE;
  inputCodeInfo.mainClassASTNode     = astConsumer.rv.m_mainClassASTNode;

  if(inputCodeInfo.mainClassASTNode == nullptr)
  {
    std::cout << "[main]: ERROR, main class " << mainClassName.c_str() << " not found" << std::endl;
    return 0;
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(2) Process control functions; extract local variables, known calls like memcpy,sort,std::fill and other " << std::endl; 
  std::cout << "{" << std::endl;

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
        finder.addMatcher(clang::ast_matchers::traverse(clang::ast_type_traits::TK_IgnoreUnlessSpelledInSource,matcher), pMatcherPrc.get());
      
      std::cout << "  process control function: " << mainFuncName.c_str() << "(...): ";
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

  std::cout << "(3) Mark data members, methods and functions which are actually used in kernels." << std::endl; 
  std::cout << "{" << std::endl;

  for(auto& kernel : inputCodeInfo.kernels)
  {
    auto kernelMatchers = inputCodeInfo.ListMatchers_KF(kernel.name);
    auto pFilter        = inputCodeInfo.MatcherHandler_KF(kernel, compiler);

    clang::ast_matchers::MatchFinder finder;
    for(auto& matcher : kernelMatchers)
      finder.addMatcher(clang::ast_matchers::traverse(clang::ast_type_traits::TK_IgnoreUnlessSpelledInSource, matcher), pFilter.get());

    auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    std::cout << "  process " << kernel.name.c_str() << ":\t" << GetClangToolingErrorCodeMessage(res) << std::endl;

    for(auto& arg : kernel.args) // it is important to run this loop after second stage at which kernel matchers are applied!
      inputCodeInfo.ProcessKernelArg(arg, kernel);

    //for(auto& initStm : kernel.loopInitStatements)
    //  inputCodeInfo.allLoopInitStatements.insert(kslicer::GetHashOfSourceRange(initStm.srcRange));
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::vector<kslicer::FuncData>    usedByKernelsFunctions = kslicer::ExtractUsedFunctions(inputCodeInfo, compiler); // recursive processing of functions used by kernel, extracting all needed functions
  std::vector<kslicer::DeclInClass> usedDecls;
  
  std::cout << "(4) Extract constants and structs from 'MainClass' " << std::endl; 
  std::cout << "{" << std::endl;
  {
    auto structMatcher = kslicer::MakeMatch_StructDeclInsideClass(inputCodeInfo.mainClassName);
    auto varMatcher    = kslicer::MakeMatch_VarDeclInsideClass(inputCodeInfo.mainClassName);
    auto tpdefMatcher  = kslicer::MakeMatch_TypedefInsideClass(inputCodeInfo.mainClassName);
    
    clang::ast_matchers::MatchFinder finder;
    kslicer::TC_Extractor typeAndConstantsHandler(inputCodeInfo, compiler);
    finder.addMatcher(clang::ast_matchers::traverse(clang::ast_type_traits::TK_IgnoreUnlessSpelledInSource, structMatcher), &typeAndConstantsHandler);
    finder.addMatcher(clang::ast_matchers::traverse(clang::ast_type_traits::TK_IgnoreUnlessSpelledInSource, varMatcher),    &typeAndConstantsHandler);
    finder.addMatcher(clang::ast_matchers::traverse(clang::ast_type_traits::TK_IgnoreUnlessSpelledInSource, tpdefMatcher),  &typeAndConstantsHandler);

    auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    std::cout << "  [TC_Extractor]: end process constants and structs:\t" << GetClangToolingErrorCodeMessage(res) << std::endl;

    usedDecls.resize(0);
    usedDecls.reserve(typeAndConstantsHandler.foundDecl.size());
    for(const auto decl : typeAndConstantsHandler.foundDecl)
      usedDecls.push_back(decl.second);

    std::sort(usedDecls.begin(), usedDecls.end(), [](const auto& a, const auto& b) { return a.order < b.order; } );
    usedDecls = kslicer::ExtractUsedTC(usedDecls, inputCodeInfo, compiler);
  }
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(5) Process control functions to generate all 'MainCmd' functions" << std::endl; 
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
  
  std::cout << "(6) Calc offsets for all class members; ingore unused members that were not marked on previous step" << std::endl; 
  std::cout << "{" << std::endl;

  // (4) calc offsets for all class variables; ingore unused members that were not marked on previous step
  //
  inputCodeInfo.dataMembers  = kslicer::MakeClassDataListAndCalcOffsets(inputCodeInfo.allDataMembers);
  auto jsonUBO               = kslicer::PrepareUBOJson(inputCodeInfo, inputCodeInfo.dataMembers);
  std::string uboIncludeName = inputCodeInfo.mainClassName + "_ubo.h";

  std::cout << "  placed classVariables num = " << inputCodeInfo.dataMembers.size() << std::endl;
  {
    std::string rawname        = GetFolderPath(inputCodeInfo.mainClassFileName);
    std::string outName        = rawname + "/include/" + uboIncludeName;
    kslicer::ApplyJsonToTemplate("templates/ubo_def.h",  outName, jsonUBO);
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;
  
  std::cout << "(7) Perform final templated text rendering to generate Vulkan calls" << std::endl; 
  std::cout << "{" << std::endl;
  {
    kslicer::PrintVulkanBasicsFile("templates/vulkan_basics.h", inputCodeInfo);

    std::string rawname = kslicer::CutOffFileExt(inputCodeInfo.mainClassFileName);
    auto json = PrepareJsonForAllCPP(inputCodeInfo, inputCodeInfo.mainFunc, rawname + "_generated.h", threadsOrder, uboIncludeName, jsonUBO); 
    
    kslicer::ApplyJsonToTemplate("templates/vk_class.h",        rawname + "_generated.h", json); 
    kslicer::ApplyJsonToTemplate("templates/vk_class.cpp",      rawname + "_generated.cpp", json);
    kslicer::ApplyJsonToTemplate("templates/vk_class_ds.cpp",   rawname + "_generated_ds.cpp", json);
    kslicer::ApplyJsonToTemplate("templates/vk_class_init.cpp", rawname + "_generated_init.cpp", json);
  }
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(8) Generate OpenCL kernels" << std::endl; 
  std::cout << "{" << std::endl;

  // if user set custom work group size for kernels via hint file we should apply it befor generating kernels
  if(hintFile != "")
  {
    std::ifstream ifs(hintFile);
    nlohmann::json hintJson = nlohmann::json::parse(ifs);
    nlohmann::json wgszJson = hintJson["WorkGroupSize"];
    
    uint32_t defaultWgSize[3] = {256,1,1};
    defaultWgSize[0] = wgszJson["default"][0];
    defaultWgSize[1] = wgszJson["default"][1];
    defaultWgSize[2] = wgszJson["default"][2];

    for(auto& kernel : inputCodeInfo.kernels)
    {
      auto it = wgszJson.find(kernel.name);
      if(it != wgszJson.end())
      {
        kernel.injectedWgSize[0] = (*it)[0];
        kernel.injectedWgSize[1] = (*it)[1];
        kernel.injectedWgSize[2] = (*it)[2];
      }
      else
      {
        kernel.injectedWgSize[0] = defaultWgSize[0];
        kernel.injectedWgSize[1] = defaultWgSize[1];
        kernel.injectedWgSize[2] = defaultWgSize[2];
      }
    }
  }

  // analize inputCodeInfo.allDescriptorSetsInfo to mark all args of each kernel that we need to apply fakeOffset(tid) inside kernel to this arg
  //
  for(const auto& call : inputCodeInfo.allDescriptorSetsInfo)
    inputCodeInfo.ProcessCallArs_KF(call);

  for(auto& k : inputCodeInfo.kernels)
    k.rewrittenText = inputCodeInfo.VisitAndRewrite_KF(k, compiler, k.rewrittenInit);

  // finally generate kernels
  //
  const std::string templatePath = inputCodeInfo.pShaderCC->TemplatePath();
  auto json = kslicer::PrepareJsonForKernels(inputCodeInfo, usedByKernelsFunctions, usedDecls, compiler, threadsOrder, uboIncludeName, jsonUBO);
  
  if(inputCodeInfo.pShaderCC->IsSingleSource())
  {
    const std::string outFileName  = GetFolderPath(inputCodeInfo.mainClassFileName) + "/" + inputCodeInfo.pShaderCC->ShaderSingleFile();
    kslicer::ApplyJsonToTemplate(templatePath, outFileName, json);

    std::ofstream buildSH(GetFolderPath(inputCodeInfo.mainClassFileName) + "/z_build.sh");
    buildSH << "#!/bin/sh" << std::endl;
    std::string build = inputCodeInfo.pShaderCC->BuildCommand();
    buildSH << build.c_str() << std::endl;
    buildSH.close();
  }
  else // if need to compile each kernel inside separate file
  { 
    nlohmann::json copy, kernels;
    for (auto& el : json.items())
    {
      //std::cout << el.key() << std::endl;
      if(std::string(el.key()) == "Kernels")
        kernels = json[el.key()];
      else
        copy[el.key()] = json[el.key()];
    }
    
    std::string folderPath = GetFolderPath(inputCodeInfo.mainClassFileName);
    std::string shaderPath = folderPath + "/" + inputCodeInfo.pShaderCC->ShaderFolder();
    #ifdef WIN32
    mkdir(shaderPath.c_str());
    #else
    mkdir(shaderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    #endif
    
    std::ofstream buildSH(shaderPath + "/build.sh");
    buildSH << "#!/bin/sh" << std::endl;
    for(auto& kernel : kernels.items())
    {
      nlohmann::json currKerneJson = copy;
      currKerneJson["Kernels"] = std::vector<std::string>();
      currKerneJson["Kernels"].push_back(kernel.value());
      
      std::string outFileName = std::string(kernel.value()["Name"]) + ".cpp";
      std::string outFilePath = shaderPath + "/" + outFileName;
      kslicer::ApplyJsonToTemplate("templates/gen_circle.cxx", outFilePath, currKerneJson);
      buildSH << "../../circle -shader -c -emit-spirv " << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DUSE_CIRCLE_CC -I.. " << std::endl;
    }
    
    nlohmann::json emptyJson;
    std::string outFileServ = shaderPath + "/" + "serv_kernels.cpp";
    kslicer::ApplyJsonToTemplate("templates/ser_circle.cxx", outFileServ, emptyJson);
    buildSH << "../../circle -shader -c -emit-spirv " << outFileServ.c_str() << " -o " << outFileServ.c_str() << ".spv" << " -DUSE_CIRCLE_CC -I.. " << std::endl;

    buildSH.close();
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;
  
  //std::ofstream file(GetFolderPath(inputCodeInfo.mainClassFileName) + "/z_kernels.json");
  //file << std::setw(2) << json; //
  //file.close();

  return 0;
}