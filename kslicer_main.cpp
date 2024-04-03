#include <stdio.h>
#include <vector>
#include <system_error>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <unordered_map>
#include <iomanip>
#include <cctype>
#include <queue>

#include "llvm/TargetParser/Host.h"
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
#include "kslicer_warnings.h"
#include "initial_pass.h"
#include "ast_matchers.h"
#include "class_gen.h"
#include "extractor.h"

using namespace clang;
#include "template_rendering.h"

using kslicer::KernelInfo;
using kslicer::DataMemberInfo;

int main(int argc, const char **argv)
{
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "[main]: work_dir = " << std::filesystem::current_path() << std::endl;
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (argc < 2)
  {
    llvm::errs() << "Usage: <filename>\n";
    return 1;
  }

  std::cout << "CMD LINE: ";
  for(int i=0;i<argc;i++)
     std::cout << argv[i] << " ";
  std::cout << std::endl;
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::vector<std::string> ignoreFiles;
  std::vector<std::string> processFiles;
  std::vector<std::string> allFiles;
  std::vector<std::string> cppIncludesAdditional;
  std::filesystem::path fileName;
  auto params = ReadCommandLineParams(argc, argv, fileName,
                                      allFiles, ignoreFiles, processFiles, cppIncludesAdditional);

  std::filesystem::path mainFolderPath  = fileName.parent_path();
  std::string mainClassName   = "TestClass";
  std::string outGenerated    = "data/generated.cl";
  std::string stdlibFolder    = "";
  std::string patternName     = "rtv";
  std::string shaderCCName    = "clspv";
  std::string hintFile        = "";
  std::string suffix          = "_Generated";
  std::string shaderFolderPrefix    = "";

  std::string composeAPIName  = "";
  std::string composeImplName = "";

  uint32_t    threadsOrder[3] = {0,1,2};
  uint32_t    warpSize        = 32;
  bool        useCppInKernels = false;
  bool        halfFloatTextures  = false;
  bool        useMegakernel      = false;
  auto        defaultVkernelType = kslicer::VKERNEL_IMPL_TYPE::VKERNEL_SWITCH;
  bool        enableSubGroupOps  = false;
  int         ispcThreadModel    = 0;
  bool        ispcExplicitIndices = false;
  bool        genGPUAPI          = false;

  kslicer::ShaderFeatures forcedFeatures;

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

  if(params.find("-shaderFolderPrefix") != params.end())
    shaderFolderPrefix = params["-shaderFolderPrefix"];

  if(suffix == "_Generated" && (shaderCCName == "ispc") || (shaderCCName == "ISPC"))
    suffix = "_ISPC";

  if(params.find("-hint") != params.end())
    hintFile = params["-hint"];

  if(params.find("-warpSize") != params.end())
    warpSize = atoi(params["-warpSize"].c_str());

  if(params.find("-enableSubgroup") != params.end())
    enableSubGroupOps = atoi(params["-enableSubgroup"].c_str());

  if(params.find("-halfTex") != params.end())
    halfFloatTextures = (params["-halfTex"] == "1");

  if(params.find("-megakernel") != params.end())
    useMegakernel = (params["-megakernel"] == "1");

  if(params.find("-cl-std=") != params.end())
    useCppInKernels = params["-cl-std="].find("++") != std::string::npos;
  else if(params.find("-cl-std") != params.end())
    useCppInKernels = params["-cl-std"].find("++") != std::string::npos;

  if(params.find("-vkernel_t=") != params.end())
  {
    if(params["-vkernel_t="] == "switch")
      defaultVkernelType = kslicer::VKERNEL_IMPL_TYPE::VKERNEL_SWITCH;
    else if(params["-vkernel_t="] == "indirect_dispatch")
      defaultVkernelType = kslicer::VKERNEL_IMPL_TYPE::VKERNEL_INDIRECT_DISPATCH;
  }

  if(params.find("-ispc_threads") != params.end())
    ispcThreadModel = atoi(params["-ispc_threads"].c_str());

  if(params.find("-ispc_explicit_id") != params.end())
    ispcExplicitIndices = (atoi(params["-ispc_explicit_id"].c_str()) == 1);

  if(params.find("-composInterface") != params.end())
    composeAPIName = params["-composInterface"];

  if(params.find("-composImplementation") != params.end())
    composeImplName = params["-composImplementation"];

  if(params.find("-suffix") != params.end())
    suffix = params["-suffix"];

  if(params.find("-forceEnableHalf") != params.end())
    forcedFeatures.useHalfType = (atoi(params["-forceEnableHalf"].c_str()) == 1);
  if(params.find("-forceEnableInt8") != params.end())
    forcedFeatures.useByteType = (atoi(params["-forceEnableInt8"].c_str()) == 1);
  if(params.find("-forceEnableInt16") != params.end())
    forcedFeatures.useShortType = (atoi(params["-forceEnableInt16"].c_str()) == 1);
  if(params.find("-forceEnableInt64") != params.end())
    forcedFeatures.useInt64Type = (atoi(params["-forceEnableInt64"].c_str()) == 1);

  if(params.find("-gen_gpu_api") != params.end())
    genGPUAPI = atoi(params["-gen_gpu_api"].c_str());

  kslicer::TextGenSettings textGenSettings;
  if(params.find("-enable_motion_blur") != params.end())
    textGenSettings.enableMotionBlur = atoi(params["-enable_motion_blur"].c_str()) != 0;
  if(params.find("-enable_ray_tracing_pipeline") != params.end())
    textGenSettings.enableRayGen = (atoi(params["-enable_ray_tracing_pipeline"].c_str()) != 0) || textGenSettings.enableMotionBlur;

  std::unordered_set<std::string> values;
  std::vector<std::string> ignoreFolders;
  std::vector<std::filesystem::path> processFolders;
  for(auto p : params)
  {
    values.insert(p.second);
    std::string folderT = p.second;
    std::transform(folderT.begin(), folderT.end(), folderT.begin(), [](unsigned char c){ return std::tolower(c); });

    if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I' && folderT == "ignore")
      ignoreFolders.push_back(p.first.substr(2));
    else if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I' && folderT == "process")
      processFolders.push_back(p.first.substr(2));
  }

  // make specific checks to be sure user don't include these files to hit project as normal files
  //
  {
    auto processFolders2 = processFolders;
    processFolders2.push_back(mainFolderPath);
    kslicer::CheckInterlanIncInExcludedFolders(processFolders2);
  }

  std::vector<const char*> argsForClang = ExcludeSlicerParams(argc, argv, params);
  llvm::ArrayRef<const char*> args(argsForClang.data(), argsForClang.data() + argsForClang.size());

  // Make sure it exists
  std::ifstream fin(fileName.c_str());
  if(!fin.is_open())
  {
    std::cout << "[main]: error, input file " << fileName.c_str() << " not found!" << std::endl;
    return 0;
  }
  fin.close();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::shared_ptr<kslicer::MainClassInfo> pImplPattern = nullptr;
  if(patternName == "rtv")
    pImplPattern = std::make_shared<kslicer::RTV_Pattern>();
  else if(patternName == "ipv")
    pImplPattern = std::make_shared<kslicer::IPV_Pattern>();
  else
  {
    std::cout << "[main]: wrong pattern name '" << patternName.c_str() << "' " << std::endl;
    exit(0);
  }
  kslicer::MainClassInfo& inputCodeInfo = *pImplPattern;
  inputCodeInfo.ignoreFolders  = ignoreFolders;  // set shader folders
  inputCodeInfo.processFolders = processFolders; // set common C/C++ folders
  inputCodeInfo.ignoreFiles    = ignoreFiles;    // set exceptions for common C/C++ folders (i.e. processFolders)
  inputCodeInfo.processFiles   = processFiles;   // set exceptions for shader folders (i.e. ignoreFolders)
  inputCodeInfo.cppIncudes     = cppIncludesAdditional;
  inputCodeInfo.genGPUAPI      = genGPUAPI;

  if(shaderCCName == "glsl" || shaderCCName == "GLSL")
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::GLSLCompiler>(inputCodeInfo.mainClassSuffix);
    inputCodeInfo.processFolders.push_back("include/");
  }
  else if(shaderCCName == "ispc" || shaderCCName == "ISPC")
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::ISPCCompiler>(useCppInKernels, inputCodeInfo.mainClassSuffix);
    inputCodeInfo.ignoreFolders.push_back("include/");
  }
  else
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::ClspvCompiler>(useCppInKernels, inputCodeInfo.mainClassSuffix);
    inputCodeInfo.ignoreFolders.push_back("include/");
  }

  inputCodeInfo.defaultVkernelType   = defaultVkernelType;
  inputCodeInfo.halfFloatTextures    = halfFloatTextures;
  inputCodeInfo.megakernelRTV        = useMegakernel;
  inputCodeInfo.mainClassSuffix      = suffix;
  inputCodeInfo.shaderFolderPrefix   = shaderFolderPrefix;
  inputCodeInfo.globalShaderFeatures = forcedFeatures;

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
    compiler.getLangOpts().CXXExceptions = 1;
    compiler.getLangOpts().RTTI        = 1;
    compiler.getLangOpts().Bool        = 1;
    compiler.getLangOpts().CPlusPlus   = 1;
    compiler.getLangOpts().CPlusPlus14 = 1;
    compiler.getLangOpts().CPlusPlus17 = 1;
  }

  compiler.createFileManager();
  compiler.createSourceManager(compiler.getFileManager());
  //g_pCompilerInstance = &compiler;

  // (0) add path dummy include files for STL and e.t.c. (we don't want to parse actually std library)
  //
  HeaderSearchOptions &headerSearchOptions = compiler.getHeaderSearchOpts();
  headerSearchOptions.AddPath(stdlibFolder.c_str(), clang::frontend::Angled, false, false);
  for(auto p : params)
  {
    if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I')
    {
      std::string includePath = p.first.substr(2);
      //std::cout << "[main]: add include folder: " << includePath.c_str() << std::endl;
      headerSearchOptions.AddPath(includePath.c_str(), clang::frontend::Angled, false, false);
    }
  }
  //headerSearchOptions.Verbose = 1;

  compiler.createPreprocessor(clang::TU_Complete);
  compiler.getPreprocessorOpts().UsePredefines = true;
  // register our header lister
  HeaderLister headerLister(&inputCodeInfo);
  compiler.getPreprocessor().addPPCallbacks(std::make_unique<HeaderLister>(headerLister));
  compiler.createASTContext();

  // const FileEntry *pFile = compiler.getFileManager().getFile(fileName).get();
  FileEntryRef file = compiler.getFileManager().getFileRef(fileName.u8string()).get();
  compiler.getSourceManager().setMainFileID( compiler.getSourceManager().createFileID( file, clang::SourceLocation(), clang::SrcMgr::C_User));
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(), &compiler.getPreprocessor());


  // init clang tooling
  //
  const std::string filenameString = fileName.u8string();
  std::vector<const char*> argv2 = {argv[0], filenameString.c_str()};
  std::vector<std::string> extraArgs; extraArgs.reserve(32);
  for(auto p : params)
  {
    if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I')  // add include folders to the Tool
    {
      extraArgs.push_back(std::string("-extra-arg=") + p.first);
      argv2.push_back(extraArgs.back().c_str());
    }
  }
  argv2.push_back("--");
  int argSize = argv2.size();

  llvm::cl::OptionCategory GDOpts("global-detect options");
  //clang::tooling::CommonOptionsParser OptionsParser(argSize, argv2.data(), GDOpts);                     // clang 12
  //clang::tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());   // clang 12
  auto OptionsParser = clang::tooling::CommonOptionsParser::create(argSize, argv2.data(), GDOpts);        // clang 14
  clang::tooling::ClangTool Tool(OptionsParser->getCompilations(), OptionsParser->getSourcePathList());   // clang 14

  // (0) find all "Main" functions, a functions which call kernels. Kernels are also listed for each mainFunc;
  //
  std::vector<std::string> cfNames;
  cfNames.reserve(20);

  std::cout << "(0) Listing main functions of " << mainClassName.c_str() << std::endl;
  auto cfList = kslicer::ListAllMainRTFunctions(Tool, mainClassName, compiler.getASTContext(), inputCodeInfo);
  std::cout << "{" << std::endl;
  for(const auto& f : cfList)
  {
    std::cout << "  found " << f.first.c_str() << std::endl;
    cfNames.push_back(f.first);
  }
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  inputCodeInfo.mainFunc.resize(cfList.size());
  inputCodeInfo.mainClassName     = mainClassName;
  inputCodeInfo.mainClassFileName = fileName;

  // create default Shader Rewriter, don't delete it please!
  //
  clang::Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
  inputCodeInfo.pShaderFuncRewriter = inputCodeInfo.pShaderCC->MakeFuncRewriter(rewrite2, compiler, &inputCodeInfo);

  std::cout << "(1) Processing class " << mainClassName.c_str() << " with initial pass" << std::endl;
  std::cout << "{" << std::endl;

  // Parse code, initial pass
  //
  std::vector<std::string> composClassNames;
  if(composeAPIName != "")
    composClassNames.push_back(composeAPIName);
  if(composeImplName != "")
    composClassNames.push_back(composeImplName);
  kslicer::InitialPassASTConsumer firstPassData(cfNames, mainClassName, composClassNames, compiler, inputCodeInfo);
  ParseAST(compiler.getPreprocessor(), &firstPassData, compiler.getASTContext());
  compiler.getDiagnosticClient().EndSourceFile(); // ??? What Is This Line For ???


  std::string composMemberName = "";
  auto pComposAPI  = firstPassData.rv.m_composedClassInfo.find(composeAPIName);
  auto pComposImpl = firstPassData.rv.m_composedClassInfo.find(composeImplName);

  if(pComposAPI != firstPassData.rv.m_composedClassInfo.end() && pComposImpl != firstPassData.rv.m_composedClassInfo.end()) // if compos classes are found
    composMemberName = kslicer::PerformClassComposition(firstPassData.rv.mci, pComposAPI->second, pComposImpl->second);     // perform class composition
  for(const auto& name : composClassNames)
    inputCodeInfo.composPrefix[name] = composMemberName;

  inputCodeInfo.mainClassFileInclude = firstPassData.rv.MAIN_FILE_INCLUDE;
  inputCodeInfo.mainClassASTNode     = firstPassData.rv.mci.astNode;
  inputCodeInfo.allKernels           = firstPassData.rv.mci.functions;
  inputCodeInfo.allOtherKernels      = firstPassData.rv.mci.otherFunctions;
  inputCodeInfo.allDataMembers       = firstPassData.rv.mci.dataMembers;
  inputCodeInfo.ctors                = firstPassData.rv.mci.ctors;
  inputCodeInfo.allMemberFunctions   = firstPassData.rv.mci.allMemberFunctions;
  inputCodeInfo.ProcessAllSetters(firstPassData.rv.mci.m_setters, compiler);

  std::vector<kslicer::DeclInClass> generalDecls = firstPassData.rv.GetExtractedDecls();
  if(inputCodeInfo.mainClassASTNode == nullptr)
  {
    std::cout << "[main]: ERROR, main class " << mainClassName.c_str() << " not found" << std::endl;
    return 0;
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(2) Process control functions; extract local variables, known calls like memcpy, sort, std::fill and other " << std::endl;
  std::cout << "{" << std::endl;

  size_t mainFuncId = 0;
  for(const auto f : cfList)
  {
    const std::string& mainFuncName = f.first;
    auto& mainFuncRef = inputCodeInfo.mainFunc[mainFuncId];
    mainFuncRef.Name  = mainFuncName;
    mainFuncRef.Node  = firstPassData.rv.mci.m_mainFuncNodes[mainFuncName];

    // Now process each main function: variables and kernel calls, if()->break and if()->return statements.
    //
    {
      auto allMatchers = inputCodeInfo.ListMatchers_CF(mainFuncName);
      auto pMatcherPrc = inputCodeInfo.MatcherHandler_CF(mainFuncRef, compiler);

      clang::ast_matchers::MatchFinder finder;
      for(auto& matcher : allMatchers)
        finder.addMatcher(clang::ast_matchers::traverse(clang::TK_IgnoreUnlessSpelledInSource,matcher), pMatcherPrc.get());

      std::cout << "  process control function: " << mainFuncName.c_str() << "(...)" << std::endl;
      auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
      std::cout << "  process control function: " << mainFuncName.c_str() << "(...) --> " << GetClangToolingErrorCodeMessage(res) << std::endl;

      // filter out unused kernels
      //
      inputCodeInfo.kernels.reserve(inputCodeInfo.allKernels.size());
      inputCodeInfo.kernels.clear();
      for (auto& k : inputCodeInfo.allKernels)
      {
        if(k.second.usedInMainFunc && inputCodeInfo.kernels.find(k.first) == inputCodeInfo.kernels.end())
          inputCodeInfo.kernels[k.first] = k.second;
      }

      if(inputCodeInfo.SupportVirtualKernels())
      {
        std::unordered_map<std::string, KernelInfo> vkernels;
        for(const auto& p : inputCodeInfo.allOtherKernels)
          vkernels[p.second.name] = p.second;

        for(auto& p : vkernels)
        {
          if(inputCodeInfo.kernels.find(p.first) == inputCodeInfo.kernels.end())
          {
            p.second.isVirtual             = true;
            inputCodeInfo.kernels[p.first] = p.second;
          }
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

  for(auto& nk : inputCodeInfo.kernels)
  {
    auto& kernel            = nk.second;
    kernel.warpSize         = warpSize;
    kernel.enableSubGroups  = enableSubGroupOps;
    kernel.singleThreadISPC = (ispcThreadModel == 1);
    kernel.openMpAndISPC    = (ispcThreadModel == 2);
    kernel.explicitIdISPC   = ispcExplicitIndices;

    auto kernelMatchers = inputCodeInfo.ListMatchers_KF(kernel.name);
    auto pFilter        = inputCodeInfo.MatcherHandler_KF(kernel, compiler);

    clang::ast_matchers::MatchFinder finder;
    for(auto& matcher : kernelMatchers)
      finder.addMatcher(clang::ast_matchers::traverse(clang::TK_IgnoreUnlessSpelledInSource, matcher), pFilter.get());

    auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    std::cout << "  process " << kernel.name.c_str() << ":\t" << GetClangToolingErrorCodeMessage(res) << std::endl;

    for(auto& arg : kernel.args) // it is important to run this loop after second stage at which kernel matchers are applied!
      inputCodeInfo.ProcessKernelArg(arg, kernel);

    kernel.isIndirect = inputCodeInfo.IsIndirect(kernel);
    if(kernel.isIndirect)
    {
      kernel.indirectBlockOffset = inputCodeInfo.m_indirectBufferSize;
      inputCodeInfo.m_indirectBufferSize++;
    }

    inputCodeInfo.VisitAndPrepare_KF(kernel, compiler);

    if(kernel.hasFinishPass) // add additional buffers for reduction
    {
      uint32_t buffNumber = 0;
      for(auto& redVar : kernel.subjectedToReduction)
      {
        if(redVar.second.SupportAtomicLastStep() || inputCodeInfo.pShaderCC->IsISPC())
          continue;

        std::stringstream strOut;
        strOut << "tmpred" << buffNumber << redVar.second.GetSizeOfDataType();
        inputCodeInfo.AddTempBufferToKernel(strOut.str(), redVar.second.dataType, kernel);
        redVar.second.tmpVarName = strOut.str();
        buffNumber++;
      }
    }
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(4) Extract functions, constants and structs from 'MainClass' " << std::endl;
  std::cout << "{" << std::endl;
  std::vector<kslicer::FuncData> usedByKernelsFunctions = kslicer::ExtractUsedFunctions(inputCodeInfo, compiler); // recursive processing of functions used by kernel, extracting all needed functions
  std::vector<kslicer::DeclInClass> usedDecls           = kslicer::ExtractTCFromClass(inputCodeInfo.mainClassName, inputCodeInfo.mainClassASTNode, compiler, Tool);


  for(const auto& usedDecl : usedDecls) // merge usedDecls with generalDecls
  {
    bool found = false;
    for(const auto& currDecl : generalDecls)
    {
      if(currDecl.name == usedDecl.name)
      {
        found = true;
        break;
      }
    }
    if(!found)
      generalDecls.push_back(usedDecl);
  }

  std::vector<std::string> usedDefines = kslicer::ExtractDefines(compiler);

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  inputCodeInfo.AddSpecVars_CF(inputCodeInfo.mainFunc, inputCodeInfo.kernels);

  bool hasMembers = false;
  for(const auto& f : usedByKernelsFunctions) {
    if(f.isMember) {
      hasMembers = true;
      break;
    }
  }

  if(hasMembers && inputCodeInfo.pShaderCC->IsGLSL()) // We don't implement this for OpenCL kernels yet ... or at all.
  {
    std::cout << "(4.1) Process Member function calls, extract data accesed in member functions " << std::endl;
    std::cout << "{" << std::endl;
    for(auto& k : inputCodeInfo.kernels)
    {
      for(const auto& f : usedByKernelsFunctions)
      {
        if(f.isMember) // and if is called from this kernel.It it is called, list all input parameters for each call!
        {
          // list all input parameters for each call of member function inside kernel; in this way we know which textures, vectors and samplers were actually used by these functions
          //
          std::unordered_map<std::string, kslicer::UsedContainerInfo> auxContainers;
          auto machedParams = kslicer::ArgMatchTraversal    (&k.second, f, usedByKernelsFunctions,                inputCodeInfo, compiler);
          auto usedMembers  = kslicer::ExtractUsedMemberData(&k.second, f, usedByKernelsFunctions, auxContainers, inputCodeInfo, compiler);

          // TODO: process bindedParams correctly
          //
          std::vector<kslicer::DataMemberInfo> samplerMembers;
          for(auto x : usedMembers)
          {
            if(kslicer::IsSamplerTypeName(x.second.type))
            {
              auto y = x.second;
              y.kind = kslicer::DATA_KIND::KIND_SAMPLER;
              samplerMembers.push_back(y);
            }
          }

          for(auto& member : usedMembers)
          {
            k.second.usedMembers.insert(member.first);
            member.second.usedInKernel = true;

            if(kslicer::IsSamplerTypeName(member.second.type)) // actually this is not correct!!!
            {
              for(auto sampler : samplerMembers)
              {
                for(auto map : machedParams)
                {
                  for(auto par : map)
                  {
                    std::string actualTextureName = par.second;
                    k.second.texAccessSampler[actualTextureName] = sampler.name;
                  }
                }
              }
            }
            else if(member.second.isContainer)
            {
              kslicer::UsedContainerInfo info;
              info.type          = member.second.type;
              info.name          = member.second.name;
              info.kind          = member.second.kind;
              info.isConst       = member.second.IsUsedTexture();      // strange thing ...
              k.second.usedContainers[info.name] = info;
            }
            else
            {
              auto p1 = inputCodeInfo.allDataMembers.find(member.first);
              auto p2 = inputCodeInfo.m_setterVars.find(member.first);
              if(p1 != inputCodeInfo.allDataMembers.end())
                p1->second.usedInKernel = true;
              else if(p2 ==  inputCodeInfo.m_setterVars.end()) // don't add setters
                inputCodeInfo.allDataMembers[member.first] = member.second;
            }
          } // end for(auto& member : usedMembers)

          for(const auto& c : auxContainers)
            k.second.usedContainers[c.first] = c.second;

        } // end if(f.isMember)
      } // end for(const auto& f : usedByKernelsFunctions)
    } // end for(auto& k : inputCodeInfo.kernels)

    for(const auto& k : inputCodeInfo.kernels) // fix this flag for members that were used in member functions but not in kernels directly
    {
      for(const auto& c : k.second.usedContainers)
      {
        auto pFound = inputCodeInfo.allDataMembers.find(c.second.name);
        if(pFound != inputCodeInfo.allDataMembers.end())
          pFound->second.usedInKernel = true;
      }
    }

    std::cout << "}" << std::endl;
    std::cout << std::endl;
  }

  std::cout << "(5) Process control functions to generate all 'MainCmd' functions" << std::endl;
  std::cout << "{" << std::endl;

  // (5) Process controll functions and generate some intermediate cpp code with Vulkan calls
  //
  ObtainKernelsDecl(inputCodeInfo.kernels, compiler, inputCodeInfo.mainClassName, inputCodeInfo);

  inputCodeInfo.allDescriptorSetsInfo.clear();
  /////////////////////////////////////////////////////////////////////////////////////////////// fakeOffset flag for local variables
  if(inputCodeInfo.megakernelRTV)
  {
    inputCodeInfo.megakernelRTV = false;
    auto auxDecriptorSets = inputCodeInfo.allDescriptorSetsInfo;
    for(auto& mainFunc : inputCodeInfo.mainFunc)
    {
      std::cout << " process subkernel " << mainFunc.Name.c_str() << std::endl;
      inputCodeInfo.VisitAndRewrite_CF(mainFunc, compiler);           // ==> output to mainFunc and inputCodeInfo.allDescriptorSetsInfo
    }
    inputCodeInfo.PlugSpecVarsInCalls_CF(inputCodeInfo.mainFunc, inputCodeInfo.kernels, inputCodeInfo.allDescriptorSetsInfo);
    for(const auto& call : inputCodeInfo.allDescriptorSetsInfo)
      inputCodeInfo.ProcessCallArs_KF(call);

    auxDecriptorSets = inputCodeInfo.allDescriptorSetsInfo;
    inputCodeInfo.allDescriptorSetsInfo.clear();

    // analize inputCodeInfo.allDescriptorSetsInfo to mark all args of each kernel that we need to apply fakeOffset(tid) inside kernel to this arg
    //
    for(const auto& call : auxDecriptorSets)
      inputCodeInfo.ProcessCallArs_KF(call);
    inputCodeInfo.megakernelRTV = true;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////// fakeOffset flag for local variables

  for(auto& mainFunc : inputCodeInfo.mainFunc)
  {
    std::cout << "  process " << mainFunc.Name.c_str() << std::endl;
    inputCodeInfo.VisitAndRewrite_CF(mainFunc, compiler);           // ==> output to mainFunc and inputCodeInfo.allDescriptorSetsInfo
  }

  inputCodeInfo.PlugSpecVarsInCalls_CF(inputCodeInfo.mainFunc, inputCodeInfo.kernels, inputCodeInfo.allDescriptorSetsInfo);

  // analize inputCodeInfo.allDescriptorSetsInfo to mark all args of each kernel that we need to apply fakeOffset(tid) inside kernel to this arg
  //
  for(const auto& call : inputCodeInfo.allDescriptorSetsInfo)
    inputCodeInfo.ProcessCallArs_KF(call);

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  if(inputCodeInfo.SupportVirtualKernels())
  {
    std::cout << "(5.1) Process Virtual Kernels hierarchies" << std::endl;
    std::cout << "(5.2) Extract Virtual Kernels hierarchies constants" << std::endl;
    std::cout << "{" << std::endl;
    inputCodeInfo.ProcessDispatchHierarchies(firstPassData.rv.m_classList, compiler);
    inputCodeInfo.ExtractHierarchiesConstants(compiler, Tool);

    for(auto& k : inputCodeInfo.kernels)
    {
      if(k.second.isMaker)
      {
        auto p = inputCodeInfo.m_vhierarchy.find(k.second.interfaceName);
        assert(p != inputCodeInfo.m_vhierarchy.end());
        k.second.indirectMakerOffset  = inputCodeInfo.m_indirectBufferSize;
        p->second.indirectBlockOffset = inputCodeInfo.m_indirectBufferSize;
        inputCodeInfo.m_indirectBufferSize += p->second.implementations.size(); // allocate place for all implementations
      }
    }

    std::cout << "}" << std::endl;
    std::cout << std::endl;
  }

  std::cout << "(6) Calc offsets for all class members; ingore unused members that were not marked on previous step" << std::endl;
  std::cout << "{" << std::endl;

  inputCodeInfo.dataMembers = kslicer::MakeClassDataListAndCalcOffsets(inputCodeInfo.allDataMembers);

  inputCodeInfo.ProcessMemberTypes(firstPassData.rv.GetOtherTypeDecls(), compiler.getSourceManager(), generalDecls);                   // ==> generalDecls
  inputCodeInfo.ProcessMemberTypesAligment(inputCodeInfo.dataMembers, firstPassData.rv.GetOtherTypeDecls(), compiler.getASTContext()); // ==> inputCodeInfo.dataMembers

  std::sort(inputCodeInfo.dataMembers.begin(), inputCodeInfo.dataMembers.end(), kslicer::DataMemberInfo_ByAligment()); // sort by aligment in GLSL

  auto jsonUBO               = kslicer::PrepareUBOJson(inputCodeInfo, inputCodeInfo.dataMembers, compiler, textGenSettings);
  std::string uboIncludeName = inputCodeInfo.mainClassName + ToLowerCase(inputCodeInfo.mainClassSuffix) + "_ubo.h";

  std::filesystem::path uboOutName = "";
  std::cout << "  placed classVariables num = " << inputCodeInfo.dataMembers.size() << std::endl;
  uboOutName = inputCodeInfo.mainClassFileName.parent_path() / "include" / uboIncludeName;

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  // if user set custom work group size for kernels via hint file we should apply it befor generating kernels
  //
  nlohmann::json wgszJson;
  uint32_t defaultWgSize[3][3] = {{256, 1, 1},
                                  {32,  8, 1},
                                  {8,   8, 8}};
  if(hintFile != "")
  {
    std::ifstream ifs(hintFile);
    nlohmann::json hintJson = nlohmann::json::parse(ifs);
    wgszJson            = hintJson["WorkGroupSize"];
    defaultWgSize[0][0] = wgszJson["default"][0];
    defaultWgSize[0][1] = wgszJson["default"][1];
    defaultWgSize[0][2] = wgszJson["default"][2];
    defaultWgSize[1][0] = wgszJson["default2D"][0];
    defaultWgSize[1][1] = wgszJson["default2D"][1];
    defaultWgSize[1][2] = wgszJson["default2D"][2];
    defaultWgSize[2][0] = wgszJson["default3D"][0];
    defaultWgSize[2][1] = wgszJson["default3D"][1];
    defaultWgSize[2][2] = wgszJson["default3D"][2];
  }

  for(auto& nk : inputCodeInfo.kernels)
  {
    auto& kernel   = nk.second;
    auto kernelDim = kernel.GetDim();
    auto it = wgszJson.find(kernel.name);
    if(it != wgszJson.end())
    {
      kernel.wgSize[0] = (*it)[0];
      kernel.wgSize[1] = (*it)[1];
      kernel.wgSize[2] = (*it)[2];
      kernel.wgSize[2] = 1;
    }
    else
    {
      kernel.wgSize[0] = defaultWgSize[kernelDim-1][0];
      kernel.wgSize[1] = defaultWgSize[kernelDim-1][1];
      kernel.wgSize[2] = defaultWgSize[kernelDim-1][2];
    }
  }

  auto& megakernelsByName = inputCodeInfo.megakernelsByName;
  if(inputCodeInfo.megakernelRTV) // join all kernels in one for each CF
  {
    for(auto& cf : inputCodeInfo.mainFunc)
    {
      cf.subkernels = kslicer::extractUsedKernelsByName(cf.UsedKernels, inputCodeInfo.kernels);
      cf.megakernel = kslicer::joinToMegaKernel(cf.subkernels, cf);
      cf.megakernel.isMega = true;
      cf.MegaKernelCall    = kslicer::GetCFMegaKernelCall(cf);
      megakernelsByName[cf.megakernel.name] = cf.megakernel;
    }

    ObtainKernelsDecl(megakernelsByName, compiler, inputCodeInfo.mainClassName, inputCodeInfo);
    for(auto& cf : inputCodeInfo.mainFunc)
      cf.megakernel.DeclCmd = megakernelsByName[cf.megakernel.name].DeclCmd;


    // fix megakernels descriptor sets
    //
    for(auto& dsInfo : inputCodeInfo.allDescriptorSetsInfo)
    {
      auto pKernelInfo = megakernelsByName.find(dsInfo.originKernelName);
      if(pKernelInfo == megakernelsByName.end())
        continue;

      dsInfo.isMega = true;
      kslicer::MainFuncInfo* pCF = nullptr;
      std::string cfName = dsInfo.originKernelName.substr(0, dsInfo.originKernelName.size()-4); // cut of "Mega"
      for(size_t i=0; i<inputCodeInfo.mainFunc.size(); i++)
      {
        if(inputCodeInfo.mainFunc[i].Name == cfName)
          pCF = &inputCodeInfo.mainFunc[i];
      }
      if(pCF == nullptr)
        continue;

      for(const auto& inout : pCF->InOuts)
      {
        kslicer::ArgReferenceOnCall arg;
        if(inout.kind == kslicer::DATA_KIND::KIND_POINTER ||
           inout.kind == kslicer::DATA_KIND::KIND_VECTOR  ||
           inout.kind == kslicer::DATA_KIND::KIND_TEXTURE)
        {
          arg.name = inout.name;
          arg.type = inout.type;
          arg.kind = inout.kind;
          arg.isConst = inout.isConst;
          arg.argType = kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG;
          arg.umpersanned = false;
          dsInfo.descriptorSetsInfo.push_back(arg);
        }
      }

      for(const auto& c : pKernelInfo->second.usedContainers)
      {
        kslicer::ArgReferenceOnCall arg;
        arg.name = c.second.name;
        arg.type = c.second.type;
        arg.kind = c.second.kind;
        arg.isConst = c.second.isConst;
        arg.argType = kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_VECTOR;
        arg.umpersanned = false;
        dsInfo.descriptorSetsInfo.push_back(arg);
      }
    }

    // enable Ray Tracing Pipeline if kernel uses accel atruct and this option is enabled in settings
    //
    for(auto& cf : inputCodeInfo.mainFunc) {
      bool hasAccelStructs = false;
      for(const auto& container : cf.megakernel.usedContainers) {
        if(container.second.isAccelStruct()) {
          hasAccelStructs = true;
          break;
        }
      }
      cf.megakernel.enableRTPipeline = hasAccelStructs && textGenSettings.enableRayGen;
    }
  }

  // enable Ray Tracing Pipeline if kernel uses accel atruct and this option is enabled in settings
  //
  for(auto& nk : inputCodeInfo.kernels)
  {
    auto& kernel = nk.second;
    bool hasAccelStructs = false;
    for(const auto& container : kernel.usedContainers) {
      if(container.second.isAccelStruct()) {
        hasAccelStructs = true;
        break;
      }
    }
    kernel.enableRTPipeline = hasAccelStructs && textGenSettings.enableRayGen;
  }

  inputCodeInfo.kernelsCallCmdDeclCached.clear();
  std::string rawname = kslicer::CutOffFileExt(allFiles[0]);
  auto jsonCPP = PrepareJsonForAllCPP(inputCodeInfo, compiler, inputCodeInfo.mainFunc, generalDecls,
                                      rawname + ToLowerCase(suffix) + ".h", threadsOrder,
                                      uboIncludeName, composeImplName,
                                      jsonUBO, textGenSettings);

  std::cout << "(7) Perform final templated text rendering to generate Vulkan calls" << std::endl;
  std::cout << "{" << std::endl;
  {
    if(!inputCodeInfo.pShaderCC->IsISPC())
    {
      kslicer::ApplyJsonToTemplate("templates/vk_class.h",        rawname + ToLowerCase(suffix) + ".h", jsonCPP);
      kslicer::ApplyJsonToTemplate("templates/vk_class.cpp",      rawname + ToLowerCase(suffix) + ".cpp", jsonCPP);
      kslicer::ApplyJsonToTemplate("templates/vk_class_ds.cpp",   rawname + ToLowerCase(suffix) + "_ds.cpp", jsonCPP);
      kslicer::ApplyJsonToTemplate("templates/vk_class_init.cpp", rawname + ToLowerCase(suffix) + "_init.cpp", jsonCPP);
      if(genGPUAPI)
        kslicer::ApplyJsonToTemplate("templates/vk_class_api.h",  rawname + ToLowerCase(suffix) + "_api.h", jsonCPP);
    }
  }
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::string shaderCCName2 = inputCodeInfo.pShaderCC->Name();
  std::cout << "(8) Generate " << shaderCCName2.c_str() << " kernels" << std::endl;
  std::cout << "{" << std::endl;

  if(inputCodeInfo.megakernelRTV) // join all kernels in one for each CF, WE MUST REPEAT THIS HERE BECAUSE OF SHITTY FUNCTIONS ARE PROCESSED DURING 'VisitAndRewrite_KF' for kernels !!!
  {
    for(auto& k : inputCodeInfo.kernels)
      k.second.rewrittenText = inputCodeInfo.VisitAndRewrite_KF(k.second, compiler, k.second.rewrittenInit, k.second.rewrittenFinish);

    for(auto& cf : inputCodeInfo.mainFunc)
    {
      cf.subkernels = kslicer::extractUsedKernelsByName(cf.UsedKernels, inputCodeInfo.kernels);
      cf.megakernel = kslicer::joinToMegaKernel(cf.subkernels, cf);
      cf.megakernel.rewrittenText = inputCodeInfo.VisitAndRewrite_KF(cf.megakernel, compiler, cf.megakernel.rewrittenInit, cf.megakernel.rewrittenFinish);

      // only here we know the full list of shitty functions (with subkernels)
      // so we should update subkernels and apply transform inside "VisitArraySubscriptExpr_Impl" to kernels in the same way we did for functions
      //
      auto oldKernels = cf.subkernels;
      std::unordered_set<std::string> processed;
      cf.subkernelsData.clear();
      for(const auto& name : cf.UsedKernels)
      {
        for(const auto& shit : cf.megakernel.shittyFunctions)
        {
          if(shit.originalName != name)
            continue;

          auto kernel          = inputCodeInfo.kernels[name];
          kernel.currentShit   = shit; // just pass shit inside GLSLKernelRewriter via 'kernel.currentShit'; don't want to break VisitAndRewrite_KF API
          kernel.rewrittenText = inputCodeInfo.VisitAndRewrite_KF(kernel, compiler, kernel.rewrittenInit, kernel.rewrittenFinish);
          cf.subkernelsData.push_back(kernel);
          processed.insert(name);
        }
      }

      // get subkernels were processed and rewritten with "shitty" transforms
      //
      cf.subkernels.clear();
      for(const auto& data : cf.subkernelsData)
        cf.subkernels.push_back(&data);

      // get back subkernels which were not processed
      //
      for(const auto& oldKernelP : oldKernels)
      {
        if(processed.find(oldKernelP->name) == processed.end())
          cf.subkernels.push_back(oldKernelP);
      }

      bool hasAccelStructs = false;
      for(const auto& container : cf.megakernel.usedContainers) {
        if(container.second.isAccelStruct()) {
          hasAccelStructs = true;
          break;
        }
      }
      cf.megakernel.enableRTPipeline = hasAccelStructs && textGenSettings.enableRayGen;
    }
  }
  else
  {
    for(auto& k : inputCodeInfo.kernels)
      k.second.rewrittenText = inputCodeInfo.VisitAndRewrite_KF(k.second, compiler, k.second.rewrittenInit, k.second.rewrittenFinish);
  }

  auto json = kslicer::PrepareJsonForKernels(inputCodeInfo, usedByKernelsFunctions, generalDecls, compiler, threadsOrder, uboIncludeName, jsonUBO, usedDefines, textGenSettings);
  if(inputCodeInfo.pShaderCC->IsISPC()) {
    json["Constructors"]        = jsonCPP["Constructors"];
    json["HasCommitDeviceFunc"] = jsonCPP["HasCommitDeviceFunc"];
    json["HasGetTimeFunc"]      = jsonCPP["HasGetTimeFunc"];
    json["ClassVars"]           = jsonCPP["ClassVars"];
    json["ClassVectorVars"]     = jsonCPP["ClassVectorVars"];
    json["MainFunctions"]       = jsonCPP["MainFunctions"];
    json["MainInclude"]         = jsonCPP["MainInclude"];
  }
  inputCodeInfo.pShaderCC->GenerateShaders(json, &inputCodeInfo);

  //std::ofstream file(inputCodeInfo.mainClassFileName.parent_path / "z_ubo.json");
  //file << std::setw(2) << json; //
  //file.close();

  {
    std::filesystem::path outName = inputCodeInfo.mainClassFileName.parent_path() / "include" / uboIncludeName;
    kslicer::ApplyJsonToTemplate("templates/ubo_def.h",  outName, jsonUBO);
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  kslicer::ApplyJsonToTemplate("templates/ubo_def.h",  uboOutName, jsonUBO); // need to call it after "GenerateShaders"
  kslicer::CheckForWarnings(inputCodeInfo);

  std::cout << "(9) Generate host code again for 'ListRequiredDeviceFeatures' " << std::endl << std::endl;
  if(true)
  {
    auto jsonCPP = PrepareJsonForAllCPP(inputCodeInfo, compiler, inputCodeInfo.mainFunc, generalDecls,
                                        rawname + ToLowerCase(suffix) + ".h", threadsOrder,
                                        uboIncludeName, composeImplName, jsonUBO, textGenSettings);
    kslicer::ApplyJsonToTemplate("templates/vk_class_init.cpp", rawname + ToLowerCase(suffix) + "_init.cpp", jsonCPP);
  }
  std::cout << "(10) Finished!  " << std::endl;

  return 0;
}