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

//using namespace clang;
#include "template_rendering.h"

using kslicer::KernelInfo;
using kslicer::DataMemberInfo;

std::vector<std::string> ListProcessedFiles(nlohmann::json a_filesArray, const std::string& a_optionsPath)
{
  std::vector<std::string> allFiles;
  if(!a_filesArray.is_array())
    return allFiles;

  std::filesystem::path optionsPath(a_optionsPath);
  std::filesystem::path optionsFolder = optionsPath.parent_path();
  
  for(const auto& param : a_filesArray) 
  {
    std::filesystem::path path = param;
    if(path.is_absolute())
      allFiles.push_back(path.string());
    else
    {
      std::filesystem::path fullPath = optionsFolder / path;
      allFiles.push_back(fullPath.string());
    }
  }
  return allFiles;
}

int main(int argc, const char **argv) // 
{
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "[main]: work_dir = " << std::filesystem::current_path() << std::endl;
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (argc < 2)
  {
    std::cout << "Normal usage: kslicer <config.json> or kslicer -config <config.json> " << std::endl;
    return 1;
  }

  std::cout << "CMD LINE: " << std::endl;
  for(int i=0;i<argc;i++)
     std::cout << i << ": " << argv[i] << std::endl;
  std::cout << std::endl;
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  // try to find config file
  //
  std::string optionsPath = "";
  {
    for(int argId = 1; argId < argc; argId++ )
    {
      if((std::string(argv[argId]) == "-options" || std::string(argv[argId]) == "-config")  && argId+1 < argc)
      {
        optionsPath = argv[argId+1];
        break;
      }
    }
  
    if(std::string(argv[1]).find(".json") != std::string::npos)
      optionsPath = argv[1];
  }
  
  nlohmann::json inputOptions;
  std::ifstream ifs(optionsPath);
  if(optionsPath == "")
    std::cout << "[main]: config is missing, which is ok in general" << std::endl;
  else if(!ifs.is_open())
    std::cout << "[main]: warning, config is not found at '" << optionsPath.c_str() << "', is it ok?" << std::endl;
  else 
    inputOptions = nlohmann::json::parse(ifs, nullptr, true, true);
  
  auto paramsFromConfig = inputOptions["options"];

  std::vector<std::string> allFiles = ListProcessedFiles(inputOptions["source"], optionsPath);

  std::vector<std::string> ignoreFiles;
  std::vector<std::string> processFiles;
  std::vector<std::string> cppIncludesAdditional;
  std::filesystem::path    fileName;
  auto paramsFromCmdLine = ReadCommandLineParams(argc, argv, fileName,                                        // ==>
                                                 allFiles, ignoreFiles, processFiles, cppIncludesAdditional); // <==

  std::unordered_map<std::string, std::string> params;
  {
    for(const auto& param : paramsFromConfig.items()) // take params initially from config
      params[param.key()] = param.value().is_string() ? param.value().get<std::string>() : param.value().dump();
                                                  
    for(const auto& param : paramsFromCmdLine)        // and overide them from command line
      params[param.first] = param.second;
  }

  auto mainClassNode = inputOptions["mainClass"];
  if(mainClassNode.is_string())
    params["-mainClass"] = mainClassNode.get<std::string>();

  std::string              composeAPIName  = "";
  std::string              composeImplName = "";
  std::vector<std::string> composeIntersections;
  {
    auto composClassNodes = inputOptions["composClasses"];
    for(auto composNode : composClassNodes) {
      composeAPIName  = composNode["interface"];
      composeImplName = composNode["implementation"];
      break; // currently support only single composition
    }
  }

  std::filesystem::path mainFolderPath  = fileName.parent_path();
  std::string mainClassName   = "TestClass";
  std::string stdlibFolder    = "";
  std::string patternName     = "rtv";
  std::string shaderCCName    = "clspv";
  std::string suffix          = "_Generated";
  std::string shaderFolderPrefix    = "";

  uint32_t    threadsOrder[3] = {0,1,2};
  uint32_t    warpSize        = 32;
  bool        useCppInKernels = false;
  bool        halfFloatTextures  = false;
  bool        useMegakernel      = false;
  bool        usePersistentThreads = false;
  auto        defaultVkernelType = kslicer::VKERNEL_IMPL_TYPE::VKERNEL_SWITCH;
  bool        enableSubGroupOps  = false;
  int         ispcThreadModel    = 0;
  bool        ispcExplicitIndices = false;
  bool        genGPUAPI           = false;

  if(params.find("-mainClass") != params.end())
    mainClassName = params["-mainClass"];

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

  if(params.find("-warpSize") != params.end())
    warpSize = atoi(params["-warpSize"].c_str());

  if(params.find("-enableSubgroup") != params.end())
    enableSubGroupOps = atoi(params["-enableSubgroup"].c_str());

  if(params.find("-halfTex") != params.end())
    halfFloatTextures = (params["-halfTex"] == "1");

  if(params.find("-megakernel") != params.end())
    useMegakernel = (params["-megakernel"] == "1");

  if(params.find("-persistent") != params.end())
    usePersistentThreads = (params["-persistent"] == "1");

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

  if(params.find("-gen_gpu_api") != params.end())
    genGPUAPI = atoi(params["-gen_gpu_api"].c_str());

  kslicer::ShaderFeatures forcedFeatures;
  {
    if(params.find("-forceEnableHalf") != params.end())
      forcedFeatures.useHalfType = (atoi(params["-forceEnableHalf"].c_str()) == 1);
    if(params.find("-forceEnableInt8") != params.end())
      forcedFeatures.useByteType = (atoi(params["-forceEnableInt8"].c_str()) == 1);
    if(params.find("-forceEnableInt16") != params.end())
      forcedFeatures.useShortType = (atoi(params["-forceEnableInt16"].c_str()) == 1);
    if(params.find("-forceEnableInt64") != params.end())
      forcedFeatures.useInt64Type = (atoi(params["-forceEnableInt64"].c_str()) == 1);
  }

  kslicer::TextGenSettings textGenSettings;
  {
    if(params.find("-enable_motion_blur") != params.end())
      textGenSettings.enableMotionBlur = atoi(params["-enable_motion_blur"].c_str()) != 0;
    if(params.find("-force_ray_tracing_pipeline") != params.end())
    {
      bool isEnabled = (atoi(params["-force_ray_tracing_pipeline"].c_str()) != 0);
      textGenSettings.enableRayGen      = isEnabled;
      textGenSettings.enableRayGenForce = isEnabled;
    }
    else if(params.find("-enable_ray_tracing_pipeline") != params.end())
      textGenSettings.enableRayGen = (atoi(params["-enable_ray_tracing_pipeline"].c_str()) != 0) || textGenSettings.enableMotionBlur;
    if(params.find("-enable_callable_shaders") != params.end()) // enable_callable_shaders
      textGenSettings.enableCallable = (atoi(params["-enable_callable_shaders"].c_str()) != 0);
    if(params.find("-timestamps") != params.end())
      textGenSettings.enableTimeStamps = (atoi(params["-timestamps"].c_str()) != 0);
    textGenSettings.genSeparateGPUAPI = genGPUAPI;
    textGenSettings.interfaceName = mainClassName;
    if(params.find("-makerInterfaceName") != params.end())
      textGenSettings.interfaceName = params["-makerInterfaceName"];
    if(params.find("-useCUB") != params.end())
      textGenSettings.useCUBforCUDA = (atoi(params["-useCUB"].c_str()) != 0);
  }

  // include and process folders
  //
  std::vector<std::filesystem::path> ignoreFolders;
  std::vector<std::filesystem::path> processFolders;
  for(auto p : params)
  {
    std::string folderT = p.second;
    std::transform(folderT.begin(), folderT.end(), folderT.begin(), [](unsigned char c){ return std::tolower(c); });

    if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I' && folderT == "ignore")
      ignoreFolders.push_back(p.first.substr(2));
    else if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I' && folderT == "process")
      processFolders.push_back(p.first.substr(2));
  }

  for(auto folder : inputOptions["includeProcess"])
  {
    if(!folder.is_string())
      continue;

    std::filesystem::path path(folder.get<std::string>());
    if(!path.is_absolute())
      path = std::filesystem::absolute(std::filesystem::path(optionsPath).parent_path() / path);

    if(std::filesystem::exists(path) && std::filesystem::is_directory(path))
      processFolders.push_back(path);
    else
      std::cout << "[main]: bad folder from 'includeProcess' list: " << path.c_str() << std::endl;
  }

  for(auto folder : inputOptions["includeIgnore"])
  {
    if(!folder.is_string())
      continue;

    std::filesystem::path path(folder.get<std::string>());
    if(!path.is_absolute())
      path = std::filesystem::absolute(std::filesystem::path(optionsPath).parent_path() / path);

    if(std::filesystem::exists(path) && std::filesystem::is_directory(path))
      ignoreFolders.push_back(path);
    else
      std::cout << "[main]: bad folder from 'includeIgnore' list: " << path.c_str() << std::endl;
  }

  // make specific checks to be sure user don't include these files to hit project as normal files
  //
  {
    auto processFolders2 = processFolders;
    processFolders2.push_back(mainFolderPath);
    kslicer::CheckInterlanIncInExcludedFolders(processFolders2);
  }

  std::vector<const char*> argsForClang = ExcludeSlicerParams(argc, argv, params, fileName.c_str());
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
  inputCodeInfo.m_timestampPoolSize = textGenSettings.enableTimeStamps ? 0 : uint32_t(-1);

  if(shaderCCName == "glsl" || shaderCCName == "GLSL")
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::GLSLCompiler>(inputCodeInfo.mainClassSuffix);
    inputCodeInfo.pHostCC   = std::make_shared<kslicer::VulkanCodeGen>();
    inputCodeInfo.processFolders.push_back("include/");
  }
  else if(shaderCCName == "slang" || shaderCCName == "SLANG" || shaderCCName == "Slang" || shaderCCName == "cuda_slang")
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::SlangCompiler>(inputCodeInfo.mainClassSuffix);
    if(shaderCCName == "cuda_slang")
      inputCodeInfo.pHostCC = std::make_shared<kslicer::CudaCodeGen>(); 
    else
      inputCodeInfo.pHostCC = std::make_shared<kslicer::VulkanCodeGen>(); 
    inputCodeInfo.processFolders.push_back("include/");
  }
  else if(shaderCCName == "cuda" || shaderCCName == "CUDA")
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::CudaCompiler>(inputCodeInfo.mainClassSuffix); 
    inputCodeInfo.pHostCC   = std::make_shared<kslicer::CudaCodeGen>();
    inputCodeInfo.ignoreFolders.push_back("include/");
  }
  else if(shaderCCName == "ispc" || shaderCCName == "ISPC")
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::ISPCCompiler>(useCppInKernels, inputCodeInfo.mainClassSuffix);
    inputCodeInfo.pHostCC   = std::make_shared<kslicer::ISPCCodeGen>();
    inputCodeInfo.ignoreFolders.push_back("include/");
  }
  else
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::ClspvCompiler>(useCppInKernels, inputCodeInfo.mainClassSuffix);
    inputCodeInfo.pHostCC   = std::make_shared<kslicer::VulkanCodeGen>();
    inputCodeInfo.ignoreFolders.push_back("include/");
  }

  inputCodeInfo.defaultVkernelType   = defaultVkernelType;
  inputCodeInfo.halfFloatTextures    = halfFloatTextures;
  inputCodeInfo.megakernelRTV        = useMegakernel;
  inputCodeInfo.persistentRTV        = usePersistentThreads;
  inputCodeInfo.mainClassSuffix      = suffix;
  inputCodeInfo.shaderFolderPrefix   = shaderFolderPrefix;
  inputCodeInfo.globalShaderFeatures = forcedFeatures;
  
  // analize multiple definitions of args which can not be processed via hash-table params
  //

  std::vector<std::string> baseClases;

  for(int argId = 1; argId < argc; argId++ )
  {
    if(std::string(argv[argId]) == "-intersectionShader" && argId+1 < argc) {
      std::string shaderClassAndFunc = argv[argId+1];
      auto splitPos = shaderClassAndFunc.find("::");
      std::string className = shaderClassAndFunc.substr(0, splitPos);
      std::string funcName  = shaderClassAndFunc.substr(splitPos + 2);
      inputCodeInfo.intersectionShaders.push_back( std::make_pair(className, funcName) );
    }
    else if(std::string(argv[argId]) == "-intersectionTriangle" && argId+1 < argc) {
      const std::string className = argv[argId+1];
      inputCodeInfo.intersectionTriangle.push_back( std::make_pair(className, className) );
    }
    else if(std::string(argv[argId]) == "-intersectionWhiteList" && argId+1 < argc) {
      const std::string className = argv[argId+1];
      inputCodeInfo.intersectionWhiteList.insert(className);
    }
    else if(std::string(argv[argId]) == "-intersectionBlackList" && argId+1 < argc) {
      const std::string className = argv[argId+1];
      inputCodeInfo.intersectionBlackList.insert(className);
    }
    else if (std::string(argv[argId]) == "-baseClass" && argId+1 < argc) {
      baseClases.push_back(argv[argId+1]);
    }
    else if(std::string(argv[argId]) == "-with_buffer_reference" && argId+1 < argc)
    {
      const std::string buffName = argv[argId+1];
      if(buffName == "all")
        inputCodeInfo.withBufferReferenceAll = true;
      else if(buffName != "")
        inputCodeInfo.withBufferReference.insert(buffName);
    }
    else if(std::string(argv[argId]) == "-without_buffer_reference" && argId+1 < argc)
    {
      const std::string buffName = argv[argId+1];
      if(buffName != "")
        inputCodeInfo.withoutBufferReference.insert(buffName);
    }
    else if(std::string(argv[argId]) == "-typedef" && argId+1 < argc)
    {
      const std::string buffName = argv[argId+1];
      std::stringstream strIn(buffName);
      char original[64];
      char replace[64];
      strIn >> original >> replace;
      inputCodeInfo.userTypedefs.push_back(std::make_pair(std::string(original), std::string(replace)));
    }
    else if(std::string(argv[argId]) == "-intersectionShaderPlace" && argId+1 < argc)
    {
      composeIntersections.push_back(argv[argId+1]);
    }
  }

  for(auto base : inputOptions["baseClasses"]) {
    if(base.is_string())
      baseClases.push_back(base.get<std::string>());
  }
  

  // read compos classes, intersection shaders and e.t.c
  {
    auto composClassNodes = inputOptions["composClasses"];
    for(auto composNode : composClassNodes) {
      composeAPIName  = composNode["interface"];
      composeImplName = composNode["implementation"];
      
      auto intersection = composNode["intersection"];
      if(intersection != nullptr)
      {
        std::string className = intersection["interface"];
        std::string funcName  = intersection["shader"];
        
        inputCodeInfo.intersectionShaders.push_back( std::make_pair(className, funcName) );
        composeIntersections.push_back(composeAPIName);
        composeIntersections.push_back(composeImplName);

        if(intersection["triangle"] != nullptr) {
          const std::string triClassName = intersection["triangle"];
          inputCodeInfo.intersectionTriangle.push_back(std::make_pair(className, className));
        }

        if(intersection["whiteList"] != nullptr) {
          for(auto node : intersection["whiteList"]) 
            inputCodeInfo.intersectionWhiteList.insert(std::string(node)); 
        }

        if(intersection["blackList"] != nullptr) {
          for(auto node : intersection["blackList"])
            inputCodeInfo.intersectionBlackList.insert(std::string(node)); 
        }
        
      }
      break; // currently support only single composition
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::unique_ptr<kslicer::MainClassInfo> pInputCodeInfoImpl = nullptr;

  clang::CompilerInstance compiler;
  clang::DiagnosticOptions diagnosticOptions;
  compiler.createDiagnostics();  //compiler.createDiagnostics(argc, argv);

  // Create an invocation that passes any flags to preprocessor
  std::shared_ptr<clang::CompilerInvocation> Invocation = std::make_shared<clang::CompilerInvocation>();
  clang::CompilerInvocation::CreateFromArgs(*Invocation, args, compiler.getDiagnostics());
  compiler.setInvocation(Invocation);

  // Set default target triple
  std::shared_ptr<clang::TargetOptions> pto = std::make_shared<clang::TargetOptions>();
  pto->Triple     = llvm::sys::getDefaultTargetTriple();
  clang::TargetInfo *pti = clang::TargetInfo::CreateTargetInfo(compiler.getDiagnostics(), pto);
  compiler.setTarget(pti);
  compiler.getLangOpts().GNUMode = 1;
  compiler.getLangOpts().CXXExceptions = 1;
  compiler.getLangOpts().RTTI        = 1;
  compiler.getLangOpts().Bool        = 1;
  compiler.getLangOpts().CPlusPlus   = 1;
  compiler.getLangOpts().CPlusPlus14 = 1;
  compiler.getLangOpts().CPlusPlus17 = 1;
  compiler.createFileManager();
  compiler.createSourceManager(compiler.getFileManager());
  
  /////////////////////////////////////////////////////////////////////////////////////////////////// -stdlibFolder
  if(stdlibFolder == "")
  {
    std::filesystem::path currentPath  = std::filesystem::current_path();
    std::filesystem::path tinystl1Path = currentPath / std::filesystem::path("TINYSTL");
    std::filesystem::path tinystl2Path = currentPath.parent_path() / std::filesystem::path("TINYSTL");
    
    if(std::filesystem::exists(tinystl1Path) && std::filesystem::is_directory(tinystl1Path))
      stdlibFolder = tinystl1Path.string();
    else if(std::filesystem::exists(tinystl2Path) && std::filesystem::is_directory(tinystl2Path))
      stdlibFolder = tinystl2Path.string();
  }
  
  auto alreadyFound = std::find(inputCodeInfo.ignoreFolders.begin(), inputCodeInfo.ignoreFolders.end(), stdlibFolder);
  if(stdlibFolder != "" && alreadyFound == inputCodeInfo.ignoreFolders.end())
    inputCodeInfo.ignoreFolders.push_back(stdlibFolder);
  /////////////////////////////////////////////////////////////////////////////////////////////////// -stdlibFolder

  // (0) add path dummy include files for STL and e.t.c. (we don't want to parse actually std library)
  //
  auto& headerSearchOptions = compiler.getHeaderSearchOpts();
  headerSearchOptions.AddPath(stdlibFolder.c_str(), clang::frontend::Angled, false, false);
  for(const auto& includePath : processFolders)
    headerSearchOptions.AddPath(includePath.c_str(), clang::frontend::Angled, false, false);
  for(const auto& includePath : ignoreFolders)
    headerSearchOptions.AddPath(includePath.c_str(), clang::frontend::Angled, false, false);

  //headerSearchOptions.Verbose = 1;

  compiler.createPreprocessor(clang::TU_Complete);
  compiler.getPreprocessorOpts().UsePredefines = true;
  // register our header lister
  HeaderLister headerLister(&inputCodeInfo);
  compiler.getPreprocessor().addPPCallbacks(std::make_unique<HeaderLister>(headerLister));
  compiler.createASTContext();

  const clang::FileEntry *pFile = compiler.getFileManager().getFile(fileName.u8string()).get();
  compiler.getSourceManager().setMainFileID( compiler.getSourceManager().createFileID( pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(), &compiler.getPreprocessor());

  // init clang tooling
  //
  const std::string filenameString = fileName.u8string();
  std::vector<const char*> argv2 = {argv[0], filenameString.c_str()};
  std::vector<std::string> extraArgs; 
  extraArgs.reserve(256);
  for(auto p : params) {
    if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I') {
      extraArgs.push_back(std::string("-extra-arg=") + p.first);
      argv2.push_back(extraArgs.back().c_str());
    }
  }
  for(const auto& includePath : processFolders) {
    extraArgs.push_back(std::string("-extra-arg=") + std::string("-I") + includePath.string());
    argv2.push_back(extraArgs.back().c_str());
  }
  for(const auto& includePath : ignoreFolders) {
    extraArgs.push_back(std::string("-extra-arg=") + std::string("-I") + includePath.string());
    argv2.push_back(extraArgs.back().c_str());
  }
  //if(optionsPath != "") 
  { 
    extraArgs.push_back(std::string("-extra-arg=") + std::string("-I") + stdlibFolder);
    argv2.push_back(extraArgs.back().c_str());
  }
  
  argv2.push_back("--");
  int argSize = argv2.size();

  llvm::cl::OptionCategory GDOpts("global-detect options");
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
  
  // Parse code, initial pass
  //
  std::cout << "(1) Processing class '" << mainClassName.c_str() << "' with initial pass" << std::endl;
  std::cout << "{" << std::endl;

  std::vector<std::string> composClassNames;
  {
    if(composeAPIName != "")
      composClassNames.push_back(composeAPIName);
    if(composeImplName != "")
      composClassNames.push_back(composeImplName);
   
    if(composeAPIName != "ISceneObject" && composeAPIName.find("ISceneObject") != std::string::npos) // need to add 'ISceneObject' if ISceneObject2 or ISceneObject_LiteRT or sms like that is used for API 
      composClassNames.push_back("ISceneObject");
  }

  kslicer::InitialPassASTConsumer firstPassData(cfNames, mainClassName, composClassNames, compiler, inputCodeInfo);
  // TODO: move this inside constructor ... 
  {
    for(size_t i=0;i<baseClases.size();i++) 
    { 
      std::string className = baseClases[i];
      firstPassData.rv.m_baseClassInfo[className] = kslicer::ClassInfo(className);
      firstPassData.rv.m_baseClassInfo[className].baseClassOrder = int(i);
    }
  }
  ParseAST(compiler.getPreprocessor(), &firstPassData, compiler.getASTContext());

  if(firstPassData.rv.mci.astNode == nullptr)
  {
    std::cout << "  [main]: critical error, main class '" << mainClassName.c_str() << "' not found" << std::endl;
    return 0;
  }

  // вызов compiler.getDiagnosticClient().EndSourceFile() 
  // обеспечивает корректное завершение обработки диагностических сообщений 
  // для текущего исходного файла в процессе компиляции с использованием Clang.
  compiler.getDiagnosticClient().EndSourceFile(); 

  auto pComposAPI  = firstPassData.rv.m_composedClassInfo.find(composeAPIName);
  auto pComposImpl = firstPassData.rv.m_composedClassInfo.find(composeImplName);

  if(pComposAPI != firstPassData.rv.m_composedClassInfo.end() && pComposImpl != firstPassData.rv.m_composedClassInfo.end()) // if compos classes are found
  {
    std::string composMemberName = kslicer::PerformClassComposition(firstPassData.rv.mci, pComposAPI->second, pComposImpl->second);     // perform class composition
    for(const auto& name : composClassNames)
      inputCodeInfo.composPrefix[name] = composMemberName;
    
    inputCodeInfo.composClassNames.insert(composeImplName);
    for(auto intersectionPlace : composeIntersections)
      inputCodeInfo.composIntersection.insert(intersectionPlace);
  }
  else if(baseClases.size() != 0)
  { 
    std::vector<const clang::CXXRecordDecl*> aux_classes;
    aux_classes.reserve(firstPassData.rv.m_composedClassInfo.size());

    // Преобразование unordered_map в vector
    std::transform(firstPassData.rv.m_baseClassInfo.begin(), firstPassData.rv.m_baseClassInfo.end(), 
                   std::back_inserter(aux_classes), [](const auto& pair) { return pair.second.astNode; });

    auto sorted = kslicer::ExtractAndSortBaseClasses(aux_classes, firstPassData.rv.mci.astNode);
    for(size_t classOrder = 0; classOrder < sorted.size(); classOrder++) {
      const auto& baseClass = sorted[classOrder];
      auto typeName = baseClass->getQualifiedNameAsString();
      const auto& classInfo = firstPassData.rv.m_baseClassInfo[typeName]; // !!!!!
      kslicer::PerformInheritanceMerge(firstPassData.rv.mci, classInfo);
      inputCodeInfo.mainClassNames[typeName] = int(classOrder) + 1; 
    }
  }
  
  inputCodeInfo.mainClassNames[inputCodeInfo.mainClassName] = 0; // put main (derived) class name in this hash-set, use 'mainClassNames' instead of 'mainClassName' later
  
  // merge mainClassNames and composClassNames in single array, add 'const Type' names to it; TODO: merge to single function
  {
    inputCodeInfo.dataClassNames.clear();
    //inputCodeInfo.dataClassNames.insert(inputCodeInfo.mainClassNames.begin(),   inputCodeInfo.mainClassNames.end());
    for(auto c : inputCodeInfo.mainClassNames)
      inputCodeInfo.dataClassNames.insert(c.first);
    inputCodeInfo.dataClassNames.insert(inputCodeInfo.composClassNames.begin(), inputCodeInfo.composClassNames.end());
    inputCodeInfo.dataClassNames.insert("ISceneObject");  // TODO: list all base classes for compose classes 
    inputCodeInfo.dataClassNames.insert("ISceneObject2"); // TODO: list all base classes for compose classes 
    
    std::vector<std::string> constVarianst; 
    constVarianst.reserve(inputCodeInfo.dataClassNames.size());
    for(auto name : inputCodeInfo.dataClassNames) {
      constVarianst.push_back(name + "*");
      constVarianst.push_back(name + " *");
      constVarianst.push_back(name + "&");
      constVarianst.push_back(name + " &");
      constVarianst.push_back(std::string("const ") + name);
      constVarianst.push_back(std::string("const ") + name + "*");
      constVarianst.push_back(std::string("const ") + name + " *");
      constVarianst.push_back(std::string("const ") + name + "&");
      constVarianst.push_back(std::string("const ") + name + " &");
    }
    for(auto name : constVarianst)
      inputCodeInfo.dataClassNames.insert(name);
  }

  inputCodeInfo.mainClassFileInclude = firstPassData.rv.MAIN_FILE_INCLUDE;
  inputCodeInfo.mainClassASTNode     = firstPassData.rv.mci.astNode;
  inputCodeInfo.allKernels           = firstPassData.rv.mci.funKernels;
  inputCodeInfo.allDataMembers       = firstPassData.rv.mci.dataMembers;
  inputCodeInfo.ctors                = firstPassData.rv.mci.ctors;
  inputCodeInfo.allMemberFunctions   = firstPassData.rv.mci.funMembers;
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
    mainFuncRef.Node  = firstPassData.rv.mci.funControls[mainFuncName].astNode;

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
    if(kernel.name.find("kernelBE") != std::string::npos)
      inputCodeInfo.ProcessBlockExpansionKernel(kernel, compiler);

    if(kernel.hasFinishPass) // add additional buffers for reduction
    {
      uint32_t buffNumber = 0;
      for(auto& redVar : kernel.subjectedToReduction)
      {
        if(inputCodeInfo.pShaderCC->SupportAtomicGlobal(redVar.second))
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

  std::vector<kslicer::FuncData> usedFunctions = kslicer::ExtractUsedFunctions(inputCodeInfo, compiler); // recursive processing of functions used by kernel, extracting all needed functions
  std::vector<kslicer::DeclInClass> usedDecls;
  for(auto name : inputCodeInfo.mainClassNames)
  {
    auto astNode = inputCodeInfo.allASTNodes.find(name.first);
    if(astNode != inputCodeInfo.allASTNodes.end())
    {
      auto declsPerClass = kslicer::ExtractTCFromClass(name.first, astNode->second, compiler, Tool);
      usedDecls.insert(usedDecls.end(), declsPerClass.begin(), declsPerClass.end());
    }
  }

  for(const auto& usedDecl : usedDecls) // merge usedDecls with generalDecls
  {
    bool found = false;
    for(auto& currDecl : generalDecls)
    {
      if(currDecl.name == usedDecl.name)
      {
        found = true;
        currDecl.inClass = true;
        break;
      }
    }
    if(!found)
      generalDecls.push_back(usedDecl);
  }

  // process virtual functions
  // 
  std::cout << "  (4.0) Process Virtual-Functions-Hierarchies:" << std::endl;
  std::unordered_map<const clang::FunctionDecl*, kslicer::FuncData> procesedFunctions;
  for(auto f  : usedFunctions)
    procesedFunctions[f.astNode] = f;
  
  size_t oldSizeOfFunctions = usedFunctions.size();

  for(auto& k : inputCodeInfo.kernels)
  {
    bool hasVirtual = false;
    for(const auto& f : k.second.usedMemberFunctions) {
      if(f.second.isVirtual) {
        hasVirtual = true;
        break;
      }
    }
    
    if(hasVirtual)
    {
      inputCodeInfo.ProcessVFH(firstPassData.rv.m_classList, compiler);
      inputCodeInfo.ExtractVFHConstants(compiler, Tool);
      
      auto sortedFunctions = kslicer::ExtractUsedFromVFH(inputCodeInfo, compiler, k.second.usedMemberFunctions);
      for(auto f : sortedFunctions) 
      {
        if(f.isMember)
          continue;
        
        if(procesedFunctions.find(f.astNode) == procesedFunctions.end()) 
        {
          procesedFunctions[f.astNode] = f;
          usedFunctions.push_back(f);
        }
      }
  
      inputCodeInfo.VisitAndPrepare_KF(k.second, compiler); // move data from usedContainersProbably to usedContainers if a kernel actually uses it
  
      std::cout << std::endl;
    }
  }

  if(usedFunctions.size() != oldSizeOfFunctions)
    std::sort(usedFunctions.begin(), usedFunctions.end(), [](const auto& a, const auto& b) { return a.depthUse > b.depthUse; });

  std::vector<std::string> usedDefines = kslicer::ExtractDefines(compiler);

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  inputCodeInfo.AddSpecVars_CF(inputCodeInfo.mainFunc, inputCodeInfo.kernels);

  if(inputCodeInfo.pShaderCC->MemberFunctionsAreSupported()) // We don't implement this for OpenCL kernels yet ... or at all.
  {
    std::cout << "(4.1) Process Member function calls, extract data accesed in member functions " << std::endl;
    std::cout << "{" << std::endl;
    for(auto& k : inputCodeInfo.kernels)
    {
      std::vector<kslicer::FuncData> usedFunctionsCopy;
      for(auto memberF : k.second.usedMemberFunctions)                    // (1) process member functions
        usedFunctionsCopy.push_back(memberF.second);                                                                                                                           
      usedFunctionsCopy.push_back(kslicer::FuncDataFromKernel(k.second)); // (2) process kernel in the same way as used member functions by this kernel  
      
      clang::Rewriter rewrite2;
      rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
    
      auto pInfoVisitor = std::make_shared<kslicer::KernelInfoVisitor>(rewrite2, compiler, &inputCodeInfo, k.second, true);

      for(const auto& f : usedFunctionsCopy)
      {
        if(!f.isMember) // and if is called from this kernel.It it is called, list all input parameters for each call!
          continue;

        // get shader features and other from used member function
        //
        pInfoVisitor->TraverseDecl(const_cast<clang::FunctionDecl*>(f.astNode));

        // list all input parameters for each call of member function inside kernel; in this way we know which textures, vectors and samplers were actually used by these functions
        //
        std::unordered_map<std::string, kslicer::UsedContainerInfo> auxContainers;
        auto machedParams = kslicer::ArgMatchTraversal    (&k.second, f, usedFunctions, inputCodeInfo, compiler);
        auto usedMembers  = kslicer::ExtractUsedMemberData(&k.second, f, usedFunctions, auxContainers, inputCodeInfo, compiler);

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
        
      } // end for(const auto& f : usedFunctions)
    } // end for(auto& k : inputCodeInfo.kernels)

    for(auto& k : inputCodeInfo.kernels) 
    {
      // fix "usedInKernel" flag for members that were used in member functions but not in kernels directly
      //
      for(const auto& c : k.second.usedContainers)
      {
        auto pFound = inputCodeInfo.allDataMembers.find(c.second.name);
        if(pFound != inputCodeInfo.allDataMembers.end())
          pFound->second.usedInKernel = true;
      }
      
      // set bind mode (explicit with descriptor set or impleceit with buffer reference)
      //
      for(auto& c : k.second.usedContainers)
      {
        auto pNoBufferReference  = inputCodeInfo.withoutBufferReference.find(c.second.name);
        auto pYesBufferReference = inputCodeInfo.withBufferReference.find(c.second.name);
        if(pNoBufferReference != inputCodeInfo.withoutBufferReference.end())
          c.second.bindWithRef = false;
        else if((pYesBufferReference != inputCodeInfo.withBufferReference.end() || inputCodeInfo.withBufferReferenceAll))
          c.second.bindWithRef = true;

        if(c.second.kind != kslicer::DATA_KIND::KIND_VECTOR)
          c.second.bindWithRef = false;

        auto pFound = inputCodeInfo.allDataMembers.find(c.second.name);
        if(pFound != inputCodeInfo.allDataMembers.end())
          pFound->second.bindWithRef = c.second.bindWithRef; 
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
      std::cout << "  process CF as megakernel " << mainFunc.Name.c_str() << std::endl;
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
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  inputCodeInfo.allDescriptorSetsInfo.clear();          // clear (1)
  if(inputCodeInfo.m_timestampPoolSize != uint32_t(-1)) // clear (2)
    inputCodeInfo.m_timestampPoolSize = 0;
  for(auto& mainFunc : inputCodeInfo.mainFunc)
  {
    std::cout << "  process " << mainFunc.Name.c_str() << std::endl;
    inputCodeInfo.VisitAndRewrite_CF(mainFunc, compiler);  // ==> output to mainFunc and inputCodeInfo.allDescriptorSetsInfo
  }
  inputCodeInfo.PlugSpecVarsInCalls_CF(inputCodeInfo.mainFunc, inputCodeInfo.kernels, inputCodeInfo.allDescriptorSetsInfo);
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////////////////////// fakeOffset flag for local variables

  // analize inputCodeInfo.allDescriptorSetsInfo to mark all args of each kernel that we need to apply fakeOffset(tid) inside kernel to this arg
  //
  for(const auto& call : inputCodeInfo.allDescriptorSetsInfo)
    inputCodeInfo.ProcessCallArs_KF(call);

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::cout << "(6) Calc offsets for all class members; ingore unused members that were not marked on previous step" << std::endl;
  std::cout << "{" << std::endl;

  inputCodeInfo.dataMembers = kslicer::MakeClassDataListAndCalcOffsets(inputCodeInfo.allDataMembers);
  inputCodeInfo.AppendAllRefsBufferIfNeeded(inputCodeInfo.dataMembers);                       // add abstract to concrete tables
  inputCodeInfo.AppendAccelStructForIntersectionShadersIfNeeded(inputCodeInfo.dataMembers, composeImplName); 

  inputCodeInfo.ProcessMemberTypes(firstPassData.rv.GetOtherTypeDecls(), compiler.getSourceManager(), generalDecls);                   // ==> generalDecls
  inputCodeInfo.ProcessMemberTypesAligment(inputCodeInfo.dataMembers, firstPassData.rv.GetOtherTypeDecls(), compiler.getASTContext()); // ==> inputCodeInfo.dataMembers

  std::sort(inputCodeInfo.dataMembers.begin(), inputCodeInfo.dataMembers.end(), kslicer::DataMemberInfo_ByAligment()); // sort by aligment in GLSL

  auto jsonUBO = kslicer::PrepareUBOJson(inputCodeInfo, inputCodeInfo.dataMembers, compiler, textGenSettings);

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  // if user set custom work group size for kernels via hint file we should apply it befor generating kernels
  //
  uint32_t defaultWgSize[3][3] = {{256, 1, 1}, {32,  8, 1}, {8,   8, 8}};

  auto kernelsOptionsAll = inputOptions["kernels"];

  if(kernelsOptionsAll["default"] != nullptr)
  {
    defaultWgSize[0][0] = kernelsOptionsAll["default"]["wgSize"][0];
    defaultWgSize[0][1] = kernelsOptionsAll["default"]["wgSize"][1];
    defaultWgSize[0][2] = kernelsOptionsAll["default"]["wgSize"][2];
  }
  else if(kernelsOptionsAll["default1D"] != nullptr)
  {
    defaultWgSize[0][0] = kernelsOptionsAll["default1D"]["wgSize"][0];
    defaultWgSize[0][1] = kernelsOptionsAll["default1D"]["wgSize"][1];
    defaultWgSize[0][2] = kernelsOptionsAll["default1D"]["wgSize"][2];
  }

  if(kernelsOptionsAll["default2D"] != nullptr)
  {
    defaultWgSize[1][0] = kernelsOptionsAll["default2D"]["wgSize"][0];
    defaultWgSize[1][1] = kernelsOptionsAll["default2D"]["wgSize"][1];
    defaultWgSize[1][2] = kernelsOptionsAll["default2D"]["wgSize"][2];
  }

  if(kernelsOptionsAll["default3D"] != nullptr)
  {
    defaultWgSize[2][0] = kernelsOptionsAll["default3D"]["wgSize"][0];
    defaultWgSize[2][1] = kernelsOptionsAll["default3D"]["wgSize"][1];
    defaultWgSize[2][2] = kernelsOptionsAll["default3D"]["wgSize"][2];
  }

  for(auto& nk : inputCodeInfo.kernels)
  {
    auto& kernel   = nk.second;
    if(kernel.be.enabled)
      continue;
    auto kernelDim = kernel.GetDim();
    auto kernelOptions = kernelsOptionsAll[kernel.name];
    if(kernelOptions != nullptr && kernelOptions["wgSize"] != nullptr)
    {
      kernel.wgSize[0] = kernelOptions["wgSize"][0];
      kernel.wgSize[1] = kernelOptions["wgSize"][1];
      kernel.wgSize[2] = kernelOptions["wgSize"][2];
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
      cf.megakernel.enableRTPipeline = (hasAccelStructs && textGenSettings.enableRayGen) || textGenSettings.enableRayGenForce;
      inputCodeInfo.globalDeviceFeatures.useRTX = inputCodeInfo.globalDeviceFeatures.useRTX || hasAccelStructs || textGenSettings.enableRayGenForce;
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
    kernel.enableRTPipeline = (hasAccelStructs && textGenSettings.enableRayGen) || textGenSettings.enableRayGenForce;
  }
  
  ///////////////////////////////////////////////////////////////////////////// fix code for seperate kernel with RT pipeline
  if(!inputCodeInfo.megakernelRTV) // && textGenSettings.enableRayGen 
  { 
    // save correct info
    //
    auto copy = inputCodeInfo.mainFunc;
    auto tmp  = inputCodeInfo.allDescriptorSetsInfo;
    
    inputCodeInfo.allDescriptorSetsInfo.clear();           // clear(1)
    if(inputCodeInfo.m_timestampPoolSize != uint32_t(-1))  // clear(2)
      inputCodeInfo.m_timestampPoolSize = 0;
    for(auto& mainFunc : inputCodeInfo.mainFunc)
    {
      std::cout << "  process " << mainFunc.Name.c_str() << std::endl;
      inputCodeInfo.VisitAndRewrite_CF(mainFunc, compiler);           // ==> output to mainFunc and inputCodeInfo.allDescriptorSetsInfo
    }
    
    // restore correct info
    //
    auto copy2 = inputCodeInfo.mainFunc;
    inputCodeInfo.mainFunc = copy;
    inputCodeInfo.allDescriptorSetsInfo = tmp;
     
    // exctract fixed text
    //
    for(size_t i=0;i<inputCodeInfo.mainFunc.size();i++)
      inputCodeInfo.mainFunc[i].CodeGenerated = copy2[i].CodeGenerated;
  }
  /////////////////////////////////////////////////////////////////////////////

  
  // apply user exclude lists for kernels
  //
  auto kernelOptions = inputOptions["kernels"];
  for(auto& k : inputCodeInfo.kernels) 
  {
    auto p = kernelOptions.find(k.second.name);
    if(p != kernelOptions.end())
    {
      for(auto excludeFunc : (*p)["ExcludeFunctions"].items()) {
        std::string name = excludeFunc.value();
        for(auto p = k.second.usedMemberFunctions.begin(); p!=  k.second.usedMemberFunctions.end(); ++p) {
          if(p->second.name == name) {
            k.second.usedMemberFunctions.erase(p);
            break;
          }
        }
      }

      for(auto excludeData : (*p)["ExcludeData"].items()) {
        std::string name = excludeData.value();
        auto p = k.second.usedContainers.find(name);
        if(p != k.second.usedContainers.end())
          k.second.usedContainers.erase(p);
      }
    }
  }
  
  std::string rawname = kslicer::CutOffFileExt(allFiles[0]);
  auto jsonCPP = PrepareJsonForAllCPP(inputCodeInfo, compiler, inputCodeInfo.mainFunc, generalDecls,
                                      rawname + ToLowerCase(suffix) + ".h", threadsOrder,
                                      composeImplName, jsonUBO, textGenSettings);
  
  std::cout << std::endl;
  std::cout << "(7) Perform final templated text rendering to generate Vulkan calls" << std::endl;
  std::cout << "{" << std::endl;
  if(!inputCodeInfo.pShaderCC->IsCUDA())
    inputCodeInfo.pHostCC->GenerateHost(rawname + ToLowerCase(suffix), jsonCPP, inputCodeInfo, textGenSettings);
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  std::string shaderCCName2 = inputCodeInfo.pShaderCC->Name();
  std::cout << "(8) Generate " << shaderCCName2.c_str() << " kernels" << std::endl;
  std::cout << "{" << std::endl;
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
      cf.megakernel.enableRTPipeline = (hasAccelStructs && textGenSettings.enableRayGen) || textGenSettings.enableRayGenForce;
    }
  }
  else
  {
    for(auto& k : inputCodeInfo.kernels)
      k.second.rewrittenText = inputCodeInfo.VisitAndRewrite_KF(k.second, compiler, k.second.rewrittenInit, k.second.rewrittenFinish);
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  auto json = kslicer::PrepareJsonForKernels(inputCodeInfo, usedFunctions, generalDecls, compiler, threadsOrder, jsonUBO, usedDefines, textGenSettings);
  
  //std::ofstream file(inputCodeInfo.mainClassFileName.parent_path() / "z_debug_kernels.json");
  //file << std::setw(2) << json; //
  //file.close();
  
  if(inputCodeInfo.pShaderCC->IsISPC()) {
    json["Constructors"]        = jsonCPP["Constructors"];
    json["HasCommitDeviceFunc"] = jsonCPP["HasCommitDeviceFunc"];
    json["HasGetTimeFunc"]      = jsonCPP["HasGetTimeFunc"];
    json["ClassVars"]           = jsonCPP["ClassVars"];
    json["ClassVectorVars"]     = jsonCPP["ClassVectorVars"];
    json["MainFunctions"]       = jsonCPP["MainFunctions"];
    json["MainInclude"]         = jsonCPP["MainInclude"];
  }
  else if(inputCodeInfo.pShaderCC->IsCUDA()) {
    jsonCPP["KernelList"]         = json["Kernels"];
    jsonCPP["LocalFunctions"]     = json["LocalFunctions"];
    jsonCPP["AllMemberFunctions"] = json["AllMemberFunctions"];
    inputCodeInfo.pHostCC->GenerateHost(rawname + ToLowerCase(suffix), jsonCPP, inputCodeInfo, textGenSettings);
  }

  inputCodeInfo.pShaderCC->GenerateShaders(json, &inputCodeInfo, textGenSettings); // not used by CUDA backend

  std::cout << "}" << std::endl;
  std::cout << std::endl;

  kslicer::CheckForWarnings(inputCodeInfo);

  std::cout << "(9) Generate host code again for 'ListRequiredDeviceFeatures' " << std::endl;
  std::cout << "{" << std::endl;
  {
    auto jsonCPP = PrepareJsonForAllCPP(inputCodeInfo, compiler, inputCodeInfo.mainFunc, generalDecls,
                                        rawname + ToLowerCase(suffix) + ".h", threadsOrder,
                                        composeImplName, jsonUBO, textGenSettings);
    
    inputCodeInfo.pHostCC->GenerateHostDevFeatures(rawname + ToLowerCase(suffix), jsonCPP, inputCodeInfo, textGenSettings);
  }
  std::cout << "}" << std::endl << std::endl;
  std::cout << "(10) Finished! " << std::endl;

  return 0;
}