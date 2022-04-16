#include <stdio.h>
#include <vector>
#include <system_error>
#include <iostream>
#include <fstream>

#include <unordered_map>
#include <iomanip>
#include <cctype>
#include <queue>

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

#ifdef WIN32
  #include <windows.h>    // for GetCurrentDirectoryW
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <unistd.h>     // for getcwd
#endif

using kslicer::KernelInfo;
using kslicer::DataMemberInfo;

//extern clang::CompilerInstance* g_pCompilerInstance;

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

std::string kslicer::GetRangeSourceCode(const clang::SourceRange a_range, const clang::SourceManager& sm) 
{
  clang::LangOptions lopt;

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

void kslicer::PrintError(const std::string& a_msg, const clang::SourceRange& a_range, const clang::SourceManager& a_sm)
{
  const auto beginLoc  = a_range.getBegin();
  const auto inFileLoc = a_sm.getPresumedLoc(beginLoc);
  
  const auto fileName = std::string(a_sm.getFilename(beginLoc));
  const auto line     = inFileLoc.getLine();
  const auto col      = inFileLoc.getColumn();

  std::string code = GetRangeSourceCode(a_range, a_sm);

  std::cout << fileName.c_str() << ":" << line << ":" << col << ": error: " << a_msg << std::endl;
  std::cout << "--> " << code.c_str() << std::endl;
}

void kslicer::PrintWarning(const std::string& a_msg, const clang::SourceRange& a_range, const clang::SourceManager& a_sm)
{
  const auto beginLoc  = a_range.getBegin();
  const auto inFileLoc = a_sm.getPresumedLoc(beginLoc);
  
  const auto fileName = std::string(a_sm.getFilename(beginLoc));
  const auto line     = inFileLoc.getLine();
  const auto col      = inFileLoc.getColumn();

  std::string code = GetRangeSourceCode(a_range, a_sm);

  std::cout << fileName.c_str() << ":" << line << ":" << col << ": warning: " << a_msg << " --> " << code.c_str() << std::endl;
}

std::string kslicer::CutOffFileExt(const std::string& a_filePath)
{
  const size_t lastindex = a_filePath.find_last_of("."); 
  if(lastindex != std::string::npos)
    return a_filePath.substr(0, lastindex); 
  else
    return a_filePath;
}

std::string kslicer::CutOffStructClass(const std::string& a_typeName)
{
  auto spacePos = a_typeName.find_last_of(" ");
  if(spacePos != std::string::npos)
    return a_typeName.substr(spacePos+1);
  return a_typeName;
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

std::unordered_map<std::string, std::string> ReadCommandLineParams(int argc, const char** argv, std::string& fileName, std::vector<std::string>& allFiles)
{
  std::unordered_map<std::string, std::string> cmdLineParams;
  for(int i=0; i<argc; i++)
  {
    std::string key(argv[i]);
    
    const bool isDefine = key.size() > 1 && key.substr(0,2) == "-D";
    const bool isKey    = key.size() > 0 && key[0] == '-';
    
    if(key != "-v" && !isDefine && isKey) // exclude special "-IfoldePath" form, exclude "-v"
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
      allFiles.push_back(key);
  }

  if(allFiles.size() == 0)
  {
    std::cout << "[kslicer]: no input file is specified " << std::endl;
    exit(0);
  }
  else if(allFiles.size() == 1)
    fileName = allFiles[0];
  else
  {
    fileName = allFiles[0];
    #ifdef WIN32
    const std::string slash = "\\";
    #else
    const std::string slash = "/";
    #endif

    size_t posSlash = fileName.find_last_of(slash); 
    auto   posCPP   = fileName.find(".cpp");
    
    assert(posSlash != std::string::npos);   
    assert(posCPP   != std::string::npos);   

    // merge files to a single temporary file
    auto folderPath = fileName.substr(0, posSlash);
    auto fileName2  = fileName.substr(posSlash+1, posCPP-posSlash-1);
    auto fileNameT  = folderPath + slash + fileName2 + "_temp.cpp";
    
    std::cout << "[kslicer]: merging input files to temporary file '" << fileName2 << "_temp.cpp' " << std::endl;
    std::ofstream fout(fileNameT);
    for(auto file : allFiles)
    {
      fout << "////////////////////////////////////////////////////" << std::endl;
      fout << "//// input file: " << file << std::endl;
      fout << "////////////////////////////////////////////////////" << std::endl;
      std::ifstream fin(file);
      std::string line;
      while (std::getline(fin, line))
        fout << line.c_str() << std::endl;
    } 
    fout.close();
    fileName = fileNameT;
    std::cout << "[kslicer]: merging finished" << std::endl;
  }

  return cmdLineParams;
}

std::vector<const char*> ExcludeSlicerParams(int argc, const char** argv, const std::unordered_map<std::string,std::string>& params)
{
  std::unordered_set<std::string> values;
  for(auto p : params) 
    values.insert(p.second);

  std::vector<const char*> argsForClang; // exclude our input from cmdline parameters and pass the rest to clang
  argsForClang.reserve(argc);
  for(int i=1;i<argc;i++)
  {
    if(params.find(argv[i]) == params.end() && values.find(argv[i]) == values.end()) 
      argsForClang.push_back(argv[i]);
  }

  return argsForClang;
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
    if(!IsAngled && File != nullptr)
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

struct less_than_key2
{
  inline bool operator() (const kslicer::DataMemberInfo& struct1, const kslicer::DataMemberInfo& struct2)
  {
    if(struct1.aligmentGLSL != struct2.aligmentGLSL)
      return (struct1.aligmentGLSL > struct2.aligmentGLSL);
    else if(struct1.sizeInBytes != struct2.sizeInBytes)
      return (struct1.sizeInBytes > struct2.sizeInBytes);
    else if(struct1.isContainerInfo && !struct2.isContainerInfo)
      return false;
    else if(!struct1.isContainerInfo && struct2.isContainerInfo)
      return true;
    else
      return struct1.name < struct2.name;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv)
{
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #ifdef WIN32
  wchar_t NPath[512];
  GetCurrentDirectoryW(512, NPath);
  std::wcout << L"[main]: curr_dir = " << NPath << std::endl;
  #else
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr)
    std::cout << "[main]: curr_dir = " << cwd << std::endl;
  #endif
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct stat sb;

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

  std::vector<std::string> allFiles;
  std::string fileName;
  auto params = ReadCommandLineParams(argc, argv, fileName, allFiles);
  
  std::string mainFolderPath  = GetFolderPath(fileName);
  std::string mainClassName   = "TestClass";
  std::string outGenerated    = "data/generated.cl";
  std::string stdlibFolder    = "";
  std::string patternName     = "rtv";
  std::string shaderCCName    = "clspv";
  std::string hintFile        = "";
  uint32_t    threadsOrder[3] = {0,1,2};
  uint32_t    warpSize        = 32;
  bool        useCppInKernels = false;
  bool        halfFloatTextures  = false;
  bool        useMegakernel      = false;
  auto        defaultVkernelType = kslicer::VKERNEL_IMPL_TYPE::VKERNEL_SWITCH;
  bool        enableSubGroupOps  = false;
  
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

  std::unordered_set<std::string> values;
  std::vector<std::string> includeFolderList;
  std::vector<std::string> includeFolderList2;
  for(auto p : params) 
  {
    values.insert(p.second);
    std::string folderT = p.second;
    std::transform(folderT.begin(), folderT.end(), folderT.begin(), [](unsigned char c){ return std::tolower(c); });

    if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I' && folderT == "ignore")
      includeFolderList.push_back(p.first.substr(2));
    else if(p.first.size() > 1 && p.first[0] == '-' && p.first[1] == 'I' && folderT == "process")
      includeFolderList2.push_back(p.first.substr(2));
  }

  // make specific checks to be sure user don't include these files to hit project as normal files
  //
  {
    auto excludeFolders = includeFolderList2;
    excludeFolders.push_back(mainFolderPath);
    kslicer::CheckInterlanIncInExcludedFolders(excludeFolders);
  }

  std::vector<const char*> argsForClang = ExcludeSlicerParams(argc, argv, params);  
  llvm::ArrayRef<const char*> args(argsForClang.data(), argsForClang.data() + argsForClang.size());

  // Make sure it exists
  #ifndef WIN32
  if (stat(fileName.c_str(), &sb) == -1)
  {
    std::cout << "[main]: error, input file " << fileName.c_str() << " not found!" << std::endl;
    return 0;
  }
  #else
  {
    std::ifstream fin(fileName.c_str());
    if(!fin.is_open())
    {
      std::cout << "[main]: error, input file " << fileName.c_str() << " not found!" << std::endl;
      return 0;
    }
    fin.close();
  }
  #endif

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
  inputCodeInfo.ignoreFolders = includeFolderList;  // set shader folders
  inputCodeInfo.includeCPPFolders       = includeFolderList2; // set common C/C++ folders

  if(shaderCCName == "glsl" || shaderCCName == "GLSL")
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::GLSLCompiler>();
    inputCodeInfo.includeCPPFolders.push_back("include/");
  }
  else
  {
    inputCodeInfo.pShaderCC = std::make_shared<kslicer::ClspvCompiler>(useCppInKernels);
    inputCodeInfo.ignoreFolders.push_back("include/");
  }

  inputCodeInfo.defaultVkernelType = defaultVkernelType;
  inputCodeInfo.halfFloatTextures  = halfFloatTextures;
  inputCodeInfo.megakernelRTV      = useMegakernel;

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
  //compiler.getPreprocessorOpts().addMacroDef("KERNEL_SLICER"); // IT DOES NOT WORKS FOR SOME REASON!!! 
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
  std::vector<const char*> argv2 = {argv[0], fileName.c_str()};
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
  clang::tooling::CommonOptionsParser OptionsParser(argSize, argv2.data(), GDOpts);                         // clang 12
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());       // clang 12
  //auto OptionsParser = clang::tooling::CommonOptionsParser::create(argSize, argv2.data(), GDOpts);        // clang 14
  //clang::tooling::ClangTool Tool(OptionsParser->getCompilations(), OptionsParser->getSourcePathList());   // clang 14

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
  kslicer::InitialPassASTConsumer firstPassData(cfNames, mainClassName, compiler, inputCodeInfo); 
  ParseAST(compiler.getPreprocessor(), &firstPassData, compiler.getASTContext());
  compiler.getDiagnosticClient().EndSourceFile(); // ??? What Is This Line For ???
  
  //#TODO: remove this copy, just pass pointer to 'inputCodeInfo' inside 'firstPassData.rv' and thats all
  //
  inputCodeInfo.allKernels           = firstPassData.rv.functions; 
  inputCodeInfo.allOtherKernels      = firstPassData.rv.otherFunctions;
  inputCodeInfo.allDataMembers       = firstPassData.rv.dataMembers;   
  inputCodeInfo.mainClassFileInclude = firstPassData.rv.MAIN_FILE_INCLUDE;
  inputCodeInfo.mainClassASTNode     = firstPassData.rv.m_mainClassASTNode;
  inputCodeInfo.ctors                = firstPassData.rv.ctors;
  inputCodeInfo.allMemberFunctions   = firstPassData.rv.allMemberFunctions;
  inputCodeInfo.ProcessAllSetters(firstPassData.rv.m_setters, compiler);

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
    mainFuncRef.Node  = firstPassData.rv.m_mainFuncNodes[mainFuncName];

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
    auto& kernel        = nk.second;
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
        if(redVar.second.SupportAtomicLastStep())
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
            if(x.second.type == "struct Sampler" || x.second.type == "struct sampler")
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

            if(member.second.type == "struct Sampler" || member.second.type == "struct sampler") // actually this is not correct!!!
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
              info.kind          = member.second.kind;;
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

  inputCodeInfo.ProcessMemberTypes(firstPassData.rv.GetOtherTypeDecls(), compiler.getSourceManager(), 
                                   generalDecls);                      // ==> generalDecls
  inputCodeInfo.ProcessMemberTypesAligment(inputCodeInfo.dataMembers, firstPassData.rv.GetOtherTypeDecls(), compiler.getASTContext()); // ==> inputCodeInfo.dataMembers

  std::sort(inputCodeInfo.dataMembers.begin(), inputCodeInfo.dataMembers.end(), less_than_key2()); // sort by aligment in GLSL

  auto jsonUBO               = kslicer::PrepareUBOJson(inputCodeInfo, inputCodeInfo.dataMembers, compiler);
  std::string uboIncludeName = inputCodeInfo.mainClassName + "_ubo.h";

  std::string uboOutName = "";
  std::cout << "  placed classVariables num = " << inputCodeInfo.dataMembers.size() << std::endl;
  {
    std::string rawname = GetFolderPath(inputCodeInfo.mainClassFileName);
    uboOutName          = rawname + "/include/" + uboIncludeName;
  }

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
    kernel.warpSize        = warpSize;
    kernel.enableSubGroups = enableSubGroupOps;
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
  }

  std::cout << "(7) Perform final templated text rendering to generate Vulkan calls" << std::endl; 
  std::cout << "{" << std::endl;
  {
    std::string rawname = kslicer::CutOffFileExt(allFiles[0]);
    auto json = PrepareJsonForAllCPP(inputCodeInfo, compiler, inputCodeInfo.mainFunc, generalDecls, rawname + "_generated.h", threadsOrder, uboIncludeName, jsonUBO); 

    kslicer::ApplyJsonToTemplate("templates/vk_class.h",        rawname + "_generated.h", json); 
    kslicer::ApplyJsonToTemplate("templates/vk_class.cpp",      rawname + "_generated.cpp", json);
    kslicer::ApplyJsonToTemplate("templates/vk_class_ds.cpp",   rawname + "_generated_ds.cpp", json);
    kslicer::ApplyJsonToTemplate("templates/vk_class_init.cpp", rawname + "_generated_init.cpp", json);    
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
    }
  }
  else
  {
    for(auto& k : inputCodeInfo.kernels)
      k.second.rewrittenText = inputCodeInfo.VisitAndRewrite_KF(k.second, compiler, k.second.rewrittenInit, k.second.rewrittenFinish);
  }
  
  auto json = kslicer::PrepareJsonForKernels(inputCodeInfo, usedByKernelsFunctions, generalDecls, compiler, threadsOrder, uboIncludeName, jsonUBO);
  inputCodeInfo.pShaderCC->GenerateShaders(json, &inputCodeInfo);

  //std::ofstream file(GetFolderPath(inputCodeInfo.mainClassFileName) + "/z_ubo.json");
  //file << std::setw(2) << json; //
  //file.close();

  {
    std::string rawname        = GetFolderPath(inputCodeInfo.mainClassFileName);
    std::string outName        = rawname + "/include/" + uboIncludeName;
    kslicer::ApplyJsonToTemplate("templates/ubo_def.h",  outName, jsonUBO);
  }

  std::cout << "}" << std::endl;
  std::cout << std::endl;
  
  kslicer::ApplyJsonToTemplate("templates/ubo_def.h",  uboOutName, jsonUBO); // need to call it after "GenerateShaders"

  return 0;
}