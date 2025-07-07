#include "kslicer.h"
#include "initial_pass.h"
#include "template_rendering.h"

#include <algorithm>
#include <cctype>
#include <string>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string kslicer::GetRangeSourceCode(const clang::SourceRange a_range, const clang::CompilerInstance& compiler)
{
  const clang::SourceManager& sm = compiler.getSourceManager();
  const clang::LangOptions& lopt = compiler.getLangOpts();

  clang::SourceLocation b(a_range.getBegin()), _e(a_range.getEnd());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, sm, lopt));
  if(e < b)
    return std::string("");
  else
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
  const uint32_t hash1 = a_range.getBegin().getRawEncoding();
  const uint32_t hash2 = a_range.getEnd().getRawEncoding();
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

void kslicer::ExtractTypeAndVarNameFromConstructor(clang::CXXConstructExpr* constructExpr, clang::ASTContext* astContext, std::string& varName, std::string& typeName) 
{
  // (0) очищаем строки
  //
  varName = ""; 
  typeName = "";
  
  // (1) Получаем имя типа
  //
  clang::CXXConstructorDecl* ctor = constructExpr->getConstructor();
  typeName = ctor->getNameInfo().getName().getAsString();
  
  // (2) Получаем имя переменной
  clang::DynTypedNodeList parents = astContext->getParents(*constructExpr);
  for (const clang::DynTypedNode& parent : parents) {
      if (const clang::VarDecl* varDecl = parent.get<clang::VarDecl>()) {
          varName = varDecl->getNameAsString();
          break;
      }
  }
}


const clang::Expr* kslicer::RemoveImplicitCast(const clang::Expr* nextNode)
{
  if(nextNode == nullptr)
    return nullptr;
  while(clang::isa<clang::ImplicitCastExpr>(nextNode))
  {
    auto cast = dyn_cast<const clang::ImplicitCastExpr>(nextNode);
    nextNode  = cast->getSubExpr();
  }
  return nextNode;
}

clang::Expr* kslicer::RemoveImplicitCast(clang::Expr* nextNode)
{
  if(nextNode == nullptr)
    return nullptr;
  while(clang::isa<clang::ImplicitCastExpr>(nextNode))
  {
    auto cast = dyn_cast<clang::ImplicitCastExpr>(nextNode);
    nextNode  = cast->getSubExpr();
  }
  return nextNode;
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

void MakeAbsolutePathRelativeTo(std::filesystem::path& a_filePath, const std::filesystem::path& a_folderPath)
{
  a_filePath = std::filesystem::relative(a_filePath, a_folderPath);
}

std::string ToLowerCase(std::string a_str)
{
  std::transform(a_str.begin(), a_str.end(), a_str.begin(), [](unsigned char c){ return std::tolower(c); });
  return a_str;
}

std::unordered_set<std::string> kslicer::GetAllServiceKernels()
{
  std::unordered_set<std::string> names;
  names.insert("copyKernelFloat");
  names.insert("matMulTranspose");
  return names;
}

std::string kslicer::ExtractSizeFromArgExpression(const std::string& a_str)
{
  auto posOfPlus = a_str.find("+");
  auto posOfEnd  = a_str.find(".end()");

  if(posOfPlus != std::string::npos)
  {
    std::string sizeExpr = a_str.substr(posOfPlus+1);
    ReplaceFirst(sizeExpr," ", "");
    return sizeExpr;
  }
  else if(posOfEnd != std::string::npos)
  {
    return a_str.substr(0, posOfEnd) + ".size()";
  }

  return a_str;
}

std::string kslicer::ClearNameFromBegin(const std::string& a_str)
{
  auto posOfBeg = a_str.find(".begin()");
  if(posOfBeg != std::string::npos)
    return a_str.substr(0, posOfBeg);

  return a_str;
}

std::string kslicer::FixLamdbaSourceCode(std::string a_str)
{
  while(ReplaceFirst(a_str, "uint32_t ", "uint "));
  while(ReplaceFirst(a_str, "unsigned int ", "uint "));

  while(ReplaceFirst(a_str, "uint2 ",    "uvec2 "));
  while(ReplaceFirst(a_str, "uint3 ",    "uvec3 "));
  while(ReplaceFirst(a_str, "uint4 ",    "uvec4 "));

  while(ReplaceFirst(a_str, "int2 ",     "ivec2 "));
  while(ReplaceFirst(a_str, "int3 ",     "ivec3 "));
  while(ReplaceFirst(a_str, "int4 ",     "ivec4 "));

  while(ReplaceFirst(a_str, "float2 ",   "vec2 "));
  while(ReplaceFirst(a_str, "float3 ",   "vec3 "));
  while(ReplaceFirst(a_str, "float4 ",   "vec4 "));

  return a_str;
}

std::string kslicer::SubstrBetween(const std::string& a_str, const std::string& first, const std::string& second)
{
  auto pos1 = a_str.find(first);
  auto pos2 = a_str.find(second);
  if(pos1 != std::string::npos && pos2 != std::string::npos)
    return a_str.substr(pos1+1, pos2 - pos1 - 1);
  return a_str;
}

bool kslicer::IsTexture(clang::QualType a_qt)
{
  if(a_qt->isReferenceType())
    a_qt = a_qt.getNonReferenceType();

  auto typeDecl = a_qt->getAsRecordDecl();
  if(typeDecl == nullptr || !clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl))
    return false;

  const std::string typeName = a_qt.getAsString();
  return (typeName.find("Texture") != std::string::npos || typeName.find("Image") != std::string::npos);
}

bool kslicer::IsAccelStruct(const std::string& a_typeName)
{
  if( (a_typeName == "struct ISceneObject")  || (a_typeName == "ISceneObject") || 
      (a_typeName == "struct ISceneObject2") || (a_typeName == "ISceneObject2") )
    return true;
  else if(a_typeName.find("ISceneObject_") != std::string::npos)  
    return true;
  else
    return false;
}

bool kslicer::IsVectorContainer(const std::string& a_typeName)
{
  return (a_typeName == "vector") || (a_typeName == "std::vector") || (a_typeName == "cvex::vector");
}

bool kslicer::IsPointerContainer(const std::string& a_typeName)
{
  return (a_typeName == "shared_ptr") || (a_typeName == "unique_ptr") ||
         (a_typeName == "std::shared_ptr") || (a_typeName == "std::unique_ptr");
}

std::string kslicer::MakeKernellCallSignature(const std::string& a_mainFuncName, const std::vector<ArgReferenceOnCall>& a_args, const std::unordered_map<std::string, UsedContainerInfo>& a_usedContainers)
{
  std::stringstream strOut;
  for(const auto& arg : a_args)
  {
    switch(arg.argType)
    {
      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL:
      strOut << "[L]";
      break;

      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG:
      strOut << "[A][" << a_mainFuncName.c_str() << "]" ;
      break;

      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_VECTOR:
      strOut << "[V]";
      break;

      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_POD:
      strOut << "[P]";
      break;

      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_THREAD_ID:
      strOut << "[T]";
      break;

      default:
      strOut << "[U]";
      break;
    };

    strOut << arg.name.c_str();
  }

  for(const auto& vecName : a_usedContainers)
    strOut << "[MV][" << vecName.second.name.c_str() << "]";

  return strOut.str();
}

std::vector<kslicer::NameFlagsPair> kslicer::ListAccessedTextures(const std::vector<kslicer::ArgReferenceOnCall>& args, const kslicer::KernelInfo& kernel)
{
  std::vector<kslicer::NameFlagsPair> accesedTextures;
  accesedTextures.reserve(16);
  for(uint32_t i=0;i<uint32_t(args.size());i++)
  {
    if(args[i].isTexture())
    {
      std::string argNameInKernel = kernel.args[i].name;
      auto pFlags = kernel.texAccessInArgs.find(argNameInKernel);
      kslicer::NameFlagsPair tex;
      tex.name  = args[i].name;
      if(pFlags != kernel.texAccessInArgs.end())
        tex.flags = pFlags->second;
      tex.isArg = true;
      tex.argId = i;
      accesedTextures.push_back(tex);
    }
  }
  for(const auto& container : kernel.usedContainers)
  {
    if(container.second.isTexture())
    {
      auto pFlags = kernel.texAccessInMemb.find(container.second.name);
      kslicer::NameFlagsPair tex;
      tex.name  = container.second.name;
      if(pFlags != kernel.texAccessInMemb.end())
        tex.flags = pFlags->second;
      else
        tex.flags = kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE;
      tex.isArg = false;
      accesedTextures.push_back(tex);
    }
  }
  return accesedTextures;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<std::string, std::string> ReadCommandLineParams(int argc, const char** argv, std::filesystem::path& fileName,
                                                                   std::vector<std::string>& allFiles,
                                                                   std::vector<std::string>& ignoreFiles,
                                                                   std::vector<std::string>& processFiles,
                                                                   std::vector<std::string>& cppIncludes)
{
  std::unordered_map<std::string, std::string> cmdLineParams;
  for(int i=0; i<argc; i++)
  {
    std::string key(argv[i]);

    const bool isDefine = key.size() > 1 && key.substr(0,2) == "-D";
    const bool isKey    = key.size() > 0 && key[0] == '-';

    if(key == "-ignore")
      ignoreFiles.push_back(argv[i+1]);
    else if (key == "-process")
      processFiles.push_back(argv[i+1]);
    else if (key == "-cpp_include")
      cppIncludes.push_back(argv[i+1]);
    else if(key != "-v" && !isDefine && isKey) // exclude special "-IfoldePath" form, exclude "-v"
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
    std::cout << "[main]: no input file is specified " << std::endl;
    exit(0);
  }
  else if(allFiles.size() == 1)
    fileName = allFiles[0];
  else
  {
    fileName = allFiles[0];

    // merge files to a single temporary file
    auto folderPath = fileName.parent_path();
    auto fileName2  = fileName.filename();
    fileName2.replace_extension("");
    fileName2.concat("_temp.cpp");
    auto fileNameT  = folderPath / fileName2;

    std::cout << "[kslicer]: merging input files to temporary file " << fileName2 << std::endl;
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

std::vector<const char*> ExcludeSlicerParams(int argc, const char** argv, const std::unordered_map<std::string,std::string>& params, const char* a_mainFileName,  const std::unordered_map<std::string,std::string>& defines)
{
  std::unordered_set<std::string> values;
  for(auto p : params)
    values.insert(p.second);

  bool foundDSlicer  = false;
  bool foundMainFile = false;

  static std::vector<std::string> g_data;
  g_data.reserve(128);

  std::vector<const char*> argsForClang; // exclude our input from cmdline parameters and pass the rest to clang
  argsForClang.reserve(argc);
  for(int i=1;i<argc;i++)
  {
    if(params.find(argv[i]) == params.end() && values.find(argv[i]) == values.end())
      argsForClang.push_back(argv[i]);

    if(std::string(argv[i]) == "-DKERNEL_SLICER")
      foundDSlicer = true;
    else if(std::string(argv[i]) == a_mainFileName)
      foundMainFile = true;
  }
  
  if(!foundMainFile)
  {
    argsForClang.insert(argsForClang.begin(), a_mainFileName);
  }

  if(!foundDSlicer)
    argsForClang.push_back("-DKERNEL_SLICER");
  
  for(auto def : defines)
  {
    std::string name = def.first;
    if(name.find("-D") != 0)
      name = std::string("-D") + name;
    
    g_data.push_back(name);
    argsForClang.push_back(g_data.back().c_str());
    if(def.second != "")
    {
      g_data.push_back(def.second);
      argsForClang.push_back(g_data.back().c_str());
    }
  }

  return argsForClang;
}

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


