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
    std::cout << "[kslicer]: no input file is specified " << std::endl;
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

    std::cout << "[kslicer]: merging input files to temporary file '" << fileName2 << std::endl;
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


