#ifndef KSLICER_AST_MATCHERS
#define KSLICER_AST_MATCHERS

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "clang/Basic/SourceManager.h"
#include "clang/AST/ASTContext.h"

#include <string>
#include <sstream>
#include <iostream>

#include <unordered_map>
#include <vector>
#include <algorithm>

#include "kslicer.h"

namespace kslicer
{
  using namespace clang::tooling;
  using namespace llvm;
  using namespace clang::ast_matchers;

  clang::ast_matchers::StatementMatcher MakeMatch_LocalVarOfMethod(std::string const& funcName); 
  clang::ast_matchers::StatementMatcher MakeMatch_MethodCallFromMethod(std::string const& funcName); 

  clang::ast_matchers::StatementMatcher MakeMatch_MemberVarOfMethod(std::string const& funcName);     
  clang::ast_matchers::StatementMatcher MakeMatch_FunctionCallFromFunction(std::string const& funcName);

  std::string locationAsString(clang::SourceLocation loc, clang::SourceManager const * const sm);
  std::string sourceRangeAsString(clang::SourceRange r, clang::SourceManager const * sm);

  /**\brief Complain if pointer is invalid.
  \param p: pointer
  \param name: name of thing checked
  \param tabs: indentation
  \return none */
  template <typename T>
  inline void
  check_ptr(T * p,
            const std::string& name,
            const std::string& tabs = "",
            std::ostream&         s = std::cout)
  {
    if(!p) { s << tabs << "Invalid pointer " << name << "\n"; }
  }

  class MainFuncAnalyzer : public clang::ast_matchers::MatchFinder::MatchCallback 
  {
  public:

    explicit MainFuncAnalyzer(std::ostream& s, kslicer::MainClassInfo& a_allInfo, const clang::ASTContext& a_astContext) : 
                              m_out(s), m_allInfo(a_allInfo), m_astContext(a_astContext) 
    {
      m_namesToIngore = kslicer::GetAllPredefinedThreadIdNames(); 
    }

    void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
    {
      using namespace clang;
     
      FunctionDecl      const * func_decl = result.Nodes.getNodeAs<FunctionDecl>     ("targetFunction");
      CXXMemberCallExpr const * kern_call = result.Nodes.getNodeAs<CXXMemberCallExpr>("functionCall");
      CXXMethodDecl     const * kern      = result.Nodes.getNodeAs<CXXMethodDecl>    ("fdecl");

      Expr              const * l_var     = result.Nodes.getNodeAs<Expr>   ("localReference");
      VarDecl           const * var       = result.Nodes.getNodeAs<VarDecl>("locVarName");

      clang::SourceManager& src_manager(const_cast<clang::SourceManager &>(result.Context->getSourceManager()));

      if(func_decl && kern_call && kern) 
      {
        m_out << "In function '" << func_decl->getNameAsString() << "' ";
        m_out << "method '" << kern->getNameAsString() << "' referred to at ";
        std::string sr(sourceRangeAsString(kern_call->getSourceRange(), &src_manager));
        m_out << sr;
        m_out << "\n";
        m_allInfo.allKernels[kern->getNameAsString()].usedInMainFunc = true; // mark this kernel is used
      }
      else if(func_decl && l_var && var)
      {
        m_out << "In function '" << func_decl->getNameAsString() << "' ";
        m_out << "variable '" << var->getNameAsString() << "' referred to at ";
        std::string sr(sourceRangeAsString(l_var->getSourceRange(), &src_manager));
        m_out << sr;
        m_out << "\n";
        
        const clang::QualType qt = var->getType();
        const auto typePtr = qt.getTypePtr(); 
        assert(typePtr != nullptr);
        
        auto elementId = std::find(m_namesToIngore.begin(), m_namesToIngore.end(), var->getNameAsString());

        if(typePtr->isPointerType() || elementId != m_namesToIngore.end()) // ignore pointers and variables with special names
          return;

        auto typeInfo = m_astContext.getTypeInfo(qt);

        DataLocalVarInfo varInfo;
        varInfo.name        = var->getNameAsString();
        varInfo.type        = qt.getAsString();
        varInfo.sizeInBytes = typeInfo.Width / 8;
        
        varInfo.isArray   = false;
        varInfo.arraySize = 0;
        varInfo.typeOfArrayElement = ""; 
        
        if(typePtr->isConstantArrayType())
        {
          auto arrayType = dyn_cast<ConstantArrayType>(typePtr); 
          assert(arrayType != nullptr);

          QualType qtOfElem = arrayType->getElementType(); 
          auto typeInfo2 = m_astContext.getTypeInfo(qtOfElem);

          varInfo.arraySize = arrayType->getSize().getLimitedValue();      
          varInfo.typeOfArrayElement = qtOfElem.getAsString();
          varInfo.sizeInBytesOfArrayElement = typeInfo2.Width / 8;
          varInfo.isArray = true;
        }

        m_allInfo.mainFuncLocals[varInfo.name] = varInfo;
      }
      else 
      {
        check_ptr(l_var,     "l_var", "", m_out);
        check_ptr(var,       "var",   "", m_out);

        check_ptr(func_decl, "func_decl", "", m_out);
        check_ptr(kern_call, "kern_call", "", m_out);
        check_ptr(kern,      "kern",      "", m_out);
      }

      return;
    }  // run
    
    std::ostream& m_out;
    MainClassInfo& m_allInfo;
    const clang::ASTContext&  m_astContext;

    std::vector<std::string> m_namesToIngore;

  };  // class MainFuncAnalyzer


  class VariableAndFunctionFilter : public clang::ast_matchers::MatchFinder::MatchCallback 
  {
  public:

    explicit VariableAndFunctionFilter(std::ostream& s, kslicer::MainClassInfo& a_allInfo, clang::SourceManager& a_sm) : 
                                       m_out(s), m_allInfo(a_allInfo), m_sourceManager(a_sm) { usedFunctions.clear(); }

    void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
    {
      using namespace clang;
     
      FunctionDecl      const * func_decl = result.Nodes.getNodeAs<FunctionDecl>("targetFunction");
      MemberExpr        const * l_var     = result.Nodes.getNodeAs<MemberExpr>("memberReference");
      FieldDecl         const * var       = result.Nodes.getNodeAs<FieldDecl>("memberName");

      CallExpr const * funcCall = result.Nodes.getNodeAs<CallExpr>("functionCall");
      FunctionDecl const * func = result.Nodes.getNodeAs<FunctionDecl>("fdecl");

      clang::SourceManager& srcMgr(const_cast<clang::SourceManager &>(result.Context->getSourceManager()));

      if(func_decl && l_var && var)
      {
        const RecordDecl* parentClass = var->getParent(); 
        if(parentClass != nullptr && parentClass->getNameAsString() == m_allInfo.mainClassName)
        {
          m_allInfo.allDataMembers[var->getNameAsString()].usedInKernel = true;
          //m_out << "In function '" << func_decl->getNameAsString() << "' ";
          //m_out << "variable '" << var->getNameAsString() << "'" << " of " <<  parentClass->getNameAsString() << std::endl;
        }
      }
      else if(func_decl && funcCall && func)
      {
        auto funcSourceRange = func->getSourceRange();
        auto fileName  = m_sourceManager.getFilename(funcSourceRange.getBegin());
        if(fileName == m_allInfo.mainClassFileName)
        {
          if(usedFunctions.find(func->getNameAsString()) == usedFunctions.end())
          {
            usedFunctions[func->getNameAsString()] = funcSourceRange;
            //m_out << "In function '" << func_decl->getNameAsString() << "' ";
            //m_out << "method '" << func->getNameAsString() << "' referred to at ";
            //std::string sr(sourceRangeAsString(funcCall->getSourceRange(), &srcMgr));
            //m_out << sr;
            //m_out << "\n";
          }
        }
        
        usedFiles[srcMgr.getFilename(func->getLocation()).str()] = true; // mark include files that used by functions; we need to put such includes in .cl file

        //std::string funcName        = func->getNameAsString();
        //std::string fileNameOfAFunc = srcMgr.getFilename(func->getLocation()).str();
        //std::cout << "[VariableAndFunctionFilter]: fileName = " << fileNameOfAFunc.c_str() << " of " << funcName.c_str() << std::endl;
      }
      else 
      {
        check_ptr(l_var,     "l_var", "", m_out);
        check_ptr(var,       "var",   "", m_out);

        check_ptr(func_decl, "func_decl", "", m_out);
        check_ptr(funcCall,  "kern_call", "", m_out);
        check_ptr(func,      "kern",      "", m_out);
      }

      return;
    }  // run
    
    std::ostream&  m_out;
    MainClassInfo& m_allInfo;
    std::string    m_mainClassName;
    clang::SourceManager& m_sourceManager;

    std::unordered_map<std::string, clang::SourceRange> usedFunctions;
    std::unordered_map<std::string, bool>               usedFiles;

    //std::vector<const clang::FunctionDecl*> GetUsedFunctions()
    //{
    //  std::vector<const clang::FunctionDecl*> res;
    //  for(auto fDecl : usedFunctions)
    //    res.push_back((const clang::FunctionDecl*)fDecl.second);
    //  return res;
    //}

  };  // class Global_Printer

}

#endif
