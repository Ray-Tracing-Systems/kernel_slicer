#ifndef KSLICER_AST_MATCHERS
#define KSLICER_AST_MATCHERS

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "clang/Basic/SourceManager.h"
#include "clang/AST/ASTContext.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <sstream>
#include <iostream>

#include "kslicer.h"

namespace kslicer
{
  using namespace clang::tooling;
  using namespace llvm;
  using namespace clang::ast_matchers;

  clang::ast_matchers::StatementMatcher MakeMatch_LocalVarOfMethod(std::string const& funcName); 
  clang::ast_matchers::StatementMatcher MakeMatch_MethodCallFromMethod(std::string const& funcName); // from method with name 'funcName'
  clang::ast_matchers::StatementMatcher MakeMatch_MethodCallFromMethod();                            // from any method

  clang::ast_matchers::StatementMatcher MakeMatch_MemberVarOfMethod(std::string const& funcName);     
  clang::ast_matchers::StatementMatcher MakeMatch_FunctionCallFromFunction(std::string const& funcName);
  clang::ast_matchers::StatementMatcher MakeMatch_SingleForLoopInsideFunction(std::string const& funcName);
  clang::ast_matchers::StatementMatcher MakeMatch_IfInsideForLoopInsideFunction(std::string const& funcName);
  clang::ast_matchers::StatementMatcher MakeMatch_FunctionCallInsideForLoopInsideFunction(std::string const& funcName);
  clang::ast_matchers::StatementMatcher MakeMatch_IfReturnFromFunction(std::string const& funcName);

  clang::ast_matchers::StatementMatcher MakeMatch_ForLoopInsideFunction(std::string const& funcName);
  clang::ast_matchers::StatementMatcher MakeMatch_BeforeForLoopInsideFunction(std::string const& funcName);

  clang::ast_matchers::DeclarationMatcher MakeMatch_StructDeclInsideClass(std::string const& className);
  clang::ast_matchers::DeclarationMatcher MakeMatch_VarDeclInsideClass(std::string const& className);
  clang::ast_matchers::DeclarationMatcher MakeMatch_TypedefInsideClass(std::string const& className);

  std::unordered_map<std::string, CFNameInfo> ListAllMainRTFunctions(clang::tooling::ClangTool& Tool, 
                                                                           const std::string& a_mainClassName, 
                                                                           const clang::ASTContext& a_astContext,
                                                                           const MainClassInfo& a_codeInfo);

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

  class MainFuncAnalyzerRT : public clang::ast_matchers::MatchFinder::MatchCallback 
  {
  public:

    explicit MainFuncAnalyzerRT(std::ostream& s, kslicer::MainClassInfo& a_allInfo, const clang::ASTContext& a_astContext, kslicer::MainFuncInfo& a_mainFuncRef) : 
                                m_out(s), m_allInfo(a_allInfo), m_astContext(a_astContext), m_mainFuncRef(a_mainFuncRef) 
    {
      m_namesToIngore = kslicer::GetAllPredefinedThreadIdNamesRTV(); 
    }

    MainFuncInfo& CurrMainFunc() { return m_mainFuncRef; }

    void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
    {
      using namespace clang;
      
      // for kernel call inside MainFunc
      //
      const FunctionDecl      * func_decl = result.Nodes.getNodeAs<FunctionDecl>     ("targetFunction");
      const CXXMemberCallExpr * kern_call = result.Nodes.getNodeAs<CXXMemberCallExpr>("functionCall");
      const CXXMethodDecl     * kern      = result.Nodes.getNodeAs<CXXMethodDecl>    ("fdecl");
      
      // for local variable decl inside MainFunc
      //
      const Expr   * l_var = result.Nodes.getNodeAs<Expr>   ("localReference");
      const VarDecl* var   = result.Nodes.getNodeAs<VarDecl>("locVarName");
      const MemberExpr* l_var2 = result.Nodes.getNodeAs<MemberExpr>("memberReference");
      const FieldDecl * var2   = result.Nodes.getNodeAs<FieldDecl>("memberName");

      // for for expression inside MainFunc
      //
      const ForStmt* forLoop = result.Nodes.getNodeAs<ForStmt>("forLoop");
      const VarDecl* forIter = result.Nodes.getNodeAs<VarDecl>("loopIter"); 

      // for if statements inside for loop, which are goint to break loop
      //
      const IfStmt    * ifCond = result.Nodes.getNodeAs<IfStmt>("ifCond"); 
      const BreakStmt * brkExp = result.Nodes.getNodeAs<BreakStmt>("breakLoop");
      const ReturnStmt* extExp = result.Nodes.getNodeAs<ReturnStmt>("exitFunction"); 


      if(func_decl && kern_call && kern) // found kernel call in MainFunc
      {
        std::string kName = kern->getNameAsString();
        if(m_allInfo.IsKernel(kName))
        {
          auto pKernel = m_allInfo.allKernels.find(kName);  
          if(pKernel != m_allInfo.allKernels.end()) 
            pKernel->second.usedInMainFunc = true;  // mark this kernel is used
          
          const QualType retType = kern->getReturnType();
          const QualType thsType = kern->getThisType();
          
          if(retType->isPointerType())                                                ////  IMaterial* pMaterial = kernel_MakeMaterial(tid, &hit);
          {
            auto qtOfClass = retType->getPointeeType(); 
            m_allInfo.AddDispatchingHierarchy(qtOfClass.getAsString(), kName);
            if(pKernel != m_allInfo.allKernels.end()) 
              pKernel->second.isMaker = true;
          }
          else if(thsType->isPointerType() && pKernel == m_allInfo.allKernels.end())  ////  pMaterial->kernel_GetColor(tid, out_color);
          {
            auto qtOfClass = thsType->getPointeeType(); 
            m_allInfo.AddDispatchingKernel(qtOfClass.getAsString(), kName);
            
            std::string typeName = qtOfClass.getAsString();
            auto pos = typeName.find(" ");
            if(pos != std::string::npos)
              typeName = typeName.substr(pos+1);
            auto pKernel2 = m_allInfo.allOtherKernels.find(typeName + "::" + kName);
            if(pKernel2 != m_allInfo.allOtherKernels.end()) 
              pKernel2->second.isVirtual = true;
          }
          
          CurrMainFunc().UsedKernels.insert(kName); // add  this kernel to list of used kernels by MainFunc 
        }
      }
      else if(func_decl && l_var2 && var2)
      {
        const RecordDecl* parentClass = var2->getParent(); 
        if(parentClass != nullptr && parentClass->getNameAsString() == m_allInfo.mainClassName)
        {
          auto varName     = var2->getNameAsString();
          auto pDataMember = m_allInfo.allDataMembers.find(varName);
          auto pSetterMemb = m_allInfo.m_setterVars.find(varName);
          
          if(pDataMember != m_allInfo.allDataMembers.end())
          {
            pDataMember->second.usedInKernel = true;
            if(pDataMember->second.isContainer)
            {
              const clang::QualType qt = var2->getType();
              kslicer::UsedContainerInfo container;
              container.name    = pDataMember->first;
              container.type    = pDataMember->second.type;
              container.kind    = pDataMember->second.kind;
              container.isConst = qt.isConstQualified();
              CurrMainFunc().usedContainers[container.name] = container;
            }
            else
              CurrMainFunc().usedMembers.insert(pDataMember->first);
          }
          else if(pSetterMemb != m_allInfo.m_setterVars.end())
          {
            //std::cout << "[TODO]: implement setter access processing '" << varName.c_str() << "'" << std::endl;
          }

          if(pDataMember == m_allInfo.allDataMembers.end() && pSetterMemb == m_allInfo.m_setterVars.end())
          {
            std::cout << "[ERROR]: accessed member " << varName.c_str() << " was not found in allDataMembers (pointer member?)" << std::endl;
            return;
          }
        }
      }
      else if(func_decl && l_var && var) // found local variable in MainFunc
      {
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
        varInfo.isArray     = false;
        varInfo.isConst     = qt.isConstQualified();
        varInfo.arraySize   = 0;
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

        if(var->isLocalVarDecl() && !var->isConstexpr() && !qt.isConstQualified()) // list only local variables, exclude function arguments and all constants 
          CurrMainFunc().Locals[varInfo.name] = varInfo;                           // can also check isCXXForRangeDecl() and check for index ... in some way ...
        else if(var->isLocalVarDecl())
          CurrMainFunc().LocalConst[varInfo.name] = varInfo;
      }
      else if(func_decl && forLoop && forIter) // found for expression inside MainFunc
      {
        CurrMainFunc().ExcludeList.insert(forIter->getNameAsString());
      }
      else if(func_decl && kern_call && ifCond)
      {
        bool hasNegativeCondition = false;
        auto conBody = ifCond->getCond();
        if(isa<UnaryOperator>(conBody))
        {
          const auto bodyOp    = dyn_cast<UnaryOperator>(conBody);
          std::string opStr    = std::string( bodyOp->getOpcodeStr(bodyOp->getOpcode()) );
          hasNegativeCondition = (opStr == "!");
        }

        auto kernDecl = kern_call->getMethodDecl();
        auto pKernel  = m_allInfo.allKernels.find(kernDecl->getNameAsString());  
        if(pKernel != m_allInfo.allKernels.end() && (brkExp || extExp)) 
        {
          pKernel->second.usedInExitExpr = true; // mark this kernel is used in exit expression
          const uint64_t hashValue1 = kslicer::GetHashOfSourceRange(ifCond->getSourceRange());
          const uint64_t hashValue2 = kslicer::GetHashOfSourceRange(kern_call->getSourceRange());

          kslicer::KernelStatementInfo info;
          info.kernelName      = kernDecl->getNameAsString();
          info.kernelCallRange = kern_call->getSourceRange();
          info.ifExprRange     = ifCond->getSourceRange();
          if(brkExp)
            info.exprKind = kslicer::KernelStmtKind::EXIT_TYPE_LOOP_BREAK;
          else
            info.exprKind = kslicer::KernelStmtKind::EXIT_TYPE_FUNCTION_RETURN;
          info.isNegative = hasNegativeCondition;
          CurrMainFunc().ExitExprIfCond[hashValue1] = info; // ExitExprIfCond[ifCond]    = kern_call;
          CurrMainFunc().ExitExprIfCall[hashValue2] = info; // ExitExprIfCond[kern_call] = kern_call;
          if(forLoop)
          CurrMainFunc().CallsInsideFor[hashValue2] = info; // CallsInsideFor[kern_call] = kern_call;
        }
      }
      else if(func_decl && kern_call && forLoop) // same as previous , but without ifCond: just a kernell call inside for loop
      {
        auto kernDecl = kern_call->getMethodDecl();
        const uint64_t hashValue2 = kslicer::GetHashOfSourceRange(kern_call->getSourceRange());
       
        if(CurrMainFunc().CallsInsideFor.find(hashValue2) == CurrMainFunc().CallsInsideFor.end()) // see previous branch, if(forLoop) ...  at the end of branch
        {
          kslicer::KernelStatementInfo info;
          info.exprKind        = kslicer::KernelStmtKind::CALL_TYPE_SIMPLE;
          info.kernelName      = kernDecl->getNameAsString();
          info.kernelCallRange = kern_call->getSourceRange();
          CurrMainFunc().CallsInsideFor[hashValue2] = info; // CallsInsideFor[kern_call] = kern_call;
        }
      }

      return;
    }  // run
    
    std::ostream&             m_out;
    MainClassInfo&            m_allInfo;
    const clang::ASTContext&  m_astContext;

    std::vector<std::string>  m_namesToIngore;
    kslicer::MainFuncInfo&    m_mainFuncRef;

  };  // class MainFuncAnalyzer


  class UsedCodeFilter : public clang::ast_matchers::MatchFinder::MatchCallback 
  {
  public:

    explicit UsedCodeFilter(std::ostream& s, kslicer::MainClassInfo& a_allInfo,  kslicer::KernelInfo* a_currKernel, const clang::CompilerInstance& a_compiler) : 
                            m_out(s), m_allInfo(a_allInfo), currKernel(a_currKernel), 
                            m_compiler(a_compiler), m_sourceManager(a_compiler.getSourceManager()), m_astContext(a_compiler.getASTContext())
    {
    
    }

    void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
    {
      using namespace clang;
     
      FunctionDecl      const * func_decl = result.Nodes.getNodeAs<FunctionDecl>("targetFunction");
      MemberExpr        const * l_var     = result.Nodes.getNodeAs<MemberExpr>("memberReference");
      FieldDecl         const * var       = result.Nodes.getNodeAs<FieldDecl>("memberName");

      CallExpr const * funcCall = result.Nodes.getNodeAs<CallExpr>("functionCall");
      FunctionDecl const * func = result.Nodes.getNodeAs<FunctionDecl>("fdecl");

      //clang::SourceManager& srcMgr(const_cast<clang::SourceManager &>(result.Context->getSourceManager()));

      if(func_decl && l_var && var)
      {
        const RecordDecl* parentClass = var->getParent(); 
        if(parentClass != nullptr && parentClass->getNameAsString() == m_allInfo.mainClassName)
        {
          auto varName     = var->getNameAsString();
          auto pDataMember = m_allInfo.allDataMembers.find(varName);
          auto pSetterMemb = m_allInfo.m_setterVars.find(varName);

          if(pDataMember != m_allInfo.allDataMembers.end())
          {
            assert(pDataMember != m_allInfo.allDataMembers.end());
            assert(currKernel  != nullptr);
  
            pDataMember->second.usedInKernel = true;
            if(pDataMember->second.isContainer)
            {
              const clang::QualType qt = var->getType();
              kslicer::UsedContainerInfo container;
              container.name    = pDataMember->first; 
              container.type    = pDataMember->second.type;           
              container.kind    = pDataMember->second.kind;
              container.isConst = qt.isConstQualified();
              currKernel->usedContainers[container.name] = container;
            }
            else
              currKernel->usedMembers.insert(pDataMember->first);
          }
          else if(pSetterMemb != m_allInfo.m_setterVars.end())
          {
            //std::cout << "[TODO]: implement setter access processing '" << pSetterMemb->first.c_str() << "' for " << currKernel->name.c_str() << std::endl; 
            //l_var->dump();
            //int a = 2;
          }

          if(pDataMember == m_allInfo.allDataMembers.end() && pSetterMemb == m_allInfo.m_setterVars.end())
          {
            std::cout << "[ERROR]: accessed member " << varName.c_str() << " was not found in allDataMembers (pointer member?)" << std::endl;
            return;
          }
        }
      }
      else if(func_decl && funcCall && func)
      {
        //
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
    kslicer::KernelInfo* currKernel = nullptr;

    const clang::CompilerInstance& m_compiler;
    const clang::SourceManager&    m_sourceManager;
    const clang::ASTContext&       m_astContext;

    std::string    m_mainClassName;
  };  // class UsedCodeFilter


  class TC_Extractor : public clang::ast_matchers::MatchFinder::MatchCallback 
  {
  public:

    explicit TC_Extractor(const clang::CompilerInstance& a_compiler) : 
                          m_compiler(a_compiler), m_sourceManager(a_compiler.getSourceManager()), m_astContext(a_compiler.getASTContext())
    {
      m_currId = 0;
    }

    void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
    {
      using namespace clang;
     
      const CXXRecordDecl* const pMainClass    = result.Nodes.getNodeAs<CXXRecordDecl>("mainClass");
      const CXXRecordDecl* const pTargetStruct = result.Nodes.getNodeAs<CXXRecordDecl>("targetStruct");
      const VarDecl*       const pTargetVar    = result.Nodes.getNodeAs<VarDecl>      ("targetVar");
      const TypedefDecl*   const pTargetTpdf   = result.Nodes.getNodeAs<TypedefDecl>  ("targetTypedef");

      if(pMainClass && pTargetStruct)
      {
        if(pTargetStruct->getNameAsString() != pMainClass->getNameAsString() && !pTargetStruct->isImplicit())
        {
          auto pDef = pTargetStruct;
          std::cout << "  [TC_Extractor]: found " << pDef->getNameAsString() << " inside " << pMainClass->getNameAsString() << std::endl;
          kslicer::DeclInClass decl;
          decl.name     = pDef->getNameAsString();
          decl.type     = pDef->getNameAsString();
          decl.srcRange = pDef->getSourceRange ();                       // (!!!) DON'T WORK (!!!)
          decl.srcHash  = kslicer::GetHashOfSourceRange(decl.srcRange);  // (!!!) DON'T WORK (!!!)
          decl.order    = m_currId;
          decl.kind     = kslicer::DECL_IN_CLASS::DECL_STRUCT;
          decl.inClass  = true;
          if(foundDecl.find(decl.name) == foundDecl.end())
          {
            foundDecl[decl.name] = decl;
            m_currId++;
          }
        }

        //bool 	isOrContainsUnion() 
      }
      else if(pMainClass && pTargetVar)
      {
        if(!pTargetVar->isImplicit() && pTargetVar->isConstexpr())
        {
          auto pDef = pTargetVar;
          const clang::QualType qt = pDef->getType();
          const auto typePtr = qt.getTypePtr(); 

          if(!typePtr->isPointerType())
          {
            kslicer::DeclInClass decl;
            decl.name     = pDef->getNameAsString();
            decl.type     = qt.getAsString();
            decl.srcRange = pDef->getSourceRange();                       // (!!!) DON'T WORK (!!!) // NEED SECOND PASS !!!
            decl.srcHash  = kslicer::GetHashOfSourceRange(decl.srcRange); // (!!!) DON'T WORK (!!!) // NEED SECOND PASS !!!
            decl.order    = m_currId;
            decl.kind     = kslicer::DECL_IN_CLASS::DECL_CONSTANT;
            
            if(typePtr->isConstantArrayType())
            {
              auto arrayType = dyn_cast<ConstantArrayType>(typePtr); 
              assert(arrayType != nullptr);
              QualType qtOfElem = arrayType->getElementType(); 
              decl.isArray   = true;
              decl.arraySize = arrayType->getSize().getLimitedValue();      
              decl.type      = qtOfElem.getAsString();
              //auto typeInfo2 = m_astContext.getTypeInfo(qtOfElem);
              //varInfo.sizeInBytesOfArrayElement = typeInfo2.Width / 8;
            }

            if(foundDecl.find(decl.name) == foundDecl.end())
            {
              foundDecl[decl.name] = decl;
              m_currId++;
            }
          }
        }
      }
      else if(pMainClass && pTargetTpdf)
      {
        const auto qt = pTargetTpdf->getUnderlyingType();
        
        kslicer::DeclInClass decl;
        decl.name     = pTargetTpdf->getNameAsString();
        decl.type     = qt.getAsString();
        decl.srcRange = pTargetTpdf->getSourceRange();                // (!!!) DON'T WORK (!!!) // NEED SECOND PASS !!!
        decl.srcHash  = kslicer::GetHashOfSourceRange(decl.srcRange); // (!!!) DON'T WORK (!!!) // NEED SECOND PASS !!!
        decl.order    = m_currId;
        decl.kind     = kslicer::DECL_IN_CLASS::DECL_TYPEDEF;
        decl.inClass  = true;
        if(foundDecl.find(decl.name) == foundDecl.end())
        {
          foundDecl[decl.name] = decl;
          m_currId++;
        }
      }
      else 
      {
        check_ptr(pMainClass,     "pMainClass",    "", std::cout);
        check_ptr(pTargetStruct,  "pTargetStruct", "", std::cout);
        check_ptr(pTargetVar,     "pTargetVar",    "", std::cout);
        check_ptr(pTargetTpdf,    "pTargetTpdf",   "", std::cout);
      }

      return;
    }  // run
    
    //MainClassInfo&                 m_allInfo;
    const clang::CompilerInstance& m_compiler;
    const clang::SourceManager&    m_sourceManager;
    const clang::ASTContext&       m_astContext;
    int m_currId = 0;

    std::unordered_map<std::string, kslicer::DeclInClass> foundDecl;

  };  // class UsedCodeFilter

}

#endif
