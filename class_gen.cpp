#include "class_gen.h"
#include "kslicer.h"

#include <sstream>
#include <algorithm>

bool kslicer::MainFuncASTVisitor::VisitCXXMethodDecl(CXXMethodDecl* f) 
{
  if (f->hasBody())
  {
    // Get name of function
    const DeclarationNameInfo dni = f->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();

    mainFuncCmdName = fname + "Cmd";
    m_rewriter.ReplaceText(dni.getSourceRange(), mainFuncCmdName);
  }

  return true; // returning false aborts the traversal
}

bool kslicer::MainFuncASTVisitor::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
{
  // Get name of function
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  auto p = fname.find("kernel_");
  if(p != std::string::npos)
  {
    std::string kernName = fname.substr(p + 7);

    // extract arguments to form correct descriptor set
    //
    auto args = ExtractArgumentsOfAKernelCall(f);
    for(auto& arg : args)
    {
      if(arg.argType != KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE)
        continue;

      auto p2 = m_argsOfMainFunc.find(arg.varName);
      if(p2 != m_argsOfMainFunc.end())
      {
        arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG;
      }
    }
    std::string callSign = MakeKernellCallSignature(args, m_mainFuncName);

    auto p2 = dsIdBySignature.find(callSign);
    if(p2 == dsIdBySignature.end())
    {
      dsIdBySignature[callSign] = m_kernCallTypes.size();
      p2 = dsIdBySignature.find(callSign);

      KernelCallInfo call;
      call.kernelName            = kernName;
      call.allDescriptorSetsInfo = args;
      m_kernCallTypes.push_back(call);
    }
    std::stringstream strOut;
    //strOut << "// call tag id = " << m_kernellCallTagId << "; argsNum = " << f->getNumArgs() << std::endl;
    
    strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ";
    strOut << kernName.c_str() << "Layout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;

    strOut << "  " << kernName.c_str() << "Cmd";
  
    m_rewriter.ReplaceText(f->getExprLoc(), strOut.str());
    m_kernellCallTagId++;
  }

  return true; 
}

std::vector<kslicer::ArgReferenceOnCall> kslicer::MainFuncASTVisitor::ExtractArgumentsOfAKernelCall(CXXMemberCallExpr* f)
{
  std::vector<kslicer::ArgReferenceOnCall> args; 
  args.reserve(20);

  auto predefinedNames = GetAllPredefinedThreadIdNames();
  
  for(size_t i=0;i<f->getNumArgs();i++)
  {
    const Expr* currArgExpr = f->getArgs()[i];
    assert(currArgExpr != nullptr);
    auto sourceRange = currArgExpr->getSourceRange();
    std::string text = GetRangeSourceCode(sourceRange, m_sm);
  
    ArgReferenceOnCall arg;    
    if(text[0] == '&')
    {
      arg.umpersanned = true;
      text = text.substr(1);
    }

    auto elementId = std::find(predefinedNames.begin(), predefinedNames.end(), text); // exclude predefined names from arguments
    if(elementId != predefinedNames.end())
      continue;

    arg.varName = text;
    args.push_back(arg); 
  }

  return args;
}

std::string kslicer::MakeKernellCallSignature(const std::vector<ArgReferenceOnCall>& a_args, const std::string& a_mainFuncName)
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

      default:
      strOut << "[U]";
      break;
    };

    strOut << arg.varName.c_str();
  }

  return strOut.str();
}


bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to) 
{
  size_t start_pos = str.find(from);
  if(start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

std::string kslicer::ProcessMainFunc(const CXXMethodDecl* a_node, clang::CompilerInstance& compiler, const std::string& a_mainClassName, const std::string& a_mainFuncName,
                                     std::string& a_outFuncDecl, std::vector<KernelCallInfo>& a_outDsInfo)
{
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  auto inOutParamList = kslicer::ListPointerParamsOfMainFunc(a_node);

  kslicer::MainFuncASTVisitor rv(rewrite2, compiler.getSourceManager(), a_mainFuncName, inOutParamList);
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  
  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  std::string sourceCode = rewrite2.getRewrittenText(clang::SourceRange(b,e));
  
  // (1) TestClass::MainFuncCmd --> TestClass_Generated::MainFuncCmd
  // 
  const std::string replaceFrom = a_mainClassName + "::" + rv.mainFuncCmdName;
  const std::string replaceTo   = a_mainClassName + "_Generated" + "::" + rv.mainFuncCmdName;

  assert(ReplaceFirst(sourceCode, replaceFrom, replaceTo));

  // (2) add input command Buffer as first argument
  //
  {
    size_t roundBracketPos = sourceCode.find("(");
    sourceCode = (sourceCode.substr(0, roundBracketPos) + "(VkCommandBuffer a_commandBuffer, " + sourceCode.substr(roundBracketPos+2)); 
  }

  // (3) set m_currCmdBuffer with input command bufer
  //
  {
    size_t bracePos = sourceCode.find("{");
    sourceCode = (sourceCode.substr(0, bracePos) + "{\n  m_currCmdBuffer = a_commandBuffer; \n\n" + sourceCode.substr(bracePos+2)); 
  }

  // (4) get function decl from full function code
  //
  std::string mainFuncDecl = sourceCode.substr(0, sourceCode.find(")")+1) + ";";
  assert(ReplaceFirst(mainFuncDecl, a_mainClassName + "_Generated" + "::", ""));

  a_outFuncDecl = "virtual " + mainFuncDecl;
  a_outDsInfo.swap(rv.m_kernCallTypes);
  return sourceCode;
}


std::unordered_map<std::string, kslicer::InOutVarInfo> kslicer::ListPointerParamsOfMainFunc(const CXXMethodDecl* a_node)
{
  std::unordered_map<std::string, InOutVarInfo> params;
  for(int i=0;i<a_node->getNumParams();i++)
  {
    const ParmVarDecl* currParam = a_node->getParamDecl(i);
    
    const clang::QualType qt = currParam->getType();
    const auto typePtr = qt.getTypePtr(); 
    assert(typePtr != nullptr);
    
    if(!typePtr->isPointerType())
      continue;
    
    InOutVarInfo var;
    var.name = currParam->getNameAsString();
    params[var.name] = var;
  }

  return params;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::KernelReplacerASTVisitor::VisitMemberExpr(MemberExpr* expr)
{
  ValueDecl* pValueDecl =	expr->getMemberDecl();
  
  if(!isa<FieldDecl>(pValueDecl))
    return true;

  FieldDecl* pFieldDecl  = dyn_cast<FieldDecl>(pValueDecl);
  assert(pFieldDecl != nullptr);
  RecordDecl* pRecodDecl = pFieldDecl->getParent();
  assert(pRecodDecl != nullptr);

  const std::string thisTypeName = pRecodDecl->getNameAsString();
  if(thisTypeName != m_mainClassName)
    return true;
 
  // (1) get type of variable itself because we need to cast pointer to this type
  //
  QualType qt = pFieldDecl->getTypeSourceInfo()->getType();
  std::string fieldType = qt.getAsString();
  kslicer::ReplaceOpenCLBuiltInTypes(fieldType);

  // (2) get variable offset in buffer by its name 
  //
  const std::string fieldName = pFieldDecl->getNameAsString(); 
  const auto p = m_variables.find(fieldName);
  if(p == m_variables.end())
    return true;

  // (3) put *(pointer+offset) instead of variable name
  //
  const std::string buffName = kslicer::GetProjPrefix() + "data"; 
  std::stringstream strOut;
  strOut << "*(  "; 
  strOut << "(__global const " << fieldType.c_str() << "*)" << "(" << buffName.c_str() << "+" << (p->second.offsetInTargetBuffer/sizeof(uint32_t)) << ")";
  strOut << "  )";
  
  m_rewriter.ReplaceText(expr->getExprLoc(), strOut.str());
  
  return true;
}

std::string kslicer::ProcessKernel(const CXXMethodDecl* a_node, clang::CompilerInstance& compiler, const kslicer::MainClassInfo& a_codeInfo)
{
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  kslicer::KernelReplacerASTVisitor rv(rewrite2, a_codeInfo.mainClassName, a_codeInfo.classVariables);
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  
  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  return rewrite2.getRewrittenText(clang::SourceRange(b,e));
}


std::vector<std::string> kslicer::ObtainKernelsDecl(const std::vector<kslicer::KernelInfo>& a_kernelsData, clang::SourceManager& sm, const std::string& a_mainClassName)
{
  std::vector<std::string> kernelsCallCmdDecl;
  for (const auto& k : a_kernelsData)  
  {
    assert(k.astNode != nullptr);
    auto sourceRange = k.astNode->getSourceRange();
    std::string kernelSourceCode = GetRangeSourceCode(sourceRange, sm);
    
    std::string kernelCmdDecl = kernelSourceCode.substr(0, kernelSourceCode.find(")")+1);
    assert(ReplaceFirst(kernelCmdDecl, a_mainClassName + "::", ""));
    assert(ReplaceFirst(kernelCmdDecl,"kernel_", ""));
    assert(ReplaceFirst(kernelCmdDecl,"(", "Cmd("));
    kernelsCallCmdDecl.push_back(kernelCmdDecl);
  }
  return kernelsCallCmdDecl;
}
