#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

bool kslicer::MainFunctionRewriter::VisitCXXMethodDecl(CXXMethodDecl* f) 
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

std::string kslicer::MainFunctionRewriter::MakeKernelCallCmdString(CXXMemberCallExpr* f)
{
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  const auto pKernelInfo = m_kernels.find(fname);
  assert(pKernelInfo != m_kernels.end());

  const std::string kernName    = m_pCodeInfo->RemoveKernelPrefix(fname);

  // extract arguments to form correct descriptor set
  // 
  //std::stringstream strOut1;
  //strOut1 << "_" << m_dsTagId;                                                    
  const auto args     = ExtractArgumentsOfAKernelCall(f);                                          
  const auto callSign = MakeKernellCallSignature(m_mainFuncName, args, pKernelInfo->second.usedVectors); // + strOut1.str();
  auto p2 = dsIdBySignature.find(callSign);
  if(p2 == dsIdBySignature.end())
  {
    dsIdBySignature[callSign] = allDescriptorSetsInfo.size();
    p2 = dsIdBySignature.find(callSign);
    KernelCallInfo call;
    call.kernelName         = kernName;
    call.originKernelName   = fname;
    call.callerName         = m_mainFuncName;
    call.descriptorSetsInfo = args;
    allDescriptorSetsInfo.push_back(call);
  }
  m_dsTagId++;

  std::string textOfCall = GetRangeSourceCode(f->getSourceRange(), m_compiler);
  std::string textOfArgs = textOfCall.substr( textOfCall.find("("));

  std::stringstream strOut;
  {
    // understand if we are inside the loop, or outside of it
    //
    auto pKernel = m_kernels.find(fname);
    assert(pKernel != m_kernels.end()); 

    auto callSourceRangeHash = kslicer::GetHashOfSourceRange(f->getSourceRange());
    auto p3 = m_mainFunc.CallsInsideFor.find(callSourceRangeHash);
    auto p4 = m_mainFunc.ExitExprIfCall.find(callSourceRangeHash);

    std::string flagsVariableName = "";
    if(p3 != m_mainFunc.CallsInsideFor.end())
    {
      flagsVariableName = "inForFlags";
      
      if(pKernel->second.isBoolTyped && p4 == m_mainFunc.ExitExprIfCall.end())
        flagsVariableName += "D";
      else if(p3->second.isNegative)
        flagsVariableName += "N";
    }
    else 
    {
      flagsVariableName = "outOfForFlags";
      if(pKernel->second.isBoolTyped && p4 == m_mainFunc.ExitExprIfCall.end())
        flagsVariableName += "D";
      else if(p4 != m_mainFunc.ExitExprIfCall.end() && p4->second.isNegative)
        flagsVariableName += "N";
    }
    
    // m_currThreadFlags
    strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ";
    strOut << kernName.c_str() << "Layout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
    if(m_pCodeInfo->NeedThreadFlags())
      strOut << "  m_currThreadFlags = " << flagsVariableName.c_str() << ";" << std::endl;
    strOut << "  " << kernName.c_str() << "Cmd" << textOfArgs.c_str();
  }
  
  return strOut.str();
}

std::string kslicer::MainFunctionRewriter::MakeServiceKernelCallCmdString(CallExpr* call)
{
  std::string kernName = "copyKernelFloat"; // extract from 'call' exact name of service function;
                                     // replace it with actual name we are going to used in generated HOST(!!!) code. 
                                     // for example it can be 'MyMemcpy' for 'memcpy' if in host code we have (MyMemcpyLayout, MyMemcpyPipeline, MyMemcpyDSLayout)
                                     // please note that you should init MyMemcpyLayout, MyMemcpyPipeline, MyMemcpyDSLayout yourself in the generated code!                                      
  
  auto memCpyArgs = ExtractArgumentsOfAKernelCall(call);

  std::vector<ArgReferenceOnCall> args(2); // TODO: extract corretc arguments from memcpy (CallExpr* call)
  {
    args[0].argType = memCpyArgs[0].argType;
    args[0].varName = memCpyArgs[0].varName;

    args[1].argType = memCpyArgs[1].argType;
    args[1].varName = memCpyArgs[1].varName;
  }

  const auto callSign = MakeKernellCallSignature(m_mainFuncName, args, std::unordered_set<std::string>()); // + strOut1.str();
  auto p2 = dsIdBySignature.find(callSign);
  if(p2 == dsIdBySignature.end())
  {
    dsIdBySignature[callSign] = allDescriptorSetsInfo.size();
    p2 = dsIdBySignature.find(callSign);
    KernelCallInfo call;
    call.kernelName         = kernName;
    call.originKernelName   = kernName;
    call.callerName         = m_mainFuncName;
    call.descriptorSetsInfo = args;
    allDescriptorSetsInfo.push_back(call);
  }
  m_dsTagId++;

  std::stringstream strOut;
  strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ";
  strOut << kernName.c_str() << "Layout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
  strOut << "  " << kernName.c_str() << "Cmd(" << memCpyArgs[2].varName << " / sizeof(float))";

  return strOut.str();
}



bool kslicer::MainFunctionRewriter::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
{
  // Get name of function
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  if(m_pCodeInfo->IsKernel(fname))
  {
    std::string callStr = MakeKernelCallCmdString(f);

    auto p2 = m_alreadyProcessedCalls.find(kslicer::GetHashOfSourceRange(f->getSourceRange()));
    if(p2 == m_alreadyProcessedCalls.end())
    {
      m_rewriter.ReplaceText(f->getSourceRange(), callStr); // getExprLoc
    }
  }

  return true; 
}

bool kslicer::MainFunctionRewriter::VisitCallExpr(CallExpr* call)
{
  if(isa<CXXMemberCallExpr>(call)) // because we process them in "VisitCXXMemberCallExpr"
    return true;

  const FunctionDecl* fDecl = call->getDirectCallee();  
  if(fDecl == nullptr)             // definitely can't process nullpointer 
    return true;

  // Get name of function
  //
  const DeclarationNameInfo dni = fDecl->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();
  
  if(fname == "memcpy")
  {
    std::string testStr = MakeServiceKernelCallCmdString(call);
    m_rewriter.ReplaceText(call->getSourceRange(), testStr);
  }

  return true;
}

bool kslicer::MainFunctionRewriter::VisitIfStmt(IfStmt* ifExpr)
{
  Expr* conBody = ifExpr->getCond();
  if(isa<UnaryOperator>(conBody)) // if(!kernel_XXX(...))
  {
    const auto bodyOp = dyn_cast<UnaryOperator>(conBody);
    conBody = bodyOp->getSubExpr();
  }

  if(isa<CXXMemberCallExpr>(conBody))
  {
    CXXMemberCallExpr* f = dyn_cast<CXXMemberCallExpr>(conBody); // extract kernel_XXX(...)
    
    // Get name of function
    const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();
  
    if(m_pCodeInfo->IsKernel(fname))
    {
      std::string callStr = MakeKernelCallCmdString(f);
      m_rewriter.ReplaceText(ifExpr->getSourceRange(), callStr);
      m_alreadyProcessedCalls.insert( kslicer::GetHashOfSourceRange(f->getSourceRange()) );
    }
  }

  return true;
}

std::vector<kslicer::ArgReferenceOnCall> kslicer::MainFunctionRewriter::ExtractArgumentsOfAKernelCall(CallExpr* f)
{
  std::vector<kslicer::ArgReferenceOnCall> args; 
  args.reserve(20);

  auto predefinedNames = GetAllPredefinedThreadIdNamesRTV();
  
  for(size_t i=0;i<f->getNumArgs();i++)
  {
    const Expr* currArgExpr = f->getArgs()[i];
    assert(currArgExpr != nullptr);
    auto sourceRange = currArgExpr->getSourceRange();
    std::string text = GetRangeSourceCode(sourceRange, m_compiler);
  
    ArgReferenceOnCall arg;    
    if(text[0] == '&')
    {
      arg.umpersanned = true;
      text = text.substr(1);

      auto pClassVar = m_allClassMembers.find(text);
      if(pClassVar != m_allClassMembers.end())
        arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_POD;
    }
    else if(text.find(".data()") != std::string::npos) 
    {
      std::string varName = text.substr(0, text.find(".data()"));
      auto pClassVar = m_allClassMembers.find(varName);
      if(pClassVar == m_allClassMembers.end())
        std::cout << "[KernelCallError]: vector<...> variable '" << varName.c_str() << "' was not found in class!" << std::endl; 
      else
        pClassVar->second.usedInMainFn = true;

      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_VECTOR;
    }

    auto elementId = std::find(predefinedNames.begin(), predefinedNames.end(), text); // exclude predefined names from arguments
    if(elementId != predefinedNames.end())
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_THREAD_ID;

    arg.varName = text;
    args.push_back(arg); 
  }

  for(auto& arg : args)  // in this loop we have to define argument (actual parameter) type
  {
    if(arg.argType != KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE)
      continue;
    
    auto p2 = m_argsOfMainFunc.find(arg.varName);
    if(p2 != m_argsOfMainFunc.end())
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG;

    auto p3 = m_mainFuncLocals.find(arg.varName);
    if(p3 != m_mainFuncLocals.end())
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL;
  }

  // for(auto& arg : args) // check we don't have unknown types of arguments // well, in fact now we can ) Use arguments
  // {
  //   if(arg.argType == KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE)
  //   {
  //     std::stringstream strOut;
  //     strOut << "  WARNING: expr '" << arg.varName.c_str() << "' was not classified"; 
  //     kslicer::PrintError(strOut.str(), f->getSourceRange(), m_sm);  
  //   }
  // }

  return args;
}

std::string kslicer::MakeKernellCallSignature(const std::string& a_mainFuncName, const std::vector<ArgReferenceOnCall>& a_args, const std::unordered_set<std::string>& a_usedVectors)
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

    strOut << arg.varName.c_str();
  }

  for(const auto& vecName : a_usedVectors)
    strOut << "[MV][" << vecName.c_str() << "]";

  return strOut.str();
}
