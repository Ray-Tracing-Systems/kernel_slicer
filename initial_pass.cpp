#include "initial_pass.h"
#include <iostream>

static kslicer::KernelInfo::Arg ProcessParameter(clang::ParmVarDecl *p) 
{
  clang::QualType q = p->getType();
  const clang::Type *typ = q.getTypePtr();
  kslicer::KernelInfo::Arg arg;
  arg.name = p->getNameAsString();
  arg.type = clang::QualType::getAsString(q.split(), clang::PrintingPolicy{ {} });
  arg.size = 1;
  if (typ->isPointerType()) {
    arg.size      = 1; // Because C always pass reference
    arg.isPointer = true;
  }
  
  return arg;
}

void kslicer::InitialPassRecursiveASTVisitor::ProcessKernelDef(const CXXMethodDecl *f) 
{
  if (!f || !f->hasBody()) 
    return;

  const QualType retType  = f->getReturnType();
  std::string retTypeName = retType.getAsString();

  DeclarationNameInfo dni = f->getNameInfo();
  DeclarationName dn      = dni.getName();
  
  KernelInfo info;
  info.name        = dn.getAsString();
  info.astNode     = f;
  info.return_type = retType.getAsString();
  info.isBoolTyped = retType.isTrivialType(m_astContext) && (retTypeName == "bool" || retTypeName == "_Bool");
  for (unsigned int i = 0; i < f->getNumParams(); ++i) {
    info.args.push_back(ProcessParameter(f->parameters()[i]));
  }
  functions[info.name] = info;
}


bool kslicer::InitialPassRecursiveASTVisitor::VisitCXXRecordDecl(CXXRecordDecl* record)
{
  if(!record->hasDefinition() || record->isImplicit() || record->isLiteral())
    return true;

  const auto pType  = record->getTypeForDecl(); 
  const QualType qt = pType->getLocallyUnqualifiedSingleStepDesugaredType();
  const std::string typeName = qt.getAsString();

  if(typeName == std::string("class ") + MAIN_CLASS_NAME || typeName == std::string("struct ") + MAIN_CLASS_NAME)
  {
    m_mainClassASTNode = record;
  }

  return true;
}

bool kslicer::InitialPassRecursiveASTVisitor::VisitCXXMethodDecl(CXXMethodDecl* f) 
{
  if (f->hasBody())
  {
    // Get name of function
    const DeclarationNameInfo dni = f->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();
    
    if(m_codeInfo.IsKernel(fname))
    {
      const QualType qThisType = f->getThisType();   
      const QualType classType = qThisType.getTypePtr()->getPointeeType();
      std::string thisTypeName = classType.getAsString();

      if(thisTypeName == std::string("class ") + MAIN_CLASS_NAME || thisTypeName == std::string("struct ") + MAIN_CLASS_NAME)
      {
        ProcessKernelDef(f);
        std::cout << "  found kernel:\t" << fname.c_str() << std::endl;
      }
    }
    else if(m_mainFuncts.find(fname) != m_mainFuncts.end())
    {
      m_mainFuncNodes[fname] = f;
      //std::cout << "main function has found:\t" << fname.c_str() << std::endl;
      //f->dump();
    }
  }

  return true; // returning false aborts the traversal
}

bool kslicer::InitialPassRecursiveASTVisitor::VisitFieldDecl(FieldDecl* fd)
{
  const clang::RecordDecl* rd = fd->getParent();
  const clang::QualType    qt = fd->getType();
 
  const std::string& thisTypeName = rd->getName().str();

  if(thisTypeName == MAIN_CLASS_NAME)
  {
    std::cout << "  found data member: " << fd->getName().str().c_str() << " of type\t" << qt.getAsString().c_str() << ", isPOD = " << qt.isCXX11PODType(m_astContext) << std::endl;

    auto funcSourceRange = rd->getSourceRange();
    auto fileName        = m_sourceManager.getFilename(funcSourceRange.getBegin());
    this->MAIN_FILE_INCLUDE = fileName;

    DataMemberInfo member;
    member.name        = fd->getName().str();
    member.type        = qt.getAsString();
    member.sizeInBytes = 0; 
    member.offsetInTargetBuffer = 0;

    // now we should check werther this field is std::vector<XXX> or just XXX; 
    //
    const Type* fieldTypePtr = qt.getTypePtr(); 
    assert(fieldTypePtr != nullptr);

    if(fieldTypePtr->isPointerType()) // we ignore pointers due to we can't pass them to GPU correctly
      return true;

    auto typeDecl = fieldTypePtr->getAsRecordDecl();  

    if(fieldTypePtr->isConstantArrayType())
    {
      auto arrayType = dyn_cast<ConstantArrayType>(fieldTypePtr); 
      assert(arrayType != nullptr);
      QualType 	qtOfElem       = arrayType->getElementType(); 
      member.containerDataType = qtOfElem.getAsString(); 
      member.arraySize         = arrayType->getSize().getLimitedValue();      
      auto typeInfo      = m_astContext.getTypeInfo(qt);
      member.sizeInBytes = typeInfo.Width / 8; 
      member.isArray     = true;
    }  
    else if (typeDecl != nullptr && isa<ClassTemplateSpecializationDecl>(typeDecl)) 
    {
      auto specDecl = dyn_cast<ClassTemplateSpecializationDecl>(typeDecl); 
      assert(specDecl != nullptr);
      
      member.isContainer   = true;
      member.containerType = specDecl->getNameAsString();      
      const auto& templateArgs = specDecl->getTemplateArgs();
      
      if(templateArgs.size() > 0)
        member.containerDataType = templateArgs[0].getAsType().getAsString();
      else
        member.containerDataType = "unknown";

      std::cout << "  found container of type " << member.containerType.c_str() << ", which data type is " <<  member.containerDataType.c_str() << std::endl;
    }
    else
    {
      auto typeInfo      = m_astContext.getTypeInfo(qt);
      member.sizeInBytes = typeInfo.Width / 8; 
    }

    dataMembers[member.name] = member;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::InitialPassASTConsumer::HandleTopLevelDecl(DeclGroupRef d)
{
  typedef DeclGroupRef::iterator iter;
  for (iter b = d.begin(), e = d.end(); b != e; ++b)
    rv.TraverseDecl(*b);
  return true; // keep going
}
