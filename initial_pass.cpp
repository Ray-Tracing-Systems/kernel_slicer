#include "initial_pass.h"
#include <iostream>

void kslicer::SplitContainerTypes(const clang::ClassTemplateSpecializationDecl* specDecl, std::string& a_containerType, std::string& a_containerDataType)
{
  a_containerType = specDecl->getNameAsString();      
  const auto& templateArgs = specDecl->getTemplateArgs();
        
  if(templateArgs.size() > 0)
    a_containerDataType = templateArgs[0].getAsType().getAsString();
  else
    a_containerDataType = "unknown";
}


kslicer::KernelInfo::ArgInfo kslicer::ProcessParameter(const clang::ParmVarDecl *p) 
{
  clang::QualType q = p->getType();

  kslicer::KernelInfo::ArgInfo arg;
  arg.name = p->getNameAsString();
  arg.type = clang::QualType::getAsString(q.split(), clang::PrintingPolicy{ {} });
  arg.size = 1;
  if (q->isPointerType()) 
  {
    arg.size      = 1; // Because C always pass reference
    arg.kind      = kslicer::DATA_KIND::KIND_POINTER;
  }
  else if(q->isReferenceType()) 
  {
    arg.isReference = true;
    auto dataType = q.getNonReferenceType(); 
    auto typeDecl = dataType->getAsRecordDecl();
      
    if(typeDecl != nullptr && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl))
    {
      arg.isContainer = true;
      auto specDecl = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(typeDecl);   
      kslicer::SplitContainerTypes(specDecl, arg.containerType, arg.containerDataType);

      if(arg.containerType == "Texture2D" || arg.containerType == "Image2D")
        arg.kind = kslicer::DATA_KIND::KIND_TEXTURE;
      else if(arg.containerType == "vector" || arg.containerType == "std::vector")
        arg.kind = kslicer::DATA_KIND::KIND_VECTOR;
      else if(arg.containerType == "unordered_map" || arg.containerType == "std::unordered_map")
        arg.kind = kslicer::DATA_KIND::KIND_HASH_TABLE;
      else if((arg.containerType == "shared_ptr" || arg.containerType == "std::shared_ptr") && arg.containerDataType == "ISceneObject")
        arg.kind = kslicer::DATA_KIND::KIND_ACCEL_STRUCT;
    }
  }

  return arg;
}

void kslicer::InitialPassRecursiveASTVisitor::ProcessKernelDef(const CXXMethodDecl *f, std::unordered_map<std::string, KernelInfo>& a_funcList, const std::string& a_className) 
{
  if (!f || !f->hasBody()) 
    return;

  const QualType retType  = f->getReturnType();
  std::string retTypeName = retType.getAsString();

  DeclarationNameInfo dni = f->getNameInfo();
  DeclarationName dn      = dni.getName();
  
  KernelInfo info;
  info.name        = dn.getAsString();
  info.className   = a_className;
  info.astNode     = f;
  info.return_type = retType.getAsString();
  info.isBoolTyped = retType.isTrivialType(m_astContext) && (retTypeName == "bool" || retTypeName == "_Bool");
  
  if(retType->isPointerType())
  {
    auto qtOfClass    = retType->getPointeeType(); 
    info.return_class = qtOfClass.getAsString();
  }

  for (unsigned int i = 0; i < f->getNumParams(); ++i) {
    info.args.push_back(kslicer::ProcessParameter(f->parameters()[i]));
  }

  if(a_className == MAIN_CLASS_NAME)
    a_funcList[info.name] = info;
  else
    a_funcList[a_className + "::" + info.name] = info;
}


bool kslicer::InitialPassRecursiveASTVisitor::VisitCXXRecordDecl(CXXRecordDecl* record)
{
  if(!record->hasDefinition())
    return true;

  const auto pType = record->getTypeForDecl(); 
  if(pType == nullptr)
    return true;

  const QualType qt   = pType->getLocallyUnqualifiedSingleStepDesugaredType();
  const auto typeName = qt.getAsString();

  if(typeName == std::string("class ") + MAIN_CLASS_NAME || typeName == std::string("struct ") + MAIN_CLASS_NAME)
    m_mainClassASTNode = record;
  else if(!record->isPOD())
    m_classList.push_back(record); // rememer for futher processing of complex classes
  
  return true;
}

bool kslicer::InitialPassRecursiveASTVisitor::NeedToProcessDeclInFile(std::string a_fileName)
{
  bool needInsertToKernels = false;                     // do we have to process this declaration to further insert it to GLSL/CL ?
  for(auto folder : m_codeInfo.includeCPPFolders)       //
  {
    if(a_fileName.find(folder) != std::string::npos)
    {
      needInsertToKernels = true;
      break;
    }
  }
  return needInsertToKernels;
}

bool kslicer::InitialPassRecursiveASTVisitor::VisitTypeDecl(TypeDecl* type)
{
  const FileEntry* Entry = m_sourceManager.getFileEntryForID(m_sourceManager.getFileID(type->getLocation()));
  std::string FileName   = Entry->getName().str();
  if(!NeedToProcessDeclInFile(FileName))
    return true;

  if(isa<CXXRecordDecl>(type)) 
  {
    // currently we don't put polimorphic C++ classes to shaders, in far future we need to process them in special way probably
    //
    CXXRecordDecl* pCXXDecl = dyn_cast<CXXRecordDecl>(type);
    //if(!pCXXDecl->isCLike())
    //  return true;
    if(pCXXDecl->isPolymorphic() || pCXXDecl->isAbstract())
      return true;   
  }

  //const clang::QualType qt = 

  kslicer::DeclInClass decl;
  if(isa<RecordDecl>(type))
  {
    RecordDecl* pRecord = dyn_cast<RecordDecl>(type);
    decl.name      = pRecord->getNameAsString();
    decl.type      = pRecord->getNameAsString();
    decl.srcRange  = pRecord->getSourceRange ();                    
    decl.srcHash   = kslicer::GetHashOfSourceRange(decl.srcRange);  
    decl.order     = m_currId;
    decl.kind      = kslicer::DECL_IN_CLASS::DECL_STRUCT;
    decl.extracted = true;
    
    if(decl.name != m_codeInfo.mainClassName && 
       decl.name != std::string("class ") + m_codeInfo.mainClassName && 
       decl.name != std::string("struct ") + m_codeInfo.mainClassName)
    {
      m_transferredDecl[decl.name] = decl;
      m_currId++;
    }
  }
  else if(isa<TypedefDecl>(type))
  {
    TypedefDecl* pTargetTpdf = dyn_cast<TypedefDecl>(type);
    const auto qt  = pTargetTpdf->getUnderlyingType();
    decl.name      = pTargetTpdf->getNameAsString();
    decl.type      = qt.getAsString();
    decl.srcRange  = pTargetTpdf->getSourceRange();                
    decl.srcHash   = kslicer::GetHashOfSourceRange(decl.srcRange);
    decl.order     = m_currId;
    decl.kind      = kslicer::DECL_IN_CLASS::DECL_TYPEDEF;
    decl.extracted = true;
    m_transferredDecl[decl.name] = decl;
    m_currId++;
  }
  else if(isa<EnumDecl>(type))
  {
    EnumDecl* pEnumDecl = dyn_cast<EnumDecl>(type);

    for(auto it = pEnumDecl->enumerator_begin(); it != pEnumDecl->enumerator_end(); ++it)
    {
      EnumConstantDecl* pConstntDecl = (*it);
      decl.name      = pConstntDecl->getNameAsString();
      decl.type      = "const uint"; 
      decl.srcRange  = pConstntDecl->getInitExpr()->getSourceRange();                    
      decl.srcHash   = kslicer::GetHashOfSourceRange(decl.srcRange);  
      decl.order     = m_currId;
      decl.kind      = kslicer::DECL_IN_CLASS::DECL_CONSTANT;
      decl.extracted = true;
      m_transferredDecl[decl.name] = decl;
      m_currId++;
    }
 
  }

  //std::string text = GetRangeSourceCode(type->getSourceRange(), m_compiler);     
  return true;
}

bool kslicer::InitialPassRecursiveASTVisitor::VisitVarDecl(VarDecl* pTargetVar)
{
  const FileEntry* Entry = m_sourceManager.getFileEntryForID(m_sourceManager.getFileID(pTargetVar->getLocation()));
  std::string FileName   = Entry->getName().str();
  if(!NeedToProcessDeclInFile(FileName))
    return true;

  const clang::QualType qt = pTargetVar->getType();

  kslicer::DeclInClass decl;
  if(pTargetVar->isConstexpr())
  {
    decl.name      = pTargetVar->getNameAsString();
    decl.type      = qt.getAsString(); 
    auto posOfDD = decl.type.find("::");
    if(posOfDD != std::string::npos)
      decl.type = decl.type.substr(posOfDD+2);

    decl.srcRange  = pTargetVar->getInit()->getSourceRange();                    
    decl.srcHash   = kslicer::GetHashOfSourceRange(decl.srcRange);  
    decl.order     = m_currId;
    decl.kind      = kslicer::DECL_IN_CLASS::DECL_CONSTANT;
    decl.extracted = true;

    if(qt->isConstantArrayType())
    {
      auto arrayType = dyn_cast<ConstantArrayType>(qt.getTypePtr()); 
      assert(arrayType != nullptr);
      QualType qtOfElem = arrayType->getElementType(); 
      decl.isArray   = true;
      decl.arraySize = arrayType->getSize().getLimitedValue();      
      decl.type      = qtOfElem.getAsString();
      //auto typeInfo2 = m_astContext.getTypeInfo(qtOfElem);
      //varInfo.sizeInBytesOfArrayElement = typeInfo2.Width / 8;
    }

    m_transferredDecl[decl.name] = decl;
    m_currId++;
  }

  return true;
}

std::vector<kslicer::DeclInClass> kslicer::InitialPassRecursiveASTVisitor::GetExtractedDecls()
{
  std::vector<kslicer::DeclInClass> generalDecls; 
  generalDecls.reserve(m_transferredDecl.size());
  for(const auto decl : m_transferredDecl)
    generalDecls.push_back(decl.second);
  std::sort(generalDecls.begin(), generalDecls.end(), [](const auto& a, const auto& b) { return a.order < b.order; } );
  return generalDecls;
}

kslicer::CPP11_ATTR kslicer::GetMethodAttr(const clang::CXXMethodDecl* f, clang::CompilerInstance& a_compiler)
{
  if(!f->hasAttrs())
    return CPP11_ATTR::ATTR_UNKNOWN;

  auto attrs = f->getAttrs();
  for(const auto& attr : attrs)
  {
    const std::string text = kslicer::GetRangeSourceCode(attr->getRange(), a_compiler);
    if(text == "kslicer::setter")
      return CPP11_ATTR::ATTR_SETTER;
    if(text == "kslicer::kernel" || text == "kslicer::kernel1D" || text == "kslicer::kernel2D" || text == "kslicer::kernel3D")
      return CPP11_ATTR::ATTR_KERNEL;
  }
  return CPP11_ATTR::ATTR_UNKNOWN;
}

bool kslicer::InitialPassRecursiveASTVisitor::VisitCXXMethodDecl(CXXMethodDecl* f) 
{
  if(f->isStatic())
    return true;

  if (f->hasBody())
  {
    // Get name of function
    const DeclarationNameInfo dni = f->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();
    
    const QualType qThisType = f->getThisType();   
    const QualType classType = qThisType.getTypePtr()->getPointeeType();
    std::string thisTypeName = classType.getAsString();

    auto attr = kslicer::GetMethodAttr(f, m_compiler);

    if(m_codeInfo.IsKernel(fname)) // 
    {
      if(thisTypeName == std::string("class ") + MAIN_CLASS_NAME || thisTypeName == std::string("struct ") + MAIN_CLASS_NAME || thisTypeName == MAIN_CLASS_NAME)
      {
        ProcessKernelDef(f, functions, MAIN_CLASS_NAME); // MAIN_CLASS_NAME::f ==> functions
        std::cout << "  found member kernel " << MAIN_CLASS_NAME.c_str() << "::" << fname.c_str() << std::endl;
        if(ctors.size() == 0)
        {
          clang::CXXRecordDecl* pClasDecl =	f->getParent();
          for(auto ctor : pClasDecl->ctors()) 
          {
            if(!ctor->isCopyOrMoveConstructor())
              ctors.push_back(ctor);
          }
        }
      }
      else // extract other kernels and classes
      {
        thisTypeName = kslicer::CutOffStructClass(thisTypeName);
        ProcessKernelDef(f, otherFunctions, thisTypeName); // thisTypeName::f ==> otherFunctions
        std::cout << "  found other kernel " << thisTypeName.c_str() << "::" << fname.c_str() << std::endl;
      }
    }
    else if(m_mainFuncts.find(fname) != m_mainFuncts.end())
    {
      m_mainFuncNodes[fname] = f;
      //std::cout << "main function has found:\t" << fname.c_str() << std::endl;
      //f->dump();
    }
    else if(attr == CPP11_ATTR::ATTR_SETTER)
    {
      m_setters[fname] = f;
    }
    else
    {
      //std::cout << "  --> found member func " <<  thisTypeName.c_str() << "::" << fname.c_str() << std::endl;
    }
  }

  return true; // returning false aborts the traversal
}

kslicer::DataMemberInfo kslicer::ExtractMemberInfo(clang::FieldDecl* fd, const clang::ASTContext& astContext)
{
  const clang::QualType qt = fd->getType();

  kslicer::DataMemberInfo member;
  member.name        = fd->getName().str();
  member.type        = qt.getAsString();
  member.sizeInBytes = 0; 
  member.offsetInTargetBuffer = 0;

  // now we should check werther this field is std::vector<XXX> or just XXX; 
  //
  const clang::Type* fieldTypePtr = qt.getTypePtr(); 
  assert(fieldTypePtr != nullptr);
  if(fieldTypePtr->isPointerType()) // we ignore pointers due to we can't pass them to GPU correctly
  {
    member.isPointer = true;
    member.kind      = kslicer::DATA_KIND::KIND_POINTER;
    return member;
  }

  auto typeDecl = fieldTypePtr->getAsRecordDecl();  
  if(fieldTypePtr->isConstantArrayType())
  {
    auto arrayType = clang::dyn_cast<clang::ConstantArrayType>(fieldTypePtr); 
    assert(arrayType != nullptr);
    clang::QualType qtOfElem = arrayType->getElementType(); 
    member.containerDataType = qtOfElem.getAsString(); 
    member.arraySize         = arrayType->getSize().getLimitedValue();      
    auto typeInfo      = astContext.getTypeInfo(qt);
    member.sizeInBytes = typeInfo.Width / 8; 
    member.isArray     = true;
    member.kind        = kslicer::DATA_KIND::KIND_POD;
  }  
  else if (typeDecl != nullptr && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl)) 
  {
    member.isContainer = true;
    auto specDecl = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(typeDecl); 
    kslicer::SplitContainerTypes(specDecl, member.containerType, member.containerDataType);
    //std::cout << "  found container of type " << member.containerType.c_str() << ", which data type is " <<  member.containerDataType.c_str() << std::endl;
    member.kind = kslicer::DATA_KIND::KIND_VECTOR;  // #TODO: probably texture or acceleration structure, check and refactor this!
  }
  else
  {
    auto typeInfo      = astContext.getTypeInfo(qt);
    member.sizeInBytes = typeInfo.Width / 8; 
    member.kind        = kslicer::DATA_KIND::KIND_POD;
  }

  return member;
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

    DataMemberInfo member = ExtractMemberInfo(fd, m_astContext);
    if(member.isPointer) // we ignore pointers due to we can't pass them to GPU correctly
      return true;

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
