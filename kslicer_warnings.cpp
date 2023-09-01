#include "kslicer_warnings.h"

#include <clang/AST/ASTContext.h>

void CheckUnalignedStructInsideVector(const kslicer::MainClassInfo& a_classInfo)
{
  for(const auto& member : a_classInfo.dataMembers)
  {
    if(member.isContainer && (member.containerType == "vector" || member.containerType == "std::vector") && member.containerDataType.find("struct") != std::string::npos) // member.pTypeDeclIfRecord != nullptr
    {
      auto dataTypeName = member.containerDataType;
      auto dataTypeDecl = member.pContainerDataTypeDeclIfRecord;
      if(dataTypeDecl != nullptr && clang::isa<clang::RecordDecl>(dataTypeDecl))
      {
        const clang::ASTContext& context    = dataTypeDecl->getASTContext();
        const clang::RecordDecl* recordDecl = clang::dyn_cast<clang::RecordDecl>(dataTypeDecl);
        // Iterate over the fields of the record
        //
        uint64_t maxSize = 0;
        bool has8BytesMember = false;
        bool has16BytesMember = false;
        for (const clang::FieldDecl* fieldDecl : recordDecl->fields()) {
          // Get the size of each member  
          const clang::Type* fieldType = fieldDecl->getType().getTypePtr();
          const uint64_t fieldSize = context.getTypeSizeInChars(fieldType).getQuantity();
          //std::cout << "Member: " << fieldDecl->getNameAsString() << ", Size: " << fieldSize << " bytes" << std::endl;
          maxSize = std::max(maxSize, fieldSize);
          has8BytesMember  = has8BytesMember || (fieldSize == 8);
          has16BytesMember = has16BytesMember || (fieldSize == 16);
        }
        
        // Get the size of the entire record
        const uint64_t recordSize = context.getTypeSizeInChars(recordDecl->getTypeForDecl()).getQuantity();
        
        uint64_t problem = 0;
        {
          if(has8BytesMember && (recordSize%8 != 0))
            problem = 8;
          else if(has16BytesMember && (recordSize%16 != 0))
            problem = 16;  
        }
        
        if(problem != 0)
          std::cout << "WARNING A1: " << "vector '" << member.name.c_str() << "', contains '" << dataTypeName.c_str() << "', which must be aligned at leat at " << problem << " bytes manually by inserting dummy fields in record" << std::endl; 
      }
    }
  }
}

void kslicer::CheckForWarnings(const MainClassInfo& a_classInfo)
{
  CheckUnalignedStructInsideVector(a_classInfo);
}
