#include "kslicer.h"
#include <locale>

std::string kslicer::KernelInfo::ReductionAccess::GetOp() const
{
  switch(type)
  {
    case REDUCTION_TYPE::ADD_ONE:
    case REDUCTION_TYPE::ADD:
    case REDUCTION_TYPE::SUB:
    case REDUCTION_TYPE::SUB_ONE:
    {
      return "+=";
    }
    break;
    case REDUCTION_TYPE::MUL:
      return "*=";
    break;
    case REDUCTION_TYPE::FUNC:
      return funcName;
    break;

    default:
      return "+=";
    break;
  };

}

std::string kslicer::KernelInfo::ReductionAccess::GetInitialValue() const // best in nomination shitty code
{
  switch(type)
  {
    case REDUCTION_TYPE::ADD_ONE:
    case REDUCTION_TYPE::ADD:
    case REDUCTION_TYPE::SUB:
    case REDUCTION_TYPE::SUB_ONE:
    {
      if(dataType == "int2")   return "make_int2(0,0)";
      if(dataType == "uint2")  return "make_uint2(0,0)";
      if(dataType == "float2") return "make_float2(0,0)";
      
      if(dataType == "int3")   return "make_int3(0,0,0)";
      if(dataType == "uint3")  return "make_uint3(0,0,0)";
      if(dataType == "float3") return "make_float3(0,0,0)";

      if(dataType == "int4")   return "make_int4(0,0,0,0)";
      if(dataType == "uint4")  return "make_uint4(0,0,0,0)";
      if(dataType == "float4") return "make_float4(0,0,0,0)";
      return "0";
    }
    break;
    case REDUCTION_TYPE::MUL:
      if(dataType == "int2")   return "make_int2(1,1)";
      if(dataType == "uint2")  return "make_uint2(1,1)";
      if(dataType == "float2") return "make_float2(1,1)";
      
      if(dataType == "int3")   return "make_int3(1,1,1)";
      if(dataType == "uint3")  return "make_uint3(1,1,1)";
      if(dataType == "float3") return "make_float3(1,1,1)";

      if(dataType == "int4")   return "make_int4(1,1,1,1)";
      if(dataType == "uint4")  return "make_uint4(1,1,1,1)";
      if(dataType == "float4") return "make_float4(1,1,1,1)";
      return "1";
    break;
    case REDUCTION_TYPE::FUNC:
    {
      if(funcName == "min" || funcName == "std::min" || funcName == "fmin") return "MAXFLOAT";
      if(funcName == "max" || funcName == "std::max" || funcName == "fmax") return "-MAXFLOAT";
      return "0";
    }
    break;

    default:
    break;
  };

  return "0";
}

bool kslicer::KernelInfo::ReductionAccess::SupportAtomicLastStep() const
{
  if(type == REDUCTION_TYPE::MUL)
    return false;

  const char* supportedTypes[] = {"int", "uint", "unsigned int", "int32_t", "uint32_t",
                                  "int2", "uint2", "int3", "uint3", "int4", "uint4"}; 

  const size_t arraySize = sizeof(supportedTypes)/sizeof(supportedTypes[0]);
  for(size_t i=0; i<arraySize; i++)
  {
    if(dataType == supportedTypes[i])
      return true;
  }

  return false;
}

std::string kslicer::KernelInfo::ReductionAccess::GetAtomicImplCode() const
{
  std::string res = "atomic_unknown";  
  switch(type)
  {
    case REDUCTION_TYPE::ADD_ONE:
    case REDUCTION_TYPE::ADD:
    res = "atomic_add";
    break;

    case REDUCTION_TYPE::SUB:
    case REDUCTION_TYPE::SUB_ONE:
    res = "atomic_sub";
    break;

    case REDUCTION_TYPE::FUNC:
    {
      if(funcName == "min" || funcName == "std::min" || funcName == "fmin") res = "atom_min";
      if(funcName == "max" || funcName == "std::max" || funcName == "fmax") res = "atom_max";
    }
    break;

    default:
    break;
  };

  auto lastSymb  = dataType[dataType.size()-1];
  auto firstSimb = dataType[0]; // 'u' or 'i'
  if(isdigit(lastSymb))
  {
    res.push_back(lastSymb);  
    if(firstSimb == 'u')
      res.push_back(firstSimb);  
  }

  return res;
}
