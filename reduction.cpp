#include "kslicer.h"
#include <locale>

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);

std::string kslicer::KernelInfo::ReductionAccess::GetOp(std::shared_ptr<IShaderCompiler> pShaderCC) const
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
      return pShaderCC->ReplaceCallFromStdNamespace(funcName, dataType);
    break;

    default:
      return "+=";
    break;
  };

}

std::string kslicer::KernelInfo::ReductionAccess::GetOp2(std::shared_ptr<IShaderCompiler> pShaderCC) const
{
  switch(type)
  {
    case REDUCTION_TYPE::ADD_ONE:
    case REDUCTION_TYPE::ADD:
    case REDUCTION_TYPE::SUB:
    case REDUCTION_TYPE::SUB_ONE:
    {
      return "+";
    }
    break;
    case REDUCTION_TYPE::MUL:
      return "*";
    break;
    case REDUCTION_TYPE::FUNC:
      return pShaderCC->ReplaceCallFromStdNamespace(funcName, dataType);
    break;

    default:
      return "+";
    break;
  };
}

std::string kslicer::KernelInfo::ReductionAccess::GetInitialValue(bool isGLSL) const // best in nomination shitty code
{
  if(isGLSL)
  {
    switch(type)
    {
      case REDUCTION_TYPE::ADD_ONE:
      case REDUCTION_TYPE::ADD:
      case REDUCTION_TYPE::SUB:
      case REDUCTION_TYPE::SUB_ONE:
      {
        if(dataType == "ivec2")  return "ivec2(0,0)";
        if(dataType == "uvec2")  return "uvec2(0,0)";
        if(dataType == "vec2")   return "vec2(0,0)";
        
        if(dataType == "ivec3")  return "ivec3(0,0,0)";
        if(dataType == "uvec3")  return "uvec3(0,0,0)";
        if(dataType == "vec3")   return "vec3(0,0,0)";
  
        if(dataType == "int4")   return "ivec4(0,0,0,0)";
        if(dataType == "uint4")  return "uvec4(0,0,0,0)";
        if(dataType == "vec4")   return "vec4(0,0,0,0)";
        return "0";
      }
      break;
      case REDUCTION_TYPE::MUL:
        if(dataType == "ivec2")  return "ivec2(1,1)";
        if(dataType == "uvec2")  return "uvec2(1,1)";
        if(dataType == "vec2")   return "vec2(1,1)";
        
        if(dataType == "ivec3")  return "ivec3(1,1,1)";
        if(dataType == "uvec3")  return "uvec3(1,1,1)";
        if(dataType == "vec3")   return "vec3(1,1,1)";
  
        if(dataType == "int4")   return "ivec4(1,1,1,1)";
        if(dataType == "uint4")  return "uvec4(1,1,1,1)";
        if(dataType == "vec4")   return "vec4(1,1,1,1)";
        return "1";
      break;
      case REDUCTION_TYPE::FUNC:
      {
        if(funcName == "fmin") return "MAXFLOAT";
        if(funcName == "fmax") return "-MAXFLOAT";
        
        if(dataType == "float" || dataType == "vec4" || dataType == "vec3" || dataType == "vec2")
        {
          if(funcName == "min" || funcName == "std::min" || funcName == "fmin") return "MAXFLOAT";
          if(funcName == "max" || funcName == "std::max" || funcName == "fmax") return "-MAXFLOAT";
        }
        else if(dataType == "int" || dataType == "ivec4" || dataType == "ivec3" || dataType == "ivec2")
        {
          if(funcName == "min" || funcName == "std::min" || funcName == "fmin") return "2147483647";
          if(funcName == "max" || funcName == "std::max" || funcName == "fmax") return "-2147483648";
        }
        else if(dataType == "uint" || dataType == "uvec4" || dataType == "uvec3" || dataType == "uvec2")
        {
          if(funcName == "min" || funcName == "std::min" || funcName == "fmin") return "4294967295";
          if(funcName == "max" || funcName == "std::max" || funcName == "fmax") return "0";
        }
        
        return "0";
      }
      break;
  
      default:
      break;
    };
  
    return "0";
  }


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

size_t kslicer::KernelInfo::ReductionAccess::GetSizeOfDataType() const
{
  size_t numElements = 1;
  auto lastSymb = dataType[dataType.size()-1];
  if(lastSymb=='2')
    numElements = 2;
  else if(lastSymb=='3')
    numElements = 3;
  else if(lastSymb == '4')
    numElements = 4;

  return sizeof(uint32_t)*numElements;
}

bool kslicer::KernelInfo::ReductionAccess::SupportAtomicLastStep() const
{
  if(type == REDUCTION_TYPE::MUL)
    return false;

  if(type == REDUCTION_TYPE::FUNC)
  {
    if(funcName != "min" && funcName == "std::min" && funcName != "max" && funcName != "std::max")
      return false;
  }

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

std::string kslicer::KernelInfo::ReductionAccess::GetAtomicImplCode(bool isGLSL) const
{
  std::string res = "atomic_unknown"; 

  if(isGLSL)
  {
    switch(type)
    {
      case REDUCTION_TYPE::ADD_ONE:
      case REDUCTION_TYPE::ADD:
      res = "atomicAdd";
      break;
  
      case REDUCTION_TYPE::SUB:
      case REDUCTION_TYPE::SUB_ONE:
      res = "atomicSub";
      break;
  
      case REDUCTION_TYPE::FUNC:
      {
        if(funcName == "min" || funcName == "std::min") res = "atomicMin";
        if(funcName == "max" || funcName == "std::max") res = "atomicMax";
      }
      break;
  
      default:
      break;
    };
  }
  else
  {
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
        if(funcName == "min" || funcName == "std::min") res = "atomic_min";
        if(funcName == "max" || funcName == "std::max") res = "atomic_max";
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
  }

  return res;
}


std::string kslicer::KernelInfo::ReductionAccess::GetSubgroupOpCode(bool isGLSL) const
{
  std::string res = "atomic_unknown"; 

  if(!isGLSL)
    return res;

  switch(type)
  {
    case REDUCTION_TYPE::ADD_ONE:
    case REDUCTION_TYPE::ADD:
    res = "subgroupAdd";
    break;

    case REDUCTION_TYPE::SUB:
    case REDUCTION_TYPE::SUB_ONE:
    res = "subgroupAdd";
    break;

    case REDUCTION_TYPE::FUNC:
    {
      if(funcName == "min" || funcName == "std::min") res = "subgroupMin";
      if(funcName == "max" || funcName == "std::max") res = "subgroupMax";
    }
    break;

    case REDUCTION_TYPE::MUL:
    res = "subgroupMul";
    break;

    default:
    break;
  };

  return res;
}
