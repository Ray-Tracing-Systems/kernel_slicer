#include "kslicer.h"
#include <locale>

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to);

std::string kslicer::KernelInfo::ReductionAccess::GetOp(std::shared_ptr<IShaderCompiler> pShaderCC) const
{
  if(pShaderCC->IsISPC())
  {
    switch(type)
    {
      case REDUCTION_TYPE::ADD_ONE:
      case REDUCTION_TYPE::ADD:
      case REDUCTION_TYPE::SUB:
      case REDUCTION_TYPE::SUB_ONE:
      {
        return "reduce_add";
      }
      break;
      case REDUCTION_TYPE::MUL:
        return "reduce_mul";
      break;
      case REDUCTION_TYPE::FUNC: 
        return std::string("reduce_") + pShaderCC->ReplaceCallFromStdNamespace(funcName, dataType);
      break;
      default:
        return "reduce_add";
      break;
    };
  }

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
  if(pShaderCC->IsISPC())
  {
    switch(type)
    {
      case REDUCTION_TYPE::ADD_ONE:
      case REDUCTION_TYPE::ADD:
      case REDUCTION_TYPE::SUB:
      case REDUCTION_TYPE::SUB_ONE:
      {
        return "add";
      }
      break;
      case REDUCTION_TYPE::MUL:
        return "mul";
      break;
      case REDUCTION_TYPE::FUNC: 
        return pShaderCC->ReplaceCallFromStdNamespace(funcName, dataType);
      break;
      default:
        return "add";
      break;
    };
  }

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

std::string kslicer::KernelInfo::ReductionAccess::GetInitialValue(bool isGLSL, const std::string& a_dataType) const // best in nomination shitty code
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
        if(a_dataType == "ivec2")  return "ivec2(0,0)";
        if(a_dataType == "uvec2")  return "uvec2(0,0)";
        if(a_dataType == "vec2")   return "vec2(0,0)";
        
        if(a_dataType == "ivec3")  return "ivec3(0,0,0)";
        if(a_dataType == "uvec3")  return "uvec3(0,0,0)";
        if(a_dataType == "vec3")   return "vec3(0,0,0)";
  
        if(a_dataType == "int4")   return "ivec4(0,0,0,0)";
        if(a_dataType == "uint4")  return "uvec4(0,0,0,0)";
        if(a_dataType == "vec4")   return "vec4(0,0,0,0)";
        return "0";
      }
      break;
      case REDUCTION_TYPE::MUL:
        if(a_dataType == "ivec2")  return "ivec2(1,1)";
        if(a_dataType == "uvec2")  return "uvec2(1,1)";
        if(a_dataType == "vec2")   return "vec2(1,1)";
        
        if(a_dataType == "ivec3")  return "ivec3(1,1,1)";
        if(a_dataType == "uvec3")  return "uvec3(1,1,1)";
        if(a_dataType == "vec3")   return "vec3(1,1,1)";
  
        if(a_dataType == "int4")   return "ivec4(1,1,1,1)";
        if(a_dataType == "uint4")  return "uvec4(1,1,1,1)";
        if(a_dataType == "vec4")   return "vec4(1,1,1,1)";
        return "1";
      break;
      case REDUCTION_TYPE::FUNC:
      {
        if(funcName == "fmin") return "FLT_MAX";
        if(funcName == "fmax") return "-FLT_MAX";
        
        if(a_dataType == "float" || a_dataType == "vec4" || a_dataType == "vec3" || a_dataType == "vec2")
        {
          if(funcName == "min" || funcName == "std::min" || funcName == "fmin") return a_dataType + "(FLT_MAX)";
          if(funcName == "max" || funcName == "std::max" || funcName == "fmax") return a_dataType + "(-FLT_MAX)";
        }
        else if(a_dataType == "int" || a_dataType == "ivec4" || a_dataType == "ivec3" || a_dataType == "ivec2")
        {
          if(funcName == "min" || funcName == "std::min" || funcName == "fmin") return a_dataType + "(2147483647)";
          if(funcName == "max" || funcName == "std::max" || funcName == "fmax") return a_dataType + "(-2147483648)";
        }
        else if(a_dataType == "uint" || a_dataType == "uvec4" || a_dataType == "uvec3" || a_dataType == "uvec2")
        {
          if(funcName == "min" || funcName == "std::min" || funcName == "fmin") return a_dataType + "(4294967295)";
          if(funcName == "max" || funcName == "std::max" || funcName == "fmax") return a_dataType + "(0)";
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
      if(funcName == "min" || funcName == "std::min" || funcName == "fmin") return "FLT_MAX";
      if(funcName == "max" || funcName == "std::max" || funcName == "fmax") return "-FLT_MAX";
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
