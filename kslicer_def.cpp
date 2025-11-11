#include "kslicer.h"
#include <array>

static std::array<std::string,7> POSSIBLE_KERNEL_NAMES = {"kernel_", "kernel1D_", "kernel2D_", "kernel3D_", "kernelBE1D_", "kernelBE2D_", "kernelBE3D_"};
static std::array<std::string,8> POSSIBLE_IMAGE_NAMES  = {"Texture1D", "Texture2D", "Texture3D", "TextureCube", "Image1D", "Image2D", "Image3D", "ImageCube"};

bool kslicer::MainClassInfo::IsKernel(const std::string& a_funcName) const
{
  for(const auto& name : POSSIBLE_KERNEL_NAMES)
    if(a_funcName.find(name) != std::string::npos)
      return true;
  return false;
}

std::string kslicer::MainClassInfo::RemoveKernelPrefix(const std::string& a_funcName) const
{
  std::string name = a_funcName;
  for(const auto& namePossible : POSSIBLE_KERNEL_NAMES)
    if(ReplaceFirst(name, namePossible, ""))
      return name;
  return a_funcName;
}

uint32_t kslicer::IPV_Pattern::GetKernelDim(const kslicer::KernelInfo& a_kernel) const
{
  const std::string& a_funcName = a_kernel.name;
  auto pos1 = a_funcName.find("1D_");
  auto pos2 = a_funcName.find("2D_");
  auto pos3 = a_funcName.find("3D_");


  if(pos1 != std::string::npos )
    return 1;
  else if(pos2 != std::string::npos) 
    return 2;
  else if(pos3 != std::string::npos)
    return 3;
  else
    return 0;
} 

bool kslicer::IsTextureContainer(const std::string& a_typeName)
{
  for(const auto& name : POSSIBLE_IMAGE_NAMES)
    if(a_typeName == name)
      return true;

  return false;
} 

bool kslicer::IsSamplerTypeName(const std::string& a_typeName)
{
  if(a_typeName == "struct Sampler")
    return true;
  auto posOfXX = a_typeName.find_last_of("::");
  auto name2   = a_typeName.substr(posOfXX+1);
  if(name2 == "Sampler")
    return true;
  return false;
}

bool kslicer::IsCombinedImageSamplerTypeName(const std::string& a_typeName)
{
  if(a_typeName == "struct ICombinedImageSampler")
    return true;
  auto posOfXX = a_typeName.find_last_of("::");
  auto name2   = a_typeName.substr(posOfXX+1);
  if(name2 == "ICombinedImageSampler")
    return true;
  return false;
}

bool kslicer::IsMatrixTypeName(const std::string& a_typeName)
{
  if(a_typeName      == "float2x2" || a_typeName == "double2x2")
    return true;
  else if(a_typeName == "float3x3" || a_typeName == "double3x3")
    return true;
  else if(a_typeName == "float4x4" || a_typeName == "double4x4")
    return true;
  else
    return false;
}

kslicer::DATA_KIND kslicer::GetContainerTypeDataKind(const std::string& a_typeName)
{
  kslicer::DATA_KIND kind = kslicer::DATA_KIND::KIND_UNKNOWN;
  if(a_typeName == "Texture2D" || a_typeName == "Image2D")
    kind = kslicer::DATA_KIND::KIND_TEXTURE;
  else if(a_typeName == "vector" || a_typeName == "std::vector")
    kind = kslicer::DATA_KIND::KIND_VECTOR;
  else if(a_typeName == "unordered_map" || a_typeName == "std::unordered_map")
    kind = kslicer::DATA_KIND::KIND_HASH_TABLE;
  else if((a_typeName == "shared_ptr" || a_typeName == "std::shared_ptr") && kslicer::IsAccelStruct(a_typeName))
    kind = kslicer::DATA_KIND::KIND_ACCEL_STRUCT;
  return kind;
}