#include "kslicer.h"
#include <array>

static std::array<std::string,7> POSSIBLE_KERNEL_NAMES = {"kernel_", "kernel1D_", "kernel2D_", "kernel3D_", "kernelBE1D_", "kernelBE2D_", "kernelBE3D_"};

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
  auto pos1 = a_funcName.find("kernel1D_");
  auto pos2 = a_funcName.find("kernel2D_");
  auto pos3 = a_funcName.find("kernel3D_");

  auto pos4 = a_funcName.find("kernelBE1D_");
  auto pos5 = a_funcName.find("kernelBE2D_");
  auto pos6 = a_funcName.find("kernelBE3D_");

  if(pos1 != std::string::npos || pos4 != std::string::npos)
    return 1;
  else if(pos2 != std::string::npos || pos5 != std::string::npos) 
    return 2;
  else if(pos3 != std::string::npos || pos6 != std::string::npos)
    return 3;
  else
    return 0;
} 

bool kslicer::IsTextureContainer(const std::string& a_typeName)
{
  if(a_typeName == "Texture1D" || a_typeName == "Image1D")
    return true;
  if(a_typeName == "Texture2D" || a_typeName == "Image2D")
    return true;
  if(a_typeName == "Texture3D" || a_typeName == "Image3D")
    return true;
  if(a_typeName == "TextureCube" || a_typeName == "ImageCube")
    return true;

  return false;
} 

bool kslicer::IsSamplerTypeName(const std::string& a_typeName)
{
  if(a_typeName == "struct Sampler")
    return true;
  auto posOfXX = a_typeName.find_last_of("::");
  auto name2   = a_typeName.substr(posOfXX+1);
  if(name2 == "Sampler" || name2 == "Sampler")
    return true;
  return false;
}

bool kslicer::IsCombinedImageSamplerTypeName(const std::string& a_typeName)
{
  if(a_typeName == "struct ICombinedImageSampler")
    return true;
  auto posOfXX = a_typeName.find_last_of("::");
  auto name2   = a_typeName.substr(posOfXX+1);
  if(name2 == "ICombinedImageSampler" || name2 == "ICombinedImageSampler")
    return true;
  return false;
}

