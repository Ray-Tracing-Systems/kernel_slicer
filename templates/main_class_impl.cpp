#include <vector>
#include <memory>

#include "vulkan_basics.h"

#include "{{IncludeClassDecl}}"

{{Includes}}

{{MainClassName}}_Generated::~{{MainClassName}}_Generated()
{
  m_pMaker    = nullptr;
  m_pBindings = nullptr;

## for Buffer in LocalVarsBuffers
  vkDestroyBuffer(device, {{Buffer.Name}}Buffer, nullptr);
## endfor

  vkDestroyBuffer(device, m_classDataBuffer, nullptr);
  vkFreeMemory   (device, m_allMem, nullptr);
}

void {{MainClassName}}_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
  m_pMaker = std::make_unique<vkfw::ComputePipelineMaker>();

  VkDescriptorType dtype = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  uint32_t dtypesize     = {{TotalDescriptorSets}};
  m_pBindings            = std::make_unique<vkfw::ProgramBindings>(device, &dtype, &dtypesize, 1);

}

void {{MainClassName}}_Generated::UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
{
## for Var in ClassVars
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, {{Var.Offset}}, &{{Var.Name}}, {{Var.Size}});
## endfor
}


void {{MainClassName}}_Generated::UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
{

}


void {{MainClassName}}_Generated::InitBuffers(size_t a_maxThreadsCount)
{
  std::vector<VkBuffer> allBuffers;
  
## for Buffer in LocalVarsBuffers
  {{Buffer.Name}}Buffer = vkfw::CreateBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back({{Buffer.Name}}Buffer);
## endfor

  m_classDataBuffer = vkfw::CreateBuffer(device, {{AllClassVarsSize}},  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffers.push_back(m_classDataBuffer);

  m_allMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, allBuffers);
}

{{KernelsCmd}}

{{MainFuncCmd}}

