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

void {{MainClassName}}::InitHelpers()
{
  vkGetPhysicalDeviceProperties(vk_data.physicalDevice, &m_devProps);
  m_pMaker = std::make_shared<vkfw::ComputePipelineMaker>();

  VkDescriptorType dtype = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  uint32_t dtypesize     = {{TotalDescriptorSets}};
  m_pBindings            = std::make_shared<vkfw::ProgramBindings>(vk_data.device, &dtype, &dtypesize, 1);

}

void {{MainClassName}}::InitBuffers(size_t a_maxThreadsCount)
{
  std::vector<VkBuffer> allBuffers;
  
## for Buffer in LocalVarsBuffers
  {{Buffer.Name}}Buffer = vkfw::CreateBuffer(vk_data.device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
## endfor

  m_classDataBuffer = vkfw::CreateBuffer(vk_data.device, {{AllClassVarsSize}},  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffers.push_back(m_classDataBuffer);

  m_allMem = vkfw::AllocateAndBindWithPadding(vk_data.device, vk_data.physicalDevice, allBuffers);
}

{{KernelsCmd}}

{{MainFuncCmd}}

