#include <vector>
#include <memory>

#include "vulkan_basics.h"

#include "{{IncludeClassDecl}}"

{{Includes}}

{{MainClassName}}_Generated::~{{MainClassName}}_Generated()
{
  m_pMaker    = nullptr;
  m_pBindings = nullptr;

## for Kernel in Kernels
  vkDestroyDescriptorSetLayout(device, {{Kernel.Name}}DSLayout, nullptr);
## endfor

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

## for Kernel in Kernels
VkDescriptorSetLayout {{MainClassName}}_Generated::Create{{Kernel.Name}}DSLayout()
{
  VkDescriptorSetLayoutBinding dsBindings[{{Kernel.ArgCount}}] = {};
  
## for KernelARG in Kernel.Args
  // binding for {{KernelARG.Name}}
  dsBindings[{{KernelARG.Id}}].binding            = {{KernelARG.Id}};
  dsBindings[{{KernelARG.Id}}].descriptorType     = {{KernelARG.Type}};
  dsBindings[{{KernelARG.Id}}].descriptorCount    = 1;
  dsBindings[{{KernelARG.Id}}].stageFlags         = {{KernelARG.Flags}};
  dsBindings[{{KernelARG.Id}}].pImmutableSamplers = nullptr;

## endfor
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t({{Kernel.ArgCount}});
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings;
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

## endfor

void {{MainClassName}}_Generated::InitKernels(const char* a_filePath, uint32_t a_blockSizeX, uint32_t a_blockSizeY, uint32_t a_blockSizeZ)
{
  VkSpecializationMapEntry specializationEntries[3] = {};
  {
    specializationEntries[0].constantID = 0;
    specializationEntries[0].offset     = 0;
    specializationEntries[0].size       = sizeof(uint32_t);
  
    specializationEntries[1].constantID = 1;
    specializationEntries[1].offset     = sizeof(uint32_t);
    specializationEntries[1].size       = sizeof(uint32_t);
  
    specializationEntries[2].constantID = 2;
    specializationEntries[2].offset     = 2 * sizeof(uint32_t);
    specializationEntries[2].size       = sizeof(uint32_t);
  }

  uint32_t specializationData[3] = {a_blockSizeX, a_blockSizeY, a_blockSizeZ};
  VkSpecializationInfo specsForWGSize = {};
  {
    specsForWGSize.mapEntryCount = 3;
    specsForWGSize.pMapEntries   = specializationEntries;
    specsForWGSize.dataSize      = 3 * sizeof(uint32_t);
    specsForWGSize.pData         = specializationData;
  }
 

## for Kernel in Kernels
  {{Kernel.Name}}DSLayout = Create{{Kernel.Name}}DSLayout();
  m_pMaker->CreateShader(device, a_filePath, &specsForWGSize, "{{Kernel.OriginalName}}");

  {{Kernel.Name}}Layout   = m_pMaker->MakeLayout(device, {{Kernel.Name}}DSLayout, sizeof(uint32_t)*2);
  {{Kernel.Name}}Pipeline = m_pMaker->MakePipeline(device);   

## endfor

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

