#include <vector>
#include <memory>

#include "vulkan_basics.h"

#include "{{IncludeClassDecl}}"

{{Includes}}

{{MainClassName}}_Generated::~{{MainClassName}}_Generated()
{
  m_pMaker = nullptr;

## for Kernel in Kernels
  vkDestroyDescriptorSetLayout(device, {{Kernel.Name}}DSLayout, nullptr);
  {{Kernel.Name}}DSLayout = VK_NULL_HANDLE;
## endfor
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;

## for MainFunc in MainFunctions
## for Buffer in MainFunc.LocalVarsBuffers
  vkDestroyBuffer(device, {{Buffer.Name}}Buffer, nullptr);
## endfor

## endfor
  vkDestroyBuffer(device, m_classDataBuffer, nullptr);
  vkFreeMemory   (device, m_allMem, nullptr);
}

void {{MainClassName}}_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
  m_pMaker = std::make_unique<vkfw::ComputePipelineMaker>();
}

## for Kernel in Kernels
VkDescriptorSetLayout {{MainClassName}}_Generated::Create{{Kernel.Name}}DSLayout()
{
  VkDescriptorSetLayoutBinding dsBindings[{{Kernel.ArgCount}}+1] = {};
  
## for KernelARG in Kernel.Args
  // binding for {{KernelARG.Name}}
  dsBindings[{{KernelARG.Id}}].binding            = {{KernelARG.Id}};
  dsBindings[{{KernelARG.Id}}].descriptorType     = {{KernelARG.Type}};
  dsBindings[{{KernelARG.Id}}].descriptorCount    = 1;
  dsBindings[{{KernelARG.Id}}].stageFlags         = {{KernelARG.Flags}};
  dsBindings[{{KernelARG.Id}}].pImmutableSamplers = nullptr;

## endfor
  dsBindings[{{Kernel.ArgCount}}].binding            = {{Kernel.ArgCount}};
  dsBindings[{{Kernel.ArgCount}}].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[{{Kernel.ArgCount}}].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[{{Kernel.ArgCount}}].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t({{Kernel.ArgCount}}+1);
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

## for MainFunc in MainFunctions
void {{MainClassName}}_Generated::InitAllGeneratedDescriptorSets_{{MainFunc.Name}}()
{
  // allocate pool
  //
  {
    VkDescriptorPoolSize buffersSize;
    buffersSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    buffersSize.descriptorCount = {{TotalDSNumber}} + 1; // add one to exclude zero case
  
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets       = {{TotalDSNumber}} + 1; // add one to exclude zero case
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes    = &buffersSize;
    
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &m_dsPool));
  }

  // allocate all descriptor sets
  //
  {
    VkDescriptorSetLayout layouts[{{TotalDSNumber}}] = {};
## for DescriptorSet in MainFunc.DescriptorSets
    layouts[{{DescriptorSet.Id}}] = {{DescriptorSet.Layout}};
## endfor

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool     = m_dsPool;  
    descriptorSetAllocateInfo.descriptorSetCount = {{TotalDSNumber}};     
    descriptorSetAllocateInfo.pSetLayouts        = layouts;
  }
 
  // now create actual bindings
  //
## for DescriptorSet in MainFunc.DescriptorSets
  // descriptor set #{{DescriptorSet.Id}} 
  {
    std::vector<VkDescriptorBufferInfo> descriptorBufferInfo({{DescriptorSet.ArgNumber}}+1);
    std::vector<VkWriteDescriptorSet>   writeDescriptorSet({{DescriptorSet.ArgNumber}}+1);

## for Arg in DescriptorSet.Args
    descriptorBufferInfo[{{Arg.Id}}]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[{{Arg.Id}}].buffer = {{Arg.Name}}Buffer;
    descriptorBufferInfo[{{Arg.Id}}].offset = {{Arg.Name}}Offset;
    descriptorBufferInfo[{{Arg.Id}}].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[{{Arg.Id}}]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[{{Arg.Id}}].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[{{Arg.Id}}].dstSet           = m_allGeneratedDS[{{DescriptorSet.Id}}];
    writeDescriptorSet[{{Arg.Id}}].dstBinding       = {{Arg.Id}};
    writeDescriptorSet[{{Arg.Id}}].descriptorCount  = 1;
    writeDescriptorSet[{{Arg.Id}}].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[{{Arg.Id}}].pBufferInfo      = &descriptorBufferInfo[{{Arg.Id}}];
    writeDescriptorSet[{{Arg.Id}}].pImageInfo       = nullptr;
    writeDescriptorSet[{{Arg.Id}}].pTexelBufferView = nullptr; 

## endfor
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].buffer = m_classDataBuffer;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].offset = 0;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[{{DescriptorSet.ArgNumber}}]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].dstSet           = m_allGeneratedDS[{{DescriptorSet.Id}}];
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].dstBinding       = {{DescriptorSet.ArgNumber}};
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].descriptorCount  = 1;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].pBufferInfo      = &descriptorBufferInfo[{{DescriptorSet.ArgNumber}}];
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].pImageInfo       = nullptr;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].pTexelBufferView = nullptr; 
   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
## endfor
}

## endfor


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

## for MainFunc in MainFunctions  
## for Buffer in MainFunc.LocalVarsBuffers
  {{Buffer.Name}}Buffer = vkfw::CreateBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back({{Buffer.Name}}Buffer);
## endfor
## endfor

  m_classDataBuffer = vkfw::CreateBuffer(device, {{AllClassVarsSize}},  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffers.push_back(m_classDataBuffer);

  m_allMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, allBuffers);
}

## for Kernel in Kernels
void {{MainClassName}}_Generated::{{Kernel.Decl}}
{
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Pipeline);
  
  uint32_t pcData[2] = {tidX, tidY};
  vkCmdPushConstants(m_currCmdBuffer, TestColorLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)*2, pcData);
  
  vkCmdDispatch(m_currCmdBuffer, tidX/m_blockSize[0], tidY/m_blockSize[1], 1);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
}

## endfor

## for MainFunc in MainFunctions
{{MainFunc.MainFuncCmd}}

## endfor

