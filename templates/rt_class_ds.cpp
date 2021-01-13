#include <vector>
#include <memory>
#include <limits>

#include <cassert>

#include "vulkan_basics.h"
#include "{{IncludeClassDecl}}"

void {{MainClassName}}_Generated::AllocateAllDescriptorSets()
{
  // allocate pool
  //
  VkDescriptorPoolSize buffersSize;
  buffersSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  buffersSize.descriptorCount = {{TotalDSNumber}}*4 + 10; // mul 4 and add 10 because of AMD bug

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.maxSets       = {{TotalDSNumber}} + 1; // add 1 to prevent zero case
  descriptorPoolCreateInfo.poolSizeCount = 1;
  descriptorPoolCreateInfo.pPoolSizes    = &buffersSize;
  
  VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &m_dsPool));
  
  // allocate all descriptor sets
  //
  VkDescriptorSetLayout layouts[{{TotalDSNumber}}] = {};
## for DescriptorSet in DescriptorSetsAll
  layouts[{{DescriptorSet.Id}}] = {{DescriptorSet.Layout}};
## endfor

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.descriptorPool     = m_dsPool;  
  descriptorSetAllocateInfo.descriptorSetCount = {{TotalDSNumber}};     
  descriptorSetAllocateInfo.pSetLayouts        = layouts;

  auto tmpRes = vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, m_allGeneratedDS);
  VK_CHECK_RESULT(tmpRes); 
}

## for MainFunc in MainFunctions
void {{MainClassName}}_Generated::InitAllGeneratedDescriptorSets_{{MainFunc.Name}}()
{
  // now create actual bindings
  //
## for DescriptorSet in MainFunc.DescriptorSets
  // descriptor set #{{DescriptorSet.Id}}: {{DescriptorSet.KernelName}}Cmd ({{DescriptorSet.ArgNames}})
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


