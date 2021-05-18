#include <vector>
#include <array>
#include <memory>
#include <limits>

#include <cassert>

#include "vulkan_basics.h"
#include "test_class_generated.h"

void TestClass_Generated::AllocateAllDescriptorSets()
{
  // allocate pool
  //
  VkDescriptorPoolSize buffersSize;
  buffersSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  buffersSize.descriptorCount = 10*4 + 100; // mul 4 and add 100 because of AMD bug

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.maxSets       = 10 + 2; // add 1 to prevent zero case and one more for internal needs
  descriptorPoolCreateInfo.poolSizeCount = 1;
  descriptorPoolCreateInfo.pPoolSizes    = &buffersSize;
  
  VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &m_dsPool));
  
  // allocate all descriptor sets
  //
  VkDescriptorSetLayout layouts[10] = {};
  layouts[0] = InitEyeRay2DSLayout;
  layouts[1] = RayTraceDSLayout;
  layouts[2] = MakeMaterialDSLayout;
  layouts[3] = NextBounceDSLayout;
  layouts[4] = ContributeToImageDSLayout;
  layouts[5] = InitEyeRayDSLayout;
  layouts[6] = RayTraceDSLayout;
  layouts[7] = MakeMaterialDSLayout;
  layouts[8] = GetColorDSLayout;
  layouts[9] = PackXYDSLayout;

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.descriptorPool     = m_dsPool;  
  descriptorSetAllocateInfo.descriptorSetCount = 10;     
  descriptorSetAllocateInfo.pSetLayouts        = layouts;

  auto tmpRes = vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, m_allGeneratedDS);
  VK_CHECK_RESULT(tmpRes);
}

void TestClass_Generated::InitAllGeneratedDescriptorSets_NaivePathTrace()
{
  // now create actual bindings
  //
  // descriptor set #0: InitEyeRay2Cmd (["in_pakedXY","rayPosAndNear","rayDirAndFar","accumColor","accumThoroughput","threadFlags"])
  {
    constexpr uint additionalSize = 1;

    std::array<VkDescriptorBufferInfo, 6 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   6 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = NaivePathTrace_local.in_pakedXYBuffer;
    descriptorBufferInfo[0].offset = NaivePathTrace_local.in_pakedXYOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[0];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = NaivePathTrace_local.rayPosAndNearBuffer;
    descriptorBufferInfo[1].offset = NaivePathTrace_local.rayPosAndNearOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[0];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = NaivePathTrace_local.rayDirAndFarBuffer;
    descriptorBufferInfo[2].offset = NaivePathTrace_local.rayDirAndFarOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[0];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = NaivePathTrace_local.accumColorBuffer;
    descriptorBufferInfo[3].offset = NaivePathTrace_local.accumColorOffset;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[0];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr; 

    descriptorBufferInfo[4]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[4].buffer = NaivePathTrace_local.accumThoroughputBuffer;
    descriptorBufferInfo[4].offset = NaivePathTrace_local.accumThoroughputOffset;
    descriptorBufferInfo[4].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[4]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[4].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[4].dstSet           = m_allGeneratedDS[0];
    writeDescriptorSet[4].dstBinding       = 4;
    writeDescriptorSet[4].descriptorCount  = 1;
    writeDescriptorSet[4].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[4].pBufferInfo      = &descriptorBufferInfo[4];
    writeDescriptorSet[4].pImageInfo       = nullptr;
    writeDescriptorSet[4].pTexelBufferView = nullptr; 

    descriptorBufferInfo[5]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[5].buffer = NaivePathTrace_local.threadFlagsBuffer;
    descriptorBufferInfo[5].offset = NaivePathTrace_local.threadFlagsOffset;
    descriptorBufferInfo[5].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[5]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[5].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[5].dstSet           = m_allGeneratedDS[0];
    writeDescriptorSet[5].dstBinding       = 5;
    writeDescriptorSet[5].descriptorCount  = 1;
    writeDescriptorSet[5].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[5].pBufferInfo      = &descriptorBufferInfo[5];
    writeDescriptorSet[5].pImageInfo       = nullptr;
    writeDescriptorSet[5].pTexelBufferView = nullptr; 

    descriptorBufferInfo[6]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[6].buffer = m_classDataBuffer;
    descriptorBufferInfo[6].offset = 0;
    descriptorBufferInfo[6].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[6]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[6].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[6].dstSet           = m_allGeneratedDS[0];
    writeDescriptorSet[6].dstBinding       = 6;
    writeDescriptorSet[6].descriptorCount  = 1;
    writeDescriptorSet[6].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[6].pBufferInfo      = &descriptorBufferInfo[6];
    writeDescriptorSet[6].pImageInfo       = nullptr;
    writeDescriptorSet[6].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
  // descriptor set #1: RayTraceCmd (["rayPosAndNear","rayDirAndFar","hit","baricentrics","threadFlags","m_vPos4f","m_indicesReordered","m_intervals","m_nodes"])
  {
    constexpr uint additionalSize = 1;

    std::array<VkDescriptorBufferInfo, 9 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   9 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = NaivePathTrace_local.rayPosAndNearBuffer;
    descriptorBufferInfo[0].offset = NaivePathTrace_local.rayPosAndNearOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = NaivePathTrace_local.rayDirAndFarBuffer;
    descriptorBufferInfo[1].offset = NaivePathTrace_local.rayDirAndFarOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = NaivePathTrace_local.hitBuffer;
    descriptorBufferInfo[2].offset = NaivePathTrace_local.hitOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = NaivePathTrace_local.baricentricsBuffer;
    descriptorBufferInfo[3].offset = NaivePathTrace_local.baricentricsOffset;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr; 

    descriptorBufferInfo[4]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[4].buffer = NaivePathTrace_local.threadFlagsBuffer;
    descriptorBufferInfo[4].offset = NaivePathTrace_local.threadFlagsOffset;
    descriptorBufferInfo[4].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[4]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[4].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[4].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[4].dstBinding       = 4;
    writeDescriptorSet[4].descriptorCount  = 1;
    writeDescriptorSet[4].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[4].pBufferInfo      = &descriptorBufferInfo[4];
    writeDescriptorSet[4].pImageInfo       = nullptr;
    writeDescriptorSet[4].pTexelBufferView = nullptr; 

    descriptorBufferInfo[5]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[5].buffer = m_vdata.m_vPos4fBuffer;
    descriptorBufferInfo[5].offset = m_vdata.m_vPos4fOffset;
    descriptorBufferInfo[5].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[5]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[5].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[5].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[5].dstBinding       = 5;
    writeDescriptorSet[5].descriptorCount  = 1;
    writeDescriptorSet[5].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[5].pBufferInfo      = &descriptorBufferInfo[5];
    writeDescriptorSet[5].pImageInfo       = nullptr;
    writeDescriptorSet[5].pTexelBufferView = nullptr; 

    descriptorBufferInfo[6]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[6].buffer = m_vdata.m_indicesReorderedBuffer;
    descriptorBufferInfo[6].offset = m_vdata.m_indicesReorderedOffset;
    descriptorBufferInfo[6].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[6]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[6].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[6].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[6].dstBinding       = 6;
    writeDescriptorSet[6].descriptorCount  = 1;
    writeDescriptorSet[6].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[6].pBufferInfo      = &descriptorBufferInfo[6];
    writeDescriptorSet[6].pImageInfo       = nullptr;
    writeDescriptorSet[6].pTexelBufferView = nullptr; 

    descriptorBufferInfo[7]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[7].buffer = m_vdata.m_intervalsBuffer;
    descriptorBufferInfo[7].offset = m_vdata.m_intervalsOffset;
    descriptorBufferInfo[7].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[7]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[7].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[7].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[7].dstBinding       = 7;
    writeDescriptorSet[7].descriptorCount  = 1;
    writeDescriptorSet[7].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[7].pBufferInfo      = &descriptorBufferInfo[7];
    writeDescriptorSet[7].pImageInfo       = nullptr;
    writeDescriptorSet[7].pTexelBufferView = nullptr; 

    descriptorBufferInfo[8]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[8].buffer = m_vdata.m_nodesBuffer;
    descriptorBufferInfo[8].offset = m_vdata.m_nodesOffset;
    descriptorBufferInfo[8].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[8]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[8].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[8].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[8].dstBinding       = 8;
    writeDescriptorSet[8].descriptorCount  = 1;
    writeDescriptorSet[8].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[8].pBufferInfo      = &descriptorBufferInfo[8];
    writeDescriptorSet[8].pImageInfo       = nullptr;
    writeDescriptorSet[8].pTexelBufferView = nullptr; 

    descriptorBufferInfo[9]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[9].buffer = m_classDataBuffer;
    descriptorBufferInfo[9].offset = 0;
    descriptorBufferInfo[9].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[9]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[9].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[9].dstSet           = m_allGeneratedDS[1];
    writeDescriptorSet[9].dstBinding       = 9;
    writeDescriptorSet[9].descriptorCount  = 1;
    writeDescriptorSet[9].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[9].pBufferInfo      = &descriptorBufferInfo[9];
    writeDescriptorSet[9].pImageInfo       = nullptr;
    writeDescriptorSet[9].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
  // descriptor set #2: MakeMaterialCmd (["hit","threadFlags","m_materialData","m_materialIds","m_materialOffsets","IMaterialObjPtrData"])
  {
    constexpr uint additionalSize = 1;

    std::array<VkDescriptorBufferInfo, 6 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   6 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = NaivePathTrace_local.hitBuffer;
    descriptorBufferInfo[0].offset = NaivePathTrace_local.hitOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[2];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = NaivePathTrace_local.threadFlagsBuffer;
    descriptorBufferInfo[1].offset = NaivePathTrace_local.threadFlagsOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[2];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = m_vdata.m_materialDataBuffer;
    descriptorBufferInfo[2].offset = m_vdata.m_materialDataOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[2];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = m_vdata.m_materialIdsBuffer;
    descriptorBufferInfo[3].offset = m_vdata.m_materialIdsOffset;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[2];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr; 

    descriptorBufferInfo[4]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[4].buffer = m_vdata.m_materialOffsetsBuffer;
    descriptorBufferInfo[4].offset = m_vdata.m_materialOffsetsOffset;
    descriptorBufferInfo[4].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[4]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[4].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[4].dstSet           = m_allGeneratedDS[2];
    writeDescriptorSet[4].dstBinding       = 4;
    writeDescriptorSet[4].descriptorCount  = 1;
    writeDescriptorSet[4].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[4].pBufferInfo      = &descriptorBufferInfo[4];
    writeDescriptorSet[4].pImageInfo       = nullptr;
    writeDescriptorSet[4].pTexelBufferView = nullptr; 

    descriptorBufferInfo[5]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[5].buffer = m_IMaterialObjPtrBuffer;
    descriptorBufferInfo[5].offset = m_IMaterialObjPtrOffset;
    descriptorBufferInfo[5].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[5]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[5].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[5].dstSet           = m_allGeneratedDS[2];
    writeDescriptorSet[5].dstBinding       = 5;
    writeDescriptorSet[5].descriptorCount  = 1;
    writeDescriptorSet[5].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[5].pBufferInfo      = &descriptorBufferInfo[5];
    writeDescriptorSet[5].pImageInfo       = nullptr;
    writeDescriptorSet[5].pTexelBufferView = nullptr; 

    descriptorBufferInfo[6]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[6].buffer = m_classDataBuffer;
    descriptorBufferInfo[6].offset = 0;
    descriptorBufferInfo[6].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[6]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[6].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[6].dstSet           = m_allGeneratedDS[2];
    writeDescriptorSet[6].dstBinding       = 6;
    writeDescriptorSet[6].descriptorCount  = 1;
    writeDescriptorSet[6].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[6].pBufferInfo      = &descriptorBufferInfo[6];
    writeDescriptorSet[6].pImageInfo       = nullptr;
    writeDescriptorSet[6].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
  // descriptor set #3: NextBounceCmd (["hit","baricentrics","m_indicesReordered.data()","m_vPos4f.data()","m_vNorm4f.data()","rayPosAndNear","rayDirAndFar","m_randomGens.data()","accumColor","accumThoroughput","threadFlags","IMaterialObjPtrData"])
  {
    constexpr uint additionalSize = 2;

    std::array<VkDescriptorBufferInfo, 12 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   12 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = NaivePathTrace_local.hitBuffer;
    descriptorBufferInfo[0].offset = NaivePathTrace_local.hitOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = NaivePathTrace_local.baricentricsBuffer;
    descriptorBufferInfo[1].offset = NaivePathTrace_local.baricentricsOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = m_vdata.m_indicesReorderedBuffer;
    descriptorBufferInfo[2].offset = m_vdata.m_indicesReorderedOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = m_vdata.m_vPos4fBuffer;
    descriptorBufferInfo[3].offset = m_vdata.m_vPos4fOffset;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr; 

    descriptorBufferInfo[4]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[4].buffer = m_vdata.m_vNorm4fBuffer;
    descriptorBufferInfo[4].offset = m_vdata.m_vNorm4fOffset;
    descriptorBufferInfo[4].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[4]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[4].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[4].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[4].dstBinding       = 4;
    writeDescriptorSet[4].descriptorCount  = 1;
    writeDescriptorSet[4].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[4].pBufferInfo      = &descriptorBufferInfo[4];
    writeDescriptorSet[4].pImageInfo       = nullptr;
    writeDescriptorSet[4].pTexelBufferView = nullptr; 

    descriptorBufferInfo[5]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[5].buffer = NaivePathTrace_local.rayPosAndNearBuffer;
    descriptorBufferInfo[5].offset = NaivePathTrace_local.rayPosAndNearOffset;
    descriptorBufferInfo[5].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[5]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[5].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[5].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[5].dstBinding       = 5;
    writeDescriptorSet[5].descriptorCount  = 1;
    writeDescriptorSet[5].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[5].pBufferInfo      = &descriptorBufferInfo[5];
    writeDescriptorSet[5].pImageInfo       = nullptr;
    writeDescriptorSet[5].pTexelBufferView = nullptr; 

    descriptorBufferInfo[6]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[6].buffer = NaivePathTrace_local.rayDirAndFarBuffer;
    descriptorBufferInfo[6].offset = NaivePathTrace_local.rayDirAndFarOffset;
    descriptorBufferInfo[6].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[6]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[6].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[6].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[6].dstBinding       = 6;
    writeDescriptorSet[6].descriptorCount  = 1;
    writeDescriptorSet[6].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[6].pBufferInfo      = &descriptorBufferInfo[6];
    writeDescriptorSet[6].pImageInfo       = nullptr;
    writeDescriptorSet[6].pTexelBufferView = nullptr; 

    descriptorBufferInfo[7]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[7].buffer = m_vdata.m_randomGensBuffer;
    descriptorBufferInfo[7].offset = m_vdata.m_randomGensOffset;
    descriptorBufferInfo[7].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[7]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[7].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[7].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[7].dstBinding       = 7;
    writeDescriptorSet[7].descriptorCount  = 1;
    writeDescriptorSet[7].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[7].pBufferInfo      = &descriptorBufferInfo[7];
    writeDescriptorSet[7].pImageInfo       = nullptr;
    writeDescriptorSet[7].pTexelBufferView = nullptr; 

    descriptorBufferInfo[8]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[8].buffer = NaivePathTrace_local.accumColorBuffer;
    descriptorBufferInfo[8].offset = NaivePathTrace_local.accumColorOffset;
    descriptorBufferInfo[8].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[8]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[8].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[8].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[8].dstBinding       = 8;
    writeDescriptorSet[8].descriptorCount  = 1;
    writeDescriptorSet[8].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[8].pBufferInfo      = &descriptorBufferInfo[8];
    writeDescriptorSet[8].pImageInfo       = nullptr;
    writeDescriptorSet[8].pTexelBufferView = nullptr; 

    descriptorBufferInfo[9]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[9].buffer = NaivePathTrace_local.accumThoroughputBuffer;
    descriptorBufferInfo[9].offset = NaivePathTrace_local.accumThoroughputOffset;
    descriptorBufferInfo[9].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[9]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[9].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[9].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[9].dstBinding       = 9;
    writeDescriptorSet[9].descriptorCount  = 1;
    writeDescriptorSet[9].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[9].pBufferInfo      = &descriptorBufferInfo[9];
    writeDescriptorSet[9].pImageInfo       = nullptr;
    writeDescriptorSet[9].pTexelBufferView = nullptr; 

    descriptorBufferInfo[10]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[10].buffer = NaivePathTrace_local.threadFlagsBuffer;
    descriptorBufferInfo[10].offset = NaivePathTrace_local.threadFlagsOffset;
    descriptorBufferInfo[10].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[10]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[10].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[10].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[10].dstBinding       = 10;
    writeDescriptorSet[10].descriptorCount  = 1;
    writeDescriptorSet[10].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[10].pBufferInfo      = &descriptorBufferInfo[10];
    writeDescriptorSet[10].pImageInfo       = nullptr;
    writeDescriptorSet[10].pTexelBufferView = nullptr; 

    descriptorBufferInfo[11]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[11].buffer = m_IMaterialObjPtrBuffer;
    descriptorBufferInfo[11].offset = m_IMaterialObjPtrOffset;
    descriptorBufferInfo[11].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[11]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[11].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[11].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[11].dstBinding       = 11;
    writeDescriptorSet[11].descriptorCount  = 1;
    writeDescriptorSet[11].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[11].pBufferInfo      = &descriptorBufferInfo[11];
    writeDescriptorSet[11].pImageInfo       = nullptr;
    writeDescriptorSet[11].pTexelBufferView = nullptr; 

    descriptorBufferInfo[12]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[12].buffer = m_vdata.m_materialDataBuffer;
    descriptorBufferInfo[12].offset = 0;
    descriptorBufferInfo[12].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[12]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[12].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[12].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[12].dstBinding       = 12;
    writeDescriptorSet[12].descriptorCount  = 1;
    writeDescriptorSet[12].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[12].pBufferInfo      = &descriptorBufferInfo[12];
    writeDescriptorSet[12].pImageInfo       = nullptr;
    writeDescriptorSet[12].pTexelBufferView = nullptr;
    
    descriptorBufferInfo[12+1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[12+1].buffer = m_classDataBuffer;
    descriptorBufferInfo[12+1].offset = 0;
    descriptorBufferInfo[12+1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[12+1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[12+1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[12+1].dstSet           = m_allGeneratedDS[3];
    writeDescriptorSet[12+1].dstBinding       = 12+1;
    writeDescriptorSet[12+1].descriptorCount  = 1;
    writeDescriptorSet[12+1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[12+1].pBufferInfo      = &descriptorBufferInfo[12+1];
    writeDescriptorSet[12+1].pImageInfo       = nullptr;
    writeDescriptorSet[12+1].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
  // descriptor set #4: ContributeToImageCmd (["accumColor","in_pakedXY","out_color","threadFlags"])
  {
    constexpr uint additionalSize = 1;

    std::array<VkDescriptorBufferInfo, 4 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   4 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = NaivePathTrace_local.accumColorBuffer;
    descriptorBufferInfo[0].offset = NaivePathTrace_local.accumColorOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[4];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = NaivePathTrace_local.in_pakedXYBuffer;
    descriptorBufferInfo[1].offset = NaivePathTrace_local.in_pakedXYOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[4];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = NaivePathTrace_local.out_colorBuffer;
    descriptorBufferInfo[2].offset = NaivePathTrace_local.out_colorOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[4];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = NaivePathTrace_local.threadFlagsBuffer;
    descriptorBufferInfo[3].offset = NaivePathTrace_local.threadFlagsOffset;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[4];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr; 

    descriptorBufferInfo[4]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[4].buffer = m_classDataBuffer;
    descriptorBufferInfo[4].offset = 0;
    descriptorBufferInfo[4].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[4]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[4].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[4].dstSet           = m_allGeneratedDS[4];
    writeDescriptorSet[4].dstBinding       = 4;
    writeDescriptorSet[4].descriptorCount  = 1;
    writeDescriptorSet[4].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[4].pBufferInfo      = &descriptorBufferInfo[4];
    writeDescriptorSet[4].pImageInfo       = nullptr;
    writeDescriptorSet[4].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
}

void TestClass_Generated::InitAllGeneratedDescriptorSets_CastSingleRay()
{
  // now create actual bindings
  //
  // descriptor set #5: InitEyeRayCmd (["in_pakedXY","rayPosAndNear","rayDirAndFar","threadFlags"])
  {
    constexpr uint additionalSize = 1;

    std::array<VkDescriptorBufferInfo, 4 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   4 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = CastSingleRay_local.in_pakedXYBuffer;
    descriptorBufferInfo[0].offset = CastSingleRay_local.in_pakedXYOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[5];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = CastSingleRay_local.rayPosAndNearBuffer;
    descriptorBufferInfo[1].offset = CastSingleRay_local.rayPosAndNearOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[5];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = CastSingleRay_local.rayDirAndFarBuffer;
    descriptorBufferInfo[2].offset = CastSingleRay_local.rayDirAndFarOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[5];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = CastSingleRay_local.threadFlagsBuffer;
    descriptorBufferInfo[3].offset = CastSingleRay_local.threadFlagsOffset;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[5];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr; 

    descriptorBufferInfo[4]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[4].buffer = m_classDataBuffer;
    descriptorBufferInfo[4].offset = 0;
    descriptorBufferInfo[4].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[4]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[4].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[4].dstSet           = m_allGeneratedDS[5];
    writeDescriptorSet[4].dstBinding       = 4;
    writeDescriptorSet[4].descriptorCount  = 1;
    writeDescriptorSet[4].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[4].pBufferInfo      = &descriptorBufferInfo[4];
    writeDescriptorSet[4].pImageInfo       = nullptr;
    writeDescriptorSet[4].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
  // descriptor set #6: RayTraceCmd (["rayPosAndNear","rayDirAndFar","hit","baricentrics","threadFlags","m_vPos4f","m_indicesReordered","m_intervals","m_nodes"])
  {
    constexpr uint additionalSize = 1;

    std::array<VkDescriptorBufferInfo, 9 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   9 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = CastSingleRay_local.rayPosAndNearBuffer;
    descriptorBufferInfo[0].offset = CastSingleRay_local.rayPosAndNearOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = CastSingleRay_local.rayDirAndFarBuffer;
    descriptorBufferInfo[1].offset = CastSingleRay_local.rayDirAndFarOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = CastSingleRay_local.hitBuffer;
    descriptorBufferInfo[2].offset = CastSingleRay_local.hitOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = CastSingleRay_local.baricentricsBuffer;
    descriptorBufferInfo[3].offset = CastSingleRay_local.baricentricsOffset;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr; 

    descriptorBufferInfo[4]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[4].buffer = CastSingleRay_local.threadFlagsBuffer;
    descriptorBufferInfo[4].offset = CastSingleRay_local.threadFlagsOffset;
    descriptorBufferInfo[4].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[4]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[4].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[4].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[4].dstBinding       = 4;
    writeDescriptorSet[4].descriptorCount  = 1;
    writeDescriptorSet[4].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[4].pBufferInfo      = &descriptorBufferInfo[4];
    writeDescriptorSet[4].pImageInfo       = nullptr;
    writeDescriptorSet[4].pTexelBufferView = nullptr; 

    descriptorBufferInfo[5]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[5].buffer = m_vdata.m_vPos4fBuffer;
    descriptorBufferInfo[5].offset = m_vdata.m_vPos4fOffset;
    descriptorBufferInfo[5].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[5]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[5].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[5].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[5].dstBinding       = 5;
    writeDescriptorSet[5].descriptorCount  = 1;
    writeDescriptorSet[5].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[5].pBufferInfo      = &descriptorBufferInfo[5];
    writeDescriptorSet[5].pImageInfo       = nullptr;
    writeDescriptorSet[5].pTexelBufferView = nullptr; 

    descriptorBufferInfo[6]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[6].buffer = m_vdata.m_indicesReorderedBuffer;
    descriptorBufferInfo[6].offset = m_vdata.m_indicesReorderedOffset;
    descriptorBufferInfo[6].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[6]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[6].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[6].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[6].dstBinding       = 6;
    writeDescriptorSet[6].descriptorCount  = 1;
    writeDescriptorSet[6].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[6].pBufferInfo      = &descriptorBufferInfo[6];
    writeDescriptorSet[6].pImageInfo       = nullptr;
    writeDescriptorSet[6].pTexelBufferView = nullptr; 

    descriptorBufferInfo[7]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[7].buffer = m_vdata.m_intervalsBuffer;
    descriptorBufferInfo[7].offset = m_vdata.m_intervalsOffset;
    descriptorBufferInfo[7].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[7]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[7].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[7].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[7].dstBinding       = 7;
    writeDescriptorSet[7].descriptorCount  = 1;
    writeDescriptorSet[7].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[7].pBufferInfo      = &descriptorBufferInfo[7];
    writeDescriptorSet[7].pImageInfo       = nullptr;
    writeDescriptorSet[7].pTexelBufferView = nullptr; 

    descriptorBufferInfo[8]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[8].buffer = m_vdata.m_nodesBuffer;
    descriptorBufferInfo[8].offset = m_vdata.m_nodesOffset;
    descriptorBufferInfo[8].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[8]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[8].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[8].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[8].dstBinding       = 8;
    writeDescriptorSet[8].descriptorCount  = 1;
    writeDescriptorSet[8].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[8].pBufferInfo      = &descriptorBufferInfo[8];
    writeDescriptorSet[8].pImageInfo       = nullptr;
    writeDescriptorSet[8].pTexelBufferView = nullptr; 

    descriptorBufferInfo[9]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[9].buffer = m_classDataBuffer;
    descriptorBufferInfo[9].offset = 0;
    descriptorBufferInfo[9].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[9]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[9].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[9].dstSet           = m_allGeneratedDS[6];
    writeDescriptorSet[9].dstBinding       = 9;
    writeDescriptorSet[9].descriptorCount  = 1;
    writeDescriptorSet[9].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[9].pBufferInfo      = &descriptorBufferInfo[9];
    writeDescriptorSet[9].pImageInfo       = nullptr;
    writeDescriptorSet[9].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
  // descriptor set #7: MakeMaterialCmd (["hit","threadFlags","m_materialData","m_materialIds","m_materialOffsets","IMaterialObjPtrData"])
  {
    constexpr uint additionalSize = 1;

    std::array<VkDescriptorBufferInfo, 6 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   6 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = CastSingleRay_local.hitBuffer;
    descriptorBufferInfo[0].offset = CastSingleRay_local.hitOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[7];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = CastSingleRay_local.threadFlagsBuffer;
    descriptorBufferInfo[1].offset = CastSingleRay_local.threadFlagsOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[7];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = m_vdata.m_materialDataBuffer;
    descriptorBufferInfo[2].offset = m_vdata.m_materialDataOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[7];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = m_vdata.m_materialIdsBuffer;
    descriptorBufferInfo[3].offset = m_vdata.m_materialIdsOffset;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[7];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr; 

    descriptorBufferInfo[4]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[4].buffer = m_vdata.m_materialOffsetsBuffer;
    descriptorBufferInfo[4].offset = m_vdata.m_materialOffsetsOffset;
    descriptorBufferInfo[4].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[4]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[4].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[4].dstSet           = m_allGeneratedDS[7];
    writeDescriptorSet[4].dstBinding       = 4;
    writeDescriptorSet[4].descriptorCount  = 1;
    writeDescriptorSet[4].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[4].pBufferInfo      = &descriptorBufferInfo[4];
    writeDescriptorSet[4].pImageInfo       = nullptr;
    writeDescriptorSet[4].pTexelBufferView = nullptr; 

    descriptorBufferInfo[5]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[5].buffer = m_IMaterialObjPtrBuffer;
    descriptorBufferInfo[5].offset = m_IMaterialObjPtrOffset;
    descriptorBufferInfo[5].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[5]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[5].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[5].dstSet           = m_allGeneratedDS[7];
    writeDescriptorSet[5].dstBinding       = 5;
    writeDescriptorSet[5].descriptorCount  = 1;
    writeDescriptorSet[5].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[5].pBufferInfo      = &descriptorBufferInfo[5];
    writeDescriptorSet[5].pImageInfo       = nullptr;
    writeDescriptorSet[5].pTexelBufferView = nullptr; 

    descriptorBufferInfo[6]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[6].buffer = m_classDataBuffer;
    descriptorBufferInfo[6].offset = 0;
    descriptorBufferInfo[6].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[6]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[6].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[6].dstSet           = m_allGeneratedDS[7];
    writeDescriptorSet[6].dstBinding       = 6;
    writeDescriptorSet[6].descriptorCount  = 1;
    writeDescriptorSet[6].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[6].pBufferInfo      = &descriptorBufferInfo[6];
    writeDescriptorSet[6].pImageInfo       = nullptr;
    writeDescriptorSet[6].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
  // descriptor set #8: GetColorCmd (["out_color","threadFlags","IMaterialObjPtrData"])
  {
    constexpr uint additionalSize = 2;

    std::array<VkDescriptorBufferInfo, 3 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   3 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = CastSingleRay_local.out_colorBuffer;
    descriptorBufferInfo[0].offset = CastSingleRay_local.out_colorOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[8];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = CastSingleRay_local.threadFlagsBuffer;
    descriptorBufferInfo[1].offset = CastSingleRay_local.threadFlagsOffset;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[8];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr; 

    descriptorBufferInfo[2]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[2].buffer = m_IMaterialObjPtrBuffer;
    descriptorBufferInfo[2].offset = m_IMaterialObjPtrOffset;
    descriptorBufferInfo[2].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[2]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[2].dstSet           = m_allGeneratedDS[8];
    writeDescriptorSet[2].dstBinding       = 2;
    writeDescriptorSet[2].descriptorCount  = 1;
    writeDescriptorSet[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[2].pBufferInfo      = &descriptorBufferInfo[2];
    writeDescriptorSet[2].pImageInfo       = nullptr;
    writeDescriptorSet[2].pTexelBufferView = nullptr; 

    descriptorBufferInfo[3]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3].buffer = m_vdata.m_materialDataBuffer;
    descriptorBufferInfo[3].offset = 0;
    descriptorBufferInfo[3].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3].dstSet           = m_allGeneratedDS[8];
    writeDescriptorSet[3].dstBinding       = 3;
    writeDescriptorSet[3].descriptorCount  = 1;
    writeDescriptorSet[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3].pBufferInfo      = &descriptorBufferInfo[3];
    writeDescriptorSet[3].pImageInfo       = nullptr;
    writeDescriptorSet[3].pTexelBufferView = nullptr;
    
    descriptorBufferInfo[3+1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[3+1].buffer = m_classDataBuffer;
    descriptorBufferInfo[3+1].offset = 0;
    descriptorBufferInfo[3+1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[3+1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[3+1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[3+1].dstSet           = m_allGeneratedDS[8];
    writeDescriptorSet[3+1].dstBinding       = 3+1;
    writeDescriptorSet[3+1].descriptorCount  = 1;
    writeDescriptorSet[3+1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[3+1].pBufferInfo      = &descriptorBufferInfo[3+1];
    writeDescriptorSet[3+1].pImageInfo       = nullptr;
    writeDescriptorSet[3+1].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
}

void TestClass_Generated::InitAllGeneratedDescriptorSets_PackXY()
{
  // now create actual bindings
  //
  // descriptor set #9: PackXYCmd (["out_pakedXY"])
  {
    constexpr uint additionalSize = 1;

    std::array<VkDescriptorBufferInfo, 1 + additionalSize> descriptorBufferInfo;
    std::array<VkWriteDescriptorSet,   1 + additionalSize> writeDescriptorSet;

    descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[0].buffer = PackXY_local.out_pakedXYBuffer;
    descriptorBufferInfo[0].offset = PackXY_local.out_pakedXYOffset;
    descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet           = m_allGeneratedDS[9];
    writeDescriptorSet[0].dstBinding       = 0;
    writeDescriptorSet[0].descriptorCount  = 1;
    writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
    writeDescriptorSet[0].pImageInfo       = nullptr;
    writeDescriptorSet[0].pTexelBufferView = nullptr; 

    descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[1].buffer = m_classDataBuffer;
    descriptorBufferInfo[1].offset = 0;
    descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;  

    writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet           = m_allGeneratedDS[9];
    writeDescriptorSet[1].dstBinding       = 1;
    writeDescriptorSet[1].descriptorCount  = 1;
    writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
    writeDescriptorSet[1].pImageInfo       = nullptr;
    writeDescriptorSet[1].pTexelBufferView = nullptr;

   
    vkUpdateDescriptorSets(device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
}



