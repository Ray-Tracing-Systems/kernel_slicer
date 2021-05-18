#include <vector>
#include <array>
#include <memory>
#include <limits>

#include <cassert>

#include "vulkan_basics.h"
#include "test_class_generated.h"
#include "include/TestClass_ubo.h"

static uint32_t ComputeReductionAuxBufferElements(uint32_t whole_size, uint32_t wg_size)
{
  uint32_t sizeTotal = 0;
  while (whole_size > 1)
  {
    whole_size  = (whole_size + wg_size - 1) / wg_size;
    sizeTotal  += std::max<uint32_t>(whole_size, 1);
  }
  return sizeTotal;
}

TestClass_Generated::~TestClass_Generated()
{
  m_pMaker = nullptr;

  vkDestroyDescriptorSetLayout(device, GetColorDSLayout, nullptr);
  GetColorDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, NextBounceDSLayout, nullptr);
  NextBounceDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, InitEyeRayDSLayout, nullptr);
  InitEyeRayDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, InitEyeRay2DSLayout, nullptr);
  InitEyeRay2DSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, RayTraceDSLayout, nullptr);
  RayTraceDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, MakeMaterialDSLayout, nullptr);
  MakeMaterialDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, PackXYDSLayout, nullptr);
  PackXYDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, ContributeToImageDSLayout, nullptr);
  ContributeToImageDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, copyKernelFloatDSLayout, nullptr);
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;

  vkDestroyBuffer(device, NaivePathTrace_local.hitBuffer, nullptr);
  vkDestroyBuffer(device, NaivePathTrace_local.accumThoroughputBuffer, nullptr);
  vkDestroyBuffer(device, NaivePathTrace_local.accumColorBuffer, nullptr);
  vkDestroyBuffer(device, NaivePathTrace_local.threadFlagsBuffer, nullptr);
  vkDestroyBuffer(device, NaivePathTrace_local.baricentricsBuffer, nullptr);
  vkDestroyBuffer(device, NaivePathTrace_local.rayDirAndFarBuffer, nullptr);
  vkDestroyBuffer(device, NaivePathTrace_local.rayPosAndNearBuffer, nullptr);

  vkDestroyBuffer(device, CastSingleRay_local.hitBuffer, nullptr);
  vkDestroyBuffer(device, CastSingleRay_local.threadFlagsBuffer, nullptr);
  vkDestroyBuffer(device, CastSingleRay_local.baricentricsBuffer, nullptr);
  vkDestroyBuffer(device, CastSingleRay_local.rayDirAndFarBuffer, nullptr);
  vkDestroyBuffer(device, CastSingleRay_local.rayPosAndNearBuffer, nullptr);


 
  vkDestroyBuffer(device, m_classDataBuffer, nullptr);

  vkDestroyBuffer(device, m_vdata.m_randomGensBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_vNorm4fBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_nodesBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_intervalsBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_materialIdsBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_indicesReorderedBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_materialOffsetsBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_vPos4fBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_materialDataBuffer, nullptr);
  vkDestroyBuffer(device, m_IMaterialObjPtrBuffer, nullptr);

  if(m_allMem != VK_NULL_HANDLE)
    vkFreeMemory(device, m_allMem, nullptr);
  
  if(m_vdata.m_vecMem != VK_NULL_HANDLE)
    vkFreeMemory(device, m_vdata.m_vecMem, nullptr);
}

void TestClass_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
  m_pMaker = std::make_unique<vkfw::ComputePipelineMaker>();
}

VkDescriptorSetLayout TestClass_Generated::CreateGetColorDSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 3+2> dsBindings;

  // binding for out_color
  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  // binding for kgen_threadFlags
  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  // binding for SomeInterfaceObjPointerData
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  // binding for kgen_objData
  dsBindings[3].binding            = 3;
  dsBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[3].descriptorCount    = 1;
  dsBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3].pImmutableSamplers = nullptr;
  
  // binding for m_classDataBuffer 
  dsBindings[3+1].binding            = 3+1;
  dsBindings[3+1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[3+1].descriptorCount    = 1;
  dsBindings[3+1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3+1].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
VkDescriptorSetLayout TestClass_Generated::CreateNextBounceDSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 12+2> dsBindings;

  // binding for in_hit
  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  // binding for in_bars
  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  // binding for in_indices
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  // binding for in_vpos
  dsBindings[3].binding            = 3;
  dsBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[3].descriptorCount    = 1;
  dsBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3].pImmutableSamplers = nullptr;

  // binding for in_vnorm
  dsBindings[4].binding            = 4;
  dsBindings[4].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[4].descriptorCount    = 1;
  dsBindings[4].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[4].pImmutableSamplers = nullptr;

  // binding for rayPosAndNear
  dsBindings[5].binding            = 5;
  dsBindings[5].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[5].descriptorCount    = 1;
  dsBindings[5].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[5].pImmutableSamplers = nullptr;

  // binding for rayDirAndFar
  dsBindings[6].binding            = 6;
  dsBindings[6].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[6].descriptorCount    = 1;
  dsBindings[6].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[6].pImmutableSamplers = nullptr;

  // binding for pGen
  dsBindings[7].binding            = 7;
  dsBindings[7].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[7].descriptorCount    = 1;
  dsBindings[7].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[7].pImmutableSamplers = nullptr;

  // binding for accumColor
  dsBindings[8].binding            = 8;
  dsBindings[8].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[8].descriptorCount    = 1;
  dsBindings[8].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[8].pImmutableSamplers = nullptr;

  // binding for accumThoroughput
  dsBindings[9].binding            = 9;
  dsBindings[9].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[9].descriptorCount    = 1;
  dsBindings[9].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[9].pImmutableSamplers = nullptr;

  // binding for kgen_threadFlags
  dsBindings[10].binding            = 10;
  dsBindings[10].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[10].descriptorCount    = 1;
  dsBindings[10].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[10].pImmutableSamplers = nullptr;

  // binding for SomeInterfaceObjPointerData
  dsBindings[11].binding            = 11;
  dsBindings[11].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[11].descriptorCount    = 1;
  dsBindings[11].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[11].pImmutableSamplers = nullptr;

  // binding for kgen_objData
  dsBindings[12].binding            = 12;
  dsBindings[12].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[12].descriptorCount    = 1;
  dsBindings[12].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[12].pImmutableSamplers = nullptr;
  
  // binding for m_classDataBuffer 
  dsBindings[12+1].binding            = 12+1;
  dsBindings[12+1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[12+1].descriptorCount    = 1;
  dsBindings[12+1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[12+1].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
VkDescriptorSetLayout TestClass_Generated::CreateInitEyeRayDSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 4+1> dsBindings;

  // binding for packedXY
  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  // binding for rayPosAndNear
  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  // binding for rayDirAndFar
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  // binding for kgen_threadFlags
  dsBindings[3].binding            = 3;
  dsBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[3].descriptorCount    = 1;
  dsBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3].pImmutableSamplers = nullptr;

  // binding for POD members stored in m_classDataBuffer
  dsBindings[4].binding            = 4;
  dsBindings[4].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[4].descriptorCount    = 1;
  dsBindings[4].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[4].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
VkDescriptorSetLayout TestClass_Generated::CreateInitEyeRay2DSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 6+1> dsBindings;

  // binding for packedXY
  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  // binding for rayPosAndNear
  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  // binding for rayDirAndFar
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  // binding for accumColor
  dsBindings[3].binding            = 3;
  dsBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[3].descriptorCount    = 1;
  dsBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3].pImmutableSamplers = nullptr;

  // binding for accumuThoroughput
  dsBindings[4].binding            = 4;
  dsBindings[4].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[4].descriptorCount    = 1;
  dsBindings[4].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[4].pImmutableSamplers = nullptr;

  // binding for kgen_threadFlags
  dsBindings[5].binding            = 5;
  dsBindings[5].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[5].descriptorCount    = 1;
  dsBindings[5].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[5].pImmutableSamplers = nullptr;

  // binding for POD members stored in m_classDataBuffer
  dsBindings[6].binding            = 6;
  dsBindings[6].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[6].descriptorCount    = 1;
  dsBindings[6].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[6].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
VkDescriptorSetLayout TestClass_Generated::CreateRayTraceDSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 9+1> dsBindings;

  // binding for rayPosAndNear
  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  // binding for rayDirAndFar
  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  // binding for out_hit
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  // binding for out_bars
  dsBindings[3].binding            = 3;
  dsBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[3].descriptorCount    = 1;
  dsBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3].pImmutableSamplers = nullptr;

  // binding for kgen_threadFlags
  dsBindings[4].binding            = 4;
  dsBindings[4].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[4].descriptorCount    = 1;
  dsBindings[4].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[4].pImmutableSamplers = nullptr;

  // binding for m_vPos4f
  dsBindings[5].binding            = 5;
  dsBindings[5].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[5].descriptorCount    = 1;
  dsBindings[5].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[5].pImmutableSamplers = nullptr;

  // binding for m_indicesReordered
  dsBindings[6].binding            = 6;
  dsBindings[6].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[6].descriptorCount    = 1;
  dsBindings[6].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[6].pImmutableSamplers = nullptr;

  // binding for m_intervals
  dsBindings[7].binding            = 7;
  dsBindings[7].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[7].descriptorCount    = 1;
  dsBindings[7].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[7].pImmutableSamplers = nullptr;

  // binding for m_nodes
  dsBindings[8].binding            = 8;
  dsBindings[8].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[8].descriptorCount    = 1;
  dsBindings[8].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[8].pImmutableSamplers = nullptr;

  // binding for POD members stored in m_classDataBuffer
  dsBindings[9].binding            = 9;
  dsBindings[9].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[9].descriptorCount    = 1;
  dsBindings[9].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[9].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
VkDescriptorSetLayout TestClass_Generated::CreateMakeMaterialDSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 6+1> dsBindings;

  // binding for in_hit
  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  // binding for kgen_threadFlags
  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  // binding for m_materialData
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  // binding for m_materialIds
  dsBindings[3].binding            = 3;
  dsBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[3].descriptorCount    = 1;
  dsBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3].pImmutableSamplers = nullptr;

  // binding for m_materialOffsets
  dsBindings[4].binding            = 4;
  dsBindings[4].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[4].descriptorCount    = 1;
  dsBindings[4].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[4].pImmutableSamplers = nullptr;

  // binding for SomeInterfaceObjPointerData
  dsBindings[5].binding            = 5;
  dsBindings[5].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[5].descriptorCount    = 1;
  dsBindings[5].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[5].pImmutableSamplers = nullptr;

  // binding for POD members stored in m_classDataBuffer
  dsBindings[6].binding            = 6;
  dsBindings[6].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[6].descriptorCount    = 1;
  dsBindings[6].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[6].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
VkDescriptorSetLayout TestClass_Generated::CreatePackXYDSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 1+1> dsBindings;

  // binding for out_pakedXY
  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  // binding for POD members stored in m_classDataBuffer
  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
VkDescriptorSetLayout TestClass_Generated::CreateContributeToImageDSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 4+1> dsBindings;

  // binding for a_accumColor
  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  // binding for in_pakedXY
  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  // binding for out_color
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  // binding for kgen_threadFlags
  dsBindings[3].binding            = 3;
  dsBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[3].descriptorCount    = 1;
  dsBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3].pImmutableSamplers = nullptr;

  // binding for POD members stored in m_classDataBuffer
  dsBindings[4].binding            = 4;
  dsBindings[4].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[4].descriptorCount    = 1;
  dsBindings[4].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[4].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

VkDescriptorSetLayout TestClass_Generated::CreatecopyKernelFloatDSLayout()
{
  std::array<VkDescriptorSetLayoutBinding, 2> dsBindings;

  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = dsBindings.size();
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

VkBufferMemoryBarrier TestClass_Generated::BarrierForObjCounters(VkBuffer a_buffer)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = a_buffer;
  bar.offset              = 0;
  bar.size                = VK_WHOLE_SIZE; // TODO: count offset and size carefully, actually we can do this!
  return bar;
}

void TestClass_Generated::InitKernel_GetColor(const char* a_filePath)
{
  std::string shaderPath = a_filePath; 
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "kernel_GetColor");
  GetColorDSLayout = CreateGetColorDSLayout();
  GetColorLayout   = m_pMaker->MakeLayout(device, GetColorDSLayout, 128); // at least 128 bytes for push constants
  GetColorPipeline = m_pMaker->MakePipeline(device);  
}

void TestClass_Generated::InitKernel_NextBounce(const char* a_filePath)
{
  std::string shaderPath = a_filePath; 
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "kernel_NextBounce");
  NextBounceDSLayout = CreateNextBounceDSLayout();
  NextBounceLayout   = m_pMaker->MakeLayout(device, NextBounceDSLayout, 128); // at least 128 bytes for push constants
  NextBouncePipeline = m_pMaker->MakePipeline(device);  
}

void TestClass_Generated::InitKernel_InitEyeRay(const char* a_filePath)
{
  std::string shaderPath = a_filePath; 
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "kernel_InitEyeRay");
  InitEyeRayDSLayout = CreateInitEyeRayDSLayout();
  InitEyeRayLayout   = m_pMaker->MakeLayout(device, InitEyeRayDSLayout, 128); // at least 128 bytes for push constants
  InitEyeRayPipeline = m_pMaker->MakePipeline(device);  
}

void TestClass_Generated::InitKernel_InitEyeRay2(const char* a_filePath)
{
  std::string shaderPath = a_filePath; 
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "kernel_InitEyeRay2");
  InitEyeRay2DSLayout = CreateInitEyeRay2DSLayout();
  InitEyeRay2Layout   = m_pMaker->MakeLayout(device, InitEyeRay2DSLayout, 128); // at least 128 bytes for push constants
  InitEyeRay2Pipeline = m_pMaker->MakePipeline(device);  
}

void TestClass_Generated::InitKernel_RayTrace(const char* a_filePath)
{
  std::string shaderPath = a_filePath; 
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "kernel_RayTrace");
  RayTraceDSLayout = CreateRayTraceDSLayout();
  RayTraceLayout   = m_pMaker->MakeLayout(device, RayTraceDSLayout, 128); // at least 128 bytes for push constants
  RayTracePipeline = m_pMaker->MakePipeline(device);  
}

void TestClass_Generated::InitKernel_MakeMaterial(const char* a_filePath)
{
  std::string shaderPath = a_filePath; 
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "kernel_MakeMaterial");
  MakeMaterialDSLayout = CreateMakeMaterialDSLayout();
  MakeMaterialLayout   = m_pMaker->MakeLayout(device, MakeMaterialDSLayout, 128); // at least 128 bytes for push constants
  MakeMaterialPipeline = m_pMaker->MakePipeline(device);  
}

void TestClass_Generated::InitKernel_PackXY(const char* a_filePath)
{
  std::string shaderPath = a_filePath; 
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "kernel_PackXY");
  PackXYDSLayout = CreatePackXYDSLayout();
  PackXYLayout   = m_pMaker->MakeLayout(device, PackXYDSLayout, 128); // at least 128 bytes for push constants
  PackXYPipeline = m_pMaker->MakePipeline(device);  
}

void TestClass_Generated::InitKernel_ContributeToImage(const char* a_filePath)
{
  std::string shaderPath = a_filePath; 
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "kernel_ContributeToImage");
  ContributeToImageDSLayout = CreateContributeToImageDSLayout();
  ContributeToImageLayout   = m_pMaker->MakeLayout(device, ContributeToImageDSLayout, 128); // at least 128 bytes for push constants
  ContributeToImagePipeline = m_pMaker->MakePipeline(device);  
}


void TestClass_Generated::InitKernels(const char* a_filePath)
{
  InitKernel_GetColor(a_filePath);
  InitKernel_NextBounce(a_filePath);
  InitKernel_InitEyeRay(a_filePath);
  InitKernel_InitEyeRay2(a_filePath);
  InitKernel_RayTrace(a_filePath);
  InitKernel_MakeMaterial(a_filePath);
  InitKernel_PackXY(a_filePath);
  InitKernel_ContributeToImage(a_filePath);

  std::string servPath = a_filePath;
  
  m_pMaker->CreateShader(device, servPath.c_str(), nullptr, "copyKernelFloat");
  copyKernelFloatDSLayout = CreatecopyKernelFloatDSLayout();
  copyKernelFloatLayout   = m_pMaker->MakeLayout(device, copyKernelFloatDSLayout, 128); // at least 128 bytes for push constants
  copyKernelFloatPipeline = m_pMaker->MakePipeline(device);

}

void TestClass_Generated::InitBuffers(size_t a_maxThreadsCount)
{
  m_maxThreadCount = a_maxThreadsCount;
  std::vector<VkBuffer> allBuffers;

  NaivePathTrace_local.hitBuffer = vkfw::CreateBuffer(device, sizeof(Lite_Hit)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(NaivePathTrace_local.hitBuffer);
  NaivePathTrace_local.accumThoroughputBuffer = vkfw::CreateBuffer(device, sizeof(struct LiteMath::float4)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(NaivePathTrace_local.accumThoroughputBuffer);
  NaivePathTrace_local.accumColorBuffer = vkfw::CreateBuffer(device, sizeof(struct LiteMath::float4)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(NaivePathTrace_local.accumColorBuffer);
  NaivePathTrace_local.threadFlagsBuffer = vkfw::CreateBuffer(device, sizeof(uint)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffers.push_back(NaivePathTrace_local.threadFlagsBuffer);
  NaivePathTrace_local.baricentricsBuffer = vkfw::CreateBuffer(device, sizeof(struct LiteMath::float2)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(NaivePathTrace_local.baricentricsBuffer);
  NaivePathTrace_local.rayDirAndFarBuffer = vkfw::CreateBuffer(device, sizeof(struct LiteMath::float4)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(NaivePathTrace_local.rayDirAndFarBuffer);
  NaivePathTrace_local.rayPosAndNearBuffer = vkfw::CreateBuffer(device, sizeof(struct LiteMath::float4)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(NaivePathTrace_local.rayPosAndNearBuffer);
  CastSingleRay_local.hitBuffer = vkfw::CreateBuffer(device, sizeof(Lite_Hit)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(CastSingleRay_local.hitBuffer);
  CastSingleRay_local.threadFlagsBuffer = vkfw::CreateBuffer(device, sizeof(uint)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffers.push_back(CastSingleRay_local.threadFlagsBuffer);
  CastSingleRay_local.baricentricsBuffer = vkfw::CreateBuffer(device, sizeof(struct LiteMath::float2)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(CastSingleRay_local.baricentricsBuffer);
  CastSingleRay_local.rayDirAndFarBuffer = vkfw::CreateBuffer(device, sizeof(struct LiteMath::float4)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(CastSingleRay_local.rayDirAndFarBuffer);
  CastSingleRay_local.rayPosAndNearBuffer = vkfw::CreateBuffer(device, sizeof(struct LiteMath::float4)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(CastSingleRay_local.rayPosAndNearBuffer);

  m_classDataBuffer = vkfw::CreateBuffer(device, sizeof(m_uboData),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | GetAdditionalFlagsForUBO());
  allBuffers.push_back(m_classDataBuffer);
  m_IMaterialObjPtrBuffer = vkfw::CreateBuffer(device, 2*sizeof(uint32_t)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(m_IMaterialObjPtrBuffer);

  if(allBuffers.size() > 0)
    m_allMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, allBuffers);
}

void TestClass_Generated::InitMemberBuffers()
{
  std::vector<VkBuffer> memberVectors;
  m_vdata.m_randomGensBuffer = vkfw::CreateBuffer(device, m_randomGens.capacity()*sizeof(struct RandomGenT), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_randomGensBuffer);
  m_vdata.m_vNorm4fBuffer = vkfw::CreateBuffer(device, m_vNorm4f.capacity()*sizeof(struct LiteMath::float4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_vNorm4fBuffer);
  m_vdata.m_nodesBuffer = vkfw::CreateBuffer(device, m_nodes.capacity()*sizeof(struct BVHNode), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_nodesBuffer);
  m_vdata.m_intervalsBuffer = vkfw::CreateBuffer(device, m_intervals.capacity()*sizeof(struct Interval), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_intervalsBuffer);
  m_vdata.m_materialIdsBuffer = vkfw::CreateBuffer(device, m_materialIds.capacity()*sizeof(unsigned int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_materialIdsBuffer);
  m_vdata.m_indicesReorderedBuffer = vkfw::CreateBuffer(device, m_indicesReordered.capacity()*sizeof(unsigned int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_indicesReorderedBuffer);
  m_vdata.m_materialOffsetsBuffer = vkfw::CreateBuffer(device, m_materialOffsets.capacity()*sizeof(unsigned int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_materialOffsetsBuffer);
  m_vdata.m_vPos4fBuffer = vkfw::CreateBuffer(device, m_vPos4f.capacity()*sizeof(struct LiteMath::float4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_vPos4fBuffer);
  m_vdata.m_materialDataBuffer = vkfw::CreateBuffer(device, m_materialData.capacity()*sizeof(unsigned int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_materialDataBuffer);
  
  if(memberVectors.size() > 0)
    m_vdata.m_vecMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, memberVectors);
}



