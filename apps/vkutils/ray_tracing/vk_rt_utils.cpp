#include "vk_rt_utils.h"
#include "vk_utils.h"
#include "vk_rt_funcs.h"

namespace vk_rt_utils
{

  VkTransformMatrixKHR transformMatrixFromRowMajArray(const std::array<float, 16> &m)
  {
    VkTransformMatrixKHR transformMatrix;
    for(int i = 0; i < 3; ++i)
    {
      for(int j = 0; j < 4; ++j)
      {
        transformMatrix.matrix[i][j] = m[i * 4 + j];
      }
    }

    return transformMatrix;
  }

  uint64_t getBufferDeviceAddress(VkDevice a_device, VkBuffer a_buffer)
  {
    VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
    bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddressInfo.buffer = a_buffer;
    return vkGetBufferDeviceAddressKHR(a_device, &bufferDeviceAddressInfo);
  }

  RTScratchBuffer allocScratchBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, VkDeviceSize size)
  {
    RTScratchBuffer scratchBuffer{};

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, nullptr, &scratchBuffer.buffer));

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(a_device, scratchBuffer.buffer, &memoryRequirements);

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDevice);

    VK_CHECK_RESULT(vkAllocateMemory(a_device, &memoryAllocateInfo, nullptr, &scratchBuffer.memory));
    VK_CHECK_RESULT(vkBindBufferMemory(a_device, scratchBuffer.buffer, scratchBuffer.memory, 0));

    VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
    bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddressInfo.buffer = scratchBuffer.buffer;
    scratchBuffer.deviceAddress = vkGetBufferDeviceAddressKHR(a_device, &bufferDeviceAddressInfo);

    return scratchBuffer;
  }

  void createAccelerationStructure(AccelStructure& accel, VkAccelerationStructureTypeKHR type,
    VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo, VkDevice a_device, VkPhysicalDevice a_physicalDevice)
  {
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = buildSizeInfo.accelerationStructureSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, nullptr, &accel.buffer));

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(a_device, accel.buffer, &memoryRequirements);
    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physicalDevice);

    VK_CHECK_RESULT(vkAllocateMemory(a_device, &memoryAllocateInfo, nullptr, &accel.memory));
    VK_CHECK_RESULT(vkBindBufferMemory(a_device, accel.buffer, accel.memory, 0));

    // Acceleration structure
    VkAccelerationStructureCreateInfoKHR accelerationStructureCreate_info{};
    accelerationStructureCreate_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    accelerationStructureCreate_info.buffer = accel.buffer;
    accelerationStructureCreate_info.size = buildSizeInfo.accelerationStructureSize;
    accelerationStructureCreate_info.type = type;
    vkCreateAccelerationStructureKHR(a_device, &accelerationStructureCreate_info, nullptr, &accel.handle);

    // AS device address
    VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
    accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationDeviceAddressInfo.accelerationStructure = accel.handle;
    accel.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(a_device, &accelerationDeviceAddressInfo);
  }

  VkStridedDeviceAddressRegionKHR getSBTStridedDeviceAddressRegion(VkDevice a_device, VkBuffer buffer,
                                                                   uint32_t handleCount, uint32_t handleSizeAligned)
  {
    VkStridedDeviceAddressRegionKHR stridedDeviceAddressRegionKHR{};
    stridedDeviceAddressRegionKHR.deviceAddress = getBufferDeviceAddress(a_device, buffer);;
    stridedDeviceAddressRegionKHR.stride = handleSizeAligned;
    stridedDeviceAddressRegionKHR.size = handleCount * handleSizeAligned;

    return stridedDeviceAddressRegionKHR;
  }

  VkPipelineLayout RTPipelineMaker::MakeLayout(VkDevice a_device, VkDescriptorSetLayout a_dslayout)
  {
    VkPipelineLayoutCreateInfo layoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutCreateInfo.pSetLayouts    = &a_dslayout;
    layoutCreateInfo.setLayoutCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &layoutCreateInfo, nullptr, &m_pipelineLayout));

    return m_pipelineLayout;
  }

  VkPipelineLayout  RTPipelineMaker::MakeLayout(VkDevice a_device, std::vector<VkDescriptorSetLayout> a_dslayouts)
  {
    VkPipelineLayoutCreateInfo layoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutCreateInfo.pSetLayouts    = a_dslayouts.data();
    layoutCreateInfo.setLayoutCount = a_dslayouts.size();
    VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &layoutCreateInfo, nullptr, &m_pipelineLayout));

    return m_pipelineLayout;
  }

  void RTPipelineMaker::LoadShaders(VkDevice a_device,  const std::vector<std::pair<VkShaderStageFlagBits, std::string>> &shader_paths)
  {
    for(auto& [stage, path] : shader_paths)
    {
      VkPipelineShaderStageCreateInfo shaderStage = {};
      shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shaderStage.stage = stage;

      auto shaderCode             = vk_utils::readSPVFile(path.c_str());
      shaderModules.push_back(vk_utils::createShaderModule(a_device, shaderCode));
      shaderStage.module = shaderModules.back();

      shaderStage.pName = "main";
      assert(shaderStage.module != VK_NULL_HANDLE);
      shaderStages.push_back(shaderStage);

      VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
      shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;

      if(stage == VK_SHADER_STAGE_MISS_BIT_KHR ||
      stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR ||
      stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR)
      {
        shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shaderGroup.generalShader = static_cast<uint32_t>(shaderStages.size()) - 1;
        shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
      }
      else if(stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
      {
        shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        shaderGroup.generalShader = VK_SHADER_UNUSED_KHR;
        shaderGroup.closestHitShader = static_cast<uint32_t>(shaderStages.size()) - 1;
      }
      // @TODO: intersection, procedural, anyhit
      shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
      shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;

      shaderGroups.push_back(shaderGroup);
    }
  }

  VkPipeline RTPipelineMaker::MakePipeline(VkDevice a_device)
  {
    VkRayTracingPipelineCreateInfoKHR createInfo {VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    createInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    createInfo.pStages = shaderStages.data();
    createInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    createInfo.pGroups = shaderGroups.data();
    createInfo.maxPipelineRayRecursionDepth = 2;
    createInfo.layout = m_pipelineLayout;
    VK_CHECK_RESULT(vkCreateRayTracingPipelinesKHR(a_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_pipeline));

    return m_pipeline;
  }

}