#include <vector>
#include <array>
#include <memory>
#include <limits>

#include <cassert>
#include "vk_copy.h"
#include "vk_context.h"

#include "{{IncludeClassDecl}}"

{% if HasRTXAccelStruct or length(IntersectionHierarhcy.Implementations) >= 1 %}
#include "VulkanRTX.h"
{% endif %}

void {{MainClassName}}{{MainClassSuffix}}::AllocateAllDescriptorSets()
{
  // allocate pool
  //
  VkDescriptorPoolSize buffersSize, combinedImageSamSize, imageStorageSize, accelStorageSize, dynamicBuffersSize;
  buffersSize.type                     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  buffersSize.descriptorCount          = {{TotalBuffersUsed}} + 64; // + 64 for reserve

  std::vector<VkDescriptorPoolSize> poolSizes = {buffersSize};
  {
    combinedImageSamSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    combinedImageSamSize.descriptorCount = {{TotalTexArrayUsed}}*GetDefaultMaxTextures() + {{TotalTexCombinedUsed}};
    imageStorageSize.type                = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageStorageSize.descriptorCount     = {{TotalTexStorageUsed}};
    accelStorageSize.type                = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    accelStorageSize.descriptorCount     = {% if TotalAccels > 0 %} {{TotalAccels+1}} {% else %} 0 {% endif %};

    if(combinedImageSamSize.descriptorCount > 0)
      poolSizes.push_back(combinedImageSamSize);
    if(imageStorageSize.descriptorCount > 0)
      poolSizes.push_back(imageStorageSize);
    if(accelStorageSize.descriptorCount > 0)
      poolSizes.push_back(accelStorageSize);
    
    {% if HaveLocalContainers %}
    dynamicBuffersSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    dynamicBuffersSize.descriptorCount = 16;
    poolSizes.push_back(dynamicBuffersSize);
    {% endif %}
  }

  {% if UniformUBO %}
  VkDescriptorPoolSize uboSize = {};
  uboSize.type             = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboSize.descriptorCount  = {{TotalKernels}} + 4; // + 4 for reserve
  poolSizes.push_back(uboSize);
  {% endif %}

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.maxSets       = {{TotalDSNumber}} + 2; // add 1 to prevent zero case and one more for internal needs
  descriptorPoolCreateInfo.poolSizeCount = poolSizes.size();
  descriptorPoolCreateInfo.pPoolSizes    = poolSizes.data();

  VK_CHECK_RESULT(vkCreateDescriptorPool(m_device, &descriptorPoolCreateInfo, NULL, &m_dsPool));

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

  auto tmpRes = vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, m_allGeneratedDS);
  VK_CHECK_RESULT(tmpRes);
}

## for Kernel in Kernels
VkDescriptorSetLayout {{MainClassName}}{{MainClassSuffix}}::Create{{Kernel.Name}}DSLayout()
{
  {% if UseSeparateUBO %}
  std::array<VkDescriptorSetLayoutBinding, {{Kernel.ArgCount}}+2> dsBindings;
  {% else %}
  std::array<VkDescriptorSetLayoutBinding, {{Kernel.ArgCount}}+1> dsBindings;
  {% endif %}
  
  {% if Kernel.UseRayGen %}
  const auto stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR;
  {% else %}
  const auto stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  {% endif %}

## for KernelARG in Kernel.Args
  // binding for {{KernelARG.Name}}
  dsBindings[{{KernelARG.Id}}].binding            = {{loop.index}};
  dsBindings[{{KernelARG.Id}}].descriptorType     = {{KernelARG.Type}};
  {% if KernelARG.IsTextureArray %}
  m_vdata.{{KernelARG.Name}}ArrayMaxSize = {{KernelARG.Count}};
  if(m_vdata.{{KernelARG.Name}}ArrayMaxSize == 0)
    m_vdata.{{KernelARG.Name}}ArrayMaxSize = GetDefaultMaxTextures();
  dsBindings[{{KernelARG.Id}}].descriptorCount    = m_vdata.{{KernelARG.Name}}ArrayMaxSize;
  {% else %}
  dsBindings[{{KernelARG.Id}}].descriptorCount    = {{KernelARG.Count}};
  {% endif %}
  dsBindings[{{KernelARG.Id}}].stageFlags         = stageFlags;
  dsBindings[{{KernelARG.Id}}].pImmutableSamplers = nullptr;

## endfor

  dsBindings[{{Kernel.ArgCount}}].binding            = {{Kernel.ArgCount}};
  dsBindings[{{Kernel.ArgCount}}].descriptorType     = {% if UniformUBO %} VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER {% else %} VK_DESCRIPTOR_TYPE_STORAGE_BUFFER {% endif %};
  dsBindings[{{Kernel.ArgCount}}].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}].stageFlags         = stageFlags;
  dsBindings[{{Kernel.ArgCount}}].pImmutableSamplers = nullptr;

  {% if UseSeparateUBO %}

  // binding for {% if UseSeparateUBO%}separate ubo{% else %}m_classDataBuffer {% endif %}

  dsBindings[{{Kernel.ArgCount}}+1].binding            = {{Kernel.ArgCount}}+1;
  dsBindings[{{Kernel.ArgCount}}+1].descriptorType     = {% if UseSeparateUBO %}VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER{% else %}VK_DESCRIPTOR_TYPE_STORAGE_BUFFER{% endif %};
  dsBindings[{{Kernel.ArgCount}}+1].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}+1].stageFlags         = stageFlags;
  dsBindings[{{Kernel.ArgCount}}+1].pImmutableSamplers = nullptr;
  {% else %}
  {% endif %}

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
## endfor

VkDescriptorSetLayout {{MainClassName}}{{MainClassSuffix}}::CreatecopyKernelFloatDSLayout()
{
  {% if UseSpecConstWgSize %}
  std::array<VkDescriptorSetLayoutBinding, 3> dsBindings;
  {% else %}
  std::array<VkDescriptorSetLayoutBinding, 2> dsBindings;
  {% endif %}

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
  {% if UseSpecConstWgSize %}

  // binding for POD arguments
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;
  {% endif %}

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = dsBindings.size();
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

VkDescriptorSetLayout {{MainClassName}}{{MainClassSuffix}}::CreatematMulTransposeDSLayout()
{
  {% if UseSpecConstWgSize %}
  std::array<VkDescriptorSetLayoutBinding, 4> dsBindings;
  {% else %}
  std::array<VkDescriptorSetLayoutBinding, 3> dsBindings;
  {% endif %}

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

  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;
  {% if UseSpecConstWgSize %}

  // binding for POD arguments
  dsBindings[3].binding            = 3;
  dsBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  dsBindings[3].descriptorCount    = 1;
  dsBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[3].pImmutableSamplers = nullptr;
  {% endif %}

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = dsBindings.size();
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

## for MainFunc in MainFunctions
void {{MainClassName}}{{MainClassSuffix}}::UpdateAllGeneratedDescriptorSets_{{MainFunc.Name}}()
{
  // now create actual bindings
  //
## for DescriptorSet in MainFunc.DescriptorSets
  // descriptor set #{{DescriptorSet.Id}}: {{DescriptorSet.KernelName}}Cmd ({{DescriptorSet.ArgNames}})
  {
    {% if not DescriptorSet.IsServiceCall and UseSeparateUBO and DescriptorSet.IsVirtual %}
    constexpr uint additionalSize = 3;
    {% else if (not DescriptorSet.IsServiceCall and UseSeparateUBO) or (not DescriptorSet.IsServiceCall and DescriptorSet.IsVirtual) %}
    constexpr uint additionalSize = 2;
    {% else if not DescriptorSet.IsServiceCall or UseSeparateUBO %}
    constexpr uint additionalSize = 1;
    {% else %}
    constexpr uint additionalSize = 0;
    {% endif %}

    std::array<VkDescriptorBufferInfo, {{DescriptorSet.ArgNumber}} + additionalSize> descriptorBufferInfo;
    std::array<VkDescriptorImageInfo,  {{DescriptorSet.ArgNumber}} + additionalSize> descriptorImageInfo;
    {% if HasRTXAccelStruct %}
    std::array<VkAccelerationStructureKHR,  {{DescriptorSet.ArgNumber}} + additionalSize> accelStructs;
    std::array<VkWriteDescriptorSetAccelerationStructureKHR,  {{DescriptorSet.ArgNumber}} + additionalSize> descriptorAccelInfo;
    {% endif %}
    std::array<VkWriteDescriptorSet,   {{DescriptorSet.ArgNumber}} + additionalSize> writeDescriptorSet;

## for Arg in DescriptorSet.Args
    {% if Arg.IsTexture %}
    descriptorImageInfo[{{Arg.Id}}].imageView   = {{Arg.Name}}View;
    descriptorImageInfo[{{Arg.Id}}].imageLayout = {{Arg.AccessLayout}};
    descriptorImageInfo[{{Arg.Id}}].sampler     = {{Arg.SamplerName}};
    {% else if Arg.IsTextureArray %}
    std::vector<VkDescriptorImageInfo> {{Arg.NameOriginal}}Info(m_vdata.{{Arg.NameOriginal}}ArrayMaxSize);
    for(size_t i=0; i<m_vdata.{{Arg.NameOriginal}}ArrayMaxSize; i++)
    {
      if(i < {{Arg.NameOriginal}}.size())
      {
        {{Arg.NameOriginal}}Info[i].sampler     = m_vdata.{{Arg.NameOriginal}}ArraySampler[i];
        {{Arg.NameOriginal}}Info[i].imageView   = m_vdata.{{Arg.NameOriginal}}ArrayView   [i];
        {{Arg.NameOriginal}}Info[i].imageLayout = {{Arg.AccessLayout}};
      }
      else
      {
        {{Arg.NameOriginal}}Info[i].sampler     = m_vdata.{{Arg.NameOriginal}}ArraySampler[0];
        {{Arg.NameOriginal}}Info[i].imageView   = m_vdata.{{Arg.NameOriginal}}ArrayView   [0];
        {{Arg.NameOriginal}}Info[i].imageLayout = {{Arg.AccessLayout}};
      }
    }
    {% else if Arg.IsAccelStruct %}
    {
      VulkanRTX* pScene = dynamic_cast<VulkanRTX*>({{Arg.Name}}->UnderlyingImpl(1));
      if(pScene == nullptr)
        std::cout << "[{{MainClassName}}{{MainClassSuffix}}::InitAllGeneratedDescriptorSets_{{MainFunc.Name}}]: fatal error, wrong accel struct type" << std::endl;
      accelStructs       [{{Arg.Id}}] = pScene->GetSceneAccelStruct();
      descriptorAccelInfo[{{Arg.Id}}] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,VK_NULL_HANDLE,1,&accelStructs[{{Arg.Id}}]};
    }
    {% else %}
    descriptorBufferInfo[{{Arg.Id}}]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[{{Arg.Id}}].buffer = {{Arg.Name}}Buffer;
    descriptorBufferInfo[{{Arg.Id}}].offset = {{Arg.Name}}Offset;
    descriptorBufferInfo[{{Arg.Id}}].range  = VK_WHOLE_SIZE;
    {% endif %}
    writeDescriptorSet[{{Arg.Id}}]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[{{Arg.Id}}].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[{{Arg.Id}}].dstSet           = m_allGeneratedDS[{{DescriptorSet.Id}}];
    writeDescriptorSet[{{Arg.Id}}].dstBinding       = {{Arg.Id}};
    writeDescriptorSet[{{Arg.Id}}].descriptorCount  = 1;
    {% if Arg.IsTexture %}
    writeDescriptorSet[{{Arg.Id}}].descriptorType   = {{Arg.AccessDSType}};
    writeDescriptorSet[{{Arg.Id}}].pBufferInfo      = nullptr;
    writeDescriptorSet[{{Arg.Id}}].pImageInfo       = &descriptorImageInfo[{{Arg.Id}}];
    writeDescriptorSet[{{Arg.Id}}].pTexelBufferView = nullptr;
    {% else if Arg.IsTextureArray %}
    writeDescriptorSet[{{Arg.Id}}].descriptorCount  = {{Arg.NameOriginal}}Info.size();
    writeDescriptorSet[{{Arg.Id}}].descriptorType   = {{Arg.AccessDSType}};
    writeDescriptorSet[{{Arg.Id}}].pBufferInfo      = nullptr;
    writeDescriptorSet[{{Arg.Id}}].pImageInfo       = {{Arg.NameOriginal}}Info.data();
    writeDescriptorSet[{{Arg.Id}}].pTexelBufferView = nullptr;
    {% else if Arg.IsAccelStruct %}
    writeDescriptorSet[{{Arg.Id}}].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writeDescriptorSet[{{Arg.Id}}].pNext          = &descriptorAccelInfo[{{Arg.Id}}];
    {% else %}
    {% if Arg.Name == "m_vdata.localTemp" %}
    writeDescriptorSet[{{Arg.Id}}].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    {% else %}
    writeDescriptorSet[{{Arg.Id}}].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    {% endif %}
    writeDescriptorSet[{{Arg.Id}}].pBufferInfo      = &descriptorBufferInfo[{{Arg.Id}}];
    writeDescriptorSet[{{Arg.Id}}].pImageInfo       = nullptr;
    writeDescriptorSet[{{Arg.Id}}].pTexelBufferView = nullptr;
    {% endif %}

## endfor
    {% if not DescriptorSet.IsServiceCall %}
    {% if DescriptorSet.IsVirtual %}
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].buffer = m_vdata.{{DescriptorSet.ObjectBufferName}}Buffer;
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

    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].buffer = m_classDataBuffer;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].offset = 0;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].range  = VK_WHOLE_SIZE;

    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1].dstSet           = m_allGeneratedDS[{{DescriptorSet.Id}}];
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1].dstBinding       = {{DescriptorSet.ArgNumber}}+1;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1].descriptorCount  = 1;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1].pBufferInfo      = &descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1];
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1].pImageInfo       = nullptr;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}+1].pTexelBufferView = nullptr;

    {% else %}
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].buffer = m_classDataBuffer;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].offset = 0;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].range  = VK_WHOLE_SIZE;

    writeDescriptorSet[{{DescriptorSet.ArgNumber}}]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].dstSet           = m_allGeneratedDS[{{DescriptorSet.Id}}];
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].dstBinding       = {{DescriptorSet.ArgNumber}};
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].descriptorCount  = 1;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].descriptorType   = {% if UniformUBO %} VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER {% else %} VK_DESCRIPTOR_TYPE_STORAGE_BUFFER {% endif %};
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].pBufferInfo      = &descriptorBufferInfo[{{DescriptorSet.ArgNumber}}];
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].pImageInfo       = nullptr;
    writeDescriptorSet[{{DescriptorSet.ArgNumber}}].pTexelBufferView = nullptr;

    {% endif %} {#/*  DescriptorSet.IsVirtual */#}
    {% endif %} {#/* not DescriptorSet.IsServiceCall */#}
    {% if UseSeparateUBO %}
    const size_t uboId = descriptorBufferInfo.size()-1;
    descriptorBufferInfo[uboId]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[uboId].buffer = m_uboArgsBuffer;
    descriptorBufferInfo[uboId].offset = 0;
    descriptorBufferInfo[uboId].range  = VK_WHOLE_SIZE;

    writeDescriptorSet[uboId]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[uboId].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[uboId].dstSet           = m_allGeneratedDS[{{DescriptorSet.Id}}];
    writeDescriptorSet[uboId].dstBinding       = uboId;
    writeDescriptorSet[uboId].descriptorCount  = 1;
    writeDescriptorSet[uboId].descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeDescriptorSet[uboId].pBufferInfo      = &descriptorBufferInfo[uboId];
    writeDescriptorSet[uboId].pImageInfo       = nullptr;
    writeDescriptorSet[uboId].pTexelBufferView = nullptr;

    {% endif %}
    vkUpdateDescriptorSets(m_device, uint32_t(writeDescriptorSet.size()), writeDescriptorSet.data(), 0, NULL);
  }
## endfor
}

## endfor

{% if length(IndirectDispatches) > 0 %}
void {{MainClassName}}{{MainClassSuffix}}::InitIndirectDescriptorSets()
{
  if(m_indirectUpdateDS != VK_NULL_HANDLE)
    return;

  // (m_classDataBuffer, m_indirectBuffer) ==> m_indirectUpdateDS
  //
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.descriptorPool     = m_dsPool;
  descriptorSetAllocateInfo.descriptorSetCount = 1;
  descriptorSetAllocateInfo.pSetLayouts        = &m_indirectUpdateDSLayout;

  auto tmpRes = vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, &m_indirectUpdateDS);
  VK_CHECK_RESULT(tmpRes);

  VkDescriptorBufferInfo descriptorBufferInfo[2];
  VkWriteDescriptorSet   writeDescriptorSet[2];

  descriptorBufferInfo[0]        = VkDescriptorBufferInfo{};
  descriptorBufferInfo[0].buffer = m_classDataBuffer;
  descriptorBufferInfo[0].offset = 0;
  descriptorBufferInfo[0].range  = VK_WHOLE_SIZE;

  descriptorBufferInfo[1]        = VkDescriptorBufferInfo{};
  descriptorBufferInfo[1].buffer = m_indirectBuffer;
  descriptorBufferInfo[1].offset = 0;
  descriptorBufferInfo[1].range  = VK_WHOLE_SIZE;

  writeDescriptorSet[0]                  = VkWriteDescriptorSet{};
  writeDescriptorSet[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeDescriptorSet[0].dstSet           = m_indirectUpdateDS;
  writeDescriptorSet[0].dstBinding       = 0;
  writeDescriptorSet[0].descriptorCount  = 1;
  writeDescriptorSet[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writeDescriptorSet[0].pBufferInfo      = &descriptorBufferInfo[0];
  writeDescriptorSet[0].pImageInfo       = nullptr;
  writeDescriptorSet[0].pTexelBufferView = nullptr;

  writeDescriptorSet[1]                  = VkWriteDescriptorSet{};
  writeDescriptorSet[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeDescriptorSet[1].dstSet           = m_indirectUpdateDS;
  writeDescriptorSet[1].dstBinding       = 1;
  writeDescriptorSet[1].descriptorCount  = 1;
  writeDescriptorSet[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writeDescriptorSet[1].pBufferInfo      = &descriptorBufferInfo[1];
  writeDescriptorSet[1].pImageInfo       = nullptr;
  writeDescriptorSet[1].pTexelBufferView = nullptr;

  vkUpdateDescriptorSets(m_device, 2, writeDescriptorSet, 0, NULL);
}
{% endif %}
