#include <vector>
#include <array>
#include <memory>
#include <limits>

#include <cassert>

#include "vulkan_basics.h"
#include "{{IncludeClassDecl}}"
#include "include/{{UBOIncl}}"

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

{{MainClassName}}_Generated::~{{MainClassName}}_Generated()
{
  m_pMaker = nullptr;

## for Kernel in Kernels
  vkDestroyDescriptorSetLayout(device, {{Kernel.Name}}DSLayout, nullptr);
  {{Kernel.Name}}DSLayout = VK_NULL_HANDLE;
## endfor
  vkDestroyDescriptorSetLayout(device, copyKernelFloatDSLayout, nullptr);
  {% if length(DispatchHierarchies) > 0 %}
  vkDestroyDescriptorSetLayout(device, ZeroCountersDSLayout, nullptr);
  {% endif %} 
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;

## for MainFunc in MainFunctions
  {% for Buffer in MainFunc.LocalVarsBuffersDecl %}
  vkDestroyBuffer(device, {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}

## endfor
 
  vkDestroyBuffer(device, m_classDataBuffer, nullptr);
  {% if UseSeparateUBO %}
  vkDestroyBuffer(device, m_uboArgsBuffer, nullptr);
  {% endif %}

  {% for Buffer in ClassVectorVars %}
  vkDestroyBuffer(device, m_vdata.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}
  {% for Buffer in RedVectorVars %}
  vkDestroyBuffer(device, m_vdata.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}
  {% if length(IndirectDispatches) > 0 %}
  vkDestroyBuffer(device, m_indirectBuffer, nullptr);
  vkDestroyDescriptorSetLayout(device, m_indirectUpdateDSLayout, nullptr);
  vkDestroyPipelineLayout(device, m_indirectUpdateLayout, nullptr);
  {% for Dispatch in IndirectDispatches %}
  vkDestroyPipeline(device, m_indirectUpdate{{Dispatch.KernelName}}Pipeline, nullptr);
  {% endfor %}
  {% endif %}
  {% for Hierarchy in DispatchHierarchies %}
  vkDestroyBuffer(device, m_{{Hierarchy.Name}}ObjPtrBuffer, nullptr);
  {% endfor %}

  if(m_allMem != VK_NULL_HANDLE)
    vkFreeMemory(device, m_allMem, nullptr);
  
  if(m_vdata.m_vecMem != VK_NULL_HANDLE)
    vkFreeMemory(device, m_vdata.m_vecMem, nullptr);
}

void {{MainClassName}}_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
  m_pMaker = std::make_unique<vkfw::ComputePipelineMaker>();
  {% if UseSpecConstWgSize %}
  {
    m_specializationEntriesWgSize[0].constantID = 0;
    m_specializationEntriesWgSize[0].offset     = 0;
    m_specializationEntriesWgSize[0].size       = sizeof(uint32_t);
  
    m_specializationEntriesWgSize[1].constantID = 1;
    m_specializationEntriesWgSize[1].offset     = sizeof(uint32_t);
    m_specializationEntriesWgSize[1].size       = sizeof(uint32_t);
  
    m_specializationEntriesWgSize[2].constantID = 2;
    m_specializationEntriesWgSize[2].offset     = 2 * sizeof(uint32_t);
    m_specializationEntriesWgSize[2].size       = sizeof(uint32_t);

    m_specsForWGSize.mapEntryCount = 3;
    m_specsForWGSize.pMapEntries   = m_specializationEntriesWgSize;
    m_specsForWGSize.dataSize      = 3 * sizeof(uint32_t);
    m_specsForWGSize.pData         = nullptr;
  }
  {% endif %}
}

## for Kernel in Kernels
VkDescriptorSetLayout {{MainClassName}}_Generated::Create{{Kernel.Name}}DSLayout()
{
  {% if UseSeparateUBO %}
  std::array<VkDescriptorSetLayoutBinding, {{Kernel.ArgCount}}+2> dsBindings;
  {% else %}
  std::array<VkDescriptorSetLayoutBinding, {{Kernel.ArgCount}}+1> dsBindings;
  {% endif %}

## for KernelARG in Kernel.Args
  // binding for {{KernelARG.Name}}
  dsBindings[{{KernelARG.Id}}].binding            = {{KernelARG.Id}};
  dsBindings[{{KernelARG.Id}}].descriptorType     = {{KernelARG.Type}};
  dsBindings[{{KernelARG.Id}}].descriptorCount    = 1;
  dsBindings[{{KernelARG.Id}}].stageFlags         = {{KernelARG.Flags}};
  dsBindings[{{KernelARG.Id}}].pImmutableSamplers = nullptr;

## endfor
  // binding for POD members stored in m_classDataBuffer
  dsBindings[{{Kernel.ArgCount}}].binding            = {{Kernel.ArgCount}};
  dsBindings[{{Kernel.ArgCount}}].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[{{Kernel.ArgCount}}].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[{{Kernel.ArgCount}}].pImmutableSamplers = nullptr;
  {% if UseSeparateUBO %}
  
  dsBindings[{{Kernel.ArgCount}}+1].binding            = {{Kernel.ArgCount}}+1;
  dsBindings[{{Kernel.ArgCount}}+1].descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  dsBindings[{{Kernel.ArgCount}}+1].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}+1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[{{Kernel.ArgCount}}+1].pImmutableSamplers = nullptr;
  {% endif %}
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t(dsBindings.size());
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}
## endfor

VkDescriptorSetLayout {{MainClassName}}_Generated::CreatecopyKernelFloatDSLayout()
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
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

{% if length(DispatchHierarchies) > 0 %}
VkDescriptorSetLayout {{MainClassName}}_Generated::CreateZeroObjCountersLayout()
{
  VkDescriptorSetLayoutBinding dsBinding = {};
  dsBinding.binding            = 0;
  dsBinding.descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBinding.descriptorCount    = 1;
  dsBinding.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBinding.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = 1;
  descriptorSetLayoutCreateInfo.pBindings    = &dsBinding;

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

VkDescriptorSet {{MainClassName}}_Generated::CreateObjCountersDS()
{
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.descriptorPool     = m_dsPool;  
  descriptorSetAllocateInfo.descriptorSetCount = 1;     
  descriptorSetAllocateInfo.pSetLayouts        = &ZeroCountersDSLayout;

  auto tmpRes = vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &ZeroCountersDS);
  VK_CHECK_RESULT(tmpRes); 

  VkDescriptorBufferInfo descriptorBufferInfo;
  VkWriteDescriptorSet   writeDescriptorSet;
  
  descriptorBufferInfo        = VkDescriptorBufferInfo{};
  descriptorBufferInfo.buffer = m_classDataBuffer;
  descriptorBufferInfo.offset = 0;
  descriptorBufferInfo.range  = VK_WHOLE_SIZE;  

  writeDescriptorSet                  = VkWriteDescriptorSet{};
  writeDescriptorSet.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeDescriptorSet.dstSet           = ZeroCountersDS;
  writeDescriptorSet.dstBinding       = 0;
  writeDescriptorSet.descriptorCount  = 1;
  writeDescriptorSet.descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writeDescriptorSet.pBufferInfo      = &descriptorBufferInfo;
  writeDescriptorSet.pImageInfo       = nullptr;
  writeDescriptorSet.pTexelBufferView = nullptr; 
  
  vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
  return ZeroCountersDS;
}

VkBufferMemoryBarrier {{MainClassName}}_Generated::BarrierForObjCounters(VkBuffer a_buffer)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.pNext               = NULL;
  bar.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
  bar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = a_buffer;
  bar.offset              = 0;
  bar.size                = VK_WHOLE_SIZE; // TODO: count offset and size carefully, actually we can do this!
  return bar;
}
{% endif %}

## for Kernel in Kernels
void {{MainClassName}}_Generated::InitKernel_{{Kernel.Name}}(const char* a_filePath)
{
  {% if MultipleSourceShaders %}
  std::string shaderPath = "{{ShaderFolder}}/{{Kernel.OriginalName}}.cpp.spv"; 
  {% else %}
  std::string shaderPath = a_filePath; 
  {% endif %}
  
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { {{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}} };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->CreateShader(device, shaderPath.c_str(), &m_specsForWGSize, "{{Kernel.OriginalName}}");
  }
  {% else %}
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "{{Kernel.OriginalName}}");
  {% endif %}
  {{Kernel.Name}}DSLayout = Create{{Kernel.Name}}DSLayout();
  {{Kernel.Name}}Layout   = m_pMaker->MakeLayout(device, {{Kernel.Name}}DSLayout, 128); // at least 128 bytes for push constants
  {{Kernel.Name}}Pipeline = m_pMaker->MakePipeline(device);  
  {% if Kernel.FinishRed %}
  
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { 256, 1, 1 };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->CreateShader(device, shaderPath.c_str(), &m_specsForWGSize, "{{Kernel.OriginalName}}_Reduction");
  }
  {% else %}
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "{{Kernel.OriginalName}}_Reduction");
  {% endif %}
  {{Kernel.Name}}ReductionPipeline = m_pMaker->MakePipeline(device);
  {% endif %} 
  {% if Kernel.HasLoopInit %}
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "{{Kernel.OriginalName}}_Init"); 
  {{Kernel.Name}}InitPipeline = m_pMaker->MakePipeline(device);
  {% if Kernel.HasLoopFinish %}
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "{{Kernel.OriginalName}}_Finish");
  {{Kernel.Name}}FinishPipeline = m_pMaker->MakePipeline(device);
  {% endif %}
  {% endif %} 
}

## endfor

void {{MainClassName}}_Generated::InitKernels(const char* a_filePath)
{
## for Kernel in Kernels
  InitKernel_{{Kernel.Name}}(a_filePath);
## endfor

  {% if MultipleSourceShaders %}
  std::string servPath = "{{ShaderFolder}}/serv_kernels.cpp.spv"; 
  {% else %}
  std::string servPath = a_filePath;
  {% endif %}
  
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { 256, 1, 1 };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->CreateShader(device, servPath.c_str(), &m_specsForWGSize, "copyKernelFloat");
  }
  {% else %}
  m_pMaker->CreateShader(device, servPath.c_str(), nullptr, "copyKernelFloat");
  {% endif %}
  copyKernelFloatDSLayout = CreatecopyKernelFloatDSLayout();
  copyKernelFloatLayout   = m_pMaker->MakeLayout(device, copyKernelFloatDSLayout, 128); // at least 128 bytes for push constants
  copyKernelFloatPipeline = m_pMaker->MakePipeline(device);
  {% if length(IndirectDispatches) > 0 %}
  InitIndirectBufferUpdateResources(a_filePath);
  {% endif %}

  {% if length(DispatchHierarchies) > 0 %}
  ZeroCountersDSLayout = CreateZeroObjCountersLayout(); 
  {% for Hierarchy in DispatchHierarchies %}
  {% if Hierarchy.IndirectDispatch %}
  m_pMaker->CreateShader(device, servPath.c_str(), nullptr, "{{Hierarchy.Name}}_ZeroObjCounters");
  {{Hierarchy.Name}}ZeroObjCountersLayout   = m_pMaker->MakeLayout(device, ZeroCountersDSLayout, 0);
  {{Hierarchy.Name}}ZeroObjCountersPipeline = m_pMaker->MakePipeline(device);
  {% endif %}  
  {% endfor %}    
  {% endif %} {# /* length(DispatchHierarchies) > 0 */ #}
}

void {{MainClassName}}_Generated::InitBuffers(size_t a_maxThreadsCount)
{
  std::vector<VkBuffer> allBuffers;

## for MainFunc in MainFunctions  
## for Buffer in MainFunc.LocalVarsBuffersDecl
  {% if Buffer.TransferDST %}
  {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer = vkfw::CreateBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  {% else %}
  {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer = vkfw::CreateBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  {% endif %}
  allBuffers.push_back({{MainFunc.Name}}_local.{{Buffer.Name}}Buffer);
## endfor
## endfor

  m_classDataBuffer = vkfw::CreateBuffer(device, sizeof(m_uboData),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | GetAdditionalFlagsForUBO());
  allBuffers.push_back(m_classDataBuffer);
  {% if UseSeparateUBO %}
  m_uboArgsBuffer = vkfw::CreateBuffer(device, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffers.push_back(m_uboArgsBuffer);
  {% endif %}
  {% for Buffer in RedVectorVars %}
  {
    const size_t sizeOfBuffer = ComputeReductionAuxBufferElements(a_maxThreadsCount, REDUCTION_BLOCK_SIZE)*sizeof({{Buffer.Type}});
    m_vdata.{{Buffer.Name}}Buffer = vkfw::CreateBuffer(device, sizeOfBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    allBuffers.push_back(m_vdata.{{Buffer.Name}}Buffer);
  }
  {% endfor %}
  {% for Hierarchy in DispatchHierarchies %}
  m_{{Hierarchy.Name}}ObjPtrBuffer = vkfw::CreateBuffer(device, sizeof(uint32_t)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back(m_{{Hierarchy.Name}}ObjPtrBuffer);
  {% endfor %}

  if(allBuffers.size() > 0)
    m_allMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, allBuffers);
}

void {{MainClassName}}_Generated::InitMemberBuffers()
{
  std::vector<VkBuffer> memberVectors;
## for Var in ClassVectorVars
  m_vdata.{{Var.Name}}Buffer = vkfw::CreateBuffer(device, {{Var.Name}}.capacity()*sizeof({{Var.TypeOfData}}), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.{{Var.Name}}Buffer);
## endfor
  
  {% if length(IndirectDispatches) > 0 %}
  m_indirectBuffer = vkfw::CreateBuffer(device, {{length(IndirectDispatches)}}*sizeof(uint32_t)*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  memberVectors.push_back(m_indirectBuffer);
  {% endif %}
  if(memberVectors.size() > 0)
    m_vdata.m_vecMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, memberVectors);
  {% if length(IndirectDispatches) > 0 %}
  InitIndirectDescriptorSets();
  {% endif %}
}

{% if length(IndirectDispatches) > 0 %}
void {{MainClassName}}_Generated::InitIndirectBufferUpdateResources(const char* a_filePath)
{
  // (1) init m_indirectUpdateDSLayout
  //
  VkDescriptorSetLayoutBinding bindings[2] = {};
  bindings[0].binding            = 0;
  bindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount    = 1;
  bindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[0].pImmutableSamplers = nullptr;

  bindings[1].binding            = 1;
  bindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount    = 1;
  bindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = 2;
  descriptorSetLayoutCreateInfo.pBindings    = bindings;

  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &m_indirectUpdateDSLayout));

  VkDescriptorSetLayout oneTwo[2] = {m_indirectUpdateDSLayout,m_indirectUpdateDSLayout};

  VkPipelineLayoutCreateInfo  pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayoutInfo.pPushConstantRanges    = nullptr;
  pipelineLayoutInfo.pSetLayouts            = oneTwo; 
  pipelineLayoutInfo.setLayoutCount         = 2;
 
  VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_indirectUpdateLayout));

  VkShaderModule tempShaderModule = VK_NULL_HANDLE;

  std::vector<uint32_t> code = vk_utils::ReadFile(a_filePath);
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.pCode    = code.data();
  createInfo.codeSize = code.size()*sizeof(uint32_t);
  VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &tempShaderModule));

  {% for Dispatch in IndirectDispatches %}
  // create indrect update pipeline for {{Dispatch.OriginalName}}
  //
  {
    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = tempShaderModule;
    shaderStageInfo.pName  = "{{Dispatch.OriginalName}}_UpdateIndirect";

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage  = shaderStageInfo;
    pipelineCreateInfo.layout = m_indirectUpdateLayout;
    VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &m_indirectUpdate{{Dispatch.KernelName}}Pipeline));
  }
  {% endfor %}

  vkDestroyShaderModule(device, tempShaderModule, VK_NULL_HANDLE);
}

VkBufferMemoryBarrier {{MainClassName}}_Generated::BarrierForIndirectBufferUpdate(VkBuffer a_buffer)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.pNext               = NULL;
  bar.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
  bar.dstAccessMask       = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = a_buffer;
  bar.offset              = 0;
  bar.size                = VK_WHOLE_SIZE;
  return bar;
}
{% endif %}

{% if UseSeparateUBO %}
VkBufferMemoryBarrier {{MainClassName}}_Generated::BarrierForArgsUBO(size_t a_size)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.pNext               = NULL;
  bar.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
  bar.dstAccessMask       = VK_ACCESS_UNIFORM_READ_BIT;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = m_uboArgsBuffer;
  bar.offset              = 0;
  bar.size                = a_size;
  return bar;
}
{% endif %}
