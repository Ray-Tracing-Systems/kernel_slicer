#include <vector>
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
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;

## for MainFunc in MainFunctions
  {% for Buffer in MainFunc.LocalVarsBuffersDecl %}
  vkDestroyBuffer(device, {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}

## endfor
 
  vkDestroyBuffer(device, m_classDataBuffer, nullptr);
  {% for Buffer in ClassVectorVars %}
  vkDestroyBuffer(device, m_vdata.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}
  {% for Buffer in RedVectorVars %}
  vkDestroyBuffer(device, m_vdata.{{Buffer.Name}}Buffer, nullptr);
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
  // binding for POD members stored in m_classDataBuffer
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

VkDescriptorSetLayout {{MainClassName}}_Generated::CreatecopyKernelFloatDSLayout()
{
  VkDescriptorSetLayoutBinding dsBindings[3] = {};

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

  // binding for POD members stored in m_classDataBuffer
  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = 3;
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings;

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

## for Kernel in Kernels
void {{MainClassName}}_Generated::InitKernel_{{Kernel.Name}}(const char* a_filePath)
{
  {% if MultipleSourceShaders %}
  std::string shaderPath = "{{ShaderFolder}}/{{Kernel.OriginalName}}.cpp.spv"; 
  {% else %}
  std::string shaderPath = a_filePath; 
  {% endif %}

  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "{{Kernel.OriginalName}}");
  {{Kernel.Name}}DSLayout = Create{{Kernel.Name}}DSLayout();
  {{Kernel.Name}}Layout   = m_pMaker->MakeLayout(device, {{Kernel.Name}}DSLayout, 128); // at least 128 bytes for push constants
  {{Kernel.Name}}Pipeline = m_pMaker->MakePipeline(device);  
  {% if Kernel.FinishRed %}
  
  m_pMaker->CreateShader(device, shaderPath.c_str(), nullptr, "{{Kernel.OriginalName}}_Reduction");
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

void {{MainClassName}}_Generated::InitKernels(const char* a_filePath, uint32_t a_blockSizeX, uint32_t a_blockSizeY, uint32_t a_blockSizeZ,
                                              KernelConfig* a_kernelConfigs, size_t a_configSize)
{
## for Kernel in Kernels
  InitKernel_{{Kernel.Name}}(a_filePath);
## endfor

  {% if MultipleSourceShaders %}
  std::string servPath = "{{ShaderFolder}}/serv_kernels.cpp.spv"; 
  {% else %}
  std::string servPath = a_filePath;
  {% endif %}

  m_pMaker->CreateShader(device, servPath.c_str(), nullptr, "copyKernelFloat");

  copyKernelFloatDSLayout = CreatecopyKernelFloatDSLayout();
  copyKernelFloatLayout   = m_pMaker->MakeLayout(device, copyKernelFloatDSLayout, 128); // at least 128 bytes for push constants
  copyKernelFloatPipeline = m_pMaker->MakePipeline(device);
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
  
  {% for Buffer in RedVectorVars %}
  {
    const size_t sizeOfBuffer = ComputeReductionAuxBufferElements(a_maxThreadsCount, REDUCTION_BLOCK_SIZE)*sizeof({{Buffer.Type}});
    m_vdata.{{Buffer.Name}}Buffer = vkfw::CreateBuffer(device, sizeOfBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    allBuffers.push_back(m_vdata.{{Buffer.Name}}Buffer);
  }
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
  
  if(memberVectors.size() > 0)
    m_vdata.m_vecMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, memberVectors);
}
