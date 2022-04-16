#include <vector>
#include <array>
#include <memory>
#include <limits>

#include <cassert>
#include "vk_copy.h"
#include "vk_context.h"

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

VkBufferUsageFlags {{MainClassName}}_Generated::GetAdditionalFlagsForUBO() const
{
  {% if HasFullImpl %}
  return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  {% else %}
  return 0;
  {% endif %}
}

uint32_t {{MainClassName}}_Generated::GetDefaultMaxTextures() const { return 256; }

{{MainClassName}}_Generated::~{{MainClassName}}_Generated()
{
  m_pMaker = nullptr;
  {% if UseServiceMemCopy %}
  vkDestroyPipeline(device, copyKernelFloatPipeline, nullptr);
  vkDestroyPipelineLayout(device, copyKernelFloatLayout, nullptr);
  {% endif %} {# /* UseServiceMemCopy */ #}
## for Kernel in Kernels
  vkDestroyDescriptorSetLayout(device, {{Kernel.Name}}DSLayout, nullptr);
  {{Kernel.Name}}DSLayout = VK_NULL_HANDLE;

  vkDestroyPipeline(device, {{Kernel.Name}}Pipeline, nullptr);
  vkDestroyPipelineLayout(device, {{Kernel.Name}}Layout, nullptr);
  {{Kernel.Name}}Layout   = VK_NULL_HANDLE;
  {{Kernel.Name}}Pipeline = VK_NULL_HANDLE;
  {% if Kernel.HasLoopInit %}
  vkDestroyPipeline(device, {{Kernel.Name}}InitPipeline, nullptr);
  {{Kernel.Name}}InitPipeline = VK_NULL_HANDLE;
  {% endif %} 
  {% if Kernel.HasLoopFinish %}
  vkDestroyPipeline(device, {{Kernel.Name}}FinishPipeline, nullptr);
  {{Kernel.Name}}FinishPipeline = VK_NULL_HANDLE;
  {% endif %} 
  {% if Kernel.FinishRed %}
  vkDestroyPipeline(device, {{Kernel.Name}}ReductionPipeline, nullptr);
  {{Kernel.Name}}ReductionPipeline = VK_NULL_HANDLE;
  {% endif %} 
  {% if Kernel.IsMaker and Kernel.Hierarchy.IndirectDispatch %}
  vkDestroyPipeline(device, {{Kernel.Name}}ZeroObjCounters, nullptr);
  vkDestroyPipeline(device, {{Kernel.Name}}CountTypeIntervals, nullptr);
  vkDestroyPipeline(device, {{Kernel.Name}}Sorter, nullptr);
  {{Kernel.Name}}ZeroObjCounters    = VK_NULL_HANDLE;
  {{Kernel.Name}}CountTypeIntervals = VK_NULL_HANDLE;
  {{Kernel.Name}}Sorter             = VK_NULL_HANDLE; 
  {% endif %}     
  {% if Kernel.IsVirtual and Kernel.Hierarchy.IndirectDispatch %}
  for(int i=0;i<{{length(Kernel.Hierarchy.Implementations)}};i++)
  {
    vkDestroyPipeline(device, {{Kernel.Name}}PipelineArray[i], nullptr);
    {{Kernel.Name}}PipelineArray[i] = nullptr;
  }
  {% endif %}  
## endfor
  vkDestroyDescriptorSetLayout(device, copyKernelFloatDSLayout, nullptr);
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;

## for MainFunc in MainFunctions
  {% if MainFunc.IsRTV and not MainFunc.IsMega %} 
  {% for Buffer in MainFunc.LocalVarsBuffersDecl %}
  vkDestroyBuffer(device, {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}
  {% endif %}
## endfor
 
  vkDestroyBuffer(device, m_classDataBuffer, nullptr);
  {% if UseSeparateUBO %}
  vkDestroyBuffer(device, m_uboArgsBuffer, nullptr);
  {% endif %}

  {% for Buffer in ClassVectorVars %}
  vkDestroyBuffer(device, m_vdata.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}
  {% for Var in ClassTextureVars %}
  vkDestroyImage    (device, m_vdata.{{Var.Name}}Texture, nullptr);
  vkDestroyImageView(device, m_vdata.{{Var.Name}}View, nullptr);
  if(m_vdata.{{Var.Name}}Sampler != VK_NULL_HANDLE)
     vkDestroySampler(device, m_vdata.{{Var.Name}}Sampler, nullptr);
  {% endfor %}
  {% for Var in ClassTexArrayVars %}
  for(auto obj : m_vdata.{{Var.Name}}ArrayTexture)
    vkDestroyImage(device, obj, nullptr);
  for(auto obj : m_vdata.{{Var.Name}}ArrayView)
    vkDestroyImageView(device, obj, nullptr);
  for(auto obj : m_vdata.{{Var.Name}}ArraySampler)
  vkDestroySampler(device, obj, nullptr);
  {% endfor %}
  {% for Sam in SamplerMembers %}
  vkDestroySampler(device, m_vdata.{{Sam}}, nullptr);
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

  FreeAllAllocations(m_allMems);
}

void {{MainClassName}}_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
  m_pMaker = std::make_unique<vk_utils::ComputePipelineMaker>();
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
  {% if UseSeparateUBO and Kernel.IsVirtual %}
  std::array<VkDescriptorSetLayoutBinding, {{Kernel.ArgCount}}+3> dsBindings;
  {% else if UseSeparateUBO or Kernel.IsVirtual %}
  std::array<VkDescriptorSetLayoutBinding, {{Kernel.ArgCount}}+2> dsBindings;
  {% else %}
  std::array<VkDescriptorSetLayoutBinding, {{Kernel.ArgCount}}+1> dsBindings;
  {% endif %}

## for KernelARG in Kernel.Args
  // binding for {{KernelARG.Name}}
  dsBindings[{{KernelARG.Id}}].binding            = {{KernelARG.Id}};
  dsBindings[{{KernelARG.Id}}].descriptorType     = {{KernelARG.Type}};
  {% if KernelARG.IsTextureArray %}
  m_vdata.{{KernelARG.Name}}ArrayMaxSize = {{KernelARG.Count}};
  if(m_vdata.{{KernelARG.Name}}ArrayMaxSize == 0)
    m_vdata.{{KernelARG.Name}}ArrayMaxSize = GetDefaultMaxTextures();
  dsBindings[{{KernelARG.Id}}].descriptorCount    = m_vdata.{{KernelARG.Name}}ArrayMaxSize;
  {% else %}
  dsBindings[{{KernelARG.Id}}].descriptorCount    = {{KernelARG.Count}};
  {% endif %}
  dsBindings[{{KernelARG.Id}}].stageFlags         = {{KernelARG.Flags}};
  dsBindings[{{KernelARG.Id}}].pImmutableSamplers = nullptr;

## endfor
  // binding for {% if Kernel.IsVirtual %}kgen_objData{% else %}POD members stored in m_classDataBuffer{% endif %}

  dsBindings[{{Kernel.ArgCount}}].binding            = {{Kernel.ArgCount}};
  dsBindings[{{Kernel.ArgCount}}].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[{{Kernel.ArgCount}}].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[{{Kernel.ArgCount}}].pImmutableSamplers = nullptr;
  {% if UseSeparateUBO and Kernel.IsVirtual %}
  
  // binding for m_classDataBuffer
  dsBindings[{{Kernel.ArgCount}}+1].binding            = {{Kernel.ArgCount}}+1;
  dsBindings[{{Kernel.ArgCount}}+1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[{{Kernel.ArgCount}}+1].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}+1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[{{Kernel.ArgCount}}+1].pImmutableSamplers = nullptr;
  
  // binding for separate ubo
  dsBindings[{{Kernel.ArgCount}}+2].binding            = {{Kernel.ArgCount}}+1;
  dsBindings[{{Kernel.ArgCount}}+2].descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  dsBindings[{{Kernel.ArgCount}}+2].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}+2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[{{Kernel.ArgCount}}+2].pImmutableSamplers = nullptr;
  {% else if UseSeparateUBO or Kernel.IsVirtual %}
  
  // binding for {% if UseSeparateUBO%}separate ubo{% else %}m_classDataBuffer {% endif %}

  dsBindings[{{Kernel.ArgCount}}+1].binding            = {{Kernel.ArgCount}}+1;
  dsBindings[{{Kernel.ArgCount}}+1].descriptorType     = {% if UseSeparateUBO %}VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER{% else %}VK_DESCRIPTOR_TYPE_STORAGE_BUFFER{% endif %};
  dsBindings[{{Kernel.ArgCount}}+1].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}+1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[{{Kernel.ArgCount}}+1].pImmutableSamplers = nullptr;
  {% else %}
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
VkBufferMemoryBarrier {{MainClassName}}_Generated::BarrierForObjCounters(VkBuffer a_buffer)
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
{% endif %}

## for Kernel in Kernels
void {{MainClassName}}_Generated::InitKernel_{{Kernel.Name}}(const char* a_filePath)
{
  {% if MultipleSourceShaders %}
  std::string shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}.comp.spv"); 
  {% else %}
  std::string shaderPath = AlterShaderPath(a_filePath); 
  {% endif %}
  
  {% if Kernel.IsVirtual and Kernel.Hierarchy.IndirectDispatch %}
  {{Kernel.Name}}DSLayout = Create{{Kernel.Name}}DSLayout();
  {% else%}
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { {{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}} };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->LoadShader(device, shaderPath.c_str(), &m_specsForWGSize, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}"{% endif %});
  }
  {% else %}
  m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}"{% endif %});
  {% endif %}
  {{Kernel.Name}}DSLayout = Create{{Kernel.Name}}DSLayout();
  {{Kernel.Name}}Layout   = m_pMaker->MakeLayout(device, { {{Kernel.Name}}DSLayout }, 128); // at least 128 bytes for push constants
  {{Kernel.Name}}Pipeline = m_pMaker->MakePipeline(device);  
  {% endif %} {# /* not Kernel.IsVirtual and Kernel.Hierarchy.IndirectDispatch */ #}
  {% if Kernel.FinishRed %}
  
  {% if ShaderGLSL %}
  shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}_Reduction.comp.spv");
  {% endif %}
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { 256, 1, 1 };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->LoadShader(device, shaderPath.c_str(), &m_specsForWGSize, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Reduction"{% endif %});
  }
  {% else %}
  m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Reduction"{% endif %});
  {% endif %}
  {{Kernel.Name}}ReductionPipeline = m_pMaker->MakePipeline(device);
  {% endif %} 
  {% if Kernel.HasLoopInit %}
  
  {% if ShaderGLSL %}
  shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}_Init.comp.spv");
  {% endif %}
  m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Init"{% endif %}); 
  {{Kernel.Name}}InitPipeline = m_pMaker->MakePipeline(device);
  {% endif %} {# /* if Kernel.HasLoopInit */ #} 
  {% if Kernel.HasLoopFinish %}
  
  {% if ShaderGLSL %}
  shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}_Finish.comp.spv");
  {% endif %}
  m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Finish"{% endif %});
  {{Kernel.Name}}FinishPipeline = m_pMaker->MakePipeline(device);
  {% endif %} {# /* if Kernel.HasLoopFinish */ #} 
  {% if Kernel.IsMaker and Kernel.Hierarchy.IndirectDispatch %}
  
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { 32, 1, 1 };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_ZeroObjCounters"{% endif %});
  }
  {% else %}
  m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_ZeroObjCounters"{% endif %});
  {% endif %}
  {{Kernel.Name}}ZeroObjCounters    = m_pMaker->MakePipeline(device);
  
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { 32, 1, 1 };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_CountTypeIntervals"{% endif %});
  }
  {% else %}
  m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_CountTypeIntervals"{% endif %});
  {% endif %}
  {{Kernel.Name}}CountTypeIntervals = m_pMaker->MakePipeline(device);
  
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { {{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}} };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->LoadShader(device, shaderPath.c_str(), &m_specsForWGSize, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Sorter"{% endif %});
  }
  {% else %}
  m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Sorter"{% endif %});
  {% endif %}
  {{Kernel.Name}}Sorter             = m_pMaker->MakePipeline(device);
  {% else if Kernel.IsVirtual and Kernel.Hierarchy.IndirectDispatch %} {# /* if Kernel.IsMaker and Kernel.Hierarchy.IndirectDispatch */ #} 
  {% for Impl in Kernel.Hierarchy.Implementations %}
  
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { {{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}} };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->LoadShader(device, shaderPath.c_str(), &m_specsForWGSize, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_{{Impl.ClassName}}"{% endif %});
  }
  {% else %}
  m_pMaker->LoadShader(device, shaderPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_{{Impl.ClassName}}"{% endif %});
  {% endif %}
  {% if loop.index == 0 %}
  {{Kernel.Name}}Layout = m_pMaker->MakeLayout(device, { {{Kernel.Name}}DSLayout }, 128); // at least 128 bytes for push constants
  {% endif %}
  {{Kernel.Name}}PipelineArray[{{loop.index}}] = m_pMaker->MakePipeline(device);  
  {% endfor %}
 
  {% endif %} {# /* if Kernel.IsMaker and Kernel.Hierarchy.IndirectDispatch */ #} 
}

## endfor

void {{MainClassName}}_Generated::InitKernels(const char* a_filePath)
{
## for Kernel in Kernels
  InitKernel_{{Kernel.Name}}(a_filePath);
## endfor
  {% if UseServiceMemCopy %}
  {% if MultipleSourceShaders %}
  std::string servPath = AlterShaderPath({% if ShaderGLSL %}"{{ShaderFolder}}/z_memcpy.comp.spv"{% else %}"{{ShaderFolder}}/serv_kernels.cpp.spv"{% endif %}); 
  {% else %}
  std::string servPath = AlterShaderPath(a_filePath);
  {% endif %}
  {% if UseSpecConstWgSize %}
  {
    uint32_t specializationData[3] = { 256, 1, 1 };
    m_specsForWGSize.pData         = specializationData;
    m_pMaker->LoadShader(device, servPath.c_str(), &m_specsForWGSize, {% if ShaderGLSL %}"main"{% else %}"copyKernelFloat"{% endif %});
  }
  {% else %}
  m_pMaker->LoadShader(device, servPath.c_str(), nullptr, {% if ShaderGLSL %}"main"{% else %}"copyKernelFloat"{% endif %});
  {% endif %}
  copyKernelFloatDSLayout = CreatecopyKernelFloatDSLayout();
  copyKernelFloatLayout   = m_pMaker->MakeLayout(device, {copyKernelFloatDSLayout}, 128); // at least 128 bytes for push constants
  copyKernelFloatPipeline = m_pMaker->MakePipeline(device);
  {% endif %} {# /* UseServiceMemCopy */ #}
  {% if length(IndirectDispatches) > 0 %}
  InitIndirectBufferUpdateResources(a_filePath);
  {% endif %}
}

void {{MainClassName}}_Generated::InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay)
{
  m_maxThreadCount = a_maxThreadsCount;
  std::vector<VkBuffer> allBuffers;
  allBuffers.reserve(64);

  struct BufferReqPair
  {
    BufferReqPair() {  }
    BufferReqPair(VkBuffer a_buff, VkDevice a_dev) : buf(a_buff) { vkGetBufferMemoryRequirements(a_dev, a_buff, &req); }
    VkBuffer             buf = VK_NULL_HANDLE;
    VkMemoryRequirements req = {};
  };

  struct LocalBuffers
  {
    std::vector<BufferReqPair> bufs;
    size_t                     size = 0;
    std::vector<VkBuffer>      bufsClean;
  };

  std::vector<LocalBuffers> groups;
  groups.reserve(16);

## for MainFunc in MainFunctions 
  {% if MainFunc.IsRTV and not MainFunc.IsMega %} 
  LocalBuffers localBuffers{{MainFunc.Name}};
  localBuffers{{MainFunc.Name}}.bufs.reserve(32);
  {% for Buffer in MainFunc.LocalVarsBuffersDecl %}
  {% if Buffer.TransferDST %}
  {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer = vk_utils::createBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  {% else %}
  {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer = vk_utils::createBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  {% endif %}
  localBuffers{{MainFunc.Name}}.bufs.push_back(BufferReqPair({{MainFunc.Name}}_local.{{Buffer.Name}}Buffer, device));
  {% endfor %}
  for(const auto& pair : localBuffers{{MainFunc.Name}}.bufs)
  {
    allBuffers.push_back(pair.buf);
    localBuffers{{MainFunc.Name}}.size += pair.req.size;
  }
  groups.push_back(localBuffers{{MainFunc.Name}});
  {% endif %}
## endfor

  size_t largestIndex = 0;
  size_t largestSize  = 0;
  for(size_t i=0;i<groups.size();i++)
  {
    if(groups[i].size > largestSize)
    {
      largestIndex = i;
      largestSize  = groups[i].size;
    }
    groups[i].bufsClean.resize(groups[i].bufs.size());
    for(size_t j=0;j<groups[i].bufsClean.size();j++)
      groups[i].bufsClean[j] = groups[i].bufs[j].buf;
  }
  
  {% if IsRTV and not IsMega %}
  auto& allBuffersRef = a_tempBuffersOverlay ? groups[largestIndex].bufsClean : allBuffers;
  {% else %}
  auto& allBuffersRef = allBuffers;
  {% endif %}

  m_classDataBuffer = vk_utils::createBuffer(device, sizeof(m_uboData),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | GetAdditionalFlagsForUBO());
  allBuffersRef.push_back(m_classDataBuffer);
  {% if UseSeparateUBO %}
  m_uboArgsBuffer = vk_utils::createBuffer(device, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffersRef.push_back(m_uboArgsBuffer);
  {% endif %}
  {% for Buffer in RedVectorVars %}
  {
    const size_t sizeOfBuffer = ComputeReductionAuxBufferElements(a_maxThreadsCount, REDUCTION_BLOCK_SIZE)*sizeof({{Buffer.Type}});
    m_vdata.{{Buffer.Name}}Buffer = vk_utils::createBuffer(device, sizeOfBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    allBuffersRef.push_back(m_vdata.{{Buffer.Name}}Buffer);
  }
  {% endfor %}
  {% for Hierarchy in DispatchHierarchies %}
  m_{{Hierarchy.Name}}ObjPtrBuffer = vk_utils::createBuffer(device, 2*sizeof(uint32_t)*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffersRef.push_back(m_{{Hierarchy.Name}}ObjPtrBuffer);
  {% endfor %}
  
  auto internalBuffersMem = AllocAndBind(allBuffersRef);
  if(a_tempBuffersOverlay)
  {
    for(size_t i=0;i<groups.size();i++)
      if(i != largestIndex)
        AssignBuffersToMemory(groups[i].bufsClean, internalBuffersMem.memObject);
  }
}

void {{MainClassName}}_Generated::InitMemberBuffers()
{
  std::vector<VkBuffer> memberVectors;
  std::vector<VkImage>  memberTextures;

  {% for Var in ClassVectorVars %}
  m_vdata.{{Var.Name}}Buffer = vk_utils::createBuffer(device, {{Var.Name}}.capacity()*sizeof({{Var.TypeOfData}}), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.{{Var.Name}}Buffer);
  {% endfor %}

  {% for Var in ClassTextureVars %}
  m_vdata.{{Var.Name}}Texture = CreateTexture2D({{Var.Name}}{{Var.AccessSymb}}width(), {{Var.Name}}{{Var.AccessSymb}}height(), VkFormat({{Var.Format}}), {{Var.Usage}});
  {% if Var.NeedSampler %}
  m_vdata.{{Var.Name}}Sampler = CreateSampler({{Var.Name}}->getSampler());
  {% endif %}
  memberTextures.push_back(m_vdata.{{Var.Name}}Texture);
  {% endfor %}
  {% for Var in ClassTexArrayVars %}
  m_vdata.{{Var.Name}}ArrayTexture.resize(0);
  m_vdata.{{Var.Name}}ArrayView.resize(0);
  m_vdata.{{Var.Name}}ArraySampler.resize(0);
  m_vdata.{{Var.Name}}ArrayTexture.reserve(64);
  m_vdata.{{Var.Name}}ArrayView.reserve(64);
  m_vdata.{{Var.Name}}ArraySampler.reserve(64);
  for(auto imageObj : {{Var.Name}}) 
  {
    auto tex = CreateTexture2D(imageObj->width(), imageObj->height(), VkFormat(imageObj->format()), {{Var.Usage}});
    auto sam = CreateSampler(imageObj->getSampler());
    m_vdata.{{Var.Name}}ArrayTexture.push_back(tex);
    m_vdata.{{Var.Name}}ArrayView.push_back(VK_NULL_HANDLE);
    m_vdata.{{Var.Name}}ArraySampler.push_back(sam);
    memberTextures.push_back(tex);
  }
  {% endfor %}
  {% for Sam in SamplerMembers %}
  m_vdata.{{Sam}} = CreateSampler({{Sam}});
  {% endfor %}

  {% if length(IndirectDispatches) > 0 %}
  m_indirectBuffer = vk_utils::createBuffer(device, {{IndirectBufferSize}}*sizeof(uint32_t)*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  memberVectors.push_back(m_indirectBuffer);
  {% endif %}
  AllocMemoryForMemberBuffersAndImages(memberVectors, memberTextures);
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
  
  {% if ShaderGLSL %}
  {% for Dispatch in IndirectDispatches %}
  // create indrect update pipeline for {{Dispatch.OriginalName}}
  //
  {
    VkShaderModule tempShaderModule = VK_NULL_HANDLE;
    std::vector<uint32_t> code      = vk_utils::readSPVFile("{{ShaderFolder}}/{{Dispatch.OriginalName}}_UpdateIndirect.comp.spv");
    
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode    = code.data();
    createInfo.codeSize = code.size()*sizeof(uint32_t);
    VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &tempShaderModule));

    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = tempShaderModule;
    shaderStageInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage  = shaderStageInfo;
    pipelineCreateInfo.layout = m_indirectUpdateLayout;
    VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &m_indirectUpdate{{Dispatch.KernelName}}Pipeline));

    vkDestroyShaderModule(device, tempShaderModule, VK_NULL_HANDLE);
  }
  {% endfor %}
  {% else %}
  VkShaderModule tempShaderModule = VK_NULL_HANDLE;
  std::vector<uint32_t> code = vk_utils::readSPVFile(a_filePath);
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
  {% endif %} {# /* end else branch of if ShaderGLSL */ #}
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

{% if length(TextureMembers) > 0 or length(ClassTexArrayVars) > 0 %}
VkImage {{MainClassName}}_Generated::CreateTexture2D(const int a_width, const int a_height, VkFormat a_format, VkImageUsageFlags a_usage)
{
  VkImage result = VK_NULL_HANDLE;
  VkImageCreateInfo imgCreateInfo = {};
  imgCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgCreateInfo.pNext         = nullptr;
  imgCreateInfo.flags         = 0; // not sure about this ...
  imgCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
  imgCreateInfo.format        = a_format;
  imgCreateInfo.extent        = VkExtent3D{uint32_t(a_width), uint32_t(a_height), 1};
  imgCreateInfo.mipLevels     = 1;
  imgCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  imgCreateInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
  imgCreateInfo.usage         = a_usage; 
  imgCreateInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imgCreateInfo.arrayLayers   = 1;
  VK_CHECK_RESULT(vkCreateImage(device, &imgCreateInfo, nullptr, &result));
  return result;
}

VkSampler {{MainClassName}}_Generated::CreateSampler(const Sampler& a_sampler) // TODO: implement this function correctly
{
  VkSampler result = VK_NULL_HANDLE;
  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.pNext        = nullptr;
  samplerInfo.flags        = 0;
  samplerInfo.magFilter    = VkFilter(int(a_sampler.filter));
  samplerInfo.minFilter    = VkFilter(int(a_sampler.filter));
  samplerInfo.mipmapMode   = (samplerInfo.magFilter == VK_FILTER_LINEAR ) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VkSamplerAddressMode(int(a_sampler.addressU));
  samplerInfo.addressModeV = VkSamplerAddressMode(int(a_sampler.addressV));
  samplerInfo.addressModeW = VkSamplerAddressMode(int(a_sampler.addressW));
  samplerInfo.mipLodBias   = a_sampler.mipLODBias;
  samplerInfo.compareOp    = VK_COMPARE_OP_NEVER;
  samplerInfo.minLod           = a_sampler.minLOD;
  samplerInfo.maxLod           = a_sampler.maxLOD;
  samplerInfo.maxAnisotropy    = a_sampler.maxAnisotropy;
  samplerInfo.anisotropyEnable = (a_sampler.maxAnisotropy > 1) ? VK_TRUE : VK_FALSE;
  samplerInfo.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &result));
  return result;
}
{% if 0 %}
void {{MainClassName}}_Generated::TrackTextureAccess(const std::vector<TexAccessPair>& a_pairs, std::unordered_map<uint64_t, VkAccessFlags>& a_currImageFlags)
{
  if(a_pairs.size() == 0)
    return;
  
  VkImageSubresourceRange rangeWholeImage = {};
  rangeWholeImage.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  rangeWholeImage.baseMipLevel   = 0;
  rangeWholeImage.levelCount     = 1;
  rangeWholeImage.baseArrayLayer = 0;
  rangeWholeImage.layerCount     = 1;

  std::vector<VkImageMemoryBarrier> barriers(a_pairs.size());
  for(size_t i=0;i<a_pairs.size();i++)
  {
    VkImageMemoryBarrier& bar = barriers[i];
    bar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.pNext               = nullptr;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image               = a_pairs[i].image;
    bar.subresourceRange    = rangeWholeImage;

    uint64_t imageHandle = uint64_t(a_pairs[i].image);
    auto pState = a_currImageFlags.find(imageHandle);
    if(pState == a_currImageFlags.end())
    {
      bar.srcAccessMask = 0;
      bar.dstAccessMask = a_pairs[i].access;
      a_currImageFlags[imageHandle] = a_pairs[i].access;
      bar.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
      bar.newLayout     = (bar.dstAccessMask == VK_ACCESS_SHADER_READ_BIT) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    }
    else
    {
      bar.srcAccessMask = pState->second;
      bar.dstAccessMask = a_pairs[i].access;
      pState->second    = a_pairs[i].access;
      bar.oldLayout     = (bar.srcAccessMask == VK_ACCESS_SHADER_READ_BIT) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
      bar.newLayout     = (bar.dstAccessMask == VK_ACCESS_SHADER_READ_BIT) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    }
  }

  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,nullptr, 0,nullptr, uint32_t(barriers.size()),barriers.data());
}
{% endif %} {# /* 0 */ #}

{% endif %} {# /* length(TextureMembers) > 0 */ #}

void {{MainClassName}}_Generated::AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem)
{
  if(a_buffers.size() == 0 || a_mem == VK_NULL_HANDLE)
    return;

  std::vector<VkMemoryRequirements> memInfos(a_buffers.size());
  for(size_t i=0;i<memInfos.size();i++)
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkGetBufferMemoryRequirements(device, a_buffers[i], &memInfos[i]);
    else
    {
      memInfos[i] = memInfos[0];
      memInfos[i].size = 0;
    }
  }
  
  for(size_t i=1;i<memInfos.size();i++)
  {
    if(memInfos[i].memoryTypeBits != memInfos[0].memoryTypeBits)
    {
      std::cout << "[{{MainClassName}}_Generated::AssignBuffersToMemory]: error, input buffers has different 'memReq.memoryTypeBits'" << std::endl;
      return;
    }
  }

  auto offsets = vk_utils::calculateMemOffsets(memInfos);
  for (size_t i = 0; i < memInfos.size(); i++)
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkBindBufferMemory(device, a_buffers[i], a_mem, offsets[i]);
  }
}

{{MainClassName}}_Generated::MemLoc {{MainClassName}}_Generated::AllocAndBind(const std::vector<VkBuffer>& a_buffers)
{
  MemLoc currLoc;
  if(a_buffers.size() > 0)
  {
    currLoc.memObject = vk_utils::allocateAndBindWithPadding(device, physicalDevice, a_buffers);
    currLoc.allocId   = m_allMems.size();
    m_allMems.push_back(currLoc);
  }
  return currLoc;
}

{{MainClassName}}_Generated::MemLoc {{MainClassName}}_Generated::AllocAndBind(const std::vector<VkImage>& a_images)
{
  MemLoc currLoc;
  if(a_images.size() > 0)
  {
    std::vector<VkMemoryRequirements> reqs(a_images.size()); 
    for(size_t i=0; i<reqs.size(); i++)
      vkGetImageMemoryRequirements(device, a_images[i], &reqs[i]);

    for(size_t i=0; i<reqs.size(); i++)
    {
      if(reqs[i].memoryTypeBits != reqs[0].memoryTypeBits)
      {
        std::cout << "{{MainClassName}}_Generated::AllocAndBind(textures): memoryTypeBits warning, need to split mem allocation (override me)" << std::endl;
        break;
      }
    } 

    auto offsets  = vk_utils::calculateMemOffsets(reqs);
    auto memTotal = offsets[offsets.size() - 1];

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = memTotal;
    allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(reqs[0].memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicalDevice);
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &currLoc.memObject));
    
    for(size_t i=0;i<a_images.size();i++) {
      VK_CHECK_RESULT(vkBindImageMemory(device, a_images[i], currLoc.memObject, offsets[i]));
    }

    currLoc.allocId = m_allMems.size();
    m_allMems.push_back(currLoc);
  }
  return currLoc;
}

void {{MainClassName}}_Generated::FreeAllAllocations(std::vector<MemLoc>& a_memLoc)
{
  // in general you may check 'mem.allocId' for unique to be sure you dont free mem twice
  // for default implementation this is not needed
  for(auto mem : a_memLoc)
    vkFreeMemory(device, mem.memObject, nullptr);
  a_memLoc.resize(0);
}     

void {{MainClassName}}_Generated::AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_images)
{
  AllocAndBind(a_buffers);
  {% if length(ClassTextureVars) > 0 or length(ClassTexArrayVars) > 0 %}
  std::vector<VkFormat>             formats;  formats.reserve({{length(ClassTextureVars)}});
  std::vector<VkImageView*>         views;    views.reserve({{length(ClassTextureVars)}});
  std::vector<VkImage>              textures; textures.reserve({{length(ClassTextureVars)}});
  VkMemoryRequirements memoryRequirements;

  {% for Var in ClassTextureVars %}
  formats.push_back(VkFormat({{Var.Format}}));
  views.push_back(&m_vdata.{{Var.Name}}View);
  textures.push_back(m_vdata.{{Var.Name}}Texture);
  {% endfor %}
  {% for Var in ClassTexArrayVars %}
  for(size_t i=0;i< m_vdata.{{Var.Name}}ArrayTexture.size(); i++) 
  {
    formats.push_back (VkFormat({{Var.Name}}[i]->format()));
    views.push_back   (&m_vdata.{{Var.Name}}ArrayView[i]);
    textures.push_back(m_vdata.{{Var.Name}}ArrayTexture[i]);
  }
  {% endfor %}

  AllocAndBind(textures);
  for(size_t i=0;i<textures.size();i++)
  {
    VkImageViewCreateInfo imageViewInfo = {};
    imageViewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewInfo.flags                           = 0;
    imageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format                          = formats[i];
    imageViewInfo.components                      = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
    imageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewInfo.subresourceRange.baseMipLevel   = 0;
    imageViewInfo.subresourceRange.baseArrayLayer = 0;
    imageViewInfo.subresourceRange.layerCount     = 1;
    imageViewInfo.subresourceRange.levelCount     = 1;
    imageViewInfo.image                           = textures[i];     // The view will be based on the texture's image
    VK_CHECK_RESULT(vkCreateImageView(device, &imageViewInfo, nullptr, views[i]));
  }
  {% endif %}
}
