#ifndef MAIN_CLASS_DECL_{{MainClassName}}_H
#define MAIN_CLASS_DECL_{{MainClassName}}_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#include "vulkan_basics.h"
#include "vk_compute_pipeline.h"
#include "vk_buffer.h"
#include "vk_utils.h"

{{Includes}}
#include "include/{{UBOIncl}}"

class {{MainClassName}}_Generated : public {{MainClassName}}
{
public:

  {{MainClassName}}_Generated() {}

  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount) 
  {
    physicalDevice = a_physicalDevice;
    device         = a_device;
    InitHelpers();
    InitBuffers(a_maxThreadsCount);
    InitKernels("{{ShaderSingleFile}}.spv");
    AllocateAllDescriptorSets();
  }

## for MainFunc in MainFunctions
  virtual void SetVulkanInOutFor_{{MainFunc.Name}}(
## for BufferName in MainFunc.InOutVars
    VkBuffer a_{{BufferName}}Buffer,
    size_t   a_{{BufferName}}Offset,
## endfor
    uint32_t dummyArgument = 0)
  {
## for BufferName in MainFunc.InOutVars
    {{MainFunc.Name}}_local.{{BufferName}}Buffer = a_{{BufferName}}Buffer;
    {{MainFunc.Name}}_local.{{BufferName}}Offset = a_{{BufferName}}Offset;
## endfor
    InitAllGeneratedDescriptorSets_{{MainFunc.Name}}();
  }

## endfor
  virtual ~{{MainClassName}}_Generated();

  virtual void InitMemberBuffers();

  virtual void UpdateAll(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
  {
    UpdatePlainMembers(a_pCopyEngine);
    UpdateVectorMembers(a_pCopyEngine);
  }

## for MainFunc in MainFunctions  
  {{MainFunc.Decl}}
## endfor

  virtual void copyKernelFloatCmd(uint32_t length);
  
  {{KernelsDecl}}

protected:
  
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice         device         = VK_NULL_HANDLE;

  VkCommandBuffer  m_currCmdBuffer   = VK_NULL_HANDLE;
  uint32_t         m_currThreadFlags = 0;

  std::unique_ptr<vkfw::ComputePipelineMaker> m_pMaker = nullptr;
  VkPhysicalDeviceProperties m_devProps;

  VkBufferMemoryBarrier BarrierForClearFlags(VkBuffer a_buffer);
  VkBufferMemoryBarrier BarrierForSingleBuffer(VkBuffer a_buffer);
  void BarriersForSeveralBuffers(VkBuffer* a_inBuffers, VkBufferMemoryBarrier* a_outBarriers, uint32_t a_buffersNum);

  virtual void InitHelpers();
  virtual void InitBuffers(size_t a_maxThreadsCount);
  virtual void InitKernels(const char* a_filePath);
  virtual void AllocateAllDescriptorSets();

## for MainFunc in MainFunctions
  virtual void InitAllGeneratedDescriptorSets_{{MainFunc.Name}}();
## endfor

  virtual void UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);

  {{PlainMembersUpdateFunctions}}
  {{VectorMembersUpdateFunctions}}

## for MainFunc in MainFunctions  
  struct {{MainFunc.Name}}_Data
  {
## for Buffer in MainFunc.LocalVarsBuffersDecl
    VkBuffer {{Buffer.Name}}Buffer = VK_NULL_HANDLE;
    size_t   {{Buffer.Name}}Offset = 0;

## endfor
## for BufferName in MainFunc.InOutVars
    VkBuffer {{BufferName}}Buffer = VK_NULL_HANDLE;
    size_t   {{BufferName}}Offset = 0;

## endfor
  } {{MainFunc.Name}}_local;

## endfor

  struct StdVectorMembersGPUData
  {
    {% for Vector in VectorMembers %}
    VkBuffer {{Vector}}Buffer = VK_NULL_HANDLE;
    size_t   {{Vector}}Offset = 0;
    {% endfor %}
    VkDeviceMemory m_vecMem = VK_NULL_HANDLE;
  } m_vdata;

  size_t m_maxThreadCount = 0;

  {% for Hierarchy in DispatchHierarchies %}
  // Auxilary data and kernels for 'VirtualKernels'; Dispatch hierarchy of '{{Hierarchy.Name}}'
  //
  VkBuffer         m_{{Hierarchy.Name}}ObjPtrBuffer = VK_NULL_HANDLE;
  size_t           m_{{Hierarchy.Name}}ObjPtrOffset = 0;
  {% if Hierarchy.IndirectDispatch %}
  VkPipelineLayout {{Hierarchy.Name}}ZeroObjCountersLayout   = VK_NULL_HANDLE;
  VkPipeline       {{Hierarchy.Name}}ZeroObjCountersPipeline = VK_NULL_HANDLE; 
  void             {{Hierarchy.Name}}ZeroObjCountersCmd();
  {% endif %} 
  {% endfor %}
  {% if length(DispatchHierarchies) > 0 %}
  VkBufferMemoryBarrier BarrierForObjCounters(VkBuffer a_buffer);
  {% endif %} 

  {% if length(IndirectDispatches) > 0 %}
  void InitIndirectBufferUpdateResources(const char* a_filePath);
  void InitIndirectDescriptorSets();
  VkBufferMemoryBarrier BarrierForIndirectBufferUpdate(VkBuffer a_buffer);
  VkBuffer              m_indirectBuffer  = VK_NULL_HANDLE;
  VkPipelineLayout      m_indirectUpdateLayout   = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_indirectUpdateDSLayout = VK_NULL_HANDLE;
  VkDescriptorSet       m_indirectUpdateDS       = VK_NULL_HANDLE;
  {% for Dispatch in IndirectDispatches %}
  VkPipeline            m_indirectUpdate{{Dispatch.KernelName}}Pipeline = VK_NULL_HANDLE; 
  {% endfor %}
  {% endif %}

  VkBuffer m_classDataBuffer = VK_NULL_HANDLE;
  {% if UseSeparateUBO %}
  VkBuffer m_uboArgsBuffer = VK_NULL_HANDLE;
  VkBufferMemoryBarrier BarrierForArgsUBO(size_t a_size);
  {% endif %}
  VkDeviceMemory m_allMem    = VK_NULL_HANDLE;

  {% for Kernel in Kernels %}
  VkPipelineLayout      {{Kernel.Name}}Layout   = VK_NULL_HANDLE;
  VkPipeline            {{Kernel.Name}}Pipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout {{Kernel.Name}}DSLayout = VK_NULL_HANDLE;
  {% if Kernel.HasLoopInit %}
  VkPipeline            {{Kernel.Name}}InitPipeline = VK_NULL_HANDLE;
  {% endif %} 
  {% if Kernel.HasLoopFinish %}
  VkPipeline            {{Kernel.Name}}FinishPipeline = VK_NULL_HANDLE;
  {% endif %}   
  {% if Kernel.FinishRed %}
  VkPipeline            {{Kernel.Name}}ReductionPipeline = VK_NULL_HANDLE; 
  {% endif %}  
  {% if Kernel.IsMaker and Kernel.Hierarchy.IndirectDispatch %}
  VkPipeline            {{Kernel.Name}}ZeroObjCounters    = VK_NULL_HANDLE;
  VkPipeline            {{Kernel.Name}}CountTypeIntervals = VK_NULL_HANDLE;
  VkPipeline            {{Kernel.Name}}Sorter             = VK_NULL_HANDLE; 
  {% endif %}  
  VkDescriptorSetLayout Create{{Kernel.Name}}DSLayout();
  void InitKernel_{{Kernel.Name}}(const char* a_filePath);
  {% if Kernel.IsIndirect %}
  void {{Kernel.Name}}_UpdateIndirect();
  {% endif %}
  {% endfor %}

  {% if UseSpecConstWgSize %}
  VkSpecializationMapEntry m_specializationEntriesWgSize[3];
  VkSpecializationInfo     m_specsForWGSize;
  {% endif %}

  virtual VkBufferUsageFlags GetAdditionalFlagsForUBO() const;

  VkPipelineLayout      copyKernelFloatLayout   = VK_NULL_HANDLE;
  VkPipeline            copyKernelFloatPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyKernelFloatDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatecopyKernelFloatDSLayout();

  VkDescriptorPool m_dsPool = VK_NULL_HANDLE;
  VkDescriptorSet  m_allGeneratedDS[{{TotalDSNumber}}];

  {{MainClassName}}_UBO_Data m_uboData;
  
  constexpr static uint32_t MEMCPY_BLOCK_SIZE = 256;
  constexpr static uint32_t REDUCTION_BLOCK_SIZE = 256;
};

#endif
