#ifndef MAIN_CLASS_DECL_{{MainClassName}}_H
#define MAIN_CLASS_DECL_{{MainClassName}}_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#include "vulkan_basics.h"

#include "vk_pipeline.h"
#include "vk_buffers.h"
#include "vk_utils.h"

{{Includes}}
#include "include/{{UBOIncl}}"

{% for SetterDecl in SettersDecl %}  
{{SetterDecl}}

{% endfor %}
class {{MainClassName}}_Generated : public {{MainClassName}}
{
public:

  {% for ctorDecl in Constructors %}
  {{ctorDecl}}
  {% endfor %}
  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount);

## for MainFunc in MainFunctions
  virtual void SetVulkanInOutFor_{{MainFunc.Name}}(
## for Arg in MainFunc.InOutVars
    {% if Arg.IsTexture %}
    VkImage     {{Arg.Name}}Text,
    VkImageView {{Arg.Name}}View,
    {% else %}
    VkBuffer {{Arg.Name}}Buffer,
    size_t   {{Arg.Name}}Offset,
    {% endif %}
## endfor
    uint32_t dummyArgument = 0)
  {
## for Arg in MainFunc.InOutVars
    {% if Arg.IsTexture %}
    {{MainFunc.Name}}_local.{{Arg.Name}}Text   = {{Arg.Name}}Text;
    {{MainFunc.Name}}_local.{{Arg.Name}}View   = {{Arg.Name}}View;
    {% else %}
    {{MainFunc.Name}}_local.{{Arg.Name}}Buffer = {{Arg.Name}}Buffer;
    {{MainFunc.Name}}_local.{{Arg.Name}}Offset = {{Arg.Name}}Offset;
    {% endif %}
## endfor
    InitAllGeneratedDescriptorSets_{{MainFunc.Name}}();
  }

## endfor
  virtual ~{{MainClassName}}_Generated();

  {% for SetterFunc in SetterFuncs %}  
  {{SetterFunc}}
  {% endfor %}

  virtual void InitMemberBuffers();

  virtual void UpdateAll(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
  {
    UpdatePlainMembers(a_pCopyEngine);
    UpdateVectorMembers(a_pCopyEngine);
    UpdateTextureMembers(a_pCopyEngine);
  }
  
  virtual void UpdatePlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateTextureMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);

  {% for MainFunc in MainFunctions %}  
  virtual {{MainFunc.ReturnType}} {{MainFunc.Decl}};
  {% endfor %}

  virtual void copyKernelFloatCmd(uint32_t length);
  
  {% for KernelDecl in KernelsDecls %}
  {{KernelDecl}}
  {% endfor %}
protected:
  
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice         device         = VK_NULL_HANDLE;

  VkCommandBuffer  m_currCmdBuffer   = VK_NULL_HANDLE;
  uint32_t         m_currThreadFlags = 0;

  std::unique_ptr<vk_utils::ComputePipelineMaker> m_pMaker = nullptr;
  VkPhysicalDeviceProperties m_devProps;

  VkBufferMemoryBarrier BarrierForClearFlags(VkBuffer a_buffer);
  VkBufferMemoryBarrier BarrierForSingleBuffer(VkBuffer a_buffer);
  void BarriersForSeveralBuffers(VkBuffer* a_inBuffers, VkBufferMemoryBarrier* a_outBarriers, uint32_t a_buffersNum);

  virtual void InitHelpers();
  virtual void InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay = true);
  virtual void InitKernels(const char* a_filePath);
  virtual void AllocateAllDescriptorSets();

## for MainFunc in MainFunctions
  virtual void InitAllGeneratedDescriptorSets_{{MainFunc.Name}}();
## endfor

  virtual void AllocMemoryForInternalBuffers(const std::vector<VkBuffer>& a_buffers);
  virtual void AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem);

  virtual void AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_image);
  
  virtual void FreeMemoryForInternalBuffers();
  virtual void FreeMemoryForMemberBuffersAndImages();
  virtual std::string AlterShaderPath(const char* in_shaderPath) { return std::string(in_shaderPath); }

  {{PlainMembersUpdateFunctions}}
  {{VectorMembersUpdateFunctions}}

## for MainFunc in MainFunctions  
  struct {{MainFunc.Name}}_Data
  {
## for Buffer in MainFunc.LocalVarsBuffersDecl
    VkBuffer {{Buffer.Name}}Buffer = VK_NULL_HANDLE;
    size_t   {{Buffer.Name}}Offset = 0;

## endfor
## for Arg in MainFunc.InOutVars
    {% if Arg.IsTexture %}
    VkImage     {{Arg.Name}}Text = VK_NULL_HANDLE;
    VkImageView {{Arg.Name}}View = VK_NULL_HANDLE;
    {% else %}
    VkBuffer {{Arg.Name}}Buffer = VK_NULL_HANDLE;
    size_t   {{Arg.Name}}Offset = 0;
    {% endif %}
## endfor
  } {{MainFunc.Name}}_local;

## endfor

  {% for var in SetterVars %}  
  {{var.Type}}Vulkan {{var.Name}}Vulkan;
  {% endfor %}

  struct MembersDataGPU
  {
    {% for Vector in VectorMembers %}
    VkBuffer {{Vector}}Buffer = VK_NULL_HANDLE;
    size_t   {{Vector}}Offset = 0;
    {% endfor %}
    {% for Tex in TextureMembers %}
    VkImage     {{Tex}}Texture = VK_NULL_HANDLE;
    VkImageView {{Tex}}View    = VK_NULL_HANDLE;
    {% endfor %}
    VkDeviceMemory m_vecMem = VK_NULL_HANDLE;
    VkDeviceMemory m_texMem = VK_NULL_HANDLE;
    {% for Sam in SamplerMembers %}
    VkSampler      {{Sam}} = VK_NULL_HANDLE;
    {% endfor %}
  } m_vdata;

  {% if length(TextureMembers) > 0 %}
  VkImage   CreateTexture2D(const int a_width, const int a_height, VkFormat a_format, VkImageUsageFlags a_usage);
  VkSampler CreateSampler(const Sampler& a_sampler);
  struct TexAccessPair
  {
    TexAccessPair() : image(VK_NULL_HANDLE), access(0) {}
    TexAccessPair(VkImage a_image, VkAccessFlags a_access) : image(a_image), access(a_access) {}
    VkImage image;
    VkAccessFlags access;  
  };
  void TrackTextureAccess(const std::vector<TexAccessPair>& a_pairs, std::unordered_map<uint64_t, VkAccessFlags>& a_currImageFlags);
  {% endif %} {# /* length(TextureMembers) > 0 */ #}
  {% if length(DispatchHierarchies) > 0 %}
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
  VkBufferMemoryBarrier BarrierForObjCounters(VkBuffer a_buffer);
  {% endif %} {# /* length(DispatchHierarchies) > 0 */ #}
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
  {% endif %} {# /* length(IndirectDispatches) > 0 */ #}
  size_t m_maxThreadCount = 0;
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
  {% if Kernel.IsVirtual and Kernel.Hierarchy.IndirectDispatch %}
  VkPipeline            {{Kernel.Name}}PipelineArray[{{length(Kernel.Hierarchy.Implementations)}}] = {};
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
