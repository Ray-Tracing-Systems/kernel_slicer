#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#include "vk_pipeline.h"
#include "vk_buffers.h"
#include "vk_utils.h"
#include "vk_copy.h"
#include "vk_context.h"

{% if length(TextureMembers) > 0 or length(ClassTexArrayVars) > 0 %}
#include "Image2d.h"
using LiteImage::Image2D;
using LiteImage::Sampler;
using namespace LiteMath;
{% endif %}

#include "{{MainInclude}}"
{% if GenGpuApi %}
#include "{{MainIncludeApi}}"
{% endif %}
{% for Include in AdditionalIncludes %}
#include "{{Include}}"
{% endfor %}

## for Decl in ClassDecls  
{% if Decl.InClass and Decl.IsType %}
using {{Decl.Type}} = {{MainClassName}}::{{Decl.Type}}; // for passing this data type to UBO
{% endif %}
## endfor

#include "include/{{UBOIncl}}"

{% for SetterDecl in SettersDecl %}  
{{SetterDecl}}

{% endfor %}
{% if GenGpuApi %}
class {{MainClassName}}{{MainClassSuffix}} : public {{MainClassName}}, public I{{MainClassName}}{{MainClassSuffix}}
{% else %}
class {{MainClassName}}{{MainClassSuffix}} : public {{MainClassName}}
{% endif %}
{
public:

  {% for ctorDecl in Constructors %}
  {% if ctorDecl.NumParams == 0 %}
  {{ctorDecl.ClassName}}{{MainClassSuffix}}() 
  {
    {% if HasPrefixData %}
    if({{PrefixDataName}} == nullptr)
      {{PrefixDataName}} = std::make_shared<{{PrefixDataClass}}>();
    {% endif %}
  }
  {% else %}
  {{ctorDecl.ClassName}}{{MainClassSuffix}}({{ctorDecl.Params}}) : {{ctorDecl.ClassName}}({{ctorDecl.PrevCall}}) 
  {
    {% if HasPrefixData %}
    if({{PrefixDataName}} == nullptr)
      {{PrefixDataName}} = std::make_shared<{{PrefixDataClass}}>();
    {% endif %}
  }
  {% endif %}
  {% endfor %}
  {% if HasNameFunc %}
  const char* Name() const override { return "{{MainClassName}}{{MainClassSuffix}}";}
  {% endif %}
  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount);
  virtual void SetVulkanContext(vk_utils::VulkanContext a_ctx) { m_ctx = a_ctx; }

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
  virtual ~{{MainClassName}}{{MainClassSuffix}}();

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
  
  {% if HasPrefixData %}
  virtual void UpdatePrefixPointers()
  {
    auto pUnderlyingImpl = dynamic_cast<{{PrefixDataClass}}*>({{PrefixDataName}}.get());
    if(pUnderlyingImpl != nullptr)
    {
      {% for Vector in VectorMembers %}
      {% if Vector.HasPrefix %}
      {{Vector.Name}} = &pUnderlyingImpl->{{Vector.CleanName}};
      {% endif %}
      {% endfor %}
    }
  }
  {% endif %}
  {% if HasCommitDeviceFunc %}
  virtual void CommitDeviceData() override // you have to define this virtual function in the original imput class
  {
    {% if HasPrefixData %}
    UpdatePrefixPointers(); 
    {% endif %}
    InitMemberBuffers();
    UpdateAll(m_ctx.pCopyHelper);
  }  
  {% endif %}
  {% if HasGetTimeFunc %}
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override; 
  {% endif %}
  {% if UpdateMembersPlainData %}
  void UpdateMembersPlainData() override { UpdatePlainMembers(m_ctx.pCopyHelper); } 
  {% endif %}
  {% if UpdateMembersVectorData %}
  void UpdateMembersVectorData() override { UpdateVectorMembers(m_ctx.pCopyHelper); }
  {% endif %}
  {% if UpdateMembersTextureData %}
  void UpdateMembersTexureData() override { UpdateTextureMembers(m_ctx.pCopyHelper); }
  {% endif %}
  
  virtual void UpdatePlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateTextureMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  {% if HasFullImpl %}
  virtual void ReadPlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  {% endif %}

  {% if not GenGpuApi %}virtual {%endif%}VkPhysicalDeviceFeatures2 ListRequiredDeviceFeatures() {% if GenGpuApi %} override{% endif %};
  {% for MainFunc in MainFunctions %}  
  {% if not GenGpuApi %}virtual {%endif%}{{MainFunc.ReturnType}}   {{MainFunc.Decl}} {% if GenGpuApi %} override{% endif %};
  {% endfor %}
  {% if HasFullImpl %}

  {% for MainFunc in MainFunctions %}  
  {% if MainFunc.OverrideMe %}
  {{MainFunc.ReturnType}} {{MainFunc.DeclOrig}} override;
  {% endif %}
  {% endfor %}

  {% for MainFunc in MainFunctions %}  
  {% if MainFunc.OverrideMe %}
  inline vk_utils::ExecTime Get{{MainFunc.Name}}ExecutionTime() const { return m_exTime{{MainFunc.Name}}; }
  {% endif %}
  {% endfor %}

  {% for MainFunc in MainFunctions %}  
  {% if MainFunc.OverrideMe %}
  vk_utils::ExecTime m_exTime{{MainFunc.Name}};
  {% endif %}
  {% endfor %}
  {% endif %} {# /* end if HasFullImpl */ #}

  virtual void copyKernelFloatCmd(uint32_t length);
  
  {% for KernelDecl in KernelsDecls %}
  {{KernelDecl}}
  {% endfor %}
  
  struct MemLoc
  {
    VkDeviceMemory memObject = VK_NULL_HANDLE;
    size_t         memOffset = 0;
    size_t         allocId   = 0;
  };

  virtual MemLoc AllocAndBind(const std::vector<VkBuffer>& a_buffers); ///< replace this function to apply custom allocator
  virtual MemLoc AllocAndBind(const std::vector<VkImage>& a_image);    ///< replace this function to apply custom allocator
  virtual void   FreeAllAllocations(std::vector<MemLoc>& a_memLoc);    ///< replace this function to apply custom allocator

protected:

  VkPhysicalDevice           physicalDevice = VK_NULL_HANDLE;
  VkDevice                   device         = VK_NULL_HANDLE;
  vk_utils::VulkanContext    m_ctx          = {};
  VkCommandBuffer            m_currCmdBuffer   = VK_NULL_HANDLE;
  uint32_t                   m_currThreadFlags = 0;
  std::vector<MemLoc>        m_allMems;
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

  virtual void AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem);

  virtual void AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_image);
  virtual std::string AlterShaderPath(const char* in_shaderPath) { return std::string("{{ShaderFolderPrefix}}") + std::string(in_shaderPath); }

  {{PlainMembersUpdateFunctions}}
  {{VectorMembersUpdateFunctions}}

## for MainFunc in MainFunctions  
  struct {{MainFunc.Name}}_Data
  {
    {% if MainFunc.IsRTV and not MainFunc.IsMega %}
    {% for Buffer in MainFunc.LocalVarsBuffersDecl %}
    VkBuffer {{Buffer.Name}}Buffer = VK_NULL_HANDLE;
    size_t   {{Buffer.Name}}Offset = 0;
    {% endfor %}
    {% endif %}
    {% for Arg in MainFunc.InOutVars %}
    {% if Arg.IsTexture %}
    VkImage     {{Arg.Name}}Text = VK_NULL_HANDLE;
    VkImageView {{Arg.Name}}View = VK_NULL_HANDLE;
    {% else %}
    VkBuffer {{Arg.Name}}Buffer = VK_NULL_HANDLE;
    size_t   {{Arg.Name}}Offset = 0;
    {% endif %}
    {% endfor %}
    bool needToClearOutput = {% if MainFunc.IsRTV %}true{% else %}false{% endif %};
  } {{MainFunc.Name}}_local;

## endfor

  {% for var in SetterVars %}  
  {{var.Type}}Vulkan {{var.Name}}Vulkan;
  {% endfor %}

  struct MembersDataGPU
  {
    {% for Vector in VectorMembers %}
    VkBuffer {{Vector.Name}}Buffer = VK_NULL_HANDLE;
    size_t   {{Vector.Name}}Offset = 0;
    {% endfor %}
    {% for Tex in TextureMembers %}
    VkImage     {{Tex}}Texture = VK_NULL_HANDLE;
    VkImageView {{Tex}}View    = VK_NULL_HANDLE;
    VkSampler   {{Tex}}Sampler = VK_NULL_HANDLE; ///<! aux sampler, may not be used
    {% endfor %}
    {% for Tex in TexArrayMembers %}
    std::vector<VkImage>     {{Tex}}ArrayTexture;
    std::vector<VkImageView> {{Tex}}ArrayView   ;
    std::vector<VkSampler>   {{Tex}}ArraySampler; ///<! samplers for texture arrays are always used
    size_t                   {{Tex}}ArrayMaxSize; ///<! used when texture array size is not known after constructor of base class is finished
    {% endfor %}
    {% for Sam in SamplerMembers %}
    VkSampler      {{Sam}} = VK_NULL_HANDLE;
    {% endfor %}
  } m_vdata;
  
  {% for Vector in VectorMembers %}
  {% if Vector.HasPrefix %}
  {{Vector.Type}}* {{Vector.Name}} = nullptr;
  {% endif %}
  {% endfor %}
  
  {% if length(TextureMembers) > 0 or length(ClassTexArrayVars) > 0 %}
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
  size_t                m_indirectOffset  = 0;
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
  virtual void InitKernel_{{Kernel.Name}}(const char* a_filePath);
  {% if Kernel.IsIndirect %}
  virtual void {{Kernel.Name}}_UpdateIndirect();
  {% endif %}
  {% endfor %}

  {% if UseSpecConstWgSize %}
  VkSpecializationMapEntry m_specializationEntriesWgSize[3];
  VkSpecializationInfo     m_specsForWGSize;
  {% endif %}

  virtual VkBufferUsageFlags GetAdditionalFlagsForUBO() const;
  virtual uint32_t           GetDefaultMaxTextures() const;

  VkPipelineLayout      copyKernelFloatLayout   = VK_NULL_HANDLE;
  VkPipeline            copyKernelFloatPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyKernelFloatDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatecopyKernelFloatDSLayout();

  VkDescriptorPool m_dsPool = VK_NULL_HANDLE;
  VkDescriptorSet  m_allGeneratedDS[{{TotalDSNumber}}];

  {{MainClassName}}{{MainClassSuffix}}_UBO_Data m_uboData;
  
  constexpr static uint32_t MEMCPY_BLOCK_SIZE = 256;
  constexpr static uint32_t REDUCTION_BLOCK_SIZE = 256;

  {% if GenerateSceneRestrictions %}
  virtual void SceneRestrictions(uint32_t a_restrictions[4]) const
  {
    uint32_t maxMeshes            = 1024;
    uint32_t maxTotalVertices     = 1'000'000;
    uint32_t maxTotalPrimitives   = 1'000'000;
    uint32_t maxPrimitivesPerMesh = 200'000;

    a_restrictions[0] = maxMeshes;
    a_restrictions[1] = maxTotalVertices;
    a_restrictions[2] = maxTotalPrimitives;
    a_restrictions[3] = maxPrimitivesPerMesh;
  }
  {% endif %}
  {% if UseServiceScan %}
  struct ScanData
  {
    VkBuffer              m_scanTempDataBuffer;
    size_t                m_scanTempDataOffset = 0;
    std::vector<size_t>   m_scanMipOffsets;
    size_t                m_scanMaxSize;
    std::vector<VkBuffer> InitTempBuffers(VkDevice a_device, size_t a_maxSize);
    void                  DeleteTempBuffers(VkDevice a_device);
    
    void ExclusiveScanCmd(VkCommandBuffer a_cmdBuffer, size_t a_size);
    void InclusiveScanCmd(VkCommandBuffer a_cmdBuffer, size_t a_size, bool actuallyExclusive = false);

    VkDescriptorSetLayout CreateInternalScanDSLayout(VkDevice a_device);
    void                  DeleteDSLayouts(VkDevice a_device);

    VkDescriptorSetLayout internalDSLayout = VK_NULL_HANDLE;
    VkPipelineLayout      scanFwdLayout   = VK_NULL_HANDLE;
    VkPipeline            scanFwdPipeline = VK_NULL_HANDLE;

    VkPipelineLayout      scanPropLayout   = VK_NULL_HANDLE;
    VkPipeline            scanPropPipeline = VK_NULL_HANDLE;

  };
  {% for Scan in ServiceScan %}
  ScanData m_scan_{{Scan.Type}};
  {% endfor %}
  {% endif %}
  {% if UseServiceSort %}
  struct BitonicSortData
  {
    VkDescriptorSetLayout CreateSortDSLayout(VkDevice a_device);
    VkDescriptorSetLayout sortDSLayout        = VK_NULL_HANDLE;
    VkPipelineLayout      bitonicPassLayout   = VK_NULL_HANDLE;
    VkPipeline            bitonicPassPipeline = VK_NULL_HANDLE;
    VkPipelineLayout      bitonic512Layout    = VK_NULL_HANDLE;
    VkPipeline            bitonic512Pipeline  = VK_NULL_HANDLE;
    VkPipelineLayout      bitonic1024Layout   = VK_NULL_HANDLE;
    VkPipeline            bitonic1024Pipeline = VK_NULL_HANDLE;
    VkPipelineLayout      bitonic2048Layout   = VK_NULL_HANDLE;
    VkPipeline            bitonic2048Pipeline = VK_NULL_HANDLE;
    void DeleteDSLayouts(VkDevice a_device);
    void BitonicSortCmd(VkCommandBuffer a_cmdBuffer, size_t a_size, uint32_t a_maxWorkGroupSize = 256);
    void BitonicSortSimpleCmd(VkCommandBuffer a_cmdBuffer, size_t a_size);
  };
  {% for Sort in ServiceSort %}
  BitonicSortData m_sort_{{Sort.Type}};
  {% endfor %}
  {% endif %}
  virtual void MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, 
                                            VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline);
  virtual void MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout, 
                                       VkPipeline* pPipeline);

  std::vector<VkPipelineLayout> m_allCreatedPipelineLayouts; ///<! remenber them here to delete later
  std::vector<VkPipeline>       m_allCreatedPipelines;       ///<! remenber them here to delete later
  {% if length(SpecConstants) > 0 %}
  std::vector<uint32_t>                  m_allSpecConstVals; ///<! require user to define "ListRequiredFeatures" func.    
  std::vector<VkSpecializationMapEntry>  m_allSpecConstInfo;
  VkSpecializationInfo                   m_allSpecInfo;
  const VkSpecializationInfo*            GetAllSpecInfo();
  {% endif %}
};

