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

  struct KernelConfig
  {
    std::string kernelName;
    uint32_t    blockSize[3] = {1,1,1};
  };

  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount, 
                                 uint32_t a_blockSizeX, uint32_t a_blockSizeY, uint32_t a_blockSizeZ, 
                                 KernelConfig* a_kernelConfigs = nullptr, size_t a_configSize = 0) 
  {
    physicalDevice = a_physicalDevice;
    device         = a_device;
    
    m_blockSize[0] = a_blockSizeX;
    m_blockSize[1] = a_blockSizeY;
    m_blockSize[2] = a_blockSizeZ;

    InitHelpers();
    InitBuffers(a_maxThreadsCount);
    InitKernels("{{ShaderSingleFile}}.spv", a_blockSizeX, a_blockSizeY, a_blockSizeZ, a_kernelConfigs, a_configSize);
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

  VkBufferMemoryBarrier BarrierForUBOUpdate();

  virtual void InitHelpers();
  virtual void InitBuffers(size_t a_maxThreadsCount);
  virtual void InitKernels(const char* a_filePath, uint32_t a_blockSizeX, uint32_t a_blockSizeY, uint32_t a_blockSizeZ,
                           KernelConfig* a_kernelConfigs, size_t a_configSize);
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
## for Vector in VectorMembers
    VkBuffer {{Vector}}Buffer = VK_NULL_HANDLE;
    size_t   {{Vector}}Offset = 0;
## endfor
    VkDeviceMemory m_vecMem = VK_NULL_HANDLE;
  } m_vdata;

  VkBuffer m_classDataBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_allMem    = VK_NULL_HANDLE;

## for Kernel in Kernels
  VkPipelineLayout      {{Kernel.Name}}Layout   = VK_NULL_HANDLE;
  VkPipeline            {{Kernel.Name}}Pipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout {{Kernel.Name}}DSLayout = VK_NULL_HANDLE;  
  VkDescriptorSetLayout Create{{Kernel.Name}}DSLayout();
  {% if Kernel.HasLoopInit %}
  VkPipelineLayout      {{Kernel.Name}}LoopInitLayout   = VK_NULL_HANDLE;
  VkPipeline            {{Kernel.Name}}LoopInitPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout {{Kernel.Name}}LoopInitDSLayout = VK_NULL_HANDLE;  
  VkDescriptorSetLayout Create{{Kernel.Name}}LoopInitDSLayout(); 
  {% endif %}
## endfor

  VkPipelineLayout      copyKernelFloatLayout   = VK_NULL_HANDLE;
  VkPipeline            copyKernelFloatPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyKernelFloatDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatecopyKernelFloatDSLayout();

  VkDescriptorPool m_dsPool = VK_NULL_HANDLE;
  VkDescriptorSet m_allGeneratedDS[{{TotalDSNumber}}];
  uint32_t m_blockSize[3];
  std::unordered_map<std::string, KernelConfig> m_kernelExceptions;

  {{MainClassName}}_UBO_Data m_uboData;
  
};

#endif
