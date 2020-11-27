#ifndef MAIN_CLASS_DECL_{{MainClassName}}_H
#define MAIN_CLASS_DECL_{{MainClassName}}_H

#include <vector>
#include <memory>

#include "vulkan_basics.h"
#include "vk_compute_pipeline.h"
#include "vk_buffer.h"
#include "vk_utils.h"

{{Includes}}

class {{MainClassName}}_Generated : public {{MainClassName}}
{
public:

  {{MainClassName}}_Generated() {}

  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount, 
                                 uint32_t a_blockSizeX, uint32_t a_blockSizeY = 1, uint32_t a_blockSizeZ = 1) 
  {
    physicalDevice = a_physicalDevice;
    device         = a_device;
    
    m_blockSize[0] = a_blockSizeX;
    m_blockSize[1] = a_blockSizeY;
    m_blockSize[2] = a_blockSizeZ;

    InitHelpers();
    InitBuffers(a_maxThreadsCount);
    InitKernels("z_generated.cl.spv", a_blockSizeX, a_blockSizeY, a_blockSizeZ);
  }

  virtual void SetVulkanInputOutput(
## for BufferName in InOutVars
    VkBuffer a_{{BufferName}}Buffer,
    size_t   a_{{BufferName}}Offset,
## endfor
    uint32_t dummyArgument = 0)
  {
## for BufferName in InOutVars
    {{BufferName}}Buffer = a_{{BufferName}}Buffer;
    {{BufferName}}Offset = a_{{BufferName}}Offset;
## endfor
    InitAllGeneratedDescriptorSets();
  }

  ~{{MainClassName}}_Generated();

  virtual void UpdateAll(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
  {
    UpdatePlainMembers(a_pCopyEngine);
    UpdateVectorMembers(a_pCopyEngine);
  }

  {{MainFuncDecl}}
  
  {{KernelsDecl}}

protected:
  
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice         device         = VK_NULL_HANDLE;

  VkCommandBuffer  m_currCmdBuffer = VK_NULL_HANDLE;

  std::unique_ptr<vkfw::ComputePipelineMaker> m_pMaker = nullptr;
  VkPhysicalDeviceProperties m_devProps;

  virtual void InitHelpers();
  virtual void InitBuffers(size_t a_maxThreadsCount);
  virtual void InitKernels(const char* a_filePath, uint32_t a_blockSizeX, uint32_t a_blockSizeY, uint32_t a_blockSizeZ);
  virtual void InitAllGeneratedDescriptorSets();

  virtual void UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);

  {{PlainMembersUpdateFunctions}}
  {{VectorMembersUpdateFunctions}}

## for BufferName in LocalVarsBuffersDecl
  VkBuffer {{BufferName}}Buffer = VK_NULL_HANDLE;
  size_t   {{BufferName}}Offset = 0;
## endfor

## for BufferName in InOutVars
  VkBuffer {{BufferName}}Buffer = VK_NULL_HANDLE;
  size_t   {{BufferName}}Offset = 0;
## endfor

  VkBuffer m_classDataBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_allMem    = VK_NULL_HANDLE;

## for KernelName in KernelNames
  VkPipelineLayout      {{KernelName}}Layout   = VK_NULL_HANDLE;
  VkPipeline            {{KernelName}}Pipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout {{KernelName}}DSLayout = VK_NULL_HANDLE;  
  VkDescriptorSetLayout Create{{KernelName}}DSLayout();

## endfor

  VkDescriptorPool m_dsPool = VK_NULL_HANDLE;
  VkDescriptorSet m_allGeneratedDS[{{TotalDSNumber}}];
  uint32_t m_blockSize[3];
  
};

#endif
