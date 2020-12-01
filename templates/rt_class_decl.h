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
                                 uint32_t a_blockSizeX, uint32_t a_blockSizeY, uint32_t a_blockSizeZ) 
  {
    physicalDevice = a_physicalDevice;
    device         = a_device;
    
    m_blockSize[0] = a_blockSizeX;
    m_blockSize[1] = a_blockSizeY;
    m_blockSize[2] = a_blockSizeZ;

    InitHelpers();
    InitBuffers(a_maxThreadsCount);
    InitKernels("z_generated.cl.spv", a_blockSizeX, a_blockSizeY, a_blockSizeZ);
    AllocateAllDescriptorSets();
  }

## for MainFunc in MainFunctions
  virtual void SetVulkanInputOutputFor_{{MainFunc.Name}}(
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
  ~{{MainClassName}}_Generated();

  virtual void UpdateAll(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
  {
    UpdatePlainMembers(a_pCopyEngine);
    UpdateVectorMembers(a_pCopyEngine);
  }

## for MainFunc in MainFunctions  
  {{MainFunc.Decl}}
## endfor
  
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
  virtual void AllocateAllDescriptorSets();

## for MainFunc in MainFunctions
  virtual void InitAllGeneratedDescriptorSets_{{MainFunc.Name}}();
  bool m_dsAllocatedFor_{{MainFunc.Name}} = false;
## endfor

  virtual void UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);

  {{PlainMembersUpdateFunctions}}
  {{VectorMembersUpdateFunctions}}

## for MainFunc in MainFunctions  
  struct {{MainFunc.Name}}_Data
  {
## for BufferName in MainFunc.LocalVarsBuffersDecl
    VkBuffer {{BufferName}}Buffer = VK_NULL_HANDLE;
    size_t   {{BufferName}}Offset = 0;

## endfor
## for BufferName in MainFunc.InOutVars
    VkBuffer {{BufferName}}Buffer = VK_NULL_HANDLE;
    size_t   {{BufferName}}Offset = 0;

## endfor
  } {{MainFunc.Name}}_local;

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
