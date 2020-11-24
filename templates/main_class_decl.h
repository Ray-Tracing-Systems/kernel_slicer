#ifndef MAIN_CLASS_DECL_{{MainClassName}}_H
#define MAIN_CLASS_DECL_{{MainClassName}}_H

#include <vector>
#include <memory>

#include "vulkan_basics.h"
#include "vk_program.h"

{{Includes}}

class {{MainClassName}}_Generated : public {{MainClassName}}
{
public:

  {{MainClassName}}_Generated(VulkanContext a_vkContext) 
  {
    instance       = a_vkContext.instance;
    physicalDevice = a_vkContext.physicalDevice;
    device         = a_vkContext.device;
    computeQueue   = a_vkContext.computeQueue;
    transferQueue  = a_vkContext.transferQueue;
    InitHelpers();
    InitBuffers();
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
  
  VkInstance       instance       = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice         device         = VK_NULL_HANDLE;

  VkQueue computeQueue  = VK_NULL_HANDLE;
  VkQueue transferQueue = VK_NULL_HANDLE;

  VkCommandBuffer m_currCmdBuffer = VK_NULL_HANDLE;

  vkfw::ComputePipelineMaker m_pMaker    = nullptr;
  vkfw::ProgramBindings      m_pBindings = nullptr;
  VkPhysicalDeviceProperties m_devProps;

  void InitHelpers();
  virtual void InitBuffers();

  virtual void UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);

  {{PlainMembersUpdateFunctions}}
  {{VectorMembersUpdateFunctions}}

  {{LocalVarsBuffersDecl}}
  VkBuffer m_classDataBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_allMem    = VK_NULL_HANDLE;
};

#endif
