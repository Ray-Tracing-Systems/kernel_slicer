#ifndef MAIN_CLASS_DECL_{{MainClassName}}_H
#define MAIN_CLASS_DECL_{{MainClassName}}_H

#include <vector>
#include <memory>

#include "vulkan_basics.h"

{{Includes}}

class {{MainClassName}}_Generated : public {{MainClassName}}
{
public:

  {{MainClassName}}_Generated(VulkanContext a_vkContext) 
  {
    instance       = a_vkContext.instance;
    physicalDevice = a_vkContext.physicalDevice;
    device         = a_vkContext.device;
    commandPool    = a_vkContext.commandPool;
    computeQueue   = a_vkContext.computeQueue;
    transferQueue  = a_vkContext.transferQueue;
  }

  ~{{MainClassName}}_Generated(){}

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
  VkCommandPool    commandPool    = VK_NULL_HANDLE; 

  VkQueue computeQueue  = VK_NULL_HANDLE;
  VkQueue transferQueue = VK_NULL_HANDLE;

  VkCommandBuffer m_currCmdBuffer = VK_NULL_HANDLE;

  virtual void UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);

  {{PlainMembersUpdateFunctions}}
  {{VectorMembersUpdateFunctions}}
};

#endif
