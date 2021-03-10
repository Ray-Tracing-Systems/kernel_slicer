#ifndef VULKAN_COPY_H
#define VULKAN_COPY_H

#if defined(__ANDROID__) // Dynamic load, use vulkan_wrapper.h to load vulkan functions
  #include "vulkan_wrapper/vulkan_wrapper.h"
#else
  #include <vulkan/vulkan.h>
#endif

#include <vector>

#include <stdexcept>
#include <sstream>

namespace vkfw
{
  // Application should implement this interface or use provided helpers 
  //
  struct ICopyEngine
  {
    ICopyEngine(){}
    virtual ~ICopyEngine(){}

    virtual void UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size) = 0;     // mandatory if ICopyEngine devivative object used for meshes
    virtual void ReadBuffer  (VkBuffer a_src, size_t a_srcOffset,       void* a_dst, size_t a_size) {};      // optional
    virtual void UpdateImage (VkImage a_image, const void* a_src, int a_width, int a_height, int a_bpp) {};  // mandatory if ICopyEngine devivative object used for textures

  protected:
    ICopyEngine(const ICopyEngine& rhs) {}
    ICopyEngine& operator=(const ICopyEngine& rhs) { return *this; }    
  };

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct SimpleCopyHelper : public ICopyEngine
  {
    SimpleCopyHelper(); // fill everything with VK_NULL_HANDLE
    SimpleCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device, VkQueue a_transferQueue, uint32_t a_transferQueueIDX, size_t a_stagingBuffSize);
    virtual ~SimpleCopyHelper();

    virtual void UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size);
    virtual void ReadBuffer  (VkBuffer a_src, size_t a_srcOffset, void* a_dst, size_t a_size);
    virtual void UpdateImage (VkImage a_image, const void* a_src, int a_width, int a_height, int a_bpp);
    //virtual void ReadImage (VkImage a_image, const void* a_src, int a_width, int a_height, int a_bpp); // TODO: implement this in future

    VkCommandBuffer CmdBuffer() { return cmdBuff; }

  protected:

    VkQueue         queue;
    VkCommandPool   cmdPool;
    VkCommandBuffer cmdBuff;

    VkBuffer        stagingBuff;
    VkDeviceMemory  stagingBuffMemory;
    size_t          stagingSize;

    VkPhysicalDevice physDev;
    VkDevice         dev;

    SimpleCopyHelper(const SimpleCopyHelper& rhs) {}
    SimpleCopyHelper& operator=(const SimpleCopyHelper& rhs) { return *this; }
  };

  struct PinPongCopyHelper : SimpleCopyHelper
  {
    PinPongCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device, VkQueue a_transferQueue, uint32_t a_transferQueueIDX, size_t a_stagingBuffSize);
    ~PinPongCopyHelper();

    void UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size) override;

  protected:

    void SubmitCopy(VkBuffer a_dst, size_t a_dstOffset, size_t a_size, int a_currStagingId);

    VkFence  fence;
    VkBuffer staging[2];
    size_t   stagingSizeHalf;
  };

  struct ComputeCopyHelper : public SimpleCopyHelper
  {
    typedef SimpleCopyHelper Base;

    ComputeCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device,
                      VkQueue a_transferQueue, uint32_t a_transferQueueIDX,
                      size_t a_stagingBuffSize, const char *a_csCopyPath,
                      size_t a_auxBuffSize);

    ~ComputeCopyHelper();

    void ReadBuffer(VkBuffer a_src, size_t a_srcOffset, void* a_dst, size_t a_size) override;

  private:
    
    VkBuffer        auxBuff;
    VkDeviceMemory  auxBuffMemory;

    VkPipeline       copyPipeline;
    VkPipelineLayout copyPipelineLayout;
    VkShaderModule   csCopy;

    VkDescriptorSet       copyDescriptorSet;
    VkDescriptorSetLayout copyDescriptorSetLayout;

    VkDescriptorPool      m_dsPool;
  };

};

#endif