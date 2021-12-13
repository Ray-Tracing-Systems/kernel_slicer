#ifndef VULKAN_COPY_H
#define VULKAN_COPY_H

#include "vk_include.h"

#include <vector>

#include <stdexcept>
#include <sstream>

namespace vk_utils
{
  // Application should implement this interface or use provided helpers 
  //
  struct ICopyEngine
  {
    ICopyEngine(){}
    virtual ~ICopyEngine(){}

    virtual void UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size) = 0;
    virtual void ReadBuffer  (VkBuffer a_src, size_t a_srcOffset,       void* a_dst, size_t a_size)
    {
      (void)a_src;
      (void)a_srcOffset;
      (void)a_dst;
      (void)a_size;
    };
    virtual void UpdateImage (VkImage a_image, const void* a_src, int a_width, int a_height, int a_bpp, VkImageLayout a_finalLayout)
    {
      (void)a_image;
      (void)a_src;
      (void)a_width;
      (void)a_height;
      (void)a_bpp;
      (void)a_finalLayout;
    };
    virtual void ReadImage (VkImage a_image, void* a_dst, int a_width, int a_height, int a_bpp, VkImageLayout a_finalLayout)
    {
      (void)a_image;
      (void)a_dst;
      (void)a_width;
      (void)a_height;
      (void)a_bpp;
      (void)a_finalLayout;
    };

    virtual VkQueue         TransferQueue() const { return VK_NULL_HANDLE; }
    virtual VkCommandBuffer CmdBuffer()     const { return VK_NULL_HANDLE; }
  protected:
    ICopyEngine(const ICopyEngine& rhs) { (void)rhs; }
    ICopyEngine& operator=(const ICopyEngine& rhs) { (void)rhs; return *this; }    
  };

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct SimpleCopyHelper : public ICopyEngine
  {
    SimpleCopyHelper(); // fill everything with VK_NULL_HANDLE
    SimpleCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device, VkQueue a_transferQueue, uint32_t a_transferQueueIDX, size_t a_stagingBuffSize);
    ~SimpleCopyHelper() override;

    void UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size) override;
    void ReadBuffer  (VkBuffer a_src, size_t a_srcOffset, void* a_dst, size_t a_size) override;
    void UpdateImage (VkImage a_image, const void* a_src, int a_width, int a_height, int a_bpp, VkImageLayout a_finalLayout) override;
    void ReadImage   (VkImage a_image, void* a_dst, int a_width, int a_height, int a_bpp, VkImageLayout a_finalLayout) override;

    VkQueue  TransferQueue() const override { return queue; }
    VkCommandBuffer CmdBuffer() const override { return cmdBuff; }

  protected:
    static constexpr uint32_t SMALL_BUFF = 65536;
    VkQueue         queue = VK_NULL_HANDLE;
    VkCommandPool   cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuff = VK_NULL_HANDLE;

    VkBuffer        stagingBuff = VK_NULL_HANDLE;
    VkDeviceMemory  stagingBuffMemory = VK_NULL_HANDLE;
    size_t          stagingSize = 0u;

    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice         dev = VK_NULL_HANDLE;

    SimpleCopyHelper(const SimpleCopyHelper& rhs) = delete;
    SimpleCopyHelper& operator=(const SimpleCopyHelper& rhs) { (void)rhs; return *this; }
  };

  struct PingPongCopyHelper : SimpleCopyHelper
  {
    PingPongCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device, VkQueue a_transferQueue, uint32_t a_transferQueueIDX, size_t a_stagingBuffSize);
    ~PingPongCopyHelper() override;

    void UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size) override;

  protected:

    void SubmitCopy(VkBuffer a_dst, size_t a_dstOffset, size_t a_size, int a_currStagingId);

    VkFence  fence = VK_NULL_HANDLE;;
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

    ~ComputeCopyHelper() override;

    void ReadBuffer(VkBuffer a_src, size_t a_srcOffset, void* a_dst, size_t a_size) override;

  private:
    
    VkBuffer        auxBuff = VK_NULL_HANDLE;;
    VkDeviceMemory  auxBuffMemory = VK_NULL_HANDLE;;

    VkPipeline       copyPipeline = VK_NULL_HANDLE;;
    VkPipelineLayout copyPipelineLayout = VK_NULL_HANDLE;;
    VkShaderModule   csCopy = VK_NULL_HANDLE;;

    VkDescriptorSet       copyDescriptorSet = VK_NULL_HANDLE;;
    VkDescriptorSetLayout copyDescriptorSetLayout = VK_NULL_HANDLE;;

    VkDescriptorPool      m_dsPool = VK_NULL_HANDLE;;
  };

}

#endif