#ifndef VULKAN_COPY_H
#define VULKAN_COPY_H

#define USE_VOLK
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

    virtual void UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size) = 0;     // mandatory if ICopyEngine devivative object used for meshes
    virtual void ReadBuffer  (VkBuffer a_src, size_t a_srcOffset,       void* a_dst, size_t a_size) {};      // optional
    virtual void UpdateImage (VkImage a_image, const void* a_src, int a_width, int a_height, int a_bpp) {};  // mandatory if ICopyEngine devivative object used for textures

    virtual VkQueue         TransferQueue() const { return VK_NULL_HANDLE; }
    virtual VkCommandBuffer CmdBuffer()     const { return VK_NULL_HANDLE; }
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

    VkQueue  TransferQueue() const override { return queue; }
    VkCommandBuffer CmdBuffer() const override { return cmdBuff; }

  protected:
    static constexpr uint32_t SMALL_BUFF = 65536;
    VkQueue         queue;
    VkCommandPool   cmdPool;
    VkCommandBuffer cmdBuff;

    VkBuffer        stagingBuff;
    VkDeviceMemory  stagingBuffMemory;
    size_t          stagingSize;

    VkPhysicalDevice physDev;
    VkDevice         dev;

    SimpleCopyHelper(const SimpleCopyHelper& rhs) = delete;
    SimpleCopyHelper& operator=(const SimpleCopyHelper& rhs) { return *this; }
  };

  struct PingPongCopyHelper : SimpleCopyHelper
  {
    PingPongCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device, VkQueue a_transferQueue, uint32_t a_transferQueueIDX, size_t a_stagingBuffSize);
    ~PingPongCopyHelper();

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