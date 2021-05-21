#ifndef MAIN_CLASS_DECL_TestClass_H
#define MAIN_CLASS_DECL_TestClass_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#include "vulkan_basics.h"
#include "vk_compute_pipeline.h"
#include "vk_buffer.h"
#include "vk_utils.h"

#include "test_class.h"

#include "include/TestClass_ubo.h"

class TestClass_Generated : public TestClass
{
public:

  TestClass_Generated() {}

  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount) 
  {
    physicalDevice = a_physicalDevice;
    device         = a_device;
    InitHelpers();
    InitBuffers(a_maxThreadsCount);
    InitKernels("z_generated.cl.spv");
    AllocateAllDescriptorSets();
  }

  virtual void SetVulkanInOutFor_NaivePathTrace(
    VkBuffer a_in_pakedXYBuffer,
    size_t   a_in_pakedXYOffset,
    VkBuffer a_out_colorBuffer,
    size_t   a_out_colorOffset,
    uint32_t dummyArgument = 0)
  {
    NaivePathTrace_local.in_pakedXYBuffer = a_in_pakedXYBuffer;
    NaivePathTrace_local.in_pakedXYOffset = a_in_pakedXYOffset;
    NaivePathTrace_local.out_colorBuffer = a_out_colorBuffer;
    NaivePathTrace_local.out_colorOffset = a_out_colorOffset;
    InitAllGeneratedDescriptorSets_NaivePathTrace();
  }

  virtual void SetVulkanInOutFor_CastSingleRay(
    VkBuffer a_in_pakedXYBuffer,
    size_t   a_in_pakedXYOffset,
    VkBuffer a_out_colorBuffer,
    size_t   a_out_colorOffset,
    uint32_t dummyArgument = 0)
  {
    CastSingleRay_local.in_pakedXYBuffer = a_in_pakedXYBuffer;
    CastSingleRay_local.in_pakedXYOffset = a_in_pakedXYOffset;
    CastSingleRay_local.out_colorBuffer = a_out_colorBuffer;
    CastSingleRay_local.out_colorOffset = a_out_colorOffset;
    InitAllGeneratedDescriptorSets_CastSingleRay();
  }

  virtual void SetVulkanInOutFor_PackXY(
    VkBuffer a_out_pakedXYBuffer,
    size_t   a_out_pakedXYOffset,
    uint32_t dummyArgument = 0)
  {
    PackXY_local.out_pakedXYBuffer = a_out_pakedXYBuffer;
    PackXY_local.out_pakedXYOffset = a_out_pakedXYOffset;
    InitAllGeneratedDescriptorSets_PackXY();
  }

  virtual ~TestClass_Generated();

  virtual void InitMemberBuffers();

  virtual void UpdateAll(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
  {
    UpdatePlainMembers(a_pCopyEngine);
    UpdateVectorMembers(a_pCopyEngine);
  }

  virtual void NaivePathTraceCmd(VkCommandBuffer a_commandBuffer, uint tid, uint a_maxDepth, uint* in_pakedXY, float4* out_color,
                                 uint tileStart, uint tileEnd);
  virtual void CastSingleRayCmd(VkCommandBuffer a_commandBuffer, uint tid, uint* in_pakedXY, uint* out_color,
                                uint tileStart, uint tileEnd);
  virtual void PackXYCmd(VkCommandBuffer a_commandBuffer, uint tidX, uint tidY, uint* out_pakedXY);

  virtual void copyKernelFloatCmd(uint32_t length);
  
  virtual void GetColorCmd(uint tid, uint* out_color, const TestClass* a_pGlobals, uint tileOffset);
  virtual void NextBounceCmd(uint tid, const Lite_Hit* in_hit, const float2* in_bars,
                             const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                             float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen,
                             float4* accumColor, float4* accumThoroughput, uint tileOffset);
  virtual void InitEyeRayCmd(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar, uint tileOffset);
  virtual void InitEyeRay2Cmd(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar,
                              float4* accumColor, float4* accumuThoroughput, uint tileOffset);
  virtual void RayTraceCmd(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                  Lite_Hit* out_hit, float2* out_bars, uint tileOffset);
  virtual void MakeMaterialCmd(uint tid, const Lite_Hit* in_hit, uint tileOffset);
  virtual void PackXYCmd(uint tidX, uint tidY, uint* out_pakedXY);
  virtual void ContributeToImageCmd(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color,
                                    uint tileOffset);


protected:
  
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice         device         = VK_NULL_HANDLE;

  VkCommandBuffer  m_currCmdBuffer   = VK_NULL_HANDLE;
  uint32_t         m_currThreadFlags = 0;

  std::unique_ptr<vkfw::ComputePipelineMaker> m_pMaker = nullptr;
  VkPhysicalDeviceProperties m_devProps;

  VkBufferMemoryBarrier BarrierForClearFlags(VkBuffer a_buffer);
  VkBufferMemoryBarrier BarrierForSingleBuffer(VkBuffer a_buffer);
  void BarriersForSeveralBuffers(VkBuffer* a_inBuffers, VkBufferMemoryBarrier* a_outBarriers, uint32_t a_buffersNum);

  virtual void InitHelpers();
  virtual void InitBuffers(size_t a_maxThreadsCount);
  virtual void InitKernels(const char* a_filePath);
  virtual void AllocateAllDescriptorSets();

  virtual void InitAllGeneratedDescriptorSets_NaivePathTrace();
  virtual void InitAllGeneratedDescriptorSets_CastSingleRay();
  virtual void InitAllGeneratedDescriptorSets_PackXY();

  virtual void UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine);



public:
  struct NaivePathTrace_Data
  {
    VkBuffer hitBuffer = VK_NULL_HANDLE;
    size_t   hitOffset = 0;

    VkBuffer accumThoroughputBuffer = VK_NULL_HANDLE;
    size_t   accumThoroughputOffset = 0;

    VkBuffer accumColorBuffer = VK_NULL_HANDLE;
    size_t   accumColorOffset = 0;

    VkBuffer threadFlagsBuffer = VK_NULL_HANDLE;
    size_t   threadFlagsOffset = 0;

    VkBuffer baricentricsBuffer = VK_NULL_HANDLE;
    size_t   baricentricsOffset = 0;

    VkBuffer rayDirAndFarBuffer = VK_NULL_HANDLE;
    size_t   rayDirAndFarOffset = 0;

    VkBuffer rayPosAndNearBuffer = VK_NULL_HANDLE;
    size_t   rayPosAndNearOffset = 0;

    VkBuffer in_pakedXYBuffer = VK_NULL_HANDLE;
    size_t   in_pakedXYOffset = 0;

    VkBuffer out_colorBuffer = VK_NULL_HANDLE;
    size_t   out_colorOffset = 0;

  } NaivePathTrace_local;


  struct CastSingleRay_Data
  {
    VkBuffer hitBuffer = VK_NULL_HANDLE;
    size_t   hitOffset = 0;

    VkBuffer threadFlagsBuffer = VK_NULL_HANDLE;
    size_t   threadFlagsOffset = 0;

    VkBuffer baricentricsBuffer = VK_NULL_HANDLE;
    size_t   baricentricsOffset = 0;

    VkBuffer rayDirAndFarBuffer = VK_NULL_HANDLE;
    size_t   rayDirAndFarOffset = 0;

    VkBuffer rayPosAndNearBuffer = VK_NULL_HANDLE;
    size_t   rayPosAndNearOffset = 0;

    VkBuffer in_pakedXYBuffer = VK_NULL_HANDLE;
    size_t   in_pakedXYOffset = 0;

    VkBuffer out_colorBuffer = VK_NULL_HANDLE;
    size_t   out_colorOffset = 0;

  } CastSingleRay_local;

protected:
  struct PackXY_Data
  {
    VkBuffer out_pakedXYBuffer = VK_NULL_HANDLE;
    size_t   out_pakedXYOffset = 0;

  } PackXY_local;


  struct StdVectorMembersGPUData
  {
    VkBuffer m_randomGensBuffer = VK_NULL_HANDLE;
    size_t   m_randomGensOffset = 0;
    VkBuffer m_vNorm4fBuffer = VK_NULL_HANDLE;
    size_t   m_vNorm4fOffset = 0;
    VkBuffer m_nodesBuffer = VK_NULL_HANDLE;
    size_t   m_nodesOffset = 0;
    VkBuffer m_intervalsBuffer = VK_NULL_HANDLE;
    size_t   m_intervalsOffset = 0;
    VkBuffer m_materialIdsBuffer = VK_NULL_HANDLE;
    size_t   m_materialIdsOffset = 0;
    VkBuffer m_indicesReorderedBuffer = VK_NULL_HANDLE;
    size_t   m_indicesReorderedOffset = 0;
    VkBuffer m_materialOffsetsBuffer = VK_NULL_HANDLE;
    size_t   m_materialOffsetsOffset = 0;
    VkBuffer m_vPos4fBuffer = VK_NULL_HANDLE;
    size_t   m_vPos4fOffset = 0;
    VkBuffer m_materialDataBuffer = VK_NULL_HANDLE;
    size_t   m_materialDataOffset = 0;
    VkDeviceMemory m_vecMem = VK_NULL_HANDLE;
  } m_vdata;

  size_t m_maxThreadCount = 0;

  // Auxilary data and kernels for 'VirtualKernels'; Dispatch hierarchy of 'IMaterial'
  //
  VkBuffer         m_IMaterialObjPtrBuffer = VK_NULL_HANDLE;
  size_t           m_IMaterialObjPtrOffset = 0;
  VkBufferMemoryBarrier BarrierForObjCounters(VkBuffer a_buffer);


  VkBuffer m_classDataBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_allMem    = VK_NULL_HANDLE;

  VkPipelineLayout      GetColorLayout   = VK_NULL_HANDLE;
  VkPipeline            GetColorPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout GetColorDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateGetColorDSLayout();
  void InitKernel_GetColor(const char* a_filePath);
  VkPipelineLayout      NextBounceLayout   = VK_NULL_HANDLE;
  VkPipeline            NextBouncePipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout NextBounceDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateNextBounceDSLayout();
  void InitKernel_NextBounce(const char* a_filePath);
  VkPipelineLayout      InitEyeRayLayout   = VK_NULL_HANDLE;
  VkPipeline            InitEyeRayPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout InitEyeRayDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateInitEyeRayDSLayout();
  void InitKernel_InitEyeRay(const char* a_filePath);
  VkPipelineLayout      InitEyeRay2Layout   = VK_NULL_HANDLE;
  VkPipeline            InitEyeRay2Pipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout InitEyeRay2DSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateInitEyeRay2DSLayout();
  void InitKernel_InitEyeRay2(const char* a_filePath);
  VkPipelineLayout      RayTraceLayout   = VK_NULL_HANDLE;
  VkPipeline            RayTracePipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout RayTraceDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateRayTraceDSLayout();
  void InitKernel_RayTrace(const char* a_filePath);
  VkPipelineLayout      MakeMaterialLayout   = VK_NULL_HANDLE;
  VkPipeline            MakeMaterialPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout MakeMaterialDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateMakeMaterialDSLayout();
  void InitKernel_MakeMaterial(const char* a_filePath);
  VkPipelineLayout      PackXYLayout   = VK_NULL_HANDLE;
  VkPipeline            PackXYPipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout PackXYDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatePackXYDSLayout();
  void InitKernel_PackXY(const char* a_filePath);
  VkPipelineLayout      ContributeToImageLayout   = VK_NULL_HANDLE;
  VkPipeline            ContributeToImagePipeline = VK_NULL_HANDLE; 
  VkDescriptorSetLayout ContributeToImageDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateContributeToImageDSLayout();
  void InitKernel_ContributeToImage(const char* a_filePath);


  virtual VkBufferUsageFlags GetAdditionalFlagsForUBO() const;

  VkPipelineLayout      copyKernelFloatLayout   = VK_NULL_HANDLE;
  VkPipeline            copyKernelFloatPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyKernelFloatDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatecopyKernelFloatDSLayout();

  VkDescriptorPool m_dsPool = VK_NULL_HANDLE;
  VkDescriptorSet  m_allGeneratedDS[10];

  TestClass_UBO_Data m_uboData;
  
  constexpr static uint32_t MEMCPY_BLOCK_SIZE = 256;
  constexpr static uint32_t REDUCTION_BLOCK_SIZE = 256;
};

#endif

