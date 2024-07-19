#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <array>

#include "vk_pipeline.h"
#include "vk_buffers.h"
#include "vk_utils.h"
#include "vk_copy.h"
#include "vk_context.h"


#include "test_class.h"

#include "include/ToneMapping_gpu_ubo.h"
class ToneMapping_GPU : public ToneMapping
{
public:

  ToneMapping_GPU()
  {
  }
  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount);

  virtual void SetVulkanContext(vk_utils::VulkanContext a_ctx) { m_ctx = a_ctx; }
  virtual void SetVulkanInOutFor_Bloom(
    VkBuffer inData4fBuffer,
    size_t   inData4fOffset,
    VkBuffer outData1uiBuffer,
    size_t   outData1uiOffset,
    uint32_t dummyArgument = 0)
  {
    Bloom_local.inData4fBuffer = inData4fBuffer;
    Bloom_local.inData4fOffset = inData4fOffset;
    Bloom_local.outData1uiBuffer = outData1uiBuffer;
    Bloom_local.outData1uiOffset = outData1uiOffset;
    InitAllGeneratedDescriptorSets_Bloom();
  }

  virtual ~ToneMapping_GPU();


  virtual void InitMemberBuffers();
  virtual void UpdateAll(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
  {
    UpdatePlainMembers(a_pCopyEngine);
    UpdateVectorMembers(a_pCopyEngine);
    UpdateTextureMembers(a_pCopyEngine);
  }

  std::shared_ptr<vk_utils::ICopyEngine> m_pLastCopyHelper = nullptr;
  virtual void CommitDeviceData(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper) // you have to define this virtual function in the original imput class
  {
    ReserveEmptyVectors();
    InitMemberBuffers();
    UpdateAll(a_pCopyHelper);
    m_pLastCopyHelper = a_pCopyHelper;
  }
  void CommitDeviceData() override { CommitDeviceData(m_ctx.pCopyHelper); }
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override;


  virtual void ReserveEmptyVectors();
  virtual void UpdatePlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateTextureMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void ReadPlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  static VkPhysicalDeviceFeatures2 ListRequiredDeviceFeatures(std::vector<const char*>& deviceExtensions);

  virtual void BloomCmd(VkCommandBuffer a_commandBuffer, int w, int h, const float4* inData4f, unsigned int* outData1ui);

  void Bloom(int w, int h, const float4* inData4f, unsigned int* outData1ui) override;

  inline vk_utils::ExecTime GetBloomExecutionTime() const { return m_exTimeBloom; }

  vk_utils::ExecTime m_exTimeBloom;

  virtual void copyKernelFloatCmd(uint32_t length);
  virtual void matMulTransposeCmd(uint32_t A_offset, uint32_t B_offset, uint32_t C_offset, uint32_t A_col_len, uint32_t B_col_len, uint32_t A_row_len);

  virtual void BlurYCmd(int width, int height, const float4* a_dataIn, float4* a_dataOut);
  virtual void ExtractBrightPixelsCmd(int width, int height, const float4* inData4f, float4* a_brightPixels);
  virtual void DownSample4xCmd(int width, int height, const float4* a_dataFullRes, float4* a_dataSmallRes);
  virtual void MixAndToneMapCmd(int width, int height, const float4* inData4f, const float4* inBrightPixels, unsigned int* outData1ui);
  virtual void BlurXCmd(int width, int height, const float4* a_dataIn, float4* a_dataOut);

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

  virtual void InitAllGeneratedDescriptorSets_Bloom();

  virtual void AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem);

  virtual void AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_image);
  virtual std::string AlterShaderPath(const char* in_shaderPath) { return std::string("") + std::string(in_shaderPath); }
  
  

  struct Bloom_Data
  {
    VkBuffer inData4fBuffer = VK_NULL_HANDLE;
    size_t   inData4fOffset = 0;
    VkBuffer outData1uiBuffer = VK_NULL_HANDLE;
    size_t   outData1uiOffset = 0;
    bool needToClearOutput = false;
  } Bloom_local;



  struct MembersDataGPU
  {
    VkBuffer m_brightPixelsBuffer = VK_NULL_HANDLE;
    size_t   m_brightPixelsOffset = 0;
    VkBuffer m_downsampledImageBuffer = VK_NULL_HANDLE;
    size_t   m_downsampledImageOffset = 0;
    VkBuffer m_filterWeightsBuffer = VK_NULL_HANDLE;
    size_t   m_filterWeightsOffset = 0;
    VkBuffer m_tempImageBuffer = VK_NULL_HANDLE;
    size_t   m_tempImageOffset = 0;
  } m_vdata;


  size_t m_maxThreadCount = 0;
  VkBuffer m_classDataBuffer = VK_NULL_HANDLE;

  VkPipelineLayout      BlurYLayout   = VK_NULL_HANDLE;
  VkPipeline            BlurYPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout BlurYDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateBlurYDSLayout();
  virtual void InitKernel_BlurY(const char* a_filePath);
  VkPipelineLayout      ExtractBrightPixelsLayout   = VK_NULL_HANDLE;
  VkPipeline            ExtractBrightPixelsPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout ExtractBrightPixelsDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateExtractBrightPixelsDSLayout();
  virtual void InitKernel_ExtractBrightPixels(const char* a_filePath);
  VkPipelineLayout      DownSample4xLayout   = VK_NULL_HANDLE;
  VkPipeline            DownSample4xPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout DownSample4xDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateDownSample4xDSLayout();
  virtual void InitKernel_DownSample4x(const char* a_filePath);
  VkPipelineLayout      MixAndToneMapLayout   = VK_NULL_HANDLE;
  VkPipeline            MixAndToneMapPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout MixAndToneMapDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateMixAndToneMapDSLayout();
  virtual void InitKernel_MixAndToneMap(const char* a_filePath);
  VkPipelineLayout      BlurXLayout   = VK_NULL_HANDLE;
  VkPipeline            BlurXPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout BlurXDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateBlurXDSLayout();
  virtual void InitKernel_BlurX(const char* a_filePath);


  virtual VkBufferUsageFlags GetAdditionalFlagsForUBO() const;
  virtual uint32_t           GetDefaultMaxTextures() const;

  VkPipelineLayout      copyKernelFloatLayout   = VK_NULL_HANDLE;
  VkPipeline            copyKernelFloatPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyKernelFloatDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatecopyKernelFloatDSLayout();

  VkPipelineLayout      matMulTransposeLayout   = VK_NULL_HANDLE;
  VkPipeline            matMulTransposePipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout matMulTransposeDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatematMulTransposeDSLayout();

  VkDescriptorPool m_dsPool = VK_NULL_HANDLE;
  VkDescriptorSet  m_allGeneratedDS[5];

  ToneMapping_GPU_UBO_Data m_uboData;

  constexpr static uint32_t MEMCPY_BLOCK_SIZE = 256;
  constexpr static uint32_t REDUCTION_BLOCK_SIZE = 256;

  virtual void MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout,
                                            VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline);
  virtual void MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout,
                                       VkPipeline* pPipeline);

  std::vector<VkPipelineLayout> m_allCreatedPipelineLayouts; ///<! remenber them here to delete later
  std::vector<VkPipeline>       m_allCreatedPipelines;       ///<! remenber them here to delete later
public:

  struct MegaKernelIsEnabled
  {
    bool dummy = 0;
  };

  static MegaKernelIsEnabled  m_megaKernelFlags;
  static MegaKernelIsEnabled& EnabledPipelines() { return m_megaKernelFlags; }

};


