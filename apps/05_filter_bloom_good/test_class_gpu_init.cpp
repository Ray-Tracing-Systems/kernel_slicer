#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <cassert>
#include "vk_copy.h"
#include "vk_context.h"
#include "test_class_gpu.h"
#include "include/ToneMapping_gpu_ubo.h"


std::shared_ptr<ToneMapping> CreateToneMapping_GPU(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<ToneMapping_GPU>();
  pObj->SetVulkanContext(a_ctx);
  pObj->InitVulkanObjects(a_ctx.device, a_ctx.physicalDevice, a_maxThreadsGenerated);
  return pObj;
}

void ToneMapping_GPU::InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount)
{
  physicalDevice = a_physicalDevice;
  device         = a_device;
  m_allCreatedPipelineLayouts.reserve(256);
  m_allCreatedPipelines.reserve(256);
  InitHelpers();
  InitBuffers(a_maxThreadsCount, true);
  InitKernels(".spv");
  AllocateAllDescriptorSets();

}

static uint32_t ComputeReductionAuxBufferElements(uint32_t whole_size, uint32_t wg_size)
{
  uint32_t sizeTotal = 0;
  while (whole_size > 1)
  {
    whole_size  = (whole_size + wg_size - 1) / wg_size;
    sizeTotal  += std::max<uint32_t>(whole_size, 1);
  }
  return sizeTotal;
}

VkBufferUsageFlags ToneMapping_GPU::GetAdditionalFlagsForUBO() const
{
  return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
}

uint32_t ToneMapping_GPU::GetDefaultMaxTextures() const { return 256; }

void ToneMapping_GPU::MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline)
{
  VkPipelineShaderStageCreateInfo shaderStageInfo = {};
  shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  auto shaderCode   = vk_utils::readSPVFile(a_shaderPath);
  auto shaderModule = vk_utils::createShaderModule(device, shaderCode);

  shaderStageInfo.module              = shaderModule;
  shaderStageInfo.pName               = a_mainName;
  shaderStageInfo.pSpecializationInfo = a_specInfo;

  VkPushConstantRange pcRange = {};
  pcRange.stageFlags = shaderStageInfo.stage;
  pcRange.offset     = 0;
  pcRange.size       = 128; // at least 128 bytes for push constants for all Vulkan implementations

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pcRange;
  pipelineLayoutInfo.pSetLayouts            = &a_dsLayout;
  pipelineLayoutInfo.setLayoutCount         = 1;

  VkResult res = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, pPipelineLayout);
  if(res != VK_SUCCESS)
  {
    std::string errMsg = vk_utils::errorString(res);
    std::cout << "[ShaderError]: vkCreatePipelineLayout have failed for '" << a_shaderPath << "' with '" << errMsg.c_str() << "'" << std::endl;
  }
  else
    m_allCreatedPipelineLayouts.push_back(*pPipelineLayout);

  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.flags              = 0;
  pipelineInfo.stage              = shaderStageInfo;
  pipelineInfo.layout             = (*pPipelineLayout);
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, pPipeline);
  if(res != VK_SUCCESS)
  {
    std::string errMsg = vk_utils::errorString(res);
    std::cout << "[ShaderError]: vkCreateComputePipelines have failed for '" << a_shaderPath << "' with '" << errMsg.c_str() << "'" << std::endl;
  }
  else
    m_allCreatedPipelines.push_back(*pPipeline);

  if (shaderModule != VK_NULL_HANDLE)
    vkDestroyShaderModule(device, shaderModule, VK_NULL_HANDLE);
}

void ToneMapping_GPU::MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout, VkPipeline* pPipeline)
{
  VkPipelineShaderStageCreateInfo shaderStageInfo = {};
  shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  auto shaderCode   = vk_utils::readSPVFile(a_shaderPath);
  auto shaderModule = vk_utils::createShaderModule(device, shaderCode);

  shaderStageInfo.module              = shaderModule;
  shaderStageInfo.pName               = a_mainName;
  shaderStageInfo.pSpecializationInfo = a_specInfo;

  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.flags              = 0;
  pipelineInfo.stage              = shaderStageInfo;
  pipelineInfo.layout             = pipelineLayout;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, pPipeline);
  if(res != VK_SUCCESS)
  {
    std::string errMsg = vk_utils::errorString(res);
    std::cout << "[ShaderError]: vkCreateComputePipelines have failed for '" << a_shaderPath << "' with '" << errMsg.c_str() << "'" << std::endl;
  }
  else
    m_allCreatedPipelines.push_back(*pPipeline);

  if (shaderModule != VK_NULL_HANDLE)
    vkDestroyShaderModule(device, shaderModule, VK_NULL_HANDLE);
}

ToneMapping_GPU::~ToneMapping_GPU()
{
  for(size_t i=0;i<m_allCreatedPipelines.size();i++)
    vkDestroyPipeline(device, m_allCreatedPipelines[i], nullptr);
  for(size_t i=0;i<m_allCreatedPipelineLayouts.size();i++)
    vkDestroyPipelineLayout(device, m_allCreatedPipelineLayouts[i], nullptr);

  vkDestroyDescriptorSetLayout(device, BlurYDSLayout, nullptr);
  BlurYDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, ExtractBrightPixelsDSLayout, nullptr);
  ExtractBrightPixelsDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, DownSample4xDSLayout, nullptr);
  DownSample4xDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, MixAndToneMapDSLayout, nullptr);
  MixAndToneMapDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, BlurXDSLayout, nullptr);
  BlurXDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;


  vkDestroyBuffer(device, m_classDataBuffer, nullptr);

  vkDestroyBuffer(device, m_vdata.m_brightPixelsBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_downsampledImageBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_filterWeightsBuffer, nullptr);
  vkDestroyBuffer(device, m_vdata.m_tempImageBuffer, nullptr);
  FreeAllAllocations(m_allMems);
}

void ToneMapping_GPU::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
}


void ToneMapping_GPU::InitKernel_BlurY(const char* a_filePath)
{
  std::string shaderPath = AlterShaderPath("shaders_gpu/kernel2D_BlurY.comp.spv");
  const VkSpecializationInfo* kspec = nullptr;
  BlurYDSLayout = CreateBlurYDSLayout();
  MakeComputePipelineAndLayout(shaderPath.c_str(), "main", kspec, BlurYDSLayout, &BlurYLayout, &BlurYPipeline);
}

void ToneMapping_GPU::InitKernel_ExtractBrightPixels(const char* a_filePath)
{
  std::string shaderPath = AlterShaderPath("shaders_gpu/kernel2D_ExtractBrightPixels.comp.spv");
  const VkSpecializationInfo* kspec = nullptr;
  ExtractBrightPixelsDSLayout = CreateExtractBrightPixelsDSLayout();
  MakeComputePipelineAndLayout(shaderPath.c_str(), "main", kspec, ExtractBrightPixelsDSLayout, &ExtractBrightPixelsLayout, &ExtractBrightPixelsPipeline);
}

void ToneMapping_GPU::InitKernel_DownSample4x(const char* a_filePath)
{
  std::string shaderPath = AlterShaderPath("shaders_gpu/kernel2D_DownSample4x.comp.spv");
  const VkSpecializationInfo* kspec = nullptr;
  DownSample4xDSLayout = CreateDownSample4xDSLayout();
  MakeComputePipelineAndLayout(shaderPath.c_str(), "main", kspec, DownSample4xDSLayout, &DownSample4xLayout, &DownSample4xPipeline);
}

void ToneMapping_GPU::InitKernel_MixAndToneMap(const char* a_filePath)
{
  std::string shaderPath = AlterShaderPath("shaders_gpu/kernel2D_MixAndToneMap.comp.spv");
  const VkSpecializationInfo* kspec = nullptr;
  MixAndToneMapDSLayout = CreateMixAndToneMapDSLayout();
  MakeComputePipelineAndLayout(shaderPath.c_str(), "main", kspec, MixAndToneMapDSLayout, &MixAndToneMapLayout, &MixAndToneMapPipeline);
}

void ToneMapping_GPU::InitKernel_BlurX(const char* a_filePath)
{
  std::string shaderPath = AlterShaderPath("shaders_gpu/kernel2D_BlurX.comp.spv");
  const VkSpecializationInfo* kspec = nullptr;
  BlurXDSLayout = CreateBlurXDSLayout();
  MakeComputePipelineAndLayout(shaderPath.c_str(), "main", kspec, BlurXDSLayout, &BlurXLayout, &BlurXPipeline);
}


void ToneMapping_GPU::InitKernels(const char* a_filePath)
{
  InitKernel_BlurY(a_filePath);
  InitKernel_ExtractBrightPixels(a_filePath);
  InitKernel_DownSample4x(a_filePath);
  InitKernel_MixAndToneMap(a_filePath);
  InitKernel_BlurX(a_filePath);
}

void ToneMapping_GPU::InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay)
{
  ReserveEmptyVectors();

  m_maxThreadCount = a_maxThreadsCount;
  std::vector<VkBuffer> allBuffers;
  allBuffers.reserve(64);

  struct BufferReqPair
  {
    BufferReqPair() {  }
    BufferReqPair(VkBuffer a_buff, VkDevice a_dev) : buf(a_buff) { vkGetBufferMemoryRequirements(a_dev, a_buff, &req); }
    VkBuffer             buf = VK_NULL_HANDLE;
    VkMemoryRequirements req = {};
  };

  struct LocalBuffers
  {
    std::vector<BufferReqPair> bufs;
    size_t                     size = 0;
    std::vector<VkBuffer>      bufsClean;
  };

  std::vector<LocalBuffers> groups;
  groups.reserve(16);


  size_t largestIndex = 0;
  size_t largestSize  = 0;
  for(size_t i=0;i<groups.size();i++)
  {
    if(groups[i].size > largestSize)
    {
      largestIndex = i;
      largestSize  = groups[i].size;
    }
    groups[i].bufsClean.resize(groups[i].bufs.size());
    for(size_t j=0;j<groups[i].bufsClean.size();j++)
      groups[i].bufsClean[j] = groups[i].bufs[j].buf;
  }

  auto& allBuffersRef = allBuffers;

  m_classDataBuffer = vk_utils::createBuffer(device, sizeof(m_uboData),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | GetAdditionalFlagsForUBO());
  allBuffersRef.push_back(m_classDataBuffer);


  auto internalBuffersMem = AllocAndBind(allBuffersRef);
  if(a_tempBuffersOverlay)
  {
    for(size_t i=0;i<groups.size();i++)
      if(i != largestIndex)
        AssignBuffersToMemory(groups[i].bufsClean, internalBuffersMem.memObject);
  }
}

void ToneMapping_GPU::ReserveEmptyVectors()
{
  if(m_brightPixels.capacity() == 0)
    m_brightPixels.reserve(4);
  if(m_downsampledImage.capacity() == 0)
    m_downsampledImage.reserve(4);
  if(m_filterWeights.capacity() == 0)
    m_filterWeights.reserve(4);
  if(m_tempImage.capacity() == 0)
    m_tempImage.reserve(4);
}

void ToneMapping_GPU::InitMemberBuffers()
{
  std::vector<VkBuffer> memberVectors;
  std::vector<VkImage>  memberTextures;

  m_vdata.m_brightPixelsBuffer = vk_utils::createBuffer(device, m_brightPixels.capacity()*sizeof(struct LiteMath::float4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_brightPixelsBuffer);
  m_vdata.m_downsampledImageBuffer = vk_utils::createBuffer(device, m_downsampledImage.capacity()*sizeof(struct LiteMath::float4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_downsampledImageBuffer);
  m_vdata.m_filterWeightsBuffer = vk_utils::createBuffer(device, m_filterWeights.capacity()*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_filterWeightsBuffer);
  m_vdata.m_tempImageBuffer = vk_utils::createBuffer(device, m_tempImage.capacity()*sizeof(struct LiteMath::float4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_tempImageBuffer);


  AllocMemoryForMemberBuffersAndImages(memberVectors, memberTextures);
}




void ToneMapping_GPU::AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem)
{
  if(a_buffers.size() == 0 || a_mem == VK_NULL_HANDLE)
    return;

  std::vector<VkMemoryRequirements> memInfos(a_buffers.size());
  for(size_t i=0;i<memInfos.size();i++)
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkGetBufferMemoryRequirements(device, a_buffers[i], &memInfos[i]);
    else
    {
      memInfos[i] = memInfos[0];
      memInfos[i].size = 0;
    }
  }

  for(size_t i=1;i<memInfos.size();i++)
  {
    if(memInfos[i].memoryTypeBits != memInfos[0].memoryTypeBits)
    {
      std::cout << "[ToneMapping_GPU::AssignBuffersToMemory]: error, input buffers has different 'memReq.memoryTypeBits'" << std::endl;
      return;
    }
  }

  auto offsets = vk_utils::calculateMemOffsets(memInfos);
  for (size_t i = 0; i < memInfos.size(); i++)
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkBindBufferMemory(device, a_buffers[i], a_mem, offsets[i]);
  }
}

ToneMapping_GPU::MemLoc ToneMapping_GPU::AllocAndBind(const std::vector<VkBuffer>& a_buffers)
{
  MemLoc currLoc;
  if(a_buffers.size() > 0)
  {
    currLoc.memObject = vk_utils::allocateAndBindWithPadding(device, physicalDevice, a_buffers);
    currLoc.allocId   = m_allMems.size();
    m_allMems.push_back(currLoc);
  }
  return currLoc;
}

ToneMapping_GPU::MemLoc ToneMapping_GPU::AllocAndBind(const std::vector<VkImage>& a_images)
{
  MemLoc currLoc;
  if(a_images.size() > 0)
  {
    std::vector<VkMemoryRequirements> reqs(a_images.size());
    for(size_t i=0; i<reqs.size(); i++)
      vkGetImageMemoryRequirements(device, a_images[i], &reqs[i]);

    for(size_t i=0; i<reqs.size(); i++)
    {
      if(reqs[i].memoryTypeBits != reqs[0].memoryTypeBits)
      {
        std::cout << "ToneMapping_GPU::AllocAndBind(textures): memoryTypeBits warning, need to split mem allocation (override me)" << std::endl;
        break;
      }
    }

    auto offsets  = vk_utils::calculateMemOffsets(reqs);
    auto memTotal = offsets[offsets.size() - 1];

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = memTotal;
    allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(reqs[0].memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicalDevice);
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &currLoc.memObject));

    for(size_t i=0;i<a_images.size();i++) {
      VK_CHECK_RESULT(vkBindImageMemory(device, a_images[i], currLoc.memObject, offsets[i]));
    }

    currLoc.allocId = m_allMems.size();
    m_allMems.push_back(currLoc);
  }
  return currLoc;
}

void ToneMapping_GPU::FreeAllAllocations(std::vector<MemLoc>& a_memLoc)
{
  // in general you may check 'mem.allocId' for unique to be sure you dont free mem twice
  // for default implementation this is not needed
  for(auto mem : a_memLoc)
    vkFreeMemory(device, mem.memObject, nullptr);
  a_memLoc.resize(0);
}

void ToneMapping_GPU::AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_images)
{
  std::vector<VkMemoryRequirements> bufMemReqs(a_buffers.size()); // we must check that all buffers have same memoryTypeBits;
  for(size_t i = 0; i < a_buffers.size(); ++i)                    // if not, split to multiple allocations
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkGetBufferMemoryRequirements(device, a_buffers[i], &bufMemReqs[i]);
    else
    {
      bufMemReqs[i] = bufMemReqs[0];
      bufMemReqs[i].size = 0;
    }
  }

  bool needSplit = false;
  for(size_t i = 1; i < bufMemReqs.size(); ++i)
  {
    if(bufMemReqs[i].memoryTypeBits != bufMemReqs[0].memoryTypeBits)
    {
      needSplit = true;
      break;
    }
  }

  if(needSplit)
  {
    std::unordered_map<uint32_t, std::vector<uint32_t> > bufferSets;
    for(uint32_t j = 0; j < uint32_t(bufMemReqs.size()); ++j)
    {
      uint32_t key = uint32_t(bufMemReqs[j].memoryTypeBits);
      bufferSets[key].push_back(j);
    }

    for(const auto& buffGroup : bufferSets)
    {
      std::vector<VkBuffer> currGroup;
      for(auto id : buffGroup.second)
        currGroup.push_back(a_buffers[id]);
      AllocAndBind(currGroup);
    }
  }
  else
    AllocAndBind(a_buffers);

}

VkPhysicalDeviceFeatures2 ToneMapping_GPU::ListRequiredDeviceFeatures(std::vector<const char*>& deviceExtensions)
{
  static VkPhysicalDeviceFeatures2 features2 = {};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = nullptr;
  features2.features.shaderInt64   = false;
  features2.features.shaderFloat64 = false;
  features2.features.shaderInt16   = false;
  void** ppNext = &features2.pNext;
  return features2;
}

ToneMapping_GPU::MegaKernelIsEnabled ToneMapping_GPU::m_megaKernelFlags;

