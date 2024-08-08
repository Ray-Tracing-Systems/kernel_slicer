#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <cassert>
#include "vk_copy.h"
#include "vk_context.h"
#include "ray_tracing/vk_rt_funcs.h"
#include "ray_tracing/vk_rt_utils.h"
#include "test_class_generated.h"
#include "include/TestClass_generated_ubo.h"

#include "VulkanRTX.h" //#ADDED 

#include "CrossRT.h"
ISceneObject* CreateVulkanRTX(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_graphicsQId, std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper,
                              uint32_t a_maxMeshes, uint32_t a_maxTotalVertices, uint32_t a_maxTotalPrimitives, uint32_t a_maxPrimitivesPerMesh,
                              bool build_as_add);

std::shared_ptr<TestClass> CreateTestClass_Generated(int w, int h, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<TestClass_Generated>(w, h);
  pObj->SetVulkanContext(a_ctx);
  pObj->InitVulkanObjects(a_ctx.device, a_ctx.physicalDevice, a_maxThreadsGenerated);
  return pObj;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BFRayTrace_RTX_Proxy : public ISceneObject
{
public:
  BFRayTrace_RTX_Proxy(std::shared_ptr<ISceneObject> a, std::shared_ptr<ISceneObject> b) { m_imps[0] = a; m_imps[1] = b; } 
  
  const char* Name() const override { return "BFRayTrace_RTX_Proxy"; }
  ISceneObject* UnderlyingImpl(uint32_t a_implId) override { return (a_implId < 2) ? m_imps[a_implId].get() : nullptr; }

  void ClearGeom() override { for(auto impl : m_imps) impl->ClearGeom(); } 

  uint32_t AddGeom_Triangles3f(const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber, uint32_t a_flags, size_t vByteStride) override
  {
    uint32_t res = 0;
    for(auto impl : m_imps) 
      res = impl->AddGeom_Triangles3f(a_vpos3f, a_vertNumber, a_triIndices, a_indNumber, a_flags, vByteStride);
    return res;
  }
                               
  void UpdateGeom_Triangles3f(uint32_t a_geomId, const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber, uint32_t a_flags, size_t vByteStride) override
  {
    for(auto impl : m_imps) 
      impl->UpdateGeom_Triangles3f(a_geomId, a_vpos3f, a_vertNumber, a_triIndices, a_indNumber, a_flags, vByteStride);
  }
  
  uint32_t AddGeom_AABB(uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber) override
  {
    uint32_t res = 0;
    for(auto impl : m_imps) 
      res = impl->AddGeom_AABB(a_typeId, boxMinMaxF8, a_boxNumber);
    return res;
  }
  
  void UpdateGeom_AABB(uint32_t a_geomId, uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber) override
  {
    for(auto impl : m_imps) 
      impl->UpdateGeom_AABB(a_geomId, a_typeId, boxMinMaxF8, a_boxNumber);
  }

  void     ClearScene() override { for(auto impl : m_imps) impl->ClearScene(); } 
  void     CommitScene(uint32_t options) override { for(auto impl : m_imps) impl->CommitScene();  }

  uint32_t AddInstanceMotion(uint32_t a_geomId, const LiteMath::float4x4* a_matrices, uint32_t a_matrixNumber) override 
  { 
    uint32_t res = 0;
    for(auto impl : m_imps) 
      res = impl->AddInstanceMotion(a_geomId, a_matrices, a_matrixNumber);
    return res; 
  }

  uint32_t AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix) override 
  { 
    uint32_t res = 0;
    for(auto impl : m_imps) 
      res = impl->AddInstance(a_geomId, a_matrix);
    return res; 
  }

  void    UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix) override   { for(auto impl : m_imps) impl->UpdateInstance(a_instanceId, a_matrix); }

  CRT_Hit RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override { return m_imps[0]->RayQuery_NearestHit(posAndNear, dirAndFar); }
  bool    RayQuery_AnyHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar)     override { return m_imps[0]->RayQuery_AnyHit(posAndNear, dirAndFar);; }  

  CRT_Hit RayQuery_NearestHitMotion(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar, float time) override { return m_imps[0]->RayQuery_NearestHit(posAndNear, dirAndFar); }
  bool    RayQuery_AnyHitMotion(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar, float time)     override { return m_imps[0]->RayQuery_AnyHit(posAndNear, dirAndFar); }

protected:
  std::array<std::shared_ptr<ISceneObject>, 2> m_imps = {nullptr, nullptr};
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass_Generated::InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount)
{
  physicalDevice = a_physicalDevice;
  device         = a_device;
  m_allCreatedPipelineLayouts.reserve(256);
  m_allCreatedPipelines.reserve(256);
  InitHelpers();
  InitBuffers(a_maxThreadsCount, true);
  InitKernels(".spv");
  AllocateAllDescriptorSets();

  auto queueAllFID = vk_utils::getQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT);
  uint32_t userRestrictions[4];
  this->SceneRestrictions(userRestrictions);
  uint32_t maxMeshes            = userRestrictions[0];
  uint32_t maxTotalVertices     = userRestrictions[1];
  uint32_t maxTotalPrimitives   = userRestrictions[2];
  uint32_t maxPrimitivesPerMesh = userRestrictions[3];

  auto pRayTraceImplOld = m_pRayTraceImpl;                                        
  m_pRayTraceImpl = std::shared_ptr<ISceneObject>(CreateVulkanRTX(a_device, a_physicalDevice, queueAllFID, m_ctx.pCopyHelper,
                                                                  maxMeshes, maxTotalVertices, maxTotalPrimitives, maxPrimitivesPerMesh, true),
                                                                  [](ISceneObject *p) { DeleteSceneRT(p); } );

  m_pRayTraceImpl = std::make_shared<BFRayTrace_RTX_Proxy>(pRayTraceImplOld, m_pRayTraceImpl); // #ADDED: wrap both user and RTX implementation with proxy object                                                                    
  
  AllocAllShaderBindingTables();
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

VkBufferUsageFlags TestClass_Generated::GetAdditionalFlagsForUBO() const
{
  return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
}

uint32_t TestClass_Generated::GetDefaultMaxTextures() const { return 256; }

void TestClass_Generated::MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline)
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

void TestClass_Generated::MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout, VkPipeline* pPipeline)
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

void TestClass_Generated::MakeRayTracingPipelineAndLayout(const std::vector< std::pair<VkShaderStageFlagBits, std::string> >& shader_paths,
                                                                           bool a_hw_motion_blur, const char* a_mainName,
                                                                           const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout,
                                                                           VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline)
{
  // (1) load shader modules
  //
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  std::vector<VkShaderModule>                       shaderModules;
  std::vector<VkPipelineShaderStageCreateInfo>      shaderStages;

  shaderGroups.reserve(shader_paths.size());
  shaderModules.reserve(shader_paths.size());
  shaderStages.reserve(shader_paths.size());

  for(const auto& [stage, path] : shader_paths)
  {
    VkPipelineShaderStageCreateInfo shaderStage = {};
    shaderStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage  = stage;
    auto shaderCode    = vk_utils::readSPVFile(path.c_str());
    shaderModules.push_back(vk_utils::createShaderModule(device, shaderCode));
    shaderStage.module = shaderModules.back();
    shaderStage.pName  = a_mainName;
    assert(shaderStage.module != VK_NULL_HANDLE);
    shaderStages.push_back(shaderStage);
  }

  for(uint32_t shaderId=0; shaderId < uint32_t(shader_paths.size()); shaderId++)
  {
    const auto stage = shader_paths[shaderId].first;

    VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
    shaderGroup.anyHitShader       = VK_SHADER_UNUSED_KHR; // #CHANGED
    shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR; // #CHANGED

    shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    if(stage == VK_SHADER_STAGE_MISS_BIT_KHR || stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR || stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR)
    {
      shaderGroup.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      shaderGroup.generalShader    = shaderId;
      shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    }
    else if(stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
    {
      shaderGroup.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      shaderGroup.generalShader    = VK_SHADER_UNUSED_KHR;
      shaderGroup.closestHitShader = shaderId;
    }
    else if(stage == VK_SHADER_STAGE_INTERSECTION_BIT_KHR) // #CHANGED / #ADDED
    {
      shaderGroup.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
      shaderGroup.generalShader      = VK_SHADER_UNUSED_KHR;
      shaderGroup.intersectionShader = shaderId + 0; //
      shaderGroup.closestHitShader   = shaderId + 1; // assume next is always 'closestHitShader' for current 'intersectionShader'
      shaderId++;
    }                                                      // #CHANGED / #ADDED

    shaderGroups.push_back(shaderGroup);
  }

  // (2) create pipeline layout
  //
  std::array<VkPushConstantRange,4> pcRanges = {};
  pcRanges[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  pcRanges[1].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR;
  pcRanges[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  pcRanges[3].stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;  // #CHANGED / #ADDED
  for(size_t i=0;i<pcRanges.size();i++) {
    pcRanges[i].offset = 0;
    pcRanges[i].size   = 128;
  }

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = uint32_t(pcRanges.size());
  pipelineLayoutInfo.pPushConstantRanges    = pcRanges.data();
  pipelineLayoutInfo.pSetLayouts            = &a_dsLayout;
  pipelineLayoutInfo.setLayoutCount         = 1;

  VkResult res = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, pPipelineLayout);
  if(res != VK_SUCCESS)
  {
    std::string errMsg = vk_utils::errorString(res);
    std::cout << "[ShaderError]: vkCreatePipelineLayout have failed for '" << shader_paths[0].second.c_str() << "' with '" << errMsg.c_str() << "'" << std::endl;
  }
  else
    m_allCreatedPipelineLayouts.push_back(*pPipelineLayout);

  // (3) create ray tracing pipeline
  //
  VkPipelineCreateFlags pipelineFlags = 0;
  if(a_hw_motion_blur)
    pipelineFlags |= VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;

  VkRayTracingPipelineCreateInfoKHR createInfo = {};
  createInfo.sType      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
  createInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
  createInfo.pStages    = shaderStages.data();
  createInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
  createInfo.pGroups    = shaderGroups.data();
  createInfo.maxPipelineRayRecursionDepth = 1;
  createInfo.layout     = (*pPipelineLayout);
  createInfo.flags      = pipelineFlags;
  VK_CHECK_RESULT(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &createInfo, nullptr, pPipeline));

  for (size_t i = 0; i < shader_paths.size(); ++i)
  {
    if(shaderModules[i] != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, shaderModules[i], VK_NULL_HANDLE);
    shaderModules[i] = VK_NULL_HANDLE;
  }
}

TestClass_Generated::~TestClass_Generated()
{
  for(size_t i=0;i<m_allCreatedPipelines.size();i++)
    vkDestroyPipeline(device, m_allCreatedPipelines[i], nullptr);
  for(size_t i=0;i<m_allCreatedPipelineLayouts.size();i++)
    vkDestroyPipelineLayout(device, m_allCreatedPipelineLayouts[i], nullptr);

  vkDestroyDescriptorSetLayout(device, BFRT_ReadAndComputeMegaDSLayout, nullptr);
  BFRT_ReadAndComputeMegaDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;

  for(size_t i=0;i<m_allShaderTableBuffers.size();i++)
    vkDestroyBuffer(device, m_allShaderTableBuffers[i], nullptr);

  vkDestroyBuffer(device, m_classDataBuffer, nullptr);

  vkFreeMemory(device, m_allShaderTableMem, nullptr);
  FreeAllAllocations(m_allMems);
}

void TestClass_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
}


void TestClass_Generated::InitKernel_BFRT_ReadAndComputeMega(const char* a_filePath)
{
  std::string shaderPath = AlterShaderPath("shaders_generated/BFRT_ReadAndComputeMega.comp.spv");
  const VkSpecializationInfo* kspec = nullptr;
  BFRT_ReadAndComputeMegaDSLayout = CreateBFRT_ReadAndComputeMegaDSLayout();
  if(m_megaKernelFlags.enableBFRT_ReadAndComputeMega)
  {
    const bool enableMotionBlur = false;

    std::string shaderPathRGEN = AlterShaderPath("shaders_generated/BFRT_ReadAndComputeMegaRGEN.glsl.spv");
    std::string shaderPathRCHT = AlterShaderPath("shaders_generated/z_trace_rchit.glsl.spv");
    std::string shaderPathRMIS = AlterShaderPath("shaders_generated/z_trace_rmiss.glsl.spv");
    
    // #ADDED 
    std::string shaderSpherePrimRCHT = AlterShaderPath("shaders_generated/z_SpherePrim_rchit.glsl.spv");
    std::string shaderSpherePrimRINT = AlterShaderPath("shaders_generated/z_SpherePrim_rcint.glsl.spv");
    // #ADDED

    std::vector< std::pair<VkShaderStageFlagBits, std::string> > shader_paths;
    {
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_RAYGEN_BIT_KHR,      shaderPathRGEN.c_str()));
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_MISS_BIT_KHR,        shaderPathRMIS.c_str()));
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, shaderPathRCHT.c_str()));
      // #ADDED 
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, shaderSpherePrimRINT.c_str()));
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,  shaderSpherePrimRCHT.c_str()));
      // #ADDED
    }

    MakeRayTracingPipelineAndLayout(shader_paths, enableMotionBlur, "main", kspec, BFRT_ReadAndComputeMegaDSLayout, &BFRT_ReadAndComputeMegaLayout, &BFRT_ReadAndComputeMegaPipeline);
  }
  else
  {
    BFRT_ReadAndComputeMegaLayout   = nullptr;
    BFRT_ReadAndComputeMegaPipeline = nullptr;
  }
}


void TestClass_Generated::InitKernels(const char* a_filePath)
{
  InitKernel_BFRT_ReadAndComputeMega(a_filePath);
}

void TestClass_Generated::InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay)
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

void TestClass_Generated::ReserveEmptyVectors()
{
}

void TestClass_Generated::UpdatePrefixPointers()
{
  static std::vector<AbtractPrimitive> g_temp(115);

  auto pUnderlyingImpl = dynamic_cast<BFRayTrace*>(m_pRayTraceImpl->UnderlyingImpl(0));  //#CHANGED
  if(pUnderlyingImpl != nullptr) 
    m_pRayTraceImpl_primitives = &pUnderlyingImpl->primitives;
  else 
    m_pRayTraceImpl_primitives = &g_temp;
}

void TestClass_Generated::InitMemberBuffers()
{
  std::vector<VkBuffer> memberVectors;
  std::vector<VkImage>  memberTextures;
  
  std::cout << "m_pRayTraceImpl_primitives->capacity() = " << m_pRayTraceImpl_primitives->capacity() << std::endl;

  m_vdata.m_pRayTraceImpl_primitivesBuffer = vk_utils::createBuffer(device, m_pRayTraceImpl_primitives->capacity()*sizeof(AbtractPrimitive), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.m_pRayTraceImpl_primitivesBuffer);

  AllocMemoryForMemberBuffersAndImages(memberVectors, memberTextures);
}




void TestClass_Generated::AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem)
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
      std::cout << "[TestClass_Generated::AssignBuffersToMemory]: error, input buffers has different 'memReq.memoryTypeBits'" << std::endl;
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

TestClass_Generated::MemLoc TestClass_Generated::AllocAndBind(const std::vector<VkBuffer>& a_buffers)
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

TestClass_Generated::MemLoc TestClass_Generated::AllocAndBind(const std::vector<VkImage>& a_images)
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
        std::cout << "TestClass_Generated::AllocAndBind(textures): memoryTypeBits warning, need to split mem allocation (override me)" << std::endl;
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

void TestClass_Generated::FreeAllAllocations(std::vector<MemLoc>& a_memLoc)
{
  // in general you may check 'mem.allocId' for unique to be sure you dont free mem twice
  // for default implementation this is not needed
  for(auto mem : a_memLoc)
    vkFreeMemory(device, mem.memObject, nullptr);
  a_memLoc.resize(0);
}

void TestClass_Generated::AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_images)
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
void TestClass_Generated::AllocAllShaderBindingTables()
{
  // (0) remember appropriate record offsets inside VulkanRTX impl. for future use them with acceleration structure //#ADDED
  //
  std::vector<uint32_t> sbtRecordOffsets = {0, 1, 2};                           //#TODO: get from m_pRayTraceImpl after move RT 'AllocAllShaderBindingTables' call to CommitDeviceData() function (?) 
  auto pRTXImpl = dynamic_cast<VulkanRTX*>(m_pRayTraceImpl->UnderlyingImpl(1)); //#TODO: should be different for each pipelite
  if(pRTXImpl != nullptr)
    pRTXImpl->SetSBTRecordOffsets(sbtRecordOffsets);
  else
    std::cout << "AllocAllShaderBindingTables(): can't get SBT data from 'UnderlyingImpl' " << std::endl; 

  m_allShaderTableBuffers.clear();

  uint32_t numHitStages    = uint32_t(sbtRecordOffsets.size()); //#CHANHED, should be different for each RT pipeline
  uint32_t numShaderGroups = 4;                                 // raygen, miss, rayhit<tri>, rayhit<spheres> 

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR  rtPipelineProperties{};
  {
    rtPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 deviceProperties2{};
    deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties2.pNext = &rtPipelineProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);
  }

  const uint32_t handleSize        = rtPipelineProperties.shaderGroupHandleSize;
  const uint32_t handleSizeAligned = vk_utils::getSBTAlignedSize(rtPipelineProperties.shaderGroupHandleSize, rtPipelineProperties.shaderGroupHandleAlignment);
  const uint32_t sbtSize           = numShaderGroups * handleSize;

  const auto rgenStride = vk_utils::getSBTAlignedSize(handleSizeAligned, rtPipelineProperties.shaderGroupBaseAlignment);
  const auto missSize   = vk_utils::getSBTAlignedSize(handleSizeAligned, rtPipelineProperties.shaderGroupBaseAlignment);
  const auto hitSize    = vk_utils::getSBTAlignedSize(numHitStages  * handleSizeAligned, rtPipelineProperties.shaderGroupBaseAlignment);

  std::vector<VkPipeline> allRTPipelines = {};
  if(BFRT_ReadAndComputeMegaPipeline != VK_NULL_HANDLE)
    allRTPipelines.push_back(BFRT_ReadAndComputeMegaPipeline);

  // (1) create buffers for SBT
  //
  for(VkPipeline rtPipeline : allRTPipelines) // todo add for loop
  {
    //std::vector<uint8_t> shaderHandleStorage(sbtSize);                                                                                   //#REMOVED
    //VK_CHECK_RESULT(vkGetRayTracingShaderGroupHandlesKHR(device, rtPipeline, 0, numShaderGroups, sbtSize, shaderHandleStorage.data()));  //#REMOVED

    VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    auto raygenBuf  = vk_utils::createBuffer(device, rgenStride, flags);
    auto raymissBuf = vk_utils::createBuffer(device, missSize, flags);
    auto rayhitBuf  = vk_utils::createBuffer(device, hitSize, flags);

    m_allShaderTableBuffers.push_back(raygenBuf);
    m_allShaderTableBuffers.push_back(raymissBuf);
    m_allShaderTableBuffers.push_back(rayhitBuf);
  }

  // (2) allocate and bind everything for 'm_allShaderTableBuffers'
  //
  std::vector<size_t> offsets;
  size_t memTotal;
  {
    auto a_buffers = m_allShaderTableBuffers; // in
    auto& res      = m_allShaderTableMem;     // in out
    auto a_dev     = device;                  // in
    auto a_physDev = physicalDevice;          // in

    std::vector<VkMemoryRequirements> memInfos(a_buffers.size());
    for(size_t i = 0; i < memInfos.size(); ++i)
    {
      if(a_buffers[i] != VK_NULL_HANDLE)
        vkGetBufferMemoryRequirements(a_dev, a_buffers[i], &memInfos[i]);
      else
      {
        memInfos[i] = memInfos[0];
        memInfos[i].size = 0;
      }
    }

    offsets  = vk_utils::calculateMemOffsets(memInfos); // out
    memTotal = offsets.back();                          // out

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    {
      memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
      memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }

    VkMemoryAllocateInfo allocateInfo = {};
    {
      allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocateInfo.pNext           = &memoryAllocateFlagsInfo;
      allocateInfo.allocationSize  = memTotal;
      allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memInfos[0].memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, a_physDev);
    }

    VK_CHECK_RESULT(vkAllocateMemory(a_dev, &allocateInfo, NULL, &res));

    for (size_t i = 0; i < memInfos.size(); i++)
    {
      if(a_buffers[i] != VK_NULL_HANDLE)
        vkBindBufferMemory(a_dev, a_buffers[i], res, offsets[i]);
    }
  }

  // (3) get all device addresses
  //
  std::vector<uint8_t> shaderHandleStorage(sbtSize);

  char* mapped = nullptr;
  VkResult result = vkMapMemory(device, m_allShaderTableMem, 0, memTotal, 0, (void**)&mapped);
  VK_CHECK_RESULT(result);

  int groupId = 0;
  if(BFRT_ReadAndComputeMegaPipeline != VK_NULL_HANDLE)
  {
    VK_CHECK_RESULT(vkGetRayTracingShaderGroupHandlesKHR(device, allRTPipelines[groupId], 0, numShaderGroups, sbtSize, shaderHandleStorage.data()));

    auto raygenBuf  = m_allShaderTableBuffers[groupId*3+0];
    auto raymissBuf = m_allShaderTableBuffers[groupId*3+1]; 
    auto rayhitBuf  = m_allShaderTableBuffers[groupId*3+2]; 

    BFRT_ReadAndComputeMegaSBTStrides.resize(4);            //#CHANGED:
    BFRT_ReadAndComputeMegaSBTStrides[0] = VkStridedDeviceAddressRegionKHR{ vk_rt_utils::getBufferDeviceAddress(device, raygenBuf),  rgenStride,         rgenStride };
    BFRT_ReadAndComputeMegaSBTStrides[1] = VkStridedDeviceAddressRegionKHR{ vk_rt_utils::getBufferDeviceAddress(device, raymissBuf), handleSizeAligned,  missSize   };
    BFRT_ReadAndComputeMegaSBTStrides[2] = VkStridedDeviceAddressRegionKHR{ vk_rt_utils::getBufferDeviceAddress(device, rayhitBuf),  handleSizeAligned,  hitSize    };
    BFRT_ReadAndComputeMegaSBTStrides[3] = VkStridedDeviceAddressRegionKHR{ 0u, 0u, 0u }; // for callable shaders

    auto *pData = shaderHandleStorage.data();
    
    memcpy(mapped + offsets[groupId*3 + 0], pData, handleSize * 1); // raygenBuf; copy raygen shader handle to SBT
    pData += handleSize * 1;
 
    memcpy(mapped + offsets[groupId*3 + 1], pData, handleSize * 1); // raymissBuf; copy miss shader handle(s) to SBT
    pData += handleSize * 1;
    
    memcpy(mapped + offsets[groupId*3 + 2], pData, handleSize);     // rchit for triangles BLAS
    pData += handleSize * 1;

    for(size_t i=1; i<sbtRecordOffsets.size(); i++)                 // rchit for spheres BLAS and boxes BLAS
    {
      memcpy(mapped + offsets[groupId*3 + 2] + i*handleSizeAligned, pData, handleSize); // rayhitBuf //#CHANGED: 3 --> 4;   copy hit shader handle(s) to SBT
    }
    pData += handleSize * 1;
  }
  groupId++;

  //  if(SomeOtherRTPipeline != VK_NULL_HANDLE) {... }

  vkUnmapMemory(device, m_allShaderTableMem);
}

VkPhysicalDeviceFeatures2 TestClass_Generated::ListRequiredDeviceFeatures(std::vector<const char*>& deviceExtensions)
{
  static VkPhysicalDeviceFeatures2 features2 = {};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = nullptr;
  features2.features.shaderInt64   = false;
  features2.features.shaderFloat64 = false;
  features2.features.shaderInt16   = false;
  void** ppNext = &features2.pNext;
  {
    static VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelStructFeatures = {};
    static VkPhysicalDeviceBufferDeviceAddressFeatures      enabledDeviceAddressFeatures = {};
    static VkPhysicalDeviceRayQueryFeaturesKHR              enabledRayQueryFeatures =  {};
    static VkPhysicalDeviceDescriptorIndexingFeatures       indexingFeatures = {};
    static VkPhysicalDeviceRayTracingPipelineFeaturesKHR    enabledRTPipelineFeatures = {};

    indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    indexingFeatures.pNext = nullptr;
    indexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE; // TODO: move bindless texture to seperate feature!
    indexingFeatures.runtimeDescriptorArray                    = VK_TRUE; // TODO: move bindless texture to seperate feature!

    enabledRayQueryFeatures.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    enabledRayQueryFeatures.rayQuery = VK_TRUE;
    enabledRayQueryFeatures.pNext    = &indexingFeatures;

    enabledDeviceAddressFeatures.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    enabledDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
    enabledDeviceAddressFeatures.pNext               = &enabledRayQueryFeatures;

    enabledAccelStructFeatures.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    enabledAccelStructFeatures.accelerationStructure = VK_TRUE;
    enabledAccelStructFeatures.pNext                 = &enabledDeviceAddressFeatures;
    enabledRTPipelineFeatures.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    enabledRTPipelineFeatures.rayTracingPipeline = VK_TRUE;
    enabledRTPipelineFeatures.pNext              = &enabledAccelStructFeatures;
    (*ppNext) = &enabledRTPipelineFeatures;
    ppNext = &indexingFeatures.pNext;

    // Required by VK_KHR_RAY_QUERY
    deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
    // Required by VK_KHR_acceleration_structure
    deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME); // TODO: move bindless texture it to seperate feature!
    // Required by VK_KHR_ray_tracing_pipeline
    deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
  }
  return features2;
}

TestClass_Generated::MegaKernelIsEnabled TestClass_Generated::m_megaKernelFlags;

