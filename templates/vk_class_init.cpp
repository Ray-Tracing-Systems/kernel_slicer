#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <limits>
{% if UseServiceScan %}
#include <utility> // for std::pair
{% endif %}
#include <cassert>
#include "vk_copy.h"
#include "vk_context.h"
{% if UseRayGen or HasAllRefs %}
#include "ray_tracing/vk_rt_funcs.h"
#include "ray_tracing/vk_rt_utils.h"
{% endif %}
#include "{{IncludeClassDecl}}"
#include "include/{{UBOIncl}}"

{% if length(SceneMembers) > 0 %}
#include "CrossRT.h"
ISceneObject* CreateVulkanRTX(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_graphicsQId, std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper,
                              uint32_t a_maxMeshes, uint32_t a_maxTotalVertices, uint32_t a_maxTotalPrimitives, uint32_t a_maxPrimitivesPerMesh,
                              bool build_as_add);
{% endif %}

{% for ctorDecl in Constructors %}
{% if ctorDecl.NumParams == 0 %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>();
  pObj->SetVulkanContext(a_ctx);
  pObj->InitVulkanObjects(a_ctx.device, a_ctx.physicalDevice, a_maxThreadsGenerated);
  return pObj;
}
{% else %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}({{ctorDecl.Params}}, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>({{ctorDecl.PrevCall}});
  pObj->SetVulkanContext(a_ctx);
  pObj->InitVulkanObjects(a_ctx.device, a_ctx.physicalDevice, a_maxThreadsGenerated);
  return pObj;
}
{% endif %}
{% endfor %}

void {{MainClassName}}{{MainClassSuffix}}::InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount)
{
  physicalDevice = a_physicalDevice;
  device         = a_device;
  m_allCreatedPipelineLayouts.reserve(256);
  m_allCreatedPipelines.reserve(256);
  {% if length(SpecConstants) > 0 %}
  m_allSpecConstVals = ListRequiredFeatures();
  {% endif %}
  InitHelpers();
  InitBuffers(a_maxThreadsCount, true);
  InitKernels("{{ShaderSingleFile}}.spv");
  AllocateAllDescriptorSets();
  
  {% if length(SceneMembers) > 0 %}
  auto queueAllFID = vk_utils::getQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT);
  {% endif %}
  {% for ScnObj in SceneMembers %}
  uint32_t userRestrictions[4];
  this->SceneRestrictions(userRestrictions);
  uint32_t maxMeshes            = userRestrictions[0];
  uint32_t maxTotalVertices     = userRestrictions[1];
  uint32_t maxTotalPrimitives   = userRestrictions[2];
  uint32_t maxPrimitivesPerMesh = userRestrictions[3];
  {% if ScnObj.HasIntersectionShader %}
  auto {{ScnObj.Name}}Old = {{ScnObj.Name}}; // save user implementation
  {% endif %}
  {{ScnObj.Name}} = std::shared_ptr<ISceneObject>(CreateVulkanRTX(a_device, a_physicalDevice, queueAllFID, m_ctx.pCopyHelper,
                                                             maxMeshes, maxTotalVertices, maxTotalPrimitives, maxPrimitivesPerMesh, true),
                                                             [](ISceneObject *p) { DeleteSceneRT(p); } );
  {% if ScnObj.HasIntersectionShader %}
  {{ScnObj.Name}} = std::make_shared<RTX_Proxy>({{ScnObj.Name}}Old, {{ScnObj.Name}}); // wrap both user and RTX implementation with proxy object 
  {% endif %}
  {% endfor %}
  {% if UseSubGroups %}
  if((m_ctx.subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) == 0)
    std::cout << "ALERT! class '{{MainClassName}}{{MainClassSuffix}}' uses subgroup operations but seems your device does not support them" << std::endl;
  if(m_ctx.subgroupProps.subgroupSize != {{SubGroupSize}}) {
    std::cout << "ALERT! class '{{MainClassName}}{{MainClassSuffix}}' uses subgroup operations with different subgroup size:" << std::endl;
    std::cout << " --> your device 'subgroupSize' = " << m_ctx.subgroupProps.subgroupSize << std::endl;
    std::cout << " --> this class  'subgroupSize' = " << {{SubGroupSize}} << std::endl;
  }
  {% endif %}
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

VkBufferUsageFlags {{MainClassName}}{{MainClassSuffix}}::GetAdditionalFlagsForUBO() const
{
  {% if HasFullImpl %}
  return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  {% else %}
  return 0;
  {% endif %}
}

uint32_t {{MainClassName}}{{MainClassSuffix}}::GetDefaultMaxTextures() const { return 256; }

void {{MainClassName}}{{MainClassSuffix}}::MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline)
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

void {{MainClassName}}{{MainClassSuffix}}::MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout, VkPipeline* pPipeline)
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
{% if UseRayGen %}

void {{MainClassName}}{{MainClassSuffix}}::MakeRayTracingPipelineAndLayout(const std::vector< std::pair<VkShaderStageFlagBits, std::string> >& shader_paths,
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

  // (2) make shader groups
  //
  for(uint32_t shaderId=0; shaderId < uint32_t(shader_paths.size()); shaderId++)
  {
    const auto stage = shader_paths[shaderId].first;

    VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
    shaderGroup.anyHitShader       = VK_SHADER_UNUSED_KHR; 
    shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR; 
    shaderGroup.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
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
    else if(stage == VK_SHADER_STAGE_INTERSECTION_BIT_KHR) 
    {
      shaderGroup.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
      shaderGroup.generalShader      = VK_SHADER_UNUSED_KHR;
      shaderGroup.intersectionShader = shaderId + 0; //
      shaderGroup.closestHitShader   = shaderId + 1; // assume next is always 'closestHitShader' for current 'intersectionShader'
      shaderId++;
    }                                                      

    shaderGroups.push_back(shaderGroup);
  }

  // (3) create pipeline layout
  //
  std::array<VkPushConstantRange,1> pcRanges = {};
  pcRanges[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  //pcRanges[1].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR;
  //pcRanges[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
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
{% endif %}

{{MainClassName}}{{MainClassSuffix}}::~{{MainClassName}}{{MainClassSuffix}}()
{
  for(size_t i=0;i<m_allCreatedPipelines.size();i++)
    vkDestroyPipeline(device, m_allCreatedPipelines[i], nullptr);
  for(size_t i=0;i<m_allCreatedPipelineLayouts.size();i++)
    vkDestroyPipelineLayout(device, m_allCreatedPipelineLayouts[i], nullptr);

  {% if UseServiceScan %}
  {% for Scan in ServiceScan %}
  m_scan_{{Scan.Type}}.DeleteDSLayouts(device);
  {% endfor %}
  {% endif %} {# /* UseServiceScan */ #}
  {% if UseServiceSort %}
  {% for Sort in ServiceSort %}
  m_sort_{{Sort.Type}}.DeleteDSLayouts(device);
  {% endfor %}
  {% endif %} {# /* UseServiceSort */ #}
## for Kernel in Kernels
  vkDestroyDescriptorSetLayout(device, {{Kernel.Name}}DSLayout, nullptr);
  {{Kernel.Name}}DSLayout = VK_NULL_HANDLE;
## endfor
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;

  {% if UseRayGen %}
  for(size_t i=0;i<m_allShaderTableBuffers.size();i++)
    vkDestroyBuffer(device, m_allShaderTableBuffers[i], nullptr);
  {% endif %}
## for MainFunc in MainFunctions
  {% if MainFunc.IsRTV and not MainFunc.IsMega %}
  {% for Buffer in MainFunc.LocalVarsBuffersDecl %}
  vkDestroyBuffer(device, {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}
  {% endif %}
## endfor

  vkDestroyBuffer(device, m_classDataBuffer, nullptr);
  {% if UseSeparateUBO %}
  vkDestroyBuffer(device, m_uboArgsBuffer, nullptr);
  {% endif %}

  {% for Buffer in ClassVectorVars %}
  vkDestroyBuffer(device, m_vdata.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}
  {% for Var in ClassTextureVars %}
  vkDestroyImage    (device, m_vdata.{{Var.Name}}Texture, nullptr);
  vkDestroyImageView(device, m_vdata.{{Var.Name}}View, nullptr);
  if(m_vdata.{{Var.Name}}Sampler != VK_NULL_HANDLE)
     vkDestroySampler(device, m_vdata.{{Var.Name}}Sampler, nullptr);
  {% endfor %}
  {% for Var in ClassTexArrayVars %}
  for(auto obj : m_vdata.{{Var.Name}}ArrayTexture)
    vkDestroyImage(device, obj, nullptr);
  for(auto obj : m_vdata.{{Var.Name}}ArrayView)
    vkDestroyImageView(device, obj, nullptr);
  for(auto obj : m_vdata.{{Var.Name}}ArraySampler)
  vkDestroySampler(device, obj, nullptr);
  {% endfor %}
  {% for Sam in SamplerMembers %}
  vkDestroySampler(device, m_vdata.{{Sam}}, nullptr);
  {% endfor %}
  {% for Buffer in RedVectorVars %}
  vkDestroyBuffer(device, m_vdata.{{Buffer.Name}}Buffer, nullptr);
  {% endfor %}
  {% if length(IndirectDispatches) > 0 %}
  vkDestroyBuffer(device, m_indirectBuffer, nullptr);
  vkDestroyDescriptorSetLayout(device, m_indirectUpdateDSLayout, nullptr);
  vkDestroyPipelineLayout(device, m_indirectUpdateLayout, nullptr);
  {% for Dispatch in IndirectDispatches %}
  vkDestroyPipeline(device, m_indirectUpdate{{Dispatch.KernelName}}Pipeline, nullptr);
  {% endfor %}
  {% endif %}
  {% if UseServiceScan %}
  {% for Scan in ServiceScan %}
  m_scan_{{Scan.Type}}.DeleteTempBuffers(device);
  {% endfor %}
  {% endif %}
  {% if UseRayGen %}
  vkFreeMemory(device, m_allShaderTableMem, nullptr);
  {% endif %}
  FreeAllAllocations(m_allMems);
}

void {{MainClassName}}{{MainClassSuffix}}::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
  {% if UseSpecConstWgSize %}
  {
    m_specializationEntriesWgSize[0].constantID = 0;
    m_specializationEntriesWgSize[0].offset     = 0;
    m_specializationEntriesWgSize[0].size       = sizeof(uint32_t);

    m_specializationEntriesWgSize[1].constantID = 1;
    m_specializationEntriesWgSize[1].offset     = sizeof(uint32_t);
    m_specializationEntriesWgSize[1].size       = sizeof(uint32_t);

    m_specializationEntriesWgSize[2].constantID = 2;
    m_specializationEntriesWgSize[2].offset     = 2 * sizeof(uint32_t);
    m_specializationEntriesWgSize[2].size       = sizeof(uint32_t);

    m_specsForWGSize.mapEntryCount = 3;
    m_specsForWGSize.pMapEntries   = m_specializationEntriesWgSize;
    m_specsForWGSize.dataSize      = 3 * sizeof(uint32_t);
    m_specsForWGSize.pData         = nullptr;
  }
  {% endif %}
}

{% if length(Hierarchies) > 0 %}
VkBufferMemoryBarrier {{MainClassName}}{{MainClassSuffix}}::BarrierForObjCounters(VkBuffer a_buffer)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = a_buffer;
  bar.offset              = 0;
  bar.size                = VK_WHOLE_SIZE; // TODO: count offset and size carefully, actually we can do this!
  return bar;
}
{% endif %}
{% if length(SpecConstants) > 0 %}
const VkSpecializationInfo* {{MainClassName}}{{MainClassSuffix}}::GetAllSpecInfo()
{
  if(m_allSpecConstInfo.size() == m_allSpecConstVals.size()) // already processed
    return &m_allSpecInfo;
  m_allSpecConstInfo.resize(m_allSpecConstVals.size());
  m_allSpecConstInfo[0].constantID = 0;
  m_allSpecConstInfo[0].size       = sizeof(uint32_t);
  m_allSpecConstInfo[0].offset     = 0;
  {% for x in SpecConstants %}
  m_allSpecConstInfo[{{loop.index1}}].constantID = {{loop.index1}};
  m_allSpecConstInfo[{{loop.index1}}].size       = sizeof(uint32_t);
  m_allSpecConstInfo[{{loop.index1}}].offset     = {{loop.index1}}*sizeof(uint32_t);
  {% endfor %}
  m_allSpecInfo.dataSize      = m_allSpecConstVals.size()*sizeof(uint32_t);
  m_allSpecInfo.mapEntryCount = static_cast<uint32_t>(m_allSpecConstInfo.size());
  m_allSpecInfo.pMapEntries   = m_allSpecConstInfo.data();
  m_allSpecInfo.pData         = m_allSpecConstVals.data();
  return &m_allSpecInfo;
}
{% endif %}

## for Kernel in Kernels
void {{MainClassName}}{{MainClassSuffix}}::InitKernel_{{Kernel.Name}}(const char* a_filePath)
{
  {% if MultipleSourceShaders %}
  std::string shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}.comp.spv");
  {% else %}
  std::string shaderPath = AlterShaderPath(a_filePath);
  {% endif %}
  const VkSpecializationInfo* kspec = {% if length(SpecConstants) > 0 %}GetAllSpecInfo(){% else %}nullptr{% endif %};
  {% if UseSpecConstWgSize %}
  uint32_t specializationData[3] = { {{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}} };
  m_specsForWGSize.pData         = specializationData;
  kspec = &m_specsForWGSize;
  {% endif %}
  {{Kernel.Name}}DSLayout = Create{{Kernel.Name}}DSLayout();
  {% if Kernel.IsMega %}
  if(m_megaKernelFlags.enable{{Kernel.Name}})
  {% else %}
  if(true)
  {% endif %}
  {
    {% if Kernel.UseRayGen %}
    const bool enableMotionBlur = {{UseMotionBlur}};
    std::string shaderPathRGEN  = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}RGEN.glsl.spv");
    std::string shaderPathRCHT  = AlterShaderPath("{{ShaderFolder}}/z_trace_rchit.glsl.spv");
    std::string shaderPathRMIS1 = AlterShaderPath("{{ShaderFolder}}/z_trace_rmiss.glsl.spv");
    std::string shaderPathRMIS2 = AlterShaderPath("{{ShaderFolder}}/z_trace_smiss.glsl.spv");
    {% for Hierarchy in Kernel.Hierarchies %}
    {% if Hierarchy.HasIntersection %}
    {% for Impl in Hierarchy.Implementations %}
    {% for Func in Impl.MemberFunctions %}
    {% if Func.IsIntersection %}

    std::string shader{{Impl.ClassName}}RINT = AlterShaderPath("{{ShaderFolder}}/{{Impl.ClassName}}_{{Func.Name}}_int.glsl.spv");
    std::string shader{{Impl.ClassName}}RHIT = AlterShaderPath("{{ShaderFolder}}/{{Impl.ClassName}}_{{Func.Name}}_hit.glsl.spv");  
    {% endif %}
    {% endfor %}
    {% endfor %}
    {% endif %}
    {% endfor %}
    std::vector< std::pair<VkShaderStageFlagBits, std::string> > shader_paths;
    {
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_RAYGEN_BIT_KHR,      shaderPathRGEN.c_str()));
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_MISS_BIT_KHR,        shaderPathRMIS1.c_str()));
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_MISS_BIT_KHR,        shaderPathRMIS2.c_str()));
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, shaderPathRCHT.c_str()));
      {% for Hierarchy in Kernel.Hierarchies %}
      {% if Hierarchy.HasIntersection %}
      {% for Impl in Hierarchy.Implementations %}
      {% for Func in Impl.MemberFunctions %}
      {% if Func.IsIntersection %}

      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, shader{{Impl.ClassName}}RINT.c_str()));
      shader_paths.emplace_back(std::make_pair(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,  shader{{Impl.ClassName}}RHIT.c_str()));
      {% endif %}
      {% endfor %}
      {% endfor %}
      {% endif %}
      {% endfor %}
    }

    MakeRayTracingPipelineAndLayout(shader_paths, enableMotionBlur, "main", kspec, {{Kernel.Name}}DSLayout, &{{Kernel.Name}}Layout, &{{Kernel.Name}}Pipeline);
    {% else %}
    MakeComputePipelineAndLayout(shaderPath.c_str(), {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}"{% endif %}, kspec, {{Kernel.Name}}DSLayout, &{{Kernel.Name}}Layout, &{{Kernel.Name}}Pipeline);
    {% endif %}
  }
  else
  {
    {{Kernel.Name}}Layout   = nullptr;
    {{Kernel.Name}}Pipeline = nullptr;
  }
  {% if Kernel.FinishRed %}
  {% if ShaderGLSL %}
  shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}_Reduction.comp.spv");
  {% endif %}
  {% if UseSpecConstWgSize %}
  uint32_t specializationData[3] = { 256, 1, 1 };
  m_specsForWGSize.pData         = specializationData;
  kspec = &m_specsForWGSize;
  {% endif %}
  MakeComputePipelineOnly(shaderPath.c_str(), {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Reduction"{% endif %}, kspec, {{Kernel.Name}}DSLayout, {{Kernel.Name}}Layout, &{{Kernel.Name}}ReductionPipeline);
  {% endif %} {# /* if Kernel.FinishRed */ #}
  {% if Kernel.HasLoopInit %}
  {% if ShaderGLSL %}
  shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}_Init.comp.spv");
  {% endif %}
  MakeComputePipelineOnly(shaderPath.c_str(), {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Init"{% endif %}, kspec, {{Kernel.Name}}DSLayout, {{Kernel.Name}}Layout, &{{Kernel.Name}}InitPipeline);
  {% endif %} {# /* if Kernel.HasLoopInit */ #}
  {% if Kernel.HasLoopFinish %}
  {% if ShaderGLSL %}
  shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Kernel.OriginalName}}_Finish.comp.spv");
  {% endif %}
  MakeComputePipelineOnly(shaderPath.c_str(), {% if ShaderGLSL %}"main"{% else %}"{{Kernel.OriginalName}}_Finish"{% endif %}, kspec, {{Kernel.Name}}DSLayout, {{Kernel.Name}}Layout, &{{Kernel.Name}}FinishPipeline);
  {% endif %} {# /* if Kernel.HasLoopFinish */ #}
}

## endfor

void {{MainClassName}}{{MainClassSuffix}}::InitKernels(const char* a_filePath)
{
## for Kernel in Kernels
  InitKernel_{{Kernel.Name}}(a_filePath);
## endfor
  {% if UseServiceMemCopy %}
  {% if MultipleSourceShaders %}
  std::string servPath = AlterShaderPath({% if ShaderGLSL %}"{{ShaderFolder}}/z_memcpy.comp.spv"{% else %}"{{ShaderFolder}}/serv_kernels.cpp.spv"{% endif %});
  {% else %}
  std::string servPath = AlterShaderPath(a_filePath);
  {% endif %}
  const VkSpecializationInfo* kspec = nullptr;
  {% if UseSpecConstWgSize %}
  uint32_t specializationData[3] = { 256, 1, 1 };
  m_specsForWGSize.pData         = specializationData;
  kspec = &m_specsForWGSize;
  {% endif %}
  copyKernelFloatDSLayout = CreatecopyKernelFloatDSLayout();
  MakeComputePipelineAndLayout(servPath.c_str(), {% if ShaderGLSL %}"main"{% else %}"copyKernelFloat"{% endif %}, kspec, copyKernelFloatDSLayout, &copyKernelFloatLayout, &copyKernelFloatPipeline);
  {% endif %} {# /* UseServiceMemCopy */ #}
  {% if UseServiceScan %}
  {% for Scan in ServiceScan %}
  // init m_scan_{{Scan.Type}}
  {
    const std::string servPathFwd         = AlterShaderPath("{{ShaderFolder}}/z_scan_{{Scan.Type}}_block.comp.spv");
    const std::string servPathProp        = AlterShaderPath("{{ShaderFolder}}/z_scan_{{Scan.Type}}_propagate.comp.spv");
    m_scan_{{Scan.Type}}.internalDSLayout = m_scan_{{Scan.Type}}.CreateInternalScanDSLayout(device);
    MakeComputePipelineAndLayout(servPathFwd.c_str(),  "main", nullptr, m_scan_{{Scan.Type}}.internalDSLayout, &m_scan_{{Scan.Type}}.scanFwdLayout,  &m_scan_{{Scan.Type}}.scanFwdPipeline);
    MakeComputePipelineAndLayout(servPathProp.c_str(), "main", nullptr, m_scan_{{Scan.Type}}.internalDSLayout, &m_scan_{{Scan.Type}}.scanPropLayout, &m_scan_{{Scan.Type}}.scanPropPipeline);
  }
  {% endfor %}
  {% endif %} {# /* UseServiceScan */ #}
  {% if UseServiceSort %}
  {% for Sort in ServiceSort %}
  // init m_sort_{{Sort.Type}}
  {
    std::string bitonicPassPath = AlterShaderPath("{{ShaderFolder}}/z_bitonic_{{Sort.Type}}_pass.comp.spv");
    std::string bitonic512Path  = AlterShaderPath("{{ShaderFolder}}/z_bitonic_{{Sort.Type}}_512.comp.spv");
    std::string bitonic1024Path = AlterShaderPath("{{ShaderFolder}}/z_bitonic_{{Sort.Type}}_1024.comp.spv");
    std::string bitonic2048Path = AlterShaderPath("{{ShaderFolder}}/z_bitonic_{{Sort.Type}}_2048.comp.spv");

    m_sort_{{Sort.Type}}.sortDSLayout = m_sort_{{Sort.Type}}.CreateSortDSLayout(device);
    MakeComputePipelineAndLayout(bitonicPassPath.c_str(),  "main", nullptr, m_sort_{{Sort.Type}}.sortDSLayout, &m_sort_{{Sort.Type}}.bitonicPassLayout, &m_sort_{{Sort.Type}}.bitonicPassPipeline);

    if(m_devProps.limits.maxComputeWorkGroupSize[0] >= 256)
    {
      MakeComputePipelineAndLayout(bitonic512Path.c_str(), "main", nullptr, m_sort_{{Sort.Type}}.sortDSLayout, &m_sort_{{Sort.Type}}.bitonic512Layout, &m_sort_{{Sort.Type}}.bitonic512Pipeline);
    }
    else
    {
      m_sort_{{Sort.Type}}.bitonic512Layout   = VK_NULL_HANDLE;
      m_sort_{{Sort.Type}}.bitonic512Pipeline = VK_NULL_HANDLE;
    }

    if(m_devProps.limits.maxComputeWorkGroupSize[0] >= 512)
    {
      MakeComputePipelineAndLayout(bitonic1024Path.c_str(), "main", nullptr, m_sort_{{Sort.Type}}.sortDSLayout, &m_sort_{{Sort.Type}}.bitonic1024Layout, &m_sort_{{Sort.Type}}.bitonic1024Pipeline);
    }
    else
    {
      m_sort_{{Sort.Type}}.bitonic1024Layout   = VK_NULL_HANDLE;
      m_sort_{{Sort.Type}}.bitonic1024Pipeline = VK_NULL_HANDLE;
    }

    if(m_devProps.limits.maxComputeWorkGroupSize[0] >= 1024)
    {
      MakeComputePipelineAndLayout(bitonic2048Path.c_str(), "main", nullptr, m_sort_{{Sort.Type}}.sortDSLayout, &m_sort_{{Sort.Type}}.bitonic2048Layout, &m_sort_{{Sort.Type}}.bitonic2048Pipeline);
    }
    else
    {
      m_sort_{{Sort.Type}}.bitonic2048Layout   = VK_NULL_HANDLE;
      m_sort_{{Sort.Type}}.bitonic2048Pipeline = VK_NULL_HANDLE;
    }
  }
  {% endfor %}
  {% endif %} {# /* UseServiceSort */ #}
  {% if UseMatMult %}
  {% if MultipleSourceShaders %}
  std::string servPath = AlterShaderPath({% if ShaderGLSL %}"{{ShaderFolder}}/z_matMulTranspose.comp.spv"{% else %}"{{ShaderFolder}}/serv_kernels.cpp.spv"{% endif %});
  {% else %}
  std::string servPath = AlterShaderPath(a_filePath);
  {% endif %}
  const VkSpecializationInfo* kspec = nullptr;
  {% if UseSpecConstWgSize %}
  uint32_t specializationData[3] = { 8, 8, 1 };
  m_specsForWGSize.pData         = specializationData;
  kspec = &m_specsForWGSize;
  {% endif %}
  matMulTransposeDSLayout = CreatematMulTransposeDSLayout();
  MakeComputePipelineAndLayout(servPath.c_str(), {% if ShaderGLSL %}"main"{% else %}"matMulTranspose"{% endif %}, kspec, matMulTransposeDSLayout, &matMulTransposeLayout, &matMulTransposePipeline);
  {% endif %} {# /* UseMatMult */ #}
  {% if length(IndirectDispatches) > 0 %}
  InitIndirectBufferUpdateResources(a_filePath);
  {% endif %}
}

void {{MainClassName}}{{MainClassSuffix}}::InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay)
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

## for MainFunc in MainFunctions
  {% if MainFunc.IsRTV and not MainFunc.IsMega %}
  LocalBuffers localBuffers{{MainFunc.Name}};
  localBuffers{{MainFunc.Name}}.bufs.reserve(32);
  {% for Buffer in MainFunc.LocalVarsBuffersDecl %}
  {% if Buffer.TransferDST %}
  {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer = vk_utils::createBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  {% else %}
  {{MainFunc.Name}}_local.{{Buffer.Name}}Buffer = vk_utils::createBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  {% endif %}
  localBuffers{{MainFunc.Name}}.bufs.push_back(BufferReqPair({{MainFunc.Name}}_local.{{Buffer.Name}}Buffer, device));
  {% endfor %}
  for(const auto& pair : localBuffers{{MainFunc.Name}}.bufs)
  {
    allBuffers.push_back(pair.buf);
    localBuffers{{MainFunc.Name}}.size += pair.req.size;
  }
  groups.push_back(localBuffers{{MainFunc.Name}});
  {% endif %}
## endfor

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

  {% if IsRTV and not IsMega %}
  auto& allBuffersRef = a_tempBuffersOverlay ? groups[largestIndex].bufsClean : allBuffers;
  {% else %}
  auto& allBuffersRef = allBuffers;
  {% endif %}

  m_classDataBuffer = vk_utils::createBuffer(device, sizeof(m_uboData),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | GetAdditionalFlagsForUBO());
  allBuffersRef.push_back(m_classDataBuffer);
  {% if UseSeparateUBO %}
  m_uboArgsBuffer = vk_utils::createBuffer(device, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffersRef.push_back(m_uboArgsBuffer);
  {% endif %}
  {% for Buffer in RedVectorVars %}
  {
    const size_t sizeOfBuffer = ComputeReductionAuxBufferElements(a_maxThreadsCount, REDUCTION_BLOCK_SIZE)*sizeof({{Buffer.Type}});
    m_vdata.{{Buffer.Name}}Buffer = vk_utils::createBuffer(device, sizeOfBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    allBuffersRef.push_back(m_vdata.{{Buffer.Name}}Buffer);
  }
  {% endfor %}

  {% if UseServiceScan %}
  {% for Scan in ServiceScan %}
  {
    auto tempBuffersForScan = m_scan_{{Scan.Type}}.InitTempBuffers(device, std::max(a_maxThreadsCount, size_t(256)));
    allBuffersRef.insert(allBuffersRef.end(), tempBuffersForScan.begin(), tempBuffersForScan.end());
  }
  {% endfor %}
  {% endif %}

  auto internalBuffersMem = AllocAndBind(allBuffersRef);
  if(a_tempBuffersOverlay)
  {
    for(size_t i=0;i<groups.size();i++)
      if(i != largestIndex)
        AssignBuffersToMemory(groups[i].bufsClean, internalBuffersMem.memObject);
  }
}

void {{MainClassName}}{{MainClassSuffix}}::ReserveEmptyVectors()
{
  {% for Var in ClassVectorVars %}
  {% if Var.AccessSymb == "." %}
  if({{Var.Name}}{{Var.AccessSymb}}capacity() == 0)
    {{Var.Name}}{{Var.AccessSymb}}reserve(4);
  {% else %}
  if({{Var.Name}} != nullptr && {{Var.Name}}{{Var.AccessSymb}}capacity() == 0)
    {{Var.Name}}{{Var.AccessSymb}}reserve(4);
  {% endif %}
  {% endfor %}
}
{% for Hierarchy in Hierarchies %} 
{% if Hierarchy.VFHLevel >= 2 %}
static size_t GetSizeByTag_{{Hierarchy.Name}}(uint32_t a_tag)
{
  switch(a_tag)
  {
    {% for Impl in Hierarchy.Implementations %}
    case {{Hierarchy.Name}}::{{Impl.TagName}}: return sizeof({{Impl.ClassName}});
    {% endfor %}
    default : return sizeof({{Hierarchy.EmptyImplementation.ClassName}});
  }
};

static size_t PackObject_{{Hierarchy.Name}}(std::vector<uint8_t>& buffer, const {{Hierarchy.Name}}* a_ptr) // todo: generate implementation via dynamic_cast or static_cast (can be used because we know the type)
{
  const size_t objSize  = GetSizeByTag_{{Hierarchy.Name}}(a_ptr->GetTag());
  const size_t currSize = buffer.size();
  const size_t nextSize = buffer.size() + objSize - sizeof(void*); // minus vptr size
  buffer.resize(nextSize);
  const char* objData = ((const char*)a_ptr) + sizeof(void*);      // do not account for vptr
  memcpy(buffer.data() + currSize, objData, objSize - sizeof(void*)); 
  return objSize;
}

{% endif %}
{% endfor %}

void {{MainClassName}}{{MainClassSuffix}}::InitMemberBuffers()
{
  std::vector<VkBuffer> memberVectorsWithDevAddr;
  std::vector<VkBuffer> memberVectors;
  std::vector<VkImage>  memberTextures;
  {% for Var in ClassVectorVars %}
  {% if Var.IsVFHBuffer and Var.VFHLevel >= 2 %}
  
  if({{Var.Name}}_sorted.size() == 0) // Pack all objects of '{{Var.Hierarchy.Name}}'
  {
    auto& bufferV = {{Var.Name}}_dataV;
    auto& sorted  = {{Var.Name}}_sorted;
    auto& vtable  = {{Var.Name}}_vtable;
    vtable.resize({{Var.Name}}{{Var.AccessSymb}}size());
    sorted.resize({{length(Var.Hierarchy.Implementations)}} + 1);
    bufferV.resize(16*4); // ({{Var.Name}}.size()*sizeof({{Var.Name}})); actual reserve may not be needed due to implementation don't have vectors. TODO: you may cvheck this explicitly in kslicer
    for(size_t arrId=0;arrId<sorted.size(); arrId++) {
      sorted[arrId].reserve({{Var.Name}}{{Var.AccessSymb}}size()*sizeof({{Var.Hierarchy.Name}}));
      sorted[arrId].resize(0);
    }
    
    std::vector<size_t> objCount(sorted.size());
    for(auto& x : objCount) x = 0;

    for(size_t i=0;i<{{Var.Name}}{{Var.AccessSymb}}size();i++) {
      const auto tag = {{Var.Name}}{{Var.AccessSymb}}at(i)->GetTag(); 
      PackObject_{{Var.Hierarchy.Name}}(sorted[tag], {{Var.Name}}{{Var.AccessSymb}}at(i));
      vtable[i] = LiteMath::uint2(tag, uint32_t(objCount[tag]));
      objCount[tag]++;
    }

    const size_t buffReferenceAlign = 16; // from EXT_buffer_reference spec: "If the layout qualifier is not specified, it defaults to 16 bytes"
    size_t objDataBufferSize = 0;
    {{Var.Name}}_obj_storage_offsets.reserve(sorted.size() + 1);
    {{Var.Name}}_obj_storage_offsets.resize(0);
    for(size_t arrId=0;arrId<sorted.size(); arrId++)
    {
      {{Var.Name}}_obj_storage_offsets.push_back(objDataBufferSize);
      objDataBufferSize += vk_utils::getPaddedSize(sorted[arrId].size(), buffReferenceAlign);
    }
    {{Var.Name}}_obj_storage_offsets.push_back(objDataBufferSize); // store total buffer size also in this array

    m_vdata.{{Var.Name}}_dataSBuffer = vk_utils::createBuffer(device, objDataBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_vdata.{{Var.Name}}_dataVBuffer = vk_utils::createBuffer(device, bufferV.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_vdata.{{Var.Name}}_dataSOffset = 0;
    m_vdata.{{Var.Name}}_dataVOffset = 0;
    memberVectorsWithDevAddr.push_back(m_vdata.{{Var.Name}}_dataSBuffer);
    memberVectors.push_back(m_vdata.{{Var.Name}}_dataVBuffer);
  }
  {% endif %}
  {% endfor %}
  {% for Table in RemapTables %}
  {
    auto pProxyObj = dynamic_cast<RTX_Proxy*>({{Table.AccelName}}.get());
    auto tablePtrs = pProxyObj->GetAABBToPrimTable({{Table.InterfaceName}}::{{Table.TagName}});
    m_vdata.{{Table.Name}}RemapTableBuffer = vk_utils::createBuffer(device, tablePtrs.tableSize*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    memberVectorsWithDevAddr.push_back(m_vdata.{{Table.Name}}RemapTableBuffer);
  }
  {% endfor %}
  
  {% if HasAllRefs %}
  all_references.resize(1); // need just single element to store all references
  {% endif %}

  {% for Var in ClassVectorVars %}
  m_vdata.{{Var.Name}}Buffer = vk_utils::createBuffer(device, {{Var.Name}}{{Var.AccessSymb}}capacity()*sizeof({{Var.TypeOfData}}), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_vdata.{{Var.Name}}Buffer);
  {% endfor %}

  {% for Var in ClassTextureVars %}
  m_vdata.{{Var.Name}}Texture = CreateTexture2D({{Var.Name}}{{Var.AccessSymb}}width(), {{Var.Name}}{{Var.AccessSymb}}height(), VkFormat({{Var.Format}}), {{Var.Usage}});
  {% if Var.NeedSampler %}
  m_vdata.{{Var.Name}}Sampler = CreateSampler({{Var.Name}}->sampler());
  {% endif %}
  memberTextures.push_back(m_vdata.{{Var.Name}}Texture);
  {% endfor %}
  
  {% for Var in ClassTexArrayVars %}
  m_vdata.{{Var.Name}}ArrayTexture.resize(0);
  m_vdata.{{Var.Name}}ArrayView.resize(0);
  m_vdata.{{Var.Name}}ArraySampler.resize(0);
  m_vdata.{{Var.Name}}ArrayTexture.reserve(64);
  m_vdata.{{Var.Name}}ArrayView.reserve(64);
  m_vdata.{{Var.Name}}ArraySampler.reserve(64);
  for(auto imageObj : {{Var.Name}})
  {
    auto tex = CreateTexture2D(imageObj->width(), imageObj->height(), VkFormat(imageObj->format()), {{Var.Usage}});
    auto sam = CreateSampler(imageObj->sampler());
    m_vdata.{{Var.Name}}ArrayTexture.push_back(tex);
    m_vdata.{{Var.Name}}ArrayView.push_back(VK_NULL_HANDLE);
    m_vdata.{{Var.Name}}ArraySampler.push_back(sam);
    memberTextures.push_back(tex);
  }
  {% endfor %}

  {% for Sam in SamplerMembers %}
  m_vdata.{{Sam}} = CreateSampler({{Sam}});
  {% endfor %}

  {% if length(IndirectDispatches) > 0 %}
  m_indirectBuffer = vk_utils::createBuffer(device, {{IndirectBufferSize}}*sizeof(uint32_t)*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  memberVectors.push_back(m_indirectBuffer);
  {% endif %}
  AllocMemoryForMemberBuffersAndImages(memberVectors, memberTextures);
  if(memberVectorsWithDevAddr.size() != 0)
    AllocAndBind(memberVectorsWithDevAddr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
  {% if HasAllRefs %}
  {
    {% for Var in ClassVectorVars %}
    {% if Var.IsVFHBuffer and Var.VFHLevel >= 2 %}
    {% for Impl in Var.Hierarchy.Implementations %}
    all_references[0].{{Impl.ClassName}}Address = vk_rt_utils::getBufferDeviceAddress(device, m_vdata.{{Var.Name}}_dataSBuffer) + {{Var.Name}}_obj_storage_offsets[{{Var.Hierarchy.Name}}::{{Impl.TagName}}];
    {% endfor %}
    {% endif %}
    {% endfor %}
    {% for Table in RemapTables %}
    all_references[0].{{Table.Name}}RemapAddr = vk_rt_utils::getBufferDeviceAddress(device, m_vdata.{{Table.Name}}RemapTableBuffer);
    {% endfor %}
  }
  {% endif %}
  {% for Var in ClassTexArrayVars %}
  for(size_t i = 0; i < {{Var.Name}}.size(); i++)
    m_vdata.{{Var.Name}}ArrayView[i] = CreateView(VkFormat({{Var.Name}}[i]->format()), m_vdata.{{Var.Name}}ArrayTexture[i]);
  {% endfor %}
  {% if length(IndirectDispatches) > 0 %}
  InitIndirectDescriptorSets();
  {% endif %}
}

{% if length(IndirectDispatches) > 0 %}
void {{MainClassName}}{{MainClassSuffix}}::InitIndirectBufferUpdateResources(const char* a_filePath)
{
  // (1) init m_indirectUpdateDSLayout
  //
  VkDescriptorSetLayoutBinding bindings[2] = {};
  bindings[0].binding            = 0;
  bindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount    = 1;
  bindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[0].pImmutableSamplers = nullptr;

  bindings[1].binding            = 1;
  bindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount    = 1;
  bindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = 2;
  descriptorSetLayoutCreateInfo.pBindings    = bindings;

  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &m_indirectUpdateDSLayout));

  VkDescriptorSetLayout oneTwo[2] = {m_indirectUpdateDSLayout,m_indirectUpdateDSLayout};

  VkPipelineLayoutCreateInfo  pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayoutInfo.pPushConstantRanges    = nullptr;
  pipelineLayoutInfo.pSetLayouts            = oneTwo;
  pipelineLayoutInfo.setLayoutCount         = 2;

  VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_indirectUpdateLayout));

  {% if ShaderGLSL %}
  {% for Dispatch in IndirectDispatches %}
  // create indrect update pipeline for {{Dispatch.OriginalName}}
  //
  {
    VkShaderModule tempShaderModule = VK_NULL_HANDLE;
    const std::string shaderPath = AlterShaderPath("{{ShaderFolder}}/{{Dispatch.OriginalName}}_UpdateIndirect.comp.spv");
    std::vector<uint32_t> code   = vk_utils::readSPVFile(shaderPath.c_str());

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode    = code.data();
    createInfo.codeSize = code.size()*sizeof(uint32_t);
    VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &tempShaderModule));

    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = tempShaderModule;
    shaderStageInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage  = shaderStageInfo;
    pipelineCreateInfo.layout = m_indirectUpdateLayout;
    VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &m_indirectUpdate{{Dispatch.KernelName}}Pipeline));

    vkDestroyShaderModule(device, tempShaderModule, VK_NULL_HANDLE);
  }
  {% endfor %}
  {% else %}
  VkShaderModule tempShaderModule = VK_NULL_HANDLE;
  std::vector<uint32_t> code = vk_utils::readSPVFile(a_filePath);
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.pCode    = code.data();
  createInfo.codeSize = code.size()*sizeof(uint32_t);
  VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &tempShaderModule));

  {% for Dispatch in IndirectDispatches %}
  // create indrect update pipeline for {{Dispatch.OriginalName}}
  //
  {
    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = tempShaderModule;
    shaderStageInfo.pName  = "{{Dispatch.OriginalName}}_UpdateIndirect";

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage  = shaderStageInfo;
    pipelineCreateInfo.layout = m_indirectUpdateLayout;
    VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &m_indirectUpdate{{Dispatch.KernelName}}Pipeline));
  }
  {% endfor %}

  vkDestroyShaderModule(device, tempShaderModule, VK_NULL_HANDLE);
  {% endif %} {# /* end else branch of if ShaderGLSL */ #}
}

VkBufferMemoryBarrier {{MainClassName}}{{MainClassSuffix}}::BarrierForIndirectBufferUpdate(VkBuffer a_buffer)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.pNext               = NULL;
  bar.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
  bar.dstAccessMask       = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = a_buffer;
  bar.offset              = 0;
  bar.size                = VK_WHOLE_SIZE;
  return bar;
}
{% endif %}

{% if UseSeparateUBO %}
VkBufferMemoryBarrier {{MainClassName}}{{MainClassSuffix}}::BarrierForArgsUBO(size_t a_size)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.pNext               = NULL;
  bar.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
  bar.dstAccessMask       = VK_ACCESS_UNIFORM_READ_BIT;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = m_uboArgsBuffer;
  bar.offset              = 0;
  bar.size                = a_size;
  return bar;
}
{% endif %}

{% if length(TextureMembers) > 0 or length(ClassTexArrayVars) > 0 %}
VkImage {{MainClassName}}{{MainClassSuffix}}::CreateTexture2D(const int a_width, const int a_height, VkFormat a_format, VkImageUsageFlags a_usage)
{
  VkImage result = VK_NULL_HANDLE;
  VkImageCreateInfo imgCreateInfo = {};
  imgCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgCreateInfo.pNext         = nullptr;
  imgCreateInfo.flags         = 0; // not sure about this ...
  imgCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
  imgCreateInfo.format        = a_format;
  imgCreateInfo.extent        = VkExtent3D{uint32_t(a_width), uint32_t(a_height), 1};
  imgCreateInfo.mipLevels     = 1;
  imgCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  imgCreateInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
  imgCreateInfo.usage         = a_usage;
  imgCreateInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imgCreateInfo.arrayLayers   = 1;
  VK_CHECK_RESULT(vkCreateImage(device, &imgCreateInfo, nullptr, &result));
  return result;
}

VkSampler {{MainClassName}}{{MainClassSuffix}}::CreateSampler(const Sampler& a_sampler) // TODO: implement this function correctly
{
  VkSampler result = VK_NULL_HANDLE;
  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.pNext        = nullptr;
  samplerInfo.flags        = 0;
  samplerInfo.magFilter    = VkFilter(int(a_sampler.filter));
  samplerInfo.minFilter    = VkFilter(int(a_sampler.filter));
  samplerInfo.mipmapMode   = (samplerInfo.magFilter == VK_FILTER_LINEAR ) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VkSamplerAddressMode(int(a_sampler.addressU));
  samplerInfo.addressModeV = VkSamplerAddressMode(int(a_sampler.addressV));
  samplerInfo.addressModeW = VkSamplerAddressMode(int(a_sampler.addressW));
  samplerInfo.mipLodBias   = a_sampler.mipLODBias;
  samplerInfo.compareOp    = VK_COMPARE_OP_NEVER;
  samplerInfo.minLod           = a_sampler.minLOD;
  samplerInfo.maxLod           = a_sampler.maxLOD;
  samplerInfo.maxAnisotropy    = a_sampler.maxAnisotropy;
  samplerInfo.anisotropyEnable = (a_sampler.maxAnisotropy > 1) ? VK_TRUE : VK_FALSE;
  samplerInfo.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &result));
  return result;
}

VkImageView {{MainClassName}}{{MainClassSuffix}}::CreateView(VkFormat a_format, VkImage a_image)
{
  VkImageView result = VK_NULL_HANDLE;
  VkImageViewCreateInfo createInfo{};
  createInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  createInfo.image    = a_image;
  createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  createInfo.format   = a_format;

  if(a_format == VK_FORMAT_R32_SFLOAT || a_format == VK_FORMAT_R8_UNORM  || a_format == VK_FORMAT_R8_SNORM ||
     a_format == VK_FORMAT_R16_SFLOAT || a_format == VK_FORMAT_R16_UNORM || a_format == VK_FORMAT_R16_SNORM)
  {
    createInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_R;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_R;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_R;
  }
  else
  {
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  }

  createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  createInfo.subresourceRange.baseMipLevel   = 0;
  createInfo.subresourceRange.levelCount     = 1;
  createInfo.subresourceRange.baseArrayLayer = 0;
  createInfo.subresourceRange.layerCount     = 1;

  VK_CHECK_RESULT(vkCreateImageView(device, &createInfo, nullptr, &result));
  return result;
}


{% if 0 %}
void {{MainClassName}}{{MainClassSuffix}}::TrackTextureAccess(const std::vector<TexAccessPair>& a_pairs, std::unordered_map<uint64_t, VkAccessFlags>& a_currImageFlags)
{
  if(a_pairs.size() == 0)
    return;

  VkImageSubresourceRange rangeWholeImage = {};
  rangeWholeImage.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  rangeWholeImage.baseMipLevel   = 0;
  rangeWholeImage.levelCount     = 1;
  rangeWholeImage.baseArrayLayer = 0;
  rangeWholeImage.layerCount     = 1;

  std::vector<VkImageMemoryBarrier> barriers(a_pairs.size());
  for(size_t i=0;i<a_pairs.size();i++)
  {
    VkImageMemoryBarrier& bar = barriers[i];
    bar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.pNext               = nullptr;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image               = a_pairs[i].image;
    bar.subresourceRange    = rangeWholeImage;

    uint64_t imageHandle = uint64_t(a_pairs[i].image);
    auto pState = a_currImageFlags.find(imageHandle);
    if(pState == a_currImageFlags.end())
    {
      bar.srcAccessMask = 0;
      bar.dstAccessMask = a_pairs[i].access;
      a_currImageFlags[imageHandle] = a_pairs[i].access;
      bar.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
      bar.newLayout     = (bar.dstAccessMask == VK_ACCESS_SHADER_READ_BIT) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    }
    else
    {
      bar.srcAccessMask = pState->second;
      bar.dstAccessMask = a_pairs[i].access;
      pState->second    = a_pairs[i].access;
      bar.oldLayout     = (bar.srcAccessMask == VK_ACCESS_SHADER_READ_BIT) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
      bar.newLayout     = (bar.dstAccessMask == VK_ACCESS_SHADER_READ_BIT) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    }
  }

  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,nullptr, 0,nullptr, uint32_t(barriers.size()),barriers.data());
}
{% endif %} {# /* 0 */ #}

{% endif %} {# /* length(TextureMembers) > 0 */ #}

void {{MainClassName}}{{MainClassSuffix}}::AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem)
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
      std::cout << "[{{MainClassName}}{{MainClassSuffix}}::AssignBuffersToMemory]: error, input buffers has different 'memReq.memoryTypeBits'" << std::endl;
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

{{MainClassName}}{{MainClassSuffix}}::MemLoc {{MainClassName}}{{MainClassSuffix}}::AllocAndBind(const std::vector<VkBuffer>& a_buffers, VkMemoryAllocateFlags a_flags)
{
  MemLoc currLoc;
  if(a_buffers.size() > 0)
  {
    currLoc.memObject = vk_utils::allocateAndBindWithPadding(device, physicalDevice, a_buffers, a_flags);
    currLoc.allocId   = m_allMems.size();
    m_allMems.push_back(currLoc);
  }
  return currLoc;
}

{{MainClassName}}{{MainClassSuffix}}::MemLoc {{MainClassName}}{{MainClassSuffix}}::AllocAndBind(const std::vector<VkImage>& a_images, VkMemoryAllocateFlags a_flags)
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
        std::cout << "{{MainClassName}}{{MainClassSuffix}}::AllocAndBind(textures): memoryTypeBits warning, need to split mem allocation (override me)" << std::endl;
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

void {{MainClassName}}{{MainClassSuffix}}::FreeAllAllocations(std::vector<MemLoc>& a_memLoc)
{
  // in general you may check 'mem.allocId' for unique to be sure you dont free mem twice
  // for default implementation this is not needed
  for(auto mem : a_memLoc)
    vkFreeMemory(device, mem.memObject, nullptr);
  a_memLoc.resize(0);
}

void {{MainClassName}}{{MainClassSuffix}}::AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_images)
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

  {% if length(ClassTextureVars) > 0 or length(ClassTexArrayVars) > 0 %}
  std::vector<VkFormat>             formats;  formats.reserve({{length(ClassTextureVars)}});
  std::vector<VkImageView*>         views;    views.reserve({{length(ClassTextureVars)}});
  std::vector<VkImage>              textures; textures.reserve({{length(ClassTextureVars)}});
  VkMemoryRequirements memoryRequirements;

  {% for Var in ClassTextureVars %}
  formats.push_back(VkFormat({{Var.Format}}));
  views.push_back(&m_vdata.{{Var.Name}}View);
  textures.push_back(m_vdata.{{Var.Name}}Texture);
  {% endfor %}
  {% for Var in ClassTexArrayVars %}
  for(size_t i=0;i< m_vdata.{{Var.Name}}ArrayTexture.size(); i++)
  {
    formats.push_back (VkFormat({{Var.Name}}[i]->format()));
    views.push_back   (&m_vdata.{{Var.Name}}ArrayView[i]);
    textures.push_back(m_vdata.{{Var.Name}}ArrayTexture[i]);
  }
  {% endfor %}

  AllocAndBind(textures);
  for(size_t i=0;i<textures.size();i++)
  {
    VkImageViewCreateInfo imageViewInfo = {};
    imageViewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewInfo.flags                           = 0;
    imageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format                          = formats[i];
    imageViewInfo.components                      = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
    imageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewInfo.subresourceRange.baseMipLevel   = 0;
    imageViewInfo.subresourceRange.baseArrayLayer = 0;
    imageViewInfo.subresourceRange.layerCount     = 1;
    imageViewInfo.subresourceRange.levelCount     = 1;
    imageViewInfo.image                           = textures[i];     // The view will be based on the texture's image
    VK_CHECK_RESULT(vkCreateImageView(device, &imageViewInfo, nullptr, views[i]));
  }
  {% endif %}
}
{% if UseRayGen %}
void {{MainClassName}}{{MainClassSuffix}}::AllocAllShaderBindingTables()
{
  m_allShaderTableBuffers.clear();
  uint32_t customStages    = {{length(IntersectionHierarhcy.Implementations)}};
  uint32_t numShaderGroups = 4 + customStages;            // (raygen, miss, miss, rchit(tris)) + ({% for Impl in IntersectionHierarhcy.Implementations %}{{Impl.ClassName}}, {% endfor %})
  uint32_t numHitStages    = uint32_t(customStages) + 1u; // 1 if we don't have actual sbtRecordOffsets at all
  uint32_t numMissStages   = 2u;                          // common mis shader and shadow miss  

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
  
  //assert(handleSize == handleSizeAligned);

  const auto rgenStride = vk_utils::getSBTAlignedSize(handleSizeAligned,                 rtPipelineProperties.shaderGroupBaseAlignment);
  const auto missSize   = vk_utils::getSBTAlignedSize(numMissStages * handleSizeAligned, rtPipelineProperties.shaderGroupBaseAlignment);
  const auto hitSize    = vk_utils::getSBTAlignedSize(numHitStages  * handleSizeAligned, rtPipelineProperties.shaderGroupBaseAlignment);

  std::vector<VkPipeline> allRTPipelines = {};
  {% for Kernel in Kernels %}
  {% if Kernel.UseRayGen %}
  if({{Kernel.Name}}Pipeline != VK_NULL_HANDLE)
    allRTPipelines.push_back({{Kernel.Name}}Pipeline);
  {% endif%}
  {% endfor %}

  // (1) create buffers for SBT
  //
  for(VkPipeline rtPipeline : allRTPipelines) // todo add for loop
  {
    VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    auto raygenBuf  = vk_utils::createBuffer(device, rgenStride, flags);
    auto raymissBuf = vk_utils::createBuffer(device, missSize * numMissStages, flags);
    auto rayhitBuf  = vk_utils::createBuffer(device, hitSize * numHitStages, flags);

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
  {% for Kernel in Kernels %}
  {% if Kernel.UseRayGen %}
  if({{Kernel.Name}}Pipeline != VK_NULL_HANDLE)
  {
    VK_CHECK_RESULT(vkGetRayTracingShaderGroupHandlesKHR(device, allRTPipelines[groupId], 0, numShaderGroups, sbtSize, shaderHandleStorage.data()));

    auto raygenBuf  = m_allShaderTableBuffers[groupId*3+0];
    auto raymissBuf = m_allShaderTableBuffers[groupId*3+1];
    auto rayhitBuf  = m_allShaderTableBuffers[groupId*3+2];

    {{Kernel.Name}}SBTStrides.resize(4);
    {{Kernel.Name}}SBTStrides[0] = VkStridedDeviceAddressRegionKHR{ vk_rt_utils::getBufferDeviceAddress(device, raygenBuf),  rgenStride,         rgenStride };
    {{Kernel.Name}}SBTStrides[1] = VkStridedDeviceAddressRegionKHR{ vk_rt_utils::getBufferDeviceAddress(device, raymissBuf), handleSizeAligned,  missSize };
    {{Kernel.Name}}SBTStrides[2] = VkStridedDeviceAddressRegionKHR{ vk_rt_utils::getBufferDeviceAddress(device, rayhitBuf),  handleSizeAligned,  hitSize};
    {{Kernel.Name}}SBTStrides[3] = VkStridedDeviceAddressRegionKHR{ 0u, 0u, 0u };

    auto *pData = shaderHandleStorage.data();

    memcpy(mapped + offsets[groupId*3 + 0], pData, handleSize * 1);             // raygenBuf
    pData += handleSize * 1;

    memcpy(mapped + offsets[groupId*3 + 1], pData, handleSize * numMissStages); // raymissBuf
    pData += handleSize * numMissStages;

    memcpy(mapped + offsets[groupId*3 + 2], pData, handleSize * 1);             // rayhitBuf part for hw accelerated triangles
    pData += handleSize * 1;
                                                                                // rayhitBuf part for custom primitives
    {% for Impl in IntersectionHierarhcy.Implementations %}                     
    memcpy(mapped + offsets[groupId*3 + 2] + handleSize*({{IntersectionHierarhcy.Name}}::{{Impl.TagName}}), pData + {{loop.index}}*handleSize, handleSize); // {{Impl.ClassName}}
    {% endfor %}
    pData += handleSize*customStages;

    groupId++;
  }
  {% endif%}
  {% endfor %}

  vkUnmapMemory(device, m_allShaderTableMem);
}
{% endif %}
{% if UseServiceScan %}

inline size_t sblocksST(size_t elems, int threadsPerBlock)
{
  if (elems % threadsPerBlock == 0 && elems >= threadsPerBlock)
    return elems / threadsPerBlock;
  else
    return (elems / threadsPerBlock) + 1;
}

inline size_t sRoundBlocks(size_t elems, int threadsPerBlock)
{
  if (elems < threadsPerBlock)
    return (size_t)threadsPerBlock;
  else
    return sblocksST(elems, threadsPerBlock) * threadsPerBlock;
}

std::vector<VkBuffer> {{MainClassName}}{{MainClassSuffix}}::ScanData::InitTempBuffers(VkDevice a_device, size_t a_maxSize)
{
  m_scanMipOffsets.resize(0);
  size_t currSize = a_maxSize;
  size_t currOffset = 0;
  for (int i = 0; i < 16; i++)
  {
    size_t size2 = sRoundBlocks(currSize, 256) / 256;
    if (currSize > 0)
    {
      size_t size3 = std::max(size2, size_t(256));
      m_scanMipOffsets.push_back(currOffset);
      currOffset += size3;
    }
    else
    {
      m_scanMipOffsets.push_back(currOffset);
      currOffset += 256;
      break;
    }
    currSize = currSize / 256;
  }

  m_scanTempDataBuffer = vk_utils::createBuffer(a_device, currOffset*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_scanTempDataOffset = 0;
  m_scanMaxSize        = a_maxSize;
  return {m_scanTempDataBuffer};
}

void {{MainClassName}}{{MainClassSuffix}}::ScanData::DeleteTempBuffers(VkDevice a_device)
{
  vkDestroyBuffer(a_device, m_scanTempDataBuffer, nullptr);
  m_scanTempDataBuffer = VK_NULL_HANDLE;
  m_scanMipOffsets.resize(0);
}

VkDescriptorSetLayout {{MainClassName}}{{MainClassSuffix}}::ScanData::CreateInternalScanDSLayout(VkDevice a_device)
{
  std::array<VkDescriptorSetLayoutBinding, 3> dsBindings;

  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = dsBindings.size();
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

void {{MainClassName}}{{MainClassSuffix}}::ScanData::DeleteDSLayouts(VkDevice a_device)
{
  vkDestroyDescriptorSetLayout(a_device, internalDSLayout, nullptr);
}

void {{MainClassName}}{{MainClassSuffix}}::ScanData::ExclusiveScanCmd(VkCommandBuffer a_cmdBuffer, size_t a_size)
{
  InclusiveScanCmd(a_cmdBuffer, a_size, true);
}

void {{MainClassName}}{{MainClassSuffix}}::ScanData::InclusiveScanCmd(VkCommandBuffer a_cmdBuffer, size_t a_size, bool actuallyExclusive)
{
  if (m_scanMaxSize < a_size)
  {
    std::cout << "InclusiveScanCmd: too big input size = " << a_size << ", maximum allowed is " << m_scanMaxSize << std::endl;
    return;
  }

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  VkBufferMemoryBarrier bufBars[2] = {};
  bufBars[0].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufBars[0].pNext               = NULL;
  bufBars[0].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
  bufBars[0].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
  bufBars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufBars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufBars[0].buffer              = m_scanTempDataBuffer;
  bufBars[0].offset              = 0;
  bufBars[0].size                = VK_WHOLE_SIZE;

  bufBars[1] = bufBars[0];
  bufBars[1].srcAccessMask = 0;                          // we don't going to read 'next' part of buffer in next kernel launch, just write it
  bufBars[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // we don't going to read 'next' part of buffer in next kernel launch, just write it


  size_t sizeOfElem = sizeof(uint32_t);

  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t iNumElementsX;
    uint32_t currMip;
    uint32_t currPassOffset;
    uint32_t nextPassOffset;
    uint32_t exclusiveFlag;
  } pcData;

  pcData.exclusiveFlag = actuallyExclusive ? 1 : 0;

  std::vector<size_t> lastSizeV;
  std::vector< std::pair<size_t,size_t> > offsets;

  vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scanFwdPipeline);

  // down, scan phase // fixed
  //
  int currMip = 0;
  size_t currOffset = 0;
  for (size_t currSize = a_size; currSize > 1; currSize = currSize / 256)
  {
    lastSizeV.push_back(currSize);

    const size_t runSize  = sRoundBlocks(currSize, 256);
    const size_t nextSize = runSize / 256;
    pcData.iNumElementsX  = uint32_t(runSize);
    pcData.currMip        = uint32_t(currMip);
    if(currMip == 0)
    {
      pcData.currPassOffset = 0;
      pcData.nextPassOffset = 0;
    }
    else
    {
      pcData.currPassOffset = currOffset;
      pcData.nextPassOffset = currOffset + runSize;
      offsets.push_back( std::make_pair(currOffset, currOffset + runSize) );
      currOffset += runSize;
    }

    vkCmdPushConstants(a_cmdBuffer, scanFwdLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    vkCmdDispatch(a_cmdBuffer, (runSize + blockSizeX - 1) / blockSizeX, 1, 1);

    if(currMip == 0)
      vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    else
    {
      bufBars[0].offset = pcData.nextPassOffset*sizeOfElem;
      bufBars[0].size   = nextSize*sizeOfElem;
      bufBars[1].offset = pcData.currPassOffset*sizeOfElem;
      bufBars[1].size   = runSize*sizeOfElem;
      vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 2, bufBars, 0, nullptr);
    }

    currMip++;
  }

  currMip--;

  bufBars[0].offset = 0;
  bufBars[0].size   = VK_WHOLE_SIZE;

  vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scanPropPipeline);

  // up, propagate phase
  //
  while (currMip >= 0)
  {
    size_t currSize = lastSizeV.back();
    lastSizeV.pop_back();

    const size_t runSize  = sRoundBlocks(currSize, 256);
    pcData.iNumElementsX  = uint32_t(runSize);
    pcData.currMip        = uint32_t(currMip);
    if(currMip == 0)
    {
      pcData.currPassOffset = 0;
      pcData.nextPassOffset = 0;
    }
    else
    {
      auto pair = offsets[currMip-1];
      pcData.currPassOffset = pair.second;
      pcData.nextPassOffset = pair.first;
    }

    vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, bufBars, 0, nullptr);
    vkCmdPushConstants(a_cmdBuffer, scanFwdLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    vkCmdDispatch(a_cmdBuffer, (runSize + blockSizeX - 1) / blockSizeX, 1, 1);
    currMip--;
  }
}
{% endif %}
{% if UseServiceSort %}
VkDescriptorSetLayout {{MainClassName}}{{MainClassSuffix}}::BitonicSortData::CreateSortDSLayout(VkDevice a_device)
{
  std::array<VkDescriptorSetLayoutBinding, 1> dsBindings;

  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = dsBindings.size();
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

void {{MainClassName}}{{MainClassSuffix}}::BitonicSortData::DeleteDSLayouts(VkDevice a_device)
{
  vkDestroyDescriptorSetLayout(a_device, sortDSLayout, nullptr);
}

static bool isPowerOfTwo(size_t n)
{
  if (n == 0)
    return false;
  return (std::ceil(std::log2(n)) == std::floor(std::log2(n)));
}

void {{MainClassName}}{{MainClassSuffix}}::BitonicSortData::BitonicSortSimpleCmd(VkCommandBuffer a_cmdBuffer, size_t a_size)
{
  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };

  if(!isPowerOfTwo(a_size))
  {
    std::cout << "BitonicSortCmd, bad input size " << a_size << ", it must be power of 2" << std::endl;
    return;
  }

  int numStages = 0;
  for (size_t temp = a_size; temp > 2; temp >>= 1)
    numStages++;

  const size_t blockSizeX = 256;
  struct KernelArgsPC
  {
    int iNumElementsX;
    int stage;
    int passOfStage;
    int invertModeOn;
  } pcData;

  vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonicPassPipeline);

  // up, form bitonic sequence with half arrays
  //
  for (int stage = 0; stage < numStages; stage++)
  {
    for (int passOfStage = stage; passOfStage >= 0; passOfStage--)
    {
      pcData.iNumElementsX = int(a_size);
      pcData.stage         = stage;
      pcData.passOfStage   = passOfStage;
      pcData.invertModeOn  = 1;
      vkCmdPushConstants(a_cmdBuffer, bitonicPassLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
      vkCmdDispatch(a_cmdBuffer, (a_size + blockSizeX - 1) / blockSizeX, 1, 1);
      vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    }
  }

  // down, finally sort it
  //
  for(int passOfStage = numStages; passOfStage >= 0; passOfStage--)
  {
    pcData.iNumElementsX = int(a_size);
    pcData.stage         = numStages - 1;
    pcData.passOfStage   = passOfStage;
    pcData.invertModeOn  = 0;
    vkCmdPushConstants(a_cmdBuffer, bitonicPassLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    vkCmdDispatch(a_cmdBuffer, (a_size + blockSizeX - 1) / blockSizeX, 1, 1);
    vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
  }
}

void {{MainClassName}}{{MainClassSuffix}}::BitonicSortData::BitonicSortCmd(VkCommandBuffer a_cmdBuffer, size_t a_size, uint32_t a_maxWorkGroupSize)
{
  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };

  if(!isPowerOfTwo(a_size))
  {
    std::cout << "BitonicSortCmd, bad input size " << a_size << ", it must be power of 2" << std::endl;
    return;
  }

  int numStages = 0;
  for (size_t temp = a_size; temp > 2; temp >>= 1)
    numStages++;

  const size_t blockSizeX = 256;
  struct KernelArgsPC
  {
    int iNumElementsX;
    int stage;
    int passOfStage;
    int invertModeOn;
  } pcData;

  // up, form bitonic sequence with half arrays
  //
  for (int stage = 0; stage < numStages; stage++)
  {
    vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonicPassPipeline);

    for (int passOfStage = stage; passOfStage >= 0; passOfStage--)
    {
      bool stopNow = false;

      if (passOfStage > 0 && passOfStage <= 10 && a_maxWorkGroupSize >= 1024)
      {
        vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic2048Pipeline);
        stopNow = true;
      }
      else if (passOfStage > 0 && passOfStage <= 9 && a_maxWorkGroupSize >= 512)
      {
        vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic1024Pipeline);
        stopNow = true;
      }
      else if (passOfStage > 0 && passOfStage <= 8 && a_maxWorkGroupSize >= 256)
      {
        vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic512Pipeline);
        stopNow = true;
      }

      pcData.iNumElementsX = int(a_size);
      pcData.stage         = stage;
      pcData.passOfStage   = passOfStage;
      pcData.invertModeOn  = 1;
      vkCmdPushConstants(a_cmdBuffer, bitonicPassLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
      vkCmdDispatch(a_cmdBuffer, (a_size + blockSizeX - 1) / blockSizeX, 1, 1);
      vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

      if(stopNow)
        break;
    }
  }

  vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonicPassPipeline);

  // down, finally sort it
  //
  for(int passOfStage = numStages; passOfStage >= 0; passOfStage--)
  {
    bool stopNow = false;

    if (passOfStage > 0 && passOfStage <= 10 && a_maxWorkGroupSize >= 1024)
    {
      vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic2048Pipeline);
      stopNow = true;
    }
    else if (passOfStage > 0 && passOfStage <= 9 && a_maxWorkGroupSize >= 512)
    {
      vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic1024Pipeline);
      stopNow = true;
    }
    else if (passOfStage > 0 && passOfStage <= 8 && a_maxWorkGroupSize >= 256)
    {
      vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic512Pipeline);
      stopNow = true;
    }

    pcData.iNumElementsX = int(a_size);
    pcData.stage         = numStages - 1;
    pcData.passOfStage   = passOfStage;
    pcData.invertModeOn  = 0;
    vkCmdPushConstants(a_cmdBuffer, bitonicPassLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    vkCmdDispatch(a_cmdBuffer, (a_size + blockSizeX - 1) / blockSizeX, 1, 1);
    vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    if(stopNow)
      break;
  }
}
{% endif %}

VkPhysicalDeviceFeatures2 {{MainClassName}}{{MainClassSuffix}}::ListRequiredDeviceFeatures(std::vector<const char*>& deviceExtensions)
{
  static VkPhysicalDeviceFeatures2 features2 = {};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = nullptr;
  features2.features.shaderInt64   = {{GlobalUseInt64}};
  features2.features.shaderFloat64 = {{GlobalUseFloat64}};
  features2.features.shaderInt16   = {{GlobalUseInt16}};

  void** ppNext = &features2.pNext;
  {% if GlobalUseInt8 or GlobalUseHalf %}
  deviceExtensions.push_back("VK_KHR_shader_float16_int8");
  {% endif %}
  {% if HasRTXAccelStruct %}
  {
    static VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelStructFeatures = {};
    static VkPhysicalDeviceBufferDeviceAddressFeatures      enabledDeviceAddressFeatures = {};
    static VkPhysicalDeviceRayQueryFeaturesKHR              enabledRayQueryFeatures =  {};
    static VkPhysicalDeviceDescriptorIndexingFeatures       indexingFeatures = {};
    {% if UseRayGen %}
    static VkPhysicalDeviceRayTracingPipelineFeaturesKHR    enabledRTPipelineFeatures = {};
    {% if UseMotionBlur %}
    static VkPhysicalDeviceRayTracingMotionBlurFeaturesNV   enabledMotionBlurFeatures = {};
    {% endif %}
    {% endif %}

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
    {% if UseRayGen %}
    {% if UseMotionBlur %}
    enabledMotionBlurFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV;
    enabledMotionBlurFeatures.rayTracingMotionBlur = VK_TRUE;
    enabledMotionBlurFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect = VK_FALSE; // not using indirect rt in this impl.
    enabledMotionBlurFeatures.pNext = &enabledAccelStructFeatures;

    enabledRTPipelineFeatures.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    enabledRTPipelineFeatures.rayTracingPipeline = VK_TRUE;
    enabledRTPipelineFeatures.pNext              = &enabledMotionBlurFeatures;
    {% else %}
    enabledRTPipelineFeatures.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    enabledRTPipelineFeatures.rayTracingPipeline = VK_TRUE;
    enabledRTPipelineFeatures.pNext              = &enabledAccelStructFeatures;
    {% endif %}
    (*ppNext) = &enabledRTPipelineFeatures;
    {% else %}
    (*ppNext) = &enabledAccelStructFeatures;
    {% endif %}
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
    {% if UseRayGen %}
    // Required by VK_KHR_ray_tracing_pipeline
    deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    {% if UseMotionBlur %}
    deviceExtensions.push_back(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME);
    {% endif %}
    {% endif %}
  }
  {% endif %}
  {% if HasVarPointers %}
  static VkPhysicalDeviceVariablePointersFeatures varPointersQuestion = {};
  varPointersQuestion.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
  (*ppNext) = &varPointersQuestion; ppNext = &varPointersQuestion.pNext;
  deviceExtensions.push_back("VK_KHR_variable_pointers");
  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info"); // for clspv
  {% endif %}
  {% if HasAllRefs and not HasRTXAccelStruct %} {# /***** buffer device address ********/ #}
  static VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeatures = {};
  bufferDeviceAddressFeatures.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
  bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
  (*ppNext) = &bufferDeviceAddressFeatures; ppNext = &bufferDeviceAddressFeatures.pNext;
  deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  {% endif %} {# /***** buffer device address ********/ #}
  return features2;
}

{{MainClassName}}{{MainClassSuffix}}::MegaKernelIsEnabled {{MainClassName}}{{MainClassSuffix}}::m_megaKernelFlags;
