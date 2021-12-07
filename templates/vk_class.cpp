#include <vector>
#include <memory>
#include <limits>
#include <cassert>

#include "vulkan_basics.h"
#include "{{IncludeClassDecl}}"
#include "include/{{UBOIncl}}"

{% if length(SceneMembers) > 0 %}
#include "CrossRT.h"
ISceneObject* CreateVulkanRTX(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_graphicsQId, std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper,
                              uint32_t a_maxMeshes, uint32_t a_maxTotalVertices, uint32_t a_maxTotalPrimitives, uint32_t a_maxPrimitivesPerMesh,
                              bool build_as_add);
{% endif %}

VkBufferUsageFlags {{MainClassName}}_Generated::GetAdditionalFlagsForUBO() const
{
  return 0;
}

static uint32_t ComputeReductionSteps(uint32_t whole_size, uint32_t wg_size)
{
  uint32_t steps = 0;
  while (whole_size > 1)
  {
    steps++;
    whole_size = (whole_size + wg_size - 1) / wg_size;
  }
  return steps;
}

void {{MainClassName}}_Generated::InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount,
                                                    std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
{
  physicalDevice = a_physicalDevice;
  device         = a_device;
  InitHelpers();
  InitBuffers(a_maxThreadsCount, true);
  InitKernels("{{ShaderSingleFile}}.spv");
  AllocateAllDescriptorSets();

  {% if length(SceneMembers) > 0 %}
  auto queueAllFID = vk_utils::getQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT);
  {% endif %}
  {% for ScnObj in SceneMembers %}
  //@TODO: calculate these somehow?
  uint32_t maxMeshes = 1024;
  uint32_t maxTotalVertices = 1'000'000;
  uint32_t maxTotalPrimitives = 1'000'000;
  uint32_t maxPrimitivesPerMesh = 200'000;
  {{ScnObj}} = std::shared_ptr<ISceneObject>(CreateVulkanRTX(a_device, a_physicalDevice, queueAllFID, a_pCopyEngine,
                                                             maxMeshes, maxTotalVertices, maxTotalPrimitives, maxPrimitivesPerMesh, true),
                                                            [](ISceneObject *p) { DeleteSceneRT(p); } );
  {% endfor %}
}

void {{MainClassName}}_Generated::UpdatePlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
{
  const size_t maxAllowedSize = std::numeric_limits<uint32_t>::max();

## for Var in ClassVars
  {% if Var.IsArray %}
  memcpy(m_uboData.{{Var.Name}},{{Var.Name}},sizeof({{Var.Name}}));
  {% else %}
  m_uboData.{{Var.Name}} = {{Var.Name}};
  {% endif %}
## endfor
## for Var in ClassVectorVars 
  m_uboData.{{Var.Name}}_size     = uint32_t( {{Var.Name}}.size() );    assert( {{Var.Name}}.size() < maxAllowedSize );
  m_uboData.{{Var.Name}}_capacity = uint32_t( {{Var.Name}}.capacity() ); assert( {{Var.Name}}.capacity() < maxAllowedSize );
## endfor
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 0, &m_uboData, sizeof(m_uboData));
}

void {{MainClassName}}_Generated::UpdateVectorMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
{
  {% for Var in ClassVectorVars %}
  if({{Var.Name}}.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.{{Var.Name}}Buffer, 0, {{Var.Name}}.data(), {{Var.Name}}.size()*sizeof({{Var.TypeOfData}}) );
  {% endfor %}
}

void {{MainClassName}}_Generated::UpdateTextureMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
{ 
  {% if length(ClassTextureVars) > 0 %}
  {% for Var in ClassTextureVars %}
  {% if Var.NeedUpdate %}
  a_pCopyEngine->UpdateImage(m_vdata.{{Var.Name}}Texture, {{Var.Name}}.getRawData(), {{Var.Name}}.width(), {{Var.Name}}.height(), {{Var.Name}}.bpp()); 
  {% endif %}
  {% endfor %}
  
  std::array<VkImageMemoryBarrier, {{length(ClassTextureVars)}}> barriers;

  {% for Var in ClassTextureVars %}
  barriers[{{loop.index}}].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barriers[{{loop.index}}].pNext               = nullptr;
  barriers[{{loop.index}}].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barriers[{{loop.index}}].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barriers[{{loop.index}}].srcAccessMask       = 0;                                        
  barriers[{{loop.index}}].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;       
  barriers[{{loop.index}}].image               = m_vdata.{{Var.Name}}Texture;
  barriers[{{loop.index}}].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barriers[{{loop.index}}].subresourceRange.baseMipLevel   = 0;
  barriers[{{loop.index}}].subresourceRange.levelCount     = 1;
  barriers[{{loop.index}}].subresourceRange.baseArrayLayer = 0;
  barriers[{{loop.index}}].subresourceRange.layerCount     = 1;
  {% if Var.NeedUpdate %}
  barriers[{{loop.index}}].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barriers[{{loop.index}}].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  {% else %}
  barriers[{{loop.index}}].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barriers[{{loop.index}}].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  {% endif %}

  {% endfor %}
  
  VkCommandBuffer cmdBuff       = a_pCopyEngine->CmdBuffer();
  VkQueue         transferQueue = a_pCopyEngine->TransferQueue();

  vkResetCommandBuffer(cmdBuff, 0);
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  if (vkBeginCommandBuffer(cmdBuff, &beginInfo) != VK_SUCCESS)
    throw std::runtime_error("{{MainClassName}}_Generated::UpdateTextureMembers: failed to begin command buffer!");
  vkCmdPipelineBarrier(cmdBuff,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,uint32_t(barriers.size()),barriers.data());
  vkEndCommandBuffer(cmdBuff);
  
  vk_utils::executeCommandBufferNow(cmdBuff, transferQueue, device);
  {% endif %}
}

## for Kernel in Kernels
{% if Kernel.IsIndirect %}
void {{MainClassName}}_Generated::{{Kernel.Name}}_UpdateIndirect()
{
  VkBufferMemoryBarrier barIndirect = BarrierForIndirectBufferUpdate(m_indirectBuffer);
  vkCmdBindDescriptorSets(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectUpdateLayout, 0, 1, &m_indirectUpdateDS, 0, nullptr);
  vkCmdBindPipeline      (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectUpdate{{Kernel.Name}}Pipeline);
  vkCmdDispatch          (m_currCmdBuffer, 1, 1, 1);
  vkCmdPipelineBarrier   (m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, 1, &barIndirect, 0, nullptr);
}

{% endif %}
void {{MainClassName}}_Generated::{{Kernel.Decl}}
{
  uint32_t blockSizeX = {{Kernel.WGSizeX}};
  uint32_t blockSizeY = {{Kernel.WGSizeY}};
  uint32_t blockSizeZ = {{Kernel.WGSizeZ}};

  struct KernelArgsPC
  {
    {% for Arg in Kernel.AuxArgs %}
    {{Arg.Type}} m_{{Arg.Name}}; 
    {% endfor %}
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  {% if Kernel.SmplX %}
  uint32_t sizeX  = uint32_t({{Kernel.tidX}});
  {% else %}
  uint32_t sizeX  = uint32_t(std::abs(int32_t({{Kernel.tidX}}) - int32_t({{Kernel.begX}})));
  {% endif %}
  {% if Kernel.SmplY %}
  uint32_t sizeY  = uint32_t({{Kernel.tidY}});
  {% else %}
  uint32_t sizeY  = uint32_t(std::abs(int32_t({{Kernel.tidY}}) - int32_t({{Kernel.begY}})));
  {% endif %}
  {% if Kernel.SmplZ %}
  uint32_t sizeZ  = uint32_t({{Kernel.tidZ}});
  {% else %}
  uint32_t sizeZ  = uint32_t(std::abs(int32_t({{Kernel.tidZ}}) - int32_t({{Kernel.begZ}})));
  {% endif %}
  
  pcData.m_sizeX  = {{Kernel.tidX}};
  pcData.m_sizeY  = {{Kernel.tidY}};
  pcData.m_sizeZ  = {{Kernel.tidZ}};
  pcData.m_tFlags = m_currThreadFlags;
  {% for Arg in Kernel.AuxArgs %}
  pcData.m_{{Arg.Name}} = {{Arg.Name}}; 
  {% endfor %}
  {% if Kernel.HasLoopFinish %}
  KernelArgsPC oldPCData = pcData;
  {% endif %}

  {% if UseSeparateUBO %}
  {
    vkCmdUpdateBuffer(m_currCmdBuffer, m_uboArgsBuffer, 0, sizeof(KernelArgsPC), &pcData);
    VkBufferMemoryBarrier barUBO2 = BarrierForArgsUBO(sizeof(KernelArgsPC));
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO2, 0, nullptr);
  }
  {% else %}
  vkCmdPushConstants(m_currCmdBuffer, {{Kernel.Name}}Layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  {% endif %}
  {% if Kernel.HasLoopInit %}
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}InitPipeline);
  vkCmdDispatch(m_currCmdBuffer, 1, 1, 1); 
  VkBufferMemoryBarrier barUBO = BarrierForSingleBuffer(m_classDataBuffer);
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO, 0, nullptr);
  {% endif %}
  
  {# /* --------------------------------------------------------------------------------------------------------------------------------------- */ #}
  {% if Kernel.IsMaker and Kernel.Hierarchy.IndirectDispatch %}
  VkBufferMemoryBarrier objCounterBar = BarrierForObjCounters(m_classDataBuffer);
  VkBufferMemoryBarrier barIndirect   = BarrierForIndirectBufferUpdate(m_indirectBuffer);

  // (1) zero obj. counters 
  //
  vkCmdBindPipeline   (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}ZeroObjCounters);
  vkCmdDispatch       (m_currCmdBuffer, 1, 1, 1); 
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &objCounterBar, 0, nullptr);
  
  // (2)  execute maker first time, count objects for each class 
  //
  vkCmdBindPipeline   (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Pipeline);
  vkCmdDispatch       (m_currCmdBuffer, (sizeX + blockSizeX - 1) / blockSizeX, (sizeY + blockSizeY - 1) / blockSizeY, (sizeZ + blockSizeZ - 1) / blockSizeZ); 
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &objCounterBar, 0, nullptr);

  // (3) small prefix summ to compute global offsets for each type region
  //
  vkCmdBindPipeline   (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}CountTypeIntervals);
  vkCmdDispatch       (m_currCmdBuffer, 1, 1, 1); 
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &objCounterBar, 0, nullptr);

  // (4) execute maker second time, count offset for each object using local prefix summ (in the work group) and put ObjPtr at this offset 
  //
  vkCmdBindPipeline   (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Sorter);
  vkCmdDispatch       (m_currCmdBuffer, (sizeX + blockSizeX - 1) / blockSizeX, (sizeY + blockSizeY - 1) / blockSizeY, (sizeZ + blockSizeZ - 1) / blockSizeZ); 

  // (5) update indirect buffer for futher vkernels dispatching
  //
  vkCmdBindDescriptorSets(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectUpdateLayout, 0, 1, &m_indirectUpdateDS, 0, nullptr);
  vkCmdBindPipeline      (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectUpdate{{Kernel.Name}}Pipeline);
  vkCmdDispatch          (m_currCmdBuffer, 1, 1, 1);
  vkCmdPipelineBarrier   (m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, 1, &barIndirect, 0, nullptr);

  {% else if Kernel.IsVirtual and Kernel.Hierarchy.IndirectDispatch %}
  // use reverse order of classes because assume heavy implementations in the end
  {
    {% for Impl in Kernel.Hierarchy.Implementations %}
    vkCmdBindPipeline    (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}PipelineArray[{{length(Kernel.Hierarchy.Implementations)}}-{{loop.index}}-1]);
    vkCmdDispatchIndirect(m_currCmdBuffer, m_indirectBuffer, ({{length(Kernel.Hierarchy.Implementations)}}-{{loop.index}}-1+{{Kernel.Hierarchy.IndirectOffset}})*sizeof(uint32_t)*4);

    {% endfor %}
  }
  {% else if Kernel.IsIndirect %}
  vkCmdBindPipeline    (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Pipeline);
  vkCmdDispatchIndirect(m_currCmdBuffer, m_indirectBuffer, {{Kernel.IndirectOffset}}*sizeof(uint32_t)*4);
  {% else %}
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Pipeline);
  vkCmdDispatch    (m_currCmdBuffer, (sizeX + blockSizeX - 1) / blockSizeX, (sizeY + blockSizeY - 1) / blockSizeY, (sizeZ + blockSizeZ - 1) / blockSizeZ);
  {% if Kernel.FinishRed %}
  {% include "inc_reduction_vulkan.cpp" %}
  {% endif %} {# /* Kernel.FinishRed      */ #}
  {% endif %} {# /* NOT INDIRECT DISPATCH */ #}
  {# /* --------------------------------------------------------------------------------------------------------------------------------------- */ #}
 
  {% if Kernel.HasLoopFinish %}
  VkBufferMemoryBarrier barUBOFin = BarrierForSingleBuffer(m_classDataBuffer);
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBOFin, 0, nullptr);
  {% if UseSeparateUBO %}
  {
    vkCmdUpdateBuffer(m_currCmdBuffer, m_uboArgsBuffer, 0, sizeof(KernelArgsPC), &oldPCData);
    VkBufferMemoryBarrier barUBO2 = BarrierForArgsUBO(sizeof(KernelArgsPC));
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO2, 0, nullptr);
  }
  {% else %}
  vkCmdPushConstants(m_currCmdBuffer, {{Kernel.Name}}Layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &oldPCData);
  {% endif %}
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}FinishPipeline);
  vkCmdDispatch(m_currCmdBuffer, 1, 1, 1); 
  
  {% endif %}   
}

## endfor

void {{MainClassName}}_Generated::copyKernelFloatCmd(uint32_t length)
{
  uint32_t blockSizeX = MEMCPY_BLOCK_SIZE;

  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, copyKernelFloatPipeline);
  {% if UseSeparateUBO %}
  {
    vkCmdUpdateBuffer(m_currCmdBuffer, m_uboArgsBuffer, 0, sizeof(uint32_t), &length);
    VkBufferMemoryBarrier barUBO2 = BarrierForArgsUBO(sizeof(uint32_t));
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO2, 0, nullptr);
    vkCmdPushConstants(m_currCmdBuffer, copyKernelFloatLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &length);
  }
  {% else %}
  vkCmdPushConstants(m_currCmdBuffer, copyKernelFloatLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &length);
  {% endif %}
  vkCmdDispatch(m_currCmdBuffer, (length + blockSizeX - 1) / blockSizeX, 1, 1);
}

VkBufferMemoryBarrier {{MainClassName}}_Generated::BarrierForClearFlags(VkBuffer a_buffer)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.pNext               = NULL;
  bar.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
  bar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = a_buffer;
  bar.offset              = 0;
  bar.size                = VK_WHOLE_SIZE;
  return bar;
}

VkBufferMemoryBarrier {{MainClassName}}_Generated::BarrierForSingleBuffer(VkBuffer a_buffer)
{
  VkBufferMemoryBarrier bar = {};
  bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bar.pNext               = NULL;
  bar.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
  bar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
  bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bar.buffer              = a_buffer;
  bar.offset              = 0;
  bar.size                = VK_WHOLE_SIZE;
  return bar;
}

void {{MainClassName}}_Generated::BarriersForSeveralBuffers(VkBuffer* a_inBuffers, VkBufferMemoryBarrier* a_outBarriers, uint32_t a_buffersNum)
{
  for(uint32_t i=0; i<a_buffersNum;i++)
  {
    a_outBarriers[i].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    a_outBarriers[i].pNext               = NULL;
    a_outBarriers[i].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    a_outBarriers[i].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    a_outBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    a_outBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    a_outBarriers[i].buffer              = a_inBuffers[i];
    a_outBarriers[i].offset              = 0;
    a_outBarriers[i].size                = VK_WHOLE_SIZE;
  }
}

## for MainFunc in MainFunctions
{{MainFunc.ReturnType}} {{MainClassName}}_Generated::{{MainFunc.MainFuncDeclCmd}}
{
  m_currCmdBuffer = a_commandBuffer;
  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT }; 
  {% if MainFunc.IsMega %}
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{MainFunc.Name}}MegaLayout, 0, 1, &m_allGeneratedDS[{{MainFunc.DSId}}], 0, nullptr);
  {{MainFunc.MegaKernelCall}}
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr); 
  {% else %}
  {% if MainFunc.IsRTV %}
  {% if MainFunc.NeedThreadFlags %}
  const uint32_t outOfForFlags  = KGEN_FLAG_RETURN;
  const uint32_t inForFlags     = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK;
  {% if MainFunc.NeedToAddThreadFlags %}
  const uint32_t outOfForFlagsN = KGEN_FLAG_RETURN | KGEN_FLAG_SET_EXIT_NEGATIVE;
  const uint32_t inForFlagsN    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_SET_EXIT_NEGATIVE;
  const uint32_t outOfForFlagsD = KGEN_FLAG_RETURN | KGEN_FLAG_DONT_SET_EXIT;
  const uint32_t inForFlagsD    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_DONT_SET_EXIT;
  vkCmdFillBuffer(a_commandBuffer, {{MainFunc.Name}}_local.threadFlagsBuffer, 0, VK_WHOLE_SIZE, 0); // zero thread flags, mark all threads to be active
  VkBufferMemoryBarrier fillBarrier = BarrierForClearFlags({{MainFunc.Name}}_local.threadFlagsBuffer); 
  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &fillBarrier, 0, nullptr); 
  {% endif %}
  {% endif %}
  {% endif %}
  {{MainFunc.MainFuncTextCmd}}
  {% endif %} {# /* end of else branch */ #}
}
## endfor

