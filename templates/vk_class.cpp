#include <vector>
#include <memory>
#include <limits>
#include <cassert>

#include "vulkan_basics.h"
#include "{{IncludeClassDecl}}"
#include "include/{{UBOIncl}}"

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

void {{MainClassName}}_Generated::UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
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

void {{MainClassName}}_Generated::UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
{
## for Var in ClassVectorVars
  if({{Var.Name}}.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.{{Var.Name}}Buffer, 0, {{Var.Name}}.data(), {{Var.Name}}.size()*sizeof({{Var.TypeOfData}}) );
## endfor
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
  
  pcData.m_sizeX  = {{Kernel.tidX}};
  pcData.m_sizeY  = {{Kernel.tidY}};
  pcData.m_sizeZ  = {{Kernel.tidZ}};
  pcData.m_tFlags = m_currThreadFlags;
  {% for Arg in Kernel.AuxArgs %}
  pcData.m_{{Arg.Name}} = {{Arg.Name}}; 
  {% endfor %}

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
  vkCmdDispatch       (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (pcData.m_sizeZ + blockSizeZ - 1) / blockSizeZ); 
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &objCounterBar, 0, nullptr);

  // (3) small prefix summ to compute global offsets for each type region
  //
  vkCmdBindPipeline   (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}CountTypeIntervals);
  vkCmdDispatch       (m_currCmdBuffer, 1, 1, 1); 
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &objCounterBar, 0, nullptr);

  // (4) execute maker second time, count offset for each object using local prefix summ (in the work group) and put ObjPtr at this offset 
  //
  vkCmdBindPipeline   (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Sorter);
  vkCmdDispatch       (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (pcData.m_sizeZ + blockSizeZ - 1) / blockSizeZ); 

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
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (pcData.m_sizeZ + blockSizeZ - 1) / blockSizeZ);
  {% if Kernel.FinishRed %}
  
  {% if Kernel.HasLoopFinish %}
  KernelArgsPC oldPCData = pcData;
  {% endif %}
  {% include "inc_reduction_vulkan.cpp" %}
  {% endif %} {# /* Kernel.FinishRed      */ #}
  {% endif %} {# /* NOT INDIRECT DISPATCH */ #}
  {# /* --------------------------------------------------------------------------------------------------------------------------------------- */ #}

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
  {% if Kernel.HasLoopFinish %}

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
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
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

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
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
{{MainFunc.MainFuncCmd}}

## endfor

