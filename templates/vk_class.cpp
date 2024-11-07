#include <vector>
#include <memory>
#include <limits>
#include <cassert>
#include <chrono>
#include <array>

#include "vk_copy.h"
#include "vk_context.h"
#include "vk_images.h"

#include "{{IncludeClassDecl}}"
#include "include/{{UBOIncl}}"
{% if UseRayGen %}
#include "VulkanRTX.h"
{% endif%}

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

{% if IsRTV %}
constexpr uint32_t KGEN_FLAG_RETURN            = 1;
constexpr uint32_t KGEN_FLAG_BREAK             = 2;
constexpr uint32_t KGEN_FLAG_DONT_SET_EXIT     = 4;
constexpr uint32_t KGEN_FLAG_SET_EXIT_NEGATIVE = 8;
{% endif %}
constexpr uint32_t KGEN_REDUCTION_LAST_STEP    = 16;


void {{MainClassName}}{{MainClassSuffix}}::UpdatePlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
{
  const size_t maxAllowedSize = std::numeric_limits<uint32_t>::max();
  {% if HasPrefixData %}
  auto pUnderlyingImpl = dynamic_cast<{{PrefixDataClass}}*>({{PrefixDataName}}->UnderlyingImpl(0));
  {% endif %}
## for Var in ClassVars
  {% if Var.IsArray %}
  {% if Var.HasPrefix %}
  memcpy(m_uboData.{{Var.Name}}, pUnderlyingImpl->{{Var.CleanName}},sizeof(m_uboData.{{Var.Name}}));
  {% else %}
  memcpy(m_uboData.{{Var.Name}},{{Var.Name}},sizeof({{Var.Name}}));
  {% endif %}
  {% else %}
  {% if Var.HasPrefix %}
  m_uboData.{{Var.Name}} = pUnderlyingImpl->{{Var.CleanName}};
  {% else %}
  m_uboData.{{Var.Name}} = {{Var.Name}};
  {% endif %}
  {% endif %}
## endfor
## for Var in ClassVectorVars
  m_uboData.{{Var.Name}}_size     = uint32_t( {{Var.Name}}{{Var.AccessSymb}}size() );     assert( {{Var.Name}}{{Var.AccessSymb}}size() < maxAllowedSize );
  m_uboData.{{Var.Name}}_capacity = uint32_t( {{Var.Name}}{{Var.AccessSymb}}capacity() ); assert( {{Var.Name}}{{Var.AccessSymb}}capacity() < maxAllowedSize );
## endfor
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 0, &m_uboData, sizeof(m_uboData));
}

{% if HasFullImpl %}
void {{MainClassName}}{{MainClassSuffix}}::ReadPlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
{
  a_pCopyEngine->ReadBuffer(m_classDataBuffer, 0, &m_uboData, sizeof(m_uboData));
  {% if HasPrefixData %}
  auto pUnderlyingImpl = dynamic_cast<{{PrefixDataClass}}*>({{PrefixDataName}}->UnderlyingImpl(0));
  {% endif %}
  {% for Var in ClassVars %}
  {% if not Var.IsConst %}
  {% if Var.IsArray %}
  {% if Var.HasPrefix %}
  memcpy(pUnderlyingImpl->{{Var.CleanName}}, m_uboData.{{Var.Name}}, sizeof(m_uboData.{{Var.Name}});
  {% else %}
  memcpy({{Var.Name}}, m_uboData.{{Var.Name}}, sizeof({{Var.Name}}));
  {% endif %}
  {% else %}
  {% if Var.HasPrefix %}
  pUnderlyingImpl->{{Var.CleanName}} = m_uboData.{{Var.Name}};
  {% else %}
  {{Var.Name}} = m_uboData.{{Var.Name}};
  {% endif %}
  {% endif %}
  {% endif %} {#/* end of if not var.IsConst */#}
  {% endfor %}
  {% for Var in ClassVectorVars %}
  {{Var.Name}}{{Var.AccessSymb}}resize(m_uboData.{{Var.Name}}_size);
  {% endfor %}
}
{% endif %}

void {{MainClassName}}{{MainClassSuffix}}::UpdateVectorMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
{
  {% for Table in RemapTables %}
  {
    auto pProxyObj = dynamic_cast<RTX_Proxy*>({{Table.AccelName}}.get());
    auto tablePtrs = pProxyObj->GetAABBToPrimTable();
    if(tablePtrs.tableSize != 0)
      a_pCopyEngine->UpdateBuffer(m_vdata.{{Table.Name}}RemapTableBuffer, 0, tablePtrs.table, tablePtrs.tableSize*sizeof(LiteMath::uint2));
  }
  {% endfor %}
  {% for Var in ClassVectorVars %}
  {% if Var.IsVFHBuffer and Var.VFHLevel >= 2 %}
  if({{Var.Name}}{{Var.AccessSymb}}size() > 0)
  {
    a_pCopyEngine->UpdateBuffer(m_vdata.{{Var.Name}}Buffer      , 0, {{Var.Name}}_vtable.data(), {{Var.Name}}_vtable.size()*sizeof(unsigned)*2 );
    a_pCopyEngine->UpdateBuffer(m_vdata.{{Var.Name}}_dataVBuffer, 0, {{Var.Name}}_dataV.data(), {{Var.Name}}_dataV.size());
    for(size_t i=0;i<{{Var.Name}}_obj_storage_offsets.size()-1;i++) {
      size_t     offset = {{Var.Name}}_obj_storage_offsets[i];
      const auto& odata = {{Var.Name}}_sorted[i];
      if(odata.size() != 0)
        a_pCopyEngine->UpdateBuffer(m_vdata.{{Var.Name}}_dataSBuffer, offset, odata.data(), odata.size());
    }
  }
  {% else %}
  if({{Var.Name}}{{Var.AccessSymb}}size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.{{Var.Name}}Buffer, 0, {{Var.Name}}{{Var.AccessSymb}}data(), {{Var.Name}}{{Var.AccessSymb}}size()*sizeof({{Var.TypeOfData}}) );
  {% endif %}
  {% endfor %}
}

{% for UpdateFun in UpdateVectorFun %}
{% if UpdateFun.NumParams == 2 %}
void {{MainClassName}}{{MainClassSuffix}}::{{UpdateFun.Name}}(size_t a_first, size_t a_size)
{
  if({{UpdateFun.VectorName}}.size() != 0 && m_pLastCopyHelper != nullptr)
    m_pLastCopyHelper->UpdateBuffer(m_vdata.{{UpdateFun.VectorName}}Buffer, a_first*sizeof({{UpdateFun.TypeOfData}}), {{UpdateFun.VectorName}}.data() + a_first, a_size*sizeof({{UpdateFun.TypeOfData}}) );
}
{%else%}
void {{MainClassName}}{{MainClassSuffix}}::{{UpdateFun.Name}}()
{
  if({{UpdateFun.VectorName}}.size() != 0 && m_pLastCopyHelper != nullptr)
    m_pLastCopyHelper->UpdateBuffer(m_vdata.{{UpdateFun.VectorName}}Buffer, 0, {{UpdateFun.VectorName}}.data(), {{UpdateFun.VectorName}}.size()*sizeof({{UpdateFun.TypeOfData}}) );
}
{%endif%}
{% endfor %}

void {{MainClassName}}{{MainClassSuffix}}::UpdateTextureMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)
{
  {% if length(ClassTextureVars) > 0 or length(ClassTexArrayVars) > 0 %}
  {% for Var in ClassTextureVars %}
  {% if Var.NeedUpdate %}
  a_pCopyEngine->UpdateImage(m_vdata.{{Var.Name}}Texture, {{Var.Name}}{{Var.AccessSymb}}data(), {{Var.Name}}{{Var.AccessSymb}}width(), {{Var.Name}}{{Var.AccessSymb}}height(), {{Var.Name}}{{Var.AccessSymb}}bpp(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  {% endif %}
  {% endfor %}
  {% for Var in ClassTexArrayVars %}
  for(int i=0;i<m_vdata.{{Var.Name}}ArrayTexture.size();i++)
    a_pCopyEngine->UpdateImage(m_vdata.{{Var.Name}}ArrayTexture[i], {{Var.Name}}[i]->data(), {{Var.Name}}[i]->width(), {{Var.Name}}[i]->height(), {{Var.Name}}[i]->bpp(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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
    throw std::runtime_error("{{MainClassName}}{{MainClassSuffix}}::UpdateTextureMembers: failed to begin command buffer!");
  vkCmdPipelineBarrier(cmdBuff,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,uint32_t(barriers.size()),barriers.data());
  vkEndCommandBuffer(cmdBuff);

  vk_utils::executeCommandBufferNow(cmdBuff, transferQueue, device);
  {% endif %}
}

## for Kernel in Kernels
{% if Kernel.IsIndirect %}
void {{MainClassName}}{{MainClassSuffix}}::{{Kernel.Name}}_UpdateIndirect()
{
  VkBufferMemoryBarrier barIndirect = BarrierForIndirectBufferUpdate(m_indirectBuffer);
  vkCmdBindDescriptorSets(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectUpdateLayout, 0, 1, &m_indirectUpdateDS, 0, nullptr);
  vkCmdBindPipeline      (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectUpdate{{Kernel.Name}}Pipeline);
  vkCmdDispatch          (m_currCmdBuffer, 1, 1, 1);
  vkCmdPipelineBarrier   (m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, 1, &barIndirect, 0, nullptr);
}

{% endif %}
void {{MainClassName}}{{MainClassSuffix}}::{{Kernel.Decl}}
{
  {% if not Kernel.UseRayGen %}
  uint32_t blockSizeX = {{Kernel.WGSizeX}};
  uint32_t blockSizeY = {{Kernel.WGSizeY}};
  uint32_t blockSizeZ = {{Kernel.WGSizeZ}};

  {% endif %}
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
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, {% if Kernel.UseRayGen %}VK_SHADER_STAGE_RAYGEN_BIT_KHR{% else %}VK_SHADER_STAGE_COMPUTE_BIT{% endif %}, 0, 0, nullptr, 1, &barUBO2, 0, nullptr);
  }
  {% else %}
  vkCmdPushConstants(m_currCmdBuffer, {{Kernel.Name}}Layout, {% if Kernel.UseRayGen %}VK_SHADER_STAGE_RAYGEN_BIT_KHR{% else %}VK_SHADER_STAGE_COMPUTE_BIT{% endif %}, 0, sizeof(KernelArgsPC), &pcData);
  {% endif %}
  {% if Kernel.HasLoopInit %}
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}InitPipeline);
  vkCmdDispatch(m_currCmdBuffer, 1, 1, 1);
  VkBufferMemoryBarrier barUBO = BarrierForSingleBuffer(m_classDataBuffer);
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO, 0, nullptr);
  {% endif %}
  {# /* --------------------------------------------------------------------------------------------------------------------------------------- */ #}
  
  {% if Kernel.IsIndirect %}
  vkCmdBindPipeline    (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Pipeline);
  vkCmdDispatchIndirect(m_currCmdBuffer, m_indirectBuffer, {{Kernel.IndirectOffset}}*sizeof(uint32_t)*4);
  {% else %}
  vkCmdBindPipeline(m_currCmdBuffer, {% if Kernel.UseRayGen %}VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR{% else %}VK_PIPELINE_BIND_POINT_COMPUTE{% endif %}, {{Kernel.Name}}Pipeline);
  {% if Kernel.UseRayGen %}
  const auto* strides = {{Kernel.Name}}SBTStrides.data();
  vkCmdTraceRaysKHR(m_currCmdBuffer, &strides[0],&strides[1],&strides[2],&strides[3], sizeX,sizeY,sizeZ);
  {% else if Kernel.EnableBlockExpansion %}
  vkCmdDispatch    (m_currCmdBuffer, pcData.m_sizeX, pcData.m_sizeY, pcData.m_sizeZ);
  {% else %}
  vkCmdDispatch    (m_currCmdBuffer, (sizeX + blockSizeX - 1) / blockSizeX, (sizeY + blockSizeY - 1) / blockSizeY, (sizeZ + blockSizeZ - 1) / blockSizeZ);
  {% endif %}
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

void {{MainClassName}}{{MainClassSuffix}}::copyKernelFloatCmd(uint32_t length)
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

void {{MainClassName}}{{MainClassSuffix}}::matMulTransposeCmd(uint32_t A_offset, uint32_t B_offset, uint32_t C_offset, uint32_t A_col_len, uint32_t B_col_len, uint32_t A_row_len)
{
  const uint32_t blockSizeX = 8;
  const uint32_t blockSizeY = 8;

  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, matMulTransposePipeline);
  struct KernelArgsPC
  {
    uint32_t m_A_row_len;
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_A_offset;
    uint32_t m_B_offset;
    uint32_t m_C_offset;
  } pcData;
  pcData.m_A_row_len = A_row_len;
  pcData.m_sizeX = B_col_len;
  pcData.m_sizeY = A_col_len;
  pcData.m_A_offset = A_offset;
  pcData.m_B_offset = B_offset;
  pcData.m_C_offset = C_offset;
  {% if UseSeparateUBO %}
  {
    vkCmdUpdateBuffer(m_currCmdBuffer, m_uboArgsBuffer, 0, sizeof(pcData), &pcData);
    VkBufferMemoryBarrier barUBO2 = BarrierForArgsUBO(sizeof(pcData));
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO2, 0, nullptr);
    vkCmdPushConstants(m_currCmdBuffer, matMulTransposeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcData), &pcData);
  }
  {% else %}
  vkCmdPushConstants(m_currCmdBuffer, matMulTransposeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcData), &pcData);
  {% endif %}
  vkCmdDispatch(m_currCmdBuffer, (B_col_len + blockSizeX - 1) / blockSizeX, (A_col_len + blockSizeY - 1) / blockSizeY, 1);
}

VkBufferMemoryBarrier {{MainClassName}}{{MainClassSuffix}}::BarrierForClearFlags(VkBuffer a_buffer)
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

VkBufferMemoryBarrier {{MainClassName}}{{MainClassSuffix}}::BarrierForSingleBuffer(VkBuffer a_buffer)
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

void {{MainClassName}}{{MainClassSuffix}}::BarriersForSeveralBuffers(VkBuffer* a_inBuffers, VkBufferMemoryBarrier* a_outBarriers, uint32_t a_buffersNum)
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
{{MainFunc.ReturnType}} {{MainClassName}}{{MainClassSuffix}}::{{MainFunc.MainFuncDeclCmd}}
{
  VkPipelineStageFlagBits prevStageBits = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  m_currCmdBuffer = a_commandBuffer;
  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  {% if MainFunc.IsMega %}
  vkCmdBindDescriptorSets(a_commandBuffer, {% if MainFunc.UseRayGen %}VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR{% else %}VK_PIPELINE_BIND_POINT_COMPUTE{% endif %}, {{MainFunc.Name}}MegaLayout, 0, 1, &m_allGeneratedDS[{{MainFunc.DSId}}], 0, nullptr);
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

## for MainFunc in MainFunctions
{% if MainFunc.OverrideMe %}

{{MainFunc.ReturnType}} {{MainClassName}}{{MainClassSuffix}}::{{MainFunc.DeclOrig}}
{
  // (1) get global Vulkan context objects
  //
  VkInstance       instance       = m_ctx.instance;
  VkPhysicalDevice physicalDevice = m_ctx.physicalDevice;
  VkDevice         device         = m_ctx.device;
  VkCommandPool    commandPool    = m_ctx.commandPool;
  VkQueue          computeQueue   = m_ctx.computeQueue;
  VkQueue          transferQueue  = m_ctx.transferQueue;
  auto             pCopyHelper    = m_ctx.pCopyHelper;
  auto             pAllocatorSpec = m_ctx.pAllocatorSpecial;

  // (2) create GPU objects
  //
  auto outFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if({{MainFunc.Name}}_local.needToClearOutput)
    outFlags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  std::vector<VkBuffer> buffers;
  std::vector<VkImage>  images2;
  std::vector<vk_utils::VulkanImageMem*> images;
  auto beforeCreateObjects = std::chrono::high_resolution_clock::now();
  {% for var in MainFunc.FullImpl.InputData %}
  {% if var.IsTexture %}
  auto {{var.Name}}Img = vk_utils::createImg(device, {{var.Name}}.width(), {{var.Name}}.height(), {{var.Format}}, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
  size_t {{var.Name}}ImgId = images.size();
  images.push_back(&{{var.Name}}Img);
  images2.push_back({{var.Name}}Img.image);
  {% else %}
  VkBuffer {{var.Name}}GPU = vk_utils::createBuffer(device, {{var.DataSize}}*sizeof({{var.DataType}}), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  buffers.push_back({{var.Name}}GPU);
  {% endif %}
  {% endfor %}
  {% for var in MainFunc.FullImpl.OutputData %}
  {% if var.IsTexture %}
  auto {{var.Name}}Img = vk_utils::createImg(device, {{var.Name}}.width(), {{var.Name}}.height(), {{var.Format}}, outFlags);
  size_t {{var.Name}}ImgId = images.size();
  images.push_back(&{{var.Name}}Img);
  images2.push_back({{var.Name}}Img.image);
  {% else %}
  VkBuffer {{var.Name}}GPU = vk_utils::createBuffer(device, {{var.DataSize}}*sizeof({{var.DataType}}), outFlags);
  buffers.push_back({{var.Name}}GPU);
  {% endif %}
  {% endfor %}

  {# /**
  VkBuffer tempBuffer = pAllocatorSpec->GetBufferBlock(0);
  size_t currOffset = 0;
  {% for var in MainFunc.FullImpl.InputData %}
  size_t {{var.Name}}Offset = currOffset;
  size_t {{var.Name}}Size   = {{var.DataSize}}*sizeof({{var.DataType}});
  currOffset += vk_utils::getPaddedSize({{var.Name}}Size, 1024);
  {% endfor %}
  {% for var in MainFunc.FullImpl.OutputData %}
  size_t {{var.Name}}Offset = currOffset;
  size_t {{var.Name}}Size   = {{var.DataSize}}*sizeof({{var.DataType}});
  currOffset += vk_utils::getPaddedSize({{var.Name}}Size, 1024);
  {% endfor %}
  **/ #}

  VkDeviceMemory buffersMem = VK_NULL_HANDLE; // vk_utils::allocateAndBindWithPadding(device, physicalDevice, buffers);
  VkDeviceMemory imagesMem  = VK_NULL_HANDLE; // vk_utils::allocateAndBindWithPadding(device, physicalDevice, std::vector<VkBuffer>(), images);

  vk_utils::MemAllocInfo tempMemoryAllocInfo;
  tempMemoryAllocInfo.memUsage = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; // TODO, select depending on device and sample/application (???)
  if(buffers.size() != 0)
    pAllocatorSpec->Allocate(tempMemoryAllocInfo, buffers);
  if(images.size() != 0)
  {
    pAllocatorSpec->Allocate(tempMemoryAllocInfo, images2);
    for(auto imgMem : images)
    {
      VkImageViewCreateInfo imageView{};
      imageView.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
      imageView.image    = imgMem->image;
      imageView.format   = imgMem->format;
      imageView.subresourceRange = {};
      imageView.subresourceRange.aspectMask     = imgMem->aspectMask;
      imageView.subresourceRange.baseMipLevel   = 0;
      imageView.subresourceRange.levelCount     = imgMem->mipLvls;
      imageView.subresourceRange.baseArrayLayer = 0;
      imageView.subresourceRange.layerCount     = 1;
      VK_CHECK_RESULT(vkCreateImageView(device, &imageView, nullptr, &imgMem->view));
    }
  }

  auto afterCreateObjects = std::chrono::high_resolution_clock::now();
  m_exTime{{MainFunc.Name}}.msAPIOverhead = std::chrono::duration_cast<std::chrono::microseconds>(afterCreateObjects - beforeCreateObjects).count()/1000.f;

  auto afterCopy2 = std::chrono::high_resolution_clock::now(); // just declare it here, replace value later

  auto afterInitBuffers = std::chrono::high_resolution_clock::now();
  m_exTime{{MainFunc.Name}}.msAPIOverhead += std::chrono::duration_cast<std::chrono::microseconds>(afterInitBuffers - afterCreateObjects).count()/1000.f;
  {% if MainFunc.FullImpl.HasImages %}
  { // move textures to initial (transfer for input and general for output) layouts
    auto imgCmdBuf = vk_utils::createCommandBuffer(device, commandPool);
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(imgCmdBuf, &beginInfo);
    {
      VkImageSubresourceRange subresourceRange = {};
      subresourceRange.levelCount = 1;
      subresourceRange.layerCount = 1;
      {% for var in MainFunc.FullImpl.InputData %}
      {% if var.IsTexture %}
      subresourceRange.aspectMask = {{var.Name}}Img.aspectMask;
      vk_utils::setImageLayout(imgCmdBuf, {{var.Name}}Img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);
      {% endif %}
      {% endfor %}
      {% for var in MainFunc.FullImpl.OutputData %}
      {% if var.IsTexture %}
      subresourceRange.aspectMask = {{var.Name}}Img.aspectMask;
      vk_utils::setImageLayout(imgCmdBuf, {{var.Name}}Img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
      {% endif %}
      {% endfor %}
    }
    vkEndCommandBuffer(imgCmdBuf);
    vk_utils::executeCommandBufferNow(imgCmdBuf, transferQueue, device);
    vkFreeCommandBuffers(device, commandPool, 1, &imgCmdBuf);
    auto afterLayoutChange = std::chrono::high_resolution_clock::now();
    m_exTime{{MainFunc.Name}}.msLayoutChange = std::chrono::duration_cast<std::chrono::microseconds>(afterLayoutChange - afterInitBuffers).count()/1000.f;
  }
  {% endif %}

  auto beforeSetInOut = std::chrono::high_resolution_clock::now();
  this->SetVulkanInOutFor_{{MainFunc.Name}}({{MainFunc.FullImpl.ArgsOnSetInOut}});

  // (3) copy input data to GPU
  //
  auto beforeCopy = std::chrono::high_resolution_clock::now();
  m_exTime{{MainFunc.Name}}.msAPIOverhead += std::chrono::duration_cast<std::chrono::microseconds>(beforeCopy - beforeSetInOut).count()/1000.f;
  {% for var in MainFunc.FullImpl.InputData %}
  {% if var.IsTexture %}
  pCopyHelper->UpdateImage({{var.Name}}Img.image, {{var.Name}}.data(), {{var.Name}}.width(), {{var.Name}}.height(), {{var.Name}}.bpp(), VK_IMAGE_LAYOUT_GENERAL);
  {% else %}
  pCopyHelper->UpdateBuffer({{var.Name}}GPU, 0, {{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}));
  {# /* pCopyHelper->UpdateBuffer(tempBuffer, {{var.Name}}Offset, {{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}})); */ #}
  {% endif %}
  {% endfor %}
  auto afterCopy = std::chrono::high_resolution_clock::now();
  m_exTime{{MainFunc.Name}}.msCopyToGPU = std::chrono::duration_cast<std::chrono::microseconds>(afterCopy - beforeCopy).count()/1000.f;
  //{% if MainFunc.FullImpl.HasImages %}
  //{ // move textures to normal layouts
  //  auto imgCmdBuf = vk_utils::createCommandBuffer(device, commandPool);
  //  VkCommandBufferBeginInfo beginInfo = {};
  //  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  //  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  //  vkBeginCommandBuffer(imgCmdBuf, &beginInfo);
  //  {
  //    VkImageSubresourceRange subresourceRange = {};
  //    subresourceRange.levelCount = 1;
  //    subresourceRange.layerCount = 1;
  //    {% for var in MainFunc.FullImpl.InputData %}
  //    {% if var.IsTexture %}
  //    subresourceRange.aspectMask = {{var.Name}}Img.aspectMask;
  //    vk_utils::setImageLayout(imgCmdBuf, {{var.Name}}Img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
  //    {% endif %}
  //    {% endfor %}
  //  }
  //  vkEndCommandBuffer(imgCmdBuf);
  //  vk_utils::executeCommandBufferNow(imgCmdBuf, transferQueue, device);
  //  vkFreeCommandBuffers(device, commandPool, 1, &imgCmdBuf);
  //  auto afterLayoutChange = std::chrono::high_resolution_clock::now();
  //  m_exTime{{MainFunc.Name}}.msLayoutChange += std::chrono::duration_cast<std::chrono::microseconds>(afterLayoutChange - afterCopy).count()/1000.f;
  //}
  //{% endif %}

  m_exTime{{MainFunc.Name}}.msExecuteOnGPU = 0;
  {% if MainFunc.IsRTV %}
  //// (3.1) clear all outputs if we are in RTV pattern
  //
  if({{MainFunc.Name}}_local.needToClearOutput)
  {
    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(device, commandPool);
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    {% if EnableTimeStamps %}
    vkCmdResetQueryPool(commandBuffer, m_queryPoolTimestamps, 0, m_timestampPoolSize);
    {% endif %}
    {% for var in MainFunc.FullImpl.OutputData %}
    {% if not var.IsTexture %}
    vkCmdFillBuffer(commandBuffer, {{var.Name}}GPU, 0, VK_WHOLE_SIZE, 0); // zero output buffer {{var.Name}}GPU
    {% endif %}
    {% endfor %}
    vkEndCommandBuffer(commandBuffer);
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    auto stop = std::chrono::high_resolution_clock::now();
    m_exTime{{MainFunc.Name}}.msExecuteOnGPU  += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
  }
  {% endif %}

  // (4) now execute algorithm on GPU
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(device, commandPool);
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    {% if EnableTimeStamps %}
    vkCmdResetQueryPool(commandBuffer, m_queryPoolTimestamps, 0, m_timestampPoolSize);
    {% endif %}
    {{MainFunc.Name}}Cmd(commandBuffer, {{MainFunc.FullImpl.ArgsOnCall}});
    vkEndCommandBuffer(commandBuffer);
    auto start = std::chrono::high_resolution_clock::now();
    {% if MainFunc.IsRTV %}
    {% if HasProgressBar %}
    if(a_numPasses > 1)
      ProgressBarStart();
    {% endif %}
    for(uint32_t pass = 0; pass < a_numPasses; pass++) {
      vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
      {% if HasProgressBar %}
      if((pass != 0) && (pass % 256 == 0))
        ProgressBarAccum(256.0f/float(a_numPasses));
      {% endif %}
    }
    {% if HasProgressBar %}
    if(a_numPasses > 1)
      ProgressBarDone();
    {% endif %}
    {% else %}
    vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
    {% endif %}
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    auto stop = std::chrono::high_resolution_clock::now();
    m_exTime{{MainFunc.Name}}.msExecuteOnGPU += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
  }

  // (5) copy output data to CPU
  //
  auto beforeCopy2 = std::chrono::high_resolution_clock::now();
  {% for var in MainFunc.FullImpl.OutputData %}
  {% if var.IsTexture %}
  //todo: change image layout before transfer?
  pCopyHelper->ReadImage({{var.Name}}Img.image, {{var.Name}}.data(), {{var.Name}}.width(), {{var.Name}}.height(), {{var.Name}}.bpp());
  {% else %}
  pCopyHelper->ReadBuffer({{var.Name}}GPU, 0, {{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}));
  {# /* pCopyHelper->ReadBuffer(tempBuffer, {{var.Name}}Offset, {{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}})); */ #}
  {% endif %}
  {% endfor %}
  this->ReadPlainMembers(pCopyHelper);
  afterCopy2 = std::chrono::high_resolution_clock::now();
  m_exTime{{MainFunc.Name}}.msCopyFromGPU = std::chrono::duration_cast<std::chrono::microseconds>(afterCopy2 - beforeCopy2).count()/1000.f;

  // (6) free resources
  //
  {% for var in MainFunc.FullImpl.InputData %}
  {% if var.IsTexture %}
  vkDestroyImageView(device, {{var.Name}}Img.view, nullptr);
  vkDestroyImage    (device, {{var.Name}}Img.image, nullptr);
  {% else %}
  vkDestroyBuffer(device, {{var.Name}}GPU, nullptr);
  {% endif %}
  {% endfor %}
  {% for var in MainFunc.FullImpl.OutputData %}
  {% if var.IsTexture %}
  vkDestroyImageView(device, {{var.Name}}Img.view, nullptr);
  vkDestroyImage    (device, {{var.Name}}Img.image, nullptr);
  {% else %}
  vkDestroyBuffer(device, {{var.Name}}GPU, nullptr);
  {% endif %}
  {% endfor %}
  if(buffersMem != VK_NULL_HANDLE)
    vkFreeMemory(device, buffersMem, nullptr);
  if(imagesMem != VK_NULL_HANDLE)
    vkFreeMemory(device, imagesMem, nullptr);

  m_exTime{{MainFunc.Name}}.msAPIOverhead += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - afterCopy2).count()/1000.f;
}
{% endif %}
## endfor
{% if HasGetTimeFunc %}

void {{MainClassName}}{{MainClassSuffix}}::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  vk_utils::ExecTime res = {};
  {% for MainFunc in MainFunctions %}
  {% if MainFunc.OverrideMe %}
  if(std::string(a_funcName) == "{{MainFunc.Name}}" || std::string(a_funcName) == "{{MainFunc.Name}}Block")
    res = m_exTime{{MainFunc.Name}};
  {% endif %}
  {% endfor %}
  a_out[0] = res.msExecuteOnGPU;
  a_out[1] = res.msCopyToGPU;
  a_out[2] = res.msCopyFromGPU;
  a_out[3] = res.msAPIOverhead;
}
{% endif %}