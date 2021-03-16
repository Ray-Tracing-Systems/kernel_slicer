#include <vector>
#include <memory>
#include <limits>
#include <cassert>

#include "vulkan_basics.h"
#include "{{IncludeClassDecl}}"
#include "include/{{UBOIncl}}"

VkBufferUsageFlags {{MainClassName}}_Generated::GetAdditionalFlagsForUBO()
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
void {{MainClassName}}_Generated::{{Kernel.Decl}}
{
  uint32_t blockSizeX = m_blockSize[0];
  uint32_t blockSizeY = m_blockSize[1];
  uint32_t blockSizeZ = m_blockSize[2];

  auto ex = m_kernelExceptions.find("{{Kernel.OriginalName}}");
  if(ex != m_kernelExceptions.end())
  {
    blockSizeX = ex->second.blockSize[0];
    blockSizeY = ex->second.blockSize[1];
    blockSizeZ = ex->second.blockSize[2];
  }

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

  vkCmdPushConstants(m_currCmdBuffer, {{Kernel.Name}}Layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  {% if Kernel.HasLoopInit %}
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}InitPipeline);
  vkCmdDispatch(m_currCmdBuffer, 1, 1, 1); 
  VkBufferMemoryBarrier barUBO = BarrierForSingleBuffer(m_classDataBuffer);
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO, 0, nullptr);
  {% endif %}
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Pipeline);
  vkCmdDispatch(m_currCmdBuffer, ({{Kernel.tidX}} + blockSizeX - 1) / blockSizeX, ({{Kernel.tidY}} + blockSizeY - 1) / blockSizeY, ({{Kernel.tidZ}} + blockSizeZ - 1) / blockSizeZ);

  {% if Kernel.FinishRed %}
  ///// complete kernel with reduction passes
  {
    VkBufferMemoryBarrier redBars   [{{Kernel.RedVarsFPNum}}]; 
    VkBuffer              redBuffers[{{Kernel.RedVarsFPNum}}+1] = { {% for RedVarName in Kernel.RedVarsFPArr %}m_vdata.{{RedVarName.Name}}Buffer, {% endfor %} VK_NULL_HANDLE};
    size_t                szOfElems[{{Kernel.RedVarsFPNum}}+1] = { {% for RedVarName in Kernel.RedVarsFPArr %}sizeof({{RedVarName.Type}}), {% endfor %} 0};
    BarriersForSeveralBuffers(redBuffers, redBars, {{Kernel.RedVarsFPNum}});
    
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, {{Kernel.RedVarsFPNum}}, redBars, 0, nullptr);
    vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}ReductionPipeline);
    
    uint32_t oldSize    = {{Kernel.tidX}};
    uint32_t wholeSize  = (oldSize + blockSizeX - 1) / blockSizeX; // assume first pass of reduction is done inside kernel itself
    uint32_t wgSize     = REDUCTION_BLOCK_SIZE;
    uint32_t currOffset = 0;
    while (wholeSize > 1)
    {
      uint32_t nextSize = (wholeSize + wgSize - 1) / wgSize;
      pcData.m_sizeX  = oldSize;                // put current size here
      pcData.m_sizeY  = currOffset;             // put input offset here
      pcData.m_sizeZ  = currOffset + wholeSize; // put output offet here
      pcData.m_tFlags = m_currThreadFlags;      // now flags:
      if(wholeSize <= wgSize)                   // stop if last pass
        pcData.m_tFlags |= KGEN_REDUCTION_LAST_STEP;
        
      vkCmdPushConstants(m_currCmdBuffer, {{Kernel.Name}}Layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
      vkCmdDispatch(m_currCmdBuffer, nextSize, 1, 1);
      
      if(wholeSize <= wgSize)
        vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO, 0, nullptr);
      else
      {
        uint32_t arrSize = sizeof(redBars)/sizeof(redBars[0]);
        for(int barId=0;barId<arrSize;barId++)
        {
          redBars[barId].offset = pcData.m_sizeZ*szOfElems[barId]; // put output offset here (for barrier)
          redBars[barId].size   = nextSize;                        // put output data size (for barrier)
        }
        vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, {{Kernel.RedVarsFPNum}}, redBars, 0, nullptr);
      } 

      currOffset += wholeSize;
      oldSize    =  wholeSize;
      wholeSize  =  nextSize;
    }
  }

  {% endif %}
  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
  {% if Kernel.HasLoopFinish %}
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

  vkCmdPushConstants(m_currCmdBuffer, copyKernelFloatLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &length);
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

