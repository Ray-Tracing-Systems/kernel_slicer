#include <vector>
#include <memory>
#include <limits>
#include <cassert>

#include "vulkan_basics.h"
#include "test_class_generated.h"
#include "include/TestClass_ubo.h"

VkBufferUsageFlags TestClass_Generated::GetAdditionalFlagsForUBO() const
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

void TestClass_Generated::UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
{
  const size_t maxAllowedSize = std::numeric_limits<uint32_t>::max();

  m_uboData.m_worldViewProjInv = m_worldViewProjInv;
  m_uboData.camPos = camPos;
  m_uboData.m_lightSphere = m_lightSphere;
  m_uboData.testColor = testColor;
  m_uboData.m_emissiveMaterialId = m_emissiveMaterialId;
  m_uboData.m_randomGens_size     = uint32_t( m_randomGens.size() );    assert( m_randomGens.size() < maxAllowedSize );
  m_uboData.m_randomGens_capacity = uint32_t( m_randomGens.capacity() ); assert( m_randomGens.capacity() < maxAllowedSize );
  m_uboData.m_vNorm4f_size     = uint32_t( m_vNorm4f.size() );    assert( m_vNorm4f.size() < maxAllowedSize );
  m_uboData.m_vNorm4f_capacity = uint32_t( m_vNorm4f.capacity() ); assert( m_vNorm4f.capacity() < maxAllowedSize );
  m_uboData.m_nodes_size     = uint32_t( m_nodes.size() );    assert( m_nodes.size() < maxAllowedSize );
  m_uboData.m_nodes_capacity = uint32_t( m_nodes.capacity() ); assert( m_nodes.capacity() < maxAllowedSize );
  m_uboData.m_intervals_size     = uint32_t( m_intervals.size() );    assert( m_intervals.size() < maxAllowedSize );
  m_uboData.m_intervals_capacity = uint32_t( m_intervals.capacity() ); assert( m_intervals.capacity() < maxAllowedSize );
  m_uboData.m_materialIds_size     = uint32_t( m_materialIds.size() );    assert( m_materialIds.size() < maxAllowedSize );
  m_uboData.m_materialIds_capacity = uint32_t( m_materialIds.capacity() ); assert( m_materialIds.capacity() < maxAllowedSize );
  m_uboData.m_indicesReordered_size     = uint32_t( m_indicesReordered.size() );    assert( m_indicesReordered.size() < maxAllowedSize );
  m_uboData.m_indicesReordered_capacity = uint32_t( m_indicesReordered.capacity() ); assert( m_indicesReordered.capacity() < maxAllowedSize );
  m_uboData.m_materialOffsets_size     = uint32_t( m_materialOffsets.size() );    assert( m_materialOffsets.size() < maxAllowedSize );
  m_uboData.m_materialOffsets_capacity = uint32_t( m_materialOffsets.capacity() ); assert( m_materialOffsets.capacity() < maxAllowedSize );
  m_uboData.m_vPos4f_size     = uint32_t( m_vPos4f.size() );    assert( m_vPos4f.size() < maxAllowedSize );
  m_uboData.m_vPos4f_capacity = uint32_t( m_vPos4f.capacity() ); assert( m_vPos4f.capacity() < maxAllowedSize );
  m_uboData.m_materialData_size     = uint32_t( m_materialData.size() );    assert( m_materialData.size() < maxAllowedSize );
  m_uboData.m_materialData_capacity = uint32_t( m_materialData.capacity() ); assert( m_materialData.capacity() < maxAllowedSize );

  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 0, &m_uboData, sizeof(m_uboData));
}

void TestClass_Generated::UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
{
  if(m_randomGens.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_randomGensBuffer, 0, m_randomGens.data(), m_randomGens.size()*sizeof(struct RandomGenT) );
  if(m_vNorm4f.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_vNorm4fBuffer, 0, m_vNorm4f.data(), m_vNorm4f.size()*sizeof(struct LiteMath::float4) );
  if(m_nodes.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_nodesBuffer, 0, m_nodes.data(), m_nodes.size()*sizeof(struct BVHNode) );
  if(m_intervals.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_intervalsBuffer, 0, m_intervals.data(), m_intervals.size()*sizeof(struct Interval) );
  if(m_materialIds.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_materialIdsBuffer, 0, m_materialIds.data(), m_materialIds.size()*sizeof(unsigned int) );
  if(m_indicesReordered.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_indicesReorderedBuffer, 0, m_indicesReordered.data(), m_indicesReordered.size()*sizeof(unsigned int) );
  if(m_materialOffsets.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_materialOffsetsBuffer, 0, m_materialOffsets.data(), m_materialOffsets.size()*sizeof(unsigned int) );
  if(m_vPos4f.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_vPos4fBuffer, 0, m_vPos4f.data(), m_vPos4f.size()*sizeof(struct LiteMath::float4) );
  if(m_materialData.size() > 0)
    a_pCopyEngine->UpdateBuffer(m_vdata.m_materialDataBuffer, 0, m_materialData.data(), m_materialData.size()*sizeof(unsigned int) );
}

void TestClass_Generated::GetColorCmd(uint tid, uint* out_color, const TestClass* a_pGlobals, uint tileOffset)
{
  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  pcData.m_sizeX  = tid;
  pcData.m_sizeY  = 1;
  pcData.m_sizeZ  = tileOffset;
  pcData.m_tFlags = m_currThreadFlags;

  vkCmdPushConstants(m_currCmdBuffer, GetColorLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, GetColorPipeline);
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (1 + blockSizeZ - 1) / blockSizeZ);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
}

void TestClass_Generated::NextBounceCmd(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                           const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                           float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen, 
                           float4* accumColor, float4* accumThoroughput, uint tileOffset)
{
  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  pcData.m_sizeX  = tid;
  pcData.m_sizeY  = 1;
  pcData.m_sizeZ  = tileOffset;
  pcData.m_tFlags = m_currThreadFlags;

  vkCmdPushConstants(m_currCmdBuffer, NextBounceLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, NextBouncePipeline);
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (1 + blockSizeZ - 1) / blockSizeZ);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

void TestClass_Generated::InitEyeRayCmd(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar, uint tileOffset)
{
  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  pcData.m_sizeX  = tid;
  pcData.m_sizeY  = 1;
  pcData.m_sizeZ  = tileOffset;
  pcData.m_tFlags = m_currThreadFlags;

  vkCmdPushConstants(m_currCmdBuffer, InitEyeRayLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, InitEyeRayPipeline);
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (1 + blockSizeZ - 1) / blockSizeZ);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
}

void TestClass_Generated::InitEyeRay2Cmd(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar,
                                         float4* accumColor, float4* accumuThoroughput, uint tileOffset)
{
  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  pcData.m_sizeX  = tid;
  pcData.m_sizeY  = 1;
  pcData.m_sizeZ  = tileOffset;
  pcData.m_tFlags = m_currThreadFlags;

  vkCmdPushConstants(m_currCmdBuffer, InitEyeRay2Layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, InitEyeRay2Pipeline);
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (1 + blockSizeZ - 1) / blockSizeZ);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

void TestClass_Generated::RayTraceCmd(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                Lite_Hit* out_hit, float2* out_bars, uint tileOffset)
{
  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  pcData.m_sizeX  = tid;
  pcData.m_sizeY  = 1;
  pcData.m_sizeZ  = tileOffset;
  pcData.m_tFlags = m_currThreadFlags;

  vkCmdPushConstants(m_currCmdBuffer, RayTraceLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, RayTracePipeline);
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (1 + blockSizeZ - 1) / blockSizeZ);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

void TestClass_Generated::MakeMaterialCmd(uint tid, const Lite_Hit* in_hit, uint tileOffset)
{
  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  pcData.m_sizeX  = tid;
  pcData.m_sizeY  = 1;
  pcData.m_sizeZ  = tileOffset;
  pcData.m_tFlags = m_currThreadFlags;

  vkCmdPushConstants(m_currCmdBuffer, MakeMaterialLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, MakeMaterialPipeline);
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (1 + blockSizeZ - 1) / blockSizeZ);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

void TestClass_Generated::PackXYCmd(uint tidX, uint tidY, uint* out_pakedXY)
{
  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  pcData.m_sizeX  = tidX;
  pcData.m_sizeY  = tidY;
  pcData.m_sizeZ  = 1;
  pcData.m_tFlags = m_currThreadFlags;

  vkCmdPushConstants(m_currCmdBuffer, PackXYLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, PackXYPipeline);
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (pcData.m_sizeZ + blockSizeZ - 1) / blockSizeZ);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
}

void TestClass_Generated::ContributeToImageCmd(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color,
                                               uint tileOffset)
{
  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
    uint32_t m_tFlags;
  } pcData;
  
  pcData.m_sizeX  = tid;
  pcData.m_sizeY  = 1;
  pcData.m_sizeZ  = tileOffset;
  pcData.m_tFlags = m_currThreadFlags;

  vkCmdPushConstants(m_currCmdBuffer, ContributeToImageLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
  
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ContributeToImagePipeline);
  vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (1 + blockSizeZ - 1) / blockSizeZ);

//  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
//  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}


void TestClass_Generated::copyKernelFloatCmd(uint32_t length)
{
  uint32_t blockSizeX = MEMCPY_BLOCK_SIZE;

  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, copyKernelFloatPipeline);
  vkCmdPushConstants(m_currCmdBuffer, copyKernelFloatLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &length);
  vkCmdDispatch(m_currCmdBuffer, (length + blockSizeX - 1) / blockSizeX, 1, 1);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

VkBufferMemoryBarrier TestClass_Generated::BarrierForClearFlags(VkBuffer a_buffer)
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

VkBufferMemoryBarrier TestClass_Generated::BarrierForSingleBuffer(VkBuffer a_buffer)
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

void TestClass_Generated::BarriersForSeveralBuffers(VkBuffer* a_inBuffers, VkBufferMemoryBarrier* a_outBarriers, uint32_t a_buffersNum)
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

void TestClass_Generated::NaivePathTraceCmd(VkCommandBuffer a_commandBuffer, uint tid, uint a_maxDepth, uint* in_pakedXY, float4* out_color,
                                            uint tileStart, uint tileEnd)
{
//  uint nThreads = tid;
//  uint tileStart2 = tileStart;
//  tileStart = 0;

  uint totalWork = tid;
  uint nThreads = tileEnd - tileStart;

  m_currCmdBuffer = a_commandBuffer;
  const uint32_t outOfForFlags  = KGEN_FLAG_RETURN;
  const uint32_t inForFlags     = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK;
  const uint32_t outOfForFlagsN = KGEN_FLAG_RETURN | KGEN_FLAG_SET_EXIT_NEGATIVE;
  const uint32_t inForFlagsN    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_SET_EXIT_NEGATIVE;
  const uint32_t outOfForFlagsD = KGEN_FLAG_RETURN | KGEN_FLAG_DONT_SET_EXIT;
  const uint32_t inForFlagsD    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_DONT_SET_EXIT;
  vkCmdFillBuffer(a_commandBuffer, NaivePathTrace_local.threadFlagsBuffer, 0, VK_WHOLE_SIZE, 0); // zero thread flags, mark all threads to be active
  VkBufferMemoryBarrier fillBarrier = BarrierForClearFlags(NaivePathTrace_local.threadFlagsBuffer);
  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &fillBarrier, 0, nullptr);

  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, InitEyeRay2Layout, 0, 1, &m_allGeneratedDS[0], 0, nullptr);
  m_currThreadFlags = outOfForFlags;
  InitEyeRay2Cmd(nThreads, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, tileStart);

  Lite_Hit hit;
  float2   baricentrics;

  for(int depth = 0; depth < a_maxDepth; depth++)
  {
    Lite_Hit hit;
    vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, RayTraceLayout, 0, 1, &m_allGeneratedDS[1], 0, nullptr);
    m_currThreadFlags = inForFlagsN;
    RayTraceCmd(nThreads, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics, tileStart);
//
    IMaterial* pMaterial = nullptr;
    vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, MakeMaterialLayout, 0, 1, &m_allGeneratedDS[2], 0, nullptr);
    m_currThreadFlags = outOfForFlags;
    MakeMaterialCmd(nThreads, &hit, tileStart);
////
    vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, NextBounceLayout, 0, 1, &m_allGeneratedDS[3], 0, nullptr);
    m_currThreadFlags = inForFlags;
    NextBounceCmd(nThreads, &hit, &baricentrics,
                                 m_indicesReordered.data(), m_vPos4f.data(), m_vNorm4f.data(),
                                 &rayPosAndNear, &rayDirAndFar, m_randomGens.data(),
                                 &accumColor, &accumThoroughput, tileStart);
  }

  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ContributeToImageLayout, 0, 1, &m_allGeneratedDS[4], 0, nullptr);
  m_currThreadFlags = outOfForFlags;
  ContributeToImageCmd(nThreads, &accumColor, in_pakedXY,
                           out_color, tileStart);
}

void TestClass_Generated::CastSingleRayCmd(VkCommandBuffer a_commandBuffer, uint tid, uint* in_pakedXY, uint* out_color,
                                           uint tileStart, uint tileEnd)
{
  uint totalWork = tid;
  uint nThreads = tileEnd - tileStart;

  m_currCmdBuffer = a_commandBuffer;
  const uint32_t outOfForFlags  = KGEN_FLAG_RETURN;
  const uint32_t inForFlags     = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK;
  const uint32_t outOfForFlagsN = KGEN_FLAG_RETURN | KGEN_FLAG_SET_EXIT_NEGATIVE;
  const uint32_t inForFlagsN    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_SET_EXIT_NEGATIVE;
  const uint32_t outOfForFlagsD = KGEN_FLAG_RETURN | KGEN_FLAG_DONT_SET_EXIT;
  const uint32_t inForFlagsD    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_DONT_SET_EXIT;
  vkCmdFillBuffer(a_commandBuffer, CastSingleRay_local.threadFlagsBuffer, 0, VK_WHOLE_SIZE, 0); // zero thread flags, mark all threads to be active
  VkBufferMemoryBarrier fillBarrier = BarrierForClearFlags(CastSingleRay_local.threadFlagsBuffer); 
  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &fillBarrier, 0, nullptr); 

  float4 rayPosAndNear, rayDirAndFar;
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, InitEyeRayLayout, 0, 1, &m_allGeneratedDS[5], 0, nullptr);
  m_currThreadFlags = outOfForFlags;
  InitEyeRayCmd(nThreads, in_pakedXY, &rayPosAndNear, &rayDirAndFar, tileStart);

  Lite_Hit hit; 
  float2   baricentrics; 
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, RayTraceLayout, 0, 1, &m_allGeneratedDS[6], 0, nullptr);
  m_currThreadFlags = outOfForFlagsN;
  RayTraceCmd(nThreads, nullptr, nullptr, &hit, &baricentrics, tileStart);

  IMaterial* pMaterial = nullptr;
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, MakeMaterialLayout, 0, 1, &m_allGeneratedDS[7], 0, nullptr);
  m_currThreadFlags = outOfForFlags;
  MakeMaterialCmd(nThreads, &hit, tileStart);

  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, GetColorLayout, 0, 1, &m_allGeneratedDS[8], 0, nullptr);
  m_currThreadFlags = outOfForFlags;
  GetColorCmd(nThreads, out_color, this, tileStart);
}

void TestClass_Generated::PackXYCmd(VkCommandBuffer a_commandBuffer, uint tidX, uint tidY, uint* out_pakedXY)
{
  m_currCmdBuffer = a_commandBuffer;
  const uint32_t outOfForFlags  = KGEN_FLAG_RETURN;
  const uint32_t inForFlags     = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK;

  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, PackXYLayout, 0, 1, &m_allGeneratedDS[9], 0, nullptr);
  m_currThreadFlags = outOfForFlags;
  PackXYCmd(tidX, tidY, out_pakedXY);
}



