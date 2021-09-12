#include "generated_userfix.h"

void nBody_GeneratedFix::performCmd(VkCommandBuffer a_commandBuffer, BodyState *out_bodies)
{
  static bool needInit = true;
  m_currCmdBuffer = a_commandBuffer;
  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };

  if(needInit)
  {
    vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, GenerateBodiesLayout, 0, 1, &m_allGeneratedDS[0], 0, nullptr);
    GenerateBodiesCmd(BODIES_COUNT);
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    needInit = false;
  }

  for (uint32_t i = 0; i < m_iters; ++i) {
    vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, UpdateVelocityLayout, 0, 1, &m_allGeneratedDS[0], 0, nullptr);
    UpdateVelocityCmd(BODIES_COUNT);
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, UpdatePositionLayout, 0, 1, &m_allGeneratedDS[0], 0, nullptr);
    UpdatePositionCmd(BODIES_COUNT);
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
  }
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, copyKernelFloatLayout, 0, 1, &m_allGeneratedDS[1], 0, nullptr);
  copyKernelFloatCmd(m_bodies.size()*sizeof(BodyState) / sizeof(float));
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}



