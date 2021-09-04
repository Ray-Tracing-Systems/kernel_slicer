#include "points_render.h"
//#include "../utils/input_definitions.h"

#include <chrono>

#include "../test_class_generated.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <ray_tracing/vk_rt_funcs.h>

void PointsRender::LoadScene(const char* path, bool transpose_inst_matrices)
{
  const uint32_t SEED       = 42;
  const uint32_t ITERATIONS = 1;

  m_pointsData.pointsCount = nBody::BODIES_COUNT;
  m_pointsData.outBodies.resize(nBody::BODIES_COUNT);

  m_pNBodySimGenerated->setParameters(SEED, ITERATIONS);
  m_pNBodySimGenerated->InitVulkanObjects(m_device, m_physicalDevice, nBody::BODIES_COUNT);
  m_pNBodySimGenerated->InitMemberBuffers();
  m_pNBodySimGenerated->UpdateAll(m_pCopy);

  m_pointsData.pointsBuf = vk_utils::createBuffer(m_device, nBody::BODIES_COUNT * sizeof(nBody::BodyState),
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  m_pointsData.pointsMem = vk_utils::allocateAndBindWithPadding(m_device, m_physicalDevice, {m_pointsData.pointsBuf});
  m_pNBodySimGenerated->SetVulkanInOutFor_perform(m_pointsData.pointsBuf, 0);

  SetupPointsPipeline();
  UpdateView();

  for (size_t i = 0; i < m_framesInFlight; ++i)
  {
    BuildCommandBufferPoints(m_cmdBuffersDrawMain[i], m_frameBuffers[i], m_pointsPipeline.pipeline);
  }
}

VkCommandBuffer PointsRender::BuildCommandBufferSimulation()
{
  VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(m_device, m_commandPoolCompute);

  VkCommandBufferBeginInfo beginCommandBufferInfo = {};
  beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);

  m_pNBodySimGenerated->performCmd(commandBuffer, m_pointsData.outBodies.data());
  vkEndCommandBuffer(commandBuffer);

  return commandBuffer;
}
