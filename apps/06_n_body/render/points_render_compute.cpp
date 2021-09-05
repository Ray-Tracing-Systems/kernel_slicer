#include "points_render.h"
//#include "../utils/input_definitions.h"

#include <chrono>

#include "../test_class_generated.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

unsigned char* loadImage(const std::string &a_filename, int &w, int &h, int &channels)
{
  unsigned char* pixels = stbi_load(a_filename.c_str(), &w, &h, &channels, STBI_rgb_alpha);

  if(w <= 0 || h <= 0 || !pixels)
  {
    return nullptr;
  }
  return pixels;
}

void PointsRender::LoadScene(const char *path, bool transpose_inst_matrices)
{
  const uint32_t SEED = 42;
  const uint32_t ITERATIONS = 1;

  m_pointsData.pointsCount = nBody::BODIES_COUNT;
  m_pointsData.outBodies.resize(nBody::BODIES_COUNT);

  m_pNBodySimGenerated->setParameters(SEED, ITERATIONS);
  m_pNBodySimGenerated->InitVulkanObjects(m_device, m_physicalDevice, nBody::BODIES_COUNT);
  m_pNBodySimGenerated->InitMemberBuffers();
  m_pNBodySimGenerated->UpdateAll(m_pCopy);

  m_pointsData.pointsBuf = vk_utils::createBuffer(m_device, nBody::BODIES_COUNT * sizeof(nBody::BodyState),
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  m_pointsData.pointsMem = vk_utils::allocateAndBindWithPadding(m_device, m_physicalDevice, {m_pointsData.pointsBuf});
  m_pNBodySimGenerated->SetVulkanInOutFor_perform(m_pointsData.pointsBuf, 0);

  if(DISPLAY_MODE == RENDER_MODE::SPRITES)
  {
    int w, h, channels;
    unsigned char *pixels = loadImage(SPRITE_TEXTURE_PATH, w, h, channels);
    if(!pixels)
    {
      vk_utils::logWarning("[PointsRender::LoadScene] Failed to load sprite texture!");
    }
    else
    {
      auto mipLevels = 1;//vk_utils::calcMipLevelsCount(w, h);
      m_sprite = vk_utils::allocateColorTextureFromDataLDR(m_device, m_physicalDevice, pixels, w, h, mipLevels,
                                                           VK_FORMAT_R8G8B8A8_UNORM, m_pCopy);

      m_spriteSampler = vk_utils::createSampler(m_device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
      auto imgCmdBuf = vk_utils::createCommandBuffer(m_device, m_commandPoolGraphics);

      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      vkBeginCommandBuffer(imgCmdBuf, &beginInfo);
      {
        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = m_sprite.aspectMask;
        subresourceRange.levelCount = mipLevels;
        subresourceRange.layerCount = 1;
        vk_utils::setImageLayout(
            imgCmdBuf,
            m_sprite.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresourceRange);
      }
      vkEndCommandBuffer(imgCmdBuf);
      vk_utils::executeCommandBufferNow(imgCmdBuf, m_graphicsQueue, m_device);
//      recordMipChainGenerationCmdBuf(m_device, mipCmdBuf, m_sprite, w, h, mipLevels,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      stbi_image_free(pixels);
    }
    SetupSpritesPipeline();
  }
  else
  {
    SetupPointsPipeline();
  }
  UpdateView();

  for (size_t i = 0; i < m_framesInFlight; ++i)
  {
    BuildDrawCommandBuffer(m_cmdBuffersDrawMain[i], m_frameBuffers[i], m_pointsPipeline.pipeline);
  }
}

VkCommandBuffer PointsRender::BuildCommandBufferSimulation()
{
  VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(m_device, m_commandPoolCompute);

  VkCommandBufferBeginInfo beginCommandBufferInfo = {};
  beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);

  m_pNBodySimGenerated->performCmd(commandBuffer, nullptr);
  vkEndCommandBuffer(commandBuffer);

  return commandBuffer;
}
