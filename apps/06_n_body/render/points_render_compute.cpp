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

void PointsRender::CreateColormapTexture()
{
  std::vector<uchar4> infernoColorMap = {uchar4(20, 6, 42, 1),
                                         uchar4(40, 11, 84, 1),
                                         uchar4(101, 21, 110, 1),
                                         uchar4(159, 42, 99, 1),
                                         uchar4(212, 72, 66, 1),
                                         uchar4(245, 125, 21, 1),
                                         uchar4(250, 193, 39, 1),
                                         uchar4(252, 255, 164, 1)};


  m_colormap.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  m_colormap.format     = VK_FORMAT_R8G8B8A8_UNORM;

  VkImageCreateInfo image{};
  image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image.imageType = VK_IMAGE_TYPE_1D;
  image.format = m_colormap.format;
  image.extent.width = infernoColorMap.size();
  image.extent.height = 1;
  image.extent.depth = 1;
  image.mipLevels = 1;
  image.arrayLayers = 1;
  image.samples = VK_SAMPLE_COUNT_1_BIT;
  image.tiling = VK_IMAGE_TILING_OPTIMAL;
  image.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  VK_CHECK_RESULT(vkCreateImage(m_device, &image, nullptr, &m_colormap.image));
  vkGetImageMemoryRequirements(m_device, m_colormap.image, &m_colormap.memReq);

  VkMemoryAllocateInfo memAlloc{};
  memAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memAlloc.allocationSize  = m_colormap.memReq.size;
  memAlloc.memoryTypeIndex = vk_utils::findMemoryType(m_colormap.memReq.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_physicalDevice);
  VK_CHECK_RESULT(vkAllocateMemory(m_device, &memAlloc, nullptr, &m_colormap.mem));

  VkImageViewCreateInfo imageView{};
  imageView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  imageView.viewType = VK_IMAGE_VIEW_TYPE_1D;
  imageView.format = m_colormap.format;
  imageView.subresourceRange = {};
  imageView.subresourceRange.aspectMask = m_colormap.aspectMask;
  imageView.subresourceRange.baseMipLevel = 0;
  imageView.subresourceRange.levelCount = 1;
  imageView.subresourceRange.baseArrayLayer = 0;
  imageView.subresourceRange.layerCount = 1;

  createImageViewAndBindMem(m_device, &m_colormap, &imageView);

  m_pCopy->UpdateImage(m_colormap.image, infernoColorMap.data(), infernoColorMap.size(), 1, 4, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

//  m_colormap = vk_utils::allocateColorTextureFromDataLDR(m_device, m_physicalDevice, infernoColorMap.data(),
//                                                         infernoColorMap.size(), 1, 1,
//                                                         VK_FORMAT_R8G8B8A8_UNORM, m_pCopy);
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

  CreateColormapTexture();
  m_colormapSampler = vk_utils::createSampler(m_device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                              VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);

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
      m_sprite = vk_utils::allocateColorTextureFromDataLDR(m_device, m_physicalDevice, pixels, w, h, 1,
                                                           VK_FORMAT_R8G8B8A8_UNORM, m_pCopy);

      m_spriteSampler = vk_utils::createSampler(m_device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);

      stbi_image_free(pixels);
    }
  }

  auto imgCmdBuf = vk_utils::createCommandBuffer(m_device, m_commandPoolGraphics);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(imgCmdBuf, &beginInfo);
  {
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = m_colormap.aspectMask;
    subresourceRange.levelCount = 1;
    subresourceRange.layerCount = 1;
    vk_utils::setImageLayout(
        imgCmdBuf,
        m_colormap.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresourceRange);
    if(DISPLAY_MODE == RENDER_MODE::SPRITES)
    {
      vk_utils::setImageLayout(
          imgCmdBuf,
          m_sprite.image,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          subresourceRange);
    }
  }
  vkEndCommandBuffer(imgCmdBuf);
  vk_utils::executeCommandBufferNow(imgCmdBuf, m_graphicsQueue, m_device);

  if(DISPLAY_MODE == RENDER_MODE::SPRITES)
    SetupSpritesPipeline();
  else
    SetupPointsPipeline();

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
