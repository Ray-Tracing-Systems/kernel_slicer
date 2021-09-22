#ifndef VK_SORT_VK_UTILS_H
#define VK_SORT_VK_UTILS_H

#define USE_VOLK
#include "vk_include.h"

#include "vk_descriptor_helpers.h"

#include <string>
#include <vector>
#include <cassert>
#include <unordered_map>

namespace vk_utils
{

  VkDescriptorSetLayout createDescriptorSetLayout(VkDevice a_device, const DescriptorTypesVec &a_descrTypes, VkShaderStageFlags a_stage = VK_SHADER_STAGE_COMPUTE_BIT);
  VkDescriptorPool createDescriptorPool(VkDevice a_device, const DescriptorTypesVec &a_descrTypes, unsigned a_maxSets);
  VkDescriptorSet createDescriptorSet(VkDevice a_device, VkDescriptorSetLayout a_pDSLayout, VkDescriptorPool a_pDSPool, const std::vector<VkDescriptorBufferInfo> &bufInfos);

  class DescriptorMaker
  {
  public:
    DescriptorMaker(VkDevice a_device, const DescriptorTypesVec &a_maxDescrTypes, uint32_t a_maxSets);
    ~DescriptorMaker();

    void BindBegin(VkShaderStageFlags a_shaderStage);
    void BindBuffer(uint32_t a_loc, VkBuffer a_buffer, VkBufferView a_buffView = VK_NULL_HANDLE, VkDescriptorType a_bindType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

#ifndef __ANDROID__
    void BindAccelStruct(uint32_t a_loc, VkAccelerationStructureKHR a_accStruct, VkDescriptorType a_bindType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
#endif
    void BindImage(uint32_t a_loc, VkImageView a_imageView, VkSampler a_sampler = VK_NULL_HANDLE, VkDescriptorType a_bindType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VkImageLayout a_imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void BindImageArray(uint32_t a_loc, const std::vector<VkImageView> &a_imageView, const std::vector<VkSampler> &a_sampler, VkDescriptorType a_bindType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VkImageLayout a_imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void BindEnd(VkDescriptorSet *a_pSet = nullptr, VkDescriptorSetLayout *a_pLayout = nullptr);

    VkDescriptorPool GetPool() const { return m_pool; }

    enum class HASHING_MODE {
      NONE,
      LAYOUTS_ONLY,
      LAYOUTS_AND_SETS
    };
    HASHING_MODE hashingMode = HASHING_MODE::LAYOUTS_AND_SETS;

  private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;

    VkShaderStageFlags m_currentStageFlags = 0u;
    std::unordered_map<uint32_t, DescriptorHandles> m_bindings;// shader location to vk handle(s)

    std::unordered_map<LayoutKey, VkDescriptorSetLayout, LayoutHash> m_layoutDict;
    std::unordered_map<SetKey, VkDescriptorSet, SetHash> m_setDict;

  };
}// namespace vk_utils

#endif// VK_SORT_VK_UTILS_H
