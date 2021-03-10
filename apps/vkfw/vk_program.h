#ifndef VULKAN_PROGRAM_H
#define VULKAN_PROGRAM_H

#include <vulkan/vulkan.h>
#include "vk_utils.h"

#include <vector>
#include <map>
#include <unordered_map>

#include "vk_program_helper.h"

namespace vkfw
{ 
  struct DSetId
  {
    vk_utils::PBKey key;
    size_t          dSetIndex;
  };

  struct ProgramBindings
  {
    ProgramBindings(VkDevice a_device, const VkDescriptorType* a_dtypes, const uint32_t* a_dtypesCount, int a_dtypesSize, int a_maxSets = 0);
    virtual ~ProgramBindings();
  
    virtual void BindBegin (VkShaderStageFlagBits a_shaderStage);
    virtual void BindBuffer(uint32_t a_loc, VkBuffer     a_buffer,    size_t a_buffOffset = 0, VkDescriptorType a_bindType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    virtual void BindImage (uint32_t a_loc, VkImageView  a_imageView, VkDescriptorType a_bindType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    virtual void BindImage (uint32_t a_loc, VkImageView  a_imageView, VkSampler a_sampler, VkDescriptorType a_bindType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    virtual DSetId BindEnd (VkDescriptorSet* a_pSet = nullptr, VkDescriptorSetLayout* a_pLayout = nullptr);

    virtual VkDescriptorSetLayout DLayout(const DSetId& a_setId) const;
    virtual VkDescriptorSet       DSet   (const DSetId& a_setId) const;

    VkDescriptorPool GetPool() const { return m_pool;}
   
  protected:

    VkDescriptorPool m_pool;
    std::vector<VkDescriptorType> m_poolTypes;

    ProgramBindings(const ProgramBindings& a_rhs){}
    ProgramBindings& operator=(const ProgramBindings& a_rhs) { return *this; }

    VkDescriptorSetLayout m_currLayout;
    VkDescriptorSet       m_currSet;
    VkShaderStageFlagBits m_stageFlags;
    VkDevice              m_device;

    std::map<int, vk_utils::PBinding> m_currBindings;
    std::map<int, VkDescriptorType>   m_currBindingsTypes;
    
    struct DSData
    {
      DSData() { sets.reserve(16); }
      DSData(VkDescriptorSetLayout a_layout) : layout(a_layout) { sets.reserve(16); }
      VkDescriptorSetLayout        layout = {};
      std::vector<VkDescriptorSet> sets;
    };

    std::unordered_map<vk_utils::PBKey, DSData> m_dsLayouts;
  };

};

#endif