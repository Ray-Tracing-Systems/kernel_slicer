#ifndef VKUTILS_VK_DESCRIPTOR_HELPERS_H
#define VKUTILS_VK_DESCRIPTOR_HELPERS_H

#include <vector>
#include <tuple>

namespace vk_utils
{
  using DescriptorTypesVec = std::vector <std::pair<VkDescriptorType, uint32_t>>;

  // from boost
  // https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
  template<class T>
  inline void hash_combine(std::size_t &seed, const T &v)
  {
    std::hash <T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  struct DescriptorHandles
  {
    VkBufferView buffView = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    std::vector <VkImageView> imageView;
    std::vector <VkSampler> imageSampler;
#ifndef __ANDROID__
    VkAccelerationStructureKHR accelStruct = VK_NULL_HANDLE;
#endif

    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    bool operator==(const DescriptorHandles &rhs) const
    {
#ifndef __ANDROID__
      return std::tie(type, buffer, buffView, accelStruct, imageLayout, imageView, imageSampler) ==
             std::tie(rhs.type, rhs.buffer, rhs.buffView, rhs.accelStruct, rhs.imageLayout, rhs.imageView,
                      rhs.imageSampler);
#else
      return std::tie(type, buffer, buffView, imageLayout, imageView, imageSampler) ==
             std::tie(rhs.type, rhs.buffer, rhs.buffView, rhs.imageLayout, rhs.imageView,
                      rhs.imageSampler);
#endif
    }
  };

  struct LayoutKey
  {
    VkShaderStageFlags stageFlags;
    DescriptorTypesVec descriptorTypes;

    bool operator==(const LayoutKey &rhs) const
    {
      return std::tie(stageFlags, descriptorTypes) == std::tie(rhs.stageFlags, rhs.descriptorTypes);
    }
  };

  struct LayoutHash
  {
    size_t operator()(const LayoutKey &key) const
    {
      size_t currHash = std::hash<VkShaderStageFlags>()(key.stageFlags);
      hash_combine(currHash, key.descriptorTypes.size());
      for (const auto &[dtype, count] : key.descriptorTypes)
      {
        hash_combine(currHash, dtype);
        hash_combine(currHash, count);
      }
      return currHash;
    }
  };

  struct SetKey
  {
    VkDescriptorSetLayout layout;
    std::vector <DescriptorHandles> handles;

    bool operator==(const SetKey &rhs) const
    {
      return std::tie(layout, handles) == std::tie(rhs.layout, rhs.handles);
    }
  };

  struct SetHash
  {
    size_t operator()(const SetKey &key) const
    {
      size_t currHash = std::hash<VkDescriptorSetLayout>()(key.layout);
      for (const auto &handle : key.handles)
      {
        hash_combine(currHash, handle.type);
        for (const auto &samp : handle.imageSampler)
          hash_combine(currHash, samp);
        for (const auto &view : handle.imageSampler)
          hash_combine(currHash, view);
        hash_combine(currHash, handle.imageLayout);
#ifndef __ANDROID__
        hash_combine(currHash, handle.accelStruct);
#endif
        hash_combine(currHash, handle.buffView);
        hash_combine(currHash, handle.buffer);
      }

      return currHash;
    }
  };
}
#endif //VKUTILS_VK_DESCRIPTOR_HELPERS_H
