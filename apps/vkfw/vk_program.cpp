#include "vk_program.h"
#include "vk_utils.h"

vkfw::ProgramBindings::ProgramBindings(VkDevice a_device, const VkDescriptorType* a_dtypes, const uint32_t* a_dtypesCount, int a_dtypesSize, 
                                           int a_maxSets) : m_device(a_device)
{
  m_poolTypes.assign(a_dtypes, a_dtypes + a_dtypesSize);

  uint32_t maxSetsEstimated = 0;
  std::vector<VkDescriptorPoolSize> descriptorPoolSize(m_poolTypes.size());
  for (size_t i = 0; i < descriptorPoolSize.size(); i++)
  {
    descriptorPoolSize[i]                 = {};
    descriptorPoolSize[i].type            = m_poolTypes[i];
    descriptorPoolSize[i].descriptorCount = a_dtypesCount[i];
    maxSetsEstimated                     += a_dtypesCount[i];
  }
  
  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.maxSets       = (a_maxSets == 0) ? maxSetsEstimated : a_maxSets;
  descriptorPoolCreateInfo.poolSizeCount = uint32_t(descriptorPoolSize.size());
  descriptorPoolCreateInfo.pPoolSizes    = descriptorPoolSize.data();
  
  VK_CHECK_RESULT(vkCreateDescriptorPool(m_device, &descriptorPoolCreateInfo, NULL, &m_pool));
}

vkfw::ProgramBindings::~ProgramBindings()
{
  assert(m_device != nullptr);

  for (auto& l : m_dsLayouts)
    vkDestroyDescriptorSetLayout(m_device, l.second.layout, NULL);

  vkDestroyDescriptorPool(m_device, m_pool, NULL);
}

void vkfw::ProgramBindings::BindBegin(VkShaderStageFlagBits a_shaderStage)
{
  m_stageFlags = a_shaderStage;
  m_currBindings.clear();
  m_currBindingsTypes.clear();
}

void vkfw::ProgramBindings::BindBuffer(uint32_t a_loc, VkBuffer a_buff, size_t a_buffOffset, VkDescriptorType a_bindType)
{
  vk_utils::PBinding bind;
  bind.buffView  = VK_NULL_HANDLE;
  bind.buffer    = a_buff;
  bind.bufferOffset = a_buffOffset;
  bind.imageView = nullptr;
  bind.type      = a_bindType;

  m_currBindings[a_loc]      = bind;
  m_currBindingsTypes[a_loc] = a_bindType;
}


void vkfw::ProgramBindings::BindImage(uint32_t a_loc, VkImageView a_imageView, VkDescriptorType a_bindType)
{
  vk_utils::PBinding bind;
  bind.buffer    = nullptr;
  bind.buffView  = nullptr;
  bind.imageView = a_imageView;
  bind.type      = a_bindType;

  m_currBindings[a_loc]      = bind;
  m_currBindingsTypes[a_loc] = a_bindType;
}

void  vkfw::ProgramBindings::BindImage(uint32_t a_loc, VkImageView a_imageView, VkSampler a_sampler, VkDescriptorType a_bindType)
{
  vk_utils::PBinding bind;
  bind.buffer       = nullptr;
  bind.buffView     = nullptr;
  bind.imageView    = a_imageView;
  bind.imageSampler = a_sampler;
  bind.type         = a_bindType;

  m_currBindings[a_loc]      = bind;
  m_currBindingsTypes[a_loc] = a_bindType;
}

vkfw::DSetId vkfw::ProgramBindings::BindEnd(VkDescriptorSet* a_pSet, VkDescriptorSetLayout* a_pLayout)
{
  // create DS layout key
  //
  auto currKey = vk_utils::PBKey(m_currBindings, m_stageFlags);
  auto p       = m_dsLayouts.find(currKey);

  // get DS layout
  //
  VkDescriptorSetLayout layout = nullptr;
  if (p != m_dsLayouts.end()) // if we have such ds layout fetch it from cache
  {
    layout = p->second.layout;
  }
  else                        // otherwise create 
  {
    // With the pool allocated, we can now allocate the descriptor set layout
    //
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBinding(m_currBindings.size());
    
    int top = 0;
    for (const auto& currB : m_currBindings)
    {
      descriptorSetLayoutBinding[top].binding            = uint32_t(currB.first);
      descriptorSetLayoutBinding[top].descriptorType     = currB.second.type;
      descriptorSetLayoutBinding[top].descriptorCount    = 1;
      descriptorSetLayoutBinding[top].stageFlags         = m_stageFlags;
      descriptorSetLayoutBinding[top].pImmutableSamplers = nullptr;
      top++;
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = uint32_t(top);
    descriptorSetLayoutCreateInfo.pBindings    = descriptorSetLayoutBinding.data();

    // Create the descriptor set layout.
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutCreateInfo, NULL, &layout));

    m_dsLayouts[currKey] = DSData(layout);
    p = m_dsLayouts.find(currKey);
  }

  assert(p != m_dsLayouts.end());

  // create descriptor set 
  //
  VkDescriptorSet ds;

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.descriptorPool     = m_pool;   // pool to allocate from.
  descriptorSetAllocateInfo.descriptorSetCount = 1;        // allocate a descriptor set for buffer and image
  descriptorSetAllocateInfo.pSetLayouts        = &layout;

  // allocate descriptor set.
  //
  auto tmpRes = vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, &ds);
  VK_CHECK_RESULT(tmpRes); //#TODO: do not allocate if you already has such descriptor set!

  p->second.sets.push_back(ds);

  // update current descriptor set, #NOTE: THE IMPLEMENTATION IS NOT FINISHED!!!! ITS PROTOTYPE!!!!
  //
  const size_t descriptorsInSet = m_currBindings.size();

  std::vector<VkDescriptorImageInfo>  descriptorImageInfo(descriptorsInSet);
  std::vector<VkDescriptorBufferInfo> descriptorBufferInfo(descriptorsInSet);
  std::vector<VkWriteDescriptorSet>   writeDescriptorSet(descriptorsInSet);
  
  int top = 0;
  for (auto& binding : m_currBindings)
  {
    descriptorImageInfo[top]             = VkDescriptorImageInfo{};
    descriptorImageInfo[top].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    descriptorImageInfo[top].imageView   = binding.second.imageView;
    descriptorImageInfo[top].sampler     = binding.second.imageSampler;

    descriptorBufferInfo[top]        = VkDescriptorBufferInfo{};
    descriptorBufferInfo[top].buffer = binding.second.buffer;
    descriptorBufferInfo[top].offset = binding.second.bufferOffset; 
    descriptorBufferInfo[top].range  = VK_WHOLE_SIZE;  // #TODO: update here!

    writeDescriptorSet[top]                  = VkWriteDescriptorSet{};
    writeDescriptorSet[top].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[top].dstSet           = ds;
    writeDescriptorSet[top].dstBinding       = binding.first;
    writeDescriptorSet[top].descriptorCount  = 1;
    writeDescriptorSet[top].descriptorType   = m_currBindingsTypes[binding.first]; //VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

    if(m_currBindingsTypes[binding.first] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || m_currBindingsTypes[binding.first] == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
    {
      writeDescriptorSet[top].pImageInfo  = &descriptorImageInfo[top];
      writeDescriptorSet[top].pBufferInfo = nullptr;
    }
    else if (m_currBindingsTypes[binding.first] == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || m_currBindingsTypes[binding.first] == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
    {
      writeDescriptorSet[top].pBufferInfo = &descriptorBufferInfo[top];
      writeDescriptorSet[top].pImageInfo  = nullptr;
    }
    writeDescriptorSet[top].pTexelBufferView = nullptr; // #TODO: update here!
    top++;
  }

  vkUpdateDescriptorSets(m_device, uint32_t(top), writeDescriptorSet.data(), 0, NULL);
  
  if(a_pSet != nullptr)
    (*a_pSet) = ds;

  if(a_pLayout != nullptr)
    (*a_pLayout) = layout;

  // return an id that we can use to retrieve same descriptor set later
  //
  DSetId res;
  res.key       = p->first;
  res.dSetIndex = p->second.sets.size() - 1;
  return res;
}

VkDescriptorSetLayout vkfw::ProgramBindings::DLayout(const DSetId& a_setId) const
{
  auto p = m_dsLayouts.find(a_setId.key);
  if(p == m_dsLayouts.end())
    return nullptr;

  return p->second.layout;
}

VkDescriptorSet vkfw::ProgramBindings::DSet(const DSetId& a_setId) const
{
  auto p = m_dsLayouts.find(a_setId.key);
  if (p == m_dsLayouts.end())
    return nullptr;

  if (a_setId.dSetIndex >= p->second.sets.size())
    return nullptr;

  return p->second.sets[a_setId.dSetIndex];
}
