#include "vk_mesh.h"

void vk_utils::AddInstanceMatrixAttributeToVertexLayout(uint32_t a_binding, uint32_t a_stride,
  VkPipelineVertexInputStateCreateInfo &a_vertexLayout)
{
  static std::vector<VkVertexInputBindingDescription>   tmpVertexInputBindings;
  static std::vector<VkVertexInputAttributeDescription> tmpVertexInputAttribDescr;

  auto oldAttrDescrCount = a_vertexLayout.vertexAttributeDescriptionCount;
  auto oldBindDescrCount = a_vertexLayout.vertexBindingDescriptionCount;

  tmpVertexInputBindings.resize(oldBindDescrCount);
  tmpVertexInputAttribDescr.resize(oldAttrDescrCount);

  for(uint32_t i = 0; i < oldBindDescrCount; ++i)
  {
    tmpVertexInputBindings[i] = a_vertexLayout.pVertexBindingDescriptions[i];
  }
  for(uint32_t i = 0; i < oldAttrDescrCount; ++i)
  {
    tmpVertexInputAttribDescr[i] = a_vertexLayout.pVertexAttributeDescriptions[i];
  }

  VkVertexInputAttributeDescription float4x4AttrDescription{};
  float4x4AttrDescription.location = oldAttrDescrCount;
  float4x4AttrDescription.binding  = a_binding;
  float4x4AttrDescription.format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  float4x4AttrDescription.offset   = 0;
  tmpVertexInputAttribDescr.push_back(float4x4AttrDescription);

  float4x4AttrDescription.location = oldAttrDescrCount + 1;
  float4x4AttrDescription.binding  = a_binding;
  float4x4AttrDescription.format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  float4x4AttrDescription.offset   = sizeof(float) * 4;
  tmpVertexInputAttribDescr.push_back(float4x4AttrDescription);

  float4x4AttrDescription.location = oldAttrDescrCount + 2;
  float4x4AttrDescription.binding  = a_binding;
  float4x4AttrDescription.format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  float4x4AttrDescription.offset   = sizeof(float) * 8;
  tmpVertexInputAttribDescr.push_back(float4x4AttrDescription);

  float4x4AttrDescription.location = oldAttrDescrCount + 3;
  float4x4AttrDescription.binding  = a_binding;
  float4x4AttrDescription.format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  float4x4AttrDescription.offset   = sizeof(float) * 12;
  tmpVertexInputAttribDescr.push_back(float4x4AttrDescription);

  VkVertexInputBindingDescription perInstanceBinding {};
  perInstanceBinding.binding   = a_binding;
  perInstanceBinding.stride    = a_stride;
  perInstanceBinding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
  tmpVertexInputBindings.push_back(perInstanceBinding);

  a_vertexLayout.vertexBindingDescriptionCount   = (uint32_t)tmpVertexInputBindings.size();
  a_vertexLayout.pVertexBindingDescriptions      = tmpVertexInputBindings.data();
  a_vertexLayout.vertexAttributeDescriptionCount = (uint32_t)tmpVertexInputAttribDescr.size();
  a_vertexLayout.pVertexAttributeDescriptions    = tmpVertexInputAttribDescr.data();
}

VkPipelineVertexInputStateCreateInfo Mesh8F::VertexInputLayout()
{
  m_inputBinding.binding   = 0;
  m_inputBinding.stride    = sizeof(vertex);
  m_inputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  m_inputAttributes[0].binding  = 0;
  m_inputAttributes[0].location = 0;
  m_inputAttributes[0].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  m_inputAttributes[0].offset   = static_cast<uint32_t >(offsetof(vertex, posNorm)); // 0

  m_inputAttributes[1].binding  = 0;
  m_inputAttributes[1].location = 1;
  m_inputAttributes[1].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  m_inputAttributes[1].offset   = static_cast<uint32_t >(offsetof(vertex, texCoordTang)); // 4 floats =  16 bytes

  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
  vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount   = 1;
  vertexInputInfo.vertexAttributeDescriptionCount = sizeof(m_inputAttributes) / sizeof(m_inputAttributes[0]);
  vertexInputInfo.pVertexBindingDescriptions      = &m_inputBinding;
  vertexInputInfo.pVertexAttributeDescriptions    = m_inputAttributes;

  return vertexInputInfo;
}


void Mesh8F::Append(const cmesh::SimpleMesh &meshData)
{
  vertices.reserve(vertices.size() + meshData.VerticesNum());
  auto old_size = indices.size();
  indices.resize(indices.size() + meshData.IndicesNum());

  memcpy(indices.data() + old_size, meshData.indices.data(), meshData.indices.size() * sizeof(meshData.indices[0]));

  for(size_t i = 0; i < meshData.VerticesNum(); ++i)
  {
    vertex v = {};

    float normal[3]  = {meshData.vNorm4f[i * 4 + 0], meshData.vNorm4f[i * 4 + 1], meshData.vNorm4f[i * 4 + 2]};
    float tangent[3] = {meshData.vTang4f[i * 4 + 0], meshData.vTang4f[i * 4 + 1], meshData.vTang4f[i * 4 + 2]};
    v.posNorm[0] = meshData.vPos4f[i * 4 + 0];
    v.posNorm[1] = meshData.vPos4f[i * 4 + 1];
    v.posNorm[2] = meshData.vPos4f[i * 4 + 2];
    v.posNorm[3] = as_float(EncodeNormal(normal));

    v.texCoordTang[0] = meshData.vTexCoord2f[i * 2 + 0];
    v.texCoordTang[1] = meshData.vTexCoord2f[i * 2 + 1];
    v.texCoordTang[2] = as_float(EncodeNormal(tangent));
    v.texCoordTang[3] = 0.0f;
    vertices.push_back(v);
  }
}

// ************************************************************************
// Mesh4F

VkPipelineVertexInputStateCreateInfo Mesh4F::VertexInputLayout()
{
  m_inputBinding.binding   = 0;
  m_inputBinding.stride    = sizeof(vertex);
  m_inputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  m_inputAttributes.binding  = 0;
  m_inputAttributes.location = 0;
  m_inputAttributes.format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  m_inputAttributes.offset   = 0;

  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
  vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount   = 1;
  vertexInputInfo.vertexAttributeDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions      = &m_inputBinding;
  vertexInputInfo.pVertexAttributeDescriptions    = &m_inputAttributes;

  return vertexInputInfo;
}


void Mesh4F::Append(const cmesh::SimpleMesh &meshData)
{
  vertices.reserve(vertices.size() + meshData.VerticesNum());
  auto old_size = indices.size();
  indices.resize(indices.size() + meshData.IndicesNum());

  memcpy(indices.data() + old_size, meshData.indices.data(), meshData.indices.size() * sizeof(meshData.indices[0]));

  for(size_t i = 0; i < meshData.VerticesNum(); ++i)
  {
    vertex v = {};
    v.pos[0] = meshData.vPos4f[i * 4 + 0];
    v.pos[1] = meshData.vPos4f[i * 4 + 1];
    v.pos[2] = meshData.vPos4f[i * 4 + 2];
    v.pos[3] = 1.0f;

    vertices.push_back(v);
  }
}