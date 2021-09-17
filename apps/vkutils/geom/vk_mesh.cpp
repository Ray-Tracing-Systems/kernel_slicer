#include "vk_mesh.h"

VkPipelineVertexInputStateCreateInfo Mesh8F::VertexInputLayout()
{
  m_inputBinding.binding   = 0;
  m_inputBinding.stride    = sizeof(vertex);
  m_inputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  m_inputAttributes[0].binding  = 0;
  m_inputAttributes[0].location = 0;
  m_inputAttributes[0].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  m_inputAttributes[0].offset   = 0;

  m_inputAttributes[1].binding  = 0;
  m_inputAttributes[1].location = 1;
  m_inputAttributes[1].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
  m_inputAttributes[1].offset   = sizeof(float) * 4;

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
  std::copy(meshData.indices.begin(), meshData.indices.end(), indices.begin() + old_size);

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