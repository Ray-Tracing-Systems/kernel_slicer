#include "vk_mesh.h"

VkPipelineVertexInputStateCreateInfo Mesh8F::VertexInputLayout()
{
  m_inputBinding.binding   = 0;
  m_inputBinding.stride    = sizeof(float) * 8;
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
  vertexInputInfo.vertexAttributeDescriptionCount = 2;
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
    vertex v;

    float normal[3] = {meshData.vNorm4f[i].x, meshData.vNorm4f[i].y, meshData.vNorm4f[i].z};
    float tangent[3] = {meshData.vTang4f[i].x, meshData.vTang4f[i].y, meshData.vTang4f[i].z};
    v.posNorm = {meshData.vPos4f[i].x, meshData.vPos4f[i].y, meshData.vPos4f[i].z, as_float(EncodeNormal(normal))};
    v.texCoordTang = {meshData.vTexCoord2f[i].x, meshData.vTexCoord2f[i].y, as_float(EncodeNormal(tangent)), 0.0f};

    vertices.push_back(v);
  }
}