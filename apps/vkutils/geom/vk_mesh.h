#ifndef VK_MESH_H
#define VK_MESH_H

#define USE_VOLK
#include "vk_include.h"
#include <cstring>

#include "cmesh.h"

struct MeshInfo
{
  uint32_t m_vertNum = 0;
  uint32_t m_indNum  = 0;

  uint32_t m_vertexOffset = 0u;
  uint32_t m_indexOffset  = 0u;

  VkDeviceSize m_vertexBufOffset = 0u;
  VkDeviceSize m_indexBufOffset  = 0u;
};

struct IMeshData
{
  virtual void Append(const cmesh::SimpleMesh &meshData) = 0;
  virtual float* VertexData() = 0;
  virtual uint32_t* IndexData() = 0;

  virtual size_t VertexDataSize() = 0;
  virtual size_t IndexDataSize() = 0;

  virtual size_t SingleVertexSize() = 0;
  virtual size_t SingleIndexSize() = 0;

  virtual VkPipelineVertexInputStateCreateInfo VertexInputLayout() = 0;

  virtual ~IMeshData() = default;
};

struct Mesh8F : IMeshData
{
  void Append(const cmesh::SimpleMesh &meshData) override;

  float*    VertexData() override { return (float*)vertices.data();}
  uint32_t* IndexData()  override { return indices.data();}

  size_t VertexDataSize() override { return vertices.size() * SingleVertexSize(); };
  size_t IndexDataSize() override { return indices.size() * SingleIndexSize(); };

  size_t SingleVertexSize() override { return sizeof(vertex); }
  size_t SingleIndexSize()  override { return sizeof(uint32_t); };

  VkPipelineVertexInputStateCreateInfo VertexInputLayout() override;

private:
  struct vertex {
    float posNorm[4];      // (pos_x, pos_y, pos_z, compressed_normal)
    float texCoordTang[4]; // (u, v, compressed_tangent, unused_val)
  };

  std::vector<vertex> vertices;
  std::vector<uint32_t> indices;

  VkVertexInputBindingDescription   m_inputBinding {};
  VkVertexInputAttributeDescription m_inputAttributes[2] {};
};

static inline unsigned int EncodeNormal(const float n[3])
{
  const int x = (int)(n[0]*32767.0f);
  const int y = (int)(n[1]*32767.0f);

  const unsigned int sign = (n[2] >= 0) ? 0 : 1;
  const unsigned int sx   = ((unsigned int)(x & 0xfffe) | sign);
  const unsigned int sy   = ((unsigned int)(y & 0xffff) << 16);

  return (sx | sy);
}

static inline unsigned int as_uint(float x)
{
  unsigned int res;
  memcpy(&res, &x, sizeof(unsigned int));
  return res;
}

static inline float as_float(unsigned int x)
{
  float res;
  memcpy(&res, &x, sizeof(float));
  return res;
}

#endif//VK_MESH_H
