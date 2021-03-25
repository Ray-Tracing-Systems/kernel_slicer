#include "test_class.h"
#include "include/crandom.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::InitSpheresScene(int a_numSpheres, int a_seed)
{ 
  spheresMaterials.resize(10);

  spheresMaterials[0].color = float4(0.5,0.5,0.5, 0.0f); // grayOverrideMat
  spheresMaterials[1].color = float4(0.078, 0, 0.156, 0.0f); // hydra_placeholder_material

  const float col = 0.75f;
  const float eps = 0.00f;

  spheresMaterials[2].color = float4(0.0235294,0.6,0.0235294,0); // Green
  spheresMaterials[3].color = float4(0.0847059,0.144706,0.265882,0); // Blue
  spheresMaterials[4].color = float4(0.6,0.0235294,0.0235294,0); // Red
  spheresMaterials[5].color = float4(0.6,0.6,0.6,0); // White
  spheresMaterials[6].color = float4(0.8,0.715294,0,0); // teaport_material, phong or ggx
  spheresMaterials[7].color = float4(0.0,0.0,0.0,0); // mirror
  spheresMaterials[8].color = float4(0,0,0,0); // environment_material
  spheresMaterials[9].color = float4(1,1,1,28); // TPhotometricLight001_material
}

int TestClass::LoadScene(const char* bvhPath, const char* meshPath)
{
  std::fstream input_file;
  input_file.open(bvhPath, std::ios::binary | std::ios::in);
  if (!input_file.is_open())
  {
    std::cout << "BVH file error <" << bvhPath << ">\n";
    return 1;
  }
  struct BVHDataHeader
  {
    uint64_t node_length;
    uint64_t indices_length;
    uint64_t depth_length;
    uint64_t geom_id;
  };
  BVHDataHeader header;
  input_file.read((char *) &header, sizeof(BVHDataHeader));

  m_nodes.resize(header.node_length);
  m_intervals.resize(header.node_length);
  m_indicesReordered.resize(header.indices_length);

  input_file.read((char *) m_nodes.data(), sizeof(BVHNode) * header.node_length);
  input_file.read((char *) m_intervals.data(), sizeof(Interval) * header.node_length);
  input_file.read((char *) m_indicesReordered.data(), sizeof(uint) * header.indices_length);
  //input_file.read((char *) m_bvhTree.depthRanges.data(), sizeof(Interval) * header.depth_length);

  std::fstream input_file_mesh;
  input_file_mesh.open(meshPath, std::ios::binary | std::ios::in);
  if (!input_file_mesh.is_open())
  {
    std::cout << "Mesh file error <" << meshPath << ">\n";
    return 1;
  }

  struct VSGFHeader
  {
    uint64_t fileSizeInBytes;
    uint32_t verticesNum;
    uint32_t indicesNum;
    uint32_t materialsNum;
    uint32_t flags;
  };
  VSGFHeader meshHeader;

  input_file_mesh.read((char*)&meshHeader, sizeof(VSGFHeader));
  SimpleMesh m_mesh = SimpleMesh(meshHeader.verticesNum, meshHeader.indicesNum);

  input_file_mesh.read((char*)m_mesh.vPos4f.data(),  m_mesh.vPos4f.size()*sizeof(float)*4);

  if(!(meshHeader.flags & HAS_NO_NORMALS))
    input_file_mesh.read((char*)m_mesh.vNorm4f.data(), m_mesh.vNorm4f.size()*sizeof(float)*4);
  else
    memset(m_mesh.vNorm4f.data(), 0, m_mesh.vNorm4f.size()*sizeof(float)*4);  

  if(meshHeader.flags & HAS_TANGENT)
    input_file_mesh.read((char*)m_mesh.vTang4f.data(), m_mesh.vTang4f.size()*sizeof(float)*4);
  else
    memset(m_mesh.vTang4f.data(), 0, m_mesh.vTang4f.size()*sizeof(float)*4);

  input_file_mesh.read((char*)m_mesh.vTexCoord2f.data(), m_mesh.vTexCoord2f.size()*sizeof(float)*2);
  input_file_mesh.read((char*)m_mesh.indices.data(),    m_mesh.indices.size()*sizeof(unsigned int));
  input_file_mesh.read((char*)m_mesh.matIndices.data(), m_mesh.matIndices.size()*sizeof(unsigned int));
  input_file_mesh.close();

  m_vPos4f      = m_mesh.vPos4f;
  m_vNorm4f     = m_mesh.vNorm4f;
  m_materialIds = m_mesh.matIndices;

  std::cout << "IndicesNum   = " << m_mesh.indices.size() << std::endl;
  std::cout << "TrianglesNum = " << m_mesh.TrianglesNum() << std::endl;
  std::cout << "MateriaIdNum = " << m_mesh.matIndices.size() << std::endl;

  //std::ofstream outFile("mid.txt");
  //for(size_t i=0;i<m_materialIds.size();i++)
  //outFile << m_materialIds[i] << std::endl;

  return 0;

}