#include "test_class.h"
#include "include/crandom.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum GEOM_FLAGS{ HAS_TANGENT    = 1,
  UNUSED2        = 2,
  UNUSED4        = 4,
  HAS_NO_NORMALS = 8};

struct SimpleMesh
{
  static const uint64_t POINTS_IN_TRIANGLE = 3;
  SimpleMesh(){}
  SimpleMesh(int a_vertNum, int a_indNum) { Resize(a_vertNum, a_indNum); }

  inline size_t VerticesNum()  const { return vPos4f.size(); }
  inline size_t IndicesNum()   const { return indices.size();  }
  inline size_t TrianglesNum() const { return IndicesNum() / POINTS_IN_TRIANGLE;  }
  inline void   Resize(uint32_t a_vertNum, uint32_t a_indNum)
  {
    vPos4f.resize(a_vertNum);
    vNorm4f.resize(a_vertNum);
    vTang4f.resize(a_vertNum);
    vTexCoord2f.resize(a_vertNum);
    indices.resize(a_indNum);
    matIndices.resize(a_indNum/3);
  };

  inline size_t SizeInBytes() const
  {
    return vPos4f.size()*sizeof(float)*4  +
           vNorm4f.size()*sizeof(float)*4 +
           vTang4f.size()*sizeof(float)*4 +
           vTexCoord2f.size()*sizeof(float)*2 +
           indices.size()*sizeof(int) +
           matIndices.size()*sizeof(int);
  }
  std::vector<LiteMath::float4> vPos4f;      //
  std::vector<LiteMath::float4> vNorm4f;     //
  std::vector<LiteMath::float4> vTang4f;     //
  std::vector<float2>                       vTexCoord2f; //
  std::vector<unsigned int>                 indices;     // size = 3*TrianglesNum() for triangle mesh, 4*TrianglesNum() for quad mesh
  std::vector<unsigned int>                 matIndices;  // size = 1*TrianglesNum()
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::InitSceneMaterials(int a_numSpheres, int a_seed)
{ 
  ///// create simple materials
  //
  m_materials.resize(10);
  m_materials[0] = float4(0.5f, 0.5f, 0.5f, 0.0f);
  m_materials[1] = float4(0.6f, 0.0235294f, 0.0235294f, 0.0f);
  m_materials[2] = float4(0.0235294f, 0.6f, 0.0235294f, 0.0f);
  m_materials[3] = float4(0.6f, 0.6f, 0.1f, 0.0f);
  m_materials[4] = float4(0.0847059, 0.144706,0.265882, 0.0f);
  m_materials[5] = float4(0.5f, 0.5f, 0.75f, 0.0f);
  m_materials[6] = float4(0.25f, 0.0f, 0.5f, 0.0f);
  m_materials[7] = m_materials[5];
  m_materials[8] = m_materials[5];
  m_materials[9] = float4(0.0, 0.0f, 0.0f, 20.0f); // emissive material
  m_emissiveMaterialId = 9;

}

int TestClass::LoadScene(const char* bvhPath, const char* meshPath, bool a_needReorder)
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
  
  std::cout << "[LoadScene]: create accel struct " << std::endl;
  
  m_materialIds      = m_mesh.matIndices;
  m_indicesReordered = m_mesh.indices;

  InitSceneMaterials(10);

  std::cout << "IndicesNum   = " << m_mesh.indices.size() << std::endl;
  std::cout << "TrianglesNum = " << m_mesh.TrianglesNum() << std::endl;
  std::cout << "MateriaIdNum = " << m_mesh.matIndices.size() << std::endl;

  m_pAccelStruct->ClearGeom();
  auto geomId = m_pAccelStruct->AddGeom_Triangles4f(m_vPos4f.data(), m_vPos4f.size(), m_indicesReordered.data(), m_indicesReordered.size());
  
  m_pAccelStruct->BeginScene();
  float4x4 identitiMatrix;
  auto instId = m_pAccelStruct->AddInstance(geomId, (const float*)&identitiMatrix);
  m_pAccelStruct->EndScene();

  return 0;

}