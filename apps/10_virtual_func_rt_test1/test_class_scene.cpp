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

void TestClass::InitSpheresScene(int a_numSpheres, int a_seed)
{ 
  spheresMaterials.resize(10);

  spheresMaterials[0].color = float4(0.5,0.5,0.5, 0.0f); // grayOverrideMat
  spheresMaterials[6].color = float4(0.078, 0, 0.156, 0.0f); // hydra_placeholder_material
  spheresMaterials[2].color = float4(0.0235294, 0.6, 0.0235294,0); // Green
  spheresMaterials[4].color = float4(0.0847059, 0.144706,0.265882,0); // Blue
  spheresMaterials[1].color = float4(0.6,0.0235294,0.0235294,0); // Red
  spheresMaterials[5].color = float4(0.6,0.6,0.6,0); // White
  spheresMaterials[3].color = float4(0.8,0.715294,0,0); // teaport_material, phong or ggx
  spheresMaterials[7].color = float4(0,0,0,0); // mirror
  spheresMaterials[8].color = float4(0,0,0,0); // environment_material
  spheresMaterials[9].color = float4(1,1,1,28); // TPhotometricLight001_material

  auto maxSize = std::max( std::max(sizeof(EmissiveMaterial),      sizeof(GGXGlossyMaterial)), 
                           std::max(sizeof(PerfectMirrorMaterial), sizeof(LambertMaterial)));
  
  // using in place new to create objects on the CPU side
  //
  m_materialData.resize(10*maxSize/sizeof(uint32_t));
  m_materialOffsets.resize(10);

  uint32_t* pData = m_materialData.data();
  
  m_materialOffsets[0] = uint32_t(pData - m_materialData.data());            // TODO: encode offset and type in single uint32_t
  auto* pGrey = new (pData) LambertMaterial(float3(0.5,0.5,0.5)); 
  pData += sizeof(LambertMaterial)/sizeof(uint32_t);
  
  m_materialOffsets[1] = uint32_t(pData - m_materialData.data());
  auto* pRed = new (pData) LambertMaterial(float3(0.6,0.0235294,0.0235294)); 
  pData += sizeof(LambertMaterial)/sizeof(uint32_t);

  m_materialOffsets[2] = uint32_t(pData - m_materialData.data());
  auto* pGreen = new (pData) LambertMaterial(float3(0.0235294, 0.6, 0.0235294)); 
  pData += sizeof(LambertMaterial)/sizeof(uint32_t);

  m_materialOffsets[3] = uint32_t(pData - m_materialData.data());
  auto* pTeapotMtl = new (pData) GGXGlossyMaterial(float3(0.0235294, 0.6, 0.0235294)); 
  pData += sizeof(GGXGlossyMaterial)/sizeof(uint32_t);

  m_materialOffsets[4] = uint32_t(pData - m_materialData.data());
  auto* pBlue = new (pData) LambertMaterial(float3(0.0847059, 0.144706,0.265882)); 
  pData += sizeof(LambertMaterial)/sizeof(uint32_t);

  m_materialOffsets[5] = uint32_t(pData - m_materialData.data());
  auto* pWhite = new (pData) LambertMaterial(float3(0.6,0.6,0.6)); 
  pData += sizeof(LambertMaterial)/sizeof(uint32_t);

  m_materialOffsets[6] = uint32_t(pData - m_materialData.data());
  auto* pSomething = new (pData) LambertMaterial(float3(0.078, 0, 0.156)); 
  pData += sizeof(LambertMaterial)/sizeof(uint32_t);

  m_materialOffsets[7] = uint32_t(pData - m_materialData.data());
  auto* pMirror = new (pData) PerfectMirrorMaterial; 
  pData += sizeof(PerfectMirrorMaterial)/sizeof(uint32_t);

  m_materialOffsets[8] = uint32_t(pData - m_materialData.data());
  auto* pEnvUnused = new (pData) PerfectMirrorMaterial; 
  pData += sizeof(PerfectMirrorMaterial)/sizeof(uint32_t);

  m_materialOffsets[9] = uint32_t(pData - m_materialData.data());
  auto* pEmissive = new (pData) EmissiveMaterial; 
  pData += sizeof(EmissiveMaterial)/sizeof(uint32_t);
  
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
  
  std::cout << "[LoadScene]: fixing material indices back ... " << std::endl;

  m_materialIds.resize(m_mesh.matIndices.size());   
  //m_materialIds = m_mesh.matIndices; // // NO!!! Need to reorder them accordimg to reorderedIndices!!!
  #pragma omp parallel for
  for(uint32_t triIdNew = 0; triIdNew < m_mesh.TrianglesNum(); triIdNew++)
  {
    const uint32_t A = m_indicesReordered[triIdNew*3+0];
    const uint32_t B = m_indicesReordered[triIdNew*3+1];
    const uint32_t C = m_indicesReordered[triIdNew*3+2];

    for(uint32_t triIdOld = 0; triIdOld < m_mesh.TrianglesNum(); triIdOld++)
    {
      const uint32_t AOld = m_mesh.indices[triIdOld*3+0];
      const uint32_t BOld = m_mesh.indices[triIdOld*3+1];
      const uint32_t COld = m_mesh.indices[triIdOld*3+2];

      if(A == AOld && B == BOld && C == COld)
      {
        m_materialIds[triIdNew] = m_mesh.matIndices[triIdOld];
        break;
      } 
    }
  }

  std::cout << "IndicesNum   = " << m_mesh.indices.size() << std::endl;
  std::cout << "TrianglesNum = " << m_mesh.TrianglesNum() << std::endl;
  std::cout << "MateriaIdNum = " << m_mesh.matIndices.size() << std::endl;

  return 0;

}