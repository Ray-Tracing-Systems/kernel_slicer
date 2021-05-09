#include "cmesh.h"

#include <cmath>
#include <cfloat>
#include <cstring>
#include <fstream>


std::vector<unsigned int> CreateQuadTriIndices(const int a_sizeX, const int a_sizeY)
{
  std::vector<unsigned int> indicesData(a_sizeY*a_sizeX * 6);
  unsigned int* indexBuf = indicesData.data();

  for (int i = 0; i < a_sizeY; i++)
  {
    for (int j = 0; j < a_sizeX; j++)
    {
      *indexBuf++ = (unsigned int)( (i + 0) * (a_sizeX + 1) + (j + 0) );
      *indexBuf++ = (unsigned int)( (i + 1) * (a_sizeX + 1) + (j + 0) );
      *indexBuf++ = (unsigned int)( (i + 1) * (a_sizeX + 1) + (j + 1) );
      *indexBuf++ = (unsigned int)( (i + 0) * (a_sizeX + 1) + (j + 0) );
      *indexBuf++ = (unsigned int)( (i + 1) * (a_sizeX + 1) + (j + 1) );
      *indexBuf++ = (unsigned int)( (i + 0) * (a_sizeX + 1) + (j + 1) );
    }
  }

  return indicesData;
}

cmesh::SimpleMesh cmesh::CreateQuad(const int a_sizeX, const int a_sizeY, const float a_size)
{
  const int vertNumX = a_sizeX + 1;
  const int vertNumY = a_sizeY + 1;

  const int quadsNum = a_sizeX*a_sizeY;
  const int vertNum  = vertNumX*vertNumY;

  cmesh::SimpleMesh res(vertNum, quadsNum*2*3);

  const float edgeLength  = a_size / float(a_sizeX);
  const float edgeLength2 = sqrtf(2.0f)*edgeLength;

  // put vertices
  //
  const float startX = -0.5f*a_size;
  const float startY = -0.5f*a_size;

  float* vPos4f_f = (float*)res.vPos4f.data();
  float* vNorm4f  = (float*)res.vNorm4f.data();

  for (int y = 0; y < vertNumY; y++)
  {
    const float ypos = startY + float(y)*edgeLength;
    const int offset = y*vertNumX;

    for (int x = 0; x < vertNumX; x++)
    { 
      const float xpos = startX + float(x)*edgeLength;
      const int i      = offset + x;

      vPos4f_f[i * 4 + 0] = xpos;
      vPos4f_f[i * 4 + 1] = ypos;
      vPos4f_f[i * 4 + 2] = 0.0f;
      vPos4f_f[i * 4 + 3] = 1.0f;

      vNorm4f[i * 4 + 0] = 0.0f;
      vNorm4f[i * 4 + 1] = 0.0f;
      vNorm4f[i * 4 + 2] = 1.0f;
      vNorm4f[i * 4 + 3] = 0.0f;

      res.vTexCoord2f[i].x = (xpos - startX) / a_size;
      res.vTexCoord2f[i].y = (ypos - startY) / a_size;
    }
  }
 
  res.indices = CreateQuadTriIndices(a_sizeX, a_sizeY);

  return res;
}

enum GEOM_FLAGS{ HAS_TANGENT    = 1,
    UNUSED2        = 2,
    UNUSED4        = 4,
    HAS_NO_NORMALS = 8};

struct Header
{
    uint64_t fileSizeInBytes;
    uint32_t verticesNum;
    uint32_t indicesNum;
    uint32_t materialsNum;
    uint32_t flags;
};

#if defined(__ANDROID__)
cmesh::SimpleMesh cmesh::LoadMeshFromVSGF(AAssetManager* mgr, const char* a_fileName)
{
  AAsset* asset = AAssetManager_open(mgr, a_fileName, AASSET_MODE_STREAMING);
  if (!asset)
  {
    LOGE("Could not load mesh from \"%s\"!", a_fileName);
    return SimpleMesh();
  }
  assert(asset);

  size_t size = AAsset_getLength(asset);

  assert(size > 0);

  Header vsgf_header;

  AAsset_read(asset, &vsgf_header, sizeof(Header));

  SimpleMesh res(vsgf_header.verticesNum, vsgf_header.indicesNum);

  auto bytesRead = AAsset_read(asset, (char*)res.vPos4f.data(), res.vPos4f.size() * sizeof(float) * 4);
  
  if(!(vsgf_header.flags & HAS_NO_NORMALS))
    bytesRead = AAsset_read(asset, (char*)res.vNorm4f.data(), res.vNorm4f.size() * sizeof(float) * 4);
  else
    memset(res.vNorm4f.data(), 0, res.vNorm4f.size()*sizeof(float)*4);
  
  if(vsgf_header.flags & HAS_TANGENT)
    bytesRead = AAsset_read(asset, (char*)res.vTang4f.data(), res.vTang4f.size() * sizeof(float) * 4);
  else
    memset(res.vTang4f.data(), 0, res.vTang4f.size()*sizeof(float)*4);
  
  bytesRead = AAsset_read(asset, (char*)res.vTexCoord2f.data(), res.vTexCoord2f.size() * sizeof(float) * 2);
  bytesRead = AAsset_read(asset, (char*)res.indices.data(), res.indices.size() * sizeof(uint32_t));
  bytesRead = AAsset_read(asset, (char*)res.matIndices.data(), res.matIndices.size() * sizeof(uint32_t));
  AAsset_close(asset);

  return res;
}

#else

cmesh::SimpleMesh cmesh::LoadMeshFromVSGF(const char* a_fileName)
{
  std::ifstream input(a_fileName, std::ios::binary);
  if(!input.is_open())
    return SimpleMesh();

  Header header;

  input.read((char*)&header, sizeof(Header));
  SimpleMesh res(header.verticesNum, header.indicesNum);

  input.read((char*)res.vPos4f.data(),  res.vPos4f.size()*sizeof(float)*4);
  
  if(!(header.flags & HAS_NO_NORMALS))
    input.read((char*)res.vNorm4f.data(), res.vNorm4f.size()*sizeof(float)*4);
  else
    memset(res.vNorm4f.data(), 0, res.vNorm4f.size()*sizeof(float)*4);            // #TODO: calc at flat normals in this case

  if(header.flags & HAS_TANGENT)
    input.read((char*)res.vTang4f.data(), res.vTang4f.size()*sizeof(float)*4);
  else
    memset(res.vTang4f.data(), 0, res.vTang4f.size()*sizeof(float)*4);

  input.read((char*)res.vTexCoord2f.data(), res.vTexCoord2f.size()*sizeof(float)*2);
  input.read((char*)res.indices.data(),    res.indices.size()*sizeof(unsigned int));
  input.read((char*)res.matIndices.data(), res.matIndices.size()*sizeof(unsigned int));
  input.close();

  return res; 
}

#endif

void cmesh::SaveMeshToVSGF(const char* a_fileName, const SimpleMesh& a_mesh)
{
  std::ofstream output(a_fileName, std::ios::binary);

  Header header;
  header.fileSizeInBytes = sizeof(header) + a_mesh.SizeInBytes();
  header.verticesNum     = a_mesh.VerticesNum();
  header.indicesNum      = a_mesh.IndicesNum();
  header.materialsNum    = a_mesh.matIndices.size();
  header.flags           = 0;

  if(a_mesh.vNorm4f.size() == 0)
    header.flags |= HAS_NO_NORMALS;

  if(a_mesh.vTang4f.size() != 0)
    header.flags |= HAS_TANGENT;

  output.write((char*)&header, sizeof(Header));
  output.write((char*)a_mesh.vPos4f.data(), a_mesh.vPos4f.size() * sizeof(float) * 4);

  if(a_mesh.vNorm4f.size() != 0)
    output.write((char*)a_mesh.vNorm4f.data(), a_mesh.vNorm4f.size() * sizeof(float) * 4);

  if (a_mesh.vTang4f.size() != 0)
    output.write((char*)a_mesh.vTang4f.data(), a_mesh.vTang4f.size() * sizeof(float) * 4);

  output.write((char*)a_mesh.vTexCoord2f.data(), a_mesh.vTexCoord2f.size() * sizeof(float) * 2);
  output.write((char*)a_mesh.indices.data(),     a_mesh.indices.size() * sizeof(unsigned int));
  output.write((char*)a_mesh.matIndices.data(),  a_mesh.matIndices.size() * sizeof(unsigned int));

  output.close();
}

float cmesh::SimpleMesh::GetAvgTriArea() const
{
  long double res = 0.0f;
  for(size_t i = 0; i < TrianglesNum(); i++)
  {
    uint32_t indA = indices[i * 3 + 0];
    uint32_t indB = indices[i * 3 + 1];
    uint32_t indC = indices[i * 3 + 2];

    LiteMath::float3 A = LiteMath::float3(vPos4f[indA].x, vPos4f[indA].y, vPos4f[indA].z);
    LiteMath::float3 B = LiteMath::float3(vPos4f[indB].x, vPos4f[indB].y, vPos4f[indB].z);
    LiteMath::float3 C = LiteMath::float3(vPos4f[indC].x, vPos4f[indC].y, vPos4f[indC].z);

    LiteMath::float3 edge1A = normalize(B - A);
    LiteMath::float3 edge2A = normalize(C - A);

    float area = 0.5f * sqrtf(powf(edge1A.y * edge2A.z - edge1A.z * edge2A.y, 2) +
        powf(edge1A.z * edge2A.x - edge1A.x * edge2A.z, 2) +
        powf(edge1A.x * edge2A.y - edge1A.y * edge2A.x, 2));

    res += area;
  }

  res /= TrianglesNum();

  return float(res);
}

float cmesh::SimpleMesh::GetAvgTriPerimeter() const
{
  long double res = 0.0f;
  for(size_t i = 0; i < TrianglesNum(); i++)
  {
    uint32_t indA = indices[i * 3 + 0];
    uint32_t indB = indices[i * 3 + 1];
    uint32_t indC = indices[i * 3 + 2];

    LiteMath::float3 A = LiteMath::float3(vPos4f[indA].x, vPos4f[indA].y, vPos4f[indA].z);
    LiteMath::float3 B = LiteMath::float3(vPos4f[indB].x, vPos4f[indB].y, vPos4f[indB].z);
    LiteMath::float3 C = LiteMath::float3(vPos4f[indC].x, vPos4f[indC].y, vPos4f[indC].z);

    float edge1len = LiteMath::length(B - A);
    float edge2len = LiteMath::length(C - A);
    float edge3len = LiteMath::length(C - B);

    res += edge1len + edge2len + edge3len;
  }

  res /= TrianglesNum();

  return float(res);
}


void cmesh::SimpleMesh::ApplyMatrix(const LiteMath::float4x4& m)
{
  LiteMath::float4x4 mRot;
  
  mRot = m;
  mRot.set_col(3, float4(0,0,0,1));

  LiteMath::float4* vPos  = vPos4f.data();
  LiteMath::float4* vNorm = vNorm4f.data();
  LiteMath::float4* vTang = vTang4f.data();

  for(size_t i=0; i<VerticesNum(); i++)
  {
    vPos [i] = mul(m,vPos[i]);
    vNorm[i] = LiteMath::normalize(mul(mRot, vNorm[i]));
    vTang[i] = LiteMath::normalize(mul(mRot, vTang[i]));
  }
}


