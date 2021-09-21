#ifndef CMESH_GEOM_H
#define CMESH_GEOM_H

#include <vector>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <cassert>


namespace cmesh
{
  // very simple utility mesh representation for working with geometry on the CPU in C++
  //
  struct SimpleMesh
  {
    static const uint64_t POINTS_IN_TRIANGLE = 3;
    SimpleMesh() = default;
    SimpleMesh(uint32_t a_vertNum, uint32_t a_indNum) { Resize(a_vertNum, a_indNum); }

    inline size_t VerticesNum()  const { return vPos4f.size() / 4; }
    inline size_t IndicesNum()   const { return indices.size();  }
    inline size_t TrianglesNum() const { return IndicesNum() / POINTS_IN_TRIANGLE;  }
    inline void   Resize(uint32_t a_vertNum, uint32_t a_indNum)
    {
      vPos4f.resize(a_vertNum * 4);
      vNorm4f.resize(a_vertNum * 4);
      vTang4f.resize(a_vertNum * 4);
      vTexCoord2f.resize(a_vertNum * 2);
      indices.resize(a_indNum);
      matIndices.resize(a_indNum/3); 
      assert(a_indNum % 3 == 0); // PLEASE NOTE THAT CURRENT IMPLEMENTATION ASSUMES ONLY TRIANGLE MESHES!
    };

    inline size_t SizeInBytes() const
    {
      return vPos4f.size()      * sizeof(float) +
             vNorm4f.size()     * sizeof(float) +
             vTang4f.size()     * sizeof(float) +
             vTexCoord2f.size() * sizeof(float) +
             indices.size()     * sizeof(int)   +
             matIndices.size()  * sizeof(int);
    }

    std::vector<float> vPos4f;      // positions (x, y, z, w)
    std::vector<float> vNorm4f;     // normals (x, y, z, w)
    std::vector<float> vTang4f;     // tangents (x, y, z, w)
    std::vector<float> vTexCoord2f; // texCoords (u, v)
    std::vector<unsigned int> indices;     // size = 3 * TrianglesNum()
    std::vector<unsigned int> matIndices;  // size = 1 * TrianglesNum()
  };

  SimpleMesh LoadMeshFromVSGF(const char* a_fileName);
  void       SaveMeshToVSGF  (const char* a_fileName, const SimpleMesh& a_mesh);

  SimpleMesh CreateQuad(const int a_sizeX, const int a_sizeY, const float a_size);
};


#endif // CMESH_GEOM_H
