#ifndef TEST_BVH_H
#define TEST_BVH_H


struct BVHDataHeader
{
  uint64_t node_length;
  uint64_t indices_length;
  uint64_t depth_length;
  uint64_t geom_id;
};

struct BVHNode
{
  BVHNode(){}
  BVHNode(float3 in_boxMin, float3 in_boxMax, uint in_leftOffset, uint in_escapeIndex) :
          boxMin(in_boxMin), boxMax(in_boxMax),
          leftOffset(in_leftOffset), escapeIndex(in_escapeIndex){}

  float3 boxMin;
  uint   leftOffset;
  float3 boxMax;
  uint   escapeIndex;

  bool operator==(const BVHNode& a_rhs) const
  {
    if(leftOffset != a_rhs.leftOffset || escapeIndex != a_rhs.escapeIndex)
      return false;

    const float diff1 = LiteMath::lengthSquare(boxMin - a_rhs.boxMin);
    const float diff2 = LiteMath::lengthSquare(boxMax - a_rhs.boxMax);

    return (diff1 < 1e-10f) && (diff2 < 1e-10f);
  }
};

static std::ostream& operator<<(std::ostream& os, const BVHNode& dt)
{
  os << dt.boxMin.x << "\t" << dt.boxMin.y << "\t" << dt.boxMin.z << "\t|\t" << dt.leftOffset << std::endl;
  os << dt.boxMax.x << "\t" << dt.boxMax.y << "\t" << dt.boxMax.z << "\t|\t" << dt.escapeIndex << std::endl;
  return os;
}

static inline bool IsLeaf (const BVHNode& a_node) { return (a_node.leftOffset == 0xFFFFFFFF); }
static inline bool IsEmpty(const BVHNode& a_node) { return (a_node.leftOffset == 0xFFFFFFFD); }
static inline bool IsValid(const BVHNode& a_node) { return (a_node.leftOffset <  0xFFFFFFFD); }

struct Interval
{
  Interval() : start(0), count(0) {}
  Interval(uint a, uint b) : start(a), count(b) {}

  uint start;
  uint count;
};

struct BVHTree
{
  std::vector<BVHNode>  nodes;
  std::vector<Interval> intervals;
  std::vector<uint>     indicesReordered;
  std::vector<Interval> depthRanges;
  unsigned int          geomID;

  int                   format;
  unsigned int          leavesNumber = 0;
};

struct VSGFHeader
{
  uint64_t fileSizeInBytes;
  uint32_t verticesNum;
  uint32_t indicesNum;
  uint32_t materialsNum;
  uint32_t flags;
};

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

  //enum SIMPLE_MESH_TOPOLOGY {SIMPLE_MESH_TRIANGLES = 0, SIMPLE_MESH_QUADS = 1};
  //SIMPLE_MESH_TOPOLOGY topology = SIMPLE_MESH_TRIANGLES;
 // LiteMath::Box4f GetAABB() const;

//  float GetAvgTriArea() const;
//  float GetAvgTriPerimeter() const;
//
//  void ApplyMatrix(const LiteMath::float4x4& m);

  std::vector<LiteMath::float4> vPos4f;      //
  std::vector<LiteMath::float4> vNorm4f;     //
  std::vector<LiteMath::float4> vTang4f;     //
  std::vector<float2>                       vTexCoord2f; //
  std::vector<unsigned int>                 indices;     // size = 3*TrianglesNum() for triangle mesh, 4*TrianglesNum() for quad mesh
  std::vector<unsigned int>                 matIndices;  // size = 1*TrianglesNum()
};


#endif //TEST_BVH_H
