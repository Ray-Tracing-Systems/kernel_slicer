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




#endif //TEST_BVH_H
