#ifndef TEST_BVH_H
#define TEST_BVH_H


struct BVHNode
{
  float3 boxMin;
  uint   leftOffset;
  float3 boxMax;
  uint   escapeIndex;
};

static inline bool IsLeaf (const struct BVHNode* a_node) { return (a_node->leftOffset == 0xFFFFFFFF); }
static inline bool IsEmpty(const struct BVHNode* a_node) { return (a_node->leftOffset == 0xFFFFFFFD); }
static inline bool IsValid(const struct BVHNode* a_node) { return (a_node->leftOffset <  0xFFFFFFFD); }

struct Interval
{
  uint start;
  uint count;
};
/*
struct BVHTree
{
  std::vector<struct BVHNode>  nodes;
  std::vector<struct Interval> intervals;
  std::vector<uint>     indicesReordered;
  std::vector<struct Interval> depthRanges;
  unsigned int          geomID;
  int                   format;
  unsigned int          leavesNumber = 0;
};*/



#endif //TEST_BVH_H
