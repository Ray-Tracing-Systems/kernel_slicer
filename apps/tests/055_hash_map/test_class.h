#pragma once

#include <cstdint>
#include <unordered_map>
#include "LiteMath.h"
using LiteMath::uint;

struct ItemData   // value type of the second hash table
{
  float    val;
  uint32_t id;
};

class HashMapTest
{
public:
  HashMapTest() { }

  // GPU code can not allocate memory, so hash tables capacity must be reserved before CommitDeviceData()
  //
  virtual void Reserve(uint32_t a_maxUniqueKeys, uint32_t a_maxItems);

  // a_keys: arbitrary (sparse, repeating) keys, counted in m_hist
  // a_itemKeys, a_itemVals: unique keys and their values, stored in m_items
  //
  virtual void Build(const int* a_keys     [[size("a_keysNum")]], uint32_t a_keysNum,
                     const int* a_itemKeys [[size("a_itemsNum")]],
                     const float* a_itemVals [[size("a_itemsNum")]], uint32_t a_itemsNum);

  // for each query: a_outCount[i]  = number of a_queries[i] in a_keys (0 if absent), via find(...)
  //                 a_outCount2[i] = the same, via operator[]
  //                 a_outVal[i]    = value of item with key a_queries[i] (-1.0f if absent), via at(...)
  //                 a_outId[i]     = index of item with key a_queries[i] (0xFFFFFFFF if absent), via operator[]
  //
  virtual void Lookup(const int* a_queries [[size("a_queryNum")]], uint32_t a_queryNum,
                      uint32_t* a_outCount  [[size("a_queryNum")]],
                      uint32_t* a_outCount2 [[size("a_queryNum")]],
                      float*    a_outVal    [[size("a_queryNum")]],
                      uint32_t* a_outId     [[size("a_queryNum")]]);

  // a_outTicket[i] = m_tickets[a_keys[i]]++ (old value), a_outNew[i] = (m_tickets2[a_keys[i]] += 2) (new value);
  // for every key the set of tickets is 0..n-1 and the set of new values is 2,4,..,2n in some order
  //
  virtual void Tickets(const int* a_keys [[size("a_keysNum")]], uint32_t a_keysNum,
                       uint32_t* a_outTicket [[size("a_keysNum")]], uint32_t* a_outNew [[size("a_keysNum")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:

  virtual void kernel1D_CountKeys(const int* a_keys, uint32_t a_keysNum);
  virtual void kernel1D_InsertItems(const int* a_itemKeys, const float* a_itemVals, uint32_t a_itemsNum);
  virtual void kernel1D_Tickets(const int* a_keys, uint32_t a_keysNum, uint32_t* a_outTicket, uint32_t* a_outNew);
  virtual void kernel1D_Lookup(const int* a_queries, uint32_t a_queryNum, uint32_t* a_outCount, uint32_t* a_outCount2, float* a_outVal, uint32_t* a_outId);

  std::unordered_map<int, uint32_t> m_hist;
  std::unordered_map<int, ItemData> m_items;
  std::unordered_map<int, uint32_t> m_tickets;  // value of 'm[k]++' is used: atomic add which returns old value
  std::unordered_map<int, uint32_t> m_tickets2; // value of 'm[k] += v' is used: returns new value
};
