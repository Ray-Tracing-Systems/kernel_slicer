#include "test_class.h"

#include "LiteMath.h"
using LiteMath::InterlockedAdd;

void HashMapTest::Reserve(uint32_t a_maxUniqueKeys, uint32_t a_maxItems)
{
  m_hist.reserve(a_maxUniqueKeys);
  m_items.reserve(a_maxItems);
}

void HashMapTest::Build(const int* a_keys, uint32_t a_keysNum, const int* a_itemKeys, const float* a_itemVals, uint32_t a_itemsNum)
{
  kernel1D_CountKeys(a_keys, a_keysNum);
  kernel1D_InsertItems(a_itemKeys, a_itemVals, a_itemsNum);
}

void HashMapTest::Lookup(const int* a_queries, uint32_t a_queryNum, uint32_t* a_outCount, uint32_t* a_outCount2, float* a_outVal, uint32_t* a_outId)
{
  kernel1D_Lookup(a_queries, a_queryNum, a_outCount, a_outCount2, a_outVal, a_outId);
}


void HashMapTest::kernel1D_CountKeys(const int* a_keys, uint32_t a_keysNum)
{
  for(uint32_t i=0; i < a_keysNum; i++)
  {
    const int key = a_keys[i];
    m_hist[key] += 1;
  }
}

void HashMapTest::kernel1D_InsertItems(const int* a_itemKeys, const float* a_itemVals, uint32_t a_itemsNum)
{
  for(uint32_t i=0; i < a_itemsNum; i++)
  {
    ItemData item;
    item.val = a_itemVals[i];
    item.id  = i;
    m_items[a_itemKeys[i]] = item;
  }
}

void HashMapTest::kernel1D_Lookup(const int* a_queries, uint32_t a_queryNum, uint32_t* a_outCount, uint32_t* a_outCount2, float* a_outVal, uint32_t* a_outId)
{
  for(uint32_t i=0; i < a_queryNum; i++)
  {
    const int key = a_queries[i];

    const auto pHist = m_hist.find(key);
    a_outCount[i]    = (pHist != m_hist.end()) ? pHist->second : 0u;

    // operator[] inserts absent key, so read with it only present keys
    //
    if(m_hist.count(key) != 0)
      a_outCount2[i] = m_hist[key];
    else
      a_outCount2[i] = 0u;

    if(m_items.count(key) != 0)
    {
      a_outVal[i] = m_items.at(key).val;
      a_outId[i]  = m_items[key].id;
    }
    else
    {
      a_outVal[i] = -1.0f;
      a_outId[i]  = 0xFFFFFFFF;
    }
  }
}
