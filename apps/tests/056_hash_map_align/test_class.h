#pragma once

#include <cstdint>
#include <unordered_map>
#include "LiteMath.h"
using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::uint;

struct Particle   // 32 bytes, aligned at 16 in shaders
{
  float4 pos;
  float4 vel;
};

// hash map values of different size and alignment; slot {X val; Key key;} needs padding in std430 for some of them
//
class HashMapAlign
{
public:
  HashMapAlign() { }

  virtual void Reserve(uint32_t a_maxKeys);

  // a_keys: unique keys; a_dupKeys: repeating keys from a_keys
  //
  virtual void Build(const uint32_t* a_keys    [[size("a_keysNum")]],    uint32_t a_keysNum,
                     const uint32_t* a_dupKeys [[size("a_dupKeysNum")]], uint32_t a_dupKeysNum);

  // a_out: 4 float4 per query
  //
  virtual void Lookup(const uint32_t* a_queries [[size("a_queryNum")]], uint32_t a_queryNum, float4* a_out [[size("a_queryNum*4")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:

  virtual void kernel1D_Fill(const uint32_t* a_keys, uint32_t a_keysNum);
  virtual void kernel1D_Accum(const uint32_t* a_dupKeys, uint32_t a_dupKeysNum);
  virtual void kernel1D_Lookup(const uint32_t* a_queries, uint32_t a_queryNum, float4* a_out);

  float GetSum(uint32_t a_key) const; // helper function: access hash map not from kernel directly

  std::unordered_map<uint32_t, float2>   m_f2;   // slot 16 bytes: 8 + 4 + 4 padding
  std::unordered_map<uint32_t, float3>   m_f3;   // slot 16 bytes: 12 + 4
  std::unordered_map<uint32_t, float4>   m_f4;   // slot 32 bytes: 16 + 4 + 12 padding
  std::unordered_map<uint32_t, Particle> m_part; // slot 48 bytes: 32 + 4 + 12 padding
  std::unordered_map<uint32_t, float>    m_sum;  // float atomic add
  std::unordered_map<uint32_t, uint32_t> m_cnt;  // ++
  std::unordered_map<uint32_t, int>      m_neg;  // -=
};

static inline float2   ValueF2(uint32_t a_key) { return float2(float(a_key % 1000), 1.0f); }
static inline float3   ValueF3(uint32_t a_key) { return float3(float(a_key % 1000), 2.0f, 3.0f); }
static inline float4   ValueF4(uint32_t a_key) { return float4(float(a_key % 1000), 4.0f, 5.0f, 6.0f); }
static inline Particle ValuePart(uint32_t a_key)
{
  Particle part;
  part.pos = float4(float(a_key % 1000), 7.0f, 8.0f, 9.0f);
  part.vel = float4(10.0f, 11.0f, 12.0f, float(a_key % 777));
  return part;
}
