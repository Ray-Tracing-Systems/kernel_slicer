#pragma once
#include <vector>
#include <cstdint>
#include "LiteMath.h"
#include "Image2d.h"
using LiteMath::float2;
using LiteMath::float4;
using LiteImage::Image2D;
using LiteImage::Sampler;

struct MaterialData
{
  float4   color;
  float    rough;
  uint32_t texId;
  float2   pad;
};

// Checks '-compact_ds 1': read-only member vectors of the same type are joined to a single buffer in shaders
// ('m_fA[i]' ==> 'kgen_data1f[ubo[0].m_fA_offset + (i)]'), so that descriptor sets have less bindings.
// Read-only means that a vector is not listed in "nonConstantData" (kmake.json) of the kernels which use it.
// Covers: direct access, '.size()', an empty vector, passing 'm_vec.data()' and 'm_vec.data() + k' to functions,
// a local pointer to a member vector, access from a member function, a member vector passed to a kernel by pointer
// (bound with offset inside of the joined buffer), memcpy from a joined vector, IPV and RTV kernels (also megakernel);
// texture 'm_texA' is bound after joined buffers: bindings of descriptor set layouts must be the same as in shaders.
// 'kmake_append.json' does the same with CommitAppendedData(): a joined buffer is inside of a single append group.
// Variants '*_types' use '-compact_ds 2': vectors of 'float', 'int' and 'uint' are joined to one buffer of 'uint' (views of other types).
class TestCompactDS
{
public:
  TestCompactDS(){}

  void InitData(int a_iter);

  virtual void Run(uint32_t a_size, float* a_out [[size("a_size")]]);
  virtual void RunRTVBlock(uint32_t a_size, float* a_out, uint32_t a_passesNum);
  void RunRTV(uint32_t tid, float* a_out [[size("tid")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void CommitAppendedData() { CommitDeviceData(); }                // will be overriden in generated class if "appendGroups" is used
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_Write(uint32_t a_size);
  void kernel1D_Main(uint32_t a_size, float* a_out);
  void kernel1D_FromArg(uint32_t a_size, const float* a_in, uint32_t a_inSize, float* a_out);

  void kernel_Eval (uint32_t tid, float* a_val);
  void kernel_Store(uint32_t tid, const float* a_val, float* a_out);

  float GetColor(uint32_t a_id) const;

  std::vector<float>        m_fA, m_fB, m_fC, m_fEmpty; // read-only, joined to 'kgen_data1f'
  std::vector<uint32_t>     m_uA, m_uB;                 // read-only, joined to 'kgen_data1ui'
  std::vector<float4>       m_colA, m_colB;             // read-only, joined to 'kgen_data4f'
  std::vector<int>          m_iA;                       // the only 'int' vector: own binding
  std::vector<MaterialData> m_materials;                // the only vector of 'MaterialData': own binding
  std::vector<float>        m_written;                  // written by kernel1D_Write: own binding
  std::vector<float>        m_copy;                     // written by memcpy: own binding
  Image2D<float4>           m_texA;
  Sampler                   m_sampler;
};
