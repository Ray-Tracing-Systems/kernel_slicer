#include <vector>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <omp.h>

#include "{{MainInclude}}"
#if defined(__GNUC__) && !defined(__clang__) && !defined(__aarch64__) && !defined(__fp16)
#define __fp16 _Float16 // ISPC header declares 'float16' as '__fp16' (clang), GCC on x86 names this type '_Float16'
#endif
#include "{{MainISPCFile}}"
## for Decl in ClassDecls  
{% if Decl.InClass and Decl.IsType %}
using {{Decl.Type}} = {{MainClassName}}::{{Decl.Type}}; // for passing this data type to UBO
{% endif %}
## endfor
#include "LiteMath.h"

// structs with 2- and 3-component vectors: these vectors take 16 bytes in ISPC (12 or 8 in C++), so such structs have
// another layout in ISPC and member vectors of them are copied to ISPC field by field
//
template<typename A, typename B>
static inline void ISPCConvertField(const A& a_src, B& a_dst) { std::memcpy((void*)&a_dst, (const void*)&a_src, std::min(sizeof(A), sizeof(B))); }
## for Decl in ConvertersISPC
static inline void ISPCConvert(const {{Decl.Type}}& a_src, ispc::{{Decl.Type}}& a_dst);
static inline void ISPCConvert(const ispc::{{Decl.Type}}& a_src, {{Decl.Type}}& a_dst);
## endfor
## for Decl in ConvertersISPC

static inline void ISPCConvert(const {{Decl.Type}}& a_src, ispc::{{Decl.Type}}& a_dst)
{
  {% for Field in Decl.FieldsISPC %}
  {% if Field.ArraySize > 0 %}
  for(size_t i = 0; i < {{Field.ArraySize}}; i++)
    {% if Field.Convert %}ISPCConvert{% else %}ISPCConvertField{% endif %}(a_src.{{Field.Name}}[i], a_dst.{{Field.Name}}[i]);
  {% else %}
  {% if Field.Convert %}ISPCConvert{% else %}ISPCConvertField{% endif %}(a_src.{{Field.Name}}, a_dst.{{Field.Name}});
  {% endif %}
  {% endfor %}
}

static inline void ISPCConvert(const ispc::{{Decl.Type}}& a_src, {{Decl.Type}}& a_dst)
{
  {% for Field in Decl.FieldsISPC %}
  {% if Field.ArraySize > 0 %}
  for(size_t i = 0; i < {{Field.ArraySize}}; i++)
    {% if Field.Convert %}ISPCConvert{% else %}ISPCConvertField{% endif %}(a_src.{{Field.Name}}[i], a_dst.{{Field.Name}}[i]);
  {% else %}
  {% if Field.Convert %}ISPCConvert{% else %}ISPCConvertField{% endif %}(a_src.{{Field.Name}}, a_dst.{{Field.Name}});
  {% endif %}
  {% endfor %}
}
## endfor

template<typename A, typename B>
static inline void ISPCConvertVector(const std::vector<A>& a_src, std::vector<B>& a_dst)
{
  a_dst.resize(a_src.size());
  for(size_t i = 0; i < a_src.size(); i++)
    ISPCConvert(a_src[i], a_dst[i]);
}

// ISPC header declares its own types for vectors and structures (with the same layout), so buffers are passed to ISPC functions via untyped pointer
//
struct ISPCPtr 
{ 
  ISPCPtr(const void* a_ptr) : ptr(const_cast<void*>(a_ptr)) {}
  template<typename T> operator T*() const { return (T*)ptr; } 
  void* ptr; 
};

// Image2D and Sampler are passed to ISPC as data pointer and arrays of uint (see LiteMathAux/LiteImageISPC.h)
//
template<typename T>
static inline std::array<uint32_t,4> ISPCImageInfo(const T& a_image) { return { a_image.width(), a_image.height(), a_image.pitch(), uint32_t(a_image.layout()) | (a_image.binShift() << 8) }; }
/// copy of std::vector<float2> or std::vector<float3> with elements of 16 bytes, as ISPC 'float<2>' and 'float<3>'
template<typename T>
static inline std::vector< std::array<float,4> > ISPCPadded16(const std::vector<T>& a_data)
{
  constexpr size_t N = sizeof(T)/sizeof(float);
  std::vector< std::array<float,4> > res(a_data.size());
  for(size_t i = 0; i < a_data.size(); i++)
  {
    res[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::memcpy(res[i].data(), (const void*)&a_data[i], N*sizeof(float));
  }
  return res;
}
static constexpr uint32_t KGEN_FLAG_RETURN            = 1; // thread flags of RTV wavefront
static constexpr uint32_t KGEN_FLAG_BREAK             = 2;
static constexpr uint32_t KGEN_FLAG_DONT_SET_EXIT     = 4;
static constexpr uint32_t KGEN_FLAG_SET_EXIT_NEGATIVE = 8;

/// elements of 4 bytes, from vector of 'float2', 'uint3', ... to padded vector with elements of 16 bytes and back
template<typename T>
static inline void ISPCPadTo16(const std::vector<T>& a_data, std::vector< std::array<uint32_t,4> >& a_padded)
{
  constexpr size_t N = sizeof(T)/sizeof(uint32_t);
  a_padded.resize(a_data.size());
  for(size_t i = 0; i < a_data.size(); i++)
    std::memcpy(a_padded[i].data(), (const void*)&a_data[i], N*sizeof(uint32_t));
}
template<typename T>
static inline void ISPCUnpadFrom16(const std::vector< std::array<uint32_t,4> >& a_padded, std::vector<T>& a_data)
{
  constexpr size_t N = sizeof(T)/sizeof(uint32_t);
  for(size_t i = 0; i < a_data.size() && i < a_padded.size(); i++)
    std::memcpy((void*)&a_data[i], a_padded[i].data(), N*sizeof(uint32_t));
}

/// descriptions of textures of 'std::vector<std::shared_ptr<ICombinedImageSampler>>' or 'std::vector<const ICombinedImageSampler*>' for ISPC
template<typename Vector, typename Desc>
static inline void ISPCTextureDescs(const Vector& a_textures, std::vector<Desc>& a_descs) // 'Desc' is 'ispc::LiteTextureDesc'
{
  a_descs.resize(a_textures.size());
  for(size_t i = 0; i < a_textures.size(); i++)
  {
    const auto* pTex = &(*a_textures[i]); // ICombinedImageSampler
    const auto sampler = pTex->sampler();
    Desc desc = {};
    desc.data     = const_cast<void*>(pTex->data());
    desc.width    = pTex->width();
    desc.height   = pTex->height();
    desc.pitch    = pTex->width(); // LINEAR layout
    desc.layout   = 0;
    switch(pTex->format())
    {
      case 37:          desc.format = 0; break; // VK_FORMAT_R8G8B8A8_UNORM: Image2D<uint32_t>
      case 43:          desc.format = 4; break; // VK_FORMAT_R8G8B8A8_SRGB: Image2D<uint32_t>, texels are converted to linear values
      case 109:         desc.format = 1; break; // VK_FORMAT_R32G32B32A32_SFLOAT: Image2D<float4>
      case 100:         desc.format = 2; break; // VK_FORMAT_R32_SFLOAT: Image2D<float>
      default:          desc.format = 3; break;
    };
    desc.addressU = uint32_t(sampler.addressU);
    desc.addressV = uint32_t(sampler.addressV);
    desc.filter   = uint32_t(sampler.filter);
    a_descs[i]    = desc;
  }
}

template<typename S>
static inline std::array<uint32_t,4> ISPCSamplerInfo(const S& a_sampler) { return { uint32_t(a_sampler.addressU), uint32_t(a_sampler.addressV), uint32_t(a_sampler.filter), 0u }; }
{% if UseAccelStruct %}

#ifdef USE_EMBREE4
#include "EmbreeRT4.h"
using EmbreeImpl = EmbreeRT4;
#else
#include "EmbreeRT.h"
using EmbreeImpl = EmbreeRT;
#endif

// ISPC kernels call Embree directly, so acceleration structure must be implemented via Embree
//
static EmbreeImpl* GetEmbreeImpl(ISceneObject* a_pScene, const char* a_name)
{
  auto pImpl = dynamic_cast<EmbreeImpl*>(a_pScene->UnderlyingImpl(0));
  if(pImpl == nullptr)
  {
    std::cout << "[{{MainClassName}}_ISPC]: '" << a_name << "' is not an Embree implementation, can't trace rays with ISPC" << std::endl;
    exit(0);
  }
  return pImpl;
}
{% endif %}

struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data
{
  {% for Field in UBO.UBOStructFields %}
  {% if Field.IsDummy %} 
  uint {{Field.Name}}; 
  {% else %}
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %}; 
  {% endif %}
  {% endfor %}
  uint dummy_last;
};

class {{MainClassName}}_ISPC : public {{MainClassName}}{% if HasGraphics %}, public LiteRS::DrawBackend{% endif %}

{
public:

  {% for ctorDecl in Constructors %}
  {% if ctorDecl.NumParams == 0 %}
  {{ctorDecl.ClassName}}_ISPC() {% if HasGraphics %}{ SetDrawBackends(); }{% else %}{}{% endif %}

  {% else %}
  {{ctorDecl.ClassName}}_ISPC({{ctorDecl.Params}}) : {{ctorDecl.ClassName}}({{ctorDecl.PrevCall}}) {% if HasGraphics %}{ SetDrawBackends(); }{% else %}{}{% endif %}

  {% endif %}
  {% endfor %}
  ~{{MainClassName}}_ISPC(){}

  {% if HasCommitDeviceFunc %}
  void CommitDeviceData() override 
  {
    UpdatePlainMembers();
  }  
  {% endif %}
  {% if HasGetTimeFunc %}
  void GetExecutionTime(const char* a_funcName, float a_out[4]); 
  {% endif %}

  {% for MainFunc in MainFunctions %}
  {{MainFunc.ReturnType}} {{MainFunc.DeclOrig}} override;
  {% if MainFunc.CallsBlockExpansion %}
  {{MainFunc.ReturnType}} {{MainFunc.DeclBE}}; // copy of '{{MainFunc.Name}}' that calls kernels of this class (block expansion, not virtual)
  {% endif %}
  {% endfor %}

protected:

  virtual void UpdatePlainMembers();
  virtual void ReadPlainMembers();
  {% if HasGraphics %}

  // graphics pipeline (LiteRS): draw commands of control functions call 'Draw' of 'LiteRS::Pipeline::backend'
  //
  void SetDrawBackends()
  {
    {% for Pipe in GPPipelinesISPC %}
    {{Pipe}}.backend = this;
    {% endfor %}
  }

  bool Draw(LiteRS::Pipeline& a_pipe, const LiteRS::ShaderKey& a_vs, const LiteRS::ShaderKey& a_ps, const uint32_t* a_indices, uint32_t a_num, const void* a_args) override
  {
    {% for Draw in GPDrawsISPC %}
    {% if not Draw.Instanced %}
    if(&a_pipe == &{{Draw.Pipeline}} && (a_indices != nullptr) == {{Draw.Indexed}} && a_vs == LiteRS::MakeShaderKey(&{{MainClassName}}_ISPC::{{Draw.VSOriginal}}) &&
       a_ps == {% if Draw.HasPS %}LiteRS::MakeShaderKey(&{{MainClassName}}_ISPC::{{Draw.PSOriginal}}){% else %}LiteRS::ShaderKey(){% endif %})
    {
      {{Draw.Name}}_ISPC(a_pipe, a_indices, a_num, 1, *static_cast<const std::tuple<{% for Arg in Draw.UserArgs %}{{Arg.Type}}{% if not loop.is_last %}, {% endif %}{% endfor %}>*>(a_args));
      return true;
    }
    {% endif %}
    {% endfor %}
    return false; // unknown shaders: C++ implementation of LiteRS
  }

  bool DrawInstanced(LiteRS::Pipeline& a_pipe, const LiteRS::ShaderKey& a_vs, const LiteRS::ShaderKey& a_ps, const uint32_t* a_indices, uint32_t a_num, uint32_t a_instNum,
                     const void* a_args) override
  {
    {% for Draw in GPDrawsISPC %}
    {% if Draw.Instanced %}
    if(&a_pipe == &{{Draw.Pipeline}} && (a_indices != nullptr) == {{Draw.Indexed}} && a_vs == LiteRS::MakeShaderKey(&{{MainClassName}}_ISPC::{{Draw.VSOriginal}}) &&
       a_ps == {% if Draw.HasPS %}LiteRS::MakeShaderKey(&{{MainClassName}}_ISPC::{{Draw.PSOriginal}}){% else %}LiteRS::ShaderKey(){% endif %})
    {
      {{Draw.Name}}_ISPC(a_pipe, a_indices, a_num, a_instNum, *static_cast<const std::tuple<{% for Arg in Draw.UserArgs %}{{Arg.Type}}{% if not loop.is_last %}, {% endif %}{% endfor %}>*>(a_args));
      return true;
    }
    {% endif %}
    {% endfor %}
    return false; // unknown shaders: C++ implementation of LiteRS
  }
  {% for Draw in GPDrawsISPC %}

  void {{Draw.Name}}_ISPC(LiteRS::Pipeline& a_pipe, const uint32_t* a_indices, uint32_t a_num, uint32_t a_instNum, const std::tuple<{% for Arg in Draw.UserArgs %}{{Arg.Type}}{% if not loop.is_last %}, {% endif %}{% endfor %}>& a_args)
  {
    using V = LiteRS::detail::FloatArray<{{Draw.NumFloats}}>;
    static_assert(sizeof(ispc::{{Draw.Name}}_Tri) == sizeof(LiteRS::detail::Triangle<V>), "different layout of triangle in C++ and ISPC");
    [[maybe_unused]] const auto& args = a_args;
    {% for Arg in Draw.UserArgs %}
    [[maybe_unused]] const auto& {{Arg.Name}} = std::get<{{loop.index}}>(a_args); // for '_Bind_ISPC'
    {% endfor %}
    {% for Kernel in Kernels %}
    {% if Kernel.Name == Draw.VSOriginal or Kernel.Name == Draw.PSOriginal %}
    {% for Arg in Kernel.Args %}
    {% if Arg.IsPointer and Arg.IsMember and (Arg.Type == "float2" or Arg.Type == "float3") %}
    const auto {{Arg.Name}}_padded = ISPCPadded16({{Arg.Name}}); // ISPC float2 and float3 take 16 bytes
    {% endif %}
    {% endfor %}
    ispc::{{Kernel.Name}}_Bind_ISPC({% include "inc_host_bind_args.cpp" %}); // class data for '{{Kernel.Name}}'
    {% endif %}
    {% endfor %}

    // vertex shader: vertex 'i' of instance 'k' of the batch is stored in verts[k*range + i - first]
    //
    {% if Draw.Instanced %}
    const uint32_t num = a_num - a_num % 3; // as in Vulkan, an incomplete last triangle is ignored
    if(num == 0 || a_instNum == 0)
      return;
    uint32_t first = 0, last = num - 1;
    {% else %}
    const uint32_t num = a_num;
    if(num == 0)
      return;
    uint32_t first = 0, last = num - 1;
    {% endif %}
    if(a_indices != nullptr)
      LiteRS::detail::IndexRange(a_indices, num, first, last);
    const uint32_t range  = last - first + 1;
    const uint32_t stride = uint32_t(sizeof(LiteRS::Vertex<V>)/sizeof(float));
    const LiteRS::detail::Target tgt = LiteRS::detail::MakeTarget(a_pipe);
    const int32_t state[12] = { int32_t(tgt.layout), int32_t(tgt.pitch), tgt.depthTest ? 1 : 0, tgt.depthWrite ? 1 : 0, int32_t(tgt.depthCompare), int32_t(tgt.blendMode),
                                int32_t(a_pipe.cpu.algo), a_pipe.cpu.smallTriArea, int32_t(tgt.binShift), 0, 0, 0 }; // see RS_STATE_* in raster_ispc.h
    const uint32_t batch = {% if Draw.Instanced %}LiteRS::detail::InstancesPerBatch(range){% else %}1{% endif %}; // instances processed together
    std::vector< LiteRS::Vertex<V> > verts;
    std::vector<uint32_t> batchIndices;
    for(uint32_t instFirst = 0; instFirst < a_instNum; instFirst += batch)
    {
      const uint32_t instNum = std::min(batch, a_instNum - instFirst);
      const uint32_t vertNum = range*instNum;
      LiteRS::detail::StageTimer vsTimer(a_pipe.cpu.collectStats);
      verts.resize(vertNum);
      constexpr uint32_t CHUNK = 4096;
      #pragma omp parallel for if(a_pipe.cpu.multithreaded && vertNum >= 16*CHUNK) // waking threads up costs about a millisecond
      for(int chunk = 0; chunk < int((vertNum + CHUNK - 1)/CHUNK); chunk++)
      {
        const uint32_t start = uint32_t(chunk)*CHUNK;
        {% if Draw.Instanced %}
        ispc::{{Draw.Name}}_VS_ISPC(first, range, instFirst, start, std::min(CHUNK, vertNum - start),{% for Arg in Draw.UserArgs %} &std::get<{{loop.index}}>(args),{% endfor %} reinterpret_cast<float*>(verts.data() + start), stride);
        {% else %}
        ispc::{{Draw.Name}}_VS_ISPC(first + start, std::min(CHUNK, vertNum - start),{% for Arg in Draw.UserArgs %} &std::get<{{loop.index}}>(args),{% endfor %} reinterpret_cast<float*>(verts.data() + start), stride);
        {% endif %}
      }
      a_pipe.stats.vertexMs += vsTimer.Ms();

      // primitive assembly and setup in C++, rasterization of bins in ISPC
      //
      const uint32_t* indices   = a_indices;
      uint32_t        indexNum  = num;
      uint32_t        indexBase = first;
      {% if Draw.Instanced %}
      indexNum  = num*instNum;
      indexBase = 0;
      if(a_indices != nullptr)
      {
        LiteRS::detail::InstancedIndices(a_indices, num, first, range, instNum, a_pipe.cpu.multithreaded, batchIndices);
        indices = batchIndices.data();
      }
      {% endif %}
      LiteRS::detail::DrawVertexStreamByBins(a_pipe, verts.data(), indices, indexNum, indexBase,
        [&](const LiteRS::detail::Triangle<V>* a_tris, const uint32_t* a_ids, uint32_t a_triNum, const LiteRS::detail::Rect& a_bin)
      {
        const int32_t bin[4] = { a_bin.x0, a_bin.y0, a_bin.x1, a_bin.y1 };
        ispc::{{Draw.Name}}_Raster_ISPC(reinterpret_cast<const ispc::{{Draw.Name}}_Tri*>(a_tris), a_ids, a_triNum, bin, state, tgt.depth{% for Out in Draw.Outputs %}, ISPCPtr(tgt.{% if Out.IsUint %}colorU{% else %}colorF{% endif %}[{{loop.index}}]){% endfor %});
      });
    }
  }
  {% endfor %}
  {% endif %}

  {% for Kernel in Kernels %}
  {% if Kernel.IsRTV and not Kernel.IsMega %}
  // RTV kernel for wavefront: all threads of chunk [kgen_start, kgen_end) (rows for 2D), payload arrays of the chunk
  //
  void {{Kernel.Name}}_Wave(int kgen_start, int kgen_end, int kgen_sizeX, uint32_t* kgen_threadFlags, uint32_t kgen_tFlagsMask{% for Arg in Kernel.Args %}{% if not Arg.IsUBO and not Arg.IsMember and not Arg.SkipISPC %}, {{Arg.HostTypeISPC}} {{Arg.Name}}{% endif %}{% endfor %})
  {
    {% for Arg in Kernel.Args %}
    {% if Arg.IsAccelStruct %}
    auto {{Arg.Name}}_impl = GetEmbreeImpl({{Arg.Name}}.get(), "{{Arg.Name}}");
    {% endif %}
    {% endfor %}
    ispc::{{Kernel.Name}}_ISPC({% include "inc_host_args.cpp" %}, kgen_threadFlags, kgen_tFlagsMask, kgen_start, kgen_end, kgen_sizeX);
  }

  void {{Kernel.Name}}_Bind() // class data for member functions, the same for all chunks
  {
    {% for Arg in Kernel.Args %}
    {% if Arg.IsAccelStruct %}
    auto {{Arg.Name}}_impl = GetEmbreeImpl({{Arg.Name}}.get(), "{{Arg.Name}}");
    {% else if not Arg.IsUBO and not Arg.IsMember and not Arg.SkipISPC %}
    [[maybe_unused]] {% if Arg.IsPointer %}ISPCPtr {{Arg.Name}}(nullptr){% else %}{{Arg.Type}} {{Arg.Name}}{}{% endif %};
    {% endif %}
    {% endfor %}
    ispc::{{Kernel.Name}}_Bind_ISPC({% include "inc_host_bind_args.cpp" %});
  }

  {% else if not Kernel.IsGraphics %}
  void {{Kernel.Name}}({% for Arg in Kernel.Args %}{% if not Arg.IsUBO and not Arg.IsMember %}{% if Arg.IsPointer %}{{Arg.Type}}* {{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %}, {% endif %}{% else %}{{Arg.Type}} {{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %}, {% endif %} {% endif %}{% endif %} {% endfor %}) {% if not Kernel.InitKPass and not Kernel.EnableBlockExpansion and not Kernel.IsMega and Kernel.IsVirtual %} override {% endif %}
  { 
    {% for Arg in Kernel.Args %}
    {% if Arg.IsAccelStruct %}
    auto {{Arg.Name}}_impl = GetEmbreeImpl({{Arg.Name}}.get(), "{{Arg.Name}}");
    {% endif %}
    {% endfor %}
    ispc::{{Kernel.Name}}_Bind_ISPC({% include "inc_host_bind_args.cpp" %});
    {% if Kernel.IsMega %}
    const int kgen_sizeX = {% if Kernel.threadDim == 2 %}int(tidX){% else %}1{% endif %};
    const int kgen_rows  = int({% if Kernel.threadDim == 2 %}tidY{% else %}{{Kernel.ThreadId0.Name}}{% endif %});
    const int kgen_rowsPerChunk = std::max(1, {{MegaChunkISPC}}/kgen_sizeX); // '-ispc_mega_chunk'
    #pragma omp parallel for schedule(dynamic, 1)
    for(int kgen_c = 0; kgen_c < (kgen_rows + kgen_rowsPerChunk - 1)/kgen_rowsPerChunk; kgen_c++)
      ispc::{{Kernel.Name}}_ISPC({% include "inc_host_args.cpp" %}, kgen_c*kgen_rowsPerChunk, std::min((kgen_c + 1)*kgen_rowsPerChunk, kgen_rows), kgen_sizeX);
    {% else if Kernel.OpenMPAndISPC %}
    constexpr int BLOCK_SIZE = 64;
    [[maybe_unused]] auto* ubo = &m_uboData; // loop bounds may refer to class members as 'ubo[0].m_member'
    {% if Kernel.HasInitISPC %}
    ispc::{{Kernel.Name}}_Init_ISPC({% include "inc_host_args.cpp" %});
    {% endif %}
    {% if Kernel.EnableBlockExpansion %}
    const int blocks = int({% if Kernel.threadDim == 1 %}{% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}(({{Kernel.ThreadId0.Size}}) - ({{Kernel.ThreadId0.Start}})){% endif %}{% else %}{% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}({% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}(({{Kernel.ThreadId0.Size}}) - ({{Kernel.ThreadId0.Start}})){% endif %}){% endif %}*{% if Kernel.ThreadId1.Start == "0" %}({{Kernel.ThreadId1.Size}}){% else %}({% if Kernel.ThreadId1.Start == "0" %}({{Kernel.ThreadId1.Size}}){% else %}(({{Kernel.ThreadId1.Size}}) - ({{Kernel.ThreadId1.Start}})){% endif %}){% endif %}{% endif %}); // block expansion: blocks are numbered linearly
    #pragma omp parallel for 
    for(int block = 0; block < blocks; block += BLOCK_SIZE)
      ispc::{{Kernel.Name}}_ISPC({% include "inc_host_args.cpp" %},block,std::min(block + BLOCK_SIZE, blocks),0);
    {% else if Kernel.threadDim == 1 %}
    #pragma omp parallel for 
    for(int {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}}; {{Kernel.ThreadId0.Name}} < {{Kernel.ThreadId0.Size}}; {{Kernel.ThreadId0.Name}} += BLOCK_SIZE) {
      const int end = std::min({{Kernel.ThreadId0.Name}} + BLOCK_SIZE, int({{Kernel.ThreadId0.Size}}));
      ispc::{{Kernel.Name}}_ISPC({% include "inc_host_args.cpp" %},{{Kernel.ThreadId0.Name}},end,0);  
    }
    {% else %}
    #pragma omp parallel for 
    for(int {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}}; {{Kernel.ThreadId0.Name}} < {{Kernel.ThreadId0.Size}}; {{Kernel.ThreadId0.Name}}++) {
      for(int {{Kernel.ThreadId1.Name}} = {{Kernel.ThreadId1.Start}}; {{Kernel.ThreadId1.Name}} < {{Kernel.ThreadId1.Size}}; {{Kernel.ThreadId1.Name}} += BLOCK_SIZE) {
        const int end = std::min({{Kernel.ThreadId1.Name}} + BLOCK_SIZE, int({{Kernel.ThreadId1.Size}}));
        ispc::{{Kernel.Name}}_ISPC({% include "inc_host_args.cpp" %},{{Kernel.ThreadId1.Name}},end,{{Kernel.ThreadId0.Name}});  
      }
    }
    {% endif %}
    {% if Kernel.HasFinishISPC %}
    ispc::{{Kernel.Name}}_Finish_ISPC({% include "inc_host_args.cpp" %});
    {% endif %}
    {% else %}
    ispc::{{Kernel.Name}}_ISPC({% include "inc_host_args.cpp" %}); 
    {% endif %}
    {% if HasGraphics %}
    ReadPlainMembers(); // LiteRS draws of control function are executed by C++ code, counts of draws may read data written by the kernel
    {% endif %}
  }
  {% endif %} {# /* not Kernel.IsGraphics */ #}
  {% endfor %}

  {% for MainFunc in MainFunctions %}
  {% if MainFunc.IsWaveISPC %}
  struct {{MainFunc.Name}}_Payload // local variables of '{{MainFunc.Name}}' for a chunk of threads, elements have the size of ISPC types
  {
    {% for Var in MainFunc.PayloadISPC %}
    std::vector<uint8_t>  {{Var.Name}};
    {% endfor %}
    std::vector<uint32_t> threadFlags;
    void Resize(size_t a_size)
    {
      if(threadFlags.size() == a_size)
        return;
      uint32_t sizes[{{length(MainFunc.PayloadISPC)}} + 1] = {};
      ispc::{{MainFunc.Name}}_PayloadSizes_ISPC(sizes);
      {% for Var in MainFunc.PayloadISPC %}
      {{Var.Name}}.resize(a_size*sizes[{{loop.index}}]);
      {% endfor %}
      threadFlags.resize(a_size);
    }
  };
  std::vector<{{MainFunc.Name}}_Payload> m_{{MainFunc.Name}}Payload; // for each OpenMP thread

  {% endif %}
  {% endfor %}
  {% for TexArray in TexArraysISPC %}
  std::vector<ispc::LiteTextureDesc> m_texDesc_{{TexArray}}; // textures of '{{TexArray}}' for ISPC
  {% endfor %}
  {% for Buff in PaddedBuffersISPC %}
  std::vector< std::array<uint32_t,4> > m_padded_{{Buff}}; // copy of '{{Buff}}' with elements of 16 bytes, as ISPC 2- and 3-component vectors
  {% endfor %}
  {% for Buff in ConvertedBuffersISPC %}
  std::vector<ispc::{{Buff.Type}}> m_conv_{{Buff.Name}}; // copy of '{{Buff.Name}}' in ISPC layout
  {% endfor %}
  ispc::{{MainClassName}}{{MainClassSuffix}}_UBO_Data m_uboData;
  std::unordered_map<std::string, float> m_exTimeISPC;
};

void {{MainClassName}}_ISPC::UpdatePlainMembers()
{
  const size_t maxAllowedSize = std::numeric_limits<uint32_t>::max();
  {% for Buff in PaddedBuffersISPC %}
  ISPCPadTo16({{Buff}}, m_padded_{{Buff}});
  {% endfor %}
  {% for Buff in ConvertedBuffersISPC %}
  ISPCConvertVector({{Buff.Name}}, m_conv_{{Buff.Name}});
  {% endfor %}
  {% for TexArray in TexArraysISPC %}
  ISPCTextureDescs({{TexArray}}, m_texDesc_{{TexArray}});
  {% endfor %}
## for Var in ClassVars
  {% if Var.IsArray %}
  memcpy((void*)m_uboData.{{Var.Name}}, (const void*){{Var.Name}}, sizeof({{Var.Name}}));
  {% else if Var.ConvertedISPC %}
  ISPCConvert({{Var.Name}}, m_uboData.{{Var.Name}});
  {% else %}
  static_assert(sizeof(m_uboData.{{Var.Name}}) >= sizeof({{Var.Name}}), "'{{Var.Name}}' has different layout in ISPC");
  memcpy((void*)&m_uboData.{{Var.Name}}, (const void*)&{{Var.Name}}, sizeof({{Var.Name}})); // LiteMath and ISPC types are different, ISPC float3 is 16 bytes
  {% endif %}
## endfor
## for Var in ClassVectorVars 
  m_uboData.{{Var.Name}}_size     = uint32_t( {{Var.Name}}{{Var.AccessSymb}}size() );     assert( {{Var.Name}}{{Var.AccessSymb}}size()     < maxAllowedSize );
  m_uboData.{{Var.Name}}_capacity = uint32_t( {{Var.Name}}{{Var.AccessSymb}}capacity() ); assert( {{Var.Name}}{{Var.AccessSymb}}capacity() < maxAllowedSize );
## endfor
}

void {{MainClassName}}_ISPC::ReadPlainMembers()
{
  {% for Buff in PaddedBuffersISPC %}
  ISPCUnpadFrom16(m_padded_{{Buff}}, {{Buff}});
  {% endfor %}
  {% for Buff in ConvertedBuffersISPC %}
  ISPCConvertVector(m_conv_{{Buff.Name}}, {{Buff.Name}});
  {% endfor %}
  {% for Var in ClassVars %}
  {% if not Var.IsConst %}
  {% if Var.IsArray %}
  memcpy((void*){{Var.Name}}, (const void*)m_uboData.{{Var.Name}}, sizeof({{Var.Name}})); // void*: GCC -Wclass-memaccess for LiteMath types
  {% else if Var.ConvertedISPC %}
  ISPCConvert(m_uboData.{{Var.Name}}, {{Var.Name}});
  {% else %}
  memcpy((void*)&{{Var.Name}}, (const void*)&m_uboData.{{Var.Name}}, sizeof({{Var.Name}}));
  {% endif %}
  {% endif %} {#/* end of if not var.IsConst */#}
  {% endfor %}
  {% for Var in ClassVectorVars %}
  {{Var.Name}}{{Var.AccessSymb}}resize(m_uboData.{{Var.Name}}_size);
  {% endfor %}
}

{% for MainFunc in MainFunctions %}
{{MainFunc.ReturnType}} {{MainClassName}}_ISPC::{{MainFunc.DeclOrig}}
{
  auto before = std::chrono::high_resolution_clock::now();
  UpdatePlainMembers();
  {% if HasGraphics %}
  SetDrawBackends(); // pipeline objects could be reassigned after constructor
  {% endif %}
  {% if MainFunc.IsMega %}
  for(uint32_t pass = 0; pass < a_numPasses; pass++) // RTV control function joined to megakernel
    {{MainFunc.Name}}Mega({% for Arg in MainFunc.InOutVars %}{{Arg.Name}}{% if loop.index1 != MainFunc.InOutVarsNum %}, {% endif %}{% endfor %});
  {% else if MainFunc.IsWaveISPC %}
  // RTV wavefront: control function is executed for chunks of threads, kernels process all threads of a chunk one by one;
  // payload of a chunk is small enough to stay in cache ('-ispc_wave_chunk')
  //
  {% for Kernel in Kernels %}
  {% if Kernel.IsRTV and not Kernel.IsMega %}
  {{Kernel.Name}}_Bind();
  {% endif %}
  {% endfor %}
  const int kgen_sizeX = {% if length(MainFunc.ThreadIdsISPC) == 2 %}int(tidX){% else %}1{% endif %};
  const int kgen_rows  = int({% if length(MainFunc.ThreadIdsISPC) == 2 %}tidY{% else %}{{MainFunc.ThreadIdsISPC.0}}{% endif %});
  const int kgen_rowsPerChunk = std::max(1, {{WaveChunkISPC}}/kgen_sizeX);
  const int kgen_chunks = (kgen_rows + kgen_rowsPerChunk - 1)/kgen_rowsPerChunk;
  m_{{MainFunc.Name}}Payload.resize(omp_get_max_threads());
  for(uint32_t pass = 0; pass < a_numPasses; pass++)
  {
    #pragma omp parallel for schedule(dynamic, 1)
    for(int kgen_c = 0; kgen_c < kgen_chunks; kgen_c++)
    {
      const int kgen_start = kgen_c*kgen_rowsPerChunk;
      const int kgen_end   = std::min(kgen_start + kgen_rowsPerChunk, kgen_rows);
      auto& kgen_payload   = m_{{MainFunc.Name}}Payload[omp_get_thread_num()];
      kgen_payload.Resize(size_t(kgen_rowsPerChunk)*size_t(kgen_sizeX));
      std::fill(kgen_payload.threadFlags.begin(), kgen_payload.threadFlags.end(), 0u); // all threads of the chunk are active
      [[maybe_unused]] const uint32_t outOfForFlags  = KGEN_FLAG_RETURN;
      [[maybe_unused]] const uint32_t inForFlags     = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK;
      [[maybe_unused]] const uint32_t outOfForFlagsN = KGEN_FLAG_RETURN | KGEN_FLAG_SET_EXIT_NEGATIVE;
      [[maybe_unused]] const uint32_t inForFlagsN    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_SET_EXIT_NEGATIVE;
      [[maybe_unused]] const uint32_t outOfForFlagsD = KGEN_FLAG_RETURN | KGEN_FLAG_DONT_SET_EXIT;
      [[maybe_unused]] const uint32_t inForFlagsD    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_DONT_SET_EXIT;
      {{MainFunc.MainFuncTextCmd}}
    }
  }
  {% else %}
  {% if MainFunc.CallsBlockExpansion %}{{MainFunc.Name}}_BE{% else %}{{MainClassName}}::{{MainFunc.Name}}{% endif %}({% for Arg in MainFunc.InOutVars %}{{Arg.Name}}{% if loop.index1 != MainFunc.InOutVarsNum %},{% endif %}{% endfor %});
  {% endif %}
  ReadPlainMembers();
  m_exTimeISPC["{{MainFunc.Name}}"] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}
{% if MainFunc.CallsBlockExpansion %}

// block expansion kernels (templates) and kernels which are not virtual can't be overriden, so this copy of '{{MainFunc.Name}}' calls them in '{{MainClassName}}_ISPC'
//
{{MainFunc.ReturnType}} {{MainClassName}}_ISPC::{{MainFunc.DeclBE}}
{
  {{MainFunc.MainFuncTextCmd}}
}
{% endif %}
{% endfor %}
{% if HasGetTimeFunc %}

void {{MainClassName}}_ISPC::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  float res = 0.0f;
  {% for MainFunc in MainFunctions %}
  if(std::string(a_funcName) == "{{MainFunc.Name}}" || std::string(a_funcName) == "{{MainFunc.Name}}Block")
    res = m_exTimeISPC["{{MainFunc.Name}}"];
  {% endfor %}
  a_out[0] = res;
  a_out[1] = 0.0f;
  a_out[2] = 0.0f;
  a_out[3] = 0.0f;             
}
{% endif %}

{% for ctorDecl in Constructors %}
{% if ctorDecl.NumParams == 0 %}
std::shared_ptr<{{MainClassName}}> Create{{MainClassName}}_ISPC() 
{ 
  auto pObj = std::make_shared<{{MainClassName}}_ISPC>(); 
  return pObj;
}
{% else %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}_ISPC({{ctorDecl.Params}}) 
{ 
  auto pObj = std::make_shared<{{MainClassName}}_ISPC>({{ctorDecl.PrevCall}}); 
  return pObj;
}
{% endif %}
{% endfor %}