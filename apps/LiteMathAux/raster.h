#pragma once
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LiteRS: minimal graphics pipeline (vertex shader + pixel shader) for kernel_slicer.
//
// This header is the CPU reference implementation. kernel_slicer replaces LiteRS commands (Clear, DrawTriangles,
// DrawIndexedTriangles) in control functions with a Vulkan graphics pipeline or with a call of the ISPC rasterizer.
// Pipeline state and render targets are set up in ordinary host functions through LiteRS::Pipeline objects.
//
// Shaders are const member functions of the main class, their names start with 'kernelVS_' and 'kernelPS_':
//   float4 kernelVS_Name(uint32_t a_vertexId, Args... a_args, Varyings* a_out) const; // returns clip space position
//   float4 kernelVS_Name(uint32_t a_vertexId, Args... a_args) const;                  // depth only pass, no pixel shader
//   float4 kernelPS_Name(const Varyings& a_in) const;                                 // returns color of target 0
//   Output kernelPS_Name(const Varyings& a_in) const;                                 // struct of float4: color of targets 0,1,...
// Varyings are a struct of float types (float, float2, float3, float4), all fields are interpolated with perspective
// correction. The vertex shader reads vertex data from members of the class by a_vertexId (vertex pulling).
//
// Conventions are the same as in Vulkan:
//   * clip space: -w <= x,y <= w, 0 <= z <= w; NDC y = -1 is the top row of the image; LiteMath (OpenGL style)
//     projection matrices are multiplied by LiteMath::OpenglToVulkanProjectionMatrixFix();
//   * pixel (x,y) has its center at (x+0.5, y+0.5); top-left fill rule; 8 bits of sub-pixel precision;
//   * front face is defined as VkFrontFace (COUNTER_CLOCKWISE by default), depth range is [minDepth, maxDepth];
//   * RGBA8 color target stores R in the lowest byte (VK_FORMAT_R8G8B8A8_UNORM).
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cmath>
#include <cassert>
#ifndef KERNEL_SLICER
#include <vector>
#include <tuple>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <cstring>
#include <chrono>
#include <memory>
#endif

#include "LiteMath.h"
#include "Image2d.h"

namespace LiteRS
{
  using LiteMath::float4;
  using LiteImage::Image2D;

  constexpr uint32_t MAX_COLOR_TARGETS = 4;

  enum class CullMode   : uint32_t { NONE = 0, FRONT = 1, BACK = 2 };                         ///< as VkCullModeFlagBits
  enum class FrontFace  : uint32_t { COUNTER_CLOCKWISE = 0, CLOCKWISE = 1 };                  ///< as VkFrontFace
  enum class CompareOp  : uint32_t { NEVER = 0, LESS = 1, EQUAL = 2, LESS_OR_EQUAL = 3,
                                     GREATER = 4, NOT_EQUAL = 5, GREATER_OR_EQUAL = 6, ALWAYS = 7 }; ///< as VkCompareOp
  enum class BlendMode  : uint32_t { NONE = 0, ALPHA = 1, ADD = 2 };
  enum class RasterAlgo : uint32_t { AUTO = 0, PIXEL = 1, BLOCK = 2 };

  struct Viewport ///< as VkViewport
  {
    float x        = 0.0f;
    float y        = 0.0f;
    float width    = 0.0f;
    float height   = 0.0f;
    float minDepth = 0.0f;
    float maxDepth = 1.0f;
  };

  /**
  \brief Settings of the CPU rasterizer (C++ and ISPC), GPU back ends ignore them.
  */
  struct CPUSettings
  {
    RasterAlgo algo          = RasterAlgo::AUTO;
    int        smallTriArea  = 64;   ///< AUTO: triangles with a bounding box smaller than this (in pixels) are rasterized per pixel, the others by 4x4 blocks
    int        binSize       = 64;   ///< the screen is split into bins of binSize x binSize pixels which are rasterized in parallel
    bool       multithreaded = true;
    bool       collectStats  = false; ///< measure time of stages of draws to Pipeline::stats
  };

  /**
  \brief Time of stages of draws (CPU rasterizer, C++ and ISPC), accumulated while CPUSettings::collectStats is true.
  */
  struct CPUStats
  {
    double   vertexMs   = 0.0; ///< vertex shader
    double   setupMs    = 0.0; ///< primitive assembly, clipping, setup and binning
    double   rasterMs   = 0.0; ///< rasterization, depth test, pixel shader and writing colors
    uint64_t triangles  = 0;   ///< triangles after setup
    uint32_t draws      = 0;
  };

  struct DrawBackend;
  namespace detail { struct BinningCache; };

  /**
  \brief Fixed function state and render targets of the graphics pipeline. Shaders are not stored here: they are given
         in every draw call, so it is always visible in a control function which shaders are used.
         All targets must have the same size and the same layout (LINEAR or TILED4x4).
  */
  struct Pipeline
  {
    void SetColorTarget(Image2D<float4>&   a_image, uint32_t a_index = 0) { assert(a_index < MAX_COLOR_TARGETS); colorF[a_index] = &a_image; colorU[a_index] = nullptr; }
    void SetColorTarget(Image2D<uint32_t>& a_image, uint32_t a_index = 0) { assert(a_index < MAX_COLOR_TARGETS); colorU[a_index] = &a_image; colorF[a_index] = nullptr; }
    void SetDepthTarget(Image2D<float>&    a_image)                       { depth = &a_image; }
    void SetViewport(int a_x, int a_y, int a_width, int a_height)
    {
      viewport.x      = float(a_x);
      viewport.y      = float(a_y);
      viewport.width  = float(a_width);
      viewport.height = float(a_height);
    }

    Viewport    viewport;
    CullMode    cullMode     = CullMode::NONE;
    FrontFace   frontFace    = FrontFace::COUNTER_CLOCKWISE;
    bool        depthTest    = true;    ///< as in Vulkan, depth is written only when the depth test is enabled
    bool        depthWrite   = true;
    CompareOp   depthCompare = CompareOp::LESS;
    BlendMode   blendMode    = BlendMode::NONE;
    float4      clearColor   = float4(0.0f, 0.0f, 0.0f, 0.0f); ///< for all color targets
    float       clearDepth   = 1.0f;
    CPUSettings cpu;
    CPUStats    stats;

    Image2D<float4>*   colorF[MAX_COLOR_TARGETS] = {}; ///< RGBA32F color targets
    Image2D<uint32_t>* colorU[MAX_COLOR_TARGETS] = {}; ///< RGBA8 color targets
    Image2D<float>*    depth = nullptr;                ///< D32 depth target
    DrawBackend*       backend = nullptr;              ///< set by generated code (ISPC): it executes draws with known shaders, see DrawBackend
#ifndef KERNEL_SLICER
    std::shared_ptr<detail::BinningCache> binningCache; ///< buffers of detail::DrawVertexStreamByBins reused by draws
#endif
  };

  /**
  \brief Vertex after vertex shader: clip space position and attributes (varyings) for pixel shader.
  */
  template<typename V>
  struct Vertex
  {
    float4 pos;
    V      attr;
  };

#ifdef KERNEL_SLICER

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // kernel_slicer sees only declarations of commands: their calls in control functions are replaced by generated code.
  // Documentation of the commands is in the implementation below.
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void Clear(Pipeline& a_pipe);

  template<typename C, typename VSRet, typename... VSArgs, typename PSRet, typename... PSArgs, typename... Args>
  void DrawTriangles(Pipeline& a_pipe, C* a_self, VSRet (C::*a_vs)(VSArgs...) const, PSRet (C::*a_ps)(PSArgs...) const, uint32_t a_vertNum, Args... a_args);

  template<typename C, typename VSRet, typename... VSArgs, typename... Args>
  void DrawTriangles(Pipeline& a_pipe, C* a_self, VSRet (C::*a_vs)(VSArgs...) const, uint32_t a_vertNum, Args... a_args);

  template<typename C, typename VSRet, typename... VSArgs, typename PSRet, typename... PSArgs, typename... Args>
  void DrawIndexedTriangles(Pipeline& a_pipe, C* a_self, VSRet (C::*a_vs)(VSArgs...) const, PSRet (C::*a_ps)(PSArgs...) const,
                            const uint32_t* a_indices, uint32_t a_indexNum, Args... a_args);

  template<typename C, typename VSRet, typename... VSArgs, typename... Args>
  void DrawIndexedTriangles(Pipeline& a_pipe, C* a_self, VSRet (C::*a_vs)(VSArgs...) const, const uint32_t* a_indices, uint32_t a_indexNum, Args... a_args);

#else

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////// implementation details ////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  namespace detail
  {
    constexpr int   SUBPIXEL_BITS = 8;
    constexpr int   SUBPIXEL_ONE  = 1 << SUBPIXEL_BITS;
    constexpr int   BLOCK_SIZE    = 4;     ///< equal to the tile size of LiteImage::Layout::TILED4x4
    constexpr float GUARD_BAND    = 64.0f; ///< triangles are clipped by x and y only outside of [-GUARD_BAND*w, GUARD_BAND*w]

    struct Rect { int x0, y0, x1, y1; };   ///< [x0, x1) x [y0, y1)

    /// milliseconds since construction if a_enabled (CPUSettings::collectStats), 0 otherwise
    struct StageTimer
    {
      explicit StageTimer(bool a_enabled) : enabled(a_enabled) { if(enabled) start = std::chrono::high_resolution_clock::now(); }
      double Ms() const { return enabled ? std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() : 0.0; }
      bool enabled;
      std::chrono::high_resolution_clock::time_point start;
    };

    struct NoVaryings { float unused; };   ///< varyings of a depth only pass

    static inline Rect Intersect(const Rect& a, const Rect& b)
    {
      return Rect{ std::max(a.x0, b.x0), std::max(a.y0, b.y0), std::min(a.x1, b.x1), std::min(a.y1, b.y1) };
    }

    static inline int64_t FloorDiv(int64_t a, int64_t b) { return (a >= 0) ? a / b : -((-a + b - 1) / b); }

    /**
    \brief Render targets and state of one draw call in the form used by rasterizer.
    */
    struct Target
    {
      float4*           colorF[MAX_COLOR_TARGETS] = {};
      uint32_t*         colorU[MAX_COLOR_TARGETS] = {};
      float*            depth  = nullptr;
      int               width  = 0;
      int               height = 0;
      LiteImage::Layout layout = LiteImage::Layout::LINEAR;
      unsigned int      pitch  = 0;
      unsigned int      binShift = 6;        ///< log2 of bin size of LiteImage::Layout::TILED4x4_BINS
      Rect              scissor;             ///< viewport rectangle clamped to the target size
      float             vpScale[3];          ///< screen = ndc*vpScale + vpOffset
      float             vpOffset[3];
      CullMode          cullMode;
      FrontFace         frontFace;
      bool              depthTest;
      bool              depthWrite;
      CompareOp         depthCompare;
      BlendMode         blendMode;

      int Offset(int x, int y) const { return int(LiteImage::PixelOffset(layout, pitch, unsigned(x), unsigned(y), binShift)); }

      /// bins of rasterizer: bins of TILED4x4_BINS targets (every thread works with its own continuous part of memory), a_settings.binSize otherwise
      int BinSize(const CPUSettings& a_settings) const { return (layout == LiteImage::Layout::TILED4x4_BINS) ? (1 << binShift) : std::max(a_settings.binSize, BLOCK_SIZE); }
    };

    template<typename T>
    static inline void AddTarget(Target& t, Image2D<T>* a_image)
    {
      if(a_image == nullptr)
        return;
      if(t.width == 0)
      {
        t.width  = int(a_image->width());
        t.height = int(a_image->height());
        t.layout   = a_image->layout();
        t.pitch    = a_image->pitch();
        t.binShift = a_image->binShift();
      }
      assert(int(a_image->width()) == t.width && int(a_image->height()) == t.height && a_image->layout() == t.layout && a_image->binShift() == t.binShift);
    }

    static inline Target MakeTarget(const Pipeline& a_pipe)
    {
      Target t;
      for(uint32_t i = 0; i < MAX_COLOR_TARGETS; i++)
      {
        AddTarget(t, a_pipe.colorF[i]);
        AddTarget(t, a_pipe.colorU[i]);
        t.colorF[i] = (a_pipe.colorF[i] != nullptr) ? a_pipe.colorF[i]->data() : nullptr;
        t.colorU[i] = (a_pipe.colorU[i] != nullptr) ? a_pipe.colorU[i]->data() : nullptr;
      }
      AddTarget(t, a_pipe.depth);
      t.depth = (a_pipe.depth != nullptr) ? a_pipe.depth->data() : nullptr;

      const Viewport& vp = a_pipe.viewport;
      t.vpScale[0]  = 0.5f*vp.width;
      t.vpScale[1]  = 0.5f*vp.height;
      t.vpScale[2]  = vp.maxDepth - vp.minDepth;
      t.vpOffset[0] = vp.x + 0.5f*vp.width;
      t.vpOffset[1] = vp.y + 0.5f*vp.height;
      t.vpOffset[2] = vp.minDepth;

      const Rect vpRect = Rect{ int(std::floor(vp.x)), int(std::floor(vp.y)), int(std::ceil(vp.x + vp.width)), int(std::ceil(vp.y + vp.height)) };
      t.scissor      = Intersect(vpRect, Rect{0, 0, t.width, t.height});
      t.cullMode     = a_pipe.cullMode;
      t.frontFace    = a_pipe.frontFace;
      t.depthTest    = a_pipe.depthTest && (t.depth != nullptr);
      t.depthWrite   = a_pipe.depthWrite;
      t.depthCompare = a_pipe.depthCompare;
      t.blendMode    = a_pipe.blendMode;
      return t;
    }

    /**
    \brief Triangle after clipping, viewport transform and setup.
    Edge function i is E_i(P) = A[i]*P.x + B[i]*P.y + C[i] (P in sub-pixel units), it is positive inside the triangle
    and E_i/area is the screen space barycentric coordinate of vertex i.
    */
    template<typename V>
    struct Triangle
    {
      int64_t A[3], B[3], C[3];
      int64_t bias[3];   ///< top-left fill rule: pixel is inside if E_i + bias[i] >= 0 for all i
      float   invArea;
      float   z[3];      ///< screen space depth
      float   invW[3];
      Rect    bbox;      ///< pixel bounding box clamped to the scissor rectangle
      V       attr[3];
    };

    template<typename V>
    static inline Vertex<V> Lerp(const Vertex<V>& a, const Vertex<V>& b, float t)
    {
      constexpr int N = int(sizeof(V)/sizeof(float));
      Vertex<V> res;
      res.pos = a.pos + (b.pos - a.pos)*t;
      const float* pa = reinterpret_cast<const float*>(&a.attr);
      const float* pb = reinterpret_cast<const float*>(&b.attr);
      float*       pr = reinterpret_cast<float*>(&res.attr);
      for(int k = 0; k < N; k++)
        pr[k] = pa[k] + (pb[k] - pa[k])*t;
      return res;
    }

    /// distance of a clip space point to the clipping plane 'a_plane', the point is inside if it is >= 0
    static inline float PlaneDist(const float4& p, int a_plane)
    {
      switch(a_plane)
      {
        case 0:  return p.z;                    // near: z >= 0
        case 1:  return p.w - p.z;              // far:  z <= w
        case 2:  return GUARD_BAND*p.w + p.x;
        case 3:  return GUARD_BAND*p.w - p.x;
        case 4:  return GUARD_BAND*p.w + p.y;
        default: return GUARD_BAND*p.w - p.y;
      };
    }

    /// bit i is set if the point is outside of the view volume plane i: -w <= x,y <= w, 0 <= z <= w
    static inline uint32_t OutCode(const float4& p)
    {
      return (p.x < -p.w ? 1u : 0u) | (p.x > p.w ? 2u : 0u) | (p.y < -p.w ? 4u : 0u) | (p.y > p.w ? 8u : 0u) |
             (p.z < 0.0f ? 16u : 0u) | (p.z > p.w ? 32u : 0u);
    }

    static inline bool NeedClipping(const float4& p)
    {
      for(int plane = 0; plane < 6; plane++)
        if(PlaneDist(p, plane) < 0.0f)
          return true;
      return false;
    }

    /// Sutherland-Hodgman clipping of a triangle by near, far and guard band planes; returns the number of polygon vertices
    template<typename V>
    static int ClipTriangle(const Vertex<V>& a, const Vertex<V>& b, const Vertex<V>& c, Vertex<V> a_out[9])
    {
      Vertex<V> buff[9];
      Vertex<V>* src = a_out;
      Vertex<V>* dst = buff;
      src[0] = a; src[1] = b; src[2] = c;
      int num = 3;
      for(int plane = 0; plane < 6 && num > 0; plane++)
      {
        int outNum = 0;
        for(int i = 0; i < num; i++)
        {
          const Vertex<V>& p0 = src[i];
          const Vertex<V>& p1 = src[(i + 1) % num];
          const float d0 = PlaneDist(p0.pos, plane);
          const float d1 = PlaneDist(p1.pos, plane);
          if(d0 >= 0.0f)
            dst[outNum++] = p0;
          if((d0 >= 0.0f) != (d1 >= 0.0f))
            dst[outNum++] = Lerp(p0, p1, d0/(d0 - d1));
        }
        std::swap(src, dst);
        num = outNum;
      }
      if(src != a_out)
        for(int i = 0; i < num; i++)
          a_out[i] = src[i];
      return (num >= 3) ? num : 0;
    }

    /// viewport transform, culling and edge function setup; returns false if the triangle is culled, degenerate or out of scissor
    template<typename V>
    static bool SetupTriangle(const Vertex<V>& a_v0, const Vertex<V>& a_v1, const Vertex<V>& a_v2, const Target& a_tgt, Triangle<V>& t)
    {
      const Vertex<V>* v[3] = {&a_v0, &a_v1, &a_v2};
      int64_t X[3], Y[3];
      float   z[3], invW[3];
      for(int i = 0; i < 3; i++)
      {
        invW[i]          = 1.0f/v[i]->pos.w;
        const float sx   = v[i]->pos.x*invW[i]*a_tgt.vpScale[0] + a_tgt.vpOffset[0];
        const float sy   = v[i]->pos.y*invW[i]*a_tgt.vpScale[1] + a_tgt.vpOffset[1];
        z[i]             = v[i]->pos.z*invW[i]*a_tgt.vpScale[2] + a_tgt.vpOffset[2];
        X[i]             = int64_t(std::llrint(double(sx)*double(SUBPIXEL_ONE)));
        Y[i]             = int64_t(std::llrint(double(sy)*double(SUBPIXEL_ONE)));
      }

      int64_t area = (X[1] - X[0])*(Y[2] - Y[0]) - (Y[1] - Y[0])*(X[2] - X[0]);
      if(area == 0)
        return false;

      // Vulkan: a = -0.5*sum(x_i*y_(i+1) - x_(i+1)*y_i) = -area/2, COUNTER_CLOCKWISE triangle is front facing if a > 0
      const bool isFront = (a_tgt.frontFace == FrontFace::COUNTER_CLOCKWISE) ? (area < 0) : (area > 0);
      if((a_tgt.cullMode == CullMode::BACK && !isFront) || (a_tgt.cullMode == CullMode::FRONT && isFront))
        return false;

      int idx[3] = {0, 1, 2};
      if(area < 0) // make area positive, then edge functions are positive inside
      {
        std::swap(idx[1], idx[2]);
        area = -area;
      }

      for(int i = 0; i < 3; i++) // edge i is opposite to vertex i: (v[i+1] -> v[i+2])
      {
        const int     ia  = idx[(i + 1) % 3];
        const int     ib  = idx[(i + 2) % 3];
        const int64_t dx  = X[ib] - X[ia];
        const int64_t dy  = Y[ib] - Y[ia];
        t.A[i]    = -dy;
        t.B[i]    = dx;
        t.C[i]    = dy*X[ia] - dx*Y[ia];
        t.bias[i] = (dy < 0 || (dy == 0 && dx > 0)) ? 0 : -1; // top or left edge includes pixels exactly on it
        t.z[i]    = z[idx[i]];
        t.invW[i] = invW[idx[i]];
        t.attr[i] = v[idx[i]]->attr;
      }
      t.invArea = float(1.0/double(area));

      const int64_t minX = std::min(X[0], std::min(X[1], X[2]));
      const int64_t maxX = std::max(X[0], std::max(X[1], X[2]));
      const int64_t minY = std::min(Y[0], std::min(Y[1], Y[2]));
      const int64_t maxY = std::max(Y[0], std::max(Y[1], Y[2]));
      const Rect    bbox = Rect{ int(FloorDiv(minX, SUBPIXEL_ONE)),     int(FloorDiv(minY, SUBPIXEL_ONE)),
                                 int(FloorDiv(maxX, SUBPIXEL_ONE)) + 1, int(FloorDiv(maxY, SUBPIXEL_ONE)) + 1 };
      t.bbox = Intersect(bbox, a_tgt.scissor);
      return (t.bbox.x0 < t.bbox.x1 && t.bbox.y0 < t.bbox.y1);
    }

    /// primitive assembly of one triangle: trivial reject, clipping, setup; appends 0..7 triangles to a_out
    template<typename V>
    static void AssembleTriangle(const Target& a_tgt, const Vertex<V>& a_v0, const Vertex<V>& a_v1, const Vertex<V>& a_v2, std::vector<Triangle<V>>& a_out)
    {
      if((OutCode(a_v0.pos) & OutCode(a_v1.pos) & OutCode(a_v2.pos)) != 0)
        return;

      Triangle<V> tri;
      if(!NeedClipping(a_v0.pos) && !NeedClipping(a_v1.pos) && !NeedClipping(a_v2.pos))
      {
        if(SetupTriangle(a_v0, a_v1, a_v2, a_tgt, tri))
          a_out.push_back(tri);
        return;
      }

      Vertex<V> poly[9];
      const int num = ClipTriangle(a_v0, a_v1, a_v2, poly);
      for(int i = 1; i + 1 < num; i++)
        if(SetupTriangle(poly[0], poly[i], poly[i + 1], a_tgt, tri))
          a_out.push_back(tri);
    }

    /// edge function values at the center of pixel (x,y)
    template<typename V>
    static inline void EdgesAt(const Triangle<V>& t, int x, int y, int64_t e[3])
    {
      const int64_t px = int64_t(x)*SUBPIXEL_ONE + SUBPIXEL_ONE/2;
      const int64_t py = int64_t(y)*SUBPIXEL_ONE + SUBPIXEL_ONE/2;
      for(int i = 0; i < 3; i++)
        e[i] = t.A[i]*px + t.B[i]*py + t.C[i];
    }

    /// per pixel half-space rasterizer, calls a_frag(x, y, b0, b1, b2) for every covered pixel of rectangle a_rect
    template<typename V, typename FragFn>
    static void RasterizePixels(const Triangle<V>& t, const Rect& a_rect, FragFn& a_frag)
    {
      int64_t row[3];
      EdgesAt(t, a_rect.x0, a_rect.y0, row);
      const int64_t stepX[3] = { t.A[0]*SUBPIXEL_ONE, t.A[1]*SUBPIXEL_ONE, t.A[2]*SUBPIXEL_ONE };
      const int64_t stepY[3] = { t.B[0]*SUBPIXEL_ONE, t.B[1]*SUBPIXEL_ONE, t.B[2]*SUBPIXEL_ONE };
      for(int y = a_rect.y0; y < a_rect.y1; y++)
      {
        int64_t e0 = row[0], e1 = row[1], e2 = row[2];
        for(int x = a_rect.x0; x < a_rect.x1; x++)
        {
          if(((e0 + t.bias[0]) | (e1 + t.bias[1]) | (e2 + t.bias[2])) >= 0)
            a_frag(x, y, float(e0)*t.invArea, float(e1)*t.invArea, float(e2)*t.invArea);
          e0 += stepX[0]; e1 += stepX[1]; e2 += stepX[2];
        }
        row[0] += stepY[0]; row[1] += stepY[1]; row[2] += stepY[2];
      }
    }

    /// block (4x4) half-space rasterizer: empty blocks are skipped by 4 corners, full blocks are processed without edge tests
    template<typename V, typename FragFn>
    static void RasterizeBlocks(const Triangle<V>& t, const Rect& a_rect, FragFn& a_frag)
    {
      constexpr int BS = BLOCK_SIZE;
      const int64_t stepX[3]  = { t.A[0]*SUBPIXEL_ONE, t.A[1]*SUBPIXEL_ONE, t.A[2]*SUBPIXEL_ONE };
      const int64_t stepY[3]  = { t.B[0]*SUBPIXEL_ONE, t.B[1]*SUBPIXEL_ONE, t.B[2]*SUBPIXEL_ONE };
      const int64_t blockX[3] = { stepX[0]*BS, stepX[1]*BS, stepX[2]*BS };
      const int64_t blockY[3] = { stepY[0]*BS, stepY[1]*BS, stepY[2]*BS };

      // offsets from the top left pixel of a block to its corner pixel with minimal/maximal edge function value
      int64_t minOffs[3], maxOffs[3];
      for(int i = 0; i < 3; i++)
      {
        minOffs[i] = std::min<int64_t>(0, stepX[i]*(BS - 1)) + std::min<int64_t>(0, stepY[i]*(BS - 1));
        maxOffs[i] = std::max<int64_t>(0, stepX[i]*(BS - 1)) + std::max<int64_t>(0, stepY[i]*(BS - 1));
      }

      const int bx0 = int(FloorDiv(a_rect.x0, BS))*BS;
      const int by0 = int(FloorDiv(a_rect.y0, BS))*BS;

      int64_t row[3];
      EdgesAt(t, bx0, by0, row);
      for(int by = by0; by < a_rect.y1; by += BS)
      {
        int64_t e[3] = { row[0], row[1], row[2] };
        for(int bx = bx0; bx < a_rect.x1; bx += BS)
        {
          bool empty = false, full = true;
          for(int i = 0; i < 3; i++)
          {
            empty = empty || (e[i] + maxOffs[i] + t.bias[i] < 0);
            full  = full  && (e[i] + minOffs[i] + t.bias[i] >= 0);
          }
          const bool inside = (bx >= a_rect.x0 && by >= a_rect.y0 && bx + BS <= a_rect.x1 && by + BS <= a_rect.y1);

          if(!empty && full && inside)  // all 16 pixels are covered: no edge tests
          {
            int64_t p[3] = { e[0], e[1], e[2] };
            for(int y = by; y < by + BS; y++)
            {
              int64_t q0 = p[0], q1 = p[1], q2 = p[2];
              for(int x = bx; x < bx + BS; x++)
              {
                a_frag(x, y, float(q0)*t.invArea, float(q1)*t.invArea, float(q2)*t.invArea);
                q0 += stepX[0]; q1 += stepX[1]; q2 += stepX[2];
              }
              p[0] += stepY[0]; p[1] += stepY[1]; p[2] += stepY[2];
            }
          }
          else if(!empty)               // partially covered block: coverage mask of 16 pixels
          {
            int64_t p[3] = { e[0], e[1], e[2] };
            for(int y = by; y < by + BS; y++)
            {
              int64_t q0 = p[0], q1 = p[1], q2 = p[2];
              for(int x = bx; x < bx + BS; x++)
              {
                const bool inRect = (x >= a_rect.x0 && x < a_rect.x1 && y >= a_rect.y0 && y < a_rect.y1);
                if(inRect && ((q0 + t.bias[0]) | (q1 + t.bias[1]) | (q2 + t.bias[2])) >= 0)
                  a_frag(x, y, float(q0)*t.invArea, float(q1)*t.invArea, float(q2)*t.invArea);
                q0 += stepX[0]; q1 += stepX[1]; q2 += stepX[2];
              }
              p[0] += stepY[0]; p[1] += stepY[1]; p[2] += stepY[2];
            }
          }
          e[0] += blockX[0]; e[1] += blockX[1]; e[2] += blockX[2];
        }
        row[0] += blockY[0]; row[1] += blockY[1]; row[2] += blockY[2];
      }
    }

    /// rasterizes triangle inside a_clip rectangle choosing the algorithm by triangle size
    template<typename V, typename FragFn>
    static void Rasterize(const Triangle<V>& t, const Rect& a_clip, const CPUSettings& a_settings, FragFn& a_frag)
    {
      const Rect r = Intersect(t.bbox, a_clip);
      if(r.x0 >= r.x1 || r.y0 >= r.y1)
        return;
      const int  bboxArea = (t.bbox.x1 - t.bbox.x0)*(t.bbox.y1 - t.bbox.y0);
      const bool perPixel = (a_settings.algo == RasterAlgo::PIXEL) || (a_settings.algo == RasterAlgo::AUTO && bboxArea < a_settings.smallTriArea);
      if(perPixel)
        RasterizePixels(t, r, a_frag);
      else
        RasterizeBlocks(t, r, a_frag);
    }

    static inline bool Compare(CompareOp op, float a, float b)
    {
      switch(op)
      {
        case CompareOp::NEVER:            return false;
        case CompareOp::LESS:             return a <  b;
        case CompareOp::EQUAL:            return a == b;
        case CompareOp::LESS_OR_EQUAL:    return a <= b;
        case CompareOp::GREATER:          return a >  b;
        case CompareOp::NOT_EQUAL:        return a != b;
        case CompareOp::GREATER_OR_EQUAL: return a >= b;
        default:                          return true;
      };
    }

    /// depth test and depth write; pixel shader can not change depth or discard pixels, so the test is done before it
    static inline bool DepthTest(const Target& a_tgt, int a_offs, float a_z)
    {
      if(!a_tgt.depthTest)
        return true;
      if(!Compare(a_tgt.depthCompare, a_z, a_tgt.depth[a_offs]))
        return false;
      if(a_tgt.depthWrite)
        a_tgt.depth[a_offs] = a_z;
      return true;
    }

    /// perspective correct interpolation of varyings; screen space barycentrics (b0, b1, b2) are used for depth only
    template<typename V>
    static inline void Interpolate(const Triangle<V>& t, float b0, float b1, float b2, V& a_res)
    {
      constexpr int N = int(sizeof(V)/sizeof(float));
      const float q0 = b0*t.invW[0];
      const float q1 = b1*t.invW[1];
      const float q2 = b2*t.invW[2];
      const float s  = 1.0f/(q0 + q1 + q2);
      const float p0 = q0*s, p1 = q1*s, p2 = q2*s;
      const float* a0 = reinterpret_cast<const float*>(&t.attr[0]);
      const float* a1 = reinterpret_cast<const float*>(&t.attr[1]);
      const float* a2 = reinterpret_cast<const float*>(&t.attr[2]);
      float*       r  = reinterpret_cast<float*>(&a_res);
      for(int k = 0; k < N; k++)
        r[k] = p0*a0[k] + p1*a1[k] + p2*a2[k];
    }

    static inline uint32_t PackRGBA8(float4 c)
    {
      const uint32_t r = uint32_t(LiteMath::clamp(c.x, 0.0f, 1.0f)*255.0f + 0.5f);
      const uint32_t g = uint32_t(LiteMath::clamp(c.y, 0.0f, 1.0f)*255.0f + 0.5f);
      const uint32_t b = uint32_t(LiteMath::clamp(c.z, 0.0f, 1.0f)*255.0f + 0.5f);
      const uint32_t a = uint32_t(LiteMath::clamp(c.w, 0.0f, 1.0f)*255.0f + 0.5f);
      return r | (g << 8) | (b << 16) | (a << 24);
    }

    static inline float4 UnpackRGBA8(uint32_t c)
    {
      const float s = 1.0f/255.0f;
      return float4(float(c & 0xFF)*s, float((c >> 8) & 0xFF)*s, float((c >> 16) & 0xFF)*s, float(c >> 24)*s);
    }

    static inline float4 Blend(BlendMode a_mode, float4 a_src, float4 a_dst)
    {
      switch(a_mode)
      {
        case BlendMode::ALPHA: return float4(a_src.x*a_src.w + a_dst.x*(1.0f - a_src.w),
                                             a_src.y*a_src.w + a_dst.y*(1.0f - a_src.w),
                                             a_src.z*a_src.w + a_dst.z*(1.0f - a_src.w),
                                             a_src.w         + a_dst.w*(1.0f - a_src.w));
        case BlendMode::ADD:   return a_src + a_dst;
        default:               return a_src;
      };
    }

    /// writes pixel shader output (float4 or a struct of float4) to color targets 0, 1, ...
    template<typename PSOut>
    static inline void WriteColors(const Target& a_tgt, int a_offs, const PSOut& a_out)
    {
      static_assert(sizeof(PSOut) % sizeof(float4) == 0, "LiteRS: pixel shader must return float4 or a struct of float4");
      constexpr uint32_t N = uint32_t(sizeof(PSOut)/sizeof(float4));
      static_assert(N <= MAX_COLOR_TARGETS, "LiteRS: too many color targets");
      const float4* colors = reinterpret_cast<const float4*>(&a_out);
      for(uint32_t i = 0; i < N; i++)
      {
        const float4 c = colors[i];
        if(a_tgt.colorF[i] != nullptr)
          a_tgt.colorF[i][a_offs] = (a_tgt.blendMode == BlendMode::NONE) ? c : Blend(a_tgt.blendMode, c, a_tgt.colorF[i][a_offs]);
        else if(a_tgt.colorU[i] != nullptr)
          a_tgt.colorU[i][a_offs] = PackRGBA8((a_tgt.blendMode == BlendMode::NONE) ? c : Blend(a_tgt.blendMode, c, UnpackRGBA8(a_tgt.colorU[i][a_offs])));
      }
    }

    /**
    \brief Rasterizes triangles in submission order. Multithreaded mode is 'sort-middle': triangles are binned to
           bins of binSize x binSize pixels and bins are processed in parallel, so the order of triangles inside
           every pixel is kept (blending is correct and the result is deterministic).
    */
    template<typename V, typename PixelFn>
    static void RasterizeAll(const CPUSettings& a_settings, const Target& a_tgt, const std::vector<Triangle<V>>& a_tris, PixelFn a_pixel)
    {
      if(a_tris.empty())
        return;

      const Rect& sc = a_tgt.scissor;
      const int   bs = a_tgt.BinSize(a_settings);
      const int   binsX = (sc.x1 - 1)/bs - sc.x0/bs + 1;
      const int   binsY = (sc.y1 - 1)/bs - sc.y0/bs + 1;

      if(!a_settings.multithreaded || binsX*binsY == 1)
      {
        for(const auto& tri : a_tris)
        {
          auto frag = [&](int x, int y, float b0, float b1, float b2) { a_pixel(tri, x, y, b0, b1, b2); };
          Rasterize(tri, sc, a_settings, frag);
        }
        return;
      }

      std::vector< std::vector<uint32_t> > bins(size_t(binsX*binsY));
      for(uint32_t triId = 0; triId < uint32_t(a_tris.size()); triId++)
      {
        const Rect& bb = a_tris[triId].bbox;
        for(int by = bb.y0/bs; by <= (bb.y1 - 1)/bs; by++)
          for(int bx = bb.x0/bs; bx <= (bb.x1 - 1)/bs; bx++)
            bins[size_t((by - sc.y0/bs)*binsX + (bx - sc.x0/bs))].push_back(triId);
      }

      #pragma omp parallel for schedule(dynamic, 1)
      for(int binId = 0; binId < binsX*binsY; binId++)
      {
        const int  bx  = binId % binsX + sc.x0/bs;
        const int  by  = binId / binsX + sc.y0/bs;
        const Rect bin = Intersect(Rect{bx*bs, by*bs, (bx + 1)*bs, (by + 1)*bs}, sc);
        for(uint32_t triId : bins[size_t(binId)])
        {
          const Triangle<V>& tri = a_tris[triId];
          auto frag = [&](int x, int y, float b0, float b1, float b2) { a_pixel(tri, x, y, b0, b1, b2); };
          Rasterize(tri, bin, a_settings, frag);
        }
      }
    }

    /**
    \brief Draws a triangle list of already transformed vertices.
    \param a_verts      -- vertices after vertex shader
    \param a_indices    -- index buffer or nullptr for non indexed draw
    \param a_num        -- number of indices (or vertices for non indexed draw)
    \param a_indexBase  -- vertex 'i' is stored in a_verts[i - a_indexBase]
    \param a_ps         -- functor 'PSOut(const V&)' or nullptr for depth only pass
    */
    template<typename V, typename PSFn>
    static void DrawVertexStream(Pipeline& a_pipe, const Vertex<V>* a_verts, const uint32_t* a_indices, uint32_t a_num, uint32_t a_indexBase, PSFn a_ps)
    {
      static_assert(sizeof(V) % sizeof(float) == 0, "LiteRS: varyings must consist of float, float2, float3, float4 fields");
      const Target tgt = MakeTarget(a_pipe);

      StageTimer setupTimer(a_pipe.cpu.collectStats);
      std::vector< Triangle<V> > tris;
      tris.reserve(a_num/3);
      for(uint32_t i = 0; i + 2 < a_num; i += 3)
      {
        if(a_indices == nullptr)
          AssembleTriangle(tgt, a_verts[i + 0], a_verts[i + 1], a_verts[i + 2], tris);
        else
          AssembleTriangle(tgt, a_verts[a_indices[i + 0] - a_indexBase], a_verts[a_indices[i + 1] - a_indexBase], a_verts[a_indices[i + 2] - a_indexBase], tris);
      }
      a_pipe.stats.setupMs   += setupTimer.Ms();
      a_pipe.stats.triangles += tris.size();
      a_pipe.stats.draws     += a_pipe.cpu.collectStats ? 1 : 0;

      auto shadePixel = [&](const Triangle<V>& tri, int x, int y, float b0, float b1, float b2)
      {
        const int   offs = tgt.Offset(x, y);
        const float z    = b0*tri.z[0] + b1*tri.z[1] + b2*tri.z[2];
        if(!DepthTest(tgt, offs, z))
          return;
        if constexpr(!std::is_same<PSFn, std::nullptr_t>::value)
        {
          V in;
          Interpolate(tri, b0, b1, b2, in);
          WriteColors(tgt, offs, a_ps(in));
        }
      };
      StageTimer rasterTimer(a_pipe.cpu.collectStats);
      RasterizeAll(a_pipe.cpu, tgt, tris, shadePixel);
      a_pipe.stats.rasterMs += rasterTimer.Ms();
    }

    /// runs vertex shader for vertices [a_first, a_first + a_num); a_vs(vertexId, Vertex<V>&) writes one vertex
    template<typename V, typename VSFn>
    static std::vector< Vertex<V> > RunVertexShader(Pipeline& a_pipe, uint32_t a_first, uint32_t a_num, VSFn a_vs)
    {
      StageTimer timer(a_pipe.cpu.collectStats);
      std::vector< Vertex<V> > verts(a_num);
      #pragma omp parallel for if(a_pipe.cpu.multithreaded && a_num >= 4096)
      for(int i = 0; i < int(a_num); i++)
        a_vs(a_first + uint32_t(i), verts[i]);
      a_pipe.stats.vertexMs += timer.Ms();
      return verts;
    }

    /// range of vertex indices [first, last] used by an index buffer
    static inline void IndexRange(const uint32_t* a_indices, uint32_t a_indexNum, uint32_t& a_first, uint32_t& a_last)
    {
      a_first = 0xFFFFFFFF;
      a_last  = 0;
      for(uint32_t i = 0; i < a_indexNum; i++)
      {
        a_first = std::min(a_first, a_indices[i]);
        a_last  = std::max(a_last,  a_indices[i]);
      }
    }

    template<typename... VSArgs>
    using VaryingsOf = typename std::remove_cv<typename std::remove_pointer<typename std::tuple_element<sizeof...(VSArgs) - 1, std::tuple<VSArgs...> >::type>::type>::type;

    template<size_t Offset, typename Tuple, size_t... I>
    auto SubTupleType(std::index_sequence<I...>) -> std::tuple<typename std::decay<typename std::tuple_element<Offset + I, Tuple>::type>::type...>;

    /// std::tuple of additional arguments of vertex shader: 'VS(uint32_t a_vertexId, Args..., V* a_out)' ==> std::tuple<Args...>
    template<size_t Count, typename... VSArgs>
    using UserArgsOf = decltype(SubTupleType<1, std::tuple<VSArgs...>>(std::make_index_sequence<Count>{}));

    /// varyings as an array of floats: back ends interpolate them without knowing their structure
    template<int N>
    struct FloatArray { float v[N]; };

    /**
    \brief For back ends of DrawBackend: assembles and sets up triangles from transformed vertices and calls
           a_binFn(tris, triIds, triNum, binRect) for every bin of the screen; bins are processed in parallel, triangles
           of a bin are given in submission order (see RasterizeAll).
    */
    /// buffer without initialization of elements, it keeps memory between draws
    struct RawBuffer
    {
      void* Get(size_t a_bytes)
      {
        if(a_bytes > capacity)
        {
          capacity = a_bytes + a_bytes/4;
          data.reset(new uint64_t[(capacity + 7)/8]);
        }
        return data.get();
      }
      std::unique_ptr<uint64_t[]> data;
      size_t capacity = 0;
    };

    /// binned triangles of one portion of input triangles: CSR lists of bins, ids are global or (SPILL_BIT | index in spill)
    struct BinningPortion
    {
      std::vector<uint8_t>  spill;     ///< Triangle<V> produced by clipping in addition to the first triangle of input triangle
      std::vector<uint32_t> pairs;     ///< (bin, id) pairs in submission order
      std::vector<uint32_t> binStart;  ///< CSR: ids of bin b are ids[binStart[b] ... binStart[b+1])
      std::vector<uint32_t> ids;
      uint32_t              produced = 0; ///< number of triangles after setup
    };

    struct BinningCache
    {
      RawBuffer                   tris;
      std::vector<BinningPortion> portions;
      std::vector<uint32_t>       binStart;
      std::vector<uint32_t>       ids;
    };

    /**
    \brief For back ends of DrawBackend: assembles and sets up triangles from transformed vertices and calls
           a_binFn(tris, triIds, triNum, binRect) for every bin of the screen; bins are processed in parallel, triangles
           of a bin are given in submission order (see RasterizeAll).
           Portions of input triangles are assembled, set up and binned in parallel: the first triangle produced from input
           triangle i is stored to tris[i], additional triangles of clipping are stored after all input triangles.
    */
    template<typename V, typename BinFn>
    static void DrawVertexStreamByBins(Pipeline& a_pipe, const Vertex<V>* a_verts, const uint32_t* a_indices, uint32_t a_num, uint32_t a_indexBase, BinFn a_binFn)
    {
      constexpr uint32_t SPILL_BIT   = 0x80000000u;
      constexpr uint32_t PORTION_TRI = 16384;
      using Tri = Triangle<V>;

      const Target tgt   = MakeTarget(a_pipe);
      const Rect&  sc    = tgt.scissor;
      const int    bs    = tgt.BinSize(a_pipe.cpu);
      const int    binsX = (sc.x1 - 1)/bs - sc.x0/bs + 1;
      const int    binsY = (sc.y1 - 1)/bs - sc.y0/bs + 1;
      const int    binsN = binsX*binsY;
      if(a_pipe.binningCache == nullptr)
        a_pipe.binningCache = std::make_shared<BinningCache>();
      BinningCache& cache = *a_pipe.binningCache;

      StageTimer setupTimer(a_pipe.cpu.collectStats);
      const uint32_t inputTris = a_num/3;
      const int      portionsN = int((inputTris + PORTION_TRI - 1)/PORTION_TRI);
      if(cache.portions.size() < size_t(portionsN))
        cache.portions.resize(size_t(portionsN));
      Tri* tris = reinterpret_cast<Tri*>(cache.tris.Get(sizeof(Tri)*size_t(inputTris)));

      // (1) assembly, setup and binning of portions in parallel
      //
      #pragma omp parallel for schedule(dynamic, 1) if(a_pipe.cpu.multithreaded && portionsN >= 4) // waking threads up costs about a millisecond
      for(int p = 0; p < portionsN; p++)
      {
        BinningPortion& portion = cache.portions[size_t(p)];
        portion.spill.clear();
        portion.pairs.clear();
        portion.produced = 0;
        const uint32_t first = uint32_t(p)*PORTION_TRI;
        const uint32_t last  = std::min(first + PORTION_TRI, inputTris);
        std::vector<Tri> produced;
        produced.reserve(8);
        for(uint32_t tri = first; tri < last; tri++)
        {
          produced.clear();
          const uint32_t i = tri*3;
          if(a_indices == nullptr)
            AssembleTriangle(tgt, a_verts[i + 0], a_verts[i + 1], a_verts[i + 2], produced);
          else
            AssembleTriangle(tgt, a_verts[a_indices[i + 0] - a_indexBase], a_verts[a_indices[i + 1] - a_indexBase], a_verts[a_indices[i + 2] - a_indexBase], produced);
          portion.produced += uint32_t(produced.size());
          for(size_t k = 0; k < produced.size(); k++)
          {
            uint32_t id = tri;
            if(k == 0)
              tris[tri] = produced[0];
            else
            {
              id = SPILL_BIT | uint32_t(portion.spill.size()/sizeof(Tri));
              const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&produced[k]);
              portion.spill.insert(portion.spill.end(), bytes, bytes + sizeof(Tri));
            }
            const Rect& bb = produced[k].bbox;
            for(int by = bb.y0/bs; by <= (bb.y1 - 1)/bs; by++)
            {
              for(int bx = bb.x0/bs; bx <= (bb.x1 - 1)/bs; bx++)
              {
                portion.pairs.push_back(uint32_t((by - sc.y0/bs)*binsX + (bx - sc.x0/bs)));
                portion.pairs.push_back(id);
              }
            }
          }
        }
        // stable counting sort of pairs by bin
        portion.binStart.assign(size_t(binsN) + 1, 0);
        for(size_t k = 0; k < portion.pairs.size(); k += 2)
          portion.binStart[portion.pairs[k] + 1]++;
        for(int b = 0; b < binsN; b++)
          portion.binStart[size_t(b) + 1] += portion.binStart[size_t(b)];
        portion.ids.resize(portion.pairs.size()/2);
        std::vector<uint32_t> fill(portion.binStart.begin(), portion.binStart.end() - 1);
        for(size_t k = 0; k < portion.pairs.size(); k += 2)
          portion.ids[fill[portion.pairs[k]]++] = portion.pairs[k + 1];
      }

      // (2) triangles of clipping are placed after input triangles, CSR of all bins in order of portions
      //
      std::vector<size_t> spillStart(size_t(portionsN) + 1, size_t(inputTris));
      for(int p = 0; p < portionsN; p++)
        spillStart[size_t(p) + 1] = spillStart[size_t(p)] + cache.portions[size_t(p)].spill.size()/sizeof(Tri);
      const size_t trisNum = spillStart.back();
      if(trisNum > inputTris)
      {
        std::vector<Tri> copy(tris, tris + inputTris); // rare case: clipping produced additional triangles
        tris = reinterpret_cast<Tri*>(cache.tris.Get(sizeof(Tri)*trisNum));
        std::copy(copy.begin(), copy.end(), tris);
        for(int p = 0; p < portionsN; p++)
          std::memcpy(tris + spillStart[size_t(p)], cache.portions[size_t(p)].spill.data(), cache.portions[size_t(p)].spill.size());
      }

      cache.binStart.assign(size_t(binsN) + 1, 0);
      for(int b = 0; b < binsN; b++)
      {
        uint32_t count = 0;
        for(int p = 0; p < portionsN; p++)
          count += cache.portions[size_t(p)].binStart[size_t(b) + 1] - cache.portions[size_t(p)].binStart[size_t(b)];
        cache.binStart[size_t(b) + 1] = cache.binStart[size_t(b)] + count;
      }
      cache.ids.resize(cache.binStart.back());
      #pragma omp parallel for schedule(dynamic, 4) if(a_pipe.cpu.multithreaded && cache.ids.size() > 65536) // waking threads up costs milliseconds
      for(int b = 0; b < binsN; b++)
      {
        uint32_t dst = cache.binStart[size_t(b)];
        for(int p = 0; p < portionsN; p++)
        {
          const BinningPortion& portion = cache.portions[size_t(p)];
          for(uint32_t k = portion.binStart[size_t(b)]; k < portion.binStart[size_t(b) + 1]; k++)
          {
            const uint32_t id = portion.ids[k];
            cache.ids[dst++] = (id & SPILL_BIT) ? uint32_t(spillStart[size_t(p)]) + (id & ~SPILL_BIT) : id;
          }
        }
      }
      a_pipe.stats.setupMs += setupTimer.Ms();
      for(int p = 0; p < portionsN; p++)
        a_pipe.stats.triangles += cache.portions[size_t(p)].produced;
      a_pipe.stats.draws     += a_pipe.cpu.collectStats ? 1 : 0;

      // (3) bins are rasterized in parallel
      //
      StageTimer rasterTimer(a_pipe.cpu.collectStats);
      #pragma omp parallel for schedule(dynamic, 1) if(a_pipe.cpu.multithreaded)
      for(int b = 0; b < binsN; b++)
      {
        const uint32_t count = cache.binStart[size_t(b) + 1] - cache.binStart[size_t(b)];
        if(count == 0)
          continue;
        const int  bx  = b % binsX + sc.x0/bs;
        const int  by  = b / binsX + sc.y0/bs;
        const Rect bin = Intersect(Rect{bx*bs, by*bs, (bx + 1)*bs, (by + 1)*bs}, sc);
        a_binFn(tris, cache.ids.data() + cache.binStart[size_t(b)], count, bin);
      }
      a_pipe.stats.rasterMs += rasterTimer.Ms();
    }

  }; // namespace detail

  /**
  \brief Shader of a draw as a key: bytes of the pointer to member function.
  */
  struct ShaderKey
  {
    uint64_t data[2] = {0, 0};
    bool operator==(const ShaderKey& a_other) const { return data[0] == a_other.data[0] && data[1] == a_other.data[1]; }
  };

  template<typename F>
  static inline ShaderKey MakeShaderKey(F a_func)
  {
    static_assert(sizeof(F) <= sizeof(ShaderKey), "LiteRS: unexpected size of pointer to member function");
    ShaderKey key;
    std::memcpy(key.data, &a_func, sizeof(F));
    return key;
  }

  /**
  \brief Alternative implementation of draws. Code generated by kernel_slicer for ISPC sets 'Pipeline::backend', then
         draw commands of the control function (which is executed on CPU as is) call Draw with the shaders of the command.
         Draw returns false if it does not know these shaders, then the draw is executed by the C++ implementation.
  \param a_args -- pointer to std::tuple of additional arguments of the vertex shader (detail::UserArgsOf)
  */
  struct DrawBackend
  {
    virtual ~DrawBackend() {}
    virtual bool Draw(Pipeline& a_pipe, const ShaderKey& a_vs, const ShaderKey& a_ps, const uint32_t* a_indices, uint32_t a_num, const void* a_args) = 0;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////// commands /////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
  \brief Clears all color targets of a_pipe with a_pipe.clearColor and the depth target with a_pipe.clearDepth.
  */
  static inline void Clear(Pipeline& a_pipe)
  {
    for(uint32_t i = 0; i < MAX_COLOR_TARGETS; i++)
    {
      if(a_pipe.colorF[i] != nullptr)
        a_pipe.colorF[i]->clear(a_pipe.clearColor);
      if(a_pipe.colorU[i] != nullptr)
        a_pipe.colorU[i]->clear(detail::PackRGBA8(a_pipe.clearColor));
    }
    if(a_pipe.depth != nullptr)
      a_pipe.depth->clear(a_pipe.clearDepth);
  }

  /**
  \brief Draws a triangle list with vertex and pixel shaders given as member functions of the main class.
  \param a_self    -- 'this' of the main class
  \param a_vs      -- vertex shader 'float4 VS(uint32_t a_vertexId, Args..., V* a_out) const', returns clip space position
  \param a_ps      -- pixel shader  'float4 PS(const V& a_in) const' (or a struct of float4 for several color targets)
  \param a_vertNum -- number of vertices, vertex ids are 0, 1, ..., a_vertNum-1
  \param a_args    -- additional arguments of the vertex shader
  Shaders are const member functions: they are executed in parallel and can not change data of the class.
  */
  template<typename C, typename VSRet, typename... VSArgs, typename PSRet, typename... PSArgs, typename... Args>
  static void DrawTriangles(Pipeline& a_pipe, C* a_self, VSRet (C::*a_vs)(VSArgs...) const, PSRet (C::*a_ps)(PSArgs...) const, uint32_t a_vertNum, Args... a_args)
  {
    if(a_pipe.backend != nullptr)
    {
      const detail::UserArgsOf<sizeof...(VSArgs) - 2, VSArgs...> args(a_args...);
      if(a_pipe.backend->Draw(a_pipe, MakeShaderKey(a_vs), MakeShaderKey(a_ps), nullptr, a_vertNum, &args))
        return;
    }
    using V = detail::VaryingsOf<VSArgs...>;
    const auto verts = detail::RunVertexShader<V>(a_pipe, 0, a_vertNum, [&](uint32_t a_id, Vertex<V>& a_vert) { a_vert.pos = (a_self->*a_vs)(a_id, a_args..., &a_vert.attr); });
    detail::DrawVertexStream(a_pipe, verts.data(), nullptr, a_vertNum, 0, [a_self, a_ps](const V& a_in) { return (a_self->*a_ps)(a_in); });
  }

  /**
  \brief Depth only pass: draws a triangle list with vertex shader 'float4 VS(uint32_t a_vertexId, Args...) const' and without pixel shader.
  */
  template<typename C, typename VSRet, typename... VSArgs, typename... Args>
  static void DrawTriangles(Pipeline& a_pipe, C* a_self, VSRet (C::*a_vs)(VSArgs...) const, uint32_t a_vertNum, Args... a_args)
  {
    if(a_pipe.backend != nullptr)
    {
      const detail::UserArgsOf<sizeof...(VSArgs) - 1, VSArgs...> args(a_args...);
      if(a_pipe.backend->Draw(a_pipe, MakeShaderKey(a_vs), ShaderKey(), nullptr, a_vertNum, &args))
        return;
    }
    using V = detail::NoVaryings;
    const auto verts = detail::RunVertexShader<V>(a_pipe, 0, a_vertNum, [&](uint32_t a_id, Vertex<V>& a_vert) { a_vert.pos = (a_self->*a_vs)(a_id, a_args...); });
    detail::DrawVertexStream(a_pipe, verts.data(), nullptr, a_vertNum, 0, nullptr);
  }

  /**
  \brief Draws an indexed triangle list: vertex shader gets a_indices[i] as a_vertexId and is executed once for every vertex
         in the range of used indices.
  \param a_indices  -- index buffer, usually 'm_indices.data()' of a member vector
  \param a_indexNum -- number of indices
  The other parameters are the same as in DrawTriangles.
  */
  template<typename C, typename VSRet, typename... VSArgs, typename PSRet, typename... PSArgs, typename... Args>
  static void DrawIndexedTriangles(Pipeline& a_pipe, C* a_self, VSRet (C::*a_vs)(VSArgs...) const, PSRet (C::*a_ps)(PSArgs...) const,
                                   const uint32_t* a_indices, uint32_t a_indexNum, Args... a_args)
  {
    if(a_pipe.backend != nullptr)
    {
      const detail::UserArgsOf<sizeof...(VSArgs) - 2, VSArgs...> args(a_args...);
      if(a_pipe.backend->Draw(a_pipe, MakeShaderKey(a_vs), MakeShaderKey(a_ps), a_indices, a_indexNum, &args))
        return;
    }
    using V = detail::VaryingsOf<VSArgs...>;
    uint32_t first, last;
    detail::IndexRange(a_indices, a_indexNum, first, last);
    if(a_indexNum == 0)
      return;
    const auto verts = detail::RunVertexShader<V>(a_pipe, first, last - first + 1, [&](uint32_t a_id, Vertex<V>& a_vert) { a_vert.pos = (a_self->*a_vs)(a_id, a_args..., &a_vert.attr); });
    detail::DrawVertexStream(a_pipe, verts.data(), a_indices, a_indexNum, first, [a_self, a_ps](const V& a_in) { return (a_self->*a_ps)(a_in); });
  }

  /**
  \brief Depth only pass with an indexed triangle list.
  */
  template<typename C, typename VSRet, typename... VSArgs, typename... Args>
  static void DrawIndexedTriangles(Pipeline& a_pipe, C* a_self, VSRet (C::*a_vs)(VSArgs...) const, const uint32_t* a_indices, uint32_t a_indexNum, Args... a_args)
  {
    if(a_pipe.backend != nullptr)
    {
      const detail::UserArgsOf<sizeof...(VSArgs) - 1, VSArgs...> args(a_args...);
      if(a_pipe.backend->Draw(a_pipe, MakeShaderKey(a_vs), ShaderKey(), a_indices, a_indexNum, &args))
        return;
    }
    using V = detail::NoVaryings;
    uint32_t first, last;
    detail::IndexRange(a_indices, a_indexNum, first, last);
    if(a_indexNum == 0)
      return;
    const auto verts = detail::RunVertexShader<V>(a_pipe, first, last - first + 1, [&](uint32_t a_id, Vertex<V>& a_vert) { a_vert.pos = (a_self->*a_vs)(a_id, a_args...); });
    detail::DrawVertexStream(a_pipe, verts.data(), a_indices, a_indexNum, first, nullptr);
  }

#endif // KERNEL_SLICER

}; // namespace LiteRS
