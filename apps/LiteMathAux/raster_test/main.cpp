// Tests and a small benchmark of LiteRS CPU rasterizer (apps/LiteMathAux/raster.h) and of tiled Image2D layout
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>

#include "LiteMath.h"
#include "Image2d.h"
#include "raster.h"

using LiteMath::float4;
using LiteMath::float2;
using LiteMath::int2;
using LiteImage::Image2D;
using LiteImage::Layout;

struct Col { float4 color; };
using Vert = LiteRS::Vertex<Col>;

static float4 PassColor(const Col& a_in) { return a_in.color; }

static void Draw(LiteRS::Pipeline& a_pipe, const std::vector<Vert>& a_verts)
{
  LiteRS::detail::DrawVertexStream(a_pipe, a_verts.data(), nullptr, uint32_t(a_verts.size()), 0, PassColor);
}

static const char* AlgoName(LiteRS::RasterAlgo a_algo)
{
  switch(a_algo)
  {
    case LiteRS::RasterAlgo::PIXEL: return "PIXEL";
    case LiteRS::RasterAlgo::BLOCK: return "BLOCK";
    default:                        return "AUTO ";
  };
}

/// grid of cells, 2 triangles per cell with shared jittered vertices; winding of triangles alternates
static std::vector<Vert> MakeGrid(int a_cellsX, int a_cellsY, std::mt19937& a_gen)
{
  std::uniform_real_distribution<float> jitter(-0.2f, 0.2f); // keeps cells convex
  std::vector<float4> nodes((a_cellsX + 1)*(a_cellsY + 1));
  for(int y = 0; y <= a_cellsY; y++)
  {
    for(int x = 0; x <= a_cellsX; x++)
    {
      const bool  border = (x == 0 || y == 0 || x == a_cellsX || y == a_cellsY);
      const float jx = border ? 0.0f : jitter(a_gen);
      const float jy = border ? 0.0f : jitter(a_gen);
      const float nx = -1.0f + 2.0f*(float(x) + jx)/float(a_cellsX);
      const float ny = -1.0f + 2.0f*(float(y) + jy)/float(a_cellsY);
      nodes[y*(a_cellsX + 1) + x] = float4(nx, ny, 0.5f, 1.0f);
    }
  }
  std::vector<Vert> verts;
  const Col one = { float4(1.0f, 0.0f, 0.0f, 0.0f) };
  for(int y = 0; y < a_cellsY; y++)
  {
    for(int x = 0; x < a_cellsX; x++)
    {
      const float4 p00 = nodes[(y + 0)*(a_cellsX + 1) + x + 0];
      const float4 p10 = nodes[(y + 0)*(a_cellsX + 1) + x + 1];
      const float4 p01 = nodes[(y + 1)*(a_cellsX + 1) + x + 0];
      const float4 p11 = nodes[(y + 1)*(a_cellsX + 1) + x + 1];
      if((x + y) % 2 == 0)
      {
        verts.push_back({p00, one}); verts.push_back({p10, one}); verts.push_back({p11, one});
        verts.push_back({p00, one}); verts.push_back({p11, one}); verts.push_back({p01, one});
      }
      else // other diagonal and opposite winding
      {
        verts.push_back({p10, one}); verts.push_back({p00, one}); verts.push_back({p01, one});
        verts.push_back({p10, one}); verts.push_back({p01, one}); verts.push_back({p11, one});
      }
    }
  }
  return verts;
}

/// random triangles with random depth and colors, 'a_size' is the approximate size in pixels
static std::vector<Vert> MakeRandomTris(int a_num, float a_size, int a_width, int a_height, std::mt19937& a_gen)
{
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  std::vector<Vert> verts;
  verts.reserve(a_num*3);
  const float sx = 2.0f*a_size/float(a_width);
  const float sy = 2.0f*a_size/float(a_height);
  for(int i = 0; i < a_num; i++)
  {
    const float cx = -1.0f + 2.0f*u01(a_gen);
    const float cy = -1.0f + 2.0f*u01(a_gen);
    const float z  = u01(a_gen);
    const Col   c  = { float4(u01(a_gen), u01(a_gen), u01(a_gen), 1.0f) };
    for(int k = 0; k < 3; k++)
    {
      const float4 p = float4(cx + sx*(u01(a_gen) - 0.5f), cy + sy*(u01(a_gen) - 0.5f), z, 1.0f);
      verts.push_back({p, c});
    }
  }
  return verts;
}

struct Target
{
  Target(int w, int h, Layout a_layout = Layout::LINEAR)
  {
    color.resize(w, h, a_layout);
    depth.resize(w, h, a_layout);
    pipe.SetColorTarget(color);
    pipe.SetDepthTarget(depth);
    pipe.SetViewport(0, 0, w, h);
  }
  Image2D<float4>  color;
  Image2D<float>   depth;
  LiteRS::Pipeline pipe;
};

static int CountDifferent(const Target& a, const Target& b)
{
  int diff = 0;
  for(int y = 0; y < int(a.color.height()); y++)
  {
    for(int x = 0; x < int(a.color.width()); x++)
    {
      const float4 ca = a.color[int2(x, y)];
      const float4 cb = b.color[int2(x, y)];
      diff += (ca.x != cb.x || ca.y != cb.y || ca.z != cb.z || ca.w != cb.w || a.depth[int2(x, y)] != b.depth[int2(x, y)]) ? 1 : 0;
    }
  }
  return diff;
}

static bool TestWatertight(int w, int h)
{
  std::mt19937 gen(17);
  const std::vector<Vert> verts = MakeGrid(37, 29, gen);
  bool allOk = true;
  for(auto algo : {LiteRS::RasterAlgo::PIXEL, LiteRS::RasterAlgo::BLOCK, LiteRS::RasterAlgo::AUTO})
  {
    for(bool mt : {false, true})
    {
      Target t(w, h, Layout::TILED4x4);
      t.pipe.depthTest         = false;
      t.pipe.blendMode         = LiteRS::BlendMode::ADD;
      t.pipe.cpu.algo          = algo;
      t.pipe.cpu.multithreaded = mt;
      LiteRS::Clear(t.pipe);
      Draw(t.pipe, verts);
      int bad = 0;
      for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++)
          bad += (std::lround(t.color[int2(x, y)].x) != 1) ? 1 : 0; // interpolated constant is not exactly 1.0f, round coverage count
      std::cout << "  watertight " << AlgoName(algo) << (mt ? " MT" : " ST") << ": pixels covered != 1 time: " << bad << std::endl;
      allOk = allOk && (bad == 0);
    }
  }
  return allOk;
}

static bool TestEquivalence(int w, int h)
{
  std::mt19937 gen(5);
  std::vector<Vert> verts = MakeRandomTris(3000, 6.0f, w, h, gen);
  const std::vector<Vert> big = MakeRandomTris(300, 120.0f, w, h, gen);
  verts.insert(verts.end(), big.begin(), big.end());

  Target ref(w, h);
  ref.pipe.cpu.algo          = LiteRS::RasterAlgo::PIXEL;
  ref.pipe.cpu.multithreaded = false;
  LiteRS::Clear(ref.pipe);
  Draw(ref.pipe, verts);

  bool allOk = true;
  for(auto layout : {Layout::LINEAR, Layout::TILED4x4})
  {
    for(auto algo : {LiteRS::RasterAlgo::PIXEL, LiteRS::RasterAlgo::BLOCK, LiteRS::RasterAlgo::AUTO})
    {
      for(bool mt : {false, true})
      {
        Target t(w, h, layout);
        t.pipe.cpu.algo          = algo;
        t.pipe.cpu.multithreaded = mt;
        LiteRS::Clear(t.pipe);
        Draw(t.pipe, verts);
        const int diff = CountDifferent(ref, t);
        std::cout << "  equal to PIXEL/ST/LINEAR: " << AlgoName(algo) << (mt ? " MT " : " ST ") << (layout == Layout::LINEAR ? "LINEAR  " : "TILED4x4")
                  << ", different pixels: " << diff << std::endl;
        allOk = allOk && (diff == 0);
      }
    }
  }

  // the same triangles as an indexed list with shared vertex array (every vertex is used once, indices are reversed)
  std::vector<uint32_t> indices(verts.size());
  std::vector<Vert>     shuffled(verts.size());
  for(size_t i = 0; i < verts.size(); i++)
  {
    indices[i] = uint32_t(verts.size() - 1 - i);
    shuffled[indices[i]] = verts[i];
  }
  Target t(w, h, Layout::TILED4x4);
  LiteRS::Clear(t.pipe);
  LiteRS::detail::DrawVertexStream(t.pipe, shuffled.data(), indices.data(), uint32_t(indices.size()), 0, PassColor);
  const int diff = CountDifferent(ref, t);
  std::cout << "  equal to PIXEL/ST/LINEAR: indexed draw, different pixels: " << diff << std::endl;
  return allOk && (diff == 0);
}

/// Vulkan style perspective: depth [0,1], y down
static float4 Project(float4 p, float a_fovY, float a_aspect, float a_near, float a_far)
{
  const float f = 1.0f/std::tan(0.5f*a_fovY);
  return float4(f/a_aspect*p.x, -f*p.y, a_far/(a_near - a_far)*p.z + a_near*a_far/(a_near - a_far), -p.z);
}

static bool TestClipping(int w, int h)
{
  // ground plane y = -1 that goes behind the camera and far beyond the guard band
  const float  S = 1000.0f;
  const float4 q[4] = { float4(-S, -1.0f, -S, 1.0f), float4(S, -1.0f, -S, 1.0f), float4(S, -1.0f, S, 1.0f), float4(-S, -1.0f, S, 1.0f) };
  const Col    one  = { float4(1.0f, 0.0f, 0.0f, 0.0f) };
  std::vector<Vert> verts;
  for(int i : {0, 1, 2, 0, 2, 3})
    verts.push_back({Project(q[i], 1.0f, float(w)/float(h), 0.1f, 5000.0f), one});

  bool allOk = true;
  for(auto algo : {LiteRS::RasterAlgo::PIXEL, LiteRS::RasterAlgo::BLOCK})
  {
    Target t(w, h);
    t.pipe.depthTest = false;
    t.pipe.blendMode = LiteRS::BlendMode::ADD;
    t.pipe.cpu.algo  = algo;
    LiteRS::Clear(t.pipe);
    Draw(t.pipe, verts);
    int moreThanOnce = 0, bottomNotCovered = 0, covered = 0;
    for(int y = 0; y < h; y++)
    {
      for(int x = 0; x < w; x++)
      {
        const long c = std::lround(t.color[int2(x, y)].x);
        moreThanOnce     += (c > 1) ? 1 : 0;
        covered          += (c == 1) ? 1 : 0;
        bottomNotCovered += (y >= h/2 + 8 && c != 1) ? 1 : 0;
      }
    }
    std::cout << "  clipping " << AlgoName(algo) << ": covered " << covered << " of " << w*h << ", covered twice: " << moreThanOnce
              << ", not covered below horizon: " << bottomNotCovered << std::endl;
    allOk = allOk && (moreThanOnce == 0) && (bottomNotCovered == 0);
  }
  return allOk;
}

/// tiled layout of Image2D: pixel access, sampling, conversion of layouts and saving give the same results as LINEAR
static bool TestTiledImage()
{
  const int w = 37, h = 23;   // not a multiple of the tile size
  Image2D<float4> lin(w, h);
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
      lin[int2(x, y)] = float4(float(x), float(y), float(x*y), 1.0f);

  Image2D<float4> tiled = lin;
  tiled.setLayout(Layout::TILED4x4);

  int bad = 0;
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
      bad += (tiled[int2(x, y)].z != lin[int2(x, y)].z) ? 1 : 0;

  LiteImage::Sampler sampler;
  std::mt19937 gen(3);
  std::uniform_real_distribution<float> u(-0.5f, 1.5f);
  for(auto filter : {LiteImage::Sampler::Filter::NEAREST, LiteImage::Sampler::Filter::LINEAR})
  {
    for(auto mode : {LiteImage::Sampler::AddressMode::WRAP, LiteImage::Sampler::AddressMode::CLAMP})
    {
      sampler.filter   = filter;
      sampler.addressU = sampler.addressV = mode;
      for(int i = 0; i < 1000; i++)
      {
        const float2 uv(u(gen), u(gen));
        const float4 a = lin.sample(sampler, uv);
        const float4 b = tiled.sample(sampler, uv);
        bad += (a.x != b.x || a.y != b.y || a.z != b.z) ? 1 : 0;
      }
    }
  }

  Image2D<float4> back = tiled;
  back.setLayout(Layout::LINEAR);
  for(size_t i = 0; i < lin.vector().size(); i++)
    bad += (back.vector()[i].z != lin.vector()[i].z) ? 1 : 0;
  bad += (LiteImage::MSE(lin, tiled) != 0.0f) ? 1 : 0;

  std::cout << "  tiled Image2D: errors in access, sampling, layout conversion and MSE: " << bad << std::endl;
  return bad == 0;
}

static void Benchmark(int w, int h)
{
  std::cout << std::endl << "Benchmark " << w << "x" << h << ", trivial pixel shader, ms (best of 5); each row covers about 8 screens" << std::endl;
  std::cout << "  tri size |  tris   | PIXEL ST | BLOCK ST | AUTO ST  | AUTO MT  | AUTO ST tiled | AUTO MT tiled" << std::endl;
  for(float size : {2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 256.0f})
  {
    std::mt19937 gen(1);
    const int numTris = int(8.0f*float(w*h)/(0.25f*size*size)) + 1;
    const std::vector<Vert> verts = MakeRandomTris(numTris, size, w, h, gen);
    std::cout << "  " << std::setw(8) << size << " | " << std::setw(7) << numTris;
    for(int run = 0; run < 6; run++)
    {
      Target t(w, h, (run >= 4) ? Layout::TILED4x4 : Layout::LINEAR);
      t.pipe.cpu.algo          = (run == 0) ? LiteRS::RasterAlgo::PIXEL : (run == 1) ? LiteRS::RasterAlgo::BLOCK : LiteRS::RasterAlgo::AUTO;
      t.pipe.cpu.multithreaded = (run == 3 || run == 5);
      float best = 1e10f;
      for(int iter = 0; iter < 5; iter++)
      {
        LiteRS::Clear(t.pipe);
        auto start = std::chrono::high_resolution_clock::now();
        Draw(t.pipe, verts);
        best = std::min(best, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f);
      }
      std::cout << " | " << std::setw(run >= 4 ? 13 : 8) << std::fixed << std::setprecision(2) << best;
    }
    std::cout << std::endl;
  }
}

int main(int argc, const char** argv)
{
  const int W = 1024, H = 768;
  bool ok = true;
  std::cout << "Tests" << std::endl;
  ok = TestWatertight(W - 3, H - 5) && ok;  // size is not a multiple of block and bin size on purpose
  ok = TestEquivalence(W - 3, H - 5) && ok;
  ok = TestClipping(W, H) && ok;
  ok = TestTiledImage() && ok;
  std::cout << (ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

  if(argc > 1 && std::string(argv[1]) == "--bench")
    Benchmark(W, H);
  return ok ? 0 : 1;
}
