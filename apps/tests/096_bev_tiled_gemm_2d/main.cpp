#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <string>
#include <cmath>

#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"
#include "test_class.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<TiledGemm2D> CreateTiledGemm2D_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif
#ifdef USE_ISPC
std::shared_ptr<TiledGemm2D> CreateTiledGemm2D_ISPC();
#endif

struct Size { uint M, N, K; };

// relative error against reference computed in double; 'false' if it is too large
//
static bool Compare(const std::vector<float>& a_C, const std::vector<double>& a_ref, uint a_K, double* a_maxErr)
{
  double maxErr = 0.0;
  for(size_t i = 0; i < a_ref.size(); i++)
    maxErr = std::max(maxErr, std::abs(double(a_C[i]) - a_ref[i])/(std::abs(a_ref[i]) + 1.0));
  *a_maxErr = maxErr;
  return maxErr < 1e-6*double(a_K) + 1e-6;
}

// Usage: testapp [--gpu | --ispc] [--bench N]
// Fixed sizes (tile borders, non square) are checked against reference, then both versions are timed on N x N x N.
//
int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  const bool onGPU  = args.hasOption("--gpu");
  const bool onISPC = args.hasOption("--ispc");
  const uint bench  = args.getOptionValue<int>("--bench", 1024);

  std::shared_ptr<TiledGemm2D> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl    = CreateTiledGemm2D_Generated(ctx, bench*bench);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateTiledGemm2D_ISPC();
  else
  #endif
    pImpl = std::make_shared<TiledGemm2D>();
  pImpl->CommitDeviceData();

  const std::vector<Size> sizes = {{1, 1, 1}, {16, 16, 16}, {17, 33, 9}, {100, 70, 130}, {256, 256, 256}, {1000, 513, 777}};
  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> valueDist(-1.0f, 1.0f);

  int failed = 0;
  for(size_t i = 0; i < sizes.size(); i++)
  {
    const auto [M, N, K] = sizes[i];
    std::vector<float> A(M*K), B(K*N), C(M*N), C2(M*N);
    for(auto& x : A) x = valueDist(gen);
    for(auto& x : B) x = valueDist(gen);

    std::vector<double> ref(M*N, 0.0);
    for(uint y = 0; y < M; y++)
      for(uint k = 0; k < K; k++)
        for(uint x = 0; x < N; x++)
          ref[y*N + x] += double(A[y*K + k])*double(B[k*N + x]);

    pImpl->MatMul     (A.data(), B.data(), C.data(),  M, N, K);
    pImpl->MatMulNaive(A.data(), B.data(), C2.data(), M, N, K);

    double err1 = 0.0, err2 = 0.0;
    const bool ok1 = Compare(C,  ref, K, &err1);
    const bool ok2 = Compare(C2, ref, K, &err2);
    failed += (ok1 ? 0 : 1) + (ok2 ? 0 : 1);
    std::cout << M << "x" << N << "x" << K << ": tiled err = " << err1 << (ok1 ? "" : " FAILED!") << ", naive err = " << err2 << (ok2 ? "" : " FAILED!") << std::endl;
    JSONLog::write("tiled_" + std::to_string(i), ok1 ? "PASSED!" : "FAILED!");
    JSONLog::write("naive_" + std::to_string(i), ok2 ? "PASSED!" : "FAILED!");
  }
  JSONLog::write("check", failed == 0 ? "PASSED!" : "FAILED!");
  JSONLog::saveToFile(onGPU ? "zout_gpu.json" : (onISPC ? "zout_ispc.json" : "zout_cpu.json"));

  if(bench > 0 && (onGPU || onISPC))                      // time of both versions; no check: it is done above
  {
    std::vector<float> A(bench*bench), B(bench*bench), C(bench*bench);
    for(auto& x : A) x = valueDist(gen);
    for(auto& x : B) x = valueDist(gen);
    for(const char* name : {"MatMul", "MatMulNaive"})
    {
      float minTime = 1e30f;
      for(int run = 0; run < 5; run++)
      {
        if(std::string(name) == "MatMul")
          pImpl->MatMul(A.data(), B.data(), C.data(), bench, bench, bench);
        else
          pImpl->MatMulNaive(A.data(), B.data(), C.data(), bench, bench, bench);
        float timings[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        pImpl->GetExecutionTime(name, timings);
        minTime = std::min(minTime, timings[0]);
      }
      std::cout << name << " " << bench << "^3: " << minTime << " ms, " << 2.0*double(bench)*bench*bench/(double(minTime)*1e6) << " GFLOPS" << std::endl;
    }
  }

  std::cout << (failed == 0 ? "PASSED" : "FAILED") << std::endl;

  pImpl = nullptr;           // Vulkan objects of the class first, then the context
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
