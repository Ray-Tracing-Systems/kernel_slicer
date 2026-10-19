#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <cstdint>

#include "conv2d.h"
#include "Image2d.h"
#include "ArgParser.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<Convolution2D> CreateConvolution2D_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures Convolution2D_Generated_ListRequiredDeviceFeatures();
#endif

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data); // imageutils.cpp

//////////////////////////////////////////////////////////////////////////////////////////////// filter bank (host side)

struct Filter
{
  std::string        name;
  int                size;        // 3, 5 or 7
  std::vector<float> weights;     // size*size for 2D, size for separable (same 1D kernel for X and Y)
  bool               separable;
  bool               normalize;   // edge-like filters: show |value| / max|value|
};

static std::vector<float> Gauss1D(int a_size, float a_sigma)
{
  std::vector<float> w(a_size);
  const int r = a_size/2;
  float summ = 0.0f;
  for(int i = -r; i <= r; i++) { w[i + r] = std::exp(-float(i*i)/(2.0f*a_sigma*a_sigma)); summ += w[i + r]; }
  for(auto& x : w) x /= summ;
  return w;
}

static std::vector<float> Outer(const std::vector<float>& a, const std::vector<float>& b)
{
  std::vector<float> res(a.size()*b.size());
  for(size_t y = 0; y < a.size(); y++)
    for(size_t x = 0; x < b.size(); x++)
      res[y*b.size() + x] = a[y]*b[x];
  return res;
}

static std::vector<Filter> MakeFilters()
{
  std::vector<Filter> f;
  f.push_back({"box3x3",     3, std::vector<float>(9, 1.0f/9.0f), false, false});
  f.push_back({"sharpen3x3", 3, { 0,-1, 0,
                                 -1, 5,-1,
                                  0,-1, 0}, false, false});
  f.push_back({"edge3x3",    3, {-1,-1,-1,
                                 -1, 8,-1,
                                 -1,-1,-1}, false, true});
  f.push_back({"gauss5x5",   5, Outer(Gauss1D(5, 1.0f), Gauss1D(5, 1.0f)), false, false});
  f.push_back({"emboss5x5",  5, {-1,-1,-1,-1, 0,
                                 -1,-1,-1, 0, 1,
                                 -1,-1, 0, 1, 1,
                                 -1, 0, 1, 1, 1,
                                  0, 1, 1, 1, 1}, false, true});
  f.push_back({"gauss7x7",   7, Outer(Gauss1D(7, 2.0f), Gauss1D(7, 2.0f)), false, false});
  f.push_back({"gauss7x7sep",7, Gauss1D(7, 2.0f),                          true,  false});
  return f;
}

//////////////////////////////////////////////////////////////////////////////////////////////// running

struct RunResult { std::vector<float4> hdr; std::vector<uint32_t> ldr; float timeMs; float maxValue; };

static RunResult ApplyFilter(std::shared_ptr<Convolution2D> pImpl, const Filter& f, int w, int h, const std::vector<float4>& image)
{
  RunResult res;
  res.hdr.resize(w*h);
  res.ldr.resize(w*h);
  if(f.separable)
    pImpl->ConvolveSeparable(w, h, image.data(), f.weights.data(), f.weights.data(), f.size, res.hdr.data());
  else
    pImpl->Convolve2D(w, h, image.data(), f.weights.data(), f.size, res.hdr.data());

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime(f.separable ? "ConvolveSeparable" : "Convolve2D", timings);
  res.timeMs = timings[0];

  pImpl->ToLDR(w, h, res.hdr.data(), f.normalize ? 1 : 0, 2.2f, res.ldr.data());
  res.maxValue = pImpl->GetMaxValue();
  return res;
}

static std::shared_ptr<Convolution2D> MakeImpl(bool onGPU, int w, int h, bool enableValidationLayers, int deviceId)
{
  std::shared_ptr<Convolution2D> pImpl = nullptr;
  if(onGPU)
  {
    #ifdef USE_VULKAN
    auto features = Convolution2D_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, deviceId);
    pImpl         = CreateConvolution2D_Generated(ctx, size_t(w)*size_t(h));
    #else
    std::cout << "[main]: built without Vulkan, --gpu is not available" << std::endl;
    #endif
  }
  else
    pImpl = std::make_shared<Convolution2D>();

  if(pImpl != nullptr)
  {
    pImpl->SetMaxImageSize(w, h); // 1) size member vectors
    pImpl->CommitDeviceData();    // 2) upload class data
  }
  return pImpl;
}

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  const bool compare  = args.hasOption("--compare");
  const bool onGPU    = args.hasOption("--gpu") || compare;
  const bool onCPU    = !args.hasOption("--gpu") || compare;
  const int  deviceId = args.getOptionValue<int>("--gpu_id", 0);
  const std::string inputPath = args.hasOption("--input") ? args.getOptionValue<std::string>("--input") : std::string("../images/nancy_church_2.hdr");

  int w = 0, h = 0;
  std::vector<float> hdrData;
  if(!LoadHDRImageFromFile(inputPath.c_str(), &w, &h, hdrData))
  {
    std::cout << "[main]: can't load '" << inputPath << "'" << std::endl;
    return -1;
  }
  std::vector<float4> image(size_t(w)*size_t(h));   // std::vector<float4> keeps 16-byte alignment
  for(size_t i = 0; i < image.size(); i++)
    image[i] = float4(hdrData[4*i+0], hdrData[4*i+1], hdrData[4*i+2], hdrData[4*i+3]);
  std::cout << "[main]: image " << w << "x" << h << std::endl;

  auto pCPU = onCPU ? MakeImpl(false, w, h, enableValidationLayers, deviceId) : nullptr;
  auto pGPU = onGPU ? MakeImpl(true,  w, h, enableValidationLayers, deviceId) : nullptr;
  if(onGPU && pGPU == nullptr)
    return -1;

  std::cout << std::left << std::setw(14) << "filter" << std::setw(12) << "cpu(ms)" << std::setw(12) << "gpu(ms)"
            << std::setw(14) << "max|diff|" << "ldr pixels differ" << std::endl;

  bool allOk = true;
  for(const auto& f : MakeFilters())
  {
    RunResult rc, rg;
    if(pCPU) { rc = ApplyFilter(pCPU, f, w, h, image); LiteImage::SaveBMP(("zout_cpu_" + f.name + ".bmp").c_str(), rc.ldr.data(), w, h); }
    if(pGPU) { rg = ApplyFilter(pGPU, f, w, h, image); LiteImage::SaveBMP(("zout_gpu_" + f.name + ".bmp").c_str(), rg.ldr.data(), w, h); }

    std::cout << std::left << std::setw(14) << f.name
              << std::setw(12) << (pCPU ? std::to_string(rc.timeMs) : "-")
              << std::setw(12) << (pGPU ? std::to_string(rg.timeMs) : "-");
    if(compare)
    {
      float maxDiff = 0.0f;   // relative to image magnitude
      for(size_t i = 0; i < rc.hdr.size(); i++)
      {
        const float4 d = rc.hdr[i] - rg.hdr[i];
        const float  s = std::max(1.0f, std::max(std::fabs(rc.hdr[i].x), std::max(std::fabs(rc.hdr[i].y), std::fabs(rc.hdr[i].z))));
        maxDiff = std::max(maxDiff, std::max(std::fabs(d.x), std::max(std::fabs(d.y), std::fabs(d.z)))/s);
      }
      size_t ldrDiff = 0;   // pixels whose 8-bit channels differ by more than 1
      for(size_t i = 0; i < rc.ldr.size(); i++)
        for(int ch = 0; ch < 3; ch++)
          if(std::abs(int((rc.ldr[i] >> (8*ch)) & 0xFF) - int((rg.ldr[i] >> (8*ch)) & 0xFF)) > 1) { ldrDiff++; break; }
      const bool ok = (maxDiff < 1e-4f) && (ldrDiff == 0);
      allOk = allOk && ok;
      std::cout << std::setw(14) << maxDiff << ldrDiff << (ok ? "  OK" : "  FAILED");
    }
    std::cout << std::endl;
  }

  pCPU = nullptr;
  pGPU = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
    vk_utils::globalContextDestroy();
  #endif
  if(compare)
    std::cout << (allOk ? "[main]: CPU and GPU results match" : "[main]: CPU and GPU results DIFFER") << std::endl;
  return (compare && !allOk) ? 1 : 0;
}
